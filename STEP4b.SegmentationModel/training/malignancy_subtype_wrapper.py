# training/malignancy_subtype_wrapper.py
import os
import shutil
import yaml
import torch
import torch.distributed as _dist
import torch.nn.functional as F
from .losses_foundation import (
    auto_distill_malignancy_loss,
    DiceLossMultiClass,
    get_lesion_channels,
    save_tensor_as_nifti,
)


def _is_main_rank():
    """Return True iff this process is the single rank responsible for
    writing files to shared disk. Used to gate the sanity-dump block so
    every DDP rank doesn't race against every other rank to rmtree + write
    the same `SanitySubtypeMalignancyLoss/<counter>/` directory — that
    race is what produced the 94-min / 195-min mtime gaps in earlier dump
    batches (the UFO sub-type niftis survived from rank-A's write because
    rank-B's rmtree didn't run, or ran before rank-A finished).

    In a non-DDP / single-process run (`dist.is_initialized()==False`),
    this returns True so the sanity dump still works when you call the
    wrapper interactively (e.g. in tests / debug scripts).
    """
    try:
        if _dist.is_available() and _dist.is_initialized():
            return _dist.get_rank() == 0
    except Exception:
        pass
    return True


# Sanity-dump counter (shared across calls, like counter_malig / counter3 in losses_foundation.py).
counter_subtype = 0
SANITY_SUBTYPE_MAX_DUMPS = 50
SANITY_SUBTYPE_DIR = 'SanitySubtypeMalignancyLoss'

# Stashed state for post-backward gradient capture. Populated at the
# end of `_write_sanity_dumps` when that block actually writes to disk
# for this batch; consumed by `save_gradient_sanity_dumps()` after the
# training step calls `loss.backward()`. Kept module-level so
# train_ddp.py can reach it without threading extra args through the
# loss signature. `None` when no dump was written this batch.
_grad_sanity_state = None


# Per-organ sub-type definition. Extend the dict to add other organs later.
DEFAULT_SUBTYPE_MAP = {
    'pancreas': {
        'lesion_class': 'pancreatic_lesion',
        'mal_classes':  ('pancreatic_pdac', 'pancreatic_pnet'),
        'ben_classes':  ('pancreatic_cyst',),
    },
}


def _build_plan(classes, subtype_map):
    plan, strip = [], set()
    for organ, spec in subtype_map.items():
        if spec['lesion_class'] not in classes:
            continue
        mal = [classes.index(c) for c in spec['mal_classes'] if c in classes]
        ben = [classes.index(c) for c in spec['ben_classes'] if c in classes]
        if not (mal or ben):
            continue
        plan.append({
            'organ': organ,
            'lesion_class': spec['lesion_class'],
            'lesion_idx':   classes.index(spec['lesion_class']),
            'mal_idx': mal,
            'ben_idx': ben,
        })
        strip.update(mal); strip.update(ben)
    return plan, strip


def _lesion_class_weights(class_weights, classes, ref_shape):
    """Collapse per-class class_weights to lesion channels, matching the pre-processing
    done inside auto_distill_malignancy_loss."""
    if class_weights is None:
        return None
    if torch.equal(class_weights, torch.ones_like(class_weights)):
        return None
    cw = class_weights
    # (B, C, D, H, W) — broadcast to full ref shape so get_lesion_channels works the same way.
    cw = cw.repeat(ref_shape[0], 1, ref_shape[2], ref_shape[3], ref_shape[4])
    cw = get_lesion_channels(cw, classes)
    return cw


def malignancy_loss_with_subtype(
    model_output, malig_benign, unk_voxels, classes, label,
    sizes_malignancy, malignancy_per_voxel, chosen_segment_mask,
    sigmoid_already_applied=False, class_weights=None, triangle_consistency=False,
    input_tensor=None, names=None, skip_cls=False, include_ball_loss=False,
    tumor_volumes=None, tumor_diameters=None, sizes_slices=None,
    ct_z_spacing_original=None, slices_mask=None, max_slice=None,
    sample_weights=None, subseg_dilation=31,
    tumor_type=None, tumor_type_organ=None,
    crop_target=None,
    *,
    subtype_map=DEFAULT_SUBTYPE_MAP,
    subtype_loss_weight=1.0,
):
    """
    Drop-in replacement for auto_distill_malignancy_loss.

    When any sample has non-empty sub-type masks (pdac/pnet/cyst) for a configured
    organ, that (sample, organ) pair's malignant/benign supervision comes from a
    standard BCE+Dice on the merged sub-type masks. All other pairs fall back to
    the original auto_distill_malignancy_loss path — unchanged.
    """
    plan, strip_idx_set = _build_plan(classes, subtype_map)

    # No sub-type channels → fully transparent pass-through.
    if not plan:
        return auto_distill_malignancy_loss(
            model_output=model_output, malig_benign=malig_benign, unk_voxels=unk_voxels,
            classes=classes, label=label, sizes_malignancy=sizes_malignancy,
            malignancy_per_voxel=malignancy_per_voxel, chosen_segment_mask=chosen_segment_mask,
            sigmoid_already_applied=sigmoid_already_applied, class_weights=class_weights,
            triangle_consistency=triangle_consistency, input_tensor=input_tensor, names=names,
            skip_cls=skip_cls, include_ball_loss=include_ball_loss,
            tumor_volumes=tumor_volumes, tumor_diameters=tumor_diameters,
            sizes_slices=sizes_slices, ct_z_spacing_original=ct_z_spacing_original,
            slices_mask=slices_mask, max_slice=max_slice,
            sample_weights=sample_weights, subseg_dilation=subseg_dilation,
        )

    device = label.device
    B = label.shape[0]

    # If this batch MIGHT trigger a sanity dump (counter not exhausted
    # and we're on rank 0), retain grad on the model outputs up-front
    # — before any loss op touches them. retain_grad() only has to be
    # called before `.backward()`, but moving it here (immediately after
    # we have references to the output tensors, before any loss math)
    # is the safe idiomatic placement: no loss op can accidentally
    # strip the graph, and we don't have to worry about intermediate
    # autograd.grad calls inside losses. The stash + NIfTI write still
    # happens later inside `_write_sanity_dumps()` (which is the true
    # gate for whether we actually save anything).
    if counter_subtype < SANITY_SUBTYPE_MAX_DUMPS and _is_main_rank():
        _seg_full = model_output.get('segmentation')
        _seg_finest = (_seg_full[0] if isinstance(_seg_full, (list, tuple))
                       else _seg_full)
        _mb_finest = (malig_benign[0] if isinstance(malig_benign, (list, tuple))
                      else malig_benign)
        _cls_finests = [v[0] if isinstance(v, (list, tuple)) else v
                        for k, v in model_output.items() if 'classif' in k]
        for _t in (_seg_finest, _mb_finest, *_cls_finests):
            if _t is not None and getattr(_t, 'requires_grad', False):
                _t.retain_grad()

    # (1) Per-sample flag: does this sample have any sub-type mask voxels for this organ?
    organ_active = []
    for p in plan:
        has = torch.zeros(B, dtype=torch.bool, device=device)
        for ci in p['mal_idx'] + p['ben_idx']:
            has = has | (label[:, ci].float().sum(dim=(-1,-2,-3)) > 0)
        organ_active.append(has)

    # (2) Suppress scalar malignancy signal for routed (sample, organ) pairs. These are atlas cases with pdac/pnet/cyst. We will treat them in (4).
    mpv = malignancy_per_voxel.clone()
    for p, has in zip(plan, organ_active):
        if has.any():
            mpv[has, p['lesion_idx']] = float('nan')

    # (3) Strip sub-type channels from all per-class tensors before the old loss.
    keep = [i for i in range(len(classes)) if i not in strip_idx_set]
    classes_old = [classes[i] for i in keep]
    def _slice(t): return t[:, keep] if t is not None else None

    out_full = model_output['segmentation']
    if isinstance(out_full, (list, tuple)):
        out_sliced = [o[:, keep] for o in out_full]
    else:
        out_sliced = _slice(out_full)
    mo_sliced = {**model_output, 'segmentation': out_sliced}

    # Populated in-place by auto_distill_malignancy_loss when ball_loss runs
    # (i.e. include_ball_loss=True and the batch isn't fully skipped). Keys:
    # 'pseudo_mask_malignant', 'pseudo_mask_benign',
    # 'penalize_malignant',    'penalize_benign'
    # — each shape (B, len(classes_old), D, H, W). classes_old: just the clases w/o cyst/ pdac/ pnet
    ball_masks_out = {}

    # Per-(B, L) loss / label / penalize breakdowns from the OLD path.
    # Populated in-place by auto_distill_malignancy_loss. Used by the sanity
    # dump to cross-check what was penalized by the OLD vs NEW paths and
    # to detect double-counting on shared channels.
    old_losses_per_bc = {}

    #our old loss, supervises malignant/benign channels
    old_loss = auto_distill_malignancy_loss(
        model_output=mo_sliced, malig_benign=malig_benign, unk_voxels=_slice(unk_voxels),
        classes=classes_old, label=_slice(label), sizes_malignancy=sizes_malignancy,
        malignancy_per_voxel=_slice(mpv), chosen_segment_mask=_slice(chosen_segment_mask),
        sigmoid_already_applied=sigmoid_already_applied, class_weights=_slice(class_weights),
        triangle_consistency=triangle_consistency, input_tensor=input_tensor, names=names,
        skip_cls=skip_cls, include_ball_loss=include_ball_loss,
        tumor_volumes=tumor_volumes, tumor_diameters=tumor_diameters,
        sizes_slices=sizes_slices, ct_z_spacing_original=ct_z_spacing_original,
        slices_mask=slices_mask, max_slice=max_slice,
        sample_weights=sample_weights, subseg_dilation=subseg_dilation,
        ball_masks_out=ball_masks_out,
        losses_per_bc_out=old_losses_per_bc,
    )

    # (4) Here, we supervise Atlas cases with cyst/dpac/pnet, using these 3 masks to 
    # create malignant/benign masks
    # New standard loss on sub-type samples (per organ).
    # Under deep supervision, `split_outputs_malignancy` returns `malig_benign`
    # as a *list* of per-scale tensors (finest first, per medformer's
    # `[out, aux_out]` convention — both scales are at input resolution
    # because aux_out is pre-interpolated inside the model; see
    # model/dim3/medformer.py:696). `auto_distill_malignancy_loss` supervises
    # EVERY scale, and this wrapper must do the same. We build mal_label /
    # ben_label ONCE from the per-voxel sub-type labels (at input resolution),
    # then apply the same supervision to each scale's malig/benign output.
    # If a scale happens to be at a coarser resolution (not medformer's
    # convention but safe to support), labels + penalize are F.interpolate'd
    # with nearest-neighbor per scale.
    malig_benign_scales = (list(malig_benign)
                           if isinstance(malig_benign, (list, tuple))
                           else [malig_benign])
    malig_benign_finest = malig_benign_scales[0]

    lesion_classes = [c for c in sorted(classes_old) if 'lesion' in c]
    L = len(lesion_classes)
    malign_out = malig_benign_finest[:, :L]
    benign_out = malig_benign_finest[:, L:]

    mal_label = torch.zeros_like(malign_out)
    ben_label = torch.zeros_like(benign_out)
    # Per-(sample, lesion-channel) activity flag. True iff this (sample, organ)
    # pair is routed to the new loss; 0 elsewhere so we don't conflict with the
    # old loss's supervision for other organs on the same sample.
    active_cell = torch.zeros(B, L, dtype=torch.bool, device=device)

    for p, has in zip(plan, organ_active):
        if p['lesion_class'] not in lesion_classes:
            continue
        li = lesion_classes.index(p['lesion_class'])
        for ci in p['mal_idx']: #add all malig subtypes to the malig label (union)
            mal_label[:, li] = torch.clamp(mal_label[:, li] + label[:, ci].float(), 0, 1)
        for ci in p['ben_idx']: #add all benign subtypes to the benign label (union)
            ben_label[:, li] = torch.clamp(ben_label[:, li] + label[:, ci].float(), 0, 1)
        active_cell[has, li] = True

    # sanity: disjoint
    assert (mal_label + ben_label).max() <= 1.0 + 1e-6

    mal_label = mal_label.detach()
    ben_label = ben_label.detach()

    # Per-(sample, lesion-channel) penalize mask, restricted to LESION voxels
    # only. The old auto_distill path retains unique responsibility for
    # background voxels on these samples (section (2) NaN's out `mpv` at the
    # lesion column, which makes auto_distill's `penalize_known_malignancy`
    # become 0 AT lesion voxels and 1 at background — pushing malig_out /
    # benign_out → 0 as background). If we also penalized the full crop here,
    # background voxels would get a double push-to-0 from the two paths.
    # By gating section (4) to lesion voxels (mal_label ∪ ben_label) we keep
    # each (sample, organ, voxel) supervised by EXACTLY ONE path:
    #   - lesion voxels (pdac/pnet/cyst): section (4) → spatial malig/benign.
    #   - background voxels (inside crop, outside lesion): old path → push 0.
    # See test_BB_no_double_penalization_atlas_subtype for the regression guard.
    lesion_mask_spatial = ((mal_label > 0) | (ben_label > 0)).to(malign_out.dtype)
    penalize = (active_cell.view(B, L, 1, 1, 1).to(malign_out.dtype)
                * lesion_mask_spatial)

    # Lesion-channel class weights at the finest scale (rebuilt per scale
    # inside the loop below when spatial shapes differ).
    # `class_weights` is (1, C_full, 1, 1, 1); slice to the stripped channel
    # layout (`classes_old`) before collapsing to lesion channels, otherwise
    # `get_lesion_channels` will trip its shape assertion.
    cw_lesion = _lesion_class_weights(_slice(class_weights), classes_old, malign_out.shape)

    # Per-sample reductions so sample_weights (shape (B,)) can weight losses
    # matching the pattern used in auto_distill_malignancy_loss / ball_loss.
    def _apply_sw(per_sample_loss):
        if sample_weights is not None:
            assert per_sample_loss.shape == sample_weights.shape, \
                f'per-sample loss shape {per_sample_loss.shape} does not match sample_weights shape {sample_weights.shape}'
            per_sample_loss = per_sample_loss * sample_weights
        return per_sample_loss.mean()

    # ---- Per-scale deep-supervision loop ----
    # Reuse the SAME mal_label / ben_label / penalize across scales (the
    # per-voxel sub-type labels don't change between output levels).
    bce_m_per_scale  = []
    bce_b_per_scale  = []
    dice_m_per_scale = []
    dice_b_per_scale = []
    # Keep finest-scale per-voxel maps for the sanity-dump block below — it
    # writes `bce_malignant_B{b}.nii.gz` and `malignant_output_B{b}.nii.gz`
    # using these references.
    bce_m = bce_b = dice_m = dice_b = None

    for scale_idx, mb in enumerate(malig_benign_scales):
        m_out = mb[:, :L]
        b_out = mb[:, L:]

        # Medformer keeps both scales at input resolution, so the no-resize
        # branch is the common path. The else-branch handles non-medformer
        # deep-sup conventions.
        if m_out.shape[-3:] == mal_label.shape[-3:]:
            mal_s, ben_s, pen_s, cw_s = mal_label, ben_label, penalize, cw_lesion
        else:
            size = tuple(m_out.shape[-3:])
            mal_s = F.interpolate(mal_label, size=size, mode='nearest')
            ben_s = F.interpolate(ben_label, size=size, mode='nearest')
            pen_s = F.interpolate(penalize,  size=size, mode='nearest')
            cw_s  = _lesion_class_weights(_slice(class_weights), classes_old, m_out.shape)

        if not sigmoid_already_applied:
            bce_m_s = F.binary_cross_entropy_with_logits(
                m_out, mal_s.float(), reduction='none', weight=cw_s)
            bce_b_s = F.binary_cross_entropy_with_logits(
                b_out, ben_s.float(), reduction='none', weight=cw_s)
        else:
            bce_m_s = F.binary_cross_entropy(
                m_out, mal_s.float(), reduction='none', weight=cw_s)
            bce_b_s = F.binary_cross_entropy(
                b_out, ben_s.float(), reduction='none', weight=cw_s)
        bce_m_s = bce_m_s * pen_s
        bce_b_s = bce_b_s * pen_s

        dice_m_s = DiceLossMultiClass(
            m_out, mal_s, pen_s,
            sigmoid=(not sigmoid_already_applied), class_weights=cw_s,
            size_average=False)
        dice_b_s = DiceLossMultiClass(
            b_out, ben_s, pen_s,
            sigmoid=(not sigmoid_already_applied), class_weights=cw_s,
            size_average=False)

        bce_m_per_scale.append(bce_m_s.mean(dim=(-1, -2, -3, -4)))   # (B,)
        bce_b_per_scale.append(bce_b_s.mean(dim=(-1, -2, -3, -4)))   # (B,)
        dice_m_per_scale.append(dice_m_s.mean(dim=-1))               # (B,)
        dice_b_per_scale.append(dice_b_s.mean(dim=-1))               # (B,)

        if scale_idx == 0:
            # Retain finest-scale per-voxel maps for the sanity-dump block.
            bce_m, bce_b = bce_m_s, bce_b_s
            dice_m, dice_b = dice_m_s, dice_b_s

    # Mean across scales per sample, then sample-weighted reduce to scalar.
    bce_m_per_sample  = torch.stack(bce_m_per_scale,  dim=0).mean(dim=0)
    bce_b_per_sample  = torch.stack(bce_b_per_scale,  dim=0).mean(dim=0)
    dice_m_per_sample = torch.stack(dice_m_per_scale, dim=0).mean(dim=0)
    dice_b_per_sample = torch.stack(dice_b_per_scale, dim=0).mean(dim=0)

    combined = dict(old_loss)
    combined['loss_subtype_malig_bce']   = _apply_sw(bce_m_per_sample)  * subtype_loss_weight
    combined['loss_subtype_benign_bce']  = _apply_sw(bce_b_per_sample)  * subtype_loss_weight
    combined['loss_subtype_malig_dice']  = _apply_sw(dice_m_per_sample) * subtype_loss_weight
    combined['loss_subtype_benign_dice'] = _apply_sw(dice_b_per_sample) * subtype_loss_weight

    # (5) Classification loss for sub-type samples on the sub-type organs only.
    # For those (sample, organ) cells the old cls loss inside
    # auto_distill_malignancy_loss contributes 0 (we set malignancy_per_voxel=NaN,
    # so those cells fall into the "unknown" branch). We fill that gap here.
    if not skip_cls:
        malignant_label_cls = (mal_label.sum(dim=(-1, -2, -3)) > 0).float()   # (B, L)
        benign_label_cls    = (ben_label.sum(dim=(-1, -2, -3)) > 0).float()   # (B, L)

        mask_f = active_cell.float()                                          # (B, L)
        per_sample_denom = mask_f.sum(dim=1).clamp_min(1.0)                   # (B,)

        # Collect per-sample cls loss across all cls keys + deep-sup scales so
        # the sanity dump can report a per-sample scalar (pre-sample-weighting)
        # instead of the batch-level `_apply_sw(...)` number.
        _s5_cls_per_sample_vals = []
        for key in model_output:
            if 'malig_benign_cls' not in key:
                continue
            classif_out = model_output[key]
            if not isinstance(classif_out, (list, tuple)):
                classif_out = [classif_out]
            vals = []
            for out_cls in classif_out:
                mal_out_cls = out_cls[:, :L]
                ben_out_cls = out_cls[:, L:]
                l_m = F.binary_cross_entropy_with_logits(
                    mal_out_cls, malignant_label_cls, reduction='none')
                l_b = F.binary_cross_entropy_with_logits(
                    ben_out_cls, benign_label_cls, reduction='none')
                l_m_per_sample = (l_m * mask_f).sum(dim=1) / per_sample_denom
                l_b_per_sample = (l_b * mask_f).sum(dim=1) / per_sample_denom
                _per_sample_this = (l_m_per_sample + l_b_per_sample) / 2
                _s5_cls_per_sample_vals.append(_per_sample_this.detach())
                vals.append(_apply_sw(_per_sample_this))
            # Distinct key so the old cls entry from auto_distill_malignancy_loss is untouched.
            combined[key + '_subtype'] = torch.stack(vals, dim=0).mean(0) * subtype_loss_weight
        if _s5_cls_per_sample_vals:
            _s5_cls_per_sample = torch.stack(_s5_cls_per_sample_vals, dim=0).mean(dim=0)  # (B,)
        else:
            _s5_cls_per_sample = None
    else:
        _s5_cls_per_sample = None

    # (5c) UFO sub-type classification supervision.
    #     Mirror of (5b) for the lesion-side classification head: for UFO
    #     samples with a pure pdac / pnet / cyst report, supervise the three
    #     sub-type cls channels (pdac=1 / others=0 per tumor_type). The
    #     existing classification_loss already does this for atlas samples
    #     via label[:, subtype_idx].sum()>0; here we fill the gap for UFO,
    #     where label is zero and unk_voxels masks the classification_loss
    #     out. Uses the same per-sample gating as the seg loss; atlas
    #     samples with per-voxel sub-type masks are skipped.
    # Enabled iff all the inputs are in place. The key-emission loop below
    # runs even when this is False so that `loss_meters` (initialized from
    # the first batch's key set) never sees a later batch add new keys.
    subtype_cls_loss_enabled = (
        (not skip_cls)
        and ('pancreatic_pdac' in classes) and ('pancreatic_pnet' in classes)
        and ('pancreatic_cyst' in classes)
        and (tumor_type is not None) and (tumor_type_organ is not None)
    )
    # Only consider cls keys the wrapper is responsible for (mirrors the
    # condition inside the loop below). Guarantees the same key set every call.
    _ufo_cls_keys = [k for k in model_output
                     if ('classif' in k) and ('malig_benign_cls' not in k)]#we have classification in multiple points of the architecture
    _zero_scalar = torch.zeros((), device=device, dtype=torch.float32)
    for k in _ufo_cls_keys:
        combined[k + '_ufo_subtype_cls'] = _zero_scalar
    # Outer-scope placeholders for the per-sub-type cls (target, mask) —
    # populated by the section 5c block below when enabled, otherwise left
    # as None so the sanity dump can detect "block didn't fire for this
    # sample". Shape: (B, 3) each; order matches ufo channel layout
    # [pdac, pnet, cyst].
    _s5c_target_cls_full = None
    _s5c_mask_cls_full   = None
    if subtype_cls_loss_enabled:
        pdac_idx_cls = classes.index('pancreatic_pdac')
        pnet_idx_cls = classes.index('pancreatic_pnet')
        cyst_idx_cls = classes.index('pancreatic_cyst')
        # Columns in the (post-split) lesion-side cls head; must match
        # split_cls_outputs_malignancy's `cls_head_main` (sorted filter).
        cls_head_main = [c for c in sorted(classes)
                         if (('background' in c) or ('lesion' in c)
                             or ('pdac' in c) or ('pnet' in c) or ('cyst' in c))]
        cls_pdac_col = cls_head_main.index('pancreatic_pdac')
        cls_pnet_col = cls_head_main.index('pancreatic_pnet')
        cls_cyst_col = cls_head_main.index('pancreatic_cyst')

        # Atlas gating (cheap, independent of seg block). We skip atlas here. This block is for UFO.
        subtype_label_sum_cls = (
            label[:, pdac_idx_cls].sum(dim=(-1, -2, -3))
            + label[:, pnet_idx_cls].sum(dim=(-1, -2, -3))
            + label[:, cyst_idx_cls].sum(dim=(-1, -2, -3))
        )
        atlas_has_subtype_cls = (subtype_label_sum_cls > 0)

        # Crop-success gating: cls is a per-CROP label ("does this crop contain
        # a sub-type X?"). For UFO samples we can only answer that question
        # when the crop was successfully targeted on pancreas — otherwise the
        # tumor location relative to the crop is unknown (the cyst could be
        # entirely outside the crop window, in which case pushing cyst_cls=1
        # would teach the model "there's a cyst here" when there isn't).
        # `chosen_segment_mask` is the dataset's "crop was organ-targeted"
        # signal: it's populated by get_chosen_segment_mask only when
        # tumor_segment != 'random' (dataset_..._UFO_multi_tumor.py line
        # 2522-2524). For UFO samples it's the right signal; for atlas
        # samples it's zero by design — but atlas samples are gated out
        # below by `atlas_has_subtype_cls`, so this check applies only to
        # the UFO branch where it's meaningful.
        pan_les_in_classes = ('pancreatic_lesion' in classes)
        if pan_les_in_classes:
            pan_les_idx_cls = classes.index('pancreatic_lesion')
            crop_targeted_pancreas_cls = (
                chosen_segment_mask[:, pan_les_idx_cls].sum(dim=(-1, -2, -3)) > 0
            )  # (B,) bool
        else:
            # Defensive: without a pancreatic_lesion channel we have no signal
            # to gate on — fall back to the old behavior (no crop check).
            crop_targeted_pancreas_cls = torch.ones(B, dtype=torch.bool, device=device)

        # Build per-sample (B, 3) target + mask for (pdac, pnet, cyst).
        target_cls = torch.zeros(B, 3, device=device, dtype=malig_benign_finest.dtype)
        mask_cls   = torch.zeros(B, 3, device=device, dtype=malig_benign_finest.dtype)
        for b in range(B):
            if tumor_type_organ[b] != 'pancreas':
                continue
            if bool(atlas_has_subtype_cls[b].item()):
                continue
            if not bool(crop_targeted_pancreas_cls[b].item()):
                # Crop was random / not pancreas-targeted on this UFO sample.
                # We cannot supervise cls because the tumor may or may not be
                # inside the crop window. Skip this sample; the absent
                # sub-types' "push to 0" signal would be unreliable too
                # (we'd be teaching the model the tumor sub-types aren't
                # present in a crop where the present sub-type might
                # actually be).
                continue
            # Per-sub-type presence parse. cls is multi-label BCE so we can
            # apply PARTIAL supervision: the *known* channels get pushed to 1
            # (their presence is certain from the report), and when the
            # composition also contains 'unknown' we simply don't penalize the
            # channels we did NOT observe — the untyped tumor could belong to
            # any of them, so pushing absent channels to 0 would be wrong.
            #
            # 'other' (typed tumor that didn't map to pdac/pnet/cyst) is
            # handled like 'unknown' at the cls level: we don't fully trust
            # the 'other' bucket to be negative for all three sub-types
            # (e.g. near-cyst fluid collections, borderline cases) — if any
            # non-known tag is present, un-observed channels go un-penalized.
            tt_set = set(tumor_type[b].split('+'))
            known = tt_set & {'pdac', 'pnet', 'cyst'}
            if not known:
                # No known sub-type to push to 1 (pure 'unknown', pure 'other',
                # or any mix with no known). Skip the sample — nothing we can
                # confidently supervise on.
                continue
            has_uncertain = bool(tt_set & {'other', 'unknown'})
            for i, k in enumerate(('pdac', 'pnet', 'cyst')):
                if k in known:
                    # Present in the report → push prediction to 1.
                    target_cls[b, i] = 1.0
                    mask_cls[b, i]   = 1.0
                elif not has_uncertain:
                    # Confidently absent (the whole composition is known sub-
                    # types) → push to 0.
                    # target stays 0; mask=1 enables the penalty.
                    mask_cls[b, i]   = 1.0
                # else: target and mask both stay 0 → no contribution for this
                # channel on this sample (we can't say it's absent when an
                # un-typed / other-type tumor is also present).

        per_sample_denom_cls = mask_cls.sum(dim=1).clamp_min(1.0)  # (B,)
        # Expose for the sanity dump's supervision matrix.
        _s5c_target_cls_full = target_cls.detach()
        _s5c_mask_cls_full   = mask_cls.detach()

        # Collect per-sample UFO cls loss across all cls keys + deep-sup scales.
        _s5c_cls_per_sample_vals = []
        for key in model_output:
            if 'classif' not in key:
                continue
            if 'malig_benign_cls' in key:
                continue   # handled by the malig/benign cls block above
            cls_out = model_output[key]
            if not isinstance(cls_out, (list, tuple)):
                cls_out_list = [cls_out]
            else:
                cls_out_list = cls_out
            vals = []
            for co in cls_out_list:
                assert co.shape == (B, len(cls_head_main)), (
                    f'expected cls output shape (B={B}, len(cls_head_main)='
                    f'{len(cls_head_main)}) so cls_pdac_col={cls_pdac_col} / '
                    f'cls_pnet_col={cls_pnet_col} / cls_cyst_col={cls_cyst_col} '
                    f'are valid column indices, but got {tuple(co.shape)} for '
                    f'key {key!r}'
                )
                co_sub = co[:, [cls_pdac_col, cls_pnet_col, cls_cyst_col]]  # (B, 3)
                l = F.binary_cross_entropy_with_logits(
                    co_sub, target_cls, reduction='none')
                l_per_sample = (l * mask_cls).sum(dim=1) / per_sample_denom_cls
                _s5c_cls_per_sample_vals.append(l_per_sample.detach())
                vals.append(_apply_sw(l_per_sample))
            combined[key + '_ufo_subtype_cls'] = torch.stack(vals, dim=0).mean(0) * subtype_loss_weight
        if _s5c_cls_per_sample_vals:
            _s5c_cls_per_sample = torch.stack(_s5c_cls_per_sample_vals, dim=0).mean(dim=0)  # (B,)
        else:
            _s5c_cls_per_sample = None
    else:
        _s5c_cls_per_sample = None

    # (5b) UFO sub-type segmentation supervision.
    #     When the per-tumor report metadata pins down a pure sub-type
    #     (tumor_type in {'pdac','pnet','cyst'} and tumor_type_organ == 'pancreas'),
    #     we have no per-voxel label for that sub-type on UFO samples. But the
    #     ball_loss has already built a tolerance-aware pseudo-mask for the
    #     malignant / benign side of the same crop. Reuse that mask to supervise
    #     the sub-type segmentation channel directly. Atlas samples (those that
    #     already carry per-voxel sub-type labels) are gated off so the more
    #     precise main seg loss handles them.
    subtype_seg_loss_enabled = (
        ('pancreatic_pdac' in classes) and ('pancreatic_pnet' in classes)
        and ('pancreatic_cyst' in classes)
        and (tumor_type is not None) and (tumor_type_organ is not None)
        and ('pseudo_mask_malignant' in ball_masks_out)
    )
    # Default placeholders so the sanity dump can reference them unconditionally.
    pan_les_col_old = None
    target = None
    penalize_seg = None
    atlas_has_subtype = None
    subtype_out = None
    # `target_full` / `penalize_full` are assigned inside the
    # `subtype_seg_loss_enabled` block; seed them here so the
    # sanity-dump's `_subtype_masks` closure doesn't raise NameError
    # when that block doesn't fire (e.g. ball_loss skipped on this
    # batch, so `pseudo_mask_malignant` isn't in `ball_masks_out`).
    target_full = None
    penalize_full = None
    # Finest-scale per-voxel UFO seg BCE + per-(sample, channel) Dice —
    # kept in scope here so the sanity-dump block can reference them even
    # when the UFO seg block doesn't fire (left as None in that case).
    ufo_bce_s_finest  = None
    ufo_dice_s_finest = None
    # Per-sample UFO seg loss (pre-sample-weighting). `None` when the block
    # doesn't fire. Used by the sanity dump to report this sample's own
    # contribution rather than the batch-level scalar.
    _s5b_bce_per_sample  = None
    _s5b_dice_per_sample = None
    # Seed UFO seg loss keys with zero so the key set is stable across
    # iterations even when this block is gated off (e.g. batches where
    # ball_loss didn't run). `loss_meters` in train_ddp is initialized from
    # the first batch's keys — if those keys disappear later we crash with
    # KeyError; if they APPEAR later we also crash. Emit unconditionally.
    _zero_seg = torch.zeros((), device=device, dtype=torch.float32)
    combined.setdefault('loss_ufo_subtype_seg_bce',  _zero_seg)
    combined.setdefault('loss_ufo_subtype_seg_dice', _zero_seg)
    if subtype_seg_loss_enabled:
        pdac_idx = classes.index('pancreatic_pdac')
        pnet_idx = classes.index('pancreatic_pnet')
        cyst_idx = classes.index('pancreatic_cyst')
        # ball_loss internally does `out = get_lesion_channels(out, classes)`
        # before allocating the batched masks, so the channel axis of
        # ball_masks_out follows get_lesion_channels' first-appearance order
        # on `classes_old` — one column per *_lesion (/ cyst / pdac / pnet)
        # class in `classes_old`. Our wrapper already stripped pdac/pnet/cyst
        # from classes_old, so this reduces to the list of bare `*_lesion`
        # classes in classes_old order, keys get the 'pancreatic'→'pancreas'
        # substitution.
        _lesion_cols = [c for c in classes_old
                        if any(s in c for s in ('lesion', 'cyst', 'pdac', 'pnet'))]
        pan_les_col_old = (_lesion_cols.index('pancreatic_lesion')
                           if 'pancreatic_lesion' in _lesion_cols else None)

        if pan_les_col_old is not None:
            # Atlas gating: skip samples that already have per-voxel sub-type masks.
            subtype_label_sum = (
                label[:, pdac_idx].sum(dim=(-1, -2, -3))
                + label[:, pnet_idx].sum(dim=(-1, -2, -3))
                + label[:, cyst_idx].sum(dim=(-1, -2, -3))
            )
            atlas_has_subtype = (subtype_label_sum > 0)  # (B,) bool

            # Ball-loss masks come from medformer's MAIN output (`scales[0]`),
            # NOT from aux_out. auto_distill_malignancy_loss recurses over
            # the scales list but is explicitly coded to populate
            # `ball_masks_out` only on the first iteration (idx == 0), so
            # the aux_out recursion cannot overwrite it. This matters
            # because ball_loss places ball centers at
            # `torch.argmax(out_spatial)` — out and aux_out differ in their
            # argmaxes, and we want the supervision target aligned with the
            # main output (what we evaluate, and what drives the finest-scale
            # loss in the wrapper's per-scale sub-type loop below).
            # Channel axis is 1 (singleton — the pancreas lesion column only).
            mal_mask = ball_masks_out['pseudo_mask_malignant'][:, pan_les_col_old:pan_les_col_old+1]
            mal_pen  = ball_masks_out['penalize_malignant']  [:, pan_les_col_old:pan_les_col_old+1]
            ben_mask = ball_masks_out['pseudo_mask_benign']  [:, pan_les_col_old:pan_les_col_old+1]
            ben_pen  = ball_masks_out['penalize_benign']     [:, pan_les_col_old:pan_les_col_old+1]

            # Non-target-channel penalize = pancreas sub-segment mask. For UFO cases
            # with a pancreatic tumor reported, `unk_voxels[pancreatic_*]` is
            # populated by assign_lesion_labels_from_report whenever ANY pancreas
            # sub-segment mask has voxels in the crop — INDEPENDENTLY of whether
            # the cropper targeted pancreas (chosen_segment_mask is the "targeted"
            # signal; unk_voxels is the "any incidental pancreas tissue" signal).
            # So this can be non-zero on random / non-targeted crops too.
            # This is OK because:
            #   - If the crop is random AND sizes_malignancy for this sample is
            #     all-padding (always true for 'random' crops via
            #     estimate_tumor_volume's short-circuit at line 4226), ball_loss
            #     produces EMPTY ball masks for this sample (mal_mask[b], mal_pen[b],
            #     ben_mask[b], ben_pen[b] all zero). The wrapper still supervises
            #     ABSENT sub-types by pushing them to 0 over non_target_pen — which
            #     is semantically correct (absent sub-types are absent everywhere,
            #     including over any incidental pancreas region we happen to see).
            #   - PRESENT sub-type gets no supervision on this sample (its target
            #     and penalize both resolve to 0 via the empty ball mask) — also
            #     correct: without a spatial ball mask we can't localize where to
            #     push the present sub-type to 1.
            # Shape: (B, 1, D, H, W).
            non_target_pen = unk_voxels[:, pdac_idx:pdac_idx+1].float()

            # Build (B, 3, D, H, W) target and penalize at the FINEST scale; we
            # downsample to each deep-supervision scale inside the scale loop.
            ref_shape = (B, 3, *mal_mask.shape[-3:])
            target_full = torch.zeros(ref_shape, device=mal_mask.device, dtype=mal_mask.dtype)
            penalize_full = torch.zeros_like(target_full)

            for b in range(B):
                if tumor_type_organ[b] != 'pancreas':
                    continue
                if bool(atlas_has_subtype[b].item()):
                    continue
                # Per-sub-type presence parse. tumor_type is either a bare
                # sub-type ('pdac'/'pnet'/'cyst'/'other') or a '+'-joined
                # composition for mixed reports (e.g. 'cyst+pdac').
                tt_set = set(tumor_type[b].split('+'))
                # Conservative: skip the whole sample if any unknown/other tag
                # is present — we cannot guarantee absence of un-mentioned
                # sub-types when the report contains a type we couldn't map.
                if tt_set & {'other', 'unknown'}:
                    continue
                known = tt_set & {'pdac', 'pnet', 'cyst'}
                if not known:
                    continue
                has_pdac = 'pdac' in known
                has_pnet = 'pnet' in known
                has_cyst = 'cyst' in known

                # pdac channel.
                if has_pdac and not has_pnet:
                    # Target = malignant ball mask; penalize = malig tolerance margin.
                    target_full[b, 0]   = mal_mask[b, 0]
                    penalize_full[b, 0] = mal_pen[b, 0]
                elif has_pdac and has_pnet:
                    # Both malignant types present — spatial assignment within
                    # the malignant mask is ambiguous. Skip pdac supervision.
                    pass
                else:  # confidently no pdac in this case
                    penalize_full[b, 0] = non_target_pen[b, 0]

                # pnet channel — symmetric to pdac.
                if has_pnet and not has_pdac:
                    target_full[b, 1]   = mal_mask[b, 0]
                    penalize_full[b, 1] = mal_pen[b, 0]
                elif has_pnet and has_pdac:
                    pass
                else:
                    penalize_full[b, 1] = non_target_pen[b, 0]

                # cyst channel — cyst is the only benign sub-type, so no
                # ambiguity on the benign side.
                if has_cyst:
                    target_full[b, 2]   = ben_mask[b, 0]
                    penalize_full[b, 2] = ben_pen[b, 0]
                else:
                    penalize_full[b, 2] = non_target_pen[b, 0]

            target_full = target_full.detach()
            penalize_full = penalize_full.detach()

            # Deep supervision: loop over every segmentation scale, resize the
            # masks with nearest-neighbor as needed. Per user: reuse the finest-
            # scale ball masks for all scales rather than rerunning ball_loss.
            full_seg = model_output['segmentation']
            scales = full_seg if isinstance(full_seg, (list, tuple)) else [full_seg]

            # `ball_masks_out` is built at the MAIN output's scale (scales[0])
            # — see the comment above and the matching idx==0 gate in
            # auto_distill_malignancy_loss. The supervision target we just
            # assembled from it is therefore at scales[0]'s spatial shape.
            # We assert matching shape against scales[0] (not against a max
            # across scales) so that if the deep-sup convention ever changes
            # — e.g. medformer starts emitting aux_out WITHOUT the internal
            # F.interpolate, or the list order flips to [aux_out, out] — we
            # fail loudly rather than silently supervising out's sub-type
            # channels with aux's ball masks.
            main_spatial = tuple(scales[0].shape[-3:])
            ball_spatial = tuple(ball_masks_out['pseudo_mask_malignant'].shape[-3:])
            assert ball_spatial == main_spatial, (
                f"ball_masks_out is at spatial shape {ball_spatial} but "
                f"medformer's main output (scales[0]) is {main_spatial}. The "
                f"wrapper assumes ball_loss ran at the MAIN output's scale "
                f"(idx==0 in auto_distill_malignancy_loss's recursion). If "
                f"this fires, either the deep-sup list order changed away "
                f"from [out, aux_out], or the idx==0 gate in "
                f"auto_distill_malignancy_loss got removed."
            )

            sub_channel_idx = [pdac_idx, pnet_idx, cyst_idx]

            # Finest-scale per-voxel UFO seg BCE + Dice — retained for the
            # sanity-dump block (saves them as per-(B, C) tensors so the
            # checker can inspect per-channel contributions).
            ufo_bce_s_finest  = None
            ufo_dice_s_finest = None

            if not penalize_full.any():
                # No sample routed to the new UFO path in this batch — emit
                # a literal 0 so the scalar loss is "fully unchanged" for
                # non-pancreas / non-UFO batches. (DiceLossMultiClass otherwise
                # returns 1.0 in the `target==pred==0` corner case via its
                # `1 - 0/smooth` fall-through; gradient is 0 but the scalar
                # value isn't.)
                zero = torch.zeros((), device=penalize_full.device,
                                   dtype=penalize_full.dtype)
                combined['loss_ufo_subtype_seg_bce']  = zero
                combined['loss_ufo_subtype_seg_dice'] = zero
                # Per-sample dumps should also see 0.0 (not None) so the YAML
                # is unambiguous: "this sample's UFO seg loss is zero".
                _zero_B = torch.zeros(B, device=penalize_full.device,
                                      dtype=penalize_full.dtype)
                _s5b_bce_per_sample  = _zero_B
                _s5b_dice_per_sample = _zero_B
            else:
                bce_per_scale = []
                dice_per_scale = []
                for scale_idx_ufo, scale_out in enumerate(scales):
                    scale_spatial = scale_out.shape[-3:]
                    if scale_spatial == target_full.shape[-3:]:
                        t_s = target_full
                        p_s = penalize_full
                    else:
                        t_s = F.interpolate(target_full, size=scale_spatial, mode='nearest')
                        p_s = F.interpolate(penalize_full, size=scale_spatial, mode='nearest')
                    sub_out_s = scale_out[:, sub_channel_idx]

                    if not sigmoid_already_applied:
                        bce_s = F.binary_cross_entropy_with_logits(
                            sub_out_s, t_s, reduction='none') * p_s
                    else:
                        bce_s = F.binary_cross_entropy(
                            sub_out_s, t_s, reduction='none') * p_s
                    dice_s = DiceLossMultiClass(
                        sub_out_s, t_s, p_s,
                        sigmoid=(not sigmoid_already_applied),
                        size_average=False)

                    # Per-(sample, channel) active flag — True iff this sample
                    # supervises this sub-type channel on this scale. A channel
                    # with `p_s[b, c].sum() == 0` contributes:
                    #   - BCE: 0 at every voxel (BCE × 0), but the spatial mean
                    #     still divides by D×H×W → a 0 numerator that dilutes
                    #     the per-sample BCE toward 0 (wrong reported value).
                    #   - Dice: DiceLossMultiClass returns 1.0 via its
                    #     `1 - 0/(0+smooth)` fall-through → adds a constant +1
                    #     per empty channel to the per-sample dice mean.
                    # Both artifacts are pure SCALAR noise (zero gradient), but
                    # they inflate/deflate the REPORTED loss values. Mask the
                    # per-channel reductions so we only average across channels
                    # that actually saw supervision on this scale.
                    channel_active = (p_s.sum(dim=(-1, -2, -3)) > 0).float()       # (B, 3)
                    active_count   = channel_active.sum(dim=-1).clamp_min(1.0)     # (B,)

                    bce_per_bc  = bce_s.mean(dim=(-1, -2, -3))                      # (B, 3)
                    dice_per_bc = dice_s                                            # (B, 3)

                    bce_per_scale.append(
                        (bce_per_bc * channel_active).sum(dim=-1) / active_count)   # (B,)
                    dice_per_scale.append(
                        (dice_per_bc * channel_active).sum(dim=-1) / active_count)  # (B,)

                    if scale_idx_ufo == 0:
                        # Finest scale: retain per-voxel BCE (B, 3, D, H, W)
                        # and per-(sample, channel) Dice (B, 3) for the dump.
                        ufo_bce_s_finest  = bce_s
                        ufo_dice_s_finest = dice_s

                bce_per_sample  = torch.stack(bce_per_scale,  dim=0).mean(dim=0)
                dice_per_sample = torch.stack(dice_per_scale, dim=0).mean(dim=0)
                _s5b_bce_per_sample  = bce_per_sample.detach()
                _s5b_dice_per_sample = dice_per_sample.detach()
                combined['loss_ufo_subtype_seg_bce']  = _apply_sw(bce_per_sample)  * subtype_loss_weight
                combined['loss_ufo_subtype_seg_dice'] = _apply_sw(dice_per_sample) * subtype_loss_weight

            # Keep references so the sanity dump block can save them.
            subtype_out = scales[0][:, sub_channel_idx]
            target = target_full
            penalize_seg = penalize_full

    # (5d) Sub-type triangle-consistency loss — mirrors the structure of
    #      `auto_distill_malignancy_loss`'s triangle (the one that
    #      enforces σ(mal) + σ(ben) ≈ σ(lesion) at unknown-malignancy
    #      voxels), but applied to the three pancreatic sub-types:
    #          σ(pdac) + σ(pnet) + σ(cyst) ≈ σ(lesion)
    #      at voxels where a lesion may be present (via atlas label OR
    #      UFO soft distillation `sigmoid(les) × unk_voxels`) but no
    #      localized sub-type label fires (no atlas per-voxel truth AND
    #      no section-5b ball pseudo-mask covers the voxel).
    #
    #      Purpose: fills in supervision for the "lesion probably here,
    #      sub-type not localized" regions (UFO unk region outside the
    #      ball, or ball_skipped batches). Pushes the model to commit
    #      its lesion-presence mass to some sub-type instead of leaving
    #      all three at 0 while σ(lesion) stays positive.
    #
    #      Complementary to standard_seg/section-5b: at voxels where
    #      those DO fire with a concrete target, the mask is 0 and the
    #      triangle is silent (no conflicting gradients). The triangle
    #      is a no-op on atlas healthy, UFO samples with no pancreas in
    #      crop, and any voxel already routed to a sub-type supervision.
    _zero_tri = torch.zeros((), device=device, dtype=torch.float32)
    combined.setdefault('loss_subtype_triangle_consistency', _zero_tri)
    # Per-sample diagnostics surfaced for the sanity dump — seeded here
    # so the dump can reference them unconditionally (and see 0 / empty
    # when the sub-type triangle doesn't fire).
    _s5d_loss_per_sample        = torch.zeros(B, device=device)
    _s5d_mask_voxels_per_sample = torch.zeros(B, device=device)
    _s5d_enabled                = False
    triangle_subtype_enabled = (
        triangle_consistency
        and ('pancreatic_lesion' in classes)
        and ('pancreatic_pdac' in classes)
        and ('pancreatic_pnet' in classes)
        and ('pancreatic_cyst' in classes)
    )
    if triangle_subtype_enabled:
        pan_les_idx = classes.index('pancreatic_lesion')
        pan_pdac_idx = classes.index('pancreatic_pdac')
        pan_pnet_idx = classes.index('pancreatic_pnet')
        pan_cyst_idx = classes.index('pancreatic_cyst')

        # Use the MAIN (finest) scale only — matches the OLD triangle,
        # which runs once per call at input resolution.
        _seg_full_tri = model_output.get('segmentation')
        seg_finest_tri = (_seg_full_tri[0]
                          if isinstance(_seg_full_tri, (list, tuple))
                          else _seg_full_tri)

        les_logit  = seg_finest_tri[:, pan_les_idx:pan_les_idx + 1]
        pdac_logit = seg_finest_tri[:, pan_pdac_idx:pan_pdac_idx + 1]
        pnet_logit = seg_finest_tri[:, pan_pnet_idx:pan_pnet_idx + 1]
        cyst_logit = seg_finest_tri[:, pan_cyst_idx:pan_cyst_idx + 1]

        # Sigmoid handling mirrors OLD's triangle — lesion detached so
        # gradients only flow into the sub-type heads.
        if not sigmoid_already_applied:
            les_sig  = torch.sigmoid(les_logit).detach()
            pdac_sig = torch.sigmoid(pdac_logit)
            pnet_sig = torch.sigmoid(pnet_logit)
            cyst_sig = torch.sigmoid(cyst_logit)
        else:
            les_sig  = les_logit.detach()
            pdac_sig = pdac_logit
            pnet_sig = pnet_logit
            cyst_sig = cyst_logit

        # Reconstruct lesion_label the same way OLD auto_distill does:
        # atlas per-voxel lesion label ∪ soft distillation over the
        # pancreas unk region. `unk_voxels` here is at input resolution
        # and full-channels, so direct indexing is safe.
        les_label_tri = (
            label[:, pan_les_idx:pan_les_idx + 1].float()
            + les_sig * unk_voxels[:, pan_les_idx:pan_les_idx + 1].float()
        ).clamp(0, 1).detach()

        # Per-voxel sub-type labels: atlas per-voxel truth (zero on UFO),
        # plus the section-5b ball pseudo-masks when they fired. The
        # `target_full` tensor from section 5b has channel order
        # [pdac, pnet, cyst] (see L690 where it was constructed).
        pdac_label_tri = label[:, pan_pdac_idx:pan_pdac_idx + 1].float()
        pnet_label_tri = label[:, pan_pnet_idx:pan_pnet_idx + 1].float()
        cyst_label_tri = label[:, pan_cyst_idx:pan_cyst_idx + 1].float()
        if target_full is not None:
            pdac_label_tri = pdac_label_tri + target_full[:, 0:1].float()
            pnet_label_tri = pnet_label_tri + target_full[:, 1:2].float()
            cyst_label_tri = cyst_label_tri + target_full[:, 2:3].float()
        pdac_label_tri = pdac_label_tri.clamp(0, 1).detach()
        pnet_label_tri = pnet_label_tri.clamp(0, 1).detach()
        cyst_label_tri = cyst_label_tri.clamp(0, 1).detach()

        # Mask: lesion-possible voxels that nothing else supervises as
        # a specific sub-type. Clamped ≥ 0 (the subtractive construction
        # can go negative if per-voxel labels spill outside lesion — in
        # practice they don't, but the clamp is a belt-and-suspenders).
        unknown_subtype_label = (
            (les_label_tri > 0).float()
            - (pdac_label_tri > 0).float()
            - (pnet_label_tri > 0).float()
            - (cyst_label_tri > 0).float()
        ).clamp_min(0).detach()

        pred_sum_tri  = pdac_sig + pnet_sig + cyst_sig
        loss_tri      = torch.abs(pred_sum_tri - les_sig) * unknown_subtype_label.detach()

        # Masked mean per sample — same normalization as the new OLD
        # triangle. clamp_min(1.0) keeps samples with empty masks at
        # zero loss (no division by zero, no spurious contribution).
        _s5d_mask_voxels_per_sample = unknown_subtype_label.detach().sum(dim=(-1, -2, -3, -4))  # (B,)
        mask_sum_tri = _s5d_mask_voxels_per_sample.clamp_min(1.0)
        loss_sum_tri = loss_tri.sum(dim=(-1, -2, -3, -4))
        _s5d_loss_per_sample = (loss_sum_tri / mask_sum_tri).detach()  # (B,)
        _s5d_enabled = True
        combined['loss_subtype_triangle_consistency'] = (
            _apply_sw(loss_sum_tri / mask_sum_tri) * subtype_loss_weight
        )

    # (6) Sanity dump — encapsulated in a nested function so the loss
    #     computation above stays self-contained. The function captures all
    #     the state it needs (active_cell, mal_label, ben_label, penalize,
    #     bce_*/dice_* tensors, old_losses_per_bc, ball_masks_out, etc.)
    #     via closure; no working-path variables are renamed or re-routed.
    def _write_sanity_dumps():
        """Write NIfTI volumes + meta YAML for this call — mirrors the
        pattern in ball_loss / auto_distill_malignancy_loss. Runs for the
        first SANITY_SUBTYPE_MAX_DUMPS calls where at least one sample is
        routed to the new loss, on rank 0 only. All state is captured via
        closure from the enclosing `malignancy_loss_with_subtype` scope."""
        global counter_subtype
        if not (counter_subtype < SANITY_SUBTYPE_MAX_DUMPS
                and active_cell.any()
                and _is_main_rank()):
            return
        # Only rank 0 writes sanity dumps. Every DDP rank increments its own
        # `counter_subtype` (it's a per-process Python global), so if every
        # rank wrote, they'd collide on the same dump folder and the
        # conditional UFO-nifti writes would leave stale-looking files
        # (rank A writes UFO at its iteration N, rank B doesn't write UFO
        # at its iteration N if its batch's subtype_seg_loss_enabled is
        # False → rank A's UFO files survive rank B's partial overwrite
        # unless rmtree fires — and even with rmtree, the race is flaky).
        # Gating on rank 0 gives one clean writer per dump folder.
        counter_subtype += 1
        out_dir = os.path.join(SANITY_SUBTYPE_DIR, str(counter_subtype))
        # Belt-and-braces: even with the rank-0 gate, a previous training run
        # (different python invocation) may have left files here. Wipe
        # before each dump so we always start from a known empty state.
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)

        # ---- Per-batch diagnostics (same for every b in this call) ----
        # Whether ball_loss would be run by auto_distill_malignancy_loss on
        # this batch. ball_loss is skipped when NO sample has a known binary
        # malignancy flag in sizes_malignancy[..., 1] AND no voxel-level
        # malignancy is set — captured here via ball_masks_out existence
        # (populated iff ball_loss ran).
        batch_ball_loss_ran = bool('pseudo_mask_malignant' in ball_masks_out)

        for b in range(B):
            # Input CT volume (if provided)
            if input_tensor is not None:
                save_tensor_as_nifti(input_tensor[b].squeeze(),
                                     os.path.join(out_dir, f'input_volume_B{b}'))

            # Sub-type masks merged into malignant/benign labels (what we supervise on)
            save_tensor_as_nifti(mal_label[b].sum(0),
                                 os.path.join(out_dir, f'malignant_label_B{b}'))
            save_tensor_as_nifti(ben_label[b].sum(0),
                                 os.path.join(out_dir, f'benign_label_B{b}'))

            # Raw sub-type channels before merging (for traceability)
            for p in plan:
                for ci in p['mal_idx']:
                    save_tensor_as_nifti(label[b, ci].float(),
                                         os.path.join(out_dir, f'{classes[ci]}_B{b}'))
                for ci in p['ben_idx']:
                    save_tensor_as_nifti(label[b, ci].float(),
                                         os.path.join(out_dir, f'{classes[ci]}_B{b}'))

            # Penalize masks — one per loss path. Each says "where this
            # specific loss term multiplies in non-zero values for sample b".
            # Viewing all four in a NIfTI viewer at once makes it obvious:
            #   (a) which voxels the OLD malig/benign BCE+Dice loss covers,
            #   (b) which voxels section (4)'s new atlas-subtype BCE+Dice covers,
            #   (c) which voxels ball_loss's malig branch covers,
            #   (d) which voxels ball_loss's benign branch covers.
            # Plus the pre-existing `ufo_subtype_penalize_B*.nii.gz` below
            # (section 5b — UFO sub-type seg). Each should be visually
            # disjoint from the others (see test BB — no overlap invariant).

            # (b) Section (4) — atlas subtype malig/benign BCE+Dice. The
            # SAME mask multiplies BOTH `malig` AND `benign` BCE+Dice in
            # section (4) (penalize is built once per sample at the
            # lesion-voxel union), so a single NIfTI suffices. Named
            # explicitly to avoid ambiguity about which loss it gates.
            save_tensor_as_nifti(penalize[b].sum(0),
                                 os.path.join(out_dir, f'section4_penalize_malig_AND_benign_B{b}'))

            # (a) OLD path — `penalize_known_malignancy` in auto_distill.
            # Same mask gates both `bce_malig` and `bce_benign` (and their
            # dice counterparts) inside auto_distill_malignancy_loss. Exposed
            # via losses_per_bc_out['penalize_known_malignancy_mask'] from
            # the OLD call. Shape: (B, L_old, D, H, W) where L_old is the
            # lesion-only channel list (kidney_lesion, liver_lesion,
            # pancreatic_lesion) — NOT including sub-types.
            if 'penalize_known_malignancy_mask' in old_losses_per_bc:
                save_tensor_as_nifti(
                    old_losses_per_bc['penalize_known_malignancy_mask'][b].sum(0),
                    os.path.join(out_dir, f'old_penalize_malig_AND_benign_B{b}'))

            # (c, d) ball_loss has SEPARATE penalize masks for its malignant
            # and benign branches — unlike section (4) and the OLD path,
            # these two ARE different tensors (one per branch). Only present
            # when ball_loss ran for the batch.
            if 'penalize_malignant' in ball_masks_out:
                save_tensor_as_nifti(
                    ball_masks_out['penalize_malignant'][b].sum(0),
                    os.path.join(out_dir, f'ball_loss_penalize_malig_B{b}'))
            if 'penalize_benign' in ball_masks_out:
                save_tensor_as_nifti(
                    ball_masks_out['penalize_benign'][b].sum(0),
                    os.path.join(out_dir, f'ball_loss_penalize_benign_B{b}'))

            # Back-compat alias: keep the old name `penalize_B{b}.nii.gz`
            # (= section (4)'s penalize) so any external tools / notebooks
            # keyed on that filename don't break. Remove later once callers
            # migrate to the explicit names.
            save_tensor_as_nifti(penalize[b].sum(0),
                                 os.path.join(out_dir, f'penalize_B{b}'))

            # OLD auto_distill labels and BCE maps — mirror the same niftis
            # that auto_distill_malignancy_loss writes into its own
            # `SanityMalignancyLoss/<counter>/` folder, but co-located with
            # the new-loss dumps for direct side-by-side inspection. The
            # OLD path's `penalize_known_malignancy` excludes the lesion
            # voxels of routed (sample, organ) pairs (section (2) NaN's out
            # mpv for them), so `old_bce_malignant` and `old_bce_benign`
            # should be 0 over the same lesion voxels where section (4)'s
            # `bce_malignant` / `bce_benign` (already saved above) are
            # non-zero — verifying the clean handoff visually.
            for src_key, dst_name in [
                ('malignant_label_mask',          'old_malignant_label'),
                ('benign_label_mask',             'old_benign_label'),
                ('unknown_malignancy_label_mask', 'old_unknown_malignancy'),
                ('lesion_label_mask',             'old_lesion_label'),
                ('lesion_output_sigmoid_mask',    'old_lesion_output'),
                ('bce_malig_mask',                'old_bce_malignant'),
                ('bce_benign_mask',               'old_bce_benign'),
            ]:
                if src_key in old_losses_per_bc:
                    save_tensor_as_nifti(
                        old_losses_per_bc[src_key][b].sum(0),
                        os.path.join(out_dir, f'{dst_name}_B{b}'))

            # ball_loss pseudo-masks (what ball_loss tried to segment).
            # Same tensors the ball_loss dumps write into `SanityBallLoss/`,
            # but co-located. The malig pseudo_mask should spatially
            # overlap with section (5b)'s ufo_subtype_target (when UFO
            # samples route) and the ball's malig penalize (saved above)
            # should bound where `ball_loss_malignant_bce` fires.
            if 'pseudo_mask_malignant' in ball_masks_out:
                save_tensor_as_nifti(
                    ball_masks_out['pseudo_mask_malignant'][b].sum(0),
                    os.path.join(out_dir, f'ball_loss_pseudo_mask_malig_B{b}'))
            if 'pseudo_mask_benign' in ball_masks_out:
                save_tensor_as_nifti(
                    ball_masks_out['pseudo_mask_benign'][b].sum(0),
                    os.path.join(out_dir, f'ball_loss_pseudo_mask_benign_B{b}'))
            if 'pseudo_mask_lesion' in ball_masks_out:
                save_tensor_as_nifti(
                    ball_masks_out['pseudo_mask_lesion'][b].sum(0),
                    os.path.join(out_dir, f'ball_loss_pseudo_mask_lesion_B{b}'))

            # Model outputs (malignant / benign channels)
            if not sigmoid_already_applied:
                save_tensor_as_nifti(torch.sigmoid(malign_out[b]).sum(0),
                                     os.path.join(out_dir, f'malignant_output_B{b}'))
                save_tensor_as_nifti(torch.sigmoid(benign_out[b]).sum(0),
                                     os.path.join(out_dir, f'benign_output_B{b}'))
            else:
                save_tensor_as_nifti(malign_out[b].sum(0),
                                     os.path.join(out_dir, f'malignant_output_B{b}'))
                save_tensor_as_nifti(benign_out[b].sum(0),
                                     os.path.join(out_dir, f'benign_output_B{b}'))

            # Per-voxel BCE maps (after penalize multiplication)
            save_tensor_as_nifti(bce_m[b].sum(0),
                                 os.path.join(out_dir, f'bce_malignant_B{b}'))
            save_tensor_as_nifti(bce_b[b].sum(0),
                                 os.path.join(out_dir, f'bce_benign_B{b}'))

            # Metadata. Python dicts preserve insertion order, and PyYAML
            # dumps keys in that order by default — so the top of this dict
            # becomes the top of the YAML file. We lead with a compact
            # `summary` block that covers: (1) crop target organ, (2) source
            # (atlas vs UFO), (3) atlas sub-type / lesion label voxel counts
            # in the crop, (4) report-derived malignancy rows, (5) report-
            # derived sub-type composition, (6) what each new loss routed,
            # and then the raw detail below for deep dives.
            meta = {}
            _name = (names[b] if names is not None
                     and isinstance(names, (list, tuple))
                     and b < len(names) else None)

            # Source detection: path-based. JHH atlas files live under
            # .../JHH_lesion_types_medformer_npz/; Merlin UFO under
            # .../merlin_processed_rsuper/merlin_medformer_pancreas_npz/.
            if _name is None:
                _source = 'unknown'
            elif 'JHH' in _name:
                _source = 'atlas (JHH)'
            elif 'Merlin' in _name or 'merlin' in _name:
                _source = 'UFO (Merlin)'
            else:
                _source = 'unknown'

            # Crop target. For UFO: `chosen_segment_mask` is populated by the
            # dataset only when the crop was organ-targeted (all-zero on
            # random / background crops). For atlas: chosen_segment_mask is
            # always zero by design, so we fall back to which lesion class
            # has any label voxels in the crop — that's the downstream signal
            # of "this atlas crop contains tumor X".
            _csm_b_nz = [classes[i] for i in range(chosen_segment_mask.shape[1])
                         if chosen_segment_mask[b, i].sum().item() > 0]
            _lesion_in_crop_b = [classes[i] for i in range(label.shape[1])
                                 if ('lesion' in classes[i] or
                                     any(s in classes[i] for s in ('pdac', 'pnet', 'cyst')))
                                 and label[b, i].sum().item() > 0]
            if _source.startswith('atlas'):
                if _lesion_in_crop_b:
                    _crop_desc = (f'atlas → contains {",".join(_lesion_in_crop_b)}')
                else:
                    _crop_desc = 'atlas → no tumor in crop'
            else:
                if _csm_b_nz:
                    _crop_desc = ','.join(_csm_b_nz)
                else:
                    _crop_desc = 'random / background (chosen_segment_mask all-zero)'

            # Atlas sub-type + lesion voxel counts in the crop (NOT organ
            # masks — only the channels the loss cares about).
            _atlas_subtype_counts = {}
            for cls_name in ('pancreatic_lesion',
                             'pancreatic_pdac', 'pancreatic_pnet', 'pancreatic_cyst'):
                if cls_name in classes:
                    ci = classes.index(cls_name)
                    _atlas_subtype_counts[cls_name] = int((label[b, ci] > 0).sum().item())

            # Report-derived malignancy: only non-padding rows from
            # sizes_malignancy. Each row is [diameter_mm, malignancy ∈ {0,1,NaN}].
            _malig_rows = []
            if sizes_malignancy is not None:
                for row in sizes_malignancy[b].detach().cpu().tolist():
                    d = row[0]
                    m = row[1]
                    if d != 0:     # skip padding rows (diameter == 0)
                        if m != m:    # NaN
                            m_str = 'unknown'
                        elif m == 0.0:
                            m_str = 'benign'
                        elif m == 1.0:
                            m_str = 'malignant'
                        else:
                            m_str = f'invalid({m})'
                        _malig_rows.append(f'diam={d} mm → {m_str}')
            _report_malignancy = _malig_rows if _malig_rows else 'no non-padding rows'

            # Report-derived sub-type composition.
            _rep_tt    = tumor_type[b] if (tumor_type is not None and b < len(tumor_type)) else None
            _rep_organ = tumor_type_organ[b] if (tumor_type_organ is not None and b < len(tumor_type_organ)) else None

            # New-loss routing summary. What did each new loss actually do
            # for this sample? (sign-posts; exact numbers live in the detail
            # section below.)
            _ufo_seg_fired = (subtype_seg_loss_enabled and pan_les_col_old is not None)
            _atlas_routed  = bool(active_cell[b].any().item())

            _s4_active_channels = [lc for i, lc in enumerate(lesion_classes)
                                   if bool(active_cell[b, i].item())]
            _s4_mal_voxels = int((mal_label[b] > 0).sum().item())
            _s4_ben_voxels = int((ben_label[b] > 0).sum().item())
            _s4_penalize_voxels = int((penalize[b] > 0).sum().item())

            if _ufo_seg_fired and target is not None:
                _s5b_target_per_channel   = target[b].sum(dim=(-1,-2,-3)).tolist()
                _s5b_penalize_per_channel = penalize_seg[b].sum(dim=(-1,-2,-3)).tolist()
            else:
                _s5b_target_per_channel = _s5b_penalize_per_channel = 'not routed'

            # cls block fires iff tumor_type_organ=='pancreas' AND the sample
            # has a known sub-type AND (for UFO) the crop was on pancreas.
            # Guard against `atlas_has_subtype_cls` / `crop_targeted_pancreas_cls`
            # not being in scope when subtype_cls_loss_enabled=False (they're
            # defined only inside that branch).
            _cls_fired = False
            _cls_targets = _cls_masks = None
            _cls_atlas_gated = None
            _cls_crop_gated  = None
            try:
                _cls_atlas_gated = bool(atlas_has_subtype_cls[b].item())
            except NameError:
                pass
            try:
                _cls_crop_gated = not bool(crop_targeted_pancreas_cls[b].item())
            except NameError:
                pass
            if (subtype_cls_loss_enabled
                and tumor_type_organ is not None
                and b < len(tumor_type_organ)
                and tumor_type_organ[b] == 'pancreas'
                and _cls_atlas_gated is False
                and _cls_crop_gated  is False):
                _tt_set = set(tumor_type[b].split('+')) if tumor_type[b] else set()
                _known = _tt_set & {'pdac', 'pnet', 'cyst'}
                if _known:
                    _cls_fired = True
                    _has_uncertain = bool(_tt_set & {'other', 'unknown'})
                    _cls_targets = [int('pdac' in _known),
                                    int('pnet' in _known),
                                    int('cyst' in _known)]
                    _cls_masks = [
                        int(k in _known or not _has_uncertain)
                        for k in ('pdac', 'pnet', 'cyst')
                    ]

            # Filename key — documents every NIfTI in this dump folder so
            # the reader doesn't have to remember what each one is. Grouped
            # by loss path. Keep in sync with the save_tensor_as_nifti
            # calls above.
            _niftis_key = {
                # ---- inputs + model outputs ----
                'input_volume_B{b}.nii.gz':
                    'Input CT volume for sample b (if `input_tensor` was provided).',
                'chosen_segment_mask_B{b}.nii.gz':
                    'Dataset-provided organ sub-segment mask (summed across '
                    'class channels). Non-empty on UFO samples whose crop '
                    'was targeted; all-zero on random / atlas samples.',
                'malignant_output_B{b}.nii.gz':
                    'Model output for the malignant channels (post-sigmoid), '
                    'summed across channels. The prediction the loss is '
                    'pushing toward the labels.',
                'benign_output_B{b}.nii.gz':
                    'Model output for the benign channels (post-sigmoid), '
                    'summed across channels.',

                # ---- OLD auto_distill_malignancy_loss path ----
                'old_malignant_label_B{b}.nii.gz':
                    'OLD path malignant_label — the spatial target for '
                    'bce_malig / dice_malig inside auto_distill. Built from '
                    'lesion_label × (malignancy_per_voxel == 1 | '
                    'malignant_only from sizes_malignancy).',
                'old_benign_label_B{b}.nii.gz':
                    'OLD path benign_label — symmetric to old_malignant_label.',
                'old_unknown_malignancy_B{b}.nii.gz':
                    'OLD path unknown_malignancy_label — lesion voxels where '
                    'malignant vs benign is ambiguous (including all lesion '
                    "voxels of routed atlas-subtype samples, because section "
                    '(2) NaN\'s out their `mpv` so both mal/ben labels become 0 '
                    'over those voxels). OLD path does NOT penalize these voxels.',
                'old_lesion_label_B{b}.nii.gz':
                    'OLD path lesion_label — the union of malignant + benign + '
                    'unknown labels on the lesion channels (kidney_lesion, '
                    'liver_lesion, pancreatic_lesion).',
                'old_lesion_output_B{b}.nii.gz':
                    'Model output for the lesion channels (post-sigmoid), '
                    'summed. Used by triangle_consistency and as the '
                    '"is there any lesion here?" prediction.',
                'old_bce_malignant_B{b}.nii.gz':
                    'OLD path `bce_malig * penalize_known_malignancy`, summed '
                    'across lesion channels. Non-zero voxels = where the OLD '
                    'malig BCE actually contributes to the gradient.',
                'old_bce_benign_B{b}.nii.gz':
                    'OLD path `bce_benign * penalize_known_malignancy`. '
                    'Symmetric to old_bce_malignant.',
                'old_penalize_malig_AND_benign_B{b}.nii.gz':
                    'OLD auto_distill `penalize_known_malignancy` — gates '
                    'bce_malig AND bce_benign AND dice_malig AND dice_benign '
                    'inside auto_distill_malignancy_loss. Shape (L_old, D, H, W) '
                    'summed across lesion channels.',

                # ---- Section (4) atlas-subtype BCE+Dice ----
                '{pancreatic_pdac|pnet|cyst}_B{b}.nii.gz':
                    'Raw per-voxel sub-type labels from the atlas (used to '
                    'build mal_label = pdac ∪ pnet and ben_label = cyst).',
                'malignant_label_B{b}.nii.gz':
                    'Section (4) mal_label — merged pdac ∪ pnet mask used as '
                    'the target for loss_subtype_malig_bce/dice.',
                'benign_label_B{b}.nii.gz':
                    'Section (4) ben_label — cyst mask used as the target '
                    'for loss_subtype_benign_bce/dice.',
                'bce_malignant_B{b}.nii.gz':
                    'Section (4) `bce_m * penalize` — where the NEW atlas-'
                    'subtype malig BCE actually contributes. Should spatially '
                    'complement old_bce_malignant (no overlap, test BB).',
                'bce_benign_B{b}.nii.gz':
                    'Section (4) `bce_b * penalize`. Symmetric to above.',
                'section4_penalize_malig_AND_benign_B{b}.nii.gz':
                    'Section (4) atlas-subtype penalize — gates '
                    'loss_subtype_malig_bce/dice AND loss_subtype_benign_bce/dice. '
                    'Same mask for both branches (lesion-voxel union of '
                    'mal_label ∪ ben_label), sum across lesion channels.',

                # ---- ball_loss masks (from auto_distill's ball_loss call) ----
                'ball_loss_pseudo_mask_malig_B{b}.nii.gz':
                    'ball_loss pseudo_mask_malignant — the tolerance-aware '
                    'ball centered at argmax of lesion_out for malignant tumors. '
                    'Present only when ball_loss ran for the batch AND this '
                    "sample's `allow_malignancy_loss_ct=True`.",
                'ball_loss_pseudo_mask_benign_B{b}.nii.gz':
                    'ball_loss pseudo_mask_benign — symmetric for benign.',
                'ball_loss_pseudo_mask_lesion_B{b}.nii.gz':
                    'ball_loss pseudo_mask (union of malig + benign + unknown '
                    'malignancy balls). The un-split lesion mask.',
                'ball_loss_penalize_malig_B{b}.nii.gz':
                    'ball_loss malignant branch penalize — gates the per-sample '
                    'BCE+Dice for the malignant ball mask. DIFFERENT tensor '
                    'from the benign branch.',
                'ball_loss_penalize_benign_B{b}.nii.gz':
                    'ball_loss benign branch penalize — DIFFERENT tensor from '
                    'the malignant branch.',

                # ---- Section (5b) UFO sub-type seg — ONE NIfTI PER CHANNEL ----
                # Each of {target, penalize, pred} is now split into 3 niftis
                # — one for each sub-type channel (pdac, pnet, cyst) — so the
                # values stay binary (for target / penalize) or [0,1] (for
                # pred, post-sigmoid). The previous `.sum(0)` saves produced
                # non-binary 0..3 values because pdac, pnet, and cyst
                # penalize masks overlap in different regions.
                'ufo_subtype_target_{pdac|pnet|cyst}_B{b}.nii.gz':
                    'Section (5b) UFO sub-type seg target for ONE sub-type '
                    'channel. Binary 0/1. For a sample where a given sub-type '
                    'is present in the report AND the crop was organ-targeted '
                    "AND ball_loss ran, that channel's target equals the ball "
                    'pseudo-mask (malig or benign as appropriate); otherwise 0.',
                'ufo_subtype_penalize_{pdac|pnet|cyst}_B{b}.nii.gz':
                    'Section (5b) UFO sub-type seg penalize for ONE channel. '
                    'Binary 0/1. For the PRESENT sub-type: equals ball_loss '
                    "'s tolerance-margin penalize. For ABSENT sub-types on a "
                    'cyst-only / pdac-only / pnet-only sample: equals the '
                    'pancreas unk_voxels region (push absent type to 0 over '
                    'the organ). For ambiguous compositions (pdac+pnet, or '
                    "anything with 'other'/'unknown'): zero on that channel.",
                'ufo_subtype_pred_{pdac|pnet|cyst}_B{b}.nii.gz':
                    'Model output for ONE sub-type channel (post-sigmoid, '
                    'finest scale). Probability in [0, 1]. The prediction the '
                    "UFO seg loss is pushing toward that channel's target.",

                # ---- misc / back-compat ----
                'penalize_B{b}.nii.gz':
                    '[alias] same tensor as section4_penalize_malig_AND_benign. '
                    'Kept for back-compat; prefer the explicit name.',
            }

            # =====================================================
            # Collect per-SAMPLE values for each loss path (indexed at b).
            # NO batch-level scalars — the dump is a per-sample file and the
            # batch scalar is a noisy mix of every sample's contribution.
            # =====================================================
            def _ps(t):
                """Extract sample-b float from a (B,) tensor, or None."""
                if t is None: return None
                try: return float(t[b].item())
                except Exception: return None

            # OLD path's per-(B, L) arrays — index into sample b.
            def _old_bl(key):
                t = old_losses_per_bc.get(key)
                if t is not None:
                    try: return t[b].detach().cpu().tolist()
                    except Exception: return None
                return None

            # OLD path's per-sample scalars, computed from per-(B, L) arrays /
            # full spatial masks in `old_losses_per_bc`.
            #   bce_malig/benign: mean over (L, D, H, W) on the full spatial
            #     per-voxel BCE map — matches what `bce_malig.mean()` would
            #     yield if restricted to this sample's slice.
            #   dice_malig/benign: mean across L of the per-(B, L) dice tensor.
            def _old_mask_mean(key):
                m = old_losses_per_bc.get(key)
                if m is None: return None
                try: return float(m[b].mean().item())
                except Exception: return None
            def _old_per_sample_mean_over_L(key):
                t = old_losses_per_bc.get(key)
                if t is None: return None
                try: return float(t[b].mean().item())
                except Exception: return None

            _old_les_order = [c for c in sorted(classes_old) if 'lesion' in c]

            # Per-lesion-channel arrays at the finest scale (section 4).
            _s4_bce_m_pc = bce_m[b].sum(dim=(-1, -2, -3)).tolist() if bce_m is not None else None
            _s4_bce_b_pc = bce_b[b].sum(dim=(-1, -2, -3)).tolist() if bce_b is not None else None
            _s4_dice_m_pc= dice_m[b].tolist() if dice_m is not None else None
            _s4_dice_b_pc= dice_b[b].tolist() if dice_b is not None else None
            # Section 4 per-sample scalars (pre-sample-weighting, pre-batch-mean).
            # These are the local per-sample means already computed at lines
            # 323-326; we just index sample b.
            _s4_bce_m_this  = _ps(bce_m_per_sample)  if 'bce_m_per_sample'  in dir() else None
            _s4_bce_b_this  = _ps(bce_b_per_sample)  if 'bce_b_per_sample'  in dir() else None
            _s4_dice_m_this = _ps(dice_m_per_sample) if 'dice_m_per_sample' in dir() else None
            _s4_dice_b_this = _ps(dice_b_per_sample) if 'dice_b_per_sample' in dir() else None

            # Per-channel arrays for section 5b UFO seg.
            _s5b_bce_pc  = ufo_bce_s_finest[b].sum(dim=(-1, -2, -3)).tolist() if ufo_bce_s_finest is not None else None
            _s5b_dice_pc = ufo_dice_s_finest[b].tolist() if ufo_dice_s_finest is not None else None
            # Section 5b per-sample scalars.
            _s5b_bce_this  = _ps(_s5b_bce_per_sample)
            _s5b_dice_this = _ps(_s5b_dice_per_sample)

            # CLS per-sample scalars (pre-sample-weighting).
            _old_cls_this = _ps(old_losses_per_bc.get('malig_benign_cls_loss_per_sample'))
            _new_cls_s4_this  = _ps(_s5_cls_per_sample)
            _new_cls_s5c_this = _ps(_s5c_cls_per_sample)

            # Triangle-consistency diagnostics. OLD triangle (malig/benign
            # axis) was computed inside auto_distill and exposed via
            # `losses_per_bc_out`; the new sub-type triangle (section 5d)
            # was computed in this wrapper. Both are per-sample so we can
            # tell *for this specific sample* whether the triangle fired
            # and what its loss value was — which is what we care about
            # for cross-checking the rules.
            _old_tri_loss_pb = old_losses_per_bc.get(
                'triangle_malig_benign_loss_per_sample')
            _old_tri_mask_pb = old_losses_per_bc.get(
                'triangle_malig_benign_mask_voxels_per_sample')
            _old_tri_loss_this = _ps(_old_tri_loss_pb)
            _old_tri_mask_this = (int(_old_tri_mask_pb[b].item())
                                  if _old_tri_mask_pb is not None else 0)
            _old_tri_fired = (triangle_consistency and _old_tri_mask_this > 0)
            _s5d_tri_loss_this = _ps(_s5d_loss_per_sample) if _s5d_enabled else None
            _s5d_tri_mask_this = (int(_s5d_mask_voxels_per_sample[b].item())
                                  if _s5d_enabled else 0)
            _s5d_tri_fired = (_s5d_enabled and _s5d_tri_mask_this > 0)

            # Cropper path taken for this sample — set by the dataset's
            # `_normalize_crop_target`. One of: 'pancreas_body' /
            # 'pancreas_head' / 'pancreas_tail' / 'pancreas' / 'random' /
            # 'atlas_per_voxel' / multi-subseg joined by '+'. Surfaces whether
            # the cropper hit the subseg path, the organ-fallback path, or
            # the deterministic random-crop path at dataset.py:3047 / 3065 /
            # 3160 — which the empty `chosen_segment_mask_active_classes`
            # alone couldn't distinguish from an organ crop.
            if crop_target is not None and b < len(crop_target):
                _crop_target_b = crop_target[b]
            else:
                _crop_target_b = None

            # =========================================================
            # Supervision matrix — per (channel, type) 'max' / 'min' / 'null'
            # =========================================================
            # For each of the 6 channels (lesion / benign / malig / cyst /
            # pdac / pnet) and each of 3 supervision types (seg foreground,
            # seg background, classification), report:
            #   'max'   → there's supervision pushing the output toward 1
            #   'min'   → there's supervision pushing the output toward 0
            #   'null'  → no supervision
            # Computed as the UNION of ALL wrapper-visible AND
            # losses_foundation-visible supervision paths:
            #   - standard_seg (calculate_loss BCE+Dice) on every channel
            #     of `label`.
            #   - classification_loss (losses_foundation) on lesion +
            #     pdac/pnet/cyst cls heads.
            #   - OLD auto_distill seg/cls (malig/benign heads) exposed
            #     via `losses_per_bc_out`.
            #   - ball_loss (lesion + malig + benign) exposed via
            #     `ball_masks_out`.
            #   - wrapper section 4 seg + cls (atlas subtype → malig/
            #     benign heads).
            #   - wrapper section 5b seg + 5c cls (UFO subtype heads).
            def _any_true(t):
                return bool((t > 0).any().item()) if t is not None else False

            def _cell_fg(pen, tgt):
                """'max' if any voxel has penalize=1 AND target=1, else 'null'."""
                if pen is None or not _any_true(pen):
                    return 'null'
                return 'max' if _any_true((pen > 0) & (tgt > 0)) else 'null'

            def _cell_bg(pen, tgt):
                """'min' if any voxel has penalize=1 AND target=0, else 'null'."""
                if pen is None or not _any_true(pen):
                    return 'null'
                return 'min' if _any_true((pen > 0) & (tgt == 0)) else 'null'

            def _cell_cls(target_val, mask_val):
                """CLS cell: 'max'=target>0 & mask=1, 'min'=target==0 & mask=1, else 'null'."""
                if mask_val is None or float(mask_val) <= 0:
                    return 'null'
                return 'max' if float(target_val) > 0 else 'min'

            # Locate the pancreas lesion column in OLD's class layout.
            # `pan_les_col_old` is only bound when `subtype_seg_loss
            # _enabled` is True (i.e. ball_loss fired this batch). For
            # atlas-only batches where ball skipped, it stays None —
            # but the matrix's malig/benign/section-4 indexing still
            # needs the column. Fall back to computing it here directly
            # from `classes_old`, which is always available.
            _pcol = pan_les_col_old
            if _pcol is None:
                _les_cols_fallback = [c for c in classes_old
                                      if any(s in c for s in
                                             ('lesion', 'cyst', 'pdac', 'pnet'))]
                if 'pancreatic_lesion' in _les_cols_fallback:
                    _pcol = _les_cols_fallback.index('pancreatic_lesion')

            # Zero-spatial placeholder sized like one spatial channel of label
            # (B, C, D, H, W → pick (D, H, W) from label[b, 0]).
            _zero_sp = torch.zeros_like(label[b, 0]) if label.ndim == 5 else None

            def _get_old(key):
                """Fetch a (B, L, D, H, W) mask from old_losses_per_bc and
                return its (D, H, W) slice at sample b, pancreas-lesion col.
                None if missing or `_pcol` is None."""
                if _pcol is None: return None
                t = old_losses_per_bc.get(key)
                if t is None: return None
                try: return t[b, _pcol]
                except Exception: return None

            def _get_ball(key):
                if _pcol is None: return None
                t = ball_masks_out.get(key)
                if t is None: return None
                try: return t[b, _pcol]
                except Exception: return None

            def _union(*masks):
                """Sum non-None masks → combined penalize/target (>0 indicates active)."""
                accum = None
                for m in masks:
                    if m is None: continue
                    if accum is None:
                        accum = (m > 0).to(m.dtype)
                    else:
                        accum = ((accum > 0) | (m > 0)).to(m.dtype)
                return accum

            # Resolve the pancreatic_lesion channel in the FULL class
            # layout (used below by standard_seg / classification_loss
            # contributions to the lesion + sub-type heads).
            _pan_les_idx_full = (classes.index('pancreatic_lesion')
                                 if 'pancreatic_lesion' in classes else None)

            def _std_seg_pen_tgt(ci):
                """standard_seg contribution at class index `ci`:
                penalize = 1-unk_voxels[ci], target = label[ci]."""
                if ci is None or unk_voxels is None:
                    return None, None
                return ((1 - unk_voxels[b, ci]).float(),
                        label[b, ci].float())

            # --- Per-channel (penalize, target) union ---
            # lesion: standard_seg (label + 1-unk) ∪ ball_loss_lesion.
            # OLD's `lesion_label_mask` and `penalize_known_malignancy_mask`
            # are NOT targets on the segmentation head's pancreatic_lesion
            # channel — they supervise the `malig_benign` output (via
            # `malignant_label = lesion_label × mpv` and `benign_label =
            # lesion_label × benign_flag` at losses_foundation.py:2591,
            # 2613). Gradient-pattern audit (sample 9/B0) confirmed:
            # every FG-push voxel on seg[pancreatic_lesion] sits inside
            # the ball; none inside the distillation blob. Including
            # `lesion_label_mask` here would be a saver-side artifact.
            # Ball's lesion-branch penalize is approximated by
            # `pseudo_mask_lesion` (same proxy the double-penalization
            # audit uses a few blocks below).
            _les_std_pen, _les_std_tgt = _std_seg_pen_tgt(_pan_les_idx_full)
            _les_tgt = _union(_get_ball('pseudo_mask_lesion'),
                              _les_std_tgt)
            _les_pen = _union(_get_ball('pseudo_mask_lesion'),
                              _les_std_pen)

            # malig: OLD mal_label + ball malig + section 4 mal_label
            _mal_tgt_s4 = (mal_label[b, _pcol]
                           if (_pcol is not None and mal_label is not None and
                               mal_label.shape[1] > _pcol) else None)
            _mal_pen_s4 = (penalize[b, _pcol]
                           if (_pcol is not None and penalize is not None and
                               penalize.shape[1] > _pcol) else None)
            _mal_tgt = _union(_get_old('malignant_label_mask'),
                              _get_ball('pseudo_mask_malignant'),
                              _mal_tgt_s4)
            _mal_pen = _union(_get_old('penalize_known_malignancy_mask'),
                              _get_ball('penalize_malignant'),
                              _mal_pen_s4)

            # benign: symmetric
            _ben_tgt_s4 = (ben_label[b, _pcol]
                           if (_pcol is not None and ben_label is not None and
                               ben_label.shape[1] > _pcol) else None)
            _ben_pen_s4 = _mal_pen_s4   # same penalize mask for both malig/benign in section 4
            _ben_tgt = _union(_get_old('benign_label_mask'),
                              _get_ball('pseudo_mask_benign'),
                              _ben_tgt_s4)
            _ben_pen = _union(_get_old('penalize_known_malignancy_mask'),
                              _get_ball('penalize_benign'),
                              _ben_pen_s4)

            # pdac/pnet/cyst: atlas label (pushed via standard seg) + section 5b
            def _subtype_masks(cls_name, s5b_idx):
                if cls_name not in classes:
                    return None, None
                ci = classes.index(cls_name)
                # Atlas target = label[b, ci] (per-voxel subtype mask).
                atlas_tgt = label[b, ci].float()
                # Standard seg penalize on this subtype = everywhere outside
                # unk_voxels[cls_name] — this is what `calculate_loss` uses.
                std_pen = (1 - unk_voxels[b, ci]).float() if unk_voxels is not None else None
                # Section 5b contribution (if fired):
                s5b_tgt = (target_full[b, s5b_idx].float()
                           if (target_full is not None and
                               target_full.shape[1] > s5b_idx) else None)
                s5b_pen = (penalize_seg[b, s5b_idx].float()
                           if (penalize_seg is not None and
                               penalize_seg.shape[1] > s5b_idx) else None)
                return (_union(atlas_tgt, s5b_tgt),
                        _union(std_pen, s5b_pen))

            _pdac_tgt, _pdac_pen = _subtype_masks('pancreatic_pdac', 0)
            _pnet_tgt, _pnet_pen = _subtype_masks('pancreatic_pnet', 1)
            _cyst_tgt, _cyst_pen = _subtype_masks('pancreatic_cyst', 2)

            # --- cls targets per channel ---
            # Every cls cell unions all loss paths that touch that cls
            # head. A cell is `max` iff any path pushes it to 1; `min`
            # iff some path pushes to 0 and none to 1; `null` iff no
            # path supervises it.
            def _has_mask(key):
                m = _get_old(key)
                return _any_true(m) if m is not None else False

            def _union_cls(sources):
                """sources: iterable of (target, mask) pairs in
                Python scalars. Returns (cell_value_label, mask_flag)."""
                has_max = any(mask > 0 and tgt > 0 for tgt, mask in sources)
                has_min = any(mask > 0 and tgt <= 0 for tgt, mask in sources)
                if has_max: return 1, 1
                if has_min: return 0, 1
                return 0, 0

            # ---- lesion cls: classification_loss (losses_foundation) ----
            # Target per losses_foundation.classification_loss:
            #   lesion_labels = (label + chosen_segment_mask).sum > 0
            # Mask: 0 iff unk.sum>0 AND lesion_labels==0, else 1.
            def _std_cls_tgt_mask(ci):
                if ci is None:
                    return 0, 0
                lsum = float(label[b, ci].sum().item())
                csm = float(chosen_segment_mask[b, ci].sum().item()
                            ) if chosen_segment_mask is not None else 0.0
                tgt = 1 if (lsum + csm) > 0 else 0
                unk = float(unk_voxels[b, ci].sum().item()
                            ) if unk_voxels is not None else 0.0
                # known_labels from classification_loss: 1 iff lesion
                # label is present OR unk is zero on that channel.
                mask = 0 if (unk > 0 and tgt == 0) else 1
                return tgt, mask

            _les_std_cls = _std_cls_tgt_mask(_pan_les_idx_full)
            _les_cls_tgt, _les_cls_mask = _union_cls([_les_std_cls])

            # ---- malig / benign cls: OLD auto_distill + wrapper sec 5 ----
            # OLD path exposes (B, L) target+mask via losses_per_bc_out;
            # index sample b, pancreatic_lesion column.
            def _old_cls_tm(tgt_key, mask_key):
                t = old_losses_per_bc.get(tgt_key)
                m = old_losses_per_bc.get(mask_key)
                if t is None or m is None or _pcol is None:
                    return 0, 0
                try:
                    return float(t[b, _pcol].item()), float(m[b, _pcol].item())
                except Exception:
                    return 0, 0

            _malig_old_cls = _old_cls_tm('malig_cls_target', 'malig_cls_mask')
            _benign_old_cls = _old_cls_tm('benign_cls_target', 'benign_cls_mask')

            # Wrapper section 5 (atlas subtype) cls: target =
            # (mal_label/ben_label).sum>0, mask = active_cell @ pcol.
            _s5_mask_flag = (1 if (_pcol is not None and active_cell is not None
                                   and bool(active_cell[b, _pcol].item()))
                             else 0)
            _s5_mal_tgt = (1 if (mal_label is not None and _pcol is not None
                                 and bool((mal_label[b, _pcol].sum() > 0).item()))
                           else 0)
            _s5_ben_tgt = (1 if (ben_label is not None and _pcol is not None
                                 and bool((ben_label[b, _pcol].sum() > 0).item()))
                           else 0)

            _malig_cls_tgt, _malig_cls_mask = _union_cls(
                [_malig_old_cls, (_s5_mal_tgt, _s5_mask_flag)])
            _benign_cls_tgt, _benign_cls_mask = _union_cls(
                [_benign_old_cls, (_s5_ben_tgt, _s5_mask_flag)])

            # ---- sub-type cls: classification_loss + section 5c ----
            def _s5c_cls(ch_idx):
                if _s5c_target_cls_full is None: return 0, 0
                return (float(_s5c_target_cls_full[b, ch_idx].item()),
                        float(_s5c_mask_cls_full[b, ch_idx].item()))

            def _subtype_cls_union(cls_name, s5c_idx):
                ci = classes.index(cls_name) if cls_name in classes else None
                std = _std_cls_tgt_mask(ci)
                s5c = _s5c_cls(s5c_idx)
                return _union_cls([std, s5c])

            _pdac_cls_tgt, _pdac_cls_mask = _subtype_cls_union('pancreatic_pdac', 0)
            _pnet_cls_tgt, _pnet_cls_mask = _subtype_cls_union('pancreatic_pnet', 1)
            _cyst_cls_tgt, _cyst_cls_mask = _subtype_cls_union('pancreatic_cyst', 2)

            _supervision_matrix = {
                'lesion': {
                    'segmentation_foreground': _cell_fg(_les_pen, _les_tgt),
                    'segmentation_background': _cell_bg(_les_pen, _les_tgt),
                    'classification':          _cell_cls(_les_cls_tgt, _les_cls_mask),
                },
                'benign': {
                    'segmentation_foreground': _cell_fg(_ben_pen, _ben_tgt),
                    'segmentation_background': _cell_bg(_ben_pen, _ben_tgt),
                    'classification':          _cell_cls(_benign_cls_tgt, _benign_cls_mask),
                },
                'malig': {
                    'segmentation_foreground': _cell_fg(_mal_pen, _mal_tgt),
                    'segmentation_background': _cell_bg(_mal_pen, _mal_tgt),
                    'classification':          _cell_cls(_malig_cls_tgt, _malig_cls_mask),
                },
                'cyst': {
                    'segmentation_foreground': _cell_fg(_cyst_pen, _cyst_tgt),
                    'segmentation_background': _cell_bg(_cyst_pen, _cyst_tgt),
                    'classification':          _cell_cls(_cyst_cls_tgt, _cyst_cls_mask),
                },
                'pdac': {
                    'segmentation_foreground': _cell_fg(_pdac_pen, _pdac_tgt),
                    'segmentation_background': _cell_bg(_pdac_pen, _pdac_tgt),
                    'classification':          _cell_cls(_pdac_cls_tgt, _pdac_cls_mask),
                },
                'pnet': {
                    'segmentation_foreground': _cell_fg(_pnet_pen, _pnet_tgt),
                    'segmentation_background': _cell_bg(_pnet_pen, _pnet_tgt),
                    'classification':          _cell_cls(_pnet_cls_tgt, _pnet_cls_mask),
                },
            }

            # =========================================================
            # Double-penalization detection — pairwise overlap between
            # penalize masks of different loss paths on the SAME model
            # output channel. Emits one entry per (channel, loss-pair)
            # that actually overlaps. When targets at the overlapping
            # voxels agree, the loss is applied twice in the same
            # direction (harmless 2× gradient magnitude); when they
            # disagree it's a CONFLICT that fights against itself.
            #
            # Loss paths considered (all wrapper-visible):
            #   * standard_seg      — calculate_loss's BCE+Dice with
            #                         `known_voxels = 1 - unk_voxels`
            #                         and target = `label`.
            #   * ball_loss_lesion  — ball_loss lesion branch: target =
            #                         `pseudo_mask_lesion`, penalize =
            #                         `pseudo_mask_lesion | (1 - unk)`
            #                         approximated via OLD's
            #                         penalize_known_malignancy_mask
            #                         (which is pre-ball-subtraction on
            #                         the lesion column when ball ran).
            #   * auto_distill_old  — auto_distill's malig/benign BCE+Dice,
            #                         penalize = penalize_known_malignancy
            #                         (post-ball-subtraction).
            #   * ball_loss_malig   — ball_loss malig branch (if ball ran).
            #   * ball_loss_benign  — symmetric for benign.
            #   * section_4         — wrapper section 4 atlas-subtype
            #                         malig/benign supervision.
            #   * section_5b        — wrapper section 5b UFO subtype seg.
            # =========================================================
            def _overlap_counts(pen_i, tgt_i, pen_j, tgt_j):
                """Return (fg_count, bg_count, conflict_count) for the
                spatial overlap between two (penalize, target) pairs."""
                if pen_i is None or pen_j is None:
                    return 0, 0, 0
                ol = (pen_i > 0) & (pen_j > 0)
                if not bool(ol.any().item()):
                    return 0, 0, 0
                ti = (tgt_i > 0) if tgt_i is not None else torch.zeros_like(ol)
                tj = (tgt_j > 0) if tgt_j is not None else torch.zeros_like(ol)
                fg  = int((ol & ti & tj).sum().item())
                bg  = int((ol & ~ti & ~tj).sum().item())
                conf = int((ol & (ti ^ tj)).sum().item())
                return fg, bg, conf

            # Build per-channel loss-path inventory at sample b.
            # Each entry: (loss_name, penalize_mask, target_mask).
            # None entries are dropped below.
            pan_les_idx_full = classes.index('pancreatic_lesion') if 'pancreatic_lesion' in classes else None
            # Standard seg's contribution at each channel — the wrapper sees
            # unk_voxels + label, which (modulo the dilation applied inside
            # get_known_voxels) gives a faithful penalize × target pair.
            def _std_seg_penalize(ci):
                if unk_voxels is None or ci is None: return None
                return (1 - unk_voxels[b, ci]).float()
            def _std_seg_target(ci):
                if ci is None: return None
                return label[b, ci].float()

            channel_paths = {}  # channel label → list of (name, pen, tgt)

            # ---- pancreatic_lesion (lesion output head) ----
            paths_les = []
            std_pen = _std_seg_penalize(pan_les_idx_full)
            std_tgt = _std_seg_target(pan_les_idx_full)
            if std_pen is not None:
                paths_les.append(('standard_seg', std_pen, std_tgt))
            if 'pseudo_mask_lesion' in ball_masks_out and _pcol is not None:
                # ball_loss lesion: use pseudo_mask_lesion as penalize proxy
                # (actual ball penalize at lesion follows `to_penalize` which
                # the wrapper can't reconstruct exactly; pseudo_mask is a
                # reliable lower bound for the FG overlap check).
                paths_les.append(('ball_loss_lesion',
                                  _get_ball('pseudo_mask_lesion'),
                                  _get_ball('pseudo_mask_lesion')))
            channel_paths['pancreatic_lesion'] = paths_les

            # ---- malig head at pancreatic_lesion ----
            paths_mal = []
            if _get_old('penalize_known_malignancy_mask') is not None:
                paths_mal.append(('auto_distill_old_malig',
                                  _get_old('penalize_known_malignancy_mask'),
                                  _get_old('malignant_label_mask')))
            if _get_ball('pseudo_mask_malignant') is not None:
                paths_mal.append(('ball_loss_malig',
                                  _get_ball('penalize_malignant'),
                                  _get_ball('pseudo_mask_malignant')))
            if (_pcol is not None and penalize is not None
                    and mal_label is not None and penalize.shape[1] > _pcol):
                paths_mal.append(('section_4',
                                  penalize[b, _pcol],
                                  mal_label[b, _pcol]))
            channel_paths['malig_head_at_pancreas'] = paths_mal

            # ---- benign head at pancreatic_lesion ----
            paths_ben = []
            if _get_old('penalize_known_malignancy_mask') is not None:
                paths_ben.append(('auto_distill_old_benign',
                                  _get_old('penalize_known_malignancy_mask'),
                                  _get_old('benign_label_mask')))
            if _get_ball('pseudo_mask_benign') is not None:
                paths_ben.append(('ball_loss_benign',
                                  _get_ball('penalize_benign'),
                                  _get_ball('pseudo_mask_benign')))
            if (_pcol is not None and penalize is not None
                    and ben_label is not None and penalize.shape[1] > _pcol):
                paths_ben.append(('section_4',
                                  penalize[b, _pcol],
                                  ben_label[b, _pcol]))
            channel_paths['benign_head_at_pancreas'] = paths_ben

            # ---- pdac / pnet / cyst (sub-type output channels) ----
            for cls_name, s5b_idx in (('pancreatic_pdac', 0),
                                       ('pancreatic_pnet', 1),
                                       ('pancreatic_cyst', 2)):
                if cls_name not in classes:
                    continue
                ci = classes.index(cls_name)
                paths_sub = []
                sp = _std_seg_penalize(ci)
                if sp is not None:
                    paths_sub.append(('standard_seg', sp, _std_seg_target(ci)))
                if (target_full is not None and target_full.shape[1] > s5b_idx):
                    paths_sub.append(('section_5b',
                                      penalize_seg[b, s5b_idx].float(),
                                      target_full[b, s5b_idx].float()))
                channel_paths[cls_name] = paths_sub

            # Pairwise overlap scan.
            _double_pen = {}
            for ch, paths in channel_paths.items():
                findings = []
                for i in range(len(paths)):
                    for j in range(i + 1, len(paths)):
                        name_i, pen_i, tgt_i = paths[i]
                        name_j, pen_j, tgt_j = paths[j]
                        fg, bg, conf = _overlap_counts(pen_i, tgt_i, pen_j, tgt_j)
                        if fg == 0 and bg == 0 and conf == 0:
                            continue
                        findings.append({
                            'losses': [name_i, name_j],
                            'overlap_voxels_foreground': fg,
                            'overlap_voxels_background': bg,
                            'overlap_voxels_conflict':   conf,
                        })
                if findings:
                    _double_pen[ch] = findings

            meta['summary'] = {
                # =========================================================
                # 1) SAMPLE INFO
                #    Who this sample is and what its crop / report look like.
                # =========================================================
                'sample': {
                    'name': _name,
                    'source_atlas_or_UFO':
                        _source,   # 'atlas (JHH)' | 'UFO (Merlin)' | 'unknown'
                    'atlas_masks_in_crop_voxels':
                        _atlas_subtype_counts,
                        # ^ Only pancreatic_lesion / _pdac / _pnet / _cyst
                        #   channels — NOT organ masks like 'pancreas'. Tells
                        #   you which per-voxel sub-type labels are actually
                        #   in this crop. Non-zero ONLY for atlas samples;
                        #   UFO samples have these channels zero by design.
                    'chosen_segment_mask_active_classes':
                        _csm_b_nz,
                        # ^ The classes with non-zero chosen_segment_mask. For
                        #   UFO: non-empty iff the crop was organ-targeted
                        #   (e.g. ['pancreatic_lesion']). For atlas: always []
                        #   because the dataset deliberately zeroes it (see
                        #   dataset_abdomenatlas_UFO_multi_tumor.py:2057).
                    'crop_target':
                        _crop_target_b,
                        # ^ 'pancreas_body' / 'pancreas_head' / 'pancreas_tail'
                        #   → subsegment crop succeeded. 'pancreas' → organ
                        #   fallback (report location was 'u'). 'random' →
                        #   deterministic random crop (no tumor-to-crop-on or
                        #   rand() < non_tumor_crop_chance). 'atlas_per_voxel'
                        #   → atlas sample, cropper bypassed entirely.
                    'report_tumor_type':
                        _rep_tt,
                        # ^ '+'-joined composition from the per-tumor report
                        #   (e.g. 'cyst', 'cyst+pdac', 'other', 'unknown',
                        #   'pdac+unknown'). From dataset.get_pancreas_lesion_type.
                    'report_tumor_type_organ':
                        _rep_organ,
                        # ^ 'pancreas' when the report has any pancreas row,
                        #   'none' otherwise.
                    'report_malignancy_rows':
                        _report_malignancy,
                        # ^ Non-padding rows of sizes_malignancy formatted as
                        #   "diam=X mm → benign/malignant/unknown". Each row
                        #   is one tumor reported in the crop's organ.
                },

                # =========================================================
                # 1b) SUPERVISION MATRIX — 6 channels × 3 types
                #     Each cell ∈ {'max', 'min', 'null'}:
                #       max  → some voxel pushed output toward 1
                #       min  → some voxel pushed output toward 0
                #       null → no supervision on that (channel, type)
                #     Designed so the eval script can produce a flat CSV
                #     per-sample without needing to re-load the NIfTIs.
                # =========================================================
                'supervision_matrix': _supervision_matrix,

                # =========================================================
                # 1c) DOUBLE-PENALIZATION AUDIT — voxel-level spatial overlap
                #     between penalize masks of different loss paths on the
                #     SAME output channel. Per (channel, loss-pair) finding:
                #       - losses: [loss_name_i, loss_name_j]
                #       - overlap_voxels_foreground: both push → 1 (SAFE 2×)
                #       - overlap_voxels_background: both push → 0 (SAFE 2×)
                #       - overlap_voxels_conflict:   push in opposite directions (BUG)
                #     Empty dict when nothing overlaps. Non-empty entries are
                #     expected in a few places by design (e.g. standard_seg +
                #     section_5b on pdac/pnet/cyst push-to-0 outside pancreas)
                #     — check the `conflict` count to detect real bugs.
                # =========================================================
                'double_penalization': _double_pen,

                # =========================================================
                # 2) LOSSES — one block per loss path, in order:
                #    OLD first, then NEW. Each block leads with `loss` (the
                #    per-sample scalar values), followed by kind / origin /
                #    supervises / per-channel breakdowns.
                #    Note (applies to every `*_dice*` field below): dice = 1.0
                #    usually means the channel was NOT evaluated (penalize mask
                #    is zero for that channel on this sample) — it's the
                #    DiceLossMultiClass fall-through `1 - 0/smooth`, not a
                #    real "worst-case" score. Check the matching
                #    `penalize_voxels` or `*_voxels` count; if zero, the
                #    1.0 is just logging noise with no gradient.
                # =========================================================
                'losses': {
                    '_note_dice_eq_1': 'dice=1.0 almost always means the loss did not run on that channel (penalize was 0 → DiceLossMultiClass fall-through `1 - 0/smooth`). Check penalize_voxels / mal_label_voxels / ben_label_voxels before interpreting dice values.',
                    # --- OLD --------------------------------------------
                    'old_seg__malig_benign': {
                        'loss': {
                            'bce_malig_voxel_mean':  _old_mask_mean('bce_malig_mask'),
                            'bce_benign_voxel_mean': _old_mask_mean('bce_benign_mask'),
                            'dice_malig_L_mean':     _old_per_sample_mean_over_L('loss_malig_dice_per_bc'),
                            'dice_benign_L_mean':    _old_per_sample_mean_over_L('loss_benign_dice_per_bc'),
                        },
                        'kind': 'seg',
                        'origin': 'old (auto_distill_malignancy_loss, standard BCE+Dice path)',
                        'supervises': (
                            "The MALIGNANT and BENIGN segmentation channels "
                            "(one per *_lesion class: kidney_lesion, "
                            "liver_lesion, pancreatic_lesion). Targets are "
                            "derived from `lesion_label × malignancy_per_voxel` "
                            "plus report-derived malignant_only/benign_only "
                            "flags. Section (2) NaN's out mpv[pancreatic_lesion] "
                            "for atlas-subtype samples so lesion voxels are "
                            "handed off to section (4); the OLD path still "
                            "penalizes BACKGROUND voxels there."
                        ),
                        'per_lesion_channel_finest_scale': {
                            'channel_order':        _old_les_order,
                            'bce_malig_spatial_sum': _old_bl('loss_malig_bce_per_bc'),
                            'bce_benign_spatial_sum':_old_bl('loss_benign_bce_per_bc'),
                            'dice_malig':            _old_bl('loss_malig_dice_per_bc'),
                            'dice_benign':           _old_bl('loss_benign_dice_per_bc'),
                            'penalize_voxels':       _old_bl('penalize_known_malignancy_per_bc'),
                            'malignant_label_voxels':_old_bl('malignant_label_per_bc'),
                            'benign_label_voxels':   _old_bl('benign_label_per_bc'),
                            'unknown_malignancy_label_voxels':
                                                     _old_bl('unknown_malignancy_label_per_bc'),
                        },
                    },
                    'old_seg__ball_loss__malig_benign': {
                        # ball_loss doesn't expose a per-sample scalar; use the
                        # per-lesion-channel ball_applied voxel count as the
                        # per-sample signal ("did ball_loss fire on this
                        # sample's pancreas/liver/kidney lesion channels?").
                        'loss': {
                            'ball_applied_voxels_per_lesion_channel':
                                _old_bl('ball_applied_per_bc'),
                        },
                        'kind': 'seg',
                        'origin': 'old (ball_loss, inside auto_distill_malignancy_loss)',
                        'supervises': (
                            "Same MALIGNANT and BENIGN segmentation channels, "
                            "but using tolerance-aware balls centered at the "
                            "model's argmax for each report-annotated tumor "
                            "(with a known size and a known malig/benign "
                            "flag). Fires iff will_skip_ball_malignancy_loss_for_all_batch_items "
                            "is False for the batch AND this sample's "
                            "allow_malignancy_loss_ct=True (e.g., no "
                            "unknown-size sentinel)."
                        ),
                    },
                    'old_cls__malig_benign': {
                        'loss': {
                            'malig_benign_cls_loss': _old_cls_this,
                        },
                        'kind': 'cls',
                        'origin': 'old (auto_distill_malignancy_loss, classification half)',
                        'supervises': (
                            "The MALIGNANT and BENIGN classification heads "
                            "(one per lesion class). Target = presence flag "
                            "per channel derived from spatial mal/ben labels "
                            "+ report-derived has_mal/has_ben flags. "
                            "Masked when malignancy is unknown."
                        ),
                    },

                    # --- NEW section (4) --------------------------------
                    'new_seg__atlas_subtype__section_4': {
                        'loss': {
                            'subtype_malig_bce':  _s4_bce_m_this,
                            'subtype_benign_bce': _s4_bce_b_this,
                            'subtype_malig_dice': _s4_dice_m_this,
                            'subtype_benign_dice':_s4_dice_b_this,
                        },
                        'kind': 'seg',
                        'origin': 'new (wrapper section 4)',
                        'supervises': (
                            "The MALIGNANT and BENIGN segmentation channels, "
                            "but FOR ATLAS SAMPLES ONLY (those with per-voxel "
                            "pancreatic_pdac / _pnet / _cyst labels). "
                            "Target: mal_label = pdac ∪ pnet, ben_label = cyst. "
                            "Penalize is restricted to lesion voxels — OLD "
                            "path retains unique responsibility for background. "
                            "Triggered iff any per-voxel sub-type label "
                            "appears in the crop (active_cell[b, "
                            "pancreatic_lesion]=True)."
                        ),
                        'routed_for_this_sample': _atlas_routed,
                        'active_lesion_channels': _s4_active_channels,
                        'mal_label_voxels':       _s4_mal_voxels,
                        'ben_label_voxels':       _s4_ben_voxels,
                        'penalize_voxels':        _s4_penalize_voxels,
                        'per_lesion_channel_finest_scale': {
                            'channel_order':         _old_les_order,
                            'bce_malig_spatial_sum': _s4_bce_m_pc,
                            'bce_benign_spatial_sum':_s4_bce_b_pc,
                            'dice_malig':            _s4_dice_m_pc,
                            'dice_benign':           _s4_dice_b_pc,
                        },
                    },
                    'new_cls__atlas_subtype__section_4': {
                        'loss': {
                            'subtype_malig_benign_cls_loss': _new_cls_s4_this,
                        },
                        'kind': 'cls',
                        'origin': 'new (wrapper section 4, classification half)',
                        'supervises': (
                            "The MALIGNANT and BENIGN classification heads "
                            "for atlas-subtype samples whose mpv was NaN'd "
                            "out by section (2). Without this, OLD cls would "
                            "also skip those cells (mpv=NaN → unknown branch), "
                            "leaving the cls head unsupervised on atlas-subtype "
                            "samples. Target = presence flag per lesion class "
                            "from mal_label/ben_label sums."
                        ),
                    },

                    # --- NEW section (5b) — UFO sub-type seg -----------
                    'new_seg__ufo_subtype__section_5b': {
                        'loss': {
                            'ufo_subtype_seg_bce':  _s5b_bce_this,
                            'ufo_subtype_seg_dice': _s5b_dice_this,
                        },
                        'kind': 'seg',
                        'origin': 'new (wrapper section 5b)',
                        'supervises': (
                            "The SUB-TYPE segmentation channels "
                            "(pancreatic_pdac, pancreatic_pnet, pancreatic_cyst). "
                            "Targets come from ball_loss pseudo-masks, routed "
                            "per report: malignant ball → pdac (if tumor_type "
                            "contains 'pdac' alone) or pnet (if 'pnet' alone). "
                            "benign ball → cyst (if 'cyst' reported). Absent "
                            "sub-types are pushed to 0 over the pancreas "
                            "sub-segment region from unk_voxels. Block fires "
                            "iff ball_loss ran AND sub-type classes are "
                            "configured."
                        ),
                        'block_fired': _ufo_seg_fired,
                        'target_voxels_per_channel_pdac_pnet_cyst':  _s5b_target_per_channel,
                        'penalize_voxels_per_channel_pdac_pnet_cyst':_s5b_penalize_per_channel,
                        'per_channel_finest_scale': {
                            'channel_order': ['pancreatic_pdac',
                                              'pancreatic_pnet',
                                              'pancreatic_cyst'],
                            'bce_spatial_sum': _s5b_bce_pc,
                            'dice':            _s5b_dice_pc,
                        },
                    },

                    # --- NEW section (5c) — UFO sub-type cls -----------
                    'new_cls__ufo_subtype__section_5c': {
                        'loss': {
                            'ufo_subtype_cls_loss': _new_cls_s5c_this,
                        },
                        'kind': 'cls',
                        'origin': 'new (wrapper section 5c)',
                        'supervises': (
                            "The SUB-TYPE classification channels "
                            "(pancreatic_pdac, pancreatic_pnet, pancreatic_cyst). "
                            "Target: 1 if the sub-type is present in the "
                            "report, 0 if confidently absent (pure known "
                            "composition without 'other'/'unknown'). Mask=0 "
                            "for ambiguous absent channels (un-observed when "
                            "the composition contains 'other'/'unknown'). "
                            "Gated on chosen_segment_mask — skipped on "
                            "failed organ crops (tumor location unknown vs. "
                            "crop)."
                        ),
                        'fired_for_this_sample':     _cls_fired,
                        'target_cls_pdac_pnet_cyst': _cls_targets,
                        'mask_cls_pdac_pnet_cyst':   _cls_masks,
                    },

                    # --- OLD triangle — malig/benign axis -----------------
                    'old__triangle__malig_benign': {
                        'loss': {
                            'triangle_malig_benign_loss': _old_tri_loss_this,
                        },
                        'kind': 'consistency',
                        'origin': 'old (auto_distill_malignancy_loss, triangle path)',
                        'supervises': (
                            "At voxels where we know a lesion MAY be present "
                            "(`lesion_label > 0`) but neither malignant_label "
                            "nor benign_label fires (the `unknown_malignancy_"
                            "label` mask), pushes "
                            "`σ(malig_out) + σ(benign_out) ≈ σ(lesion_out)`. "
                            "Forces the model to commit its lesion-presence "
                            "mass to one side (or split probabilistically) "
                            "instead of leaving both malig/benign at 0 while "
                            "lesion_out is non-zero. Complementary to BCE/"
                            "Dice on malig_benign output: fires EXACTLY where "
                            "`penalize_known_malignancy = 0` masks BCE off. "
                            "Enabled globally by `--triangle_consistency`; "
                            "per-sample, fires iff any (mal/ben)-unknown "
                            "voxels exist in the crop."
                        ),
                        'fired_for_this_sample': _old_tri_fired,
                        'triangle_consistency_flag': bool(triangle_consistency),
                        'unknown_malignancy_mask_voxels': _old_tri_mask_this,
                    },

                    # --- NEW section (5d) — sub-type triangle -------------
                    'new__triangle__subtype': {
                        'loss': {
                            'subtype_triangle_loss': _s5d_tri_loss_this,
                        },
                        'kind': 'consistency',
                        'origin': 'new (wrapper section 5d)',
                        'supervises': (
                            "Sub-type analogue of OLD's triangle. At voxels "
                            "where `lesion_label > 0` but no per-voxel sub-"
                            "type label fires (atlas label = 0 AND section-"
                            "5b `target_full` = 0 on all three channels), "
                            "pushes `σ(pdac) + σ(pnet) + σ(cyst) ≈ "
                            "σ(lesion)`. Forces the model to commit its "
                            "lesion-presence mass to SOME sub-type when the "
                            "location is unknown (UFO unk region outside "
                            "the ball, or ball_skipped batches). Enabled "
                            "globally by `--triangle_consistency`; per-"
                            "sample, fires iff any (no-subtype-label)-but-"
                            "(lesion-possible) voxels exist."
                        ),
                        'fired_for_this_sample': _s5d_tri_fired,
                        'triangle_consistency_flag': bool(triangle_consistency),
                        'unknown_subtype_mask_voxels': _s5d_tri_mask_this,
                    },
                },

                # =========================================================
                # 3) NIfTI FILENAME KEY
                #    Documents every *.nii.gz in this dump folder.
                # =========================================================
                'niftis_key': _niftis_key,
            }

            # --- detail section ---------------------------------------------------
            # Everything below the `summary` block is the raw / granular
            # detail. Top-line questions should be answerable from `summary`
            # alone; drop down here when you need per-(B,L) arrays, old-loss
            # scalars, class weights, etc.
            meta['name'] = _name
            meta['lesion_classes'] = list(lesion_classes)
            meta['classes_subtype'] = [classes[i] for i in sorted(strip_idx_set)]
            meta['sample_routed_new_loss'] = bool(active_cell[b].any().item())
            meta['active_cell_per_lesion'] = {
                lc: bool(active_cell[b, i].item())
                for i, lc in enumerate(lesion_classes)
            }
            # Per-sub-type voxel counts
            meta['mask_voxel_counts'] = {}
            for p in plan:
                for ci in p['mal_idx'] + p['ben_idx']:
                    meta['mask_voxel_counts'][classes[ci]] = int(
                        (label[b, ci] > 0).sum().item())
            meta['mal_label_voxels'] = int((mal_label[b] > 0).sum().item())
            meta['ben_label_voxels'] = int((ben_label[b] > 0).sum().item())
            meta['penalize_voxels'] = int((penalize[b] > 0).sum().item())

            meta['malignant_label_max'] = float(mal_label[b].max().item())
            meta['benign_label_max']    = float(ben_label[b].max().item())

            # Model-output stats (post-sigmoid for readability)
            if not sigmoid_already_applied:
                mo_s = torch.sigmoid(malign_out[b])
                bo_s = torch.sigmoid(benign_out[b])
            else:
                mo_s, bo_s = malign_out[b], benign_out[b]
            meta['malignant_output_max'] = float(mo_s.max().item())
            meta['benign_output_max']    = float(bo_s.max().item())

            # NOTE: batch-level scalars intentionally omitted. Every per-loss
            # entry under `summary.losses.*.loss` already
            # captures what this sample contributed pre-sample-weighting and
            # pre-batch-mean. The old batch scalars (e.g. loss_subtype_*,
            # old_loss_scalars) were identical across every B-th dump in a
            # given batch, not per-sample, and hid the inflation caused by
            # `_apply_sw` with unnormalized sample_weights (see the dice>1
            # investigation).
            meta['subtype_loss_weight'] = float(subtype_loss_weight)
            meta['sigmoid_already_applied'] = bool(sigmoid_already_applied)

            # ---- Per-(sample, lesion-channel) loss breakdowns (NEW losses) ----
            # For each new-loss key that the wrapper computes per-voxel, we
            # sum over spatial dims to get (B, C) and slice out row `b` → a
            # list of C scalars, one per lesion class. Gives per-channel
            # insight (e.g. is loss concentrated in pancreatic_lesion vs
            # kidney_lesion?).
            #
            # Section (4) — subtype malig/benign on atlas samples:
            #   bce_m / bce_b shape (B, L, D, H, W), L = len(lesion_classes).
            #   dice_m / dice_b shape (B, L) already (from DiceLossMultiClass
            #   with size_average=False).
            if bce_m is not None:
                meta['subtype_bce_malig_per_lesion']  = bce_m[b].sum(dim=(-1, -2, -3)).tolist()
                meta['subtype_bce_benign_per_lesion'] = bce_b[b].sum(dim=(-1, -2, -3)).tolist()
            if dice_m is not None:
                meta['subtype_dice_malig_per_lesion']  = dice_m[b].tolist()
                meta['subtype_dice_benign_per_lesion'] = dice_b[b].tolist()

            # Section (5b) — UFO sub-type seg (pdac / pnet / cyst channels):
            #   ufo_bce_s_finest  shape (B, 3, D, H, W).
            #   ufo_dice_s_finest shape (B, 3).
            # Only present when the UFO seg path actually fired for this
            # batch (penalize_full.any() == True and ball_loss ran).
            if ufo_bce_s_finest is not None:
                meta['ufo_subtype_bce_per_channel']  = ufo_bce_s_finest[b].sum(dim=(-1, -2, -3)).tolist()
            if ufo_dice_s_finest is not None:
                meta['ufo_subtype_dice_per_channel'] = ufo_dice_s_finest[b].tolist()
            # Label for the per-channel order in the UFO seg breakdowns.
            meta['ufo_subtype_channel_order'] = ['pancreatic_pdac', 'pancreatic_pnet', 'pancreatic_cyst']

            # ---- OLD-path per-(B, L) breakdowns (from auto_distill_malignancy_loss) ----
            # These tell us, per lesion channel: where labels existed, where
            # penalization was applied, and what the per-channel loss was.
            # The `classes_old` for the OLD path has stripped pdac/pnet/cyst,
            # so L here = #lesion classes in classes_old (kidney, liver,
            # pancreatic_lesion). Order matches the `lesion_classes` meta key.
            meta['old_path_channel_order'] = [c for c in sorted(classes_old) if 'lesion' in c]
            for _k in ('loss_malig_bce_per_bc', 'loss_benign_bce_per_bc',
                       'loss_malig_dice_per_bc', 'loss_benign_dice_per_bc',
                       'malignant_label_per_bc', 'benign_label_per_bc',
                       'unknown_malignancy_label_per_bc',
                       'penalize_known_malignancy_per_bc',
                       'ball_applied_per_bc'):
                if _k in old_losses_per_bc:
                    meta[f'old_{_k}'] = old_losses_per_bc[_k][b].detach().cpu().tolist()

            # ---- NEW-path (section 4) per-(B, L) breakdowns ----
            # Mirrors the OLD fields. `penalize` here is section (4)'s active-
            # cell broadcast (1 everywhere for active atlas-subtype pairs,
            # 0 otherwise). Counts therefore saturate at D×H×W for active.
            meta['new_section4_mal_label_per_lesion'] = mal_label[b].sum(dim=(-1, -2, -3)).tolist()
            meta['new_section4_ben_label_per_lesion'] = ben_label[b].sum(dim=(-1, -2, -3)).tolist()
            meta['new_section4_penalize_per_lesion']  = penalize[b].sum(dim=(-1, -2, -3)).tolist()

            # ---- NEW-path (section 5b UFO seg) per-(B, 3) breakdowns ----
            # `target` / `penalize_seg` are shape (B, 3, D, H, W) when the UFO
            # seg path fired; None otherwise (gated off on atlas batches).
            if target is not None:
                meta['new_ufo_target_per_channel']   = target[b].sum(dim=(-1, -2, -3)).tolist()
            if penalize_seg is not None:
                meta['new_ufo_penalize_per_channel'] = penalize_seg[b].sum(dim=(-1, -2, -3)).tolist()

            # ---- Overlap detection — DOUBLE-PENALIZATION check ----
            # For every (sample, lesion-channel) we want:
            #   1. OLD penalize ∩ NEW section-4 penalize  (per lesion class)
            #   2. OLD penalize ∩ NEW section-5b penalize (per lesion-->subtype mapping)
            # If both paths penalize the same voxels in the same channel, the
            # loss is applied twice — the checker / tests flag this.
            # Both masks live at input resolution (medformer convention) and
            # share shape (B, L_old, D, H, W) vs (B, 3, D, H, W). The overlap
            # calc is per-sample b.
            if 'penalize_known_malignancy_mask' in old_losses_per_bc:
                old_pen_mask = old_losses_per_bc['penalize_known_malignancy_mask'][b]  # (L_old, D, H, W)
                # Find pancreatic_lesion column in OLD's lesion layout.
                old_les_cols = [c for c in sorted(classes_old) if 'lesion' in c]
                old_pan_idx = (old_les_cols.index('pancreatic_lesion')
                               if 'pancreatic_lesion' in old_les_cols else None)
                # Section (4) overlap per lesion channel — the NEW path's
                # penalize is 1 over full spatial when active_cell, 0 otherwise.
                # Overlap voxel count per OLD lesion channel.
                overlap_s4 = []
                for i, _ in enumerate(old_les_cols):
                    # section (4) penalize uses the SAME lesion ordering.
                    new_pen_s4 = penalize[b, i] if i < penalize.shape[1] else None
                    if new_pen_s4 is None:
                        overlap_s4.append(None)
                    else:
                        overlap_s4.append(int(((old_pen_mask[i] > 0) & (new_pen_s4 > 0)).sum().item()))
                meta['overlap_old_vs_section4_per_lesion'] = overlap_s4
                # Section (5b) CROSS-CHANNEL spatial overlap — this is NOT a
                # double-penalty check. OLD supervises the pancreatic_lesion
                # OUTPUT channel; section 5b supervises the pdac / pnet / cyst
                # OUTPUT channels. Those are different channels in the model
                # output, so overlapping penalize regions spatially does NOT
                # produce conflicting or doubled gradients — each gradient
                # flows through its own output channel. Non-zero entries here
                # are expected and benign (e.g. any Merlin cyst batch where
                # ball_loss fires will show non-zero overlap on the absent
                # pdac/pnet channels, because section 5b's `non_target_pen =
                # unk_voxels[pancreatic_pdac]` covers the pancreas sub-segment
                # and OLD's penalize also covers most of the crop outside the
                # ball). The real double-penalty check is
                # `overlap_old_vs_section4_per_lesion` (same channel, both
                # paths), which should always be 0.
                if penalize_seg is not None and old_pan_idx is not None:
                    overlap_s5b = []
                    for sub_ch in range(penalize_seg.shape[1]):
                        overlap_s5b.append(int(
                            ((old_pen_mask[old_pan_idx] > 0) &
                             (penalize_seg[b, sub_ch] > 0)).sum().item()))
                    meta['cross_channel_spatial_overlap_old_lesion_vs_section5b'] = overlap_s5b

            # ---- Crop-target inference (what was this sample cropped on?) ----
            # IMPORTANT atlas/UFO distinction:
            #   - For UFO samples, the dataset populates
            #     `chosen_segment_mask[b, organ_channel]` with the organ's
            #     sub-segment mask when the crop is organ-targeted; it's
            #     all-zero on random / background crops. So for UFO samples,
            #     this IS the reliable "crop target" signal.
            #   - For ATLAS samples, the dataset explicitly zeros
            #     `chosen_segment_mask` (dataset_abdomenatlas_UFO_multi_tumor.py
            #     line 2057, comment: "it is important to define this as 0 —
            #     or it will cause loss problems"). So an empty mask on atlas
            #     says NOTHING about the crop target — atlas crops can still
            #     be tumor-targeted via `--crop_on_tumor` on per-voxel labels.
            # We derive a second signal — `any lesion voxel in crop` — which
            # is a reliable proxy for "tumor-targeted crop" on atlas samples.
            cs = chosen_segment_mask[b]        # (C, D, H, W)
            nonzero_classes = [classes[i] for i in range(cs.shape[0])
                               if cs[i].sum().item() > 0]
            meta['chosen_segment_mask_nonzero_classes'] = nonzero_classes
            meta['chosen_segment_mask_empty'] = (len(nonzero_classes) == 0)
            # Pancreas tissue presence in THIS crop — separate from
            # "crop targets pancreas". unk_voxels[pancreatic_lesion]
            # is populated by `assign_lesion_labels_from_report`
            # exactly when a pancreas voxel lives in the crop (per
            # the user's dataset contract). Any of these signals
            # firing means the crop has pancreas tissue in it:
            #   (a) label[pancreatic_lesion].sum > 0   (atlas per-voxel)
            #   (b) chosen_segment_mask[pancreatic_lesion].sum > 0
            #                                          (UFO targeted crop)
            #   (c) unk_voxels[pancreatic_lesion].sum > 0
            #                                          (UFO with report,
            #                                           pancreas overlap)
            pan_idx = (classes.index('pancreatic_lesion')
                       if 'pancreatic_lesion' in classes else None)
            if pan_idx is not None:
                a = float(label[b, pan_idx].sum().item())
                c = float(chosen_segment_mask[b, pan_idx].sum().item())
                u = (float(unk_voxels[b, pan_idx].sum().item())
                     if unk_voxels is not None else 0.0)
                meta['pancreas_labels_in_crop'] = bool(a + c + u > 0)
            else:
                meta['pancreas_labels_in_crop'] = False
            # Atlas-friendly proxy: does the crop contain ANY lesion voxel?
            # (If yes → tumor-targeted for atlas, or a random-lucky crop.
            # If no → background / random / empty-tumor crop.)
            lesion_in_crop = [classes[i] for i in range(label.shape[1])
                              if ('lesion' in classes[i] and label[b, i].sum().item() > 0)]
            meta['lesion_classes_in_crop'] = lesion_in_crop
            # Save the mask itself so you can visually inspect the crop
            # extent (distinct from `penalize` which is plan-driven).
            save_tensor_as_nifti(cs.sum(0),
                                 os.path.join(out_dir, f'chosen_segment_mask_B{b}'))

            # ---- Report-derived sub-type (always, not only when UFO fires) ----
            # Lets us see the pancreas report's classification outcome
            # regardless of whether ball_loss ran.
            if tumor_type is not None and b < len(tumor_type):
                meta['tumor_type'] = tumor_type[b]
            if tumor_type_organ is not None and b < len(tumor_type_organ):
                meta['tumor_type_organ'] = tumor_type_organ[b]

            # ---- ball_loss-relevant diagnostics (mirrors the keys ball_loss
            # and auto_distill save in their own sanity dumps) ----
            meta['ball_loss_ran_for_batch'] = batch_ball_loss_ran
            if tumor_volumes is not None:
                meta['tumor_volumes']   = tumor_volumes[b].detach().cpu().tolist()
            if tumor_diameters is not None:
                meta['tumor_diameters'] = tumor_diameters[b].detach().cpu().tolist()
            if sizes_malignancy is not None:
                sm = sizes_malignancy[b].detach().cpu()
                meta['sizes_malignancy'] = sm.tolist()
                # Summarize the malignancy axis (orthogonal to the
                # tumor-type axis). These three flags drive the
                # malig/benign cls supervision independently of the
                # sub-type identity — they're what OLD auto_distill
                # actually consumes (OLD cls keys off `has_mal_scalar`
                # / `has_ben_scalar`, which DON'T filter by diameter).
                #
                # A `sizes_malignancy` row counts as "known" when the
                # malignancy flag itself is 0 or 1 — even if the
                # diameter is a sentinel (e.g. -9999999, meaning
                # "malignancy known, size unknown"). The pre-fix
                # `diam > 0` gate here diverged from OLD cls and made
                # sentinel-diameter samples (which still trigger OLD
                # cls toward max) falsely match the purely-uncertain
                # rule family in the validator.
                #
                # `has_unk_row` keeps the `diam > 0` guard — a
                # non-padding row with NaN malignancy means "tumor
                # seen but malignancy genuinely unknown"; padding
                # (diam=0, mal=NaN) doesn't count.
                diam = sm[:, 0]
                mal  = sm[:, 1]
                non_pad = (diam > 0)
                has_mal_row = bool((mal == 1).any().item())
                has_ben_row = bool((mal == 0).any().item())
                has_unk_row = bool((non_pad & (mal != mal)).any().item())
                meta['report_has_mal_row'] = has_mal_row
                meta['report_has_ben_row'] = has_ben_row
                meta['report_has_unknown_malignancy_row'] = has_unk_row
                # Legacy alias: "known" = has_mal OR has_ben.
                meta['sample_has_known_malignancy_flag'] = (
                    has_mal_row or has_ben_row)
            # Atlas-has-subtype flag — independent of ball_loss. Tells us why
            # the wrapper skipped the UFO seg/cls routing on this sample even
            # when the gating conditions were otherwise satisfied.
            if atlas_has_subtype is not None:
                meta['atlas_has_subtype'] = bool(atlas_has_subtype[b].item())

            # UFO sub-type seg targets (only present when the new path fired).
            # `tumor_type`, `tumor_type_organ`, `atlas_has_subtype` are now
            # emitted earlier (unconditionally when in scope) so downstream
            # diagnosis doesn't depend on the seg block having fired.
            if subtype_seg_loss_enabled and pan_les_col_old is not None:
                # Save ONE nifti per sub-type channel for target / penalize /
                # pred — summing across the 3 pdac/pnet/cyst channels into a
                # single nifti used to produce non-binary values (e.g. 0-3
                # depending on channel overlap) that were confusing to
                # interpret. Splitting gives you a clean binary target nifti
                # per channel, a binary penalize nifti per channel, and a
                # probability pred nifti per channel — all directly
                # interpretable in a nifti viewer.
                _sub_names = ['pdac', 'pnet', 'cyst']
                for _ch, _sub_name in enumerate(_sub_names):
                    save_tensor_as_nifti(
                        target[b, _ch],
                        os.path.join(out_dir, f'ufo_subtype_target_{_sub_name}_B{b}'))
                    save_tensor_as_nifti(
                        penalize_seg[b, _ch],
                        os.path.join(out_dir, f'ufo_subtype_penalize_{_sub_name}_B{b}'))

                # Model predictions for pdac/pnet/cyst (finest scale).
                # Save post-sigmoid one-nifti-per-channel.
                if not sigmoid_already_applied:
                    sub_pred_b = torch.sigmoid(subtype_out[b])
                else:
                    sub_pred_b = subtype_out[b]
                for _ch, _sub_name in enumerate(_sub_names):
                    save_tensor_as_nifti(
                        sub_pred_b[_ch],
                        os.path.join(out_dir, f'ufo_subtype_pred_{_sub_name}_B{b}'))
                # Count UNION across the 3 sub-type channels (matches the nifti
                # save which uses `.sum(0)`); otherwise two channels sharing
                # the same voxels (e.g. pdac/pnet both receiving
                # non_target_pen) double-count vs what the nifti shows.
                meta['ufo_subtype_target_voxels']   = int((target[b].sum(0) > 0).sum().item())
                meta['ufo_subtype_penalize_voxels'] = int((penalize_seg[b].sum(0) > 0).sum().item())

            # =================================================================
            # Unified `out_{channel}` and `target_{channel}` NIfTIs per sample.
            # - `out_*` = post-sigmoid model prediction at that channel.
            # - `target_*` = union of every target mask that any loss applies
            #   to that channel (atlas label ∪ OLD path label ∪ ball pseudo
            #   mask ∪ wrapper section-4/5b target). Binary.
            # =================================================================
            _sig = lambda t: (torch.sigmoid(t)
                              if not sigmoid_already_applied else t)

            # Gather finest-scale seg output for this sample.
            _seg_full_raw = model_output.get('segmentation')
            _seg_finest_b = None
            if _seg_full_raw is not None:
                _seg_finest_b = (_seg_full_raw[0]
                                 if isinstance(_seg_full_raw, (list, tuple))
                                 else _seg_full_raw)
            _pan_les_ci = (classes.index('pancreatic_lesion')
                           if 'pancreatic_lesion' in classes else None)
            _pan_pdac_ci = (classes.index('pancreatic_pdac')
                            if 'pancreatic_pdac' in classes else None)
            _pan_pnet_ci = (classes.index('pancreatic_pnet')
                            if 'pancreatic_pnet' in classes else None)
            _pan_cyst_ci = (classes.index('pancreatic_cyst')
                            if 'pancreatic_cyst' in classes else None)
            _pan_les_mb_col = (lesion_classes.index('pancreatic_lesion')
                               if 'pancreatic_lesion' in lesion_classes else None)

            def _save_out(name, t):
                if t is None: return
                save_tensor_as_nifti(
                    t.detach().cpu(),
                    os.path.join(out_dir, f'out_{name}_B{b}'))

            def _save_tgt(name, t):
                """Binarize at >= 0.5. OLD `auto_distill` emits SOFT
                distillation targets on the malig/benign outputs —
                `lesion_label × mpv` at losses_foundation.py:2591,2613
                where `lesion_label` accumulates `sigmoid(out) ×
                unk_voxels`. A naive `> 0` threshold lights up every
                voxel with any residual probability (1e-9 counts),
                producing a low-res organ blob over the real ball
                mask. Thresholding at 0.5 keeps only voxels the loss
                is meaningfully pushing toward 1. Sub-type targets
                are binary already (atlas label ∪ section 5b ball),
                so the threshold is a no-op for them."""
                if t is None: return
                save_tensor_as_nifti(
                    (t > 0.5).float().detach().cpu(),
                    os.path.join(out_dir, f'target_{name}_B{b}'))

            # --- Outputs (post-sigmoid) ------------------------------------
            if _seg_finest_b is not None and _pan_les_ci is not None:
                _save_out('pancreatic_lesion', _sig(_seg_finest_b[b, _pan_les_ci]))
            if _seg_finest_b is not None and _pan_pdac_ci is not None:
                _save_out('pancreatic_pdac',   _sig(_seg_finest_b[b, _pan_pdac_ci]))
            if _seg_finest_b is not None and _pan_pnet_ci is not None:
                _save_out('pancreatic_pnet',   _sig(_seg_finest_b[b, _pan_pnet_ci]))
            if _seg_finest_b is not None and _pan_cyst_ci is not None:
                _save_out('pancreatic_cyst',   _sig(_seg_finest_b[b, _pan_cyst_ci]))
            if malig_benign_finest is not None and _pan_les_mb_col is not None:
                _save_out(
                    'pancreatic_malignant',
                    _sig(malig_benign_finest[b, _pan_les_mb_col]))
                _save_out(
                    'pancreatic_benign',
                    _sig(malig_benign_finest[b, L + _pan_les_mb_col]))

            # --- Targets (binary union across every contributing loss) -----
            # `_les_tgt / _mal_tgt / _ben_tgt / _pdac_tgt / _pnet_tgt /
            # _cyst_tgt` were computed earlier for the supervision matrix
            # and already union OLD auto_distill + ball + section-4 +
            # section-5b + standard_seg target masks.
            #
            # For the malig/benign NIfTI saves we drop OLD's soft
            # distillation contribution (`malignant_label_mask` /
            # `benign_label_mask`). Those are `lesion_label × mpv/flag`
            # where `lesion_label = sigmoid(out) × unk_voxels` — strictly
            # > 0 across the entire pancreas unk region, which
            # `_union`'s `> 0` binarization picks up in full. The
            # resulting NIfTI is the whole pancreas sub-segment rather
            # than the meaningful "push toward 1" region. Ball pseudo-
            # masks and section-4 atlas labels are the only hard targets
            # worth visualizing here. The supervision matrix (above)
            # still uses `_mal_tgt` / `_ben_tgt` with OLD folded in, so
            # rule semantics for "benign fg = max via soft distillation"
            # are unaffected.
            _mal_tgt_viz = _union(_get_ball('pseudo_mask_malignant'),
                                  _mal_tgt_s4)
            _ben_tgt_viz = _union(_get_ball('pseudo_mask_benign'),
                                  _ben_tgt_s4)
            _save_tgt('pancreatic_lesion',    _les_tgt)
            _save_tgt('pancreatic_pdac',      _pdac_tgt)
            _save_tgt('pancreatic_pnet',      _pnet_tgt)
            _save_tgt('pancreatic_cyst',      _cyst_tgt)
            _save_tgt('pancreatic_malignant', _mal_tgt_viz)
            _save_tgt('pancreatic_benign',    _ben_tgt_viz)

            # =================================================================
            # CLS heads — per-channel {out, target, grad} summary. grad is
            # filled in post-backward by `save_gradient_sanity_dumps()`;
            # here we populate `out` (post-sigmoid scalar) and `target`
            # (union of all cls-loss targets). Placed at the TOP of
            # `summary` so it reads first in the YAML.
            # =================================================================
            _cls_head_main = [c for c in sorted(classes)
                              if (('background' in c) or ('lesion' in c)
                                  or ('pdac' in c) or ('pnet' in c)
                                  or ('cyst' in c))]
            def _cls_head_col(name):
                return (_cls_head_main.index(name)
                        if name in _cls_head_main else None)

            _main_cls_t = None
            _mb_cls_t = None
            for _k, _v in model_output.items():
                if 'classif' not in _k: continue
                _t = _v[0] if isinstance(_v, (list, tuple)) else _v
                if 'malig_benign_cls' in _k:
                    _mb_cls_t = _t
                else:
                    _main_cls_t = _t

            def _cls_out_scalar(t, col):
                if t is None or col is None: return None
                try:
                    return float(torch.sigmoid(t[b, col]).item())
                except Exception:
                    return None

            def _cls_target_or_none(tgt, mask):
                """If the cls loss masks this channel off (mask<=0),
                report None. Otherwise report the target (0/1)."""
                if mask is None or float(mask) <= 0:
                    return None
                return int(float(tgt) > 0)

            _cls_heads_block = {
                'lesion': {
                    'out':    _cls_out_scalar(_main_cls_t,
                                              _cls_head_col('pancreatic_lesion')),
                    'target': _cls_target_or_none(_les_cls_tgt, _les_cls_mask),
                    'grad':   None,
                },
                'malig': {
                    'out':    _cls_out_scalar(_mb_cls_t, _pan_les_mb_col),
                    'target': _cls_target_or_none(_malig_cls_tgt, _malig_cls_mask),
                    'grad':   None,
                },
                'benign': {
                    'out':    _cls_out_scalar(
                        _mb_cls_t,
                        (L + _pan_les_mb_col) if _pan_les_mb_col is not None else None),
                    'target': _cls_target_or_none(_benign_cls_tgt, _benign_cls_mask),
                    'grad':   None,
                },
                'pdac': {
                    'out':    _cls_out_scalar(_main_cls_t,
                                              _cls_head_col('pancreatic_pdac')),
                    'target': _cls_target_or_none(_pdac_cls_tgt, _pdac_cls_mask),
                    'grad':   None,
                },
                'pnet': {
                    'out':    _cls_out_scalar(_main_cls_t,
                                              _cls_head_col('pancreatic_pnet')),
                    'target': _cls_target_or_none(_pnet_cls_tgt, _pnet_cls_mask),
                    'grad':   None,
                },
                'cyst': {
                    'out':    _cls_out_scalar(_main_cls_t,
                                              _cls_head_col('pancreatic_cyst')),
                    'target': _cls_target_or_none(_cyst_cls_tgt, _cyst_cls_mask),
                    'grad':   None,
                },
            }
            # Prepend so cls_heads is the first entry under `summary`.
            if 'summary' in meta:
                meta['summary'] = {'cls_heads': _cls_heads_block,
                                   **meta['summary']}
            else:
                meta['summary'] = {'cls_heads': _cls_heads_block}

            with open(os.path.join(out_dir, f'meta_B{b}.yaml'), 'w') as fh:
                # `sort_keys=False` preserves insertion order so the YAML
                # reads in the order we built the dict: `summary` (sample,
                # losses, niftis_key) first, then raw detail. Without this,
                # PyYAML sorts alphabetically and scatters the structure.
                yaml.dump(meta, fh, sort_keys=False, default_flow_style=False,
                          width=100, allow_unicode=True)

        # ---- stash refs for the post-backward gradient capture ----
        # Called exactly when the writer committed meta_B*.yaml files
        # above. `retain_grad()` was already called up-front on the
        # output tensors (top of `malignancy_loss_with_subtype`), so
        # `.grad` will be populated after the trainer's `loss.backward()`.
        # `save_gradient_sanity_dumps()` then reads `.grad` and writes
        # NIfTIs + meta updates. Gradient tensors are shape
        # (B, ..., D, H, W) → `grad[b]` is one sample's contribution.
        global _grad_sanity_state
        seg_full = model_output.get('segmentation')
        seg_finest = (seg_full[0] if isinstance(seg_full, (list, tuple))
                      else seg_full)
        mb_finest = malig_benign_finest
        cls_outputs = {k: (v[0] if isinstance(v, (list, tuple)) else v)
                       for k, v in model_output.items() if 'classif' in k}
        _grad_sanity_state = {
            'dir': out_dir,
            'seg': seg_finest,
            'malig_benign': mb_finest,
            'cls_outputs': cls_outputs,
            'label': label,
            'unk_voxels': unk_voxels,
            'chosen_segment_mask': chosen_segment_mask,
            'classes': list(classes),
            'lesion_classes': list(lesion_classes),
            'B': B,
            'L': L,
        }

    # Ensure stale state from a previous batch can't leak through.
    global _grad_sanity_state
    _grad_sanity_state = None

    # Fire the sanity dump (no-op when gated off by rank / counter / active_cell).
    # When it DOES write, it also populates `_grad_sanity_state` for the
    # post-backward gradient-capture hook.
    _write_sanity_dumps()

    return combined


def _normalize_grad_for_viz(g):
    """Two-sided per-tensor normalization that keeps 0 at 0. Positives
    are rescaled so their max becomes +1; negatives are rescaled so
    their most-negative value becomes -1. If either side is empty
    (no positive voxels, or no negative voxels), that side is left
    untouched. End result is in `[-1, 1]`, `[0, 1]`, or `[-1, 0]`.
    The independent scaling keeps small-magnitude pushes visible when
    the other sign dominates."""
    out = g.clone()
    pos_mask = g > 0
    neg_mask = g < 0
    if pos_mask.any():
        pos_max = g[pos_mask].max()
        if pos_max > 0:
            out = torch.where(pos_mask, g / pos_max, out)
    if neg_mask.any():
        neg_min = g[neg_mask].min()
        if neg_min < 0:
            out = torch.where(neg_mask, g / neg_min.abs(), out)
    return out


def _build_pancreas_mask(b, label, unk_voxels, chosen_segment_mask, classes):
    """Approximate pancreas-tissue mask for sample b: union of label,
    unk_voxels, and chosen_segment_mask on every pancreatic_* channel
    in `classes`. Returns (D, H, W) bool tensor."""
    pan_cls_names = [c for c in classes if isinstance(c, str)
                     and c.startswith('pancreatic')]
    mask = None
    for c in pan_cls_names:
        ci = classes.index(c)
        if label is not None:
            cur = label[b, ci] > 0
        else:
            cur = None
        if unk_voxels is not None:
            u = unk_voxels[b, ci] > 0
            cur = u if cur is None else (cur | u)
        if chosen_segment_mask is not None:
            csm = chosen_segment_mask[b, ci] > 0
            cur = csm if cur is None else (cur | csm)
        if cur is None:
            continue
        mask = cur if mask is None else (mask | cur)
    if mask is None and label is not None:
        mask = torch.zeros_like(label[b, 0], dtype=torch.bool)
    return mask


def save_gradient_sanity_dumps():
    """Post-`loss.backward()` hook. Reads `.grad` from the output
    tensors the wrapper retained grad on during this batch's sanity
    dump, writes one NIfTI per channel, and appends gradient stats +
    a "real-numbers" supervision matrix to each `meta_B*.yaml`. No-op
    when `_grad_sanity_state` wasn't stashed this batch (i.e. the
    sanity dump didn't fire).

    Channels saved (one NIfTI per sample per channel):
      grad_pancreatic_lesion     — segmentation head
      grad_pancreatic_pdac       — segmentation head
      grad_pancreatic_pnet       — segmentation head
      grad_pancreatic_cyst       — segmentation head
      grad_pancreatic_malignant  — malig_benign head @ pancreatic_lesion
      grad_pancreatic_benign     — malig_benign head @ pancreatic_lesion

    NIfTI values are normalized to [0, 2] (0-grad → 1, positive → [1,2],
    negative → [0, 1)) for easy overlay in a viewer. The unnormalized
    stats live inside `meta_B*.yaml` under `gradient_stats`.
    """
    global _grad_sanity_state
    state = _grad_sanity_state
    _grad_sanity_state = None
    if state is None:
        return
    out_dir = state['dir']
    if not os.path.isdir(out_dir):
        return

    seg = state['seg']
    mb  = state['malig_benign']
    cls_outs = state['cls_outputs']
    label = state['label']
    unk_voxels = state['unk_voxels']
    csm = state['chosen_segment_mask']
    classes = state['classes']
    lesion_classes = state['lesion_classes']
    B = state['B']
    L = state['L']

    seg_grad = seg.grad if (seg is not None and seg.grad is not None) else None
    mb_grad  = mb.grad  if (mb  is not None and mb.grad  is not None) else None
    cls_grads = {k: (t.grad if t is not None else None)
                 for k, t in cls_outs.items()}

    # Resolve pancreas indices / columns.
    def _cls_idx(name):
        return classes.index(name) if name in classes else None
    pan_les_idx  = _cls_idx('pancreatic_lesion')
    pan_pdac_idx = _cls_idx('pancreatic_pdac')
    pan_pnet_idx = _cls_idx('pancreatic_pnet')
    pan_cyst_idx = _cls_idx('pancreatic_cyst')
    pan_les_mb_col = (lesion_classes.index('pancreatic_lesion')
                      if 'pancreatic_lesion' in lesion_classes else None)

    # cls_head_main (order matches split_cls_outputs_malignancy)
    cls_head_main = [c for c in sorted(classes)
                     if (('background' in c) or ('lesion' in c)
                         or ('pdac' in c) or ('pnet' in c) or ('cyst' in c))]
    def _cls_head_col(name):
        return cls_head_main.index(name) if name in cls_head_main else None

    # Per-channel spec: (nifti_name, source_tensor, channel_index_fn)
    # source is a callable returning (B, D, H, W) slice at channel.
    spec = []
    if seg_grad is not None:
        for name, ci in (('pancreatic_lesion', pan_les_idx),
                         ('pancreatic_pdac',   pan_pdac_idx),
                         ('pancreatic_pnet',   pan_pnet_idx),
                         ('pancreatic_cyst',   pan_cyst_idx)):
            if ci is not None:
                spec.append((f'grad_{name}', seg_grad, ci, 'seg'))
    if mb_grad is not None and pan_les_mb_col is not None:
        spec.append(('grad_pancreatic_malignant',
                     mb_grad, pan_les_mb_col, 'mb_malig'))
        spec.append(('grad_pancreatic_benign',
                     mb_grad, L + pan_les_mb_col, 'mb_benign'))

    # For the cls grads we grab scalars per (sample, head, channel).
    # Both `classification` (multi-class: bg + lesion + subtypes) and
    # `*_malig_benign_cls` heads may be present; consume whichever fired.
    cls_grads_per_sample = {b: {} for b in range(B)}
    for key, g in cls_grads.items():
        if g is None:
            continue
        # Determine which head this is.
        if 'malig_benign_cls' in key:
            # shape (B, 2L). First L = malig, next L = benign.
            if pan_les_mb_col is not None and g.shape[-1] >= 2 * L:
                for b in range(B):
                    cls_grads_per_sample[b]['malig'] = float(
                        g[b, pan_les_mb_col].item())
                    cls_grads_per_sample[b]['benign'] = float(
                        g[b, L + pan_les_mb_col].item())
        else:
            # shape (B, len(cls_head_main)).
            for b in range(B):
                for cls_name, short in (
                        ('pancreatic_lesion', 'lesion'),
                        ('pancreatic_pdac',   'pdac'),
                        ('pancreatic_pnet',   'pnet'),
                        ('pancreatic_cyst',   'cyst')):
                    col = _cls_head_col(cls_name)
                    if col is not None and g.shape[-1] > col:
                        cls_grads_per_sample[b][short] = float(
                            g[b, col].item())

    # ---- per-sample pass: write niftis, build stats, update yaml ----
    for b in range(B):
        meta_path = os.path.join(out_dir, f'meta_B{b}.yaml')
        if not os.path.isfile(meta_path):
            continue

        pancreas = _build_pancreas_mask(b, label, unk_voxels, csm, classes)
        grad_stats = {}
        real_matrix = {ch: {} for ch in
                       ('lesion', 'malig', 'benign',
                        'pdac', 'pnet', 'cyst')}

        # Map nifti-name → matrix row key
        name_to_row = {
            'grad_pancreatic_lesion':    'lesion',
            'grad_pancreatic_pdac':      'pdac',
            'grad_pancreatic_pnet':      'pnet',
            'grad_pancreatic_cyst':      'cyst',
            'grad_pancreatic_malignant': 'malig',
            'grad_pancreatic_benign':    'benign',
        }

        for nifti_name, grad_tensor, ch_idx, _kind in spec:
            g_b = grad_tensor[b, ch_idx]          # (D, H, W)
            # Sign-flip (×-1) then two-sided normalize. After the flip
            # POSITIVE values = loss pushing output toward 1 (max) and
            # NEGATIVE values = pushing toward 0 (min). Each side is
            # then rescaled independently (positives→[0, +1],
            # negatives→[-1, 0]) so that a dominant side doesn't
            # visually drown out the other. Zero stays at zero; if
            # either sign is absent the nifti is simply [0, +1] or
            # [-1, 0]. Raw magnitudes live in `gradient_stats`.
            g_viz = _normalize_grad_for_viz(-g_b)
            save_tensor_as_nifti(
                g_viz.detach().cpu(),
                os.path.join(out_dir, f'{nifti_name}_B{b}'))
            # Stats
            if pancreas is not None:
                in_pan = g_b[pancreas]
                out_pan = g_b[~pancreas]
            else:
                in_pan = g_b.reshape(-1)
                out_pan = g_b.reshape(-1)[:0]
            max_in_pan = (float(in_pan.max().item())
                          if in_pan.numel() > 0 else 0.0)
            min_in_pan = (float(in_pan.min().item())
                          if in_pan.numel() > 0 else 0.0)
            max_out_pan = (float(out_pan.max().item())
                           if out_pan.numel() > 0 else 0.0)
            min_out_pan = (float(out_pan.min().item())
                           if out_pan.numel() > 0 else 0.0)
            grad_stats[nifti_name] = {
                'max_grad_in_pancreas': max_in_pan,
                'min_grad_in_pancreas': min_in_pan,
                'max_grad_background':  max_out_pan,
                'min_grad_background':  min_out_pan,
                'pancreas_voxels':      int(pancreas.sum().item()
                                             if pancreas is not None else 0),
                'abs_max':              float(g_b.abs().max().item()),
            }

            # "Real-number" matrix: FG ≈ most-negative gradient in the
            # FG region (target=1 voxels, approximated by pancreas
            # interior where this channel could be present → use
            # min_in_pancreas). BG ≈ most-positive gradient in the BG
            # region (outside pancreas → max_out_pan). Sign convention:
            # negative grad = push-to-1, positive grad = push-to-0.
            row = name_to_row[nifti_name]
            real_matrix[row]['segmentation_foreground'] = min_in_pan
            real_matrix[row]['segmentation_background'] = max_out_pan

        # Fill in cls row of the real-matrix.
        for short, g_val in cls_grads_per_sample.get(b, {}).items():
            if short in real_matrix:
                real_matrix[short]['classification'] = g_val

        # Also expose the raw cls grads in stats.
        grad_stats['classification_grads'] = cls_grads_per_sample.get(b, {})

        # Merge into the existing meta YAML.
        try:
            with open(meta_path) as fh:
                meta = yaml.safe_load(fh) or {}
        except Exception:
            continue
        summary = meta.setdefault('summary', {})
        summary['gradient_stats'] = grad_stats
        summary['supervision_matrix_grads'] = real_matrix

        # Fill in `grad` for each cls_heads entry (populated pre-backward
        # in `_write_sanity_dumps`, `grad` was None). Sign flip matches
        # the NIfTI convention: negative raw grad → push-to-1, and after
        # ×-1 it comes out positive in the YAML too.
        if isinstance(summary.get('cls_heads'), dict):
            for short, g_val in cls_grads_per_sample.get(b, {}).items():
                if short in summary['cls_heads'] and g_val is not None:
                    summary['cls_heads'][short]['grad'] = float(-g_val)

        with open(meta_path, 'w') as fh:
            yaml.dump(meta, fh, sort_keys=False, default_flow_style=False,
                      width=100, allow_unicode=True)