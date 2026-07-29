"""
EvoTumor -- volume x sphericity sweep test.

Generates a grid of synthetic tumors across a cross product of
(volume_scale, target_sphericity) values, using the EXACT pipeline
functions from TumorGeneration.tumor_gen_utils (synthesize_tumor,
prepare_mask_model, prepare_tumor_model, sample_radiomics, get_size, etc.)
-- nothing here reimplements or duplicates their internals.

To control size/sphericity per-cell, we monkeypatch
`tumor_gen_utils.sample_radiomics` for the duration of each call to
`synthesize_tumor`, so `synthesize_tumor` still runs its normal body
unmodified, it just receives our controlled radiomics dict instead of a
fresh GMM draw. This guarantees the sweep exercises the identical code
path as your real inference script (generate_samples.py), not a
hand-copied reimplementation.

Run directly:
    python tumor_volume_sphericity_sweep.py
Assumes it runs from the same working directory as your hydra entrypoint
(TumorGeneration importable, config/ discoverable by hydra), same as
generate_samples.py.
"""

import copy
import json
import math
import os


import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import marching_cubes

from TumorGeneration import tumor_gen_utils

from TumorGeneration.tumor_gen_utils import (
    get_size,
    prepare_mask_model,
    prepare_tumor_model,
    sample_radiomics,
    synthesize_tumor,
)
from dataset.dataloader import get_healthy_loader



# --------------------------------------------------------------------------- #
# Config -- edit these
# --------------------------------------------------------------------------- #
CONFIG_DIR = os.path.abspath("config")   # adjust if your config/ lives elsewhere
CONFIG_NAME = "synthesis"                # same config_name used in generate_samples.py


VOLUME_SCALES = [1.0, 2.0, 3.0, 4.0, 5.0]           # linear scale factors (not raw volume multipliers)
SPHERICITY_TARGETS = [0.30, 0.50, 0.70, 0.90]  # low (irregular) -> near-spherical (1.0)  # low (irregular) -> near-spherical (1.0)

OUT_DIR = "sweep_outputs"


# --------------------------------------------------------------------------- #
# Radiomics variant helpers
#
# These only build/modify plain dicts in the same shape sample_radiomics()
# returns ({"mask_radiomics": {...}, "tumor_radiomics": {...}}) -- no
# pipeline logic is duplicated here.
# --------------------------------------------------------------------------- #
_LINEAR_FIELDS = [
    "diameter_x_mm", "diameter_y_mm", "diameter_z_mm",
    "original_shape_LeastAxisLength", "original_shape_MajorAxisLength",
    "original_shape_MinorAxisLength",
    "original_shape_Maximum2DDiameterColumn", "original_shape_Maximum2DDiameterRow",
    "original_shape_Maximum2DDiameterSlice", "original_shape_Maximum3DDiameter",
]
_AREA_FIELDS = ["original_shape_SurfaceArea"]
_VOLUME_FIELDS = ["original_shape_MeshVolume", "original_shape_VoxelVolume"]
# Elongation / Flatness / Sphericity / SurfaceVolumeRatio are shape-normalized
# ratios and are left untouched by a pure size scale (a bigger tumor of the
# same shape has the same elongation/flatness/sphericity). SurfaceVolumeRatio
# does change under isotropic scaling (~ 1/s), handled explicitly below.


def make_size_variant(base_mask_dict, scale_factor):
    """Scale a mask-radiomics dict as if the tumor grew by linear factor
    `scale_factor`, treating it as a similar ellipsoid: linear extents ~ s,
    surface area ~ s^2, volume ~ s^3. Keeps shape ratios (elongation,
    flatness, sphericity) fixed."""
    m = copy.deepcopy(base_mask_dict)
    for k in _LINEAR_FIELDS:
        m[k] = base_mask_dict[k] * scale_factor
    for k in _AREA_FIELDS:
        m[k] = base_mask_dict[k] * (scale_factor ** 2)
    for k in _VOLUME_FIELDS:
        m[k] = base_mask_dict[k] * (scale_factor ** 3)
    m["original_shape_SurfaceVolumeRatio"] = (
        base_mask_dict["original_shape_SurfaceVolumeRatio"] / scale_factor
    )
    return m


def make_sphericity_variant(base_mask_dict, target_sphericity):
    """Hold volume fixed, solve for the surface area implied by
    target_sphericity (sphericity = (36*pi*V^2)^(1/3) / SurfaceArea), and
    overwrite Sphericity + SurfaceArea + SurfaceVolumeRatio consistently so
    the triple never becomes internally inconsistent."""
    m = copy.deepcopy(base_mask_dict)
    volume = base_mask_dict["original_shape_MeshVolume"]

    ideal_numerator = (36.0 * math.pi * (volume ** 2)) ** (1.0 / 3.0)
    target_sphericity = min(max(target_sphericity, 1e-3), 1.0)
    implied_surface_area = ideal_numerator / target_sphericity

    m["original_shape_Sphericity"] = target_sphericity
    m["original_shape_SurfaceArea"] = implied_surface_area
    m["original_shape_SurfaceVolumeRatio"] = implied_surface_area / volume
    return m


def make_joint_variant(base_mask_dict, scale_factor, target_sphericity):
    """Apply size scaling first, then re-solve surface area for the target
    sphericity AT that scaled volume, so (volume, surface_area, sphericity)
    stay consistent together."""
    sized = make_size_variant(base_mask_dict, scale_factor)
    return make_sphericity_variant(sized, target_sphericity)


# --------------------------------------------------------------------------- #
# Monkeypatch hook: forces synthesize_tumor's internal sample_radiomics()
# call to return our controlled radiomics dict instead of a fresh GMM draw.
# synthesize_tumor's own code is untouched and unduplicated.
# --------------------------------------------------------------------------- #
def run_synthesize_tumor_with_override(radiomics_override, **synth_kwargs):
    original_fn = tumor_gen_utils.sample_radiomics

    def _patched(gmm_bank_path, organ):
        return radiomics_override

    tumor_gen_utils.sample_radiomics = _patched
    try:
        return synthesize_tumor(**synth_kwargs, just_mask=True, cond_scale=4.0)
    finally:
        tumor_gen_utils.sample_radiomics = original_fn


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def extract_mask_shell(mask_np, upsample_factor=5, smooth_sigma=0.85):
    """Extracts the tumor mask's outer surface as a triangle mesh via
    marching cubes, for a full 3D shell render. Returns (verts, faces), or
    (None, None) if the mask is empty (marching_cubes needs a non-trivial
    isosurface).

    Small masks (a handful of voxels) produce blocky, barely-3D-looking
    shells straight off the raw binary grid, so we upsample the mask onto a
    finer grid and gaussian-smooth it before running marching cubes -- this
    gives a proper rounded surface instead of a jagged voxel cluster.
    `upsample_factor` was bumped from 3 -> 5 and `smooth_sigma` trimmed
    slightly (1.0 -> 0.85 grid units, i.e. roughly the same *physical*
    smoothing once you account for the finer grid) so the resulting mesh
    keeps more real surface undulation instead of getting smoothed into a
    featureless blob -- that loss of surface detail was part of why shells
    looked "dull" even before considering lighting.
    Coordinates are returned in ORIGINAL voxel units (rescaled back down),
    so they stay comparable across cells.
    """
    if mask_np.sum() == 0:
        return None, None

    padded = np.pad(mask_np, pad_width=1, mode="constant", constant_values=0)

    zoomed = zoom(padded.astype(np.float32), zoom=upsample_factor, order=1)
    smoothed = gaussian_filter(zoomed, sigma=smooth_sigma * upsample_factor)

    verts, faces, _, _ = marching_cubes(smoothed, level=smoothed.max() * 0.35)
    verts = verts / upsample_factor - 1  # undo upsampling + padding offset
    return verts, faces


def shade_faces(verts, faces, light_dir=(0.4, -0.5, 0.75), base_color=(1.0, 0.35, 0.05)):
    """Computes a per-face lighting color (Lambertian diffuse + a
    Blinn-Phong specular kick + a soft rim light) so the mesh reads as a
    glossy, detailed 3D object even when rendered small inside a subplot
    grid.

    The old version clamped diffuse intensity to a narrow [0.55, 1.0]
    band, so nearly every face came out roughly the same brightness --
    that's exactly what "dull, low-detail" looks like once the mesh is
    shrunk into a thumbnail (the shape info is all still there, it just
    doesn't show up as contrast). This version:
      - widens the diffuse range so shadowed vs lit faces actually differ,
      - adds a Blinn-Phong specular highlight, which puts a bright glossy
        streak on curved regions and reads as "detail" at a glance,
      - adds a faint rim light from behind so the silhouette edge doesn't
        go flat/dark against the background.
    Returns an (n_faces, 4) RGBA array to pass as Poly3DCollection
    facecolors.
    """
    tris = verts[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    normals = normals / norm_len

    light = np.array(light_dir) / np.linalg.norm(light_dir)
    view = np.array([0.0, -0.3, 1.0])  # roughly matches ax.view_init(elev=20, azim=-60)
    view = view / np.linalg.norm(view)

    # Diffuse: pushed even wider (0.08 -> 1.0) so shadowed faces go
    # genuinely dark instead of just "a bit dimmer" -- this is the single
    # biggest lever for making the shape read as distinct/sculpted.
    diffuse = np.clip(normals @ light, 0.0, 1.0)
    diffuse = diffuse ** 0.8  # gamma bend: brightens mid-tones, keeps deep shadows
    diffuse = 0.08 + 0.92 * diffuse

    # Specular: Blinn-Phong highlight, now brighter and a bit broader
    # (lower exponent) so it reads as a clear glossy patch rather than a
    # tiny pinprick.
    half_vec = light + view
    half_vec = half_vec / np.linalg.norm(half_vec)
    spec = np.clip(normals @ half_vec, 0.0, 1.0) ** 16
    specular = 0.9 * spec

    # Rim light: stronger secondary light from behind so the silhouette
    # edge pops distinctly against the background instead of just avoiding
    # going flat.
    rim_light = -light
    rim = np.clip(normals @ rim_light, 0.0, 1.0) ** 1.5
    rim_boost = 0.35 * rim

    base = np.array(base_color)
    colors = base[None, :] * diffuse[:, None]
    colors = colors + specular[:, None] * np.array([1.0, 1.0, 1.0])[None, :]
    colors = colors + rim_boost[:, None] * base[None, :]
    colors = np.clip(colors, 0.0, 1.0)
    rgba = np.concatenate([colors, np.ones((len(colors), 1))], axis=1)
    return rgba


def process_mask(tumor_mask):
    mask_np = tumor_mask[0, 0].detach().cpu().numpy().astype(np.uint8)
    return mask_np


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ----------------------------------------------------------------- #
    # 1. Load config via Hydra (same config as generate_samples.py)
    # ----------------------------------------------------------------- #
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg: DictConfig = compose(config_name=CONFIG_NAME)

    torch.cuda.set_device(cfg.inference.gpu_idxs)
    device = torch.device(f"cuda:{cfg.inference.gpu_idxs}")

    # ----------------------------------------------------------------- #
    # 2. Load models -- identical calls to generate_samples.py
    # ----------------------------------------------------------------- #
    mask_tester = prepare_mask_model(device, cfg)
    tumor_tester = prepare_tumor_model(device, cfg)
    gmm_bank_path = cfg.paths.radiomics_gmm_bank

    with open(cfg.dataset.tumor_norm_stats) as f:
        tumor_norm_stats = json.load(f)
    with open(cfg.dataset.mask_norm_stats) as f:
        mask_norm_stats = json.load(f)

    # ----------------------------------------------------------------- #
    # 3. Grab one real healthy sample -- identical to generate_samples.py's
    #    batch-loop unpacking, just for a single batch
    # ----------------------------------------------------------------- #
    healthy_loader, _, _ = get_healthy_loader(cfg.dataset)
    batch = next(iter(healthy_loader))

    ct = batch["image"].to(device)
    organ_mask = batch["organ_mask"].to(device)
    m_organ_mask = batch["m_organ_mask"].to(device)
    heatmap = batch["heatmap"].to(device)
    organ = batch["organ"][0]
    bdmap_id = batch["bdmap_id"][0]

    print(f"Using bdmap_id={bdmap_id}, organ={organ}")

    # ----------------------------------------------------------------- #
    # 4. Base radiomics draw -- exact call used by synthesize_tumor
    #    internally, done once up front so every grid cell shares the
    #    same non-swept features (texture/appearance, unperturbed shape
    #    columns) and only volume/sphericity vary between cells.
    # ----------------------------------------------------------------- #
    base_radiomics = sample_radiomics(gmm_bank_path, organ)
    base_mask = base_radiomics["mask_radiomics"]

    print("Base mask radiomics (unperturbed draw):")
    for k in ["diameter_x_mm", "diameter_y_mm", "diameter_z_mm",
              "original_shape_MeshVolume", "original_shape_VoxelVolume",
              "original_shape_SurfaceArea", "original_shape_Sphericity"]:
        print(f"  {k}: {base_mask[k]:.3f}")

    if base_mask["original_shape_MeshVolume"] <= 0 or base_mask["original_shape_VoxelVolume"] <= 0:
        raise ValueError(
            "Base GMM draw has non-positive volume "
            f"(MeshVolume={base_mask['original_shape_MeshVolume']:.3f}, "
            f"VoxelVolume={base_mask['original_shape_VoxelVolume']:.3f}). "
            "The size/sphericity scaling assumes a positive-volume base draw "
            "to scale from -- this GMM sample is likely an outlier or the "
            "bank's volume feature isn't in the units this script assumes. "
            "Try re-running (sample_radiomics draws randomly) or inspect "
            "gmm_bank_path's VoxelVolume distribution directly."
        )

    # ----------------------------------------------------------------- #
    # 5. Run the grid -- each cell calls synthesize_tumor() directly
    #    (via the monkeypatch), so this exercises literally the same
    #    code path as generate_samples.py.
    # ----------------------------------------------------------------- #
    grid = [[None for _ in SPHERICITY_TARGETS] for _ in VOLUME_SCALES]

    for i, scale in enumerate(VOLUME_SCALES):
        for j, sph in enumerate(SPHERICITY_TARGETS):
            variant_mask = make_joint_variant(base_mask, scale, sph)
            radiomics_override = {
                "mask_radiomics": variant_mask,
                "tumor_radiomics": copy.deepcopy(base_radiomics["tumor_radiomics"]),
            }
            tag = f"vol{scale:.2f}_sph{sph:.2f}"

            _, tumor_mask, _ = run_synthesize_tumor_with_override(
                radiomics_override,
                ct_volume=ct,
                organ_mask=organ_mask,
                heatmap=heatmap,
                m_organ_mask=m_organ_mask,
                organ_type=organ,
                mask_tester=mask_tester,
                tumor_tester=tumor_tester,
                gmm_bank_path=gmm_bank_path,
                tumor_norm_stats=tumor_norm_stats,
                mask_norm_stats=mask_norm_stats,
            )

            n_voxels = int(tumor_mask.sum().item())
            print(f"[{tag}] tumor mask voxels: {n_voxels}  "
                  f"(bucket={get_size(radiomics_override)}, "
                  f"target VoxelVolume={variant_mask['original_shape_VoxelVolume']:.1f})")

            mask_np = process_mask(tumor_mask)

            grid[i][j] = {
                "tag": tag,
                "scale": scale,
                "target_sphericity": sph,
                "voxel_volume": variant_mask["original_shape_VoxelVolume"],
                "achieved_mask_voxels": n_voxels,
                "mask": mask_np,
            }

    print("Grid synthesis complete:", len(VOLUME_SCALES), "x", len(SPHERICITY_TARGETS))

    # ----------------------------------------------------------------- #
    # 6. Figure -- rows = volume scale, cols = sphericity target, each
    #    subplot a full 3D shell render of the tumor mask surface
    # ----------------------------------------------------------------- #
    n_rows, n_cols = len(VOLUME_SCALES), len(SPHERICITY_TARGETS)
    # Bigger per-cell size than before (3.6in -> 4.4in) since the higher
    # mesh resolution / shading contrast needs more pixels to actually
    # show through -- at the old size + old dpi a lot of the added detail
    # was getting anti-aliased away again.
    fig = plt.figure(figsize=(4.4 * n_cols, 4.4 * n_rows))

    # Precompute every cell's mesh up front so the shared zoom level can be
    # based on actual tumor geometry (mesh extent), not the full mask array
    # shape -- using the full array shape was the source of the excess
    # padding, since a small tumor sitting in a large volume was being
    # zoomed out to fit the whole volume instead of just the tumor.
    meshes = [[None for _ in SPHERICITY_TARGETS] for _ in VOLUME_SCALES]
    max_extent = 1.0  # fallback if every cell is empty
    for i in range(n_rows):
        for j in range(n_cols):
            verts, faces = extract_mask_shell(grid[i][j]["mask"])
            meshes[i][j] = (verts, faces)
            if verts is not None:
                extent = (verts.max(axis=0) - verts.min(axis=0)).max()
                max_extent = max(max_extent, extent)

    # small margin so the shell doesn't touch the subplot edge
    half = (max_extent / 2.0) * 1.15

    for i in range(n_rows):
        for j in range(n_cols):
            cell = grid[i][j]
            verts, faces = meshes[i][j]
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection="3d")

            if verts is not None:
                face_colors = shade_faces(verts, faces)
                mesh = Poly3DCollection(
                    verts[faces], facecolor=face_colors,
                    edgecolor=(0.05, 0.05, 0.05, 0.25), linewidth=0.15,
                )
                # Slightly denser shading + no antialiasing on edges keeps
                # the specular highlight crisp instead of getting blurred
                # into the diffuse base color at small sizes.
                mesh.set_antialiased(False)
                ax.add_collection3d(mesh)
                center = verts.mean(axis=0)
            else:
                center = np.array(cell["mask"].shape) / 2.0

            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_zlim(center[2] - half, center[2] + half)

            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
            ax.view_init(elev=20, azim=-60)

            if i == 0:
                ax.set_title(f"sphericity={cell['target_sphericity']:.2f}", fontsize=11, pad=0)
            if j == 0:
                ax.text2D(
                    -0.05, 0.5, f"vol scale={cell['scale']:.2f}",
                    transform=ax.transAxes, fontsize=11, rotation=90,
                    ha="center", va="center",
                )

            ax.text2D(
                0.5, 0.02, f"{cell['achieved_mask_voxels']} vox",
                transform=ax.transAxes, fontsize=8, ha="center", va="bottom",
                color="dimgray",
            )

    fig.suptitle(f"EvoTumor sweep -- organ={organ}, bdmap_id={bdmap_id}", fontsize=13)
    fig.subplots_adjust(wspace=0.02, hspace=0.05)
    fig.tight_layout(rect=[0.02, 0.01, 1, 0.96], pad=0.3)

    fig_path = os.path.join(OUT_DIR, "sweep_grid.png")
    # dpi bumped 150 -> 220 to match the larger figsize / finer mesh so the
    # extra geometric + shading detail survives into the saved PNG instead
    # of being downsampled back out.
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    print("Figure saved to:", fig_path)

    print("Outputs written to:", OUT_DIR)
    print(os.listdir(OUT_DIR))


if __name__ == "__main__":
    main()