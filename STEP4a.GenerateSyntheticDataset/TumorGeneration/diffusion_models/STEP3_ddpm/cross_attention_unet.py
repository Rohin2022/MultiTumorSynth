import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels, groups=32):
    return GroupNorm32(groups, channels)


def conv_nd(*args, **kwargs):
    return nn.Conv3d(*args, **kwargs)


def avg_pool_nd(*args, **kwargs):
    return nn.AvgPool3d(*args, **kwargs)


def checkpoint(func, inputs, params, flag):
    if flag:
        return torch.utils.checkpoint.checkpoint(func, *inputs)
    return func(*inputs)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        
        # Excellent fix here on your part for the context_dim handling
        if context_dim is None:
            self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        else:
            self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
            
        self.to_k_self = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v_self = nn.Linear(query_dim, inner_dim, bias=False)

        # --- CORRECTED: QK-Normalization Layers ---
        # Applied to dim_head, not inner_dim
        self.q_norm = nn.LayerNorm(dim_head, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(dim_head, elementwise_affine=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        
        if context.shape[1] == 1:
            k = self.to_k_self(x)
            v = self.to_v_self(x)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        # 1. Split into multi-head format FIRST
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # 2. Apply QK-Normalization PER-HEAD
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 3. Calculate Similarity
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        if self.training:
            with torch.no_grad():
                ent = -(attn * (attn + 1e-9).log()).sum(-1).mean()
                self.last_entropy = ent.detach()

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                     heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv3d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
             for _ in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv3d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context=None):
        b, c, hh, ww, dd = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w d -> b (h w d) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=hh, w=ww, d=dd)
        x = self.proj_out(x)
        return x + x_in


class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = conv_nd(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = conv_nd(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Block3D(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=True,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class Unet3D_CA(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels=8,
        out_dim=None,
        num_organs=9,
        num_continuous_conditioners=10,
        cond_channels=0,
        num_res_blocks=2,
        attention_resolutions=(1, 2, 4),
        dropout=0.0,
        num_heads=8,
        transformer_depth=1,
        use_scale_shift_norm=True,
        use_checkpoint=False,
        conv_resample=True,
    ):
        super().__init__()
        self.channels = channels
        self.cond_channels = cond_channels
        self.out_channels = out_dim or channels
        self.model_channels = dim
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions

        self.tabular_cond_dim = num_organs + num_continuous_conditioners
        self.num_heads = num_heads
        # NOTE: unlike a fixed tabular_emb_dim, SpatialTransformer's inner_dim
        # (= num_heads * dim_head) now varies per resolution level -- see the
        # `legacy` dim_head formula below, matching the source's
        # `dim_head = ch // num_heads if use_spatial_transformer else num_head_channels`
        # under `legacy=True`. inner_dim == ch at every level, by construction.
        context_dim = 1

        # NOTE: no learned null-conditioning parameter anymore. Null
        # conditioning is now implemented via the single-dummy-token /
        # self-attention-fallback mechanism in forward() below, matching the
        # source repo's approach (see forward() docstring-style comment).

        time_embed_dim = dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        in_channels = channels + cond_channels

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(in_channels, dim, 3, padding=1))]
        )
        input_block_chans = [dim]
        ch = dim
        ds = 1
        for level, mult in enumerate(dim_mults):
            for _ in range(num_res_blocks):
                layers = [
                    Block3D(
                        ch, time_embed_dim, dropout,
                        out_channels=mult * dim,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * dim
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(dim_mults) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        mid_dim_head = ch // num_heads
        self.middle_block = TimestepEmbedSequential(
            Block3D(ch, time_embed_dim, dropout, use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm),
            SpatialTransformer(ch, num_heads, mid_dim_head, depth=transformer_depth, context_dim=context_dim),
            Block3D(ch, time_embed_dim, dropout, use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(dim_mults))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    Block3D(
                        ch + ich, time_embed_dim, dropout,
                        out_channels=dim * mult,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = dim * mult
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(Upsample(ch, conv_resample, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dim, self.out_channels, 3, padding=1)),
        )

    def forward(self, x, t, cond=None, tabular_cond=None, null_cond_prob=0.0):
        if self.cond_channels > 0:
            assert cond is not None, "cond_channels > 0 but no cond tensor was passed"
            x = torch.cat([x, cond], dim=1)

        b = x.shape[0]

        # Decide, for this whole forward call, whether to use null conditioning.
        # tabular_cond=None always means null; otherwise null_cond_prob is a
        # per-call (effectively per-batch) dropout probability. This is a
        # single decision for the whole batch, not per-sample -- matching the
        # source repo's dataset-duplication approach, whose stochastic mix
        # emerges over many batches/epochs rather than within one batch.
        use_null = tabular_cond is None
        if not use_null and null_cond_prob > 0:
            use_null = bool(torch.rand(1).item() < null_cond_prob)

        if use_null:
            # Null-conditioning mechanism matching the source repo (TS-Radiomics)
            # rather than a learned null embedding: pass a single dummy token
            # (context.shape[1] == 1) instead of tabular_cond_dim real tokens.
            # CrossAttention's `if context.shape[1] == 1:` branch then bypasses
            # to_k/to_v entirely and falls back to pure self-attention
            # (to_k_self/to_v_self on x itself). This means the real-radiomics
            # projection weights (to_k/to_v) never have to also learn to
            # represent "absence of conditioning" -- a more stable joint
            # optimization target than a learned null vector routed through
            # the same weights as real conditioning.
            context = torch.ones(b, 1, 1, device=x.device, dtype=x.dtype)
        else:
            # Clamp z-scored radiomics to a fixed range before they become
            # attention tokens. Each scalar is projected independently via
            # Linear(1, inner_dim) with no other values to dampen it (unlike
            # a shared FiLM-MLP), so a single heavy-tailed feature (e.g.
            # glcm_contrast) can otherwise produce one outlier key/value that
            # destabilizes the softmax. This clamp is a no-op for
            # well-behaved samples and only affects rare outliers.
            #tabular_cond = tabular_cond.clamp(-tabular_clamp, tabular_clamp)
            context = tabular_cond.unsqueeze(-1)

        hs = []
        t_emb = timestep_embedding(t, self.model_channels)
        emb = self.time_embed(t_emb)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        return self.out(h)

    def forward_with_cond_scale(self, x, t, cond=None, tabular_cond=None, cond_scale=1.):
        """
        Classifier-free-guidance inference wrapper, matching the call pattern used by
        p_mean_variance:
            denoise_fn.forward_with_cond_scale(x, t, cond=cond, tabular_cond=tabular_cond,
                                                cond_scale=cond_scale)

        Runs one conditional pass (real tabular_cond) and, if cond_scale != 1, one
        unconditional pass (tabular_cond forced to the learned null embedding via
        null_cond_prob=1.), then extrapolates:
            out = null_out + (cond_out - null_out) * cond_scale
        which is standard CFG. At cond_scale == 1. this reduces to the plain
        conditional forward pass (no extra compute).
        """
        cond_out = self.forward(x, t, cond=cond, tabular_cond=tabular_cond, null_cond_prob=0.)

        if cond_scale == 1. or tabular_cond is None:
            return cond_out

        null_out = self.forward(x, t, cond=cond, tabular_cond=tabular_cond, null_cond_prob=1.)
        return null_out + (cond_out - null_out) * cond_scale