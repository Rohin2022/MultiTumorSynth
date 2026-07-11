import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import AdamW
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding

from ddpm.text import tokenize, bert_embed, BERT_MODEL_DIM
from torch.utils.data import Dataset, DataLoader
from vq_gan_3d.model.vqgan import VQGAN

import matplotlib.pyplot as plt

from ddpm.cross_attention_unet import CrossAttention

def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
 
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)
 
 
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        # time_emb_dim is now the dim of the CONCATENATED [t, tab_emb]
        # vector, not t alone. One shared projection produces scale/shift
        # for BOTH blocks -> 4*dim_out total, same as before.
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 4)
        ) if exists(time_emb_dim) else None
 
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
 
    def forward(self, x, time_emb=None):
        scale_shift1 = scale_shift2 = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            emb = self.mlp(time_emb)
            emb = rearrange(emb, 'b c -> b c 1 1 1')
            scale1, shift1, scale2, shift2 = emb.chunk(4, dim=1)
            scale_shift1 = (scale1, shift1)
            scale_shift2 = (scale2, shift2)
 
        h = self.block1(x, scale_shift=scale_shift1)
        h = self.block2(h, scale_shift=scale_shift2)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=129,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8,
        num_organs=9,
        num_continuous_conditioners=10,
        tabular_emb_dim=None,   # NEW: size of the tabular embedding before concat.
                                 # Defaults to time_dim // 2 below if not given —
                                 # tune this; it's now an independent hyperparameter
                                 # instead of being forced to match time_dim.
    ):
        super().__init__()
        self.channels = channels
 
        # temporal attention and its relative positional encoding
 
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
 
        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
 
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)
 
        # initial conv
 
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)
 
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size,
                                   init_kernel_size), padding=(0, init_padding, init_padding))
 
        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))
 
        # dimensions
 
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
 
        # time conditioning
 
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
 
        self.num_organs = num_organs
        self.num_continuous_conditioners = num_continuous_conditioners
        self.tabular_cond_dim = self.num_organs + self.num_continuous_conditioners
 
        # Tabular embedding is now sized independently of time_dim — it no
        # longer has to "win" a shared additive channel, it just needs to
        # carry enough information for the concat + shared FiLM-MLP inside
        # each ResnetBlock to route it into per-block scale/shift. Default
        # to time_dim // 2 as a reasonable starting point; treat as a
        # tunable hyperparameter.
        self.tabular_emb_dim = default(tabular_emb_dim, time_dim // 2)
 
        self.tabular_cond_mlp = nn.Sequential(
            nn.Linear(self.tabular_cond_dim, self.tabular_emb_dim),
            nn.SiLU(),
            nn.LayerNorm(self.tabular_emb_dim),
            nn.Linear(self.tabular_emb_dim, self.tabular_emb_dim)
        )
 
        self.tabular_null_cond_emb = nn.Parameter(
            torch.randn(1, self.tabular_cond_dim))
 
        # This is the dim every ResnetBlock's FiLM-MLP now expects —
        # concatenated [t, tab_emb], not t alone.
        fused_emb_dim = time_dim + self.tabular_emb_dim
 
        # layers
 
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
 
        num_resolutions = len(in_out)
        # block type
 
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=fused_emb_dim)
 
        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
 
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
 
        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
 
        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))
 
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))
 
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
 
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
 
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
 
        out_dim = default(out_dim, channels)
        # NOTE: final_conv's Block still takes time_emb_dim implicitly via
        # block_klass (not block_klass_cond) below, so it is UNCONDITIONED
        # on time/tabular — same as your original code. Flagging only
        # because it's easy to assume every block is conditioned; this one
        # isn't, and that isn't new to this patch.
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )
 
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or (
            kwargs.get('cond') is None and kwargs.get('tabular_cond') is None
        ):
            return logits
 
        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale
 
    def forward(
        self,
        x,
        time,
        cond=None,
        tabular_cond=None,
        textual_cond_embed=None,
        null_cond_prob=0.10,
        focus_present_mask=None,
        prob_focus_present=0.
    ):
 
        batch, device = x.shape[0], x.device
 
        drop_mask = torch.rand(batch, device=device) < null_cond_prob
 
        #if exists(cond):
        #    spatial_mask = drop_mask.view(batch, 1, 1, 1, 1)
        #    cond = torch.where(spatial_mask, torch.zeros_like(cond), cond)
        x = torch.cat([x, cond], dim=1)
 
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))
 
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
 
        x = self.init_conv(x)
        r = x.clone()
 
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
 
        t = self.time_mlp(time)
 
        # classifier free guidance — tabular conditioning is now CONCATENATED
        # with t, not added into it. Every downstream block receives the
        # fused vector and the per-block FiLM-MLP learns how to split its
        # attention between "what timestep am I at" and "what attenuation
        # was requested" instead of the two competing inside one shared
        # additive channel.
        if exists(tabular_cond):
            emb_mask = drop_mask.view(batch, 1)
 
            tabular_cond = torch.where(
                emb_mask.bool(),
                self.tabular_null_cond_emb.expand(batch, -1),
                tabular_cond
            )
 
            tab_emb = self.tabular_cond_mlp(tabular_cond)
            t = torch.cat([t, tab_emb], dim=-1)

            
        else:
            # Keep dims consistent even when tabular_cond isn't provided
            # (e.g. an ablation run) — pad with zeros rather than letting
            # ResnetBlock's Linear(fused_emb_dim, ...) receive the wrong
            # shape and crash.
            zeros = torch.zeros(batch, self.tabular_emb_dim, device=device, dtype=t.dtype)
            t = torch.cat([t, zeros], dim=-1)
 
        h = []
 
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)
 
        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)
 
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)
 
        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)

# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls=False,
        channels=3,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False, 
        dynamic_thres_percentile=0.9,
        vqgan_ckpt=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        if vqgan_ckpt:
            self.vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt,weights_only=False).cuda()
            self.vqgan.eval()
        else:
            self.vqgan = None

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, tabular_cond=None, textual_cond_embed=None, cond_scale=1.):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, tabular_cond=tabular_cond, cond_scale=cond_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, tabular_cond=None, textual_cond_embed=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, tabular_cond=tabular_cond, textual_cond_embed=textual_cond_embed, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, tabular_cond=None, textual_cond=None, cond_scale=1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        print('cond', cond.shape)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long), cond=cond, tabular_cond=tabular_cond, textual_cond_embed=textual_cond, cond_scale=cond_scale)

        return img

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1., batch_size=16):
        raise NotImplementedError("IMPLEMENT ROHIN")
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = batch_size 
        image_size = self.image_size
        channels = 8 # self.channels
        num_frames = self.num_frames
        
        _sample = self.p_sample_loop(
            (batch_size, channels, num_frames, image_size, image_size), cond=cond, cond_scale=cond_scale)

        if isinstance(self.vqgan, VQGAN):
            _sample = (((_sample + 1.0) / 2.0) * (self.vqgan.codebook.embeddings.max() -
                                                  self.vqgan.codebook.embeddings.min())) + self.vqgan.codebook.embeddings.min()

            _sample = self.vqgan.decode(_sample, quantize=True)
        else:
            unnormalize_img(_sample)

        return _sample

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond=None, tabular_cond=None, textual_cond=None, noise=None, null_cond_prob=0.10, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        textual_cond_embed = None
        if exists(textual_cond):
            textual_cond_embed = bert_embed(
                tokenize(textual_cond), return_cls_repr=self.text_use_bert_cls)
            textual_cond_embed = textual_cond_embed.to(device)

        x_recon = self.denoise_fn(x_noisy, t, cond=cond, tabular_cond=tabular_cond, null_cond_prob=null_cond_prob, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, img, mask, tabular_cond, textual_cond=None, null_cond_prob=0.10, *args, **kwargs):
        # 1. Extract binary tumor mask from ternary {0,1,2}
        tumor_mask = (mask == 2).float().detach()   # {0,1} binary, tumor region only
        mask_ = (1 - tumor_mask).detach()           # 1=keep, 0=zero-out tumor region
        masked_img = (img * mask_).detach()         # CT with tumor region zeroed out

        # 2. Permute from (B, C, H, W, D) → (B, C, D, H, W) for VQGAN
        masked_img  = masked_img.permute(0, 1, 4, 2, 3)
        img = img.permute(0, 1, 4, 2, 3)
        tumor_mask  = tumor_mask.permute(0, 1, 4, 2, 3)

        # 3. Encode through VQGAN and normalize with codebook min/max
        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                emb_min   = self.vqgan.codebook.embeddings.min()
                emb_max   = self.vqgan.codebook.embeddings.max()
                emb_denom = emb_max - emb_min

                img        = self.vqgan.encode(img,        quantize=False, include_embeddings=True)
                masked_img = self.vqgan.encode(masked_img, quantize=False, include_embeddings=True)

                img        = ((img        - emb_min) / emb_denom) * 2.0 - 1.0
                masked_img = ((masked_img - emb_min) / emb_denom) * 2.0 - 1.0
        else:
            raise RuntimeError("PLEASE USE VQGAN")

        # 4. Build spatial conditioning
        cc = torch.nn.functional.interpolate(
            tumor_mask * 2.0 - 1.0,        # {0,1} → {-1,1} ✅
            size=masked_img.shape[-3:],
            mode='nearest'
        )
        cond = torch.cat((masked_img, cc), dim=1)

        # 5. Timestep sampling and loss
        b, device = img.shape[0], img.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(
            img, t, cond=cond, tabular_cond=tabular_cond,
            null_cond_prob=null_cond_prob, *args, **kwargs
        )

# trainer class

def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5

# trainer clas
from tensorboardX import SummaryWriter
import os
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        cfg,
        folder=None,
        dataset=None,
        *,
        ema_decay=0.995,
        num_frames=16,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        num_sample_rows=1,
        max_grad_norm=None,
        num_workers=20,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.cfg = cfg
        dl=dataset

        self.len_dataloader = len(dl)
        self.dl = cycle(dl)

        self.device = "cuda" if torch.cuda.is_available() else ""
                
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith('.bias') or 'norm' in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        print(f"Parameters with decay: {len(decay)}, Parameters withOUT decay: {len(no_decay)}")

        self.opt = AdamW([
            {'params': decay, 'weight_decay': 1e-4},
            {'params': no_decay, 'weight_decay': 0.0},
        ], lr=train_lr)


        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(str(self.results_folder)+'/logs'):
            os.makedirs(str(self.results_folder)+'/logs')
        self.writer = SummaryWriter(str(self.results_folder)+'/logs')
        
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'{milestone}.pt'))

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [
                int(p.stem)
                for p in Path(self.results_folder).glob('*.pt')
                if p.stem.isdigit()
            ]
            assert len(all_milestones) > 0, \
                'need to have at least one numeric milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        # if milestone is an integer like 5, convert it to checkpoints/5.pt
        if isinstance(milestone, int):
            milestone = str(Path(self.results_folder) / f'{milestone}.pt')

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def prepare_conditional_vector(self, data, device):
        """
        Extracts tabular features into a single tensor, one-hot encoding the organ.
        Output shape: (Batch, 18) -> 9 organ classes + 9 numerical features
        """
        numerical_features = [
            "attenuation_mean", "attenuation_stdev", "attenuation_delta", # attenuation_delta is (mean_tumor - mean_organ) / std_organ
            "attenuation_skew", "attenuation_10th", "attenuation_uniformity",
            "glcm_contrast", "glcm_autocorrelation", "glcm_idm", "num_components"
        ]

        # 1. Handle the categorical "organ" feature
        organ_idx = torch.as_tensor(
            data["organ"], dtype=torch.long, device=device).view(-1)

        # One-hot encode to shape (Batch, 9) and cast back to float32
        organ_one_hot = F.one_hot(organ_idx, num_classes=9).float()

        # 2. Handle the remaining continuous numerical features
        num_tensors = []
        for key in numerical_features:
            val = torch.as_tensor(
                data[key], dtype=torch.float32, device=device).view(-1)
            num_tensors.append(val)

        # Stack continuous features to shape (Batch, 10)
        continuous_vector = torch.stack(num_tensors, dim=1)

        # 3. Concatenate the one-hot organ with the continuous features
        # Resulting shape: (Batch, 18)
        cond_vector = torch.cat([organ_one_hot, continuous_vector], dim=1)

        return cond_vector


    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)
        best_train_loss = 0.90

        loss_history = []

        while self.step < self.train_num_steps:
            skip_step = False

            for i in range(self.gradient_accumulate_every):
                data = next(self.dl)

                image = data['image'].to(self.device)
                mask = data['label'].to(self.device)

                tabular_cond = self.prepare_conditional_vector(data, device=self.device)
                
                # -- diagnostics, logged regardless of whether we skip --
                tab_max = tabular_cond.abs().max().item()
                tab_argmax = tabular_cond.abs().view(-1).argmax().item()
                img_max = image.abs().max().item()
                sample_ids = data.get('bdmap_id', None)  # adjust key to match your dataset

                with autocast(enabled=self.amp, dtype=torch.bfloat16):
                    loss = self.model(
                        image,
                        mask,
                        tabular_cond,
                        null_cond_prob=0.1
                    )

                loss_val = loss.item()

                is_nonfinite = not torch.isfinite(loss)
                is_spike = False
                median = float('nan')
                if len(loss_history) >= 50:
                    median = torch.tensor(loss_history[-50:]).median().item()
                    is_spike = loss_val > median * 5

                if is_nonfinite or is_spike:
                    reason = "non-finite" if is_nonfinite else "spike"
                    print(f"[step {self.step}] {reason} loss {loss_val:.4f} "
                        f"(recent median {median:.4f}) -- skipping. "
                        f"tab_max={tab_max:.3f} (feature idx {tab_argmax}), "
                        f"img_max={img_max:.3f}, sample_ids={sample_ids}")
                    self.writer.add_scalar('Skipped_batch/loss', loss_val, self.step)
                    self.writer.add_scalar('Skipped_batch/tabular_max', tab_max, self.step)
                    self.writer.add_scalar('Skipped_batch/image_max', img_max, self.step)
                    #self.opt.zero_grad(set_to_none=True)
                    #skip_step = True
                    #break  # abandon remaining grad-accumulation micro-batches this step

                loss.backward()

                if self.step % 10 == 0:
                    print(f'{self.step}: {loss_val}')
                    for name, module in self.model.named_modules():
                        if isinstance(module, CrossAttention) and hasattr(module, 'last_entropy'):
                            self.writer.add_scalar(f'attn_entropy/{name}', module.last_entropy.item(), self.step)

            if skip_step:
                self.step += 1
                log_fn({'loss': loss_val, 'skipped': True})
                continue  # this now continues the OUTER while loop --
                        # correctly skips opt.step, EMA, checkpointing, inference

            # -- only reached if no micro-batch this step was skipped --
            log = {'loss': loss_val}

            grad_norm = None
            if exists(self.max_grad_norm):
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            loss_history.append(loss_val)
            if len(loss_history) > 100:
                loss_history = loss_history[-75:]

            self.opt.step()
            self.opt.zero_grad()

            lr = self.opt.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('Train_Loss', loss_val, self.step)
            self.writer.add_scalar('Learning_rate', lr, self.step)
            self.writer.add_scalar('Tabular_max', tab_max, self.step)
            self.writer.add_scalar('Image_max', img_max, self.step)
            if grad_norm is not None:
                self.writer.add_scalar('Grad_norm', grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, self.step)

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

                if loss_val < best_train_loss:
                    best_train_loss = loss_val
                    self.save('model_best')
                    print(f'New best model found at step {self.step}')

            if self.step % 2000 == 0:
                print(f"\n--- Running inference at step {self.step} ---")
                self.ema_model.eval()
                
                with torch.no_grad():
                    vqgan = self.model.vqgan
                    n_samples = min(3, image.shape[0])
                    cond_scale = 3.0
                    
                    image_s = image[:n_samples]
                    mask_s  = mask[:n_samples]
                    tabular_cond_s = tabular_cond[:n_samples] if tabular_cond is not None else None
                    
                    # 1. Build Spatial Conditioning for Diffusion
                    # Extract binary tumor mask from ternary {0,1,2} — matches forward()
                    tumor_mask = (mask_s == 2).float().detach()
                    mask_      = (1 - tumor_mask).detach()
                    masked_img = (image_s * mask_).detach()
                    
                    masked_img_p = masked_img.permute(0, 1, 4, 2, 3)
                    tumor_mask_p = tumor_mask.permute(0, 1, 4, 2, 3)
                    
                    emb_min   = vqgan.codebook.embeddings.min()
                    emb_max   = vqgan.codebook.embeddings.max()
                    emb_denom = emb_max - emb_min
                    
                    latent   = vqgan.encode(masked_img_p, quantize=False, include_embeddings=True)
                    latent_n = ((latent - emb_min) / emb_denom) * 2.0 - 1.0
                    
                    cc = F.interpolate(
                        tumor_mask_p * 2.0 - 1.0,   # {0,1} → {-1,1} ✅
                        size=latent_n.shape[-3:],
                        mode='nearest'
                    )
                    spatial_cond = torch.cat([latent_n, cc], dim=1)
                    latent_shape = latent_n.shape
                    
                    # 2. Reverse Diffusion in Latent Space
                    noisy_latent = torch.randn(latent_shape, device=self.device)
                    
                    for i in tqdm(reversed(range(self.ema_model.num_timesteps)), desc=f"Sampling cfg={cond_scale}", leave=False):
                        t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)  # CHANGED: n_samples instead of batch_size
                        noisy_latent = self.ema_model.p_sample(
                            noisy_latent, t,
                            cond=spatial_cond,
                            tabular_cond=tabular_cond_s,  # CHANGED: sliced tabular cond
                            cond_scale=cond_scale,
                            clip_denoised=False
                        )
                        
                    # 3. Decode Diffusion Latent to CT
                    latent_denorm = ((noisy_latent + 1.0) / 2.0) * emb_denom + emb_min
                    decoded = vqgan.decode(latent_denorm, quantize=True)
                    ct_synth = decoded.permute(0, 1, 3, 4, 2).contiguous()

                    # ---------------------------------------------------
                    # NEW: VQGAN Autoencode (Original Image In & Out)
                    # ---------------------------------------------------
                    image_p = image_s.permute(0, 1, 4, 2, 3) # Permute to VQGAN shape  # CHANGED: image_s
                    latent_orig = vqgan.encode(image_p, quantize=False, include_embeddings=True)
                    decoded_orig = vqgan.decode(latent_orig, quantize=True)
                    ct_vqgan_recon = decoded_orig.permute(0, 1, 3, 4, 2).contiguous()
                    
                    # 4. Save NIfTI outputs
                    import nibabel as nib
                    import numpy as np
                    
                    debug_folder = self.results_folder / 'debug_masks' 
                    debug_folder.mkdir(exist_ok=True)
                    spacing = (1.0, 1.0, 1.0)
                    affine = np.diag([*spacing, 1.0])
                    
                    # Move tensors to CPU and convert to NumPy
                    ct_np = ct_synth.cpu().numpy()
                    mask_np = mask_s.cpu().numpy()  # CHANGED: mask_s
                    orig_ct_np = image_s.cpu().numpy()  # CHANGED: image_s
                    vqgan_recon_np = ct_vqgan_recon.cpu().numpy()
                    
                    for b in range(n_samples):  # CHANGED: no need to re-min() against batch_size
                        # Extract 3D volumes
                        ct_3d = ct_np[b, 0]
                        mask_3d = mask_np[b, 0]
                        orig_3d = orig_ct_np[b, 0]
                        vqgan_3d = vqgan_recon_np[b, 0]
                        
                        stem = f"step{self.step:04d}_b{b}"
                        
                        # Save Diffusion Output & Mask
                        nib.save(nib.Nifti1Image(ct_3d.astype(np.float32), affine), str(debug_folder / f"{stem}_cfg{cond_scale}_diffusion_ct.nii.gz"))
                        nib.save(nib.Nifti1Image(mask_3d.astype(np.uint8), affine), str(debug_folder / f"{stem}_mask.nii.gz"))
                        
                        # Save Original & VQGAN Reconstruction
                        nib.save(nib.Nifti1Image(orig_3d.astype(np.float32), affine), str(debug_folder / f"{stem}_original_ct.nii.gz"))
                        #nib.save(nib.Nifti1Image(vqgan_3d.astype(np.float32), affine), str(debug_folder / f"{stem}_vqgan_recon_ct.nii.gz"))
                        
                #self.ema_model.train() # Switch back to training mode
                print("--- Inference complete, resuming training ---\n")
            # =======================================================
            
            log_fn(log)
            self.step += 1

        print('training completed')

class Tester(object):
    def __init__(
        self,
        diffusion_model,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema_model = copy.deepcopy(self.model)
        self.step=0
        self.image_size = diffusion_model.image_size

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())


    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)