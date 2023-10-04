# Source: https://github.com/addtt/variational-diffusion-models
#
# MIT License
#
# Copyright (c) 2022 Andrea Dittadi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modifications:
# - Added data_adapters to UNetVDM to preprocess the inputs and postprocess the outputs
# - Replaced `timesteps` argument of `UNetModel.forward()` with time `t`, which is used to compute the `timesteps`
# - Added 1/1000 to t before computing timesteps embeddings so t isn't 0
# - Added concatenation of input and output of the network before the final projection

import numpy as np
import torch
from torch import einsum, nn, pi, softmax

from utils_model import sandwich


@torch.no_grad()
def zero_init(module: nn.Module) -> nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        nn.init.zeros_(p.data)
    return module


class UNetVDM(nn.Module):
    def __init__(
        self,
        data_adapters,
        embedding_dim: int = 128,
        n_blocks: int = 32,
        n_attention_heads: int = 1,
        dropout_prob: float = 0.1,
        norm_groups: int = 32,
        input_channels: int = 3,
        use_fourier_features: bool = True,
        attention_everywhere: bool = False,
        image_size: int = 32,
    ):
        super().__init__()
        self.input_adapter = data_adapters["input_adapter"]
        self.output_adapter = data_adapters["output_adapter"]
        attention_params = dict(
            n_heads=n_attention_heads,
            n_channels=embedding_dim,
            norm_groups=norm_groups,
        )
        resnet_params = dict(
            ch_in=embedding_dim,
            ch_out=embedding_dim,
            condition_dim=4 * embedding_dim,
            dropout_prob=dropout_prob,
            norm_groups=norm_groups,
        )
        if use_fourier_features:
            self.fourier_features = FourierFeatures()
        self.embed_conditioning = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 4),
            nn.SiLU(),
        )
        total_input_ch = input_channels
        if use_fourier_features:
            total_input_ch *= 1 + self.fourier_features.num_features
        self.conv_in = nn.Conv2d(total_input_ch, embedding_dim, 3, padding=1)

        # Down path: n_blocks blocks with a resnet block and maybe attention.
        self.down_blocks = nn.ModuleList(
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_params),
                attention_block=AttentionBlock(**attention_params) if attention_everywhere else None,
            )
            for _ in range(n_blocks)
        )

        self.mid_resnet_block_1 = ResnetBlock(**resnet_params)
        self.mid_attn_block = AttentionBlock(**attention_params)
        self.mid_resnet_block_2 = ResnetBlock(**resnet_params)

        # Up path: n_blocks+1 blocks with a resnet block and maybe attention.
        resnet_params["ch_in"] *= 2  # double input channels due to skip connections
        self.up_blocks = nn.ModuleList(
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_params),
                attention_block=AttentionBlock(**attention_params) if attention_everywhere else None,
            )
            for _ in range(n_blocks + 1)
        )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=embedding_dim),
            nn.SiLU(),
            zero_init(nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1)),
        )
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        self.image_size = image_size
        self.use_fourier_features = use_fourier_features

    def forward(
        self,
        data: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        flat_x = self.input_adapter(data, t)
        x = flat_x.reshape(flat_x.size(0), self.image_size, self.image_size, self.input_channels)
        x_perm = x.permute(0, 3, 1, 2).contiguous()
        t = t.float().flatten(start_dim=1)[:, 0]
        t_embedding = get_timestep_embedding(t + 0.001, self.embedding_dim)
        # We will condition on time embedding.
        cond = self.embed_conditioning(t_embedding)

        h = self.maybe_concat_fourier(x_perm)
        h = self.conv_in(h)  # (B, embedding_dim, H, W)
        hs = []
        for down_block in self.down_blocks:  # n_blocks times
            hs.append(h)
            h = down_block(h, cond)
        hs.append(h)
        h = self.mid_resnet_block_1(h, cond)
        h = self.mid_attn_block(h)
        h = self.mid_resnet_block_2(h, cond)
        for up_block in self.up_blocks:  # n_blocks+1 times
            h = torch.cat([h, hs.pop()], dim=1)
            h = up_block(h, cond)
        out = sandwich(self.conv_out(h).permute(0, 2, 3, 1).contiguous())
        out = torch.cat([sandwich(x), out], -1)
        out = self.output_adapter(out)
        return out

    def maybe_concat_fourier(self, z):
        if self.use_fourier_features:
            return torch.cat([z, self.fourier_features(z)], dim=1)
        return z


class ResnetBlock(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out=None,
        condition_dim=None,
        dropout_prob=0.0,
        norm_groups=32,
    ):
        super().__init__()
        ch_out = ch_in if ch_out is None else ch_out
        self.ch_out = ch_out
        self.condition_dim = condition_dim
        self.net1 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
        )
        if condition_dim is not None:
            self.cond_proj = zero_init(nn.Linear(condition_dim, ch_out, bias=False))
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out),
            nn.SiLU(),
            nn.Dropout(dropout_prob),
            zero_init(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)),
        )
        if ch_in != ch_out:
            self.skip_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x, condition):
        h = self.net1(x)
        if condition is not None:
            assert condition.shape == (x.shape[0], self.condition_dim)
            condition = self.cond_proj(condition)
            condition = condition[:, :, None, None]
            h = h + condition
        h = self.net2(h)
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)
        assert x.shape == h.shape
        return x + h


def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    timesteps *= 1000.0  # In DDPM the time step is in [0, 1000], here [0, 1]
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)


class FourierFeatures(nn.Module):
    def __init__(self, first=5.0, last=6.0, step=1.0):
        super().__init__()
        self.freqs_exponent = torch.arange(first, last + 1e-8, step)

    @property
    def num_features(self):
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        assert len(x.shape) >= 2

        # Compute (2pi * 2^n) for n in freqs.
        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)  # (F, )
        freqs = 2.0**freqs_exponent * 2 * pi  # (F, )
        freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, ...)

        # Compute (2pi * 2^n * x) for n in freqs.
        features = freqs * x.unsqueeze(1)  # (B, F, X1, X2, ...)
        features = features.flatten(1, 2)  # (B, F * C, X1, X2, ...)

        # Output features are cos and sin of above. Shape (B, 2 * F * C, H, W).
        return torch.cat([features.sin(), features.cos()], dim=1)


def attention_inner_heads(qkv, num_heads):
    """Computes attention with heads inside of qkv in the channel dimension.

    Args:
        qkv: Tensor of shape (B, 3*H*C, T) with Qs, Ks, and Vs, where:
            H = number of heads,
            C = number of channels per head.
        num_heads: number of heads.

    Returns:
        Attention output of shape (B, H*C, T).
    """

    bs, width, length = qkv.shape
    ch = width // (3 * num_heads)

    # Split into (q, k, v) of shape (B, H*C, T).
    q, k, v = qkv.chunk(3, dim=1)

    # Rescale q and k. This makes them contiguous in memory.
    scale = ch ** (-1 / 4)  # scale with 4th root = scaling output by sqrt
    q = q * scale
    k = k * scale

    # Reshape qkv to (B*H, C, T).
    new_shape = (bs * num_heads, ch, length)
    q = q.view(*new_shape)
    k = k.view(*new_shape)
    v = v.reshape(*new_shape)

    # Compute attention.
    weight = einsum("bct,bcs->bts", q, k)  # (B*H, T, T)
    weight = softmax(weight.float(), dim=-1).to(weight.dtype)  # (B*H, T, T)
    out = einsum("bts,bcs->bct", weight, v)  # (B*H, C, T)
    return out.reshape(bs, num_heads * ch, length)  # (B, H*C, T)


class Attention(nn.Module):
    """Based on https://github.com/openai/guided-diffusion."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        assert qkv.dim() >= 3, qkv.dim()
        assert qkv.shape[1] % (3 * self.n_heads) == 0
        spatial_dims = qkv.shape[2:]
        qkv = qkv.view(*qkv.shape[:2], -1)  # (B, 3*H*C, T)
        out = attention_inner_heads(qkv, self.n_heads)  # (B, H*C, T)
        return out.view(*out.shape[:2], *spatial_dims).contiguous()


class AttentionBlock(nn.Module):
    """Self-attention residual block."""

    def __init__(self, n_heads, n_channels, norm_groups):
        super().__init__()
        assert n_channels % n_heads == 0
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=n_channels),
            nn.Conv2d(n_channels, 3 * n_channels, kernel_size=1),  # (B, 3 * C, H, W)
            Attention(n_heads),
            zero_init(nn.Conv2d(n_channels, n_channels, kernel_size=1)),
        )

    def forward(self, x):
        return self.layers(x) + x


class UpDownBlock(nn.Module):
    def __init__(self, resnet_block, attention_block=None):
        super().__init__()
        self.resnet_block = resnet_block
        self.attention_block = attention_block

    def forward(self, x, cond):
        x = self.resnet_block(x, cond)
        if self.attention_block is not None:
            x = self.attention_block(x)
        return x
