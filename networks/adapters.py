# Copyright 2023 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from utils_model import sandwich, pe_encode, pe_encode_float


class TextInputAdapter(nn.Module):
    """
    A module to convert sequences of text class tokens to embedding tokens with learned positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        output_size: int = 256,
        learn_pos_embedding: bool = False,
    ):
        super().__init__()
        self.learn_pos_embedding = learn_pos_embedding
        if learn_pos_embedding:
            self.pos_embedding = nn.Embedding(seq_len, output_size)
        else:
            self.register_buffer("pos_embedding", pe_encode(seq_len, output_size))
        self.inp_embedding = nn.Linear(vocab_size, output_size)
        self.t_embedding = nn.Linear(1, output_size)

    def forward(self, probs: torch.Tensor, t: torch.Tensor) -> Tensor:
        inp_emb = self.inp_embedding(2 * probs - 1)
        if self.learn_pos_embedding:
            pos_emb = self.pos_embedding(
                torch.arange(0, probs.size(1)).to(probs.device)
            )
        else:
            pos_emb = self.pos_embedding
        pos_emb = pos_emb.unsqueeze(0).expand(inp_emb.size(0), -1, -1)
        t_emb = self.t_embedding((2 * t - 1).unsqueeze(-1))
        output = inp_emb + pos_emb + t_emb

        return output


class FourierImageInputAdapter(nn.Module):
    """
    A module to convert 2D image coordinates into a set of vectors represented as a matrix, with fourier position codes.
    """

    def __init__(
        self,
        input_channels: int = 3,
        input_shape: Tuple[int, int] = (224, 224),
        n_freq_bands: int = 64,
        output_height: int = 256,
        value_res: int = -1,
        mask_res: int = -1,
        add_pos_feats: bool = True,
        add_mask: bool = True,
        learn_pos_feats: bool = False,
        pos_embed_size: int = 32,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.n_freq_bands = n_freq_bands
        self.value_res = value_res
        self.mask_res = mask_res
        self.add_pos_feats = add_pos_feats
        self.add_mask = add_mask
        if learn_pos_feats:
            pos_feats = nn.Parameter(
                init_scale
                * torch.randn(1, input_shape[0] * input_shape[1], pos_embed_size)
            )
            self.register_parameter("pos_feats", pos_feats)
        else:
            x = torch.linspace(-1.0, 1.0, steps=input_shape[0])
            y = torch.linspace(-1.0, 1.0, steps=input_shape[1])
            x_pos, y_pos = torch.meshgrid(x, y, indexing="ij")
            pos = torch.stack((x_pos, y_pos), dim=-1)
            pos = pos.reshape(-1, 2)
            x_bands = torch.linspace(1.0, input_shape[0] / 2, steps=n_freq_bands)
            y_bands = torch.linspace(1.0, input_shape[1] / 2, steps=n_freq_bands)
            bands = torch.stack((x_bands, y_bands), dim=0)
            vals = pos[:, :, None] * bands[None, :, :]
            vals = math.pi * vals.reshape(vals.shape[0], -1)
            pos_feats = torch.cat([vals.sin(), vals.cos()], dim=-1)
            pos_feats = torch.cat([pos_feats, pos], dim=-1)
            self.register_buffer("pos_feats", pos_feats)
        img_feat_height = input_channels
        pos_feat_height = pos_feats.size(-1)
        if self.mask_res > 0:
            mask_feat_height = (n_freq_bands * 2) + 1
        else:
            mask_feat_height = 1
        all_feat_height = img_feat_height
        if add_mask:
            all_feat_height += mask_feat_height
        if add_pos_feats:
            all_feat_height += pos_feat_height
        self.output_projection = None
        if output_height != all_feat_height:
            self.output_projection = nn.Linear(all_feat_height, output_height)

    def forward(self, img: Tensor, t: Tensor) -> Tensor:
        flat_img = sandwich(img)
        flat_t = sandwich(t)
        t_feats = (flat_t.float()[..., :1] * 2) - 1
        if self.mask_res > 0:
            t_feats = torch.cat(
                [
                    t_feats,
                    pe_encode_float(
                        t_feats, self.mask_res, self.n_freq_bands * 2
                    ).flatten(start_dim=2),
                ],
                -1,
            )
        fourier_feats = self.pos_feats.expand(img.size(0), -1, -1)
        all_feat_list = [flat_img]
        if self.add_mask:
            all_feat_list.append(t_feats)
        if self.add_pos_feats:
            all_feat_list.append(fourier_feats)
        all_feats = torch.cat(all_feat_list, dim=-1)
        if self.output_projection is None:
            output = all_feats
        else:
            output = self.output_projection(all_feats)
        return output


class OutputAdapter(nn.Module):
    def __init__(self, input_height: int, output_channels: int, output_height: int):
        super().__init__()
        self.output_channels = output_channels
        self.output_height = output_height
        self.output_projection = nn.Linear(
            input_height, output_channels * output_height
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        output = self.output_projection(inp)
        return output.reshape(
            output.size(0), -1, self.output_channels, self.output_height
        )
