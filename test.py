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
from typing import Optional, Tuple

import torch
from omegaconf import OmegaConf, DictConfig
from rich import print
from torch import nn
from torch.utils.data import DataLoader

from data import make_datasets
from model import BFN
from utils_train import seed_everything, make_config, make_bfn, worker_init_function, get_generator, make_progress_bar

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def setup(seed, cfg: DictConfig) -> Tuple[nn.Module, DataLoader]:
    test_ds = make_datasets(cfg.data)[-1]
    test_dl = DataLoader(
        dataset=test_ds,
        worker_init_fn=worker_init_function,
        generator=get_generator(seed),
        batch_size=100,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    model = make_bfn(cfg.model)
    return model, test_dl


@torch.inference_mode()
def test(model: BFN, dataloader: DataLoader, n_steps: int, n_repeats: int) -> tuple[float, float, float, float]:
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    losses, recon_losses = [], []
    pbar = make_progress_bar(True, "[red]loss: {task.fields[loss]:.4f} repeat: {task.fields[r]}")
    with pbar:
        task_id = pbar.add_task("Test", visible=True, total=n_repeats * len(dataloader), loss=math.nan, r=0)
        for r in range(n_repeats):
            _losses, _recon_losses = [], []
            for eval_batch in dataloader:
                eval_batch = eval_batch.to("cuda") if torch.cuda.is_available() else eval_batch
                loss = model(eval_batch, n_steps=n_steps).item()
                recon_loss = model.compute_reconstruction_loss(eval_batch).item()
                _losses.append(loss)
                _recon_losses.append(recon_loss)
                pbar.update(task_id, advance=1, loss=torch.tensor(_losses).mean() + torch.tensor(_recon_losses).mean(), r=r+1)
            losses.append(torch.tensor(_losses).mean())
            recon_losses.append(torch.tensor(_recon_losses).mean())
    losses = torch.stack(losses)
    loss_mean, loss_err = losses.mean(), losses.std(correction=0).item() / math.sqrt(len(losses))
    recon_losses = torch.stack(recon_losses)
    recon_mean, recon_err = recon_losses.mean(), recon_losses.std(correction=0).item() / math.sqrt(len(recon_losses))
    return loss_mean, loss_err, recon_mean, recon_err


def main(cfg: DictConfig) -> tuple[float, float, float, float]:
    """
    Config entries:
        seed (int): Optional
        config_file (str): Name of config file containing model and data config for a saved checkpoint
        load_model (str): Path to a saved checkpoint to be tested
        n_steps (int): Number of Bayesian flow steps. Set to None for continuous time Bayesian flow loss.
        n_repeats (int): Number of times to iterate through the dataset.
    """
    seed_everything(cfg.seed)
    print(f"Seeded everything with seed {cfg.seed}")

    # Get model and data config from the training config file
    train_cfg = make_config(cfg.config_file)
    model, dataloader = setup(cfg.seed, train_cfg)

    model.load_state_dict(torch.load(cfg.load_model, weights_only=True, map_location="cpu"))
    loss_mean, loss_err, recon_mean, recon_err = test(model, dataloader, cfg.n_steps, cfg.n_repeats)
    print(f"For {cfg.n_steps} steps with {cfg.n_repeats} repeats:")
    print(f"Loss is {loss_mean:.6f} +- {loss_err:.6f}")
    print(f"Reconstruction Loss is {recon_mean:.6f} +- {recon_err:.6f}")
    print(f"Total loss mean = {loss_mean + recon_mean}")
    return loss_mean, loss_err, recon_mean, recon_err


if __name__ == "__main__":
    main(OmegaConf.from_cli())
