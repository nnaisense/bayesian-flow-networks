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

import json
import math
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional, Generator, Union

try:
    import neptune
    from neptune.utils import stringify_unsupported
except ImportError:
    neptune = None

    def stringify_unsupported(x):
        return x


import numpy as np
import torch
from accelerate.logging import get_logger
from omegaconf import OmegaConf, DictConfig
from rich.progress import Progress, SpinnerColumn, MofNCompleteColumn, TimeElapsedColumn, TextColumn
from torch.utils.data import DataLoader

import model
import networks
import probability
from data import make_datasets
from networks import adapters

logger = get_logger(__name__)


def seed_everything(seed: Optional[int]):
    assert seed is not None
    seed += torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_function(worker_id: int) -> None:
    """https://pytorch.org/docs/stable/notes/randomness.html#dataloader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_checkpointing(checkpoint_dir: Union[str, Path, None], run_id: str) -> Optional[Path]:
    if checkpoint_dir is None:
        return None
    checkpoint_dir = Path(checkpoint_dir) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_dir = checkpoint_dir / "last"
    last_dir.mkdir(parents=True, exist_ok=True)
    best_dir = checkpoint_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def checkpoint_training_state(checkpoint_dir, accelerator, ema_model, step: int, run_id: str):
    if checkpoint_dir is None:
        return
    logger.info(f"Checkpointing training state to {checkpoint_dir} at step {step}")
    accelerator.save_state(checkpoint_dir)
    with open(checkpoint_dir / "info.json", "w") as f:
        json.dump({"step": step, "run_id": run_id}, f)
    if ema_model is not None:
        ema_checkpoint_path = checkpoint_dir / "ema_model.pt"
        torch.save(ema_model.state_dict(), ema_checkpoint_path)


def log(key_handler, value, step, cond=True):
    """Log series to neptune only if cond is True. Helps with distributed training and conditional logging."""
    if not isinstance(key_handler, defaultdict) and cond and math.isfinite(value):
        key_handler.log(value, step=step)


def log_cfg(cfg, run: "neptune.Run"):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_temp_filename: Path = Path(tmpdir) / "cfg.yaml"
        cfg_temp_filename.write_text(OmegaConf.to_yaml(cfg, resolve=True))
        run["cfg"].upload(str(cfg_temp_filename), wait=True)
    run["hyperparameters"] = stringify_unsupported(OmegaConf.to_container(cfg, resolve=True))


@torch.no_grad()
def update_ema(ema_model, model, ema_decay):
    if ema_model is not None and ema_decay > 0:
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.sub_((1 - ema_decay) * (ema_param - model_param))


def ddict():
    """Infinite default dict to fake neptune run on non-main processes"""
    return defaultdict(ddict)


def make_infinite(dataloader: DataLoader) -> Generator[dict, None, None]:
    while True:
        for data in dataloader:
            yield data


def make_progress_bar(is_main: bool, text="[red]loss: {task.fields[loss]:.3f}"):
    return Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        TextColumn(text),
        disable=not is_main,
    )


def make_dataloaders(cfg: DictConfig):
    train_set, val_set, _ = make_datasets(cfg.data)
    dataloaders = {
        "train": DataLoader(
            dataset=train_set,
            worker_init_fn=worker_init_function,
            **cfg.train_loader,
        ),
        "val": DataLoader(
            dataset=val_set,
            worker_init_fn=worker_init_function,
            **cfg.val_loader,
        ),
    }
    return dataloaders


def make_from_cfg(module, cfg, **parameters):
    return getattr(module, cfg.class_name)(**cfg.parameters, **parameters) if cfg is not None else None


def make_bfn(cfg: DictConfig):
    data_adapters = {
        "input_adapter": make_from_cfg(adapters, cfg.input_adapter),
        "output_adapter": make_from_cfg(adapters, cfg.output_adapter),
    }
    net = make_from_cfg(networks, cfg.net, data_adapters=data_adapters)
    bayesian_flow = make_from_cfg(model, cfg.bayesian_flow)
    distribution_factory = make_from_cfg(probability, cfg.distribution_factory)
    loss = make_from_cfg(model, cfg.loss, bayesian_flow=bayesian_flow, distribution_factory=distribution_factory)
    bfn = model.BFN(net=net, bayesian_flow=bayesian_flow, loss=loss)
    return bfn


default_train_config = {
    "meta": {
        "neptune": None,
        "debug": False,
        "root_dir": ".",
    },
    "data": {
        "dataset": "",
        "data_dir": "./data",
    },
    "train_loader": {
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": False,
    },
    "training": {
        "accumulate": 1,
        "checkpoint_dir": "./checkpoints",
        "checkpoint_interval": None,
        "ema_decay": -1,
        "grad_clip_norm": -1,
        "log_interval": 50,
        "max_val_batches": -1,
        "seed": 666,
        "start_step": 1,
        "val_repeats": 1,
    },
}


def make_config(cfg_file: str):
    cli_conf = OmegaConf.load(cfg_file)
    # Start with default config
    cfg = OmegaConf.create(default_train_config)
    # Merge into default config
    cfg = OmegaConf.merge(cfg, cli_conf)
    return cfg
