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

import torch
from omegaconf import OmegaConf, DictConfig

from utils_train import seed_everything, make_config, make_bfn

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def main(cfg: DictConfig) -> torch.Tensor:
    """
    Config entries:
        seed (int): Optional
        config_file (str): Name of config file containing model and data config for a saved checkpoint
        load_model (str): Path to a saved checkpoint to be tested
        sample_shape (list): Shape of sample batch, e.g.:
            (3, 256) for sampling 3 sequences of length 256 from the text8 model.
            (2, 32, 32, 3) for sampling 2 images from the CIFAR10 model.
            (4, 28, 28, 1) for sampling 4 images from the MNIST model.
        n_steps (int): Number of sampling steps (positive integer).
        save_file (str): File path to save the generated sample tensor. Skip saving if None.
    """
    seed_everything(cfg.seed)
    print(f"Seeded everything with seed {cfg.seed}")

    # Get model config from the training config file
    train_cfg = make_config(cfg.config_file)
    bfn = make_bfn(train_cfg.model)

    bfn.load_state_dict(torch.load(cfg.load_model, weights_only=True, map_location="cpu"))
    if torch.cuda.is_available():
        bfn.to("cuda")
    samples = bfn.sample(cfg.samples_shape, cfg.n_steps)

    if cfg.save_file is not None:
        torch.save(samples.to("cpu"), cfg.save_file)

    return samples


if __name__ == "__main__":
    main(OmegaConf.from_cli())
