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
import os
import pathlib
import pickle
import zipfile
from typing import Union

import numpy as np
import requests
import torch
import torchvision
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.utils import make_grid

from utils_model import quantize

TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz")


def bin_mnist_transform(x):
    return torch.bernoulli(x.permute(1, 2, 0).contiguous()).int()


def bin_mnist_cts_transform(x):
    return torch.bernoulli(x.permute(1, 2, 0).contiguous()) - 0.5


def rgb_image_transform(x, num_bins=256):
    return quantize((x * 2) - 1, num_bins).permute(1, 2, 0).contiguous()


class MyLambda(torchvision.transforms.Lambda):
    def __init__(self, lambd, arg1):
        super().__init__(lambd)
        self.arg1 = arg1

    def __call__(self, x):
        return self.lambd(x, self.arg1)


class CIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]


class MNIST(torchvision.datasets.MNIST):
    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]


def make_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset, Dataset]:
    """
    Mandatory keys: dataset (must be cifar10, mnist, bin_mnist, bin_mnist_cts or text8), data_dir
    Optional for vision: num_bins (default 256), val_frac (default 0.01), horizontal_flip (default: False)
    Mandatory for text: seq_len
    """
    num_bins = cfg.get("num_bins", 256)
    if cfg.dataset == "cifar10":
        train_transform_list = [transforms.ToTensor()]
        if cfg.get("horizontal_flip", False):
            train_transform_list.append(transforms.RandomHorizontalFlip())
        train_transform_list.append(MyLambda(rgb_image_transform, num_bins))
        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose([transforms.ToTensor(), MyLambda(rgb_image_transform, num_bins)])
        train_set = CIFAR10(root=cfg.data_dir, train=True, download=True, transform=train_transform)
        val_set = CIFAR10(root=cfg.data_dir, train=True, download=True, transform=test_transform)
        test_set = CIFAR10(root=cfg.data_dir, train=False, download=True, transform=test_transform)

    elif cfg.dataset == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                MyLambda(rgb_image_transform, num_bins),
            ]
        )
        train_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        val_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        test_set = MNIST(root=cfg.data_dir, train=False, download=True, transform=transform)

    elif cfg.dataset == "bin_mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(bin_mnist_transform)])
        train_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        val_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        test_set = MNIST(root=cfg.data_dir, train=False, download=True, transform=transform)

    elif cfg.dataset == "bin_mnist_cts":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(bin_mnist_cts_transform)])
        train_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        val_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        test_set = MNIST(root=cfg.data_dir, train=False, download=True, transform=transform)

    elif cfg.dataset == "text8":
        train_set = Text8Dataset(cfg.data_dir, "train", download=True, seq_len=cfg.seq_len)
        val_set = Text8Dataset(cfg.data_dir, "val", download=True, seq_len=cfg.seq_len)
        test_set = Text8Dataset(cfg.data_dir, "test", download=True, seq_len=cfg.seq_len)
    else:
        raise NotImplementedError(cfg.dataset)

    if cfg.dataset != "text8":
        # For vision datasets we split the train set into train and val
        val_frac = cfg.get("val_frac", 0.01)
        train_val_split = [1.0 - val_frac, val_frac]
        seed = 2147483647
        train_set = random_split(train_set, train_val_split, generator=torch.Generator().manual_seed(seed))[0]
        val_set = random_split(val_set, train_val_split, generator=torch.Generator().manual_seed(seed))[1]

    return train_set, val_set, test_set


def prepare_text8(data_dir: pathlib.Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    data_url = "http://mattmahoney.net/dc/text8.zip"
    with open(data_dir / "text8.zip", "wb") as f:
        print("Downloading text8")
        f.write(requests.get(data_url).content)
        print("Done")
    with zipfile.ZipFile(data_dir / "text8.zip") as f:
        f.extractall(data_dir)
    os.remove(data_dir / "text8.zip")
    data = (data_dir / "text8").read_text()

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    # encode both to integers
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) : int(n * 0.95)]
    test_data = data[int(n * 0.95) :]
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    test_ids = encode(test_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    print(f"test has {len(test_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    test_ids = np.array(test_ids, dtype=np.uint16)
    train_ids.tofile(data_dir / "train.bin")
    val_ids.tofile(data_dir / "val.bin")
    test_ids.tofile(data_dir / "test.bin")
    print(f"Saved to {data_dir / 'train.bin'}, {data_dir / 'val.bin'}, {data_dir / 'test.bin'}")

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(data_dir / "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"text8 dataset downloaded and prepared in dir {data_dir}")


class Text8Dataset(Dataset):
    def __init__(self, data_dir: Union[str, pathlib.Path], split: str, download: bool, seq_len: int):
        """
        seq_len should include context length. Example: seq_len=512 for modeling 256 chars with 256 char of context.
        context is only used for correct preparation of val/test sets.
        """
        self.root_dir = pathlib.Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        fname = {"train": "train.bin", "val": "val.bin", "test": "test.bin"}[self.split]
        assert self.split in ["train", "val", "test"]
        data_dir = self.root_dir / "text8"
        if not os.path.exists(data_dir):
            if download:
                prepare_text8(data_dir)
            else:
                raise NotADirectoryError(f"dir {data_dir} does not exist and download is False")
        self.data = np.memmap(data_dir / fname, np.uint16, "r")

    def __getitem__(self, index) -> torch.Tensor:
        seq = torch.from_numpy(self.data[index : index + self.seq_len].astype(np.int64))
        return seq

    def __len__(self):
        return self.data.size - self.seq_len


def char_ids_to_str(char_ids: Union[list[int], np.array, torch.Tensor]) -> str:
    """Decode a 1D sequence of character IDs to a string."""
    return "".join([TEXT8_CHARS[i] for i in char_ids])


def batch_to_str(text_batch: Union[list[list], np.array, torch.Tensor]) -> list[str]:
    """Decode a batch of character IDs to a list of strings."""
    return [char_ids_to_str(row_char_ids) for row_char_ids in text_batch]


def batch_to_images(image_batch: torch.Tensor, ncols: int = None) -> plt.Figure:
    if ncols is None:
        ncols = math.ceil(math.sqrt(len(image_batch)))
    if image_batch.size(-1) == 3:  # for color images (CIFAR-10)
        image_batch = (image_batch + 1) / 2
    grid = make_grid(image_batch.permute(0, 3, 1, 2), ncols, pad_value=1).permute(1, 2, 0)
    fig = plt.figure(figsize=(grid.size(1) / 30, grid.size(0) / 30))
    plt.imshow(grid.cpu().clip(min=0, max=1), interpolation="nearest")
    plt.grid(False)
    plt.axis("off")
    return fig
