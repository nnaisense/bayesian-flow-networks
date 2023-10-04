# Bayesian Flow Networks

This is the official code release for [Bayesian Flow Networks](https://arxiv.org/abs/2308.07037) by Alex Graves, Rupesh Kumar Srivastava, Timothy Atkinson and Faustino Gomez.

<img src="bfn.gif" alt="Overview of BFN process" style="width:600px;"/>

## Reading Guide

- `model.py` contains all the main contributions of the paper. These include definitions, for both continuous and discrete data, of Bayesian Flows as well as loss functions for both continuous-time and discrete-time. See comments in the base classes in that file for details.
- `probability.py` defines the probability distributions used by the models.
- `train.py`, `test.py` and `sample.py` are scripts for training, testing and sampling (see below for usage).
- `data.py` contains utilities related to data loading and processing.
- `networks/` contains implementations of the network architectures used by the models. 

## Setup

```shell
# Create a new conda env with all dependencies including pytorch and CUDA
conda env create -f env.yml
conda activate bfn

# Or, install additional dependencies into an existing pytorch env
pip install accelerate==0.19.0 matplotlib omegaconf rich

# Optional, if you want to enable logging to neptune.ai
pip install neptune 
```

## Training

The models in the paper can be trained using the configs provided in the `configs` dir as follows:

```shell
# mnist experiment on 1 GPU
accelerate launch train.py config_file=configs/mnist_discrete.yaml
# cifar10 experiment on 1 GPU (A100)
accelerate launch train.py config_file=configs/cifar10_discretized_256bins.yaml
# text8 experiment on 8 GPUs (A100)
accelerate launch --multi_gpu --num_processes=8 --num_machines=1 --dynamo_backend=no --mixed_precision=fp16 train.py config_file=configs/text8_discrete.yaml 
```

## Testing
> [!NOTE]
> Depending on your GPU, you may wish to adjust the batch size used for testing in `test.py`.
```shell
# Optional: Download pretrained checkpoints (make sure you have git-lfs installed: https://git-lfs.com/)
git clone git@hf.co:rupspace/pretrained-BFNs
# Compute 784-step loss on MNIST
python test.py seed=1 config_file=./configs/mnist_discrete.yaml load_model=./pretrained-BFNs/mnist_ema.pt n_steps=784 n_repeats=2000
# Compute 10-step loss on CIFAR-10
python test.py seed=1 config_file=./configs/cifar10_discretized_256bins.yaml load_model=./pretrained-BFNs/cifar10_256d_ema.pt n_steps=10 n_repeats=100
# Compute continuous-time loss on text8
python test.py seed=1 config_file=./configs/text8_discrete.yaml load_model=./pretrained-BFNs/text8_ema.pt n_steps=0 n_repeats=1
```
> [!IMPORTANT]
> All computed results will be in nats-per-data-dimension. To convert to bits, divide by ln(2).

## Sampling

You can sample from a pre-trained model as follows (change options as desired):

```shell
# Sample 4 binarized MNIST images using 100 steps
python sample.py seed=1 config_file=./configs/mnist_discrete.yaml load_model=./pretrained-BFNs/mnist_ema.pt samples_shape="[4, 28, 28, 1]" n_steps=100 save_file=./samples_mnist.pt
# Sample 4 CIFAR-10 16-bit images modeled as discretized data using 1000 steps
python sample.py seed=1 config_file=./configs/cifar10_discretized_16bins.yaml load_model=./pretrained-BFNs/cifar10_16d_ema.pt samples_shape="[4, 32, 32, 3]" n_steps=1000 save_file=./samples_cifar.pt
# Sample 2 text8 sequences of length 256 using 100 steps
python sample.py seed=1 config_file=./configs/text8_discrete.yaml load_model=./pretrained-BFNs/text8_ema.pt samples_shape="[2, 256]" n_steps=100 save_file=./samples_text8.pt
```

The samples are stored as PyTorch tensors in the `save_file`, and can be visualized by loading them and then using the utilities `batch_to_images` and `batch_to_str` in `data.py`.
For example: 
```shell
# batch_to_images returns a matplotlib Figure object
python -c "import torch; from data import batch_to_images; batch_to_images(torch.load('./samples_mnist.pt')).savefig('mnist.png')"
python -c "import torch; from data import batch_to_images; batch_to_images(torch.load('./samples_cifar.pt')).savefig('cifar.png')"
# batch_to_str returns a list of str
python -c "import torch; from data import batch_to_str; print(batch_to_str(torch.load('./samples_text8.pt')))"
```

## Reproducibility 

If a high degree of reproducibility is desired (e.g. during sampling), set the following:

```python
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

## Acknowledgements

We are grateful to [@Higgcz](https://github.com/Higgcz) for generous support with the experiment infrastructure and code release.
