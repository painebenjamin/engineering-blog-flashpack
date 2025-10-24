<img width="2791" height="417" alt="image" src="https://github.com/user-attachments/assets/95de0980-36f3-49e3-9469-2346c9fd7983" />

# Introducing FlashPack: High-Throughput Tensor Loading for PyTorch

We introduce FlashPack, a new format and loading mechanism for PyTorch that significantly accelerates model checkpoint I/O in environments where GPU Direct Storage (GDS) is not available.

FlashPack achieves a 3–6× improvement in load times compared to current state-of-the-art tools such as Accelerate and the standard `load_state_dict()`.

# How It Works

Conventional PyTorch checkpoints, including `.pt`, `.bin`, `.ckpt` and `.safetensors` formats, store each tensor as an independent object. This leads to fragmented reads, frequent synchronization, and inefficient CPU–GPU data transfer.

FlashPack takes a different approach:

- Single contiguous block: The entire `state_dict` is flattened into one continuous block of data.
- Lightweight memory map: A compact map records each tensor’s key, shape, and offset.
- Pinned memory with rotating CUDA streams: The raw data is pushed to the GPU from a single pinned memory map using multiple rotating CUDA streams to maximize throughput.
- One synchronization step: All streams are synchronized exactly once, after all transfers complete, so data movement is kept parallelized as long as possible.
- Reconstruction: We then iterate over the map and reassign parameter and buffer storage to reproduce the original state dictionary exactly.

# Benchmark

<img width="4163" height="3563" alt="image" src="https://github.com/user-attachments/assets/4ac4a2ed-33e8-4214-9123-336be0d54487" />

# Installation

FlashPack is available now on PyPI:

```
pip install flashpack
```

It can also be installed directly from source like so:

```
pip install git+https://github.com/fal-ai/flashpack
```

# Conversion

Converting checkpoints to FlashPack format is as easy as installing the Python package and running `flashpack convert` in your CLI. This will accept:
1. The path to a torch or safetensors checkpoint, or
2. The path to a `diffusers` or `transformers` model directory, or
3. A HuggingFace repository of the same.

Run `flashpack convert --help` for more information on the `convert` command, and `flashpack --help` for information on all available commands.

# Integration

FlashPack provides several different means of integrating into your current pipeline to make it as frictionless as possible. This includes:
1. A mixin class you can add to a standard `nn.Module`
2. Mixin classes for `diffusers` and `transformers` models
3. Single-function `load`/`save` methods for all other cases.

See the public repository at https://github.com/fal/flashpack for more documentation and integrations guides.
