<img width="2791" height="417" alt="FlashPack logo" src="https://github.com/fal-ai/flashpack/blob/refs/heads/main/media/flashpack-logo-white.png?raw=true" />

# Introducing FlashPack: Lightning-Fast Model Loading for PyTorch

When using machine learning models in the real world, performance isn’t just about how fast your GPU can crunch numbers — it’s also about how quickly you can get your model there. Every second spent waiting on a checkpoint to load is a second your GPUs sit idle instead of working for you or your users.

That’s why we built FlashPack — a new, high-throughput file format and loading mechanism for PyTorch that makes model checkpoint I/O blazingly fast, even on systems without access to GPU Direct Storage (GDS).

With FlashPack, loading any model can be 3–6× faster than with the current state-of-the-art methods like `accelerate` or the standard `load_state_dict()` and `to()` flow — all wrapped in a lightweight, pure-Python package that works anywhere.

# The Current Landscape

If you’ve ever waited 30 seconds for a large model to load, you already know the pain of model I/O - most checkpoints, whether `.pt` or `.safetensors`, store each weight tensor as distinct objects in memory. When you load them, PyTorch has to read, deserialize, and move each tensor one by one. The process generally looks something like this:

1. CPU reads a chunk from disk.
2. Data is moved into RAM.
3. CPU sends it to GPU memory.
4. Repeat thousands of times in series.

It’s a stop-and-go pipeline — full of synchronization points and unnecessary overhead. We realized that if we treated all those weights not as individual files but instead as a single data stream, we could load them in one continuous motion. Although `.safetensors` improved upon this in numerous ways and reduced load times significantly, we can make things even faster - that's where FlashPack comes in.

# How It Works

FlashPack rethinks checkpoint loading from the ground up. It’s built on a few key observations:

1. Most models use the same data type throughout (e.g., float16 or bfloat16).
2. Tensor reshaping is an O(1) operation — it doesn’t require copying data.
3. CPU and GPU can work in parallel if given the right structure.

Here’s what happens under the hood:

## 1. Flatten everything into one block

FlashPack takes the model’s entire state_dict and flattens it into a single, contiguous stream of bytes.

At the end of the file, it stores a compact weight map that knows where every parameter and buffer lives — its key, shape, and offset. It’s like creating a single, perfectly indexed file instead of thousands of tiny ones.

## 2. Stream smartly with memory-mapped reads

When it’s time to load, FlashPack doesn’t do a slow read() loop. It memory-maps the file and divides it into a few mid-sized CPU buffers (≤64MB each). These buffers are loaded in a round-robin pattern, keeping disk reads continuous and efficient.

## 3. Overlap disk, CPU, and GPU with CUDA streams

Each CPU buffer is paired with a dedicated CUDA stream. As soon as one buffer is filled, it’s flushed asynchronously to the GPU — no waiting. While one stream writes to VRAM, another buffer is already being loaded from disk.

By the time we loop back, that first stream is done and ready for more. It’s a fully pipelined system — disk, CPU, and GPU all working in parallel.

## 4. Rebuild parameters and buffers

Once the data’s on the GPU, FlashPack reconstructs each tensor as a view into the flat memory block and creates new `nn.Parameter` and buffer instances with direct references to the exact bytes you loaded. There are no copies and no moves needed; just a new pointer and the model is immediately ready to run.

The Result: **2–6× faster checkpoint loading** compared to existing methods on all tested conditions and machines. Even without GDS or specialized hardware, it dramatically cuts startup and reload times for small and large models alike.

# Benchmarks

## `load_file() -> load_state_dict()` vs. `assign_from_file()`

We follow up from [HuggingFace's Safetensors Benchmark](https://huggingface.co/docs/safetensors/en/speed) and compare loading the state dictionary for [GPT2](https://huggingface.co/openai-community/gpt2).

<img alt="Comparison results" src="https://github.com/fal-ai/flashpack/blob/refs/heads/main/media/load-state-dict-comparison-white.png?raw=true" />

See the code for this benchmark in the publc repository: [test_speed_comparison.py](https://github.com/fal-ai/flashpack/blob/main/tests/test_speed_comparison.py)

## End-to-End Loading

This benchmark compares `from_pretrained` which instantiates, configures and loads the model and state using `accelerate` to `from_pretrained_flashpack` which does the same using FlashPack.

<img alt="Benchmark results" src="https://github.com/fal-ai/flashpack/blob/refs/heads/main/media/benchmark-white.png?raw=true" />

See the code for this benchmark in the publc repository: [run_benchmark.py](https://github.com/fal-ai/flashpack/blob/main/scripts/run_benchmark.py).

# Limitations

Although FlashPack can accelerate your model loading today, there are some limitations as of the current release:

1. All weights must be the same data type. This precludes some checkpoints from being able to be used, especially when using other accelerators like quantization.
2. FlashPack assumes all GPUs will get the same weights, and does not provide a means to provide a device map or mesh for loading different sections on different GPUs (e.g. using pipeline parallelism.)
3. State dictionary transformations are currently not possible, as the state dixctionary is never returned to the user. This means weights are specific to a particular implementation of a model. 

If your model requirements can fit within these limitations, FlashPack can be integrated and your weights converted in no time.

# Getting Started

You can install FlashPack right now from PyPI:

```
pip install flashpack
```

Or, if you prefer to pull straight from GitHub:

```
pip install git+https://github.com/fal-ai/flashpack
```

# Converting Your Models

Converting an existing checkpoint takes just one command:

```
flashpack convert
```

You can point it to:
1. A .pt or .safetensors checkpoint,
2. A diffusers or transformers model directory, or
3. A Hugging Face repository.

FlashPack will handle the rest, flattening your tensors and generating a .flashpack file ready for fast loading. If the model is a `diffusers` or `transformers` model, it will also save `config.json`.

Need more control? Run:

```
flashpack convert --help
```

for all available options.

# Integrating FlashPack

FlashPack was built to slide right into your existing workflow. You can integrate it in several ways:

1. As a mixin: Add a simple mixin to your nn.Module for automatic FlashPack support.
2. For Hugging Face models: Use our built-in diffusers and transformers mixins.
3. Standalone: Call `save_to_file` and `assign_from_file` directly on any `nn.Module`.

Everything is pure Python, dependency-light, and designed to “just work.”

You can explore the full repo, read the docs, or check out integration examples here at https://github.com/fal-ai/flashpack.
