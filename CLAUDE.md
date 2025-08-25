# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### Standard CPU Build
```bash
cmake -B build
cmake --build build --config Release -j $(nproc)
```

### GPU Backend Builds
For CUDA:
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)
```

For Metal (macOS):
```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j $(nproc)
```

For Vulkan:
```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j $(nproc)
```

For ROCm/HIP (AMD GPU):
```bash
cmake -B build -DGGML_HIP=ON
cmake --build build --config Release -j $(nproc)
```

### Debug Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Testing

### Run All Tests
```bash
ctest --test-dir build --output-on-failure -j $(nproc)
```

### Run Specific Test
```bash
ctest --test-dir build -R test-tokenizer --output-on-failure
```

### Server Tests (requires Python environment)
```bash
# First create and activate Python environment if not exists
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements-all.txt -r tools/server/tests/requirements.txt

# Run server tests
cd tools/server/tests
./tests.sh
```

## Code Formatting and Linting

### C++ Formatting
```bash
git clang-format
```

### Python Linting (if .venv exists)
```bash
source .venv/bin/activate
flake8
pyright
```

### Pre-commit Hooks
```bash
pre-commit run --all-files
```

## High-Level Architecture

### Core Components

**libllama** (`src/` and `include/llama.h`)
- Main inference library providing C API for LLM operations
- Key files: `src/llama.cpp` (core implementation), `include/llama.h` (public API)
- Architecture-specific optimizations in `src/llama-*.cpp` files
- Model loading, tokenization, KV cache, sampling, and inference

**ggml** (`ggml/`)
- Tensor computation library (vendored dependency)
- Provides low-level operations for various hardware backends
- Backend implementations in `ggml/src/ggml-*/`
- Core tensor operations in `ggml/src/ggml.c`

**Backend System**
- CPU: Optimized with AVX/NEON/AMX instructions (`ggml/src/ggml-cpu/`)
- CUDA: NVIDIA GPU support (`ggml/src/ggml-cuda/`)
- Metal: Apple Silicon GPU (`ggml/src/ggml-metal/`)
- Vulkan: Cross-platform GPU (`ggml/src/ggml-vulkan/`)
- SYCL: Intel GPU support (`ggml/src/ggml-sycl/`)
- HIP: AMD GPU support (`ggml/src/ggml-hip/`)

### Key Executables

**Primary Tools** (in `build/bin/` after building):
- `llama-cli`: Main inference CLI for interactive chat and text completion
- `llama-server`: OpenAI-compatible HTTP API server
- `llama-quantize`: Convert models to quantized formats (Q4_0, Q5_K_M, etc.)
- `llama-perplexity`: Evaluate model quality metrics
- `llama-bench`: Performance benchmarking tool

**Conversion Tools**:
- `convert_hf_to_gguf.py`: Convert Hugging Face models to GGUF format
- `convert_lora_to_gguf.py`: Convert LoRA adapters

### Model Format

**GGUF** (GGML Universal Format)
- Custom binary format for efficient model storage and loading
- Supports various quantization levels (1.5-8 bits)
- Metadata storage for model architecture and parameters
- Implementation in `gguf-py/` Python package

### Project Structure

```
llama.cpp/
├── src/                 # Core library implementation
├── include/             # Public API headers
├── ggml/               # Tensor computation library
├── common/             # Shared utilities for examples
├── examples/           # Example applications
├── tools/              # Development and utility tools
│   ├── server/         # HTTP API server
│   ├── quantize/       # Quantization tool
│   └── perplexity/     # Model evaluation
├── tests/              # Test suite
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## Common Development Tasks

### Adding Model Support
1. Update `src/llama-model.cpp` for model architecture
2. Add tokenizer support in `src/llama-vocab.cpp`
3. Update conversion script `convert_hf_to_gguf.py`
4. Add test cases in `tests/`

### Modifying Server API
1. Server implementation: `tools/server/server.cpp`
2. Update OpenAPI compatibility as needed
3. Add tests in `tools/server/tests/`
4. Update documentation in `tools/server/README.md`

### Performance Optimization
1. Profile with `llama-bench` for baseline
2. Implement optimization in appropriate backend (`ggml/src/ggml-*/`)
3. Validate with `test-backend-ops`
4. Benchmark improvements with `llama-bench`

## Important Notes

- The Makefile is deprecated - always use CMake for building
- Built binaries are placed in `build/bin/`
- Use ccache if available for faster incremental builds
- Network-dependent tests may fail in isolated environments (expected)
- Always format C++ code with `git clang-format` before committing
- The project prioritizes minimal dependencies and cross-platform compatibility