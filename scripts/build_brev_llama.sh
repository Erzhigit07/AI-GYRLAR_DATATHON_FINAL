#!/usr/bin/env bash
# Build only public upstream code on the test VM; no competition files are used.
set -euo pipefail

task_root="${1:-$PWD/qwen-public-smoke}"
task_python="$PWD/qwen-smoke-venv/bin/python"
task_cmake="$PWD/qwen-smoke-venv/bin/cmake"
expected_commit="620d2225cc436ab0b001e3200e590e3b286ef9ab"

"$task_python" -m pip install --disable-pip-version-check cmake==3.31.6
if [[ ! -e "$task_root/source" ]]; then
  git clone --depth 1 --branch b11669 https://github.com/ggml-org/llama.cpp.git "$task_root/source"
fi
test "$(git -C "$task_root/source" rev-parse HEAD)" = "$expected_commit"
"$task_cmake" -S "$task_root/source" -B "$task_root/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
"$task_cmake" --build "$task_root/build" --target llama-server -j 8
sha256sum "$task_root/build/bin/llama-server"
