# Compiler setup

Use [TVM 0.23.0](https://github.com/apache/tvm) at the revision below, with its matching FFI submodule on
host and target. Install CMake, Ninja, a C++ compiler, and LLVM 18 with AArch64
support. Put LLVM's `bin` directory on `PATH`.

Run from the EdgeKernelBench repository root:

```bash
export EDGEKERNELBENCH_ROOT="$(pwd)"
export TVM_ROOT=/path/to/tvm
git clone --recursive https://github.com/apache/tvm.git "$TVM_ROOT"
git -C "$TVM_ROOT" checkout 75ce1b4d46053b28fa926a6e29aa715b2463dc92
git -C "$TVM_ROOT" submodule update --init --recursive
git -C "$TVM_ROOT" apply --check "$EDGEKERNELBENCH_ROOT/patches/tvm-worker-errors.patch"
git -C "$TVM_ROOT" apply "$EDGEKERNELBENCH_ROOT/patches/tvm-worker-errors.patch"
cmake -S "$TVM_ROOT" -B "$TVM_ROOT/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DUSE_LLVM="$(command -v llvm-config)" \
  -DUSE_RPC=ON -DUSE_CUDA=OFF -DUSE_OPENCL=OFF
cmake --build "$TVM_ROOT/build" --parallel 4
python -m pip install -e "$TVM_ROOT/3rdparty/tvm-ffi" -e "$TVM_ROOT/python"
python -c 'import tvm; from tvm import relax, dlight; from tvm.script import tir; print(tvm.__version__)'
```

This host build does not need a GPU. Enable CUDA or OpenCL in the corresponding
target build as described in [device setup](DEVICES.md).
The [integration patch](../patches/README.md) handles compiler errors during
candidate evaluation.
