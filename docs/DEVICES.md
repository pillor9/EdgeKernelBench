# Device setup

These settings reproduce the three supplied execution paths. For a different
device, follow [Adding hardware](HARDWARE.md).

## Jetson CUDA

The target is Jetson Orin (`sm_87`). Install CUDA and the matching [TVM](https://github.com/apache/tvm) revision
with CUDA, LLVM, and RPC enabled. The noninteractive SSH environment's `python3`
must import `tvm` and NumPy. The worker expects CUDA tools under `/usr/local/cuda/bin`.

Set `LEKA_JETSON_SSH_ALIAS` to a configured SSH alias. Hardware feedback requires
Nsight Compute (`LEKA_NCU_BIN`) and scoped `sudo -n` permission for the profiler.
Use `LEKA_JETSON_PYTHON` and `LEKA_JETSON_PYTHONPATH` to select the profiling
interpreter and its imports.

## RK3588 OpenCL

Install the Mali OpenCL driver and headers, plus the matching [TVM](https://github.com/apache/tvm) revision with
OpenCL, LLVM, and RPC enabled. The noninteractive SSH environment's `python3`
must import `tvm` and NumPy. Set `LEKA_RK3588_SSH_ALIAS` to your SSH alias.

Feedback uses [TVM](https://github.com/apache/tvm) profiling. Optional HWCPipe counters require an external
`mali-sampler`, selected by `LEKA_MALI_SAMPLER`.

## Android CPU

The host needs ADB, NDK r26d, and LLVM 18 `llc`. Set `VIVO_SERIAL` to the device
identifier and `TVM_NDK_CC` to the NDK's `aarch64-linux-android28-clang`.

Build the matching [TVM](https://github.com/apache/tvm) runtime for `arm64-v8a`, API 28, with `USE_RPC=ON` and
`USE_CPP_RPC=ON`. Place `tvm_rpc`, `libtvm_runtime.so`, and their shared-library
dependencies in `LEKA_VIVO_DEVICE_LIB_DIR` (default `/data/local/tmp/vivo_tvm`).
Dynamically linked builds also need `libtvm_ffi.so` and `libc++_shared.so`.

Use `VIVO_RPC_PORT` for the RPC port and `LEKA_VIVO_STATE_DIR` for a separate
host state directory. The device needs `taskset`, `ss`, readable CPU-frequency
and thermal information, and simpleperf with access to CPU performance events.

## Measurements

Keep the target idle and use consistent power and clock settings during a
comparison. Transport checks do not configure those settings or reserve the
device against other users. Confirm that the required profiler works before
enabling hardware feedback.
