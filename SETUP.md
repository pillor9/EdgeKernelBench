# Setup

Run the commands from the repository root in a Python 3.11+ environment.
The reference host is macOS; LEKA's model client needs its OS sandbox.

## Compiler and packages

Prepare [TVM 0.23.0](https://github.com/apache/tvm) with Relax, DLight, LLVM 18, and RPC support using the
[compiler build instructions](docs/TVM.md). Then install the project:

```bash
python -m pip install -e ./EdgeKernelBench -e ./LEKA
```

`LEKA_TVM_PYTHON` selects the Python interpreter for compilation and evaluation
workers. Both this interpreter and the main LEKA environment need matching [TVM](https://github.com/apache/tvm),
NumPy, and PyTorch installations.

## Model client

Install and authenticate Claude Code, or a compatible client with persistent
`stream-json` input/output and the required sandbox options. `leka doctor`
checks the client interface; the reference client is Claude Code 2.1.257.

Set `LEKA_MODEL` to your provider's model identifier. Use `LEKA_AGENT_BINARY`
when the executable is not `claude` on `PATH`. Keep credentials in your local
environment.

## Device

Prepare a matching [TVM](https://github.com/apache/tvm) runtime on the target and configure an execution path:

| Backend | Runtime and connection |
| --- | --- |
| `jetson_gpu` | CUDA on Jetson Orin; set `LEKA_JETSON_SSH_ALIAS` to your SSH alias |
| `rk3588_gpu` | OpenCL on RK3588; set `LEKA_RK3588_SSH_ALIAS` to your SSH alias |
| `vivo_cpu` | Android AArch64 RPC; set `VIVO_SERIAL` and `TVM_NDK_CC` |

GPU paths require SSH, rsync, and a remote `python3` with [TVM](https://github.com/apache/tvm) and NumPy.
Hardware feedback uses Nsight Compute on Jetson, [TVM](https://github.com/apache/tvm) profiling on RK3588,
and simpleperf on Android. See [device setup](docs/DEVICES.md) for runtime
and profiler configuration, or [Adding hardware](docs/HARDWARE.md) for other targets.

## First run

The example uses the paper's Jetson scenario. Keep the target idle and use a
fresh output directory for each run.

```bash
python EdgeKernelBench/tools/materialize.py --scenario embodied_robotics \
  --task ekb_l1_001_abs --output /tmp/ekb-abs.py
python -m leka doctor --backend jetson_gpu
python -m leka run --backend jetson_gpu --task /tmp/ekb-abs.py \
  --run-dir /tmp/leka-abs --model "$LEKA_MODEL" \
  --agents relax,tir,l2 --scheduling async --wall-minutes 90 \
  --hardware-feedback --inline-hardware-feedback --no-review
```

`doctor` checks the local environment. Add `--check-device` for a transport
probe. `run` invokes the model and target device. The example keeps numerical
and representation checks enabled; `--no-review` disables the optional
independent source review. Enable it with `--review --review-model MODEL`.
