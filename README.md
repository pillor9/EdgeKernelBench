# EdgeKernelBench

Code and tasks for **EdgeKernelBench: Unified LLM-Driven Kernel Generation for Edge Backends**.

[Paper](https://arxiv.org/abs/2610.24648) · [Project page](https://pillor9.github.io/EdgeKernelBench/)

EdgeKernelBench provides a shared task interface and evaluation workflow for
LLM-generated kernels across edge devices. **LEKA (Layered Edge Kernel Agent)**
uses hardware feedback to coordinate graph, tensor-program, and device-code
optimization. Compilation uses [TVM 0.23.0](https://github.com/apache/tvm).

## Benchmark

The benchmark contains **331 canonical tasks** and **650 scenario-specific instances**.

| Level | Scope | Tasks |
| --- | --- | ---: |
| 1 | Basic Compute Patterns | 128 |
| 2 | Composed Operator Graphs | 162 |
| 3 | Compact Model Graphs | 41 |

The paper uses three preconfigured execution paths:

| Scenario | Device | Execution path | Instances |
| --- | --- | --- | ---: |
| Mobile and Personal Intelligence | iQOO Neo9S Pro+ (Snapdragon 8 Gen 3) | CPU / LLVM | 204 |
| Embodied Robotics | Jetson AGX Orin 64GB | GPU / CUDA | 218 |
| Smart AIoT | Firefly ROC-RK3588S-PC (Mali-G610) | GPU / OpenCL | 228 |

EdgeKernelBench can support additional edge devices through backend adapters,
reusing its task interface, correctness checks, and layered optimization workflow.
The supplied configurations cover the paper's devices. For other devices, adapt
the execution path and provide device-specific hardware descriptions and prompts;
see [Adding hardware](docs/HARDWARE.md).

## Setup

Use Python 3.11+, [TVM 0.23.0](https://github.com/apache/tvm), and a compatible Claude Code client.
The [setup guide](SETUP.md) covers the compiler and device prerequisites.

```bash
git clone https://github.com/pillor9/EdgeKernelBench.git
cd EdgeKernelBench
python -m pip install -e ./EdgeKernelBench -e ./LEKA
```

## Run

Materialize a task for the Jetson scenario, then run LEKA with your configured
model and device:

```bash
python EdgeKernelBench/tools/materialize.py --scenario embodied_robotics \
  --task ekb_l1_001_abs --output /tmp/ekb-abs.py
python -m leka doctor --backend jetson_gpu
python -m leka run --backend jetson_gpu --task /tmp/ekb-abs.py \
  --run-dir /tmp/leka-abs --model "$LEKA_MODEL" \
  --agents relax,tir,l2 --scheduling async --wall-minutes 90 \
  --hardware-feedback --inline-hardware-feedback --no-review
```

Runs save generated candidates, correctness checks, timings, and hardware
feedback. Use `python -m leka run --help` for options.

## Guides

- [Benchmark tasks and shapes](EdgeKernelBench/README.md)
- [LEKA](LEKA/README.md)
- [Evaluation interfaces](EdgeKernelBench/pipeline/README.md)
- [Adding hardware](docs/HARDWARE.md)

This release includes the task set, evaluation components, and LEKA workflow.
The paper's one-shot and inference-engine experiment drivers, tuned baseline
databases, and historical run outputs are not included. Task metadata and the
hardware-knowledge registry retain upstream references.

## License

Original framework code and LEKA are licensed under [Apache-2.0](LICENSE).
Benchmark tasks retain applicable third-party terms; see
[licensing scope](LICENSE_SCOPE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
