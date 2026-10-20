# Adding hardware

EdgeKernelBench supports extending its task evaluation and LEKA optimization
workflow to additional edge devices. An adapter connects the device's compiler,
runtime, and hardware feedback to the shared task and correctness interfaces.
The release provides the configurations used in the paper:

| Configuration | Compilation path | Execution |
| --- | --- | --- |
| `vivo_cpu` | LLVM IR → Android AArch64 CPU | Host cross-compilation and RPC over ADB |
| `jetson_gpu` | CUDA C → Jetson GPU | SSH worker |
| `rk3588_gpu` | OpenCL C → Mali GPU | SSH worker |

Similar devices can reuse these adapters. Compilation currently uses
[TVM 0.23.0](https://github.com/apache/tvm); check its
[build options](https://github.com/apache/tvm/blob/v0.23.0/cmake/config.cmake)
for the new target. Other devices need their own adapter and validation.

## Adapt an existing path

1. Add an entry to `BACKENDS` in
   [`LEKA/leka/backends.py`](../LEKA/leka/backends.py), using the closest
   configuration as a starting point. Set the compilation target, source
   language, transport, and runtime settings. Register the name in
   `doctor()` in [`LEKA/leka/__main__.py`](../LEKA/leka/__main__.py).

2. `MeasureService` in
   [`evaluation/evaluation_service.py`](../LEKA/leka/evaluation/evaluation_service.py)
   selects the worker. CUDA and OpenCL use `SSHRunner` and `device_worker.py`;
   Android CPU uses `VivoHostRunner`, `android_worker.py`, and `runtime/`.
   Adapt the connection, compiler/library paths, and device selection. A Linux
   CPU needs a CPU worker. Update `transport_probe.py` if the transport changes.

3. Under
   [`LEKA/leka/skills/wiki/`](../LEKA/leka/skills/wiki), add the new backend's
   folders in `L0_relax`, `L1_tir`, `L2_source`, `hardware_feedback` and
   `sources`. Describe the execution model, resource limits, runtime, and
   profiling tools. Record sources in
   `PROVENANCE.json`. `mount_wiki()` in `orchestration/optimization_loop.py`
   loads these folders by backend name. Adapt the relevant files in
   [`prompts/`](../LEKA/leka/prompts) and select the L2 prompt with `l2_agent`.
   **Provide hardware descriptions and prompts for the new device.** The
   included prompts are tailored to the paper's configurations.

4. Extend the hardware-feedback collector dispatch in
   `orchestration/optimization_loop.py` and the adapters in
   [`hardware_feedback/`](../LEKA/leka/hardware_feedback), including
   `collectors/device_profiler.py`, `attribution.py`, `quantity_contract.py`
   and `interpretation.py`. Add the new backend's profiler conventions and
   associate measurements with the candidate that produced them.

For example, another CUDA GPU needs its architecture target, connection settings,
resource limits, and profiler configuration. OpenCL and Android CPU ports also
need device-specific target and runtime settings.

## Extend compilation support

For a different compilation path, first check that a small task compiles and
runs on the device, then extend:

- `lowering/initial_schedule_worker.py`: an executable initial TIR schedule.
- `evaluation/`: compilation, source extraction/replacement, device execution
  and paired timing. `device_worker.py` currently selects CUDA or OpenCL.
- `lowering/abi_contract.py` and `lowering/interface_facts.py`: interface checks
  for the new L2 language. The current handlers cover CUDA C, OpenCL C and LLVM IR.

These adapters let the new device reuse EdgeKernelBench's task interface,
correctness checks, and LEKA search workflow.

## Check the port

After installing the compiler/runtime and registering `my_backend`:

```bash
python -m leka doctor --backend my_backend --check-device
python -m leka run --backend my_backend --task /path/to/small_task.py \
  --run-dir /tmp/leka-my-backend --model "$LEKA_MODEL" \
  --agents relax,tir --wall-minutes 10 --finalize-minutes 2 --max-rounds 1 \
  --no-hardware-feedback --no-review
```

The short run checks graph and tensor-program optimization. Verify correctness
and stable timings, then add `l2` to test device-code optimization. Enable
hardware feedback once its collector and interpretation checks pass.

Report new-device results with their hardware, software and measurement settings.
The paper's results apply to the three supplied configurations.
