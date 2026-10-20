# LEKA: Layered Edge Kernel Agent

LEKA uses hardware feedback to optimize Relax graphs, TIR programs, and device
source code.

## Usage

Complete the [setup](../SETUP.md) and set `LEKA_MODEL` to your model identifier.
Use a [benchmark task](../EdgeKernelBench/README.md) or a Python file defining
`Model`, `get_init_inputs()`, and `get_inputs()`.

```bash
python -m leka doctor --backend jetson_gpu
python -m leka run --backend jetson_gpu --task /path/to/task.py \
  --run-dir /tmp/leka-run --model "$LEKA_MODEL" \
  --agents relax,tir,l2 --scheduling async --wall-minutes 90 \
  --hardware-feedback --inline-hardware-feedback --no-review
```

Choose `jetson_gpu`, `rk3588_gpu`, or `vivo_cpu` for a supplied configuration.
See [Adding hardware](../docs/HARDWARE.md) for other devices.
The [paper's agent experiments](https://arxiv.org/html/2610.24648v1#A2.SS4)
use GLM-5.3-Flash with a 90-minute budget for Level 1 and 155 minutes for
Levels 2 and 3; set `LEKA_MODEL` and `--wall-minutes` accordingly.
Set `LEKA_AGENT_BINARY` if the model client executable is not `claude`.
Optional task constants `ATOL` and `RTOL` set numerical tolerances; both default
to `1e-4`.

## Results

The run directory saves candidates, correctness checks, timings, and hardware
feedback. Use these records to compare candidates and inspect the optimization
process.

Run `python -m leka run --help` for all options.
