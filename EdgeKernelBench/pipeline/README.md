# Evaluation pipeline

Inspect task and candidate interfaces before device evaluation.

## Usage

Complete the [setup](../../SETUP.md), then run from the repository root.
Use a [materialized benchmark task](../README.md) or a Python file defining
`Model`, `get_init_inputs()`, and `get_inputs()`.

Inspect the reference task:

```bash
python -m pipeline --task /path/to/task.py
```

Validate a candidate defining `build(task) -> pipeline.interfaces.BuildResult`:

```bash
python -m pipeline --task /path/to/task.py --candidate /path/to/case.py
```

These commands check the task and candidate interfaces on the host.
For model-driven optimization and device measurements, use [LEKA](../../LEKA/README.md).
