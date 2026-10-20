# EdgeKernelBench

EdgeKernelBench provides 331 PyTorch tasks with 650 scenario-specific instances.
Choose `mobile_personal`, `smart_aiot`, or `embodied_robotics` to generate a task
with the corresponding shapes and input constants.

## Generate a task

Run from this directory. Materialization requires PyYAML; task execution requires
PyTorch and NumPy. See the [setup guide](../SETUP.md) for environment setup.

```bash
python tools/materialize.py --list
python tools/materialize.py --describe ekb_l1_001_abs
python tools/validate.py
python tools/materialize.py --scenario mobile_personal \
  --task ekb_l1_001_abs --output /tmp/ekb-abs.py
```

`--describe` shows supported scenarios and required profile fields. The output
path must end in `.py` and must not already exist. For custom shapes, replace
`--scenario mobile_personal` with `--profile /path/to/profile.json`; use trusted
profiles because they can contain Python expressions.

## Use the task

Each generated task defines `Model`, `get_init_inputs()` for constructor
arguments, and `get_inputs()` for model inputs:

```python
model = Model(*get_init_inputs()).eval()
output = model(*get_inputs())
```

Pass the resulting task to [LEKA](../LEKA/README.md) for model-driven generation
and device evaluation, or use the [pipeline](pipeline/README.md) interfaces to
build an evaluation workflow.
