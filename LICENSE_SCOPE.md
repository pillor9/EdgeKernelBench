# Licensing scope

Original EdgeKernelBench framework code, LEKA, tools, configurations, and project
contributions are licensed under [Apache-2.0](LICENSE), subject to the exceptions
below. Third-party code retains its original license and copyright notices.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSES/sources.json](LICENSES/sources.json) identify the applicable files and
upstream texts. File headers record the corresponding license terms.

## Task-specific terms

These self-contained benchmark task implementations retain their source licenses:

| Task file | License |
| --- | --- |
| [095_multi_stream_stem_conv.py.tpl](EdgeKernelBench/tasks/level1/095_multi_stream_stem_conv.py.tpl) | `AGPL-3.0-only` |
| [012_c2f_chunk_concat.py.tpl](EdgeKernelBench/tasks/level2/012_c2f_chunk_concat.py.tpl) | `AGPL-3.0-only` |
| [014_c3_block.py.tpl](EdgeKernelBench/tasks/level2/014_c3_block.py.tpl) | `AGPL-3.0-only` |
| [018_cbs_conv_bn_silu.py.tpl](EdgeKernelBench/tasks/level2/018_cbs_conv_bn_silu.py.tpl) | `AGPL-3.0-only` |
| [050_conv_bn_leaky_relu.py.tpl](EdgeKernelBench/tasks/level2/050_conv_bn_leaky_relu.py.tpl) | `AGPL-3.0-only` |
| [069_detect_head_static_decode.py.tpl](EdgeKernelBench/tasks/level2/069_detect_head_static_decode.py.tpl) | `AGPL-3.0-only` |
| [071_dfl_expectation.py.tpl](EdgeKernelBench/tasks/level2/071_dfl_expectation.py.tpl) | `AGPL-3.0-only` |
| [083_focus_space_to_depth.py.tpl](EdgeKernelBench/tasks/level2/083_focus_space_to_depth.py.tpl) | `AGPL-3.0-only` |
| [111_lss_depth_head.py.tpl](EdgeKernelBench/tasks/level2/111_lss_depth_head.py.tpl) | `LicenseRef-NVIDIA-LSS` |
| [118_panet_down_fusion.py.tpl](EdgeKernelBench/tasks/level2/118_panet_down_fusion.py.tpl) | `AGPL-3.0-only` |
| [142_sppf_pool_concat.py.tpl](EdgeKernelBench/tasks/level2/142_sppf_pool_concat.py.tpl) | `AGPL-3.0-only` |
| [158_unet_skip_unit.py.tpl](EdgeKernelBench/tasks/level2/158_unet_skip_unit.py.tpl) | `GPL-3.0-only` |
| [162_yolo_bottleneck.py.tpl](EdgeKernelBench/tasks/level2/162_yolo_bottleneck.py.tpl) | `AGPL-3.0-only` |

`LicenseRef-NVIDIA-LSS` refers to the [NVIDIA Lift-Splat-Shoot license](LICENSES/nv-tlabs--lift-splat-shoot/LICENSE.txt).
Its use is limited to non-commercial research or evaluation. The GPL and AGPL
tasks are supplied as source under their respective terms. These exceptions are
not covered by the project's Apache-2.0 grant.

The Python packages contain the framework and LEKA; task templates are supplied
separately in the repository. A combined or derived program must still satisfy
all applicable licenses. Directory placement alone does not determine those
obligations. Preserve the task notices and applicable license texts when sharing
materialized task files.

## Algorithm and architecture references

The local implementations below are licensed under Apache-2.0. They express
published formulas or compact network patterns using PyTorch operations. Their
paper and GitHub references are retained in task metadata and the source inventory.

| Task | Local implementation scope | Paper reference |
| --- | --- | --- |
| [SimAM](EdgeKernelBench/tasks/level2/137_simam.py.tpl) | Spatial energy-attention formula | [Equations (5)-(6), Figure 3](https://proceedings.mlr.press/v139/yang21o.html) |
| [IMDN block](EdgeKernelBench/tasks/level2/100_imdn_block.py.tpl) | Four-stage feature distillation; CCA omitted | [Section 3.2, Table 1](https://arxiv.org/abs/1909.11856) |
| [Compact depth encoder-decoder](EdgeKernelBench/tasks/level3/023_monodepth2.py.tpl) | Three-scale inference workload inspired by Monodepth2 | [Section 3.3](https://arxiv.org/abs/1806.01260) |

These entries describe the local computation and its references. They do not
grant rights to the upstream implementations, full models or pretrained weights.

## Scope of the packages and other materials

The `edgekernelbench` and `leka` Python distributions contain Apache-2.0 framework
code and their license text. Benchmark task templates, scenario profiles and the
TVM patch are not included in those distributions. The patch is accompanied by
its [TVM license and notices](LICENSES/apache--tvm/).

External dependencies remain under their own licenses. References to model
architectures do not grant rights to pretrained weights. This software license
does not grant rights to separately distributed papers, figures or third-party
logos.
