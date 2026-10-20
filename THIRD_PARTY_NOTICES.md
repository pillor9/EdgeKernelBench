# Third-party notices

Original EdgeKernelBench code is licensed under the terms and scope described in
[LICENSE](LICENSE) and [LICENSE_SCOPE.md](LICENSE_SCOPE.md). Third-party material
retains its own terms. In particular, some task templates have copyleft or
non-commercial conditions.

The [source inventory](LICENSES/sources.json) maps upstream references to local
files and identifies implementation adaptations where established. Semantic
references are listed separately from evidence of adapted code. Listed revisions
are review snapshots unless marked as a recorded source revision or patch base;
they are not claims about when code was originally imported.

## Benchmark task sources

| Source | Retained terms and attribution |
| --- | --- |
| [MobileKernelBench](https://github.com/zeezou-isee/MobileKernelBench_git) | [Apache-2.0](LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt). Operator templates include adaptations of MobileKernelBench implementations. |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | [MIT](LICENSES/ScalingIntelligence--KernelBench/LICENSE.txt). Copyright (c) 2023 Anne Ouyang, Simon Guo, Azalia Mirhoseini; Scaling Intelligence Lab, Stanford University. |
| [torchvision](https://github.com/pytorch/vision) | [BSD-3-Clause](LICENSES/pytorch--vision/LICENSE.txt). Copyright (c) Soumith Chintala 2016. All rights reserved. |
| [MLC LLM](https://github.com/mlc-ai/mlc-llm) | [Apache-2.0](LICENSES/mlc-ai--mlc-llm/LICENSE.txt) and [NOTICE](LICENSES/mlc-ai--mlc-llm/NOTICE.txt). Copyright (c) 2023-2025 by MLC LLM Contributors. Task references use recorded revision `2008fe8343e1f40ef89ee57b9287aebcf1b86c98`. |
| [timm](https://github.com/huggingface/pytorch-image-models) | [Apache-2.0](LICENSES/huggingface--pytorch-image-models/LICENSE.txt), with [file-level notices](LICENSES/huggingface--pytorch-image-models/SOURCE_HEADERS.txt), including Ross Wightman's contributions and Google LLC's ResNetV2 notice. |
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer) | [MIT](LICENSES/microsoft--Swin-Transformer/LICENSE.txt). Copyright (c) 2021 Microsoft. Written by Ze Liu. This original-source attribution is retained through the timm Swin references. |

Other referenced implementations include Non-local PyTorch, ECANet, RepLKNet,
RepVGG, CBAM, NanoDet, Honk, DETR, PointNet, GhostNet, Diffusers, Transformers,
Kornia, TCN, LoRA, AWQ, MLPerf Tiny, OpenCLIP, MMDetection3D, Anomalib, RAFT,
PyTorch Examples, Conformer, ECBSR, GCNet, DDRNet, PSMNet and EfficientFormer. Their license texts,
applicable source headers and notices are listed in [LICENSES](LICENSES/README.md).

Task adaptations use scenario parameters and common input/output wrappers;
individual forward paths may also be simplified for evaluation. These changes
were made for EdgeKernelBench. Including an upstream reference does not imply
that its entire implementation, dependencies or pretrained weights are bundled.

## Tasks with separate conditions

The exact task assignments are in [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and the
task file headers. These terms are not replaced by the project's Apache license.

| Source | Retained terms |
| --- | --- |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | [GNU AGPL version 3](LICENSES/ultralytics--ultralytics/LICENSE.txt); [original source notice](LICENSES/ultralytics--ultralytics/SOURCE_HEADERS.txt). |
| [YOLOv5](https://github.com/ultralytics/yolov5) | [GNU AGPL version 3](LICENSES/ultralytics--yolov5/LICENSE.txt); [original source notice](LICENSES/ultralytics--yolov5/SOURCE_HEADERS.txt). The retained source copy already carries an AGPL notice. |
| [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) | [GNU GPL version 3](LICENSES/milesial--Pytorch-UNet/LICENSE.txt). |
| [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot) | [NVIDIA Source Code License](LICENSES/nv-tlabs--lift-splat-shoot/LICENSE.txt), including its non-commercial conditions. Copyright (C) 2020 NVIDIA Corporation. All rights reserved. Authors: Jonah Philion and Sanja Fidler. |

SimAM, the compact IMDN block and the three-scale depth encoder-decoder are
algorithm or architecture references. Their local implementations use Apache-2.0;
see [implementation scope and paper references](LICENSE_SCOPE.md#algorithm-and-architecture-references).
Their upstream repositories and pretrained weights retain their own terms.

## Apache TVM patch

`patches/tvm-worker-errors.patch` modifies these Apache TVM files at revision
`75ce1b4d46053b28fa926a6e29aa715b2463dc92`:

- `src/meta_schedule/postproc/verify_gpu_code.cc`
- `src/support/parallel_for.cc`

The changes join parallel workers during exception unwinding and reject invalid
or dynamic GPU thread extents during legality checks. The upstream
[LICENSE](LICENSES/apache--tvm/LICENSE.txt) and
[NOTICE](LICENSES/apache--tvm/NOTICE.txt) are reproduced unchanged.

Apache TVM — Copyright 2019-2023 The Apache Software Foundation. This product
includes software developed at The Apache Software Foundation
(http://www.apache.org/).

The retained TVM license contains its own bundled-component inventory. That
inventory describes upstream TVM; this repository distributes the patch, not
those compiler dependencies. External dependencies installed separately remain
subject to their own licenses.
