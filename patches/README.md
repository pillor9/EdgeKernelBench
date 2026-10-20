# Compiler integration

`tvm-worker-errors.patch` targets [Apache TVM](https://github.com/apache/tvm) revision
`75ce1b4d46053b28fa926a6e29aa715b2463dc92` and makes compiler failure handling
suitable for iterative candidate evaluation. It joins parallel workers during
exception unwinding and rejects invalid or dynamic GPU thread extents through
the legality checker.

Apply it before building the compiler as described in [the build guide](../docs/TVM.md).
[Apache TVM](https://github.com/apache/tvm) is an external dependency distributed
under Apache-2.0; its upstream license and notices apply to the patched compiler.
The patch includes modified upstream source fragments. The corresponding
[license](../LICENSES/apache--tvm/LICENSE.txt) and
[ASF notice](../LICENSES/apache--tvm/NOTICE.txt) are included in this repository.
