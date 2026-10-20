"""Semantic repairs for trusted imported Relax.

Only unscheduled high-level Relax is produced. No candidate policy is relaxed.
"""

def normalize_imported_module(module):
    import numpy as np
    import tvm
    from tvm import relax

    @relax.expr_functor.mutator
    class Normalize(relax.PyExprMutator):
        def visit_constant_(self, constant):
            values = constant.data.numpy()
            # Importer-generated affine-less norm scales/biases are uniform.
            # Express them as ordinary graph construction, not serialized
            # metadata (which the candidate-source policy correctly rejects).
            if values.ndim and values.size and np.all(values == values.flat[0]):
                scalar = relax.const(values.flat[0].item(), str(values.dtype))
                return self.builder_.normalize(
                    relax.op.full(relax.ShapeExpr(values.shape), scalar, str(values.dtype))
                )
            return constant

        def visit_call_(self, call):
            call = self.visit_expr_post_order(call)
            if not isinstance(call.op, tvm.ir.Op) or call.op.name != "relax.nn.group_norm":
                return call
            data, gamma, beta = call.args
            shape = data.struct_info.shape
            rank = data.struct_info.ndim
            axis = int(call.attrs.channel_axis) % rank
            spatial = [int(a) % rank for a in call.attrs.axes]
            if (shape is None or axis != 1 or spatial != list(range(2, rank))
                    or data.struct_info.dtype != "float32"):
                return call
            groups = int(call.attrs.num_groups)
            dims = list(shape.values)
            # Bound every reduction's accumulation length. A serial FP32
            # mean over ~100k sigmoid values can itself lose enough precision
            # to fail the original tolerance even with two-pass variance.
            if not all(isinstance(d, tvm.tir.IntImm) for d in dims[1:]):
                return call
            import math
            extent = math.prod(int(d) for d in dims[1:]) // groups
            grouped_shape = [dims[0], groups, extent]
            emit = self.builder_.emit
            grouped = emit(relax.op.reshape(data, grouped_shape))

            def stable_mean(value):
                remaining = extent
                while remaining > 256:
                    blocks = (remaining + 255) // 256
                    padding = blocks * 256 - remaining
                    if padding:
                        zeros = emit(relax.op.full(
                            relax.ShapeExpr([dims[0], groups, padding]),
                            relax.const(0, data.struct_info.dtype),
                            data.struct_info.dtype,
                        ))
                        value = emit(relax.op.concat([value, zeros], axis=2))
                    value = emit(relax.op.reshape(value, [dims[0], groups, blocks, 256]))
                    value = emit(relax.op.sum(value, axis=[3]))
                    remaining = blocks
                total = emit(relax.op.sum(value, axis=[2], keepdims=True))
                return emit(relax.op.divide(total, relax.const(extent, data.struct_info.dtype)))

            mean = stable_mean(grouped)
            centered = emit(relax.op.subtract(grouped, mean))
            squares = emit(relax.op.multiply(centered, centered))
            variance = stable_mean(squares)
            epsilon = relax.const(float(call.attrs.epsilon), data.struct_info.dtype)
            inv_std = emit(relax.op.rsqrt(emit(relax.op.add(variance, epsilon))))
            normalized = emit(relax.op.reshape(emit(relax.op.multiply(centered, inv_std)), dims))
            affine_shape = [1, dims[1], *([1] * (rank - 2))]
            if call.attrs.scale:
                normalized = emit(relax.op.multiply(
                    normalized, emit(relax.op.reshape(gamma, affine_shape))))
            if call.attrs.center:
                normalized = emit(relax.op.add(
                    normalized, emit(relax.op.reshape(beta, affine_shape))))
            return normalized

    normalizer = Normalize(module)
    for global_var, function in module.functions.items():
        if isinstance(function, relax.Function):
            normalizer.builder_.update_func(global_var, normalizer.visit_expr(function))
    return normalizer.builder_.get()
