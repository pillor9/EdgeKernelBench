"""The bounded L1 graph/TIR contract, checked on parsed TVM IR.

L0 remains the owner of the canonical graph.  The optional L1 exception is one
closed, consecutive, single-use chain of two or three ``relax.call_tir``
bindings.  L1 can implement that chain as one kernel, but cannot alter its
external tensor interface or any other Relax computation.  Passing this gate
is a *structural* qualification only: ordinary T1, paired promotion and release
correctness are still mandatory.

Importing this module does not import TVM, so the ordinary host process need
not have TVM installed.  The checker runs in the existing isolated TVM parser.
"""

from __future__ import annotations


FUSION_ATTR = "leka.l1_fuse_of"
MAX_LOCAL_BINDINGS = 3


class _Violation(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise _Violation(message)


def _functions(module):
    return {str(gv.name_hint): (gv, function) for gv, function in module.functions.items()}


def _attrs(function):
    return {} if function.attrs is None else {
        str(key): value for key, value in function.attrs.items()
    }


def _primfunc_shell(function):
    import tvm
    from tvm import tir

    attrs = {key: value for key, value in _attrs(function).items()
             if key != "tir.is_scheduled"}
    return tir.PrimFunc(
        function.params, tir.Evaluate(0), function.ret_type, function.buffer_map,
        tvm.ir.make_node("ir.DictAttrs", **attrs) if attrs else None,
    )


def _module_shell(module, replacements=None, removed=()):
    import tvm
    from tvm import tir

    replacements = replacements or {}
    functions = {}
    for name, (gv, function) in _functions(module).items():
        if name in removed:
            continue
        function = replacements.get(name, function)
        functions[gv] = _primfunc_shell(function) if isinstance(function, tir.PrimFunc) else function
    return tvm.IRModule(functions, attrs=module.attrs, global_infos=module.global_infos)


def _tensor_signature(info):
    from tvm import relax, tir

    _require(isinstance(info, relax.TensorStructInfo), "local fusion requires tensor-only bindings")
    _require(isinstance(info.shape, relax.ShapeExpr), "dynamic tensor shapes require ESCALATE L0")
    shape = []
    for dim in info.shape.values:
        _require(isinstance(dim, tir.IntImm), "symbolic tensor shapes require ESCALATE L0")
        shape.append(int(dim))
    return shape, str(info.dtype)


def _call(binding):
    import tvm
    from tvm import relax

    if not isinstance(binding, relax.VarBinding):
        return None
    call = binding.value
    if not (isinstance(call, relax.Call) and isinstance(call.op, tvm.ir.Op)
            and call.op.name == "relax.call_tir" and len(call.args) == 2
            and isinstance(call.args[0], tvm.ir.GlobalVar)
            and isinstance(call.args[1], relax.Tuple) and len(call.sinfo_args) == 1):
        return None
    if not isinstance(call.sinfo_args[0], relax.TensorStructInfo):
        return None
    if not all(isinstance(arg, relax.Var) for arg in call.args[1].fields):
        return None
    return str(call.args[0].name_hint), list(call.args[1].fields)


def _check_pure_tir_storage(function):
    """Prove input read-only through buffer/pointer aliases, or fail closed.

    BufferLoad is a scalar/vector value, not an escaping pointer. This keeps
    normal TIR vectorization, thread bindings and arithmetic unrestricted while
    refusing opaque pointer operations whose write effects cannot be proved.
    """
    import tvm
    from tvm import tir

    inputs = {function.buffer_map[p].data for p in function.params[:-1]}
    owned = {function.buffer_map[function.params[-1]].data}
    aliases = {}
    explicit_aliases = set()
    calls = []
    accesses = []

    def link(variable, sources):
        aliases.setdefault(variable, set()).update(sources)

    for parameter, buffer in function.buffer_map.items():
        link(parameter, {buffer.data})

    def pointer_sources(expression):
        result = set()

        def visit(node):
            if isinstance(node, tir.Var):
                # Include integer views of pointers, even under arithmetic,
                # Select, Broadcast/Ramp or other expression wrappers.
                result.add(node)
                return node
            if isinstance(node, tir.BufferLoad):
                # Loading a numeric tensor element does not expose its data
                # pointer. Prune here rather than collecting its buffer Var.
                return node
            if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op):
                if node.op.name == "tir.address_of":
                    arg = node.args[0] if len(node.args) == 1 else None
                    if isinstance(arg, tir.BufferLoad):
                        result.add(arg.buffer.data)
                    return node
                if node.op.name == "tir.tvm_access_ptr":
                    if len(node.args) == 5:
                        result.update(pointer_sources(node.args[1]))
                    return node
            return None

        # Use TVM's complete expression traversal, not a hand-maintained list
        # of wrapper nodes that could miss pointer-to-integer laundering.
        tir.stmt_functor.ir_transform(tir.Evaluate(expression), visit, None)
        return result

    def collect(node):
        if isinstance(node, tir.Block):
            owned.update(buffer.data for buffer in node.alloc_buffers)
            for match in node.match_buffers:
                explicit_aliases.add(match.buffer.data)
                link(match.buffer.data, {match.source.buffer.data})
        if isinstance(node, (tir.Allocate, tir.AllocateConst)):
            owned.add(node.buffer_var)
        if isinstance(node, (tir.Let, tir.LetStmt)):
            sources = pointer_sources(node.value)
            if sources:
                explicit_aliases.add(node.var)
                link(node.var, sources)
        if isinstance(node, (tir.BufferLoad, tir.BufferStore)):
            accesses.append(node)
        if isinstance(node, tir.Call):
            calls.append(node)

    tir.stmt_functor.post_order_visit(function.body, collect)
    _require(not explicit_aliases.intersection(owned),
             "pointer/buffer alias rebinds owned storage and cannot prove purity")

    def roots(variable, seen=None):
        if variable in inputs or variable in owned:
            return {variable}
        seen = set() if seen is None else seen
        _require(variable not in seen, "cyclic pointer/buffer alias cannot prove purity")
        if variable not in aliases:
            return {variable}
        result = set()
        for source in aliases[variable]:
            result.update(roots(source, seen | {variable}))
        return result

    known = inputs | owned
    for access in accesses:
        storage = roots(access.buffer.data)
        _require(storage and storage <= known,
                 "unknown buffer storage/opaque pointer cannot prove pure fused kernel")
        if isinstance(access, tir.BufferStore):
            _require(not storage.intersection(inputs),
                     "write to external input of a pure call_tir region through buffer alias")

    def pointer_argument(argument):
        if isinstance(argument, tir.StringImm):
            return False  # e.g. the standard storage_sync("shared") scope
        sources = pointer_sources(argument)
        for source in sources:
            if roots(source).intersection(known) or str(source.dtype) == "handle":
                return True
        return str(getattr(argument, "dtype", "")) == "handle"

    for access in accesses:
        _require(not any(pointer_argument(index) for index in access.indices),
                 "opaque pointer used as buffer index cannot prove pure fused kernel")
        if isinstance(access, tir.BufferStore):
            _require(not pointer_argument(access.value),
                     "opaque pointer escape through BufferStore cannot prove pure fused kernel")

    for call in calls:
        _require(isinstance(call.op, tvm.ir.Op), "new fused kernel adds external/helper authority: helper/global call")
        name = call.op.name
        _require(not any(marker in name for marker in (
            "extern", "packed", "trace", "callback", "cpacked", "throw_last_error")),
            "new fused kernel adds external/helper authority: " + name)
        if name == "tir.address_of":
            _require(len(call.args) == 1 and isinstance(call.args[0], tir.BufferLoad),
                     "opaque address_of cannot prove pure fused kernel")
            # Forming an address is not a write. Its consumers are separately
            # checked and Let/MatchBuffer consumers retain its storage root.
            continue
        if name == "tir.tvm_access_ptr":
            _require(len(call.args) == 5 and isinstance(call.args[1], tir.Var)
                     and isinstance(call.args[4], tir.IntImm),
                     "opaque access_ptr cannot prove pure fused kernel")
            storage = roots(call.args[1])
            mask = int(call.args[4])
            _require(storage and storage <= known and mask in (1, 2, 3),
                     "unknown access_ptr storage/mode cannot prove pure fused kernel")
            _require(not (mask & 2 and storage.intersection(inputs)),
                     "write access pointer to external input of a pure call_tir region")
            continue
        _require(str(call.dtype) != "handle" and not any(pointer_argument(arg) for arg in call.args),
                 "opaque pointer intrinsic/escape cannot prove pure fused kernel: " + name)


def _check_fused_abi(function, fused_name, inputs, output):
    import tvm
    from tvm import tir

    attrs = _attrs(function)
    _require(set(attrs) <= {FUSION_ATTR, "global_symbol", "tir.noalias", "tir.is_scheduled"},
             "new fused PrimFunc adds unsupported attributes/native authority")
    if "global_symbol" in attrs:
        _require(str(attrs["global_symbol"]) == fused_name, "fused global_symbol must match its kernel name")
    if "tir.noalias" in attrs:
        _require(int(attrs["tir.noalias"]) == 1, "fused tir.noalias must be true")
    _require(isinstance(function.ret_type, tvm.ir.TupleType) and len(function.ret_type.fields) == 0,
             "fused kernel must have the standard void return ABI")
    tensors = list(inputs) + [output]
    _require(len(function.params) == len(tensors) and len(function.buffer_map) == len(tensors),
             "fused buffer ABI must be external inputs in first-use order, then one output")
    _require(len({buffer.data for buffer in function.buffer_map.values()}) == len(tensors),
             "fused input/output buffers cannot alias their data variables")
    for parameter, tensor in zip(function.params, tensors):
        _require(parameter in function.buffer_map, "scalar/opaque fused parameters require ESCALATE L0")
        _require(str(parameter.dtype) == "handle", "fused tensor parameters must be handles")
        buffer = function.buffer_map[parameter]
        shape, dtype = _tensor_signature(tensor.struct_info)
        _require(len(buffer.shape) == len(shape)
                 and all(isinstance(dim, tir.IntImm) and int(dim) == expected
                         for dim, expected in zip(buffer.shape, shape))
                 and str(buffer.dtype) == dtype,
                 "fused buffer shape/dtype differs from the canonical boundary")
        _require(not buffer.strides and isinstance(buffer.elem_offset, tir.IntImm)
                 and int(buffer.elem_offset) == 0 and buffer.scope() == "global"
                 and int(buffer.buffer_type) == 1 and not buffer.axis_separators
                 and int(buffer.data_alignment) == 64 and int(buffer.offset_factor) == 1,
                 "fused boundary layout/strides/scope must remain ordinary contiguous tensors")
    _check_pure_tir_storage(function)


def _find_region(canonical, names):
    from tvm import relax

    uses = {name: 0 for name in names}
    matches = []
    for function_name, (_, function) in _functions(canonical).items():
        if not isinstance(function, relax.Function):
            continue

        def count(node):
            import tvm
            if isinstance(node, relax.Call):
                for arg in node.args:
                    if isinstance(arg, tvm.ir.GlobalVar) and str(arg.name_hint) in uses:
                        uses[str(arg.name_hint)] += 1
                if isinstance(node.op, tvm.ir.GlobalVar) and str(node.op.name_hint) in uses:
                    uses[str(node.op.name_hint)] += 1

        relax.analysis.post_order_visit(function.body, count)
        if not isinstance(function.body, relax.SeqExpr):
            continue
        for block_index, block in enumerate(function.body.blocks):
            for index in range(len(block.bindings) - len(names) + 1):
                region = list(block.bindings[index:index + len(names)])
                calls = [_call(binding) for binding in region]
                if all(call is not None for call in calls) and [call[0] for call in calls] == names:
                    matches.append((function_name, function, block_index, index, region, calls))
    _require(all(count == 1 for count in uses.values()),
             "local fusion requires each original kernel to have exactly one call site")
    _require(len(matches) == 1, "fusion declaration must match one consecutive 2–3 call_tir region")
    return matches[0]


def _contract_region(function, block_index, index, region, calls, fused_gv):
    from tvm import relax

    intermediate = {binding.var for binding in region[:-1]}
    outputs = {binding.var for binding in region}
    inputs = []
    for offset, (_, args) in enumerate(calls):
        if offset:
            _require(region[offset - 1].var in args,
                     "local fusion must be a connected consecutive chain, not unrelated kernels")
        available = {binding.var for binding in region[:offset]}
        for arg in args:
            _tensor_signature(arg.struct_info)
            _require(arg not in outputs or arg in available, "region contains a forward tensor dependency")
            if arg not in outputs and arg not in inputs:
                inputs.append(arg)
    for block_position, block in enumerate(function.body.blocks):
        for binding_position, binding in enumerate(block.bindings):
            if block_position == block_index and index <= binding_position < index + len(region):
                continue
            _require(not intermediate.intersection(relax.analysis.free_vars(binding.value)),
                     "an intermediate tensor escapes the proposed local fusion region")
    _require(not intermediate.intersection(relax.analysis.free_vars(function.body.body)),
             "an intermediate tensor escapes as a function result")
    last = region[-1]
    _tensor_signature(last.var.struct_info)
    fused_call = relax.Call(last.value.op, [fused_gv, relax.Tuple(inputs)],
                            attrs=last.value.attrs, sinfo_args=last.value.sinfo_args)
    replacement = relax.VarBinding(last.var, fused_call)
    blocks = []
    for position, block in enumerate(function.body.blocks):
        if position == block_index:
            bindings = list(block.bindings[:index]) + [replacement] + list(block.bindings[index + len(region):])
            block = (relax.DataflowBlock(bindings) if isinstance(block, relax.DataflowBlock)
                     else relax.BindingBlock(bindings))
        blocks.append(block)
    contracted = relax.Function(function.params, relax.SeqExpr(blocks, function.body.body),
                                function.ret_struct_info, function.is_pure, function.attrs)
    return contracted, inputs, last.var


def check_l1_contract(candidate_module, canonical_module, allow_local_fusion=False):
    """Return a JSON-serializable structural gate result, never final correctness.

    With fusion disabled, only existing PrimFunc bodies and ``tir.is_scheduled``
    may change.  With fusion enabled, the sole additional capability is the
    bounded closed-region contraction documented above.  Unsupported graph work
    is rejected with an explicit request to escalate to L0.
    """
    import tvm
    from tvm import relax, tir

    try:
        _require(isinstance(candidate_module, tvm.IRModule) and isinstance(canonical_module, tvm.IRModule),
                 "L1 contract requires candidate and canonical IRModules")
        _require(tvm.ir.structural_equal(candidate_module.attrs, canonical_module.attrs),
                 "module attributes/parameter plan must remain unchanged")
        _require(tvm.ir.structural_equal(candidate_module.global_infos, canonical_module.global_infos),
                 "module global information must remain unchanged")
        candidates, canonicals = _functions(candidate_module), _functions(canonical_module)
        _require(all(isinstance(function, (relax.Function, tir.PrimFunc))
                     for _, function in candidates.values()), "unsupported function kind")
        declared = [(name, gv, function) for name, (gv, function) in candidates.items()
                    if isinstance(function, tir.PrimFunc) and FUSION_ATTR in _attrs(function)]
        if not declared:
            _require(set(candidates) == set(canonicals), "function set changed without a local fusion declaration")
            # Give the agent an actionable function name, retaining the
            # pre-joint contract's diagnostics instead of an opaque module diff.
            for name, (_, original) in canonicals.items():
                proposed = candidates[name][1]
                same_kind = isinstance(proposed, type(original))
                if isinstance(original, tir.PrimFunc) and same_kind:
                    unchanged = tvm.ir.structural_equal(
                        _primfunc_shell(proposed), _primfunc_shell(original),
                        map_free_vars=True,
                    )
                elif isinstance(original, relax.Function) and same_kind:
                    unchanged = tvm.ir.structural_equal(proposed, original, map_free_vars=True)
                else:
                    unchanged = False
                _require(unchanged, f"non-PrimFunc-body contract changed at {name!r}: "
                         "Relax graph/function ABI are frozen")
            _require(tvm.ir.structural_equal(_module_shell(candidate_module),
                                            _module_shell(canonical_module), map_free_vars=True),
                     "non-PrimFunc-body contract changed: Relax graph/function ABI are frozen")
            return {"ok": True, "local_rewrite": None}
        _require(allow_local_fusion, "local graph/TIR fusion is disabled at this strict boundary")
        _require(len(declared) == 1, "only one local fusion region is allowed per canonical graph")
        fused_name, fused_gv, fused_function = declared[0]
        raw_names = _attrs(fused_function)[FUSION_ATTR]
        _require(isinstance(raw_names, (tvm.ir.container.Array, list, tuple)),
                 "leka.l1_fuse_of must be an ordered list of original kernel names")
        names = [str(name) for name in raw_names]
        _require(2 <= len(names) <= MAX_LOCAL_BINDINGS and len(set(names)) == len(names),
                 "leka.l1_fuse_of must name two or three distinct original kernels")
        _require(fused_name not in canonicals, "fused kernel must have a new name")
        _require(all(name in canonicals and isinstance(canonicals[name][1], tir.PrimFunc) for name in names),
                 "fusion declaration references a noncanonical/non-TIR function")
        _require(set(candidates) == (set(canonicals) - set(names)) | {fused_name},
                 "fusion must remove its original kernels and cannot add/retain other functions")
        function_name, function, block_index, index, region, calls = _find_region(canonical_module, names)
        contracted, inputs, output = _contract_region(function, block_index, index, region, calls, fused_gv)
        _check_fused_abi(fused_function, fused_name, inputs, output)
        expected = _module_shell(canonical_module, {function_name: contracted}, removed=names)
        expected[fused_gv] = _primfunc_shell(fused_function)
        _require(tvm.ir.structural_equal(_module_shell(candidate_module), expected, map_free_vars=True),
                 "candidate graph differs from the declared local contraction or changes a frozen ABI")
        return {"ok": True, "local_rewrite": {
            "kind": "bounded_call_tir_fusion", "relax_function": function_name,
            "original_kernels": names, "fused_kernel": fused_name,
            "binding_count": len(names), "external_input_count": len(inputs),
            "output_count": 1,
        }}
    except _Violation as error:
        return {"ok": False, "fault_stage": "parse", "error": str(error) + "; ESCALATE L0 for broader graph work"}
    except (TypeError, ValueError, tvm.error.TVMError) as error:
        return {"ok": False, "fault_stage": "parse", "error": "L1 contract could not be validated: " + str(error)}
