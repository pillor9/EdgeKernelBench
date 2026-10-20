"""Static safety checks for agent-authored TVMScript candidates.

Accept only the supported declarative subset before worker import. Calls target
the ``I``, ``R`` or ``T`` namespaces, declared Relax functions, or allowed pure
builtins. Later checks validate the resulting IR and numerical behavior.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterable


class CandidateSourceError(ValueError):
    """Raised when candidate source is outside the safe TVMScript subset."""


_DSL_IMPORTS = {"ir": "I", "relax": "R", "tir": "T"}
_SAFE_BUILTINS = frozenset({"range", "min", "max", "int", "float", "bool", "len"})
_RESERVED_BINDINGS = frozenset({"I", "R", "T", "Module", "tvm", "relax"})
_FORBIDDEN_DSL_CALL_MARKERS = (
    "builtin",
    "closure",
    "extern",
    "packed",
    "py_func",
    "system_lib",
)
_DANGEROUS_GLOBALS = frozenset(
    {
        "__import__",
        "breakpoint",
        "builtins",
        "compile",
        "ctypes",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "importlib",
        "input",
        "locals",
        "marshal",
        "memoryview",
        "open",
        "os",
        "pathlib",
        "pickle",
        "setattr",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "vars",
    }
)


def _location(node: ast.AST | None) -> str:
    if node is None or not hasattr(node, "lineno"):
        return ""
    return f" at line {node.lineno}, column {getattr(node, 'col_offset', 0) + 1}"


def _reject(message: str, node: ast.AST | None = None) -> None:
    raise CandidateSourceError(message + _location(node))


def _is_dunder(name: str) -> bool:
    return name.startswith("__") or name.endswith("__")


def _check_identifier(name: str, node: ast.AST) -> None:
    if _is_dunder(name):
        _reject(f"dunder identifier {name!r} is forbidden", node)


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        _check_identifier(target.id, target)
        if target.id in _RESERVED_BINDINGS:
            _reject(f"rebinding reserved name {target.id!r} is forbidden", target)
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_target_names(item))
        return names
    if isinstance(target, ast.Subscript):
        # A TIR buffer store is an IR statement, not a Python name binding.
        return set()
    _reject(f"assignment target {type(target).__name__} is forbidden", target)


def _collect_locals(statements: Iterable[ast.stmt], names: set[str]) -> None:
    """Collect Python-local bindings so builtin-looking tensor names stay valid."""

    for statement in statements:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                names.update(_target_names(target))
        elif isinstance(statement, ast.AnnAssign):
            names.update(_target_names(statement.target))
        elif isinstance(statement, ast.AugAssign):
            # Augmented subscript stores do not bind; augmented names do.
            names.update(_target_names(statement.target))
        elif isinstance(statement, ast.For):
            names.update(_target_names(statement.target))
            _collect_locals(statement.body, names)
            _collect_locals(statement.orelse, names)
        elif isinstance(statement, ast.If):
            _collect_locals(statement.body, names)
            _collect_locals(statement.orelse, names)
        elif isinstance(statement, ast.With):
            for item in statement.items:
                if item.optional_vars is not None:
                    names.update(_target_names(item.optional_vars))
            _collect_locals(statement.body, names)


class _Validator:
    def __init__(self) -> None:
        self.dsl_roots: set[str] = set()
        self._saw_tvm_import = False
        self._saw_definition = False
        self._saw_module = False
        self._module_functions: dict[str, str] = {}
        self._relax_calls: dict[str, dict[str, ast.Call]] = {}
        self._current_function: str | None = None
        self._current_function_kind: str | None = None
        self._current_module_aliases: frozenset[str] = frozenset()

    def validate(self, tree: ast.Module) -> None:
        module_count = 0
        build_count = 0
        for index, statement in enumerate(tree.body):
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                if index == 0 and isinstance(statement.value.value, str):
                    continue  # An optional module docstring is inert.
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                if self._saw_definition:
                    _reject("imports must precede all candidate definitions", statement)
                self._validate_import(statement)
                continue

            self._saw_definition = True
            if isinstance(statement, ast.ClassDef):
                module_count += 1
                self._validate_module_class(statement)
            elif isinstance(statement, ast.FunctionDef) and statement.name == "build_relax":
                build_count += 1
                self._validate_builder(statement)
            elif isinstance(statement, ast.AsyncFunctionDef):
                _reject("async functions are forbidden", statement)
            else:
                _reject(
                    "top-level candidate code may contain only approved imports, "
                    "@I.ir_module class Module, and build_relax",
                    statement,
                )

        if not self._saw_tvm_import:
            _reject("candidate must contain exactly the approved 'import tvm' import")
        if "I" not in self.dsl_roots:
            _reject("candidate must import tvm.script.ir as I")
        if module_count != 1:
            _reject(f"candidate must define exactly one Module class, found {module_count}")
        if build_count > 1:
            _reject("candidate may define build_relax at most once")

    def _validate_import(self, statement: ast.Import | ast.ImportFrom) -> None:
        if isinstance(statement, ast.Import):
            if len(statement.names) != 1:
                _reject("only a standalone 'import tvm' is allowed", statement)
            alias = statement.names[0]
            if alias.name != "tvm" or alias.asname is not None:
                _reject("only the exact import 'import tvm' is allowed", statement)
            if self._saw_tvm_import:
                _reject("duplicate 'import tvm' is forbidden", statement)
            self._saw_tvm_import = True
            return

        if statement.level != 0:
            _reject("relative imports are forbidden", statement)
        if statement.module == "tvm":
            # A legacy naive task contains this unused import.  The bound name is
            # intentionally unusable elsewhere by the expression validator.
            if len(statement.names) == 1:
                alias = statement.names[0]
                if alias.name == "relax" and alias.asname is None:
                    return
            _reject("only the compatibility import 'from tvm import relax' is allowed", statement)
        if statement.module != "tvm.script":
            _reject("imports are restricted to tvm and tvm.script DSL aliases", statement)
        if not statement.names:
            _reject("empty tvm.script import", statement)
        for alias in statement.names:
            expected = _DSL_IMPORTS.get(alias.name)
            if expected is None or alias.asname != expected:
                _reject(
                    "tvm.script imports must be ir as I, relax as R, or tir as T",
                    statement,
                )
            if expected in self.dsl_roots:
                _reject(f"duplicate DSL alias {expected!r} is forbidden", statement)
            self.dsl_roots.add(expected)

    def _validate_module_class(self, node: ast.ClassDef) -> None:
        if self._saw_module:
            _reject("candidate may define only one Module class", node)
        self._saw_module = True
        _check_identifier(node.name, node)
        if node.name != "Module":
            _reject("the @I.ir_module class must be named Module", node)
        if node.bases or node.keywords:
            _reject("Module may not have bases or class keywords", node)
        if getattr(node, "type_params", ()):  # Python 3.12+
            _reject("generic class parameters are forbidden", node)
        if len(node.decorator_list) != 1 or not self._is_exact_attribute(
            node.decorator_list[0], "I", "ir_module"
        ):
            _reject("Module must have exactly the decorator @I.ir_module", node)
        if "I" not in self.dsl_roots:
            _reject("@I.ir_module requires 'from tvm.script import ir as I'", node)

        functions: list[ast.FunctionDef] = []
        for index, statement in enumerate(node.body):
            if (
                index == 0
                and isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if isinstance(statement, ast.FunctionDef):
                functions.append(statement)
            elif isinstance(statement, ast.AsyncFunctionDef):
                _reject("async TVMScript functions are forbidden", statement)
            else:
                _reject("Module class body may contain only decorated TVMScript functions", statement)
        if not functions:
            _reject("Module must contain at least one TVMScript function", node)

        # Collect every declaration before validating bodies.  TVMScript permits
        # a Relax function to call a later-declared Relax helper, and its printer
        # commonly emits ``cls = Module; cls.helper(...)``.  The complete static
        # symbol table keeps that narrow form possible without admitting Python
        # object dispatch or an undeclared class attribute.
        for function in functions:
            _check_identifier(function.name, function)
            if function.name in self._module_functions:
                _reject(
                    f"duplicate Module function {function.name!r} is forbidden",
                    function,
                )
            kind = self._script_function_kind(function)
            self._module_functions[function.name] = kind
            if kind == "R":
                self._relax_calls[function.name] = {}
        for function in functions:
            self._validate_script_function(function)
        self._reject_recursive_relax_calls()

    def _script_function_kind(self, node: ast.FunctionDef) -> str:
        if len(node.decorator_list) != 1:
            _reject(
                "TVMScript functions require exactly one R.function or "
                "T.prim_func decorator",
                node,
            )
        decorator = node.decorator_list[0]
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if self._is_exact_attribute(target, "R", "function"):
            kind = "R"
        elif self._is_exact_attribute(target, "T", "prim_func"):
            kind = "T"
        else:
            _reject("function decorator must be R.function or T.prim_func", decorator)
        if kind not in self.dsl_roots:
            _reject(f"decorator requires the imported {kind} DSL alias", decorator)
        return kind

    def _validate_script_function(self, node: ast.FunctionDef) -> None:
        _check_identifier(node.name, node)
        if getattr(node, "type_params", ()):  # Python 3.12+
            _reject("generic function parameters are forbidden", node)
        required_root = self._script_function_kind(node)
        decorator = node.decorator_list[0]
        if isinstance(decorator, ast.Call):
            self._validate_call(decorator, frozenset())
        else:
            self._validate_expr(decorator, frozenset())

        argument_names = self._validate_arguments(node.args)
        local_names = set(argument_names)
        _collect_locals(node.body, local_names)
        aliases = self._static_module_aliases(
            node, argument_names=argument_names
        )
        if node.returns is not None:
            # Function calls in annotations execute at Python definition time;
            # keep the module-helper exception scoped to TVMScript bodies.
            self._validate_expr(node.returns, frozenset(local_names))
        previous = (
            self._current_function,
            self._current_function_kind,
            self._current_module_aliases,
        )
        self._current_function = node.name
        self._current_function_kind = required_root
        self._current_module_aliases = aliases
        try:
            for statement in node.body:
                self._validate_statement(statement, frozenset(local_names))
        finally:
            (
                self._current_function,
                self._current_function_kind,
                self._current_module_aliases,
            ) = previous

    @staticmethod
    def _static_module_aliases(
        node: ast.FunctionDef, *, argument_names: set[str]
    ) -> frozenset[str]:
        """Return aliases proven to be a single top-level ``name = Module``.

        A second assignment anywhere in the function makes the name ineligible.
        This covers the exact alias emitted by TVMScript while rejecting branch,
        loop, destructuring, parameter, and reassignment-based receiver tricks.
        """

        stores: Counter[str] = Counter()
        loads: dict[str, list[int]] = {}
        for statement in node.body:
            for child in ast.walk(statement):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    stores[child.id] += 1
                elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    loads.setdefault(child.id, []).append(child.lineno)
        aliases: set[str] = set()
        for statement in node.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Name)
                and statement.value.id == "Module"
                and stores[statement.targets[0].id] == 1
                and statement.targets[0].id not in argument_names
                and all(
                    line > statement.lineno
                    for line in loads.get(statement.targets[0].id, ())
                )
            ):
                aliases.add(statement.targets[0].id)
        return frozenset(aliases)

    def _reject_recursive_relax_calls(self) -> None:
        visited: set[str] = set()
        active: set[str] = set()

        def visit(function: str) -> None:
            if function in visited:
                return
            active.add(function)
            for callee, call in self._relax_calls.get(function, {}).items():
                if callee in active:
                    _reject("recursive Relax helper calls are forbidden", call)
                visit(callee)
            active.remove(function)
            visited.add(function)

        for function in self._relax_calls:
            visit(function)

    def _validate_arguments(self, arguments: ast.arguments) -> set[str]:
        if arguments.vararg is not None or arguments.kwarg is not None:
            _reject("*args and **kwargs are forbidden in TVMScript functions", arguments)
        names: set[str] = set()
        all_args = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        for argument in all_args:
            _check_identifier(argument.arg, argument)
            if argument.arg in _RESERVED_BINDINGS:
                _reject(f"reserved parameter name {argument.arg!r} is forbidden", argument)
            if argument.arg in names:
                _reject(f"duplicate parameter name {argument.arg!r}", argument)
            names.add(argument.arg)
            if argument.annotation is not None:
                self._validate_expr(argument.annotation, frozenset(names))
        for default in arguments.defaults:
            self._validate_expr(default, frozenset(names))
        for default in arguments.kw_defaults:
            if default is not None:
                self._validate_expr(default, frozenset(names))
        return names

    def _validate_builder(self, node: ast.FunctionDef) -> None:
        if node.decorator_list or node.returns is not None or node.type_comment is not None:
            _reject("build_relax may not have decorators or annotations", node)
        if getattr(node, "type_params", ()):  # Python 3.12+
            _reject("build_relax may not be generic", node)
        args = node.args
        if (
            args.posonlyargs
            or args.args
            or args.vararg is not None
            or args.kwonlyargs
            or args.kw_defaults
            or args.kwarg is not None
            or args.defaults
        ):
            _reject("build_relax must take no arguments", node)
        if len(node.body) != 1:
            _reject("build_relax body must contain only 'return Module'", node)
        result = node.body[0]
        if not isinstance(result, ast.Return) or not isinstance(result.value, ast.Name):
            _reject("build_relax body must contain only 'return Module'", result)
        if result.value.id != "Module":
            _reject("build_relax must return Module directly", result.value)

    def _validate_statement(self, node: ast.stmt, locals_: frozenset[str]) -> None:
        if isinstance(node, ast.Expr):
            self._validate_expr(node.value, locals_)
        elif isinstance(node, ast.Assign):
            self._validate_expr(node.value, locals_)
            for target in node.targets:
                self._validate_store_target(target, locals_)
        elif isinstance(node, ast.AnnAssign):
            self._validate_store_target(node.target, locals_)
            self._validate_expr(node.annotation, locals_)
            if node.value is not None:
                self._validate_expr(node.value, locals_)
        elif isinstance(node, ast.AugAssign):
            self._validate_store_target(node.target, locals_)
            self._validate_operator(node.op, node)
            self._validate_expr(node.value, locals_)
        elif isinstance(node, ast.Return):
            if node.value is not None:
                self._validate_expr(node.value, locals_)
        elif isinstance(node, ast.For):
            self._validate_store_target(node.target, locals_)
            self._validate_expr(node.iter, locals_)
            for statement in [*node.body, *node.orelse]:
                self._validate_statement(statement, locals_)
        elif isinstance(node, ast.If):
            self._validate_expr(node.test, locals_)
            for statement in [*node.body, *node.orelse]:
                self._validate_statement(statement, locals_)
        elif isinstance(node, ast.With):
            for item in node.items:
                self._validate_expr(item.context_expr, locals_)
                if item.optional_vars is not None:
                    self._validate_store_target(item.optional_vars, locals_)
            for statement in node.body:
                self._validate_statement(statement, locals_)
        elif isinstance(node, ast.Pass):
            return
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            _reject("global and nonlocal statements are forbidden", node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            _reject("imports inside TVMScript functions are forbidden", node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            _reject("nested functions, classes, lambdas, and async code are forbidden", node)
        else:
            _reject(f"statement {type(node).__name__} is forbidden in TVMScript functions", node)

    def _validate_store_target(self, node: ast.expr, locals_: frozenset[str]) -> None:
        if isinstance(node, ast.Name):
            _check_identifier(node.id, node)
            if node.id in _RESERVED_BINDINGS:
                _reject(f"rebinding reserved name {node.id!r} is forbidden", node)
            return
        if isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                self._validate_store_target(item, locals_)
            return
        if isinstance(node, ast.Subscript):
            self._validate_expr(node.value, locals_)
            self._validate_expr(node.slice, locals_)
            return
        _reject(f"assignment target {type(node).__name__} is forbidden", node)

    def _validate_expr(self, node: ast.expr, locals_: frozenset[str]) -> None:
        if isinstance(node, ast.Constant):
            return
        if isinstance(node, ast.Name):
            _check_identifier(node.id, node)
            if isinstance(node.ctx, ast.Load):
                if node.id in {"tvm", "relax"}:
                    _reject(f"direct use of imported module {node.id!r} is forbidden", node)
                if node.id in _DANGEROUS_GLOBALS and node.id not in locals_:
                    _reject(f"dangerous Python global {node.id!r} is forbidden", node)
                allowed = locals_ | self.dsl_roots | _SAFE_BUILTINS | {"Module"}
                if node.id not in allowed:
                    _reject(f"unbound Python global {node.id!r} is forbidden", node)
            return
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for item in node.elts:
                self._validate_expr(item, locals_)
            return
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if key is None:
                    _reject("dictionary unpacking is forbidden", node)
                self._validate_expr(key, locals_)
                self._validate_expr(value, locals_)
            return
        if isinstance(node, ast.Attribute):
            _check_identifier(node.attr, node)
            root = self._attribute_root(node)
            module_member = self._static_module_member(node)
            if root == "Module" or root in self._current_module_aliases:
                if module_member is None:
                    _reject("dynamic or chained Module attributes are forbidden", node)
                if module_member not in self._module_functions:
                    _reject(
                        f"undeclared Module function {module_member!r} is forbidden",
                        node,
                    )
                self._validate_expr(node.value, locals_)
                return
            lowered = node.attr.lower()
            if root in self.dsl_roots and any(
                marker in lowered for marker in _FORBIDDEN_DSL_CALL_MARKERS
            ):
                _reject(
                    f"external/dynamic DSL primitive {root}.{node.attr} is forbidden",
                    node,
                )
            self._validate_expr(node.value, locals_)
            return
        if isinstance(node, ast.Subscript):
            self._validate_expr(node.value, locals_)
            self._validate_expr(node.slice, locals_)
            return
        if isinstance(node, ast.Slice):
            for part in (node.lower, node.upper, node.step):
                if part is not None:
                    self._validate_expr(part, locals_)
            return
        if isinstance(node, ast.BinOp):
            self._validate_operator(node.op, node)
            self._validate_expr(node.left, locals_)
            self._validate_expr(node.right, locals_)
            return
        if isinstance(node, ast.UnaryOp):
            self._validate_operator(node.op, node)
            self._validate_expr(node.operand, locals_)
            return
        if isinstance(node, ast.BoolOp):
            self._validate_operator(node.op, node)
            for value in node.values:
                self._validate_expr(value, locals_)
            return
        if isinstance(node, ast.Compare):
            for operator in node.ops:
                self._validate_operator(operator, node)
            self._validate_expr(node.left, locals_)
            for comparator in node.comparators:
                self._validate_expr(comparator, locals_)
            return
        if isinstance(node, ast.IfExp):
            self._validate_expr(node.test, locals_)
            self._validate_expr(node.body, locals_)
            self._validate_expr(node.orelse, locals_)
            return
        if isinstance(node, ast.Call):
            self._validate_call(node, locals_)
            return
        if isinstance(node, ast.Lambda):
            _reject("lambda expressions are forbidden", node)
        if isinstance(node, ast.NamedExpr):
            _reject("assignment expressions are forbidden", node)
        if isinstance(node, (ast.Await, ast.Yield, ast.YieldFrom)):
            _reject("async and generator expressions are forbidden", node)
        _reject(f"expression {type(node).__name__} is forbidden", node)

    def _validate_call(self, node: ast.Call, locals_: frozenset[str]) -> None:
        target = node.func
        if isinstance(target, ast.Name):
            _check_identifier(target.id, target)
            if target.id not in _SAFE_BUILTINS or target.id in locals_:
                if target.id in _DANGEROUS_GLOBALS:
                    _reject(f"dangerous builtin call {target.id!r} is forbidden", target)
                _reject("calls are restricted to I/R/T DSL functions and safe builtins", target)
        elif isinstance(target, ast.Attribute):
            module_member = self._static_module_member(target)
            if module_member is not None:
                self._validate_expr(target, locals_)
                if (
                    self._current_function_kind != "R"
                    or self._current_function is None
                    or self._module_functions.get(module_member) != "R"
                ):
                    _reject(
                        "direct Module calls are restricted to declared Relax functions",
                        target,
                    )
                self._relax_calls[self._current_function][module_member] = node
            else:
                root = self._attribute_root(target)
                if root == "tvm":
                    _reject("tvm.* calls are forbidden", target)
                # Validate the complete receiver before reporting a generic target
                # error.  This makes nested tricks such as
                # ``__import__('os').system(...)`` fail at the dangerous primitive.
                self._validate_expr(target, locals_)
                if root not in self.dsl_roots:
                    _reject("calls are restricted to I/R/T DSL functions and safe builtins", target)
        else:
            _reject("dynamic call targets are forbidden", target)
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                _reject("starred call arguments are forbidden", argument)
            if (
                isinstance(target, ast.Attribute)
                and self._attribute_root(target) == "T"
                and target.attr in {"call_intrin", "call_llvm_intrin", "call_llvm_pure_intrin"}
                and isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and any(
                    marker in argument.value.lower()
                    for marker in _FORBIDDEN_DSL_CALL_MARKERS
                )
            ):
                _reject("external/dynamic TIR intrinsic is forbidden", argument)
            self._validate_expr(argument, locals_)
        for keyword in node.keywords:
            if keyword.arg is None:
                _reject("**keyword expansion is forbidden", keyword)
            _check_identifier(keyword.arg, keyword)
            self._validate_expr(keyword.value, locals_)

    def _static_module_member(self, node: ast.Attribute) -> str | None:
        """Resolve only ``Module.name`` or a proven ``alias = Module`` receiver."""

        if not isinstance(node.value, ast.Name):
            return None
        receiver = node.value.id
        if receiver == "Module" or receiver in self._current_module_aliases:
            return node.attr
        return None

    @staticmethod
    def _attribute_root(node: ast.Attribute) -> str | None:
        value: ast.expr = node
        while isinstance(value, ast.Attribute):
            if _is_dunder(value.attr):
                _reject(f"dunder attribute {value.attr!r} is forbidden", value)
            value = value.value
        if isinstance(value, ast.Name):
            return value.id
        return None

    @staticmethod
    def _is_exact_attribute(node: ast.expr, root: str, attribute: str) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == root
            and node.attr == attribute
        )

    @staticmethod
    def _validate_operator(node: ast.AST, owner: ast.AST) -> None:
        allowed = (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.MatMult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.LShift,
            ast.RShift,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
            ast.And,
            ast.Or,
            ast.Invert,
            ast.Not,
            ast.UAdd,
            ast.USub,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Is,
            ast.IsNot,
            ast.In,
            ast.NotIn,
        )
        if not isinstance(node, allowed):
            _reject(f"operator {type(node).__name__} is forbidden", owner)


def validate_candidate_source(source: str) -> None:
    """Validate *source* as side-effect-constrained TVMScript.

    The function returns ``None`` when the source is accepted and raises
    :class:`CandidateSourceError` for syntax errors or any construct outside the
    whitelist.  It never imports or executes the candidate.
    """

    if not isinstance(source, str):
        raise CandidateSourceError("candidate source must be a string")
    try:
        tree = ast.parse(source, filename="<candidate>", mode="exec")
    except (SyntaxError, ValueError) as error:
        detail = error.msg if isinstance(error, SyntaxError) else str(error)
        raise CandidateSourceError(f"candidate is not valid Python syntax: {detail}") from error
    _Validator().validate(tree)


__all__ = ["CandidateSourceError", "validate_candidate_source"]
