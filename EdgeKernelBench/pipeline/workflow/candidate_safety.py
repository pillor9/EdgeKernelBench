"""Restrict generated builders' imports, capabilities, and process resources.

These checks supplement OS isolation; they do not replace a container.
"""

from __future__ import annotations

import ast
import builtins
import math
from dataclasses import dataclass
from typing import Callable, Iterable

from .errors import FaultRecord, FaultStage, PipelineFault


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    timeout_seconds: float = 30.0
    memory_bytes: int = 2 * 1024**3
    cpu_seconds: int = 30
    max_source_bytes: int = 2 * 1024**2
    max_output_bytes: int = 256 * 1024
    max_param_bytes: int = 2 * 1024**3
    max_artifact_bytes: int = 128 * 1024**2

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        for name in (
            "memory_bytes",
            "cpu_seconds",
            "max_source_bytes",
            "max_output_bytes",
            "max_param_bytes",
            "max_artifact_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    allowed_import_roots: tuple[str, ...] = (
        "__future__",
        "tvm",
        "numpy",
        "math",
        "typing",
        "pipeline.interfaces",
    )
    isolation_strength: str = "process_static_capability_gate_v2"


@dataclass(frozen=True, slots=True)
class SafetyViolation:
    code: str
    message: str
    line: int
    column: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "column": self.column,
        }


@dataclass(frozen=True, slots=True)
class SafetyReport:
    policy: str
    build_function_count: int
    violations: tuple[SafetyViolation, ...]

    @property
    def accepted(self) -> bool:
        return not self.violations


_DENIED_IMPORT_ROOTS = {
    "asyncio",
    "builtins",
    "ctypes",
    "http",
    "importlib",
    "inspect",
    "marshal",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "urllib",
}
_DENIED_BUILTIN_CALLS = {
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
_DENIED_FILE_ATTRIBUTES = {
    # Avoid generic method names also used by legal IR or array operations.
    "fromfile",
    "memmap",
    "savez",
    "savez_compressed",
    "tofile",
}
_DENIED_RUNTIME_ATTRIBUTES = {
    # Device/build/timing capabilities, not IR construction syntax.
    "benchmark",
    "build",
    "compile",
    "cpu",
    "create_shared",
    "cuda",
    "device",
    "export_library",
    "get_global_func",
    "load_module",
    "load_static_library",
    "metal",
    "opencl",
    "register_func",
    "rocm",
    "time_evaluator",
    "tune_relay",
    "tune_tasks",
    "tune_tir",
    "vulkan",
}
_DENIED_SCHEDULE_ATTRIBUTES = {
    # Candidates supply scheduled TVMScript without invoking a scheduler.
    "Schedule",
    "ApplyDefaultSchedule",
    "ApplyDatabase",
    "ApplyHistoryBest",
}
_DENIED_TVM_CALLS = {
    "tvm.build",
    "tvm.compile",
    "tvm.cpu",
    "tvm.cuda",
    "tvm.device",
    "tvm.metal",
    "tvm.opencl",
    "tvm.rocm",
    "tvm.vulkan",
    "tvm.runtime.load_module",
    "tvm.runtime.load_static_library",
    "tvm.contrib.cc.create_shared",
    "tvm.contrib.ndk.create_shared",
    "tvm.get_global_func",
    "tvm.register_func",
}
_DENIED_FILE_CALLS = {
    "numpy.fromfile",
    "numpy.load",
    "numpy.memmap",
    "numpy.save",
    "numpy.savez",
    "numpy.savez_compressed",
    "torch.load",
    "torch.save",
    "tvm.contrib.download.download",
    "tvm.contrib.utils.tempdir",
}
_DENIED_SCHEDULE_PREFIXES = (
    "tvm.auto_scheduler",
    "tvm.autotvm",
    "tvm.dlight",
    "tvm.meta_schedule",
)
_DENIED_NETWORK_PREFIXES = (
    "torch.distributed",
    "torch.hub",
    "torch.utils.model_zoo",
    "tvm.contrib.download",
    "tvm.rpc",
)
_DENIED_NETWORK_ATTRIBUTES = {
    "connect",
    "connect_tracker",
    "download",
    "download_url_to_file",
    "load_url",
    "request_remote",
}
_DENIED_RUNTIME_PREFIXES = (
    "torch.cuda",
    "torch.utils.cpp_extension",
)
_DENIED_RUNTIME_CALLS = {
    "torch.compile",
    "torch.device",
}

# Check attribute paths and aliases as well as imports for host capabilities.
_DENIED_HOST_MODULE_SEGMENTS = frozenset(
    {
        "asyncio",
        "builtins",
        "ctypes",
        "ctypeslib",
        "fcntl",
        "http",
        "importlib",
        "inspect",
        "io",
        "marshal",
        "multiprocessing",
        "nt",
        "ops",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "posix",
        "resource",
        "runpy",
        "selectors",
        "serialization",
        "shutil",
        "signal",
        "socket",
        "ssl",
        "subprocess",
        "sys",
        "tarfile",
        "tempfile",
        "urllib",
        "webbrowser",
        "zipfile",
    }
)
_DENIED_HOST_CAPABILITY_PREFIXES = (
    "numpy.ctypeslib",
    "numpy.distutils",
    "numpy.f2py",
    "numpy.lib",
    "numpy.testing",
    "torch.classes",
    "torch.cuda",
    "torch.distributed",
    "torch.export",
    "torch.hub",
    "torch.jit",
    "torch.ops",
    "torch.package",
    "torch.serialization",
    "torch.utils.cpp_extension",
    "torch.utils.model_zoo",
)
_DENIED_HOST_CAPABILITY_ATTRIBUTES = frozenset(
    {
        # Process creation, command execution, and native-code loading.
        "CDLL",
        "LibraryLoader",
        "LoadLibrary",
        "OleDLL",
        "PyDLL",
        "WinDLL",
        "call",
        "check_call",
        "check_output",
        "dlopen",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "fork",
        "forkpty",
        "load_library",
        "popen",
        "posix_spawn",
        "posix_spawnp",
        "spawn",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "system",
        # File, environment, and serialization access.
        "DataSource",
        "NpzFile",
        "chdir",
        "chmod",
        "chown",
        "dump",
        "dumps",
        "environ",
        "fdopen",
        "from_file",
        "fromfile",
        "genfromtxt",
        "getenv",
        "listdir",
        "load",
        "loads",
        "loadtxt",
        "makedirs",
        "memmap",
        "mkdir",
        "open",
        "open_memmap",
        "pread",
        "putenv",
        "pwrite",
        "read",
        "readlink",
        "readv",
        "remove",
        "rmdir",
        "save",
        "savetxt",
        "scandir",
        "symlink",
        "tofile",
        "unlink",
        "unsetenv",
        "walk",
        "write",
        "writev",
        # Network helpers with unambiguous capability-bearing names.
        "urlopen",
        "urlretrieve",
    }
)

# Combined attribute set used by prompt-coverage checks.
_DENIED_CALL_ATTRIBUTES = (
    _DENIED_FILE_ATTRIBUTES
    | _DENIED_RUNTIME_ATTRIBUTES
    | _DENIED_SCHEDULE_ATTRIBUTES
    | _DENIED_NETWORK_ATTRIBUTES
    | _DENIED_HOST_CAPABILITY_ATTRIBUTES
)


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self, policy: SandboxPolicy) -> None:
        self.policy = policy
        self.violations: list[SafetyViolation] = []
        self.aliases: dict[str, str] = {}

    def reject(self, node: ast.AST, code: str, message: str) -> None:
        self.violations.append(
            SafetyViolation(
                code=code,
                message=message,
                line=int(getattr(node, "lineno", 0)),
                column=int(getattr(node, "col_offset", 0)),
            )
        )

    def _allowed_import(self, module: str) -> bool:
        return any(
            module == root or module.startswith(root + ".")
            for root in self.policy.allowed_import_roots
        )

    def _canonical_name(self, node: ast.AST) -> str | None:
        dotted = _dotted_name(node)
        if dotted is None:
            return None
        first, *rest = dotted.split(".")
        canonical = self.aliases.get(first, first)
        return ".".join((canonical, *rest))

    @staticmethod
    def _has_prefix(dotted: str, prefixes: tuple[str, ...]) -> bool:
        return any(dotted == prefix or dotted.startswith(prefix + ".") for prefix in prefixes)

    def _reject_capability(self, node: ast.AST, dotted: str) -> bool:
        """Reject restricted attributes before they can be assigned to aliases."""

        parts = dotted.split(".")
        leaf = parts[-1]
        normalized_parts = tuple(part.lower() for part in parts)
        if any(part.startswith("_") for part in parts[1:]):
            self.reject(
                node,
                "private_attribute",
                f"private module/object paths are not available: {dotted}",
            )
            return True
        schedule_identifier = any(
            part in {"auto_scheduler", "autotvm", "dlight", "meta_schedule", "metaschedule"}
            or part.startswith("metaschedule")
            or part.startswith("dlight")
            for part in normalized_parts
        )
        if (
            self._has_prefix(dotted, _DENIED_SCHEDULE_PREFIXES)
            or schedule_identifier
            or leaf in _DENIED_SCHEDULE_ATTRIBUTES
        ):
            self.reject(
                node,
                "denied_schedule_rewrite",
                "schedule rewriting is not available: write final scheduled "
                f"TVMScript directly instead of acquiring {dotted}",
            )
            return True
        if (
            self._has_prefix(dotted, _DENIED_NETWORK_PREFIXES)
            or "rpc" in normalized_parts
            or leaf in _DENIED_NETWORK_ATTRIBUTES
        ):
            self.reject(
                node,
                "denied_network_capability",
                f"network/RPC capability is not available: {dotted}",
            )
            return True
        if (
            self._has_prefix(dotted, _DENIED_HOST_CAPABILITY_PREFIXES)
            or any(part in _DENIED_HOST_MODULE_SEGMENTS for part in normalized_parts)
            or leaf in _DENIED_HOST_CAPABILITY_ATTRIBUTES
        ):
            self.reject(
                node,
                "denied_host_capability",
                f"host process/file/network capability is not available: {dotted}",
            )
            return True
        if (
            dotted in _DENIED_TVM_CALLS
            or dotted in _DENIED_RUNTIME_CALLS
            or self._has_prefix(dotted, _DENIED_RUNTIME_PREFIXES)
            or leaf in _DENIED_RUNTIME_ATTRIBUTES
        ):
            self.reject(
                node,
                "denied_tvm_capability",
                f"device/compile capability is runner-owned: {dotted}",
            )
            return True
        if dotted in _DENIED_FILE_CALLS or leaf in _DENIED_FILE_ATTRIBUTES:
            self.reject(
                node,
                "denied_file_capability",
                f"host file capability is not available: {dotted}",
            )
            return True
        return False

    def _bind_assignment(self, target: ast.AST, canonical: str | None) -> None:
        if isinstance(target, ast.Name):
            if canonical is None:
                self.aliases.pop(target.id, None)
            else:
                self.aliases[target.id] = canonical
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._bind_assignment(item, None)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in _DENIED_IMPORT_ROOTS or not self._allowed_import(alias.name):
                self.reject(node, "denied_import", f"import is outside sandbox allowlist: {alias.name}")
            elif self._reject_capability(node, alias.name):
                continue
            else:
                # ``import tvm.tir`` binds ``tvm`` while ``import tvm.tir as T``
                # binds ``T`` to the complete module.
                self.aliases[alias.asname or root] = alias.name if alias.asname else root
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".", 1)[0]
        if node.level or root in _DENIED_IMPORT_ROOTS or not self._allowed_import(module):
            self.reject(node, "denied_import", f"import is outside sandbox allowlist: {module or '.'}")
        elif self._reject_capability(node, module):
            pass
        for alias in node.names:
            if alias.name == "*":
                self.reject(node, "wildcard_import", "wildcard imports are not permitted")
                continue
            canonical = f"{module}.{alias.name}"
            if not self._reject_capability(node, canonical):
                self.aliases[alias.asname or alias.name] = canonical
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self.reject(
                node,
                "private_attribute",
                "private or dunder attribute access is not permitted",
            )
        dotted = self._canonical_name(node)
        if dotted is not None:
            self._reject_capability(node, dotted)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        canonical = self._canonical_name(node.value)
        for target in node.targets:
            self._bind_assignment(target, canonical)
            self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.annotation)
        self._bind_assignment(
            node.target,
            None if node.value is None else self._canonical_name(node.value),
        )
        self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind_assignment(node.target, self._canonical_name(node.value))
        self.visit(node.target)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = self._canonical_name(node.func)
        if dotted:
            if dotted in _DENIED_BUILTIN_CALLS:
                self.reject(node, "denied_call", f"call is outside build capability: {dotted}")
            self._reject_capability(node, dotted)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value is Ellipsis:
            self.reject(node, "ellipsis", "incomplete source containing ellipsis is not accepted")


def inspect_source(source: str, policy: SandboxPolicy | None = None) -> SafetyReport:
    active = policy or SandboxPolicy()
    if not isinstance(source, str):
        raise TypeError("source must be text")
    try:
        tree = ast.parse(source, filename="candidate_case.py", mode="exec")
    except SyntaxError as error:
        raise PipelineFault(
            FaultRecord(
                FaultStage.PYTHON_SYNTAX,
                "invalid_python",
                f"{error.msg} at line {error.lineno}, column {error.offset}",
                exception_type="SyntaxError",
            )
        ) from error
    visitor = _SafetyVisitor(active)
    visitor.visit(tree)
    builds = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build"
    ]
    if len(builds) != 1:
        visitor.reject(tree, "build_contract", "source must define exactly one top-level build function")
    elif isinstance(builds[0], ast.AsyncFunctionDef) or (
        len(builds[0].args.args) != 1
        or builds[0].args.posonlyargs
        or builds[0].args.kwonlyargs
        or builds[0].args.vararg is not None
        or builds[0].args.kwarg is not None
    ):
        visitor.reject(builds[0], "build_signature", "build must be synchronous and accept exactly one positional argument")
    return SafetyReport(active.isolation_strength, len(builds), tuple(visitor.violations))


def require_safe_source(
    source: str,
    *,
    policy: SandboxPolicy | None = None,
    limits: SandboxLimits | None = None,
) -> SafetyReport:
    active_limits = limits or SandboxLimits()
    if len(source.encode("utf-8")) > active_limits.max_source_bytes:
        raise PipelineFault(
            FaultRecord(FaultStage.SANDBOX_SECURITY, "source_too_large", "candidate source exceeds sandbox limit")
        )
    report = inspect_source(source, policy)
    if not report.accepted:
        raise PipelineFault(
            FaultRecord(
                FaultStage.SANDBOX_SECURITY,
                "capability_denied",
                "; ".join(item.message for item in report.violations[:5]),
                details={"violations": [item.to_dict() for item in report.violations]},
            )
        )
    return report


_SAFE_BUILTIN_NAMES = {
    "ArithmeticError",
    "AssertionError",
    "Exception",
    "KeyError",
    "RuntimeError",
    "TypeError",
    "ValueError",
    "abs",
    "all",
    "any",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "classmethod",
    "complex",
    "format",
    "dict",
    "enumerate",
    "float",
    "frozenset",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "staticmethod",
    "sum",
    "super",
    "tuple",
    "zip",
}


def restricted_importer(policy: SandboxPolicy) -> Callable[..., object]:
    def restricted_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: Iterable[str] = (),
        level: int = 0,
    ) -> object:
        if level or not any(
            name == root or name.startswith(root + ".")
            for root in policy.allowed_import_roots
        ):
            raise ImportError(f"sandbox import denied: {name}")
        # Delegate the already-authorized import to CPython so ``from x import y``
        # retains its normal submodule-loading semantics.
        return builtins.__import__(name, globals_, locals_, tuple(fromlist or ()), level)

    return restricted_import


def restricted_builtins(policy: SandboxPolicy | None = None) -> dict[str, object]:
    active = policy or SandboxPolicy()
    result = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
    result["__build_class__"] = builtins.__build_class__
    result["__import__"] = restricted_importer(active)
    return result


def apply_process_resource_limits(limits: SandboxLimits) -> None:
    """Apply best-effort POSIX limits; callers still need timeout termination."""

    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1))
        # RLIMIT_AS is unreliable for macOS/Python dylibs and is therefore used
        # only where the platform reports Linux semantics.
        import sys

        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    except (ImportError, OSError, ValueError):
        pass


__all__ = [
    "SafetyReport",
    "SafetyViolation",
    "SandboxLimits",
    "SandboxPolicy",
    "apply_process_resource_limits",
    "inspect_source",
    "require_safe_source",
    "restricted_builtins",
]
