"""Bounded agent-facing diagnostics with ownership and repair context.

``error_feedback`` identifies the responsible layer and repair location.
``lines`` contains a bounded compiler excerpt and remains untrusted data.
Tracebacks, paths, credentials, and candidate-authored exception text cannot
supply instructions to agents.
"""

from __future__ import annotations

import re

MAX_BYTES = 2048
MAX_LINES = 24
BEGIN = "UNTRUSTED_DIAGNOSTIC_BEGIN"
END = "UNTRUSTED_DIAGNOSTIC_END"

_QUOTED_PATH_RE = re.compile(r'''(?P<quote>["'])(?P<path>/[^"'\n]+|[A-Za-z]:[\\/][^"'\n]+)(?P=quote)''')
_PATH_RE = re.compile(r"(?<![\w:/])(?:/(?:[\w.@+~=\-]+/)+[\w.@+~=\-]*|[A-Za-z]:[\\/](?:[\w.@+~=\-]+[\\/])*[\w.@+~=\-]+)")
_SPACED_SOURCE_PATH_RE = re.compile(
    r'''(?<![\w:/])/(?:Users|home|private|tmp|data|opt|usr|var)/[^\n\t"'<>]*?/'''
    r"[\w.\-]+\.(?:py|cc|cpp|c|h|hpp|cu|cl|ll|so|dylib|dll|o|obj|bc|ptx|cubin|json|log|csv|txt)\b"
)
_ERROR_LINE_RE = re.compile(
    r"^\s*(?:\S*(?:Error|Exception)\s*:|LLVM ERROR:|"
    r"[^\n]*?\b(?:fatal error|error)\s*:|ptxas\s+fatal\s*:)", re.MULTILINE
)
_NATIVE_STACK_FRAME_RE = re.compile(r"^\s*#\d+\s+(?:0x[0-9a-fA-F]+\b|\S+\s*\()")
_EXCEPTION_TYPE_RE = re.compile(
    r"(?m)^\s*(?:[A-Za-z_][A-Za-z0-9_.]*\.)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\s*:"
)
_LOCATION_PATTERNS = (
    re.compile(
        r'(?P<file>(?:relax_mod|scheduled_tir|canonical_tir|candidate_module|'
        r'candidate_case|case|patched|generated)\.(?:py|cu|cl|ll))'
        r'(?::(?P<line>[1-9][0-9]{0,6}))'
    ),
    re.compile(
        r'(?m)^\s*File\s+["\'](?:[^"\'\n]*/)?'
        r'(?P<file>(?:relax_mod|scheduled_tir|canonical_tir|candidate_module|'
        r'candidate_case|case|patched|generated)\.(?:py|cu|cl|ll))["\'],'
        r'\s+line\s+(?P<line>[1-9][0-9]{0,6})'
    ),
)
_KNOWN_EXCEPTION_TYPES = frozenset(
    {
        "AssertionError",
        "AttributeError",
        "DiagnosticError",
        "IndexError",
        "InternalError",
        "KeyError",
        "RuntimeError",
        "SyntaxError",
        "TVMError",
        "TypeError",
        "ValueError",
    }
)
_SECRET_PATTERNS = (
    (
        re.compile(
            r"(?i)\bsk-[A-Za-z0-9._-]{8,}\b"
        ),
        "<redacted-provider-token>",
    ),
    (
        re.compile(
            r"(?i)\b((?:[A-Z][A-Z0-9_]*_(?:TOKEN|KEY|SECRET|PASSWORD))[\"']?\s*[=:]\s*[\"']?)"
            r"([^\s'\",}]+)"
        ),
        r"\1<redacted>",
    ),
    (
        re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]+"),
        r"\1<redacted>",
    ),
)

_STAGE_TO_LAYER = {
    "exec_l0": "L0",
    "expected_l0_policy": "L0",
    "l0_contract": "L0",
    "relax_lowering": "L0",
    "dlight_default": "L0",
    "relax_to_tir_worker": "L0",
    "t01_protocol": "L0",
    "t01": "L0",
    "parse": "L1",
    "compile": "lowest_edited",
    "compile_reference": "infra",
    "load_patch": "L2",
    "init": "infra",
    "rpc": "infra",
    "device_setup": "infra",
    "run_correctness": "lowest_edited",
    "reference_correctness": "infra",
    "timing": "infra",
    "profile": "infra",
    "transport": "infra",
    "vivo_worker": "infra",
}

_ARTIFACT_OF_LAYER = {
    "L0": "relax_mod.py",
    "L1": "scheduled_tir.py",
    "L2": "patched.<backend-source-extension>",
}

_SIGNATURE_RULES = (
    (re.compile(r"^\s*LLVM ERROR:", re.M), "llvm.compiler_error"),
    (re.compile(r"Unexpected type for TIR Arg", re.I), "tvm.tir_arg.unexpected_type"),
    (re.compile(r"Undefined variable|variable is not defined", re.I), "tvm.undefined_variable"),
    (re.compile(r"Block frame or PrimFunc frame not find", re.I), "tvm.tir.frame_missing"),
    (re.compile(r"\b(?:buffer|shape)\b", re.I), "candidate.shape_or_buffer"),
    (re.compile(r"\bdtype\b", re.I), "candidate.dtype"),
    (re.compile(r"CL_INVALID_WORK_(?:GROUP_SIZE|ITEM_SIZE)", re.I), "opencl.invalid_work_group"),
    (re.compile(r"CL_OUT_OF_RESOURCES|CL_MEM_OBJECT_ALLOCATION_FAILURE", re.I), "opencl.out_of_resources"),
    (re.compile(r"CL_BUILD_PROGRAM_FAILURE|OpenCL build error", re.I), "opencl.build_program_failure"),
    (re.compile(r"entry .* differs|ABI|kernel .* changed", re.I), "candidate.abi_freeze"),
)


def owning_layer(fault_stage: str | None, lowest_edited: str = "L1") -> str:
    layer = _STAGE_TO_LAYER.get(fault_stage or "", "infra")
    return lowest_edited if layer == "lowest_edited" else layer


def _redact(raw: str) -> str:
    for pattern, replacement in _SECRET_PATTERNS:
        raw = pattern.sub(replacement, raw)
    return raw


def scrub_report_text(raw: str, *, path_replacer=None) -> str:
    """Scrub diagnostic data, never command/launcher operational interfaces.

    An optional callback gives hardware reports opaque path aliases. No numeric
    parsing, unit conversion, truncation or whitespace normalization occurs here.
    """
    replace = path_replacer or (lambda value: "<path>/" + value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1])
    raw = _redact(raw)
    # Quoted paths can contain spaces; the source-file pattern also covers the
    # usual unquoted compiler location with a space in a project/user name.
    raw = _QUOTED_PATH_RE.sub(lambda m: m["quote"] + replace(m["path"]) + m["quote"], raw)
    raw = _SPACED_SOURCE_PATH_RE.sub(lambda m: replace(m[0]), raw)
    return _PATH_RE.sub(lambda m: replace(m[0]), raw)


def sanitize_report_text(value: object, *, max_chars: int = 1200) -> str:
    """Redact and bound model-authored HELP/ESCALATE prose before relaying it."""

    text = scrub_report_text(str(value or ""))
    text = " ".join(text.replace("\x00", "").split())
    return text[:max_chars]


def _signature(raw: str, fault_stage: str | None) -> str:
    # Native backtraces name APIs such as MemoryBufferRef; these are not a
    # compiler diagnosis of the candidate's shape or dtype.
    raw = "\n".join(line for line in raw.splitlines()
                    if not _NATIVE_STACK_FRAME_RE.match(line))
    for pattern, signature in _SIGNATURE_RULES:
        if pattern.search(raw):
            return signature
    return {
        "exec_l0": "relax.execution",
        "expected_l0_policy": "relax.contract",
        "l0_contract": "relax.contract",
        "relax_lowering": "relax.lowering",
        "dlight_default": "tir.default_schedule",
        "relax_to_tir_worker": "relax_to_tir.worker",
        "t01_protocol": "relax_to_tir.protocol",
        "t01": "relax_to_tir.lowering",
        "parse": "tvm.script.parse",
        "compile": "tvm.compile",
        "compile_reference": "infra.reference_compile",
        "load_patch": "source.abi_freeze",
        "init": "infra.worker_init",
        "rpc": "infra.rpc",
        "device_setup": "infra.device_setup",
        "transport": "infra.transport",
        "vivo_worker": "infra.vivo_worker",
        "run_correctness": "candidate.correctness",
        "reference_correctness": "infra.reference_correctness",
        "timing": "infra.timing",
        "profile": "infra.profile",
    }.get(fault_stage or "", "unknown")


def _primary_location(raw: str, edited_layer: str) -> dict | None:
    matches = []
    for pattern in _LOCATION_PATTERNS:
        matches.extend(pattern.finditer(raw))
    if not matches:
        return None
    match = max(matches, key=lambda item: item.start())
    filename = match.group("file")
    if filename in {"candidate_module.py", "candidate_case.py", "case.py"}:
        filename = _ARTIFACT_OF_LAYER.get(edited_layer, filename)
    return {"file": filename, "line": int(match.group("line"))}


def _feedback(raw: str, fault_stage: str | None, edited_layer: str) -> dict:
    owner = owning_layer(fault_stage, edited_layer)
    exception_matches = list(_EXCEPTION_TYPE_RE.finditer(raw))
    exception_type = None
    if exception_matches:
        candidate = exception_matches[-1].group("name")
        exception_type = candidate if candidate in _KNOWN_EXCEPTION_TYPES else "other"
    location = _primary_location(raw, edited_layer)
    can_repair = owner in _ARTIFACT_OF_LAYER
    return {
        "schema_version": "leka.error_feedback.v1",
        "signature": _signature(raw, fault_stage),
        "fault_stage": fault_stage,
        "owning_layer": owner,
        "exception_type": exception_type,
        "primary_location": location,
        "editable_artifact": _ARTIFACT_OF_LAYER.get(owner),
        "agent_can_repair": can_repair,
        "next_protocol": (
            "edit_then_measure_t0"
            if owner == edited_layer and can_repair
            else "ESCALATE.json"
            if can_repair
            else "HELP.json"
        ),
    }


def bounded_error_excerpt(raw: str, *, max_bytes: int, max_lines: int) -> tuple[str, bool]:
    """Retain the innermost error anchor before trimming diagnostic context.

    This selects data, not a proven cause. In particular LLVM fatal headers
    precede native stacks, whereas Python exceptions normally follow frames.
    Worker capture may call this before the separate public redaction step.
    """
    if max_bytes < 1 or max_lines < 1:
        raise ValueError("positive diagnostic bounds required")
    lines = raw.splitlines()
    if not lines:
        return "", False
    anchors = [i for i, line in enumerate(lines) if _ERROR_LINE_RE.search(line)]
    if not anchors:
        tail = "\n".join(lines[-max_lines:]).encode("utf-8")
        return (tail[-max_bytes:].decode("utf-8", errors="ignore"),
                len(lines) > max_lines or len(tail) > max_bytes)
    anchor = anchors[-1]
    start = max(0, anchor - min(6, max_lines - 1))
    end = min(len(lines), start + max_lines)
    # Context cannot consume the byte allowance before the error is reached.
    while start < anchor and len("\n".join(lines[start:anchor + 1]).encode("utf-8")) > max_bytes:
        start += 1
    selected = "\n".join(lines[start:end]).encode("utf-8")
    truncated = start > 0 or end < len(lines) or len(selected) > max_bytes
    return selected[:max_bytes].decode("utf-8", errors="ignore"), truncated


def sanitize(raw: str | None, fault_stage: str | None, *, lowest_edited: str = "L1") -> dict:
    """Innermost-error excerpt, paths scrubbed, bounded, wrapped as data-not-instructions."""
    if not raw:
        return {
            "owning_layer": owning_layer(fault_stage, lowest_edited),
            "fault_stage": fault_stage,
            "error_feedback": _feedback("", fault_stage, lowest_edited),
            "lines": [],
        }
    raw = _redact(raw)
    text, truncated = bounded_error_excerpt(
        scrub_report_text(raw), max_bytes=MAX_BYTES, max_lines=MAX_LINES)
    return {
        "owning_layer": owning_layer(fault_stage, lowest_edited),
        "fault_stage": fault_stage,
        "error_feedback": _feedback(raw, fault_stage, lowest_edited),
        "lines": [BEGIN, *text.splitlines(), END],
        "truncated": truncated,
    }
