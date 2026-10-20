"""Check that L2 source edits preserve kernel entry points.

Kernel names and parameter lists must match the generated source exactly;
Launch geometry remains in the unchanged host stub. Reject interface changes
before compilation and direct them to the appropriate upper layer.
"""

from __future__ import annotations

import re

_SIG_RE = {
    "cuda": re.compile(r'__global__\s+(?:void|__launch_bounds__\([^)]*\)\s*void|void\s+__launch_bounds__\([^)]*\))\s+(\w+)\s*\(([^)]*)\)'),
    "opencl": re.compile(r"__kernel\s+void\s+(\w+)\s*\(([^)]*)\)"),
    "llvmir": re.compile(r"define[^@\n]*@([\w.$-]+)\s*\(([^)]*)\)"),
}

_LLVM_DECLARE_RE = re.compile(
    r"^\s*declare\s+([^@\n]+)@([\w.$-]+)\s*\(([^)]*)\)", re.MULTILINE
)
_LLVM_CALL_RE = re.compile(r"\b(?:call|invoke)\b[^@\n]*@([\w.$-]+)\s*\(")
_LLVM_FORBIDDEN_NATIVE_RE = re.compile(
    r"(?:^\s*module\s+asm\b|\b(?:call|invoke)\b[^\n]*\basm\b|"
    r"@llvm\.global_(?:ctors|dtors)\b|\bifunc\b)",
    re.MULTILINE,
)


def _normalize(params: str) -> str:
    return re.sub(r"\s+", " ", params).strip()


def signatures(lang: str, source: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for name, params in _SIG_RE[lang].findall(source):
        found.setdefault(name, _normalize(params))
    return found


def _llvm_declarations(source: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for result, name, params in _LLVM_DECLARE_RE.findall(source):
        declarations.setdefault(
            name, f"{_normalize(result)} ({_normalize(params)})"
        )
    return declarations


def _check_llvm_native_surface(generated: str, patched: str) -> dict | None:
    """Freeze native escape hatches beyond the exported function ABI.

    L2 is intentionally allowed to rewrite LLVM instruction bodies, but it
    must not grow the module's host/native authority.  New LLVM intrinsics are
    permitted because they lower inside LLVM; new non-intrinsic declarations,
    external call targets, constructors, ifuncs, and inline/module assembly
    are not.
    """

    generated_native_lines = sorted(
        line.strip()
        for line in generated.splitlines()
        if _LLVM_FORBIDDEN_NATIVE_RE.search(line)
    )
    patched_native_lines = sorted(
        line.strip()
        for line in patched.splitlines()
        if _LLVM_FORBIDDEN_NATIVE_RE.search(line)
    )
    if patched_native_lines != generated_native_lines:
        return {
            "ok": False,
            "error": (
                "ABI freeze violation: LLVM patch changes module/inline assembly, "
                "global constructors/destructors, or ifunc native escape hatches. "
                "L2 may rewrite ordinary instructions only; escalate structural/native "
                "changes."
            ),
        }

    generated_declarations = _llvm_declarations(generated)
    patched_declarations = _llvm_declarations(patched)
    added_declarations = sorted(
        name
        for name in set(patched_declarations) - set(generated_declarations)
        if not name.startswith("llvm.")
    )
    if added_declarations:
        return {
            "ok": False,
            "error": (
                "ABI freeze violation: LLVM patch adds external declarations "
                f"{added_declarations}. L2 may not expand native authority."
            ),
        }
    changed_declarations = sorted(
        name
        for name in set(patched_declarations) & set(generated_declarations)
        if patched_declarations[name] != generated_declarations[name]
    )
    if changed_declarations:
        return {
            "ok": False,
            "error": (
                "ABI freeze violation: LLVM external declaration signatures changed "
                f"for {changed_declarations}."
            ),
        }

    generated_definitions = set(signatures("llvmir", generated))
    patched_definitions = set(signatures("llvmir", patched))
    generated_external_calls = {
        name
        for name in _LLVM_CALL_RE.findall(generated)
        if name not in generated_definitions and not name.startswith("llvm.")
    }
    patched_external_calls = {
        name
        for name in _LLVM_CALL_RE.findall(patched)
        if name not in patched_definitions and not name.startswith("llvm.")
    }
    added_calls = sorted(patched_external_calls - generated_external_calls)
    if added_calls:
        return {
            "ok": False,
            "error": (
                "ABI freeze violation: LLVM patch adds external call targets "
                f"{added_calls}. L2 may not expand native authority."
            ),
        }
    return None


def check(lang: str, generated: str, patched: str) -> dict:
    """Returns {ok} or {ok: False, error} with an educational message."""
    gen = signatures(lang, generated)
    pat = signatures(lang, patched)
    if not gen:
        return {"ok": False, "error": f"no {lang} kernel signatures found in generated source "
                                      "(driver-side extraction problem, not your patch)"}
    missing = sorted(set(gen) - set(pat))
    added = sorted(set(pat) - set(gen))
    if missing or added:
        return {"ok": False, "error":
                f"ABI freeze violation: kernel set changed (missing={missing}, added={added}). "
                "L2 may only rewrite kernel bodies; to change kernel structure or launch "
                "geometry, write ESCALATE to the TIR layer."}
    for name in gen:
        if gen[name] != pat[name]:
            return {"ok": False, "error":
                    f"ABI freeze violation: parameter list of kernel '{name}' changed. "
                    f"generated: ({gen[name]}) vs patched: ({pat[name]}). "
                    "Signatures are frozen (host launch stub depends on them); "
                    "write ESCALATE to the TIR layer if the interface must change."}
    if lang == "llvmir":
        native_violation = _check_llvm_native_surface(generated, patched)
        if native_violation is not None:
            return native_violation
    return {"ok": True, "kernels": sorted(gen)}
