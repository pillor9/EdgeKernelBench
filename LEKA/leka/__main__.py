"""Commands for checking the environment and running LEKA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


_CLIENT_OPTIONS = frozenset({
    "--input-format", "--output-format", "--restricted", "--strict-mcp-config",
    "--tools", "--settings", "--setting-sources", "--append-system-prompt",
    "--name", "--model", "--verbose",
})


def client_capabilities(executable: str, *, extra_options: tuple[str, ...] = ()) -> tuple[bool, str]:
    """Inspect local CLI help without starting a model session."""
    try:
        result = subprocess.run(
            [executable, "--help"], capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, type(error).__name__
    if result.returncode != 0:
        return False, "client help returned a nonzero exit status"
    help_text = result.stdout + "\n" + result.stderr
    options = set(re.findall(r"--[a-z][a-z0-9-]*", help_text))
    missing = sorted((_CLIENT_OPTIONS | frozenset(extra_options)) - options)
    if missing:
        return False, "missing options: " + ", ".join(missing)
    if "stream-json" not in help_text:
        return False, "missing format: stream-json"
    return True, "required CLI options and stream-json are available"


def doctor(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="leka doctor")
    parser.add_argument("--backend", choices=("jetson_gpu", "rk3588_gpu", "vivo_cpu"))
    parser.add_argument("--check-device", action="store_true",
                        help="Also perform a bounded read-only SSH/ADB transport probe; requires --backend")
    args = parser.parse_args(argv)
    if args.check_device and not args.backend:
        parser.error("--check-device requires --backend")
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check("python", sys.version_info >= (3, 11), sys.version.split()[0])
    for name in ("numpy", "torch", "tvm"):
        spec = importlib.util.find_spec(name)
        check(name, spec is not None, spec.origin if spec else "not installed")
    for name in ("git", "claude"):
        override = os.environ.get("LEKA_AGENT_BINARY") if name == "claude" else None
        path = override or shutil.which(name)
        ok = Path(path).is_file() if override else path is not None
        check(name, ok, path or "not on PATH")
        if name == "claude" and ok:
            capable, detail = client_capabilities(path)
            check("model_client_protocol", capable, detail)

    python = os.environ.get("LEKA_TVM_PYTHON", sys.executable)
    probe = (
        "import json,tvm,numpy,torch; from tvm import relax,dlight; "
        "from tvm.script import tir; "
        "print(json.dumps({'tvm':tvm.__version__, 'torch':torch.__version__, "
        "'numpy':numpy.__version__, 'llvm':tvm.support.libinfo().get('LLVM_VERSION')}))"
    )
    try:
        result = subprocess.run([python, "-c", probe], capture_output=True, text=True, timeout=30)
        versions = json.loads(result.stdout.strip()) if result.returncode == 0 else {}
        check("tvm_worker", result.returncode == 0 and versions.get("tvm") == "0.23.0",
              json.dumps(versions) if versions else "TVM 0.23.0 with Relax/DLight is required")
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        check("tvm_worker", False, type(error).__name__)

    import leka
    from leka.backends import BACKENDS

    package_root = Path(leka.__file__).resolve().parent
    wiki_root = package_root / "skills" / "wiki"
    profiles = package_root / "prompts"
    check("knowledge", (wiki_root / "PROVENANCE.json").is_file()
          and all((profiles / f"{role}.md").is_file() for role in
                  ("relax_agent", "tir_agent", "cuda_agent", "opencl_agent", "llvm_agent", "hardware_agent")),
          str(wiki_root))
    if args.backend:
        config = BACKENDS[args.backend]
        commands = ("adb", "llc") if args.backend == "vivo_cpu" else ("ssh", "rsync")
        for name in commands:
            path = shutil.which(name)
            check(name, path is not None, path or "not on PATH")
        if args.backend == "vivo_cpu":
            check("vivo_serial", bool(config["vivo_serial"]), "configured" if config["vivo_serial"] else "set VIVO_SERIAL")
            check("ndk_clang", Path(config["ndk_clang"]).is_file(), "set TVM_NDK_CC to the Android compiler")
    if args.check_device:
        from leka.transport_probe import probe_device
        transport = probe_device(BACKENDS[args.backend])
        check("device_transport", transport["ok"], json.dumps(transport, sort_keys=True))
    scope = ("local checks plus read-only device transport probe; no provider request or device measurement"
             if args.check_device else "local only; no provider request, SSH, ADB or device measurement")
    print(json.dumps({"ok": all(item["ok"] for item in checks), "checks": checks,
                      "scope": scope},
                     ensure_ascii=False, indent=2))
    return 0 if all(item["ok"] for item in checks) else 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="leka", description="LEKA layered kernel optimization")
    parser.add_argument("command", choices=("doctor", "run"))
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    if args.command == "doctor":
        raise SystemExit(doctor(rest))
    from leka.orchestration.optimization_loop import main as run
    sys.argv = ["leka run", *rest]
    run()


if __name__ == "__main__":
    main()
