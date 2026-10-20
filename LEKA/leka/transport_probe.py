"""Bounded, read-only device transport checks. No provider or measurement calls."""
from __future__ import annotations
import math
import re
import subprocess
from leka.ssh_transport import network_unavailable_marker


def probe_device(config: dict, *, timeout_s: float = 10) -> dict:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
        raise ValueError("probe timeout must be finite, positive and at most 60 seconds")
    runner = config.get("runner")
    if runner == "ssh":
        alias = config.get("ssh_alias", "")
        if not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z0-9_.@:\[\]-]+", alias) or alias.startswith("-"):
            return {"ok": False, "category": "configuration_error", "transport": "ssh"}
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", alias, "true"]
    elif runner == "vivo_rpc":
        serial = config.get("vivo_serial", "")
        if not isinstance(serial, str) or not serial.strip():
            return {"ok": False, "category": "configuration_error", "transport": "adb"}
        command = ["adb", "-s", serial, "get-state"]
    else:
        return {"ok": False, "category": "unsupported_transport", "transport": str(runner)}
    transport = "ssh" if runner == "ssh" else "adb"
    result = {"ok": False, "transport": transport, "scope": "transport_only; no runtime, computation or provider validation"}
    try:
        proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {**result, "category": "probe_deadline_exceeded", "timeout_s": timeout_s}
    except FileNotFoundError:
        return {**result, "category": "local_tool_missing"}
    except OSError:
        return {**result, "category": "local_execution_error"}
    result["returncode"] = proc.returncode
    if proc.returncode == 0:
        if runner == "vivo_rpc" and proc.stdout.strip() != "device":
            return {**result, "category": "device_not_ready"}
        return {**result, "ok": True, "category": "connected"}
    stderr = proc.stderr or ""
    if runner == "ssh":
        reason = network_unavailable_marker(proc)
        if reason is not None:
            return {**result, "category": "network_unavailable", "network_reason": reason}
        if proc.returncode == 255 and re.search(r"(?:^|\n)(?:[^\n]*@[^\n]*: )?Permission denied \([^\n]+\)\.?$", stderr.strip(), re.I):
            return {**result, "category": "authentication_failed"}
        if proc.returncode == 255 and re.search(r"(?:^|\n)Host key verification failed\.", stderr, re.I):
            return {**result, "category": "host_key_verification_failed"}
        if proc.returncode == 255 and re.search(r"(?:^|\n)ssh: Could not resolve hostname [^\n]+", stderr):
            return {**result, "category": "hostname_resolution_failed"}
    elif "unauthorized" in stderr.lower():
        return {**result, "category": "device_unauthorized"}
    elif "offline" in stderr.lower():
        return {**result, "category": "device_offline"}
    return {**result, "category": "transport_error_unclassified"}
