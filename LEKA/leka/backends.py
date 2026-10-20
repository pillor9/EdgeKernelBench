"""Targets, measurement policies, and connection settings for bundled devices.

Host names, device identifiers, and tool locations come from the environment.
"""

import os
from pathlib import Path
import tempfile


PACKAGE_ROOT = Path(__file__).resolve().parent


def _setting(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


BACKENDS = {
    "jetson_gpu": {
        "target": "cuda -arch=sm_87",
        "l2_ext": "cu",
        "l2_lang": "cuda",
        "l2_agent": "cuda_agent",
        "runner": "ssh",
        "ssh_alias": _setting("LEKA_JETSON_SSH_ALIAS", "jetson"),
        "device": "cuda",
        "worker_path_prefix": "PATH=/usr/local/cuda/bin:$PATH ",
        "hw_sense": "hw_sense_jetson",
        "gpu": True,
        # Measure paired calls to the saved Relax entry.
        "timing_protocol": "saved_relax_entry_ab_ba_v1",
        "timing_scored_pair_count": 20,
        "timing_evaluator_target_window_ms": 100,
        "timing_evaluator_maximum_number": 1000,
        "timing_minimum_warmup_invocations": 1,
    },
    "rk3588_gpu": {
        "target": "opencl -device=mali",
        "l2_ext": "cl",
        "l2_lang": "opencl",
        "l2_agent": "opencl_agent",
        "runner": "ssh",
        "ssh_alias": _setting("LEKA_RK3588_SSH_ALIAS", "rk3588"),
        "device": "opencl",
        "worker_path_prefix": "",
        "hw_sense": "hw_sense_rk3588",
        "gpu": True,
        "timing_protocol": "saved_relax_entry_ab_ba_v1",
        "timing_scored_pair_count": 100,
        "timing_evaluator_target_window_ms": 100,
        # The Mali g24p0 driver is only stable at one dispatch per evaluator.
        "timing_evaluator_maximum_number": 1,
        "timing_minimum_warmup_invocations": 10,
    },
    "vivo_cpu": {
        "target": "llvm -keys=arm_cpu,cpu -mcpu=generic -mtriple=aarch64-linux-android -num-cores=8",
        "l2_ext": "ll",
        "l2_lang": "llvmir",
        "l2_agent": "llvm_agent",
        "runner": "vivo_rpc",
        "rpc_port": int(_setting("VIVO_RPC_PORT", "9490")),
        "device": "cpu",
        "hw_sense": "hw_sense_vivo",
        "gpu": False,
        "runtime_num_threads": 8,
        # TVM kBig policy; the requested thread count is not a core count.
        "runtime_affinity_mode": 1,
        "runtime_policy_id": "vivo_cpu_8threads_affinity_mode1_v1",
        "timing_protocol": "saved_relax_entry_ab_ba_v1",
        "timing_scored_pair_count": 20,
        "timing_evaluator_target_window_ms": 100,
        "timing_evaluator_maximum_number": 1000,
        "timing_minimum_warmup_invocations": 1,
        "ndk_clang": _setting("TVM_NDK_CC"),
        "vivo_serial": _setting("VIVO_SERIAL"),
        "rpc_refresh_script": str(PACKAGE_ROOT / "runtime" / "vivo_refresh_rpc.sh"),
        "rpc_state_dir": _setting(
            "LEKA_VIVO_STATE_DIR", str(Path(tempfile.gettempdir()) / "LEKA" / "vivo_rpc")
        ),
        "device_lib_dir": _setting("LEKA_VIVO_DEVICE_LIB_DIR", "/data/local/tmp/vivo_tvm"),
    },
}
