"""Convert verified job bundles to payloads for device workers.

Payloads preserve the bundle's data and hashes so workers can run without
importing PyTorch or executing the generated ``case.py``. The original bundle
records their provenance. Candidate and baseline retain separate entry names,
parameter orders, and packing.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterator, Mapping, Sequence

import numpy as np

from .tensor_utils import as_c_contiguous

from pipeline.interfaces import NamedArray

from .job_bundle import JobBundleContents, JobBundleReader
from .numerical_policy import contract_from_metadata


@dataclass(frozen=True, slots=True)
class NativeFlatPayload:
    root: Path
    candidate_mod_json: Path
    candidate_abi_json: Path
    candidate_params_npz: Path
    inputs_by_seed: Mapping[int, Path]
    references_by_seed: Mapping[int, Path]
    candidate_common_manifest_sha256: str
    correctness_contract: Mapping[str, object]
    baseline_mod_json: Path | None = None
    baseline_abi_json: Path | None = None
    baseline_params_npz: Path | None = None
    baseline_common_manifest_sha256: str | None = None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_identity(item: NamedArray) -> tuple[str, tuple[int, ...], str, bytes]:
    value = as_c_contiguous(item.value)
    return item.name, tuple(int(dim) for dim in value.shape), value.dtype.str, value.tobytes()


def _require_same_hidden_data(
    candidate: JobBundleContents, baseline: JobBundleContents
) -> None:
    if len(candidate.hidden_cases) != len(baseline.hidden_cases):
        raise ValueError("candidate/baseline hidden seed counts differ")
    for left, right in zip(candidate.hidden_cases, baseline.hidden_cases, strict=True):
        if left.seed != right.seed:
            raise ValueError("candidate/baseline hidden seed identities differ")
        if tuple(map(_array_identity, left.inputs)) != tuple(
            map(_array_identity, right.inputs)
        ):
            raise ValueError("candidate/baseline hidden input bytes differ")
        if tuple(map(_array_identity, left.reference)) != tuple(
            map(_array_identity, right.reference)
        ):
            raise ValueError("candidate/baseline reference bytes differ")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _write_npz(path: Path, values: Sequence[NamedArray]) -> None:
    arrays = {item.name: as_c_contiguous(item.value) for item in values}
    if len(arrays) != len(values):
        raise ValueError(f"duplicate named array while materializing {path.name}")
    np.savez(path, **arrays)


@contextmanager
def materialize_native_flat_payload(
    candidate_bundle_path: str | Path,
    *,
    baseline_bundle_path: str | Path | None = None,
) -> Iterator[NativeFlatPayload]:
    """Verify common bundles and yield a temporary device-worker payload."""

    candidate_path = Path(candidate_bundle_path).resolve(strict=True)
    candidate = JobBundleReader(candidate_path).verify()
    contract = contract_from_metadata(candidate.metadata)
    baseline_path = (
        None
        if baseline_bundle_path is None
        else Path(baseline_bundle_path).resolve(strict=True)
    )
    baseline = None if baseline_path is None else JobBundleReader(baseline_path).verify()
    if baseline is not None:
        if contract_from_metadata(baseline.metadata) != contract:
            raise ValueError("candidate/baseline original-task correctness contracts differ")
        _require_same_hidden_data(candidate, baseline)

    with tempfile.TemporaryDirectory(prefix="pipeline-device-payload-") as temporary:
        root = Path(temporary)
        candidate_mod = root / "candidate_mod.json"
        candidate_abi = root / "candidate_abi.json"
        candidate_params = root / "candidate_params.npz"
        candidate_mod.write_text(candidate.module_json, encoding="utf-8")
        _write_json(candidate_abi, candidate.build_result_payload["abi"])
        _write_npz(candidate_params, candidate.params)

        inputs_by_seed: dict[int, Path] = {}
        references_by_seed: dict[int, Path] = {}
        for case in candidate.hidden_cases:
            input_path = root / f"inputs_seed_{case.seed}.npz"
            reference_path = root / f"reference_seed_{case.seed}.npz"
            _write_npz(input_path, case.inputs)
            _write_npz(reference_path, case.reference)
            inputs_by_seed[case.seed] = input_path
            references_by_seed[case.seed] = reference_path

        baseline_mod: Path | None = None
        baseline_abi: Path | None = None
        baseline_params: Path | None = None
        baseline_manifest_sha: str | None = None
        if baseline is not None and baseline_path is not None:
            baseline_mod = root / "baseline_mod.json"
            baseline_abi = root / "baseline_abi.json"
            baseline_params = root / "baseline_params.npz"
            baseline_mod.write_text(baseline.module_json, encoding="utf-8")
            _write_json(baseline_abi, baseline.build_result_payload["abi"])
            _write_npz(baseline_params, baseline.params)
            baseline_manifest_sha = _file_sha256(
                baseline_path / "job_manifest.json"
            )

        yield NativeFlatPayload(
            root=root,
            candidate_mod_json=candidate_mod,
            candidate_abi_json=candidate_abi,
            candidate_params_npz=candidate_params,
            inputs_by_seed=inputs_by_seed,
            references_by_seed=references_by_seed,
            candidate_common_manifest_sha256=_file_sha256(
                candidate_path / "job_manifest.json"
            ),
            correctness_contract=contract,
            baseline_mod_json=baseline_mod,
            baseline_abi_json=baseline_abi,
            baseline_params_npz=baseline_params,
            baseline_common_manifest_sha256=baseline_manifest_sha,
        )


__all__ = ["NativeFlatPayload", "materialize_native_flat_payload"]
