"""No-clobber, hash-verified host/device job bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from .tensor_utils import as_c_contiguous
from .numerical_policy import (
    CONTRACT_KEY, CONTRACT_SHA_KEY, contract_from_metadata,
    correctness_contract_sha256, validate_correctness_contract,
)

from pipeline.interfaces import BuildResult, JSONValue, NamedArray, RuntimePolicy
from pipeline.interfaces.tensor_spec import _json_value

from .task_loader import (
    CANONICAL_CORRECTNESS_SEEDS,
    HiddenCase,
    correctness_input_identities,
)
from .ir_validation import TVMValidationReport


JOB_SCHEMA = "pipeline.job_bundle.v1"
# Exactly the original unscaled task draws; no extra stress population.
FIXED_CORRECTNESS_SEEDS = CANONICAL_CORRECTNESS_SEEDS
INPUT_IDENTITIES_METADATA_KEY = "correctness_input_identity_sha256_by_seed"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    normalized = _json_value(value)
    return (
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_secret_keys(value: object, *, path: str = "$") -> None:
    secret_keys = {"api_key", "access_token", "authorization", "password", "secret", "bearer"}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in secret_keys:
                raise ValueError(f"secret-bearing metadata key is not artifact-safe: {path}.{key}")
            _reject_secret_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _reject_secret_keys(child, path=f"{path}[{index}]")


def _array_digest(value: np.ndarray) -> str:
    array = as_c_contiguous(value)
    identity = (
        array.dtype.str.encode("ascii")
        + b"\0"
        + json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        + b"\0"
        + array.tobytes(order="C")
    )
    return _sha256(identity)


def _npz_bytes(items: Sequence[NamedArray]) -> tuple[bytes, list[dict[str, JSONValue]]]:
    arrays: dict[str, np.ndarray] = {}
    manifest: list[dict[str, JSONValue]] = []
    for index, item in enumerate(items):
        key = f"array_{index:03d}"
        value = as_c_contiguous(item.value)
        arrays[key] = value
        manifest.append(
            {
                "key": key,
                "name": item.name,
                "shape": [int(dim) for dim in value.shape],
                "dtype": value.dtype.name,
                "nbytes": int(value.nbytes),
                "value_sha256": _array_digest(value),
            }
        )
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue(), manifest


def _safe_file(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"bundle member is not a regular no-symlink file: {path.name}")


@dataclass(frozen=True, slots=True)
class JobBundle:
    path: Path
    manifest: Mapping[str, JSONValue]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class JobBundleContents:
    path: Path
    case_source: str
    module_json: str
    build_result_payload: Mapping[str, JSONValue]
    params: tuple[NamedArray, ...]
    hidden_cases: tuple[HiddenCase, ...]
    metadata: Mapping[str, JSONValue]
    manifest: Mapping[str, JSONValue]

    def to_build_result(self) -> BuildResult:
        import tvm

        abi = self.build_result_payload["abi"]
        assert isinstance(abi, Mapping)
        policy = abi.get("runtime_policy")
        profile_id = policy.get("profile_id") if isinstance(policy, Mapping) else None
        return BuildResult(
            mod=tvm.ir.load_json(self.module_json),
            params=self.params,
            input_names=tuple(str(name) for name in abi["input_names"]),
            entry_name=str(abi["entry_name"]),
            runtime_policy=None if profile_id is None else RuntimePolicy(str(profile_id)),
            notes=dict(abi.get("notes", {})),
        )


class JobBundleWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _atomic_write(self, name: str, data: bytes) -> None:
        if "/" in name or "\\" in name or name in {"", ".", ".."}:
            raise ValueError(f"unsafe bundle member name: {name!r}")
        temporary = self.path / f".{name}.tmp.{uuid.uuid4().hex}"
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path / name)

    def write(
        self,
        *,
        case_source: str,
        validation: TVMValidationReport,
        build_result: BuildResult,
        hidden_cases: Sequence[HiddenCase],
        parameter_seed: int,
        metadata: Mapping[str, JSONValue] | None = None,
        correctness_contract: Mapping[str, Any] | None = None,
    ) -> JobBundle:
        if not case_source.strip():
            raise ValueError("case_source must be non-empty")
        if tuple(case.seed for case in hidden_cases) != FIXED_CORRECTNESS_SEEDS:
            raise ValueError(
                f"job bundle requires correctness seeds {FIXED_CORRECTNESS_SEEDS}, got "
                f"{tuple(case.seed for case in hidden_cases)}"
            )
        input_identities = correctness_input_identities(hidden_cases)
        metadata_value = {} if metadata is None else dict(metadata)
        if CONTRACT_KEY in metadata_value or CONTRACT_SHA_KEY in metadata_value:
            raise ValueError("correctness contract metadata is writer-reserved")
        contract = None
        if correctness_contract is not None:
            contract = validate_correctness_contract(
                correctness_contract,
                expected_task_source_sha256=metadata_value.get("task_source_sha256"),
            )
            metadata_value["task_source_sha256"] = contract["task_source_sha256"]
            metadata_value[CONTRACT_KEY] = contract
            metadata_value[CONTRACT_SHA_KEY] = correctness_contract_sha256(contract)
        if INPUT_IDENTITIES_METADATA_KEY in metadata_value:
            raise ValueError(
                f"metadata key {INPUT_IDENTITIES_METADATA_KEY!r} is writer-reserved"
            )
        metadata_value[INPUT_IDENTITIES_METADATA_KEY] = {
            str(seed): identity for seed, identity in input_identities.items()
        }
        _reject_secret_keys(metadata_value)
        normalized_metadata = _json_value(metadata_value)
        assert isinstance(normalized_metadata, dict)
        if not self.path.parent.exists() or not self.path.parent.is_dir():
            raise FileNotFoundError("job bundle parent directory must already exist")
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(f"job bundle is no-clobber: {self.path}")
        self.path.mkdir(mode=0o700)

        file_payloads: dict[str, bytes] = {
            "case.py": case_source.encode("utf-8"),
            "module.json": validation.module_json.encode("utf-8"),
        }
        array_sets: dict[str, JSONValue] = {}
        params_bytes, params_manifest = _npz_bytes(build_result.params)
        file_payloads["params.npz"] = params_bytes
        array_sets["params.npz"] = params_manifest
        for case in hidden_cases:
            input_name = f"inputs_seed_{case.seed}.npz"
            reference_name = f"reference_seed_{case.seed}.npz"
            input_bytes, input_manifest = _npz_bytes(case.inputs)
            reference_bytes, reference_manifest = _npz_bytes(case.reference)
            file_payloads[input_name] = input_bytes
            file_payloads[reference_name] = reference_bytes
            array_sets[input_name] = input_manifest
            array_sets[reference_name] = reference_manifest

        build_payload: dict[str, JSONValue] = {
            "schema_version": "pipeline.build_result.v1",
            "parameter_seed": int(parameter_seed),
            "abi": build_result.abi_manifest(),
            "validation": validation.to_manifest(),
            "module_file": "module.json",
            "params_file": "params.npz",
        }
        if contract is not None:
            # A second, file-hash-bound copy prevents metadata-only rebinding.
            build_payload[CONTRACT_KEY] = contract
            build_payload[CONTRACT_SHA_KEY] = correctness_contract_sha256(contract)
        file_payloads["build_result.json"] = _json_bytes(build_payload)
        for name, data in file_payloads.items():
            self._atomic_write(name, data)

        file_manifest: dict[str, JSONValue] = {
            name: {"sha256": _sha256(data), "size_bytes": len(data)}
            for name, data in sorted(file_payloads.items())
        }
        manifest: dict[str, JSONValue] = {
            "schema_version": JOB_SCHEMA,
            "parameter_seed": int(parameter_seed),
            "correctness_seeds": list(FIXED_CORRECTNESS_SEEDS),
            "files": file_manifest,
            "array_sets": array_sets,
            "metadata": normalized_metadata,
        }
        manifest_bytes = _json_bytes(manifest)
        # The manifest is the commit marker and is always written last.
        self._atomic_write("job_manifest.json", manifest_bytes)
        return JobBundle(
            path=self.path.resolve(),
            manifest=manifest,
            manifest_sha256=_sha256(manifest_bytes),
        )


class JobBundleReader:
    def __init__(
        self,
        path: str | Path,
        *,
        expected_manifest_sha256: str | None = None,
        max_total_bytes: int = 4 * 1024**3,
    ) -> None:
        self.path = Path(path)
        self.expected_manifest_sha256 = expected_manifest_sha256
        self.max_total_bytes = max_total_bytes

    def _read(self, name: str) -> bytes:
        if "/" in name or "\\" in name or name in {"", ".", ".."}:
            raise ValueError(f"unsafe manifest member: {name!r}")
        path = self.path / name
        _safe_file(path)
        return path.read_bytes()

    def verify(self) -> JobBundleContents:
        if self.path.is_symlink() or not self.path.is_dir():
            raise ValueError("job bundle root must be a real directory")
        manifest_bytes = self._read("job_manifest.json")
        if (
            self.expected_manifest_sha256 is not None
            and _sha256(manifest_bytes) != self.expected_manifest_sha256
        ):
            raise ValueError("job manifest differs from the expected transport identity")
        manifest = json.loads(manifest_bytes)
        _json_value(manifest)
        if manifest.get("schema_version") != JOB_SCHEMA:
            raise ValueError("unsupported job bundle schema")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("job manifest files must be an object")
        expected = {
            "case.py",
            "module.json",
            "build_result.json",
            "params.npz",
            *(f"inputs_seed_{seed}.npz" for seed in FIXED_CORRECTNESS_SEEDS),
            *(f"reference_seed_{seed}.npz" for seed in FIXED_CORRECTNESS_SEEDS),
        }
        if set(files) != expected:
            raise ValueError(f"job bundle fixed file set differs: {sorted(set(files) ^ expected)}")
        actual_members = {path.name for path in self.path.iterdir() if not path.name.startswith(".")}
        if actual_members != expected | {"job_manifest.json"}:
            raise ValueError("job bundle contains missing or unmanifested public members")
        payloads: dict[str, bytes] = {}
        total = len(manifest_bytes)
        for name, identity in files.items():
            data = self._read(name)
            total += len(data)
            if total > self.max_total_bytes:
                raise ValueError("job bundle exceeds reader byte limit")
            if identity.get("size_bytes") != len(data) or identity.get("sha256") != _sha256(data):
                raise ValueError(f"job bundle hash/size mismatch: {name}")
            payloads[name] = data

        array_sets = manifest.get("array_sets")
        if not isinstance(array_sets, dict):
            raise ValueError("job manifest array_sets must be an object")

        def read_arrays(name: str) -> tuple[NamedArray, ...]:
            entries = array_sets.get(name)
            if not isinstance(entries, list):
                raise ValueError(f"missing array manifest for {name}")
            with np.load(io.BytesIO(payloads[name]), allow_pickle=False) as archive:
                if set(archive.files) != {entry["key"] for entry in entries}:
                    raise ValueError(f"NPZ member set mismatch: {name}")
                result: list[NamedArray] = []
                for entry in entries:
                    value = as_c_contiguous(archive[entry["key"]])
                    if (
                        list(value.shape) != entry["shape"]
                        or value.dtype.name != entry["dtype"]
                        or value.nbytes != entry["nbytes"]
                        or _array_digest(value) != entry["value_sha256"]
                    ):
                        raise ValueError(f"array identity mismatch: {name}/{entry['key']}")
                    result.append(NamedArray(str(entry["name"]), value))
                return tuple(result)

        build_payload = json.loads(payloads["build_result.json"])
        validation = build_payload.get("validation", {})
        if validation.get("module_json_sha256") != _sha256(payloads["module.json"]):
            raise ValueError("module.json differs from validated module identity")
        params = read_arrays("params.npz")
        abi = build_payload.get("abi", {})
        abi_params = abi.get("params", []) if isinstance(abi, dict) else []
        if [item.get("name") for item in abi_params] != [item.name for item in params]:
            raise ValueError("params.npz order/names differ from BuildResult ABI")
        hidden_cases = tuple(
            HiddenCase(
                seed=seed,
                inputs=read_arrays(f"inputs_seed_{seed}.npz"),
                reference=read_arrays(f"reference_seed_{seed}.npz"),
            )
            for seed in FIXED_CORRECTNESS_SEEDS
        )
        identities = correctness_input_identities(hidden_cases)
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("job manifest metadata must be an object")
        contract = contract_from_metadata(metadata, required=False)
        if (build_payload.get(CONTRACT_KEY) != contract
                or build_payload.get(CONTRACT_SHA_KEY) != (
                    None if contract is None else correctness_contract_sha256(contract))):
            raise ValueError("bundle correctness contract differs from sealed build payload")
        claimed_identities = metadata.get(INPUT_IDENTITIES_METADATA_KEY)
        expected_identities = {
            str(seed): identity for seed, identity in identities.items()
        }
        if claimed_identities != expected_identities:
            raise ValueError(
                "job bundle correctness input identities are missing or differ"
            )
        return JobBundleContents(
            path=self.path.resolve(),
            case_source=payloads["case.py"].decode("utf-8"),
            module_json=payloads["module.json"].decode("utf-8"),
            build_result_payload=build_payload,
            params=params,
            hidden_cases=hidden_cases,
            metadata=dict(metadata),
            manifest=manifest,
        )


__all__ = [
    "FIXED_CORRECTNESS_SEEDS",
    "INPUT_IDENTITIES_METADATA_KEY",
    "JOB_SCHEMA",
    "JobBundle",
    "JobBundleContents",
    "JobBundleReader",
    "JobBundleWriter",
]
