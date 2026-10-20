"""Assemble task prompts, demonstrations, and public backend information."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from pipeline.interfaces import BuildTask, JSONValue, TensorSpec
from pipeline.interfaces.tensor_spec import _json_value

from .response_parser import normalize_response


@dataclass(frozen=True, slots=True)
class PromptMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported prompt role: {self.role}")
        if not self.content:
            raise ValueError("prompt message content cannot be empty")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class ShotExample:
    shot_id: str
    backend: str
    task: BuildTask
    case_source: str
    level: int | None = None
    mode: str = "full_deployment_case"

    def __post_init__(self) -> None:
        if not self.shot_id or not self.backend:
            raise ValueError("shot_id and backend must be non-empty")
        if self.level is not None and self.level not in {1, 2, 3}:
            raise ValueError("shot level must be 1, 2, or 3")
        _normalize_shot_source(self.case_source, self.mode)


@dataclass(frozen=True, slots=True)
class PromptPackage:
    messages: tuple[PromptMessage, ...]
    prompt_sha256: str
    target_task_id: str
    shot_ids: tuple[str, ...]
    public_task_payload: Mapping[str, JSONValue]

    def provider_messages(self) -> list[dict[str, str]]:
        return [message.to_dict() for message in self.messages]


def _tensor_catalog(task: BuildTask, names: Sequence[str], getter: object) -> list[JSONValue]:
    result: list[JSONValue] = []
    for name in names:
        array = getter(name)  # type: ignore[operator]
        result.append(TensorSpec.from_array(array, name=name).to_dict())
    return result


def public_task_payload(task: BuildTask) -> dict[str, JSONValue]:
    """Return the complete public ABI without values, references, or seed metadata."""

    input_names = tuple(
        getattr(
            task,
            "input_names",
            tuple(spec.name or f"input_{index}" for index, spec in enumerate(task.input_specs)),
        )
    )
    payload: dict[str, JSONValue] = {
        "task_id": str(getattr(task, "task_id", "anonymous_task")),
        "init_args": _json_value(task.init_args),
        "input_names": list(input_names),
        "input_specs": [spec.to_dict() for spec in task.input_specs],
        "parameters": _tensor_catalog(task, task.parameter_names(), task.parameter),
        "buffers": _tensor_catalog(task, task.buffer_names(), task.buffer),
        "output_spec": task.output_spec.to_dict(),
        "execution_mode": str(task.execution_mode),
    }
    return payload


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


OUTPUT_CONTRACTS: Mapping[str, Mapping[str, str]] = {
    "full_deployment_case": {
        "form": "complete Python case.py",
        "entry": "def build(task) -> BuildResult",
        "return": "pipeline.interfaces.BuildResult",
        "accepted_transport": "plain Python or one whole-file Python Markdown fence",
    },
    "restricted_irmodule": {
        "form": "one strict JSON object",
        "entry": '{"tvm_module": "<one complete TVMScript IRModule string>"}',
        "return": "strict JSON, single field tvm_module",
        "accepted_transport": "plain JSON or one whole-response JSON Markdown fence",
    },
}


def _normalize_shot_source(source: str, mode: str) -> str:
    """Check a demonstration against its declared output format."""
    if mode not in OUTPUT_CONTRACTS:
        raise ValueError(f"unsupported demonstration mode: {mode!r}")
    if mode == "full_deployment_case":
        return normalize_response(source).source
    if not isinstance(source, str):
        raise TypeError("demonstration source must be a string")
    text = source.strip()
    if text.startswith("```"):
        fence = re.fullmatch(r"```json[ \t]*\r?\n([\s\S]*?)\r?\n```", text, re.IGNORECASE)
        if fence is None:
            raise ValueError("restricted demonstration must be plain JSON or one JSON fence")
        text = fence.group(1)

    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        fields = dict(pairs)
        if len(fields) != len(pairs):
            raise ValueError("restricted demonstration contains duplicate JSON fields")
        return fields

    def reject_constant(value: str) -> None:
        raise ValueError(f"restricted demonstration contains a non-JSON constant: {value}")

    payload = json.loads(text, object_pairs_hook=unique_fields, parse_constant=reject_constant)
    if (not isinstance(payload, dict) or set(payload) != {"tvm_module"}
            or not isinstance(payload["tvm_module"], str) or not payload["tvm_module"].strip()):
        raise ValueError("restricted demonstration requires one non-empty tvm_module string")
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _copy_present(source: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    return {name: source[name] for name in names if name in source}


def model_visible_backend_facts(
    *,
    backend: str,
    hardware_profile: Mapping[str, JSONValue],
    build_environment: Mapping[str, JSONValue],
    timing_contract: Mapping[str, JSONValue],
    runtime_policy_ids: Sequence[str],
) -> dict[str, JSONValue]:
    """Select public compilation and execution facts for the model prompt.

    Keep raw device observations and measurement records in the evaluator.
    """

    profile = _mapping(hardware_profile.get("profile"))
    resolved = _mapping(hardware_profile.get("resolved_capabilities"))
    compiler = _mapping(build_environment.get("device_compiler_identity"))

    targets: dict[str, Any] = {}
    for name in ("target", "gpu_target", "host_target", "opencl_target_base"):
        for source in (resolved, profile, compiler, hardware_profile):
            value = source.get(name)
            if value is not None:
                targets[name] = value
                break

    physical = _mapping(resolved.get("physical_limits"))
    if not physical:
        physical = _mapping(profile.get("expected_physical_limits"))
    if not physical:
        physical = _mapping(hardware_profile.get("physical_limits"))

    physical_keys = {
        "jetson_gpu": (
            "max_threads_per_block",
            "max_thread_extent",
            "max_shared_memory_per_block_bytes",
            "thread_warp_size",
            "registers_per_block",
        ),
        "rk3588_gpu": (
            "max_workgroup_size",
            "max_work_item_sizes",
            "local_memory_bytes",
            "max_function_args",
        ),
        "vivo_gpu": (
            "max_threads_per_workgroup",
            "max_thread_extent",
            "max_shared_memory_per_workgroup_bytes",
            "max_function_args",
        ),
        "vivo_cpu": (),
    }
    physical = _copy_present(physical, physical_keys.get(backend, ()))

    capabilities = _mapping(resolved.get("capabilities"))
    features = _copy_present(
        capabilities,
        (
            "fp16_supported",
            "tvm_supported_tensor_intrinsics",
            "registered_tensor_intrinsics",
        ),
    )
    registered = _mapping(resolved.get("physical_limits")).get(
        "registered_tensor_intrinsics"
    )
    if registered is not None:
        features["registered_tensor_intrinsics"] = registered
    safe_isa = _mapping(resolved.get("safe_isa")) or _mapping(
        profile.get("safe_isa_contract")
    )

    execution: dict[str, Any] = {
        "allowed_runtime_policy_ids": list(runtime_policy_ids),
        "post_candidate_optimization_allowed": False,
        "timed_region": timing_contract.get(
            "measurement_boundary", "exact_saved_relax_entry_closure"
        ),
        "inputs_and_params_resident_before_timing": bool(
            timing_contract.get("device_inputs_and_params_resident_before_timing", True)
        ),
        "static_parameter_prepacking": "allowed_during_build_task",
        "dynamic_input_dependent_work": "must_remain_inside_timed_relax_entry",
    }
    runtime_threads = resolved.get("runtime_num_threads")
    if runtime_threads is not None:
        execution["runtime_num_threads"] = runtime_threads
    execution_domain = _mapping(profile.get("execution_domain"))
    execution.update(
        _copy_present(
            execution_domain,
            ("profile_id", "runtime_num_threads", "placement_class"),
        )
    )

    compiler_facts = _copy_present(
        compiler,
        (
            "tvm_version",
            "cuda_runtime_version",
            "cuda_driver_version",
            "tvm_opencl_enabled",
        ),
    )
    if "tvm_version" not in compiler_facts:
        compiler_facts["tvm_version"] = build_environment.get(
            "host_tvm_version", profile.get("tvm_version")
        )
    compiler_facts.update(
        _copy_present(
            profile,
            (
                "compiler_location",
                "compile_location",
                "cuda_arch",
                "cuda_toolkit",
                "runtime_device",
            ),
        )
    )

    compiler_facts = {
        name: value for name, value in compiler_facts.items() if value is not None
    }
    identity = _copy_present(
        hardware_profile,
        ("device", "device_name", "soc", "online_cpu_count"),
    )
    facts: dict[str, JSONValue] = {
        "schema_version": "pipeline.model_visible_backend_facts.v2",
        "backend": backend,
        "targets": _json_value(targets),
        "execution": _json_value(execution),
    }
    for name, value in (
        ("physical_limits", dict(physical)),
        ("features", features),
        ("safe_isa", dict(safe_isa)),
        ("compiler", compiler_facts),
    ):
        if value:
            facts[name] = _json_value(value)
    if identity:
        facts["identity"] = _json_value(identity)
    return facts


def _task_user_message(
    task: BuildTask,
    *,
    label: str,
    mode: str = "full_deployment_case",
    extra_facts: Mapping[str, JSONValue] | None = None,
) -> str:
    facts = {
        "schema_version": "pipeline.prompt_task.v1",
        "label": label,
        "task": public_task_payload(task),
        "output_contract": dict(OUTPUT_CONTRACTS[mode]),
    }
    if extra_facts:
        facts.update(_json_value(dict(extra_facts)))
    return_line = (
        "Return only one complete case.py; do not include prose, patches, or multiple candidates."
        if mode == "full_deployment_case"
        else (
            "Return only one strict JSON object with the single field tvm_module; "
            "do not include prose, patches, or multiple candidates."
        )
    )
    return (
        "Implement the following fixed-shape operator for the stated backend facts.\n"
        + return_line
        + "\n\n"
        "TASK_FACTS_JSON\n"
        + json.dumps(facts, ensure_ascii=False, sort_keys=True, indent=2)
        + "\nEND_TASK_FACTS_JSON\n\n"
        "PYTORCH_SEMANTIC_SOURCE\n"
        + task.source_text
        + ("" if task.source_text.endswith("\n") else "\n")
        + "END_PYTORCH_SEMANTIC_SOURCE\n"
    )


class PromptBuilder:
    def __init__(
        self,
        system_prompt: str,
        *,
        required_shot_count: int = 1,
        backend_card: str = "",
    ) -> None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")
        if required_shot_count not in {0, 1, 2, 3}:
            raise ValueError("required_shot_count must be 0, 1, 2, or 3")
        if not isinstance(backend_card, str):
            raise ValueError("backend_card must be a string")
        message = system_prompt.strip()
        if backend_card.strip():
            message += "\n\n# BACKEND CAPABILITY CARD\n\n" + backend_card.strip()
        self.system_prompt = message + "\n"
        self.required_shot_count = required_shot_count

    def build(
        self,
        task: BuildTask,
        *,
        backend: str,
        hardware_profile: Mapping[str, JSONValue],
        build_environment: Mapping[str, JSONValue],
        timing_contract: Mapping[str, JSONValue],
        runtime_policy_ids: Sequence[str] = (),
        shots: Sequence[ShotExample] = (),
        mode: str = "full_deployment_case",
        extra_facts: Mapping[str, JSONValue] | None = None,
    ) -> PromptPackage:
        if mode not in OUTPUT_CONTRACTS:
            raise ValueError(f"unsupported prompt mode: {mode!r}")
        if not backend:
            raise ValueError("backend must be non-empty")
        if len(shots) != self.required_shot_count:
            raise ValueError(
                f"prompt requires exactly {self.required_shot_count} fixed shots, got {len(shots)}"
            )
        if len({shot.shot_id for shot in shots}) != len(shots):
            raise ValueError("shot_ids must be unique")
        if any(shot.backend != backend for shot in shots):
            raise ValueError("all shots must match the target backend")
        if any(shot.mode != mode for shot in shots):
            raise ValueError("all shots must match the target output mode")
        if runtime_policy_ids and len(set(runtime_policy_ids)) != len(runtime_policy_ids):
            raise ValueError("runtime_policy_ids must be unique")
        backend_facts = model_visible_backend_facts(
            backend=backend,
            hardware_profile=hardware_profile,
            build_environment=build_environment,
            timing_contract=timing_contract,
            runtime_policy_ids=runtime_policy_ids,
        )
        system_message = (
            self.system_prompt.rstrip()
            + "\n\n# FROZEN MODEL-VISIBLE BACKEND FACTS\n\n"
            + "The following compact JSON contains the trusted backend facts made "
            + "available for both demonstrations and the final target. Runner-side "
            + "attestation and measurement details are intentionally omitted. An "
            + "omitted fact does not create an additional prohibition; obey the "
            + "explicit contracts and compiler acceptance.\n\n"
            + "BACKEND_FACTS_JSON\n"
            + json.dumps(backend_facts, ensure_ascii=False, sort_keys=True, indent=2)
            + "\nEND_BACKEND_FACTS_JSON\n"
        )
        messages: list[PromptMessage] = [PromptMessage("system", system_message)]
        for shot in shots:
            messages.append(
                PromptMessage(
                    "user",
                    _task_user_message(
                        shot.task,
                        label=f"demonstration:{shot.shot_id}",
                        mode=mode,
                    ),
                )
            )
            messages.append(
                PromptMessage("assistant", _normalize_shot_source(shot.case_source, mode))
            )
        messages.append(
            PromptMessage(
                "user",
                _task_user_message(
                    task,
                    label="target",
                    mode=mode,
                    extra_facts=extra_facts,
                ),
            )
        )
        serialized = _canonical_json([message.to_dict() for message in messages])
        return PromptPackage(
            messages=tuple(messages),
            prompt_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            target_task_id=str(getattr(task, "task_id", "anonymous_task")),
            shot_ids=tuple(shot.shot_id for shot in shots),
            public_task_payload=public_task_payload(task),
        )


__all__ = [
    "PromptBuilder",
    "PromptMessage",
    "PromptPackage",
    "ShotExample",
    "model_visible_backend_facts",
    "public_task_payload",
]
