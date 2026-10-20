"""Shared, backend-neutral core for the evaluation pipeline."""

from .job_bundle import JobBundleReader, JobBundleWriter
from .baseline import BaselineClass, BaselineDescriptor, BaselineReplayArtifact
from .candidate_builder import BuildExecution, run_build_isolated
from .errors import FaultOwner, FaultRecord, FaultStage, PipelineFault
from .device_payload import NativeFlatPayload, materialize_native_flat_payload
from .paired_timing import (
    PairedTimingResult,
    TimingConfig,
    TimingDomain,
    TimingEnvironment,
    WarmupEvidence,
    run_paired_timing,
)
from .prompt_builder import PromptBuilder, PromptPackage, ShotExample
from .response_parser import RawResponse, normalize_response
from .candidate_safety import SandboxLimits, SandboxPolicy, require_safe_source
from .task_loader import (
    HiddenCase,
    LoadedTask,
    MultiSeedLoadedTask,
    TaskLoadConfig,
    load_task,
    load_task_isolated,
)
from .ir_validation import TVMValidationReport, validate_build_result

__all__ = [
    "BaselineClass",
    "BaselineDescriptor",
    "BaselineReplayArtifact",
    "BuildExecution",
    "FaultOwner",
    "FaultRecord",
    "FaultStage",
    "HiddenCase",
    "JobBundleReader",
    "JobBundleWriter",
    "LoadedTask",
    "MultiSeedLoadedTask",
    "NativeFlatPayload",
    "PairedTimingResult",
    "PipelineFault",
    "PromptBuilder",
    "PromptPackage",
    "RawResponse",
    "SandboxLimits",
    "SandboxPolicy",
    "ShotExample",
    "TVMValidationReport",
    "TaskLoadConfig",
    "TimingConfig",
    "TimingDomain",
    "TimingEnvironment",
    "WarmupEvidence",
    "load_task",
    "load_task_isolated",
    "materialize_native_flat_payload",
    "normalize_response",
    "require_safe_source",
    "run_build_isolated",
    "run_paired_timing",
    "validate_build_result",
]
