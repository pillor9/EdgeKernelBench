"""Explicit model selection for an externally configured coding client."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path


def require_model(model: str | None) -> None:
    """Require an explicit model identifier without choosing a default."""
    if not isinstance(model, str) or not model or model != model.strip() or any(
        ord(character) < 32 for character in model
    ):
        raise ValueError("an explicit model identifier is required")


def validate_model_effort(model_effort: str | None) -> None:
    if model_effort is not None and model_effort not in {"low", "high"}:
        raise ValueError("model_effort must be low, high, or None")


def provider_policy(model: str | None, *, model_effort: str | None = None) -> dict:
    """Record public launch choices without connection details or credentials."""
    require_model(model)
    validate_model_effort(model_effort)
    return {
        "schema_version": "leka.provider_policy.v1",
        "requested_model": model,
        "automatic_model_fallback": False,
        "requested_cli_effort": model_effort,
        "configuration": "external_client_environment",
    }


def claude_provider_environment(model: str | None, environment: Mapping[str, str]) -> dict[str, str]:
    """Use the caller's external client configuration without selecting a service."""
    require_model(model)
    return dict(environment)


def secret_environment_names(environment: Mapping[str, str]) -> tuple[str, ...]:
    """Identify credential variables without retaining or reporting their values."""
    pattern = re.compile(r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)(?:_|$)", re.I)
    return tuple(sorted(name for name in environment if pattern.search(name)))


def validate_claude_provider_argv(model: str | None, argv: Iterable[str], *,
                                  model_effort: str | None = None) -> None:
    """Keep model choice explicit and prevent workspace settings from overriding it."""
    require_model(model)
    validate_model_effort(model_effort)
    args = list(argv)
    actual_effort = [args[i + 1] if i + 1 < len(args) else None
                     for i, arg in enumerate(args) if arg == "--effort"]
    if actual_effort != ([] if model_effort is None else [model_effort]) or any(
        arg.startswith("--effort=") for arg in args
    ):
        raise ValueError("CLI effort must match the explicit model_effort option")
    if (args.count("--model") != 1 or args[args.index("--model") + 1:][:1] != [model]
            or any(arg.startswith("--model=") for arg in args)):
        raise ValueError("launch requires one explicit model selector")
    if any(arg == "--fallback-model" or arg.startswith("--fallback-model=") for arg in args):
        raise ValueError("automatic model fallback is disabled")
    for index, arg in enumerate(args):
        if arg == "--settings":
            if index + 1 >= len(args):
                raise ValueError("settings path is missing")
            source = args[index + 1]
        elif arg.startswith("--settings="):
            source = arg.partition("=")[2]
        else:
            continue
        try:
            payload = json.loads(source if source.lstrip().startswith("{")
                                 else Path(source).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("client settings cannot be validated") from None
        forbidden = {"apiKeyHelper", "awsAuthRefresh", "awsCredentialExport", "model",
                     "availableModels", "alwaysThinkingEnabled", "effortLevel"}
        if not isinstance(payload, dict) or forbidden.intersection(payload) or payload.get("env"):
            raise ValueError("workspace settings cannot override external client configuration")


def is_claude_synthetic_api_error(event: Mapping) -> bool:
    """Recognize Claude CLI's local error envelope, never a model response.

    Both the error flag and synthetic model ID are required. Claude's internal
    event uses camelCase; its stream-json wire serializer uses snake_case.
    An error flag on a real model ID does
    not exempt that ID from the selected-model identity check. The envelope
    alone cannot distinguish a local proxy failure from an upstream API error.
    """
    message = event.get("message")
    flags = [event[key] for key in ("isApiErrorMessage", "is_api_error_message")
             if key in event]
    return (event.get("type") == "assistant"
            and bool(flags) and all(flag is True for flag in flags)
            and isinstance(message, Mapping)
            and message.get("model") == "<synthetic>")


def claude_synthetic_error_class(event: Mapping) -> str | None:
    if not is_claude_synthetic_api_error(event):
        return None
    # This is a local CLI output-limit envelope, not an HTTP/model mismatch.
    if event.get("api_error", event.get("apiError")) == "max_output_tokens":
        return "claude_cli_max_output_tokens"
    # Claude Code currently reports an upstream HTTP 429 as a
    # synthetic assistant message with ``error: rate_limit``.  Preserve that
    # distinction so a caller can wait and resume the same conversation rather
    # than treating a transient capacity rejection as a terminal model result.
    error = event.get("error")
    if isinstance(error, str) and error.strip().lower() in {
        "rate_limit",
        "rate_limited",
    }:
        return "claude_cli_rate_limited"
    message = event.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, list):
        text = "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, Mapping) and isinstance(item.get("text"), str)
        )
        if re.search(r"(?:\b429\b|rate[_ -]?limit)", text, re.I):
            return "claude_cli_rate_limited"
    return "claude_cli_api_transport_failure"


def response_model_evidence(requested_model: str | None, events: Iterable[Mapping]) -> dict:
    """Validate returned model IDs when the Claude transport exposes them.

    Missing model fields remain explicitly unverified. Startup/init fields
    are not response evidence; only assistant messages and result usage count.
    CLI synthetic API errors are transport failures, not returned model IDs.
    """
    models: set[str] = set()
    assistant_models: set[str] = set()
    synthetic_api_error = False
    synthetic_classes: set[str] = set()
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if is_claude_synthetic_api_error(event):
            synthetic_api_error = True
            synthetic_classes.add(claude_synthetic_error_class(event))
            continue
        if event.get("type") == "assistant":
            message = event.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("model"), str):
                models.add(message["model"])
                assistant_models.add(message["model"])
        if event.get("type") == "result" or "modelUsage" in event:
            usage = event.get("modelUsage")
            if isinstance(usage, Mapping):
                models.update(name for name in usage if isinstance(name, str))
    if requested_model and any(name != requested_model for name in models):
        raise ValueError("response model differs from the explicitly requested model")
    evidence = {
        "requested_model": requested_model,
        "returned_models": sorted(models),
        "response_model_verified": (bool(models)
                                    and all(name == requested_model for name in models)
                                    and not (synthetic_api_error and not assistant_models)),
    }
    if synthetic_api_error:
        # A CLI result can list the configured model in its usage bookkeeping
        # after a failed request. That alone cannot turn a synthetic error into
        # a verified response. Preserve real earlier response evidence, if any.
        evidence["response_error_class"] = (
            "claude_cli_max_output_tokens"
            if synthetic_classes == {"claude_cli_max_output_tokens"}
            else (
                "claude_cli_rate_limited"
                if synthetic_classes == {"claude_cli_rate_limited"}
                else "claude_cli_api_transport_failure"
            )
        )
    return evidence
