"""Lossless provider-response capture and strict whole-file source extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import ast
import hashlib
import re
from typing import Mapping, Protocol, runtime_checkable

from pipeline.interfaces.tensor_spec import JSONValue, _json_value

from .errors import FaultRecord, FaultStage, PipelineFault


_WHOLE_FILE_FENCE = re.compile(
    r"\A[ \t\r\n]*```(?P<language>python|py)[ \t]*\r?\n"
    r"(?P<source>[\s\S]*?)\r?\n```[ \t\r\n]*\Z",
    re.IGNORECASE,
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RawResponse:
    text: str
    provider: str = "static"
    model: str = "unknown"
    request_id: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("response text must be a string")
        normalized = _json_value(self.metadata)
        assert isinstance(normalized, dict)
        object.__setattr__(self, "metadata", normalized)

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)


@dataclass(frozen=True, slots=True)
class NormalizedSource:
    source: str
    raw_sha256: str
    source_sha256: str
    normalization: str


@runtime_checkable
class ResponseSource(Protocol):
    def obtain(self) -> RawResponse: ...


@dataclass(frozen=True, slots=True)
class StaticResponseSource:
    response: RawResponse

    def obtain(self) -> RawResponse:
        return self.response


def normalize_response(response: RawResponse | str) -> NormalizedSource:
    raw = response if isinstance(response, RawResponse) else RawResponse(response)
    text = raw.text
    if not text.strip():
        raise PipelineFault(
            FaultRecord(FaultStage.RESPONSE_SOURCE, "empty_response", "response is empty")
        )
    fence = _WHOLE_FILE_FENCE.fullmatch(text)
    if fence:
        source = fence.group("source") + "\n"
        normalization = "whole_file_python_fence"
    else:
        # Any fence marker outside the exact accepted wrapper indicates commentary,
        # multiple candidates, or an incomplete fence rather than Python source.
        if "```" in text:
            raise PipelineFault(
                FaultRecord(
                    FaultStage.RESPONSE_SOURCE,
                    "invalid_fence",
                    "response must be plain Python or one whole-file Python fence",
                )
            )
        source = text
        normalization = "plain_python"
    if not source.endswith("\n"):
        source += "\n"
    try:
        ast.parse(source, filename="candidate_case.py", mode="exec")
    except SyntaxError as error:
        message = f"{error.msg} at line {error.lineno}, column {error.offset}"
        raise PipelineFault(
            FaultRecord(
                FaultStage.PYTHON_SYNTAX,
                "invalid_python",
                message,
                exception_type="SyntaxError",
            )
        ) from error
    return NormalizedSource(
        source=source,
        raw_sha256=raw.sha256,
        source_sha256=sha256_text(source),
        normalization=normalization,
    )


__all__ = [
    "NormalizedSource",
    "RawResponse",
    "ResponseSource",
    "StaticResponseSource",
    "normalize_response",
    "sha256_text",
]
