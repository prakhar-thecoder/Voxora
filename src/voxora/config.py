"""Configuration validation for the public STT constructor."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .stt import ConfigurationError


@dataclass(frozen=True)
class STTConfig:
    engine: str
    api_key: str
    stt_model: str
    format: bool
    format_base_url: str
    format_model: str
    format_api_key: Optional[str]
    language: Optional[str]
    vad: bool
    vad_mode: int
    silence_duration: float
    pre_speech_duration: float
    sample_rate: int
    frame_duration_ms: int
    max_recording_duration: Optional[float]
    speech_start_timeout: Optional[float]
    log_level: int


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


def _positive(value: object, name: str, *, allow_none: bool = False) -> Optional[float]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        suffix = " or None" if allow_none else ""
        raise ConfigurationError(f"{name} must be positive{suffix}")
    return float(value)


def validate_config(**values: object) -> STTConfig:
    engine = values["engine"]
    if engine != "groq":
        raise ConfigurationError("unsupported engine; only 'groq' is supported")

    api_key = _non_empty(values["api_key"], "api_key")
    stt_model = _non_empty(values["stt_model"], "stt_model")
    formatting = _bool(values["format"], "format")
    format_base_url = _non_empty(values["format_base_url"], "format_base_url")
    format_model = _non_empty(values["format_model"], "format_model")
    format_api_key = values["format_api_key"]
    if format_api_key is not None:
        format_api_key = _non_empty(format_api_key, "format_api_key")
    if formatting and format_api_key is None:
        format_api_key = api_key

    language = values["language"]
    if language is not None:
        language = _non_empty(language, "language")
    vad = _bool(values["vad"], "vad")

    vad_mode = values["vad_mode"]
    if isinstance(vad_mode, bool) or not isinstance(vad_mode, int) or not 0 <= vad_mode <= 3:
        raise ConfigurationError("vad_mode must be an integer from 0 through 3")

    silence_duration = _positive(values["silence_duration"], "silence_duration")
    pre_speech_duration = values["pre_speech_duration"]
    if isinstance(pre_speech_duration, bool) or not isinstance(pre_speech_duration, (int, float)) or pre_speech_duration < 0:
        raise ConfigurationError("pre_speech_duration must be non-negative")

    sample_rate = values["sample_rate"]
    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ConfigurationError("sample_rate must be one of 8000, 16000, 32000, or 48000")
    frame_duration_ms = values["frame_duration_ms"]
    if frame_duration_ms not in (10, 20, 30):
        raise ConfigurationError("frame_duration_ms must be 10, 20, or 30")

    max_duration = _positive(values["max_recording_duration"], "max_recording_duration", allow_none=True)
    speech_timeout = _positive(values["speech_start_timeout"], "speech_start_timeout", allow_none=True)

    log_level = values["log_level"]
    if isinstance(log_level, str):
        level_names = getattr(logging, "getLevelNamesMapping", lambda: logging._nameToLevel)()
        level = level_names.get(log_level.upper())
        if level is None:
            raise ConfigurationError("log_level must be a valid logging level")
        log_level = level
    elif isinstance(log_level, int) and not isinstance(log_level, bool) and log_level >= 0:
        level = log_level
    else:
        raise ConfigurationError("log_level must be a valid logging level")

    return STTConfig(
        engine=engine,
        api_key=api_key,
        stt_model=stt_model,
        format=formatting,
        format_base_url=format_base_url,
        format_model=format_model,
        format_api_key=format_api_key,
        language=language,
        vad=vad,
        vad_mode=vad_mode,
        silence_duration=silence_duration,
        pre_speech_duration=float(pre_speech_duration),
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration_ms,
        max_recording_duration=max_duration,
        speech_start_timeout=speech_timeout,
        log_level=level,
    )
