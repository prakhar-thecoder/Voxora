"""The public Voxora STT facade and its Phase 1 lifecycle foundation."""

from __future__ import annotations

import logging
import os
import threading
from enum import Enum, auto
from typing import Optional


class VoxoraError(Exception):
    """Base class for errors owned by Voxora."""


class ConfigurationError(VoxoraError):
    """The STT configuration is invalid."""


class RecordingStateError(VoxoraError):
    """An operation is incompatible with the current recording state."""


class AudioCaptureError(VoxoraError):
    """Audio capture failed."""


class NoSpeechError(VoxoraError):
    """No usable speech was captured."""


class RecordingTimeoutError(VoxoraError):
    """Recording exceeded a configured timeout."""


class FormattingError(VoxoraError):
    """The formatter returned an invalid result."""


class _State(Enum):
    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()


class STT:
    """Configure a microphone-based speech-to-text session."""

    def __init__(
        self,
        *,
        engine: str = "groq",
        api_key: Optional[str] = None,
        stt_model: str = "whisper-large-v3-turbo",
        format: bool = False,
        format_base_url: str = "https://api.groq.com/openai/v1",
        format_model: str = "openai/gpt-oss-120b",
        format_api_key: Optional[str] = None,
        language: Optional[str] = None,
        vad: bool = True,
        vad_mode: int = 2,
        silence_duration: float = 1.0,
        pre_speech_duration: float = 0.3,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        max_recording_duration: Optional[float] = 300.0,
        speech_start_timeout: Optional[float] = 10.0,
        log_level: int = logging.WARNING,
    ) -> None:
        from .config import validate_config

        resolved_api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
        self.config = validate_config(
            engine=engine,
            api_key=resolved_api_key,
            stt_model=stt_model,
            format=format,
            format_base_url=format_base_url,
            format_model=format_model,
            format_api_key=format_api_key,
            language=language,
            vad=vad,
            vad_mode=vad_mode,
            silence_duration=silence_duration,
            pre_speech_duration=pre_speech_duration,
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
            max_recording_duration=max_recording_duration,
            speech_start_timeout=speech_start_timeout,
            log_level=log_level,
        )
        self._logger = logging.getLogger("voxora")
        self._logger.setLevel(self.config.log_level)
        self._state = _State.IDLE
        self._state_lock = threading.RLock()

    def listen(self) -> str:
        """Record with VAD and return a transcript once implemented."""
        if not self.config.vad:
            raise ValueError("listen() is only available when vad=True")
        raise NotImplementedError("VAD recording will be implemented in a later phase")

    def start(self) -> None:
        """Start manual recording once microphone capture is implemented."""
        if self.config.vad:
            raise ValueError("start() is only available when vad=False")
        with self._state_lock:
            if self._state in (_State.RECORDING, _State.PROCESSING):
                raise RecordingStateError("a recording is already active")
            self._state = _State.RECORDING
        raise NotImplementedError("microphone recording will be implemented in a later phase")

    def stop(self) -> str:
        """Stop manual recording and return a transcript once implemented."""
        if self.config.vad:
            raise ValueError("stop() is only available when vad=False")
        with self._state_lock:
            if self._state not in (_State.RECORDING, _State.PROCESSING):
                raise RecordingStateError("no active recording")
        raise NotImplementedError("microphone recording will be implemented in a later phase")
