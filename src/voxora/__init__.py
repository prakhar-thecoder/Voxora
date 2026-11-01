"""Public package interface for Voxora."""

from .exceptions import (
    AudioCaptureError,
    ConfigurationError,
    FormattingError,
    NoSpeechError,
    RecordingStateError,
    RecordingTimeoutError,
    TranscriptionError,
    VoxoraError,
)
from .stt import STT

__all__ = [
    "STT",
    "VoxoraError",
    "ConfigurationError",
    "RecordingStateError",
    "AudioCaptureError",
    "NoSpeechError",
    "RecordingTimeoutError",
    "FormattingError",
    "TranscriptionError",
]
