"""Exceptions raised by Voxora-owned validation and lifecycle code."""


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
