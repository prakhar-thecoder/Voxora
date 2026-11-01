"""Groq Whisper transcription and in-memory WAV preparation."""

from __future__ import annotations

import io
import struct
import wave
from typing import Optional

from .audio import AudioResult
from .exceptions import TranscriptionError

try:
    from groq import Groq
except ImportError:  # pragma: no cover - exercised when optional installation is incomplete
    Groq = None


def audio_result_to_wav(audio: AudioResult) -> io.BytesIO:
    """Convert a PCM result to a seekable, in-memory WAV file."""
    if not isinstance(audio, AudioResult):
        raise TranscriptionError("audio result has an invalid type")
    if not isinstance(audio.pcm_data, bytes) or not audio.pcm_data:
        raise TranscriptionError("audio result contains no PCM data")
    if (
        isinstance(audio.sample_rate, bool)
        or not isinstance(audio.sample_rate, int)
        or audio.sample_rate <= 0
        or isinstance(audio.channels, bool)
        or not isinstance(audio.channels, int)
        or not 1 <= audio.channels <= 65535
        or isinstance(audio.sample_width, bool)
        or not isinstance(audio.sample_width, int)
        or audio.sample_width not in (1, 2, 3, 4)
        or audio.sample_rate > 0xFFFFFFFF
    ):
        raise TranscriptionError("audio result has invalid WAV metadata")
    block_align = audio.channels * audio.sample_width
    if len(audio.pcm_data) % block_align:
        raise TranscriptionError("audio result contains an incomplete PCM frame")

    output = io.BytesIO()
    try:
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(audio.channels)
            wav_file.setsampwidth(audio.sample_width)
            wav_file.setframerate(audio.sample_rate)
            wav_file.writeframes(audio.pcm_data)
    except (wave.Error, OSError, ValueError, struct.error) as exc:
        raise TranscriptionError("could not prepare WAV audio") from exc
    output.seek(0)
    output.name = "voxora-recording.wav"
    return output


class GroqTranscriber:
    """Use one synchronous Groq client for an STT instance's requests."""

    def __init__(self, api_key: str, *, client=None) -> None:
        if client is not None:
            self._client = client
            return
        if Groq is None:
            raise TranscriptionError("the Groq SDK is required for transcription")
        self._client = Groq(api_key=api_key)

    def transcribe(self, audio: AudioResult, *, model: str, language: Optional[str] = None) -> str:
        wav_file = audio_result_to_wav(audio)
        request = {"file": wav_file, "model": model}
        if language is not None:
            request["language"] = language
        response = self._client.audio.transcriptions.create(**request)
        text = response.get("text") if isinstance(response, dict) else getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise TranscriptionError("Groq returned an empty or malformed transcription")
        return text.strip()
