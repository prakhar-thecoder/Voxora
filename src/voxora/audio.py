"""Internal microphone capture, buffering, and WebRTC VAD."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List

from .exceptions import AudioCaptureError, NoSpeechError


@dataclass(frozen=True)
class AudioResult:
    """Internal PCM recording result for the future transcription layer."""

    pcm_data: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2


class AudioRecorder:
    """Capture mono 16-bit PCM from the default microphone."""

    def __init__(self, config, logger, *, pyaudio_module=None, vad_factory=None, clock: Callable[[], float] = time.monotonic):
        self.config = config
        self.logger = logger
        self._pyaudio_module = pyaudio_module
        self._vad_factory = vad_factory
        self._clock = clock
        self._audio = None
        self._stream = None
        self.samples_per_frame = config.sample_rate * config.frame_duration_ms // 1000
        if self.samples_per_frame * 1000 != config.sample_rate * config.frame_duration_ms:
            raise AudioCaptureError("frame duration does not produce a whole number of samples")
        self.frame_bytes = self.samples_per_frame * 2

    def _load_pyaudio(self):
        if self._pyaudio_module is None:
            try:
                import pyaudio
            except Exception as exc:
                raise AudioCaptureError("PyAudio is required for microphone recording") from exc
            self._pyaudio_module = pyaudio
        return self._pyaudio_module

    def _load_vad(self):
        if self._vad_factory is None:
            try:
                import webrtcvad
            except Exception as exc:
                raise AudioCaptureError("WebRTC VAD is required for VAD recording") from exc
            self._vad_factory = webrtcvad.Vad
        return self._vad_factory

    def open(self) -> None:
        if self._stream is not None:
            raise AudioCaptureError("microphone stream is already open")
        pyaudio = self._load_pyaudio()
        try:
            self._audio = pyaudio.PyAudio()
            self._stream = self._audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.samples_per_frame,
            )
            self.logger.debug("microphone initialized")
        except Exception as exc:
            self.close()
            raise AudioCaptureError("could not initialize microphone") from exc

    def close(self) -> None:
        stream, audio = self._stream, self._audio
        self._stream = None
        self._audio = None
        cleanup_errors = []
        try:
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception as exc:
                    cleanup_errors.append(exc)
                try:
                    stream.close()
                except Exception as exc:
                    cleanup_errors.append(exc)
        finally:
            if audio is not None:
                try:
                    audio.terminate()
                except Exception as exc:
                    cleanup_errors.append(exc)
        self.logger.debug("microphone resources cleaned up")
        if cleanup_errors:
            self.logger.error("microphone cleanup encountered an error")

    def _read_frame(self) -> bytes:
        if self._stream is None:
            raise AudioCaptureError("microphone stream is not open")
        try:
            frame = self._stream.read(self.samples_per_frame, exception_on_overflow=False)
        except Exception as exc:
            raise AudioCaptureError("microphone read failed") from exc
        if not isinstance(frame, bytes) or len(frame) != self.frame_bytes:
            raise AudioCaptureError("microphone returned an invalid audio frame")
        return frame

    def _new_vad(self):
        try:
            return self._load_vad()(self.config.vad_mode)
        except Exception as exc:
            raise AudioCaptureError("invalid WebRTC VAD configuration") from exc

    def record_vad(self) -> AudioResult:
        """Record until speech followed by configured silence is detected."""
        self.open()
        frames: List[bytes] = []
        pre_speech_frames: Deque[bytes] = deque(
            maxlen=max(0, math.ceil(self.config.pre_speech_duration * 1000 / self.config.frame_duration_ms))
        )
        frame_seconds = self.config.frame_duration_ms / 1000.0
        silence_frames_needed = max(1, math.ceil(self.config.silence_duration / frame_seconds))
        speech_started = False
        consecutive_silence = 0
        started_at = self._clock()
        vad = None
        try:
            vad = self._new_vad()
            self.logger.debug("recording started with VAD")
            while True:
                frame = self._read_frame()
                elapsed = self._clock() - started_at
                if self.config.max_recording_duration is not None and elapsed >= self.config.max_recording_duration:
                    self.logger.debug("maximum recording duration reached")
                    if not speech_started:
                        raise NoSpeechError("no speech detected before maximum recording duration")
                    break

                if not speech_started and (
                    self.config.speech_start_timeout is not None
                    and elapsed >= self.config.speech_start_timeout
                ):
                    raise NoSpeechError("no speech detected before speech_start_timeout")

                try:
                    speaking = vad.is_speech(frame, self.config.sample_rate)
                except Exception as exc:
                    raise AudioCaptureError("WebRTC VAD processing failed") from exc
                if not speech_started:
                    if speaking:
                        speech_started = True
                        frames.extend(pre_speech_frames)
                        frames.append(frame)
                        self.logger.debug("speech detected")
                    else:
                        pre_speech_frames.append(frame)
                    continue

                frames.append(frame)
                if speaking:
                    consecutive_silence = 0
                else:
                    consecutive_silence += 1
                    if consecutive_silence >= silence_frames_needed:
                        self.logger.debug("silence detected; recording stopped")
                        break

            if not frames:
                raise NoSpeechError("no usable audio was captured")
            return AudioResult(b"".join(frames), self.config.sample_rate)
        finally:
            self.close()
            self.logger.debug("audio preparation completed")

    def record_manual(self, stop_event: threading.Event) -> AudioResult:
        """Record until stopped or the configured maximum duration is reached."""
        if self._stream is None:
            raise AudioCaptureError("microphone stream is not open")
        frames: List[bytes] = []
        started_at = self._clock()
        self.logger.debug("manual recording started")
        try:
            while not stop_event.is_set():
                if self.config.max_recording_duration is not None and self._clock() - started_at >= self.config.max_recording_duration:
                    self.logger.debug("maximum recording duration reached")
                    break
                frames.append(self._read_frame())
            if not frames:
                raise NoSpeechError("no usable audio was captured")
            return AudioResult(b"".join(frames), self.config.sample_rate)
        finally:
            self.close()
            self.logger.debug("manual recording stopped")
