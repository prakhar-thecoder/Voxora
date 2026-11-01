"""The public Voxora STT facade and its Phase 1 lifecycle foundation."""

from __future__ import annotations

import logging
import os
import threading
from enum import Enum, auto
from typing import Optional

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
from .audio import AudioRecorder, AudioResult
from .formatter import TranscriptFormatter
from .groq import GroqTranscriber


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
        self._logger = logging.getLogger(f"voxora.stt.{id(self)}")
        self._logger.setLevel(self.config.log_level)
        self._state = _State.IDLE
        self._state_lock = threading.RLock()
        self._stop_lock = threading.Lock()
        self._recorder = None
        self._worker = None
        self._stop_event = None
        self._audio_result = None
        self._worker_error = None
        self._session_generation = 0
        self._active_session = None
        self._groq_transcriber = None
        self._groq_init_lock = threading.Lock()
        self._formatter = None
        self._formatter_init_lock = threading.Lock()

    def listen(self) -> str:
        """Record with VAD and reach the Phase 3 transcription boundary."""
        if not self.config.vad:
            raise ValueError("listen() is only available when vad=True")
        with self._state_lock:
            session = self._prepare_new_session()
            self._state = _State.RECORDING
        try:
            recorder = AudioRecorder(self.config, self._logger)
            with self._state_lock:
                if self._active_session == session:
                    self._recorder = recorder
            result = recorder.record_vad()
            with self._state_lock:
                if self._active_session != session:
                    raise RecordingStateError("recording session is no longer active")
                self._audio_result = result
                self._state = _State.PROCESSING
            transcript = self._transcribe_audio(result)
            with self._state_lock:
                self._clear_session_references(session)
                self._state = _State.COMPLETED
            return transcript
        except BaseException:
            with self._state_lock:
                if self._active_session == session:
                    self._clear_session_references(session)
                    self._state = _State.FAILED
            raise

    def start(self) -> None:
        """Start manual microphone recording in a background worker."""
        if self.config.vad:
            raise ValueError("start() is only available when vad=False")
        with self._state_lock:
            if self._state in (_State.RECORDING, _State.PROCESSING):
                raise RecordingStateError("a recording is already active")
            session = self._prepare_new_session()
            recorder = AudioRecorder(self.config, self._logger)
            try:
                recorder.open()
            except BaseException:
                self._clear_session_references(session)
                self._state = _State.FAILED
                raise
            self._recorder = recorder
            self._stop_event = threading.Event()
            self._state = _State.RECORDING
            try:
                self._worker = threading.Thread(
                    target=self._record_manual_worker,
                    args=(session, recorder, self._stop_event),
                    name="voxora-recording",
                    daemon=False,
                )
                self._worker.start()
            except BaseException:
                self._stop_event.set()
                try:
                    recorder.close()
                except BaseException:
                    self._logger.exception("recording startup cleanup failed")
                self._clear_session_references(session)
                self._state = _State.FAILED
                raise

    def stop(self) -> str:
        """Stop manual recording and reach the Phase 3 transcription boundary."""
        if self.config.vad:
            raise ValueError("stop() is only available when vad=False")
        acquired = self._stop_lock.acquire(blocking=False)
        try:
            if not acquired:
                raise RecordingStateError("another stop() call is already handling this recording")
            with self._state_lock:
                if self._state not in (_State.RECORDING, _State.PROCESSING, _State.FAILED):
                    raise RecordingStateError("no active recording")
                worker = self._worker
                stop_event = self._stop_event
                session = self._active_session
            if worker is None or stop_event is None:
                raise RecordingStateError("no active recording")
            stop_event.set()
            worker.join()
            with self._state_lock:
                if self._worker_error is not None:
                    error = self._worker_error
                    self._clear_session_references(session)
                    self._state = _State.FAILED
                    raise error
                result = self._audio_result
                self._state = _State.PROCESSING
            try:
                transcript = self._transcribe_audio(result)
                with self._state_lock:
                    self._clear_session_references(session)
                    self._state = _State.COMPLETED
                return transcript
            except BaseException:
                with self._state_lock:
                    if self._active_session == session:
                        self._clear_session_references(session)
                        self._state = _State.FAILED
                raise
        finally:
            if acquired:
                self._stop_lock.release()

    def _prepare_new_session(self) -> int:
        if self._state in (_State.RECORDING, _State.PROCESSING) or self._active_session is not None:
            raise RecordingStateError("a recording is already active")
        self._session_generation += 1
        self._active_session = self._session_generation
        self._state = _State.IDLE
        self._audio_result = None
        self._worker_error = None
        self._worker = None
        self._stop_event = None
        return self._active_session

    def _clear_session_references(self, session) -> None:
        if self._active_session != session:
            return
        self._recorder = None
        self._worker = None
        self._stop_event = None
        self._audio_result = None
        self._worker_error = None
        self._active_session = None

    def _record_manual_worker(self, session, recorder, stop_event) -> None:
        try:
            result = recorder.record_manual(stop_event)
            with self._state_lock:
                if self._active_session == session:
                    self._audio_result = result
                    self._state = _State.PROCESSING
        except BaseException as exc:
            with self._state_lock:
                if self._active_session == session:
                    self._worker_error = exc
                    self._state = _State.FAILED

    def _transcribe_audio(self, result: AudioResult) -> str:
        """Transcribe prepared audio through the configured Groq engine."""
        if self._groq_transcriber is None:
            with self._groq_init_lock:
                if self._groq_transcriber is None:
                    self._groq_transcriber = GroqTranscriber(self.config.api_key)
        transcriber = self._groq_transcriber
        transcript = transcriber.transcribe(
            result,
            model=self.config.stt_model,
            language=self.config.language,
        )
        if not self.config.format:
            return transcript
        if self._formatter is None:
            with self._formatter_init_lock:
                if self._formatter is None:
                    self._formatter = TranscriptFormatter(
                        self.config.format_api_key,
                        self.config.format_base_url,
                        self.config.format_model,
                    )
        return self._formatter.format(transcript)
