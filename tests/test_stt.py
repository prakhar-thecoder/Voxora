import logging
import threading

import pytest

from voxora import ConfigurationError, RecordingStateError, STT
from voxora.stt import _State


def test_public_import_and_defaults(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    stt = STT()
    assert stt.config.stt_model == "whisper-large-v3-turbo"
    assert stt.config.vad is True
    assert stt.config.api_key == "env-key"


def test_vad_workflow_returns_groq_transcript(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    class Recorder:
        def __init__(self, config, logger):
            pass
        def record_vad(self):
            from voxora.audio import AudioResult
            return AudioResult(b"audio", 16000)

    class Transcriber:
        def __init__(self, api_key):
            self.api_key = api_key
        def transcribe(self, audio, *, model, language=None):
            assert model == "whisper-test"
            assert language == "en"
            return "spoken text"

    monkeypatch.setattr("voxora.stt.AudioRecorder", Recorder)
    monkeypatch.setattr("voxora.stt.GroqTranscriber", Transcriber)
    stt = STT(stt_model="whisper-test", language="en")
    assert stt.listen() == "spoken text"
    assert stt._state is _State.COMPLETED


def test_manual_workflow_returns_groq_transcript(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    class Recorder:
        def __init__(self, config, logger):
            pass
        def open(self):
            pass
        def record_manual(self, stop_event):
            stop_event.wait(1)
            from voxora.audio import AudioResult
            return AudioResult(b"audio", 16000)

    class Transcriber:
        def __init__(self, api_key):
            pass
        def transcribe(self, audio, *, model, language=None):
            return "manual text"

    monkeypatch.setattr("voxora.stt.AudioRecorder", Recorder)
    monkeypatch.setattr("voxora.stt.GroqTranscriber", Transcriber)
    stt = STT(vad=False)
    stt.start()
    assert stt.stop() == "manual text"
    assert stt._state is _State.COMPLETED


def test_formatting_is_applied_in_vad_and_manual_workflows(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")

    class Recorder:
        def __init__(self, config, logger):
            pass
        def record_vad(self):
            from voxora.audio import AudioResult
            return AudioResult(b"audio", 16000)
        def open(self):
            pass
        def record_manual(self, stop_event):
            stop_event.wait(1)
            from voxora.audio import AudioResult
            return AudioResult(b"audio", 16000)

    class Transcriber:
        def __init__(self, api_key):
            pass
        def transcribe(self, audio, *, model, language=None):
            return "raw transcript"

    class Formatter:
        def __init__(self, api_key, base_url, model):
            pass
        def format(self, text):
            assert text == "raw transcript"
            return "Formatted transcript."

    monkeypatch.setattr("voxora.stt.AudioRecorder", Recorder)
    monkeypatch.setattr("voxora.stt.GroqTranscriber", Transcriber)
    monkeypatch.setattr("voxora.stt.TranscriptFormatter", Formatter)

    vad = STT(format=True)
    assert vad.listen() == "Formatted transcript."
    manual = STT(vad=False, format=True)
    manual.start()
    assert manual.stop() == "Formatted transcript."


def test_formatting_disabled_does_not_initialize_formatter(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    class Transcriber:
        def __init__(self, api_key):
            pass
        def transcribe(self, audio, *, model, language=None):
            return "raw transcript"

    class Recorder:
        def __init__(self, config, logger):
            pass
        def record_vad(self):
            from voxora.audio import AudioResult
            return AudioResult(b"audio", 16000)

    class UnexpectedFormatter:
        def __init__(self, *args):
            raise AssertionError("formatter must remain lazy when disabled")

    monkeypatch.setattr("voxora.stt.AudioRecorder", Recorder)
    monkeypatch.setattr("voxora.stt.GroqTranscriber", Transcriber)
    monkeypatch.setattr("voxora.stt.TranscriptFormatter", UnexpectedFormatter)
    stt = STT()
    assert stt.listen() == "raw transcript"
    assert stt._formatter is None


def test_formatting_failure_clears_session_and_allows_reuse(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    class Recorder:
        def __init__(self, config, logger):
            pass
        def record_vad(self):
            from voxora.audio import AudioResult
            return AudioResult(b"audio", 16000)

    class Transcriber:
        def __init__(self, api_key):
            pass
        def transcribe(self, audio, *, model, language=None):
            return "raw"

    class Formatter:
        calls = 0
        def __init__(self, api_key, base_url, model):
            pass
        def format(self, text):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("format provider failed")
            return "formatted"

    monkeypatch.setattr("voxora.stt.AudioRecorder", Recorder)
    monkeypatch.setattr("voxora.stt.GroqTranscriber", Transcriber)
    monkeypatch.setattr("voxora.stt.TranscriptFormatter", Formatter)
    stt = STT(format=True)
    with pytest.raises(RuntimeError, match="format provider failed"):
        stt.listen()
    assert stt._state is _State.FAILED
    assert stt._recorder is None and stt._audio_result is None and stt._active_session is None
    assert stt.listen() == "formatted"


def test_manual_formatting_failure_clears_session_and_allows_reuse(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    class Recorder:
        def __init__(self, config, logger):
            pass
        def open(self):
            pass
        def record_manual(self, stop_event):
            stop_event.wait(1)
            from voxora.audio import AudioResult
            return AudioResult(b"audio", 16000)

    class Transcriber:
        def __init__(self, api_key):
            pass
        def transcribe(self, audio, *, model, language=None):
            return "raw"

    class Formatter:
        calls = 0
        def __init__(self, api_key, base_url, model):
            pass
        def format(self, text):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("format provider failed")
            return "formatted"

    monkeypatch.setattr("voxora.stt.AudioRecorder", Recorder)
    monkeypatch.setattr("voxora.stt.GroqTranscriber", Transcriber)
    monkeypatch.setattr("voxora.stt.TranscriptFormatter", Formatter)
    stt = STT(vad=False, format=True)
    stt.start()
    with pytest.raises(RuntimeError, match="format provider failed"):
        stt.stop()
    assert stt._state is _State.FAILED
    assert stt._recorder is None and stt._worker is None and stt._audio_result is None
    stt.start()
    assert stt.stop() == "formatted"


def test_concurrent_formatter_initialization_creates_one_formatter(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    constructed = threading.Event()
    allow_construct = threading.Event()
    instances = []

    class Transcriber:
        def __init__(self, api_key):
            pass
        def transcribe(self, audio, *, model, language=None):
            return "raw"

    class Formatter:
        def __init__(self, api_key, base_url, model):
            instances.append(self)
            constructed.set()
            assert allow_construct.wait(1)
        def format(self, text):
            return "formatted"

    monkeypatch.setattr("voxora.stt.GroqTranscriber", Transcriber)
    monkeypatch.setattr("voxora.stt.TranscriptFormatter", Formatter)
    from voxora.audio import AudioResult
    stt = STT(format=True)
    results = []
    first = threading.Thread(target=lambda: results.append(stt._transcribe_audio(AudioResult(b"audio", 16000))))
    second = threading.Thread(target=lambda: results.append(stt._transcribe_audio(AudioResult(b"audio", 16000))))
    first.start()
    assert constructed.wait(1)
    second.start()
    allow_construct.set()
    first.join(1)
    second.join(1)
    assert results == ["formatted", "formatted"]
    assert len(instances) == 1


def test_stt_reuses_one_groq_transcriber(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    instances = []

    class Transcriber:
        def __init__(self, api_key):
            instances.append(self)
        def transcribe(self, audio, *, model, language=None):
            return "text"

    monkeypatch.setattr("voxora.stt.GroqTranscriber", Transcriber)
    from voxora.audio import AudioResult
    stt = STT()
    assert stt._transcribe_audio(AudioResult(b"audio", 16000)) == "text"
    assert stt._transcribe_audio(AudioResult(b"audio", 16000)) == "text"
    assert len(instances) == 1


def test_concurrent_transcription_initialization_creates_one_transcriber(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    constructed = threading.Event()
    allow_construct = threading.Event()
    instances = []

    class Transcriber:
        def __init__(self, api_key):
            instances.append(self)
            constructed.set()
            assert allow_construct.wait(1)
        def transcribe(self, audio, *, model, language=None):
            return "text"

    monkeypatch.setattr("voxora.stt.GroqTranscriber", Transcriber)
    from voxora.audio import AudioResult
    stt = STT()
    results = []
    first = threading.Thread(target=lambda: results.append(stt._transcribe_audio(AudioResult(b"audio", 16000))))
    second = threading.Thread(target=lambda: results.append(stt._transcribe_audio(AudioResult(b"audio", 16000))))
    first.start()
    assert constructed.wait(1)
    second.start()
    allow_construct.set()
    first.join(1)
    second.join(1)
    assert results == ["text", "text"]
    assert len(instances) == 1


def test_vad_provider_failure_clears_session_and_allows_reuse(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")

    class Recorder:
        def __init__(self, config, logger):
            pass
        def record_vad(self):
            from voxora.audio import AudioResult
            return AudioResult(b"audio", 16000)

    class Transcriber:
        calls = 0
        def __init__(self, api_key):
            pass
        def transcribe(self, audio, *, model, language=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("provider failure")
            return "reused transcript"

    monkeypatch.setattr("voxora.stt.AudioRecorder", Recorder)
    monkeypatch.setattr("voxora.stt.GroqTranscriber", Transcriber)
    stt = STT()
    with pytest.raises(RuntimeError, match="provider failure"):
        stt.listen()
    assert stt._state is _State.FAILED
    assert stt._recorder is None and stt._audio_result is None and stt._active_session is None
    assert stt.listen() == "reused transcript"
    assert stt._state is _State.COMPLETED


def test_manual_provider_failure_clears_session_and_worker_before_reuse(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")

    class Recorder:
        def __init__(self, config, logger):
            pass
        def open(self):
            pass
        def record_manual(self, stop_event):
            stop_event.wait(1)
            from voxora.audio import AudioResult
            return AudioResult(b"audio", 16000)

    class Transcriber:
        calls = 0
        def __init__(self, api_key):
            pass
        def transcribe(self, audio, *, model, language=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("provider failure")
            return "reused transcript"

    monkeypatch.setattr("voxora.stt.AudioRecorder", Recorder)
    monkeypatch.setattr("voxora.stt.GroqTranscriber", Transcriber)
    stt = STT(vad=False)
    stt.start()
    with pytest.raises(RuntimeError, match="provider failure"):
        stt.stop()
    assert stt._state is _State.FAILED
    assert stt._recorder is None and stt._worker is None and stt._audio_result is None
    stt.start()
    assert stt.stop() == "reused transcript"
    assert stt._worker is None and stt._state is _State.COMPLETED


def test_explicit_key_wins(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    assert STT(api_key="explicit").config.api_key == "explicit"


@pytest.mark.parametrize("kwargs", [{"engine": "other"}, {"stt_model": ""}])
def test_invalid_core_configuration(monkeypatch, kwargs):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    with pytest.raises(ConfigurationError):
        STT(**kwargs)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        STT()


@pytest.mark.parametrize("kwargs", [
    {"vad_mode": -1}, {"vad_mode": 4}, {"sample_rate": 44100},
    {"frame_duration_ms": 15}, {"silence_duration": 0},
    {"pre_speech_duration": -1}, {"max_recording_duration": 0},
    {"speech_start_timeout": 0},
])
def test_audio_configuration_validation(monkeypatch, kwargs):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    with pytest.raises(ConfigurationError):
        STT(**kwargs)


@pytest.mark.parametrize("name", ["silence_duration", "pre_speech_duration", "max_recording_duration", "speech_start_timeout"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_durations_rejected(monkeypatch, name, value):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    with pytest.raises(ConfigurationError):
        STT(**{name: value})


@pytest.mark.parametrize("name", ["sample_rate", "frame_duration_ms"])
@pytest.mark.parametrize("value", [16000.0, True])
def test_non_integer_audio_settings_rejected(monkeypatch, name, value):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    with pytest.raises(ConfigurationError):
        STT(**{name: value})


def test_optional_timeouts_preserve_none(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    stt = STT(max_recording_duration=None, speech_start_timeout=None)
    assert stt.config.max_recording_duration is None
    assert stt.config.speech_start_timeout is None


def test_formatter_key_fallback_and_validation(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    assert STT(format=False, format_base_url="", format_model="", format_api_key="").config.format_model == ""
    assert STT(format=True).config.format_api_key == "key"
    with pytest.raises(ConfigurationError):
        STT(format=True, format_model="")


def test_workflow_restrictions(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    class FakeRecorder:
        def __init__(self, config, logger):
            pass
        def record_vad(self):
            from voxora.audio import AudioResult
            return AudioResult(b"audio", 16000)

    monkeypatch.setattr("voxora.stt.AudioRecorder", FakeRecorder)
    vad = STT()
    vad._transcribe_audio = lambda result: "text"
    with pytest.raises(ValueError):
        vad.start()
    with pytest.raises(ValueError):
        vad.stop()
    assert vad.listen() == "text"

    manual = STT(vad=False)
    with pytest.raises(ValueError):
        manual.listen()
    with pytest.raises(RecordingStateError):
        manual.stop()


def test_start_launches_worker_and_stop_reaches_transcription_boundary(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    class FakeRecorder:
        def __init__(self, config, logger):
            self.event = None
        def open(self):
            pass
        def record_manual(self, event):
            self.event = event
            event.wait(1)
            from voxora.audio import AudioResult
            return AudioResult(b"audio", 16000)
        def close(self):
            pass

    monkeypatch.setattr("voxora.stt.AudioRecorder", FakeRecorder)
    stt = STT(vad=False)
    stt._transcribe_audio = lambda result: "text"
    stt.start()
    assert stt._state is _State.RECORDING
    assert stt.stop() == "text"
    assert stt._state is _State.COMPLETED


def test_active_state_rejects_repeated_start_and_stop_is_unimplemented(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    stt = STT(vad=False)
    stt._state = _State.RECORDING
    with pytest.raises(RecordingStateError):
        stt.start()
    with pytest.raises(RecordingStateError):
        stt.stop()


def test_instance_logger_does_not_configure_root_or_package_logger(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    root_handlers = list(logging.getLogger().handlers)
    package_logger = logging.getLogger("voxora")
    package_level = package_logger.level
    first = STT(log_level=logging.INFO)
    second = STT(log_level=logging.DEBUG)
    assert first._logger.name.startswith("voxora.stt.")
    assert first._logger.level == logging.INFO
    assert second._logger.level == logging.DEBUG
    assert package_logger.level == package_level
    assert list(logging.getLogger().handlers) == root_handlers


class ManualRecorderFake:
    def __init__(self, config, logger, *, failure=None, started=None, completed=None):
        self.failure = failure
        self.started = started
        self.completed = completed
        self.closed = False

    def open(self):
        pass

    def record_manual(self, stop_event):
        if self.started:
            self.started.set()
        if self.failure:
            raise self.failure
        stop_event.wait(1)
        if self.completed:
            self.completed.set()
        from voxora.audio import AudioResult
        return AudioResult(b"audio", 16000)

    def close(self):
        self.closed = True


def test_two_concurrent_stops_only_one_claims_session(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    started = threading.Event()
    transcribing = threading.Event()
    release_transcription = threading.Event()
    monkeypatch.setattr("voxora.stt.AudioRecorder", lambda config, logger: ManualRecorderFake(config, logger, started=started))
    stt = STT(vad=False)
    stt._transcribe_audio = lambda result: (transcribing.set(), release_transcription.wait(1), "ok")[2]
    stt.start()
    assert started.wait(1)
    first_result = []
    first = threading.Thread(target=lambda: first_result.append(stt.stop()))
    first.start()
    assert transcribing.wait(1)
    with pytest.raises(RecordingStateError):
        stt.stop()
    release_transcription.set()
    first.join(1)
    assert first_result == ["ok"]
    assert stt._worker is None


def test_worker_failure_propagates_and_failed_instance_is_reusable(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    calls = []
    started = threading.Event()

    def factory(config, logger):
        calls.append(len(calls))
        return ManualRecorderFake(config, logger, failure=RuntimeError("capture failed") if len(calls) == 1 else None, started=started)

    monkeypatch.setattr("voxora.stt.AudioRecorder", factory)
    stt = STT(vad=False)
    stt._transcribe_audio = lambda result: "ok"
    stt.start()
    assert started.wait(1)
    with pytest.raises(RuntimeError, match="capture failed"):
        stt.stop()
    assert stt._state is _State.FAILED
    stt.start()
    assert stt.stop() == "ok"
    assert stt._state is _State.COMPLETED


def test_maximum_duration_worker_can_be_stopped_after_it_finishes(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    completed = threading.Event()
    monkeypatch.setattr("voxora.stt.AudioRecorder", lambda config, logger: ManualRecorderFake(config, logger, completed=completed))
    stt = STT(vad=False)
    stt._transcribe_audio = lambda result: "ok"
    stt.start()
    stt._stop_event.set()
    assert completed.wait(1)
    assert stt.stop() == "ok"
    assert stt._worker is None


def test_start_failure_and_thread_start_failure_clean_up(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")

    class StartFailRecorder(ManualRecorderFake):
        def open(self):
            raise RuntimeError("microphone unavailable")

    monkeypatch.setattr("voxora.stt.AudioRecorder", StartFailRecorder)
    stt = STT(vad=False)
    with pytest.raises(RuntimeError, match="microphone unavailable"):
        stt.start()
    assert stt._state is _State.FAILED

    closed = []
    recorder = ManualRecorderFake(None, None)
    recorder.close = lambda: closed.append(True)
    monkeypatch.setattr("voxora.stt.AudioRecorder", lambda config, logger: recorder)

    class BrokenThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            raise RuntimeError("thread start failed")

    monkeypatch.setattr("voxora.stt.threading.Thread", BrokenThread)
    with pytest.raises(RuntimeError, match="thread start failed"):
        stt.start()
    assert closed == [True]
    assert stt._state is _State.FAILED


def test_old_manual_stop_cannot_clear_a_new_session(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    started = threading.Event()
    transcribing = threading.Event()
    release = threading.Event()
    recorders = []

    def factory(config, logger):
        recorder = ManualRecorderFake(config, logger, started=started)
        recorders.append(recorder)
        return recorder

    monkeypatch.setattr("voxora.stt.AudioRecorder", factory)
    stt = STT(vad=False)
    calls = []

    def transcribe(result):
        calls.append(result)
        if len(calls) == 1:
            transcribing.set()
            release.wait(1)
        return "ok"

    stt._transcribe_audio = transcribe
    stt.start()
    assert started.wait(1)
    first_result = []
    stopper = threading.Thread(target=lambda: first_result.append(stt.stop()))
    stopper.start()
    assert transcribing.wait(1)
    with pytest.raises(RecordingStateError):
        stt.start()
    release.set()
    stopper.join(1)
    assert first_result == ["ok"]
    assert stt._recorder is None and stt._worker is None and stt._stop_event is None
    stt.start()
    assert stt._recorder is recorders[1]
    assert stt.stop() == "ok"
    assert stt._recorder is None and stt._active_session is None


def test_old_vad_listen_cannot_clear_a_new_listen(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    transcribing = threading.Event()
    release = threading.Event()
    calls = []

    class VadRecorder:
        def __init__(self, config, logger):
            pass
        def record_vad(self):
            from voxora.audio import AudioResult
            return AudioResult(b"audio", 16000)

    monkeypatch.setattr("voxora.stt.AudioRecorder", VadRecorder)
    stt = STT(vad=True)

    def transcribe(result):
        calls.append(result)
        if len(calls) == 1:
            transcribing.set()
            release.wait(1)
        return "ok"

    stt._transcribe_audio = transcribe
    first_result = []
    first = threading.Thread(target=lambda: first_result.append(stt.listen()))
    first.start()
    assert transcribing.wait(1)
    with pytest.raises(RecordingStateError):
        stt.listen()
    release.set()
    first.join(1)
    assert first_result == ["ok"]
    assert stt._recorder is None and stt._active_session is None
    assert stt.listen() == "ok"


def test_failed_transcription_cleanup_is_atomic_before_reuse(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    started = threading.Event()
    transcribing = threading.Event()
    release = threading.Event()
    monkeypatch.setattr("voxora.stt.AudioRecorder", lambda config, logger: ManualRecorderFake(config, logger, started=started))
    stt = STT(vad=False)

    def fail_once(result):
        transcribing.set()
        release.wait(1)
        raise RuntimeError("transcription failed")

    stt._transcribe_audio = fail_once
    stt.start()
    assert started.wait(1)
    outcome = []

    def stop_and_capture():
        try:
            outcome.append(stt.stop())
        except Exception as exc:
            outcome.append(exc)

    stopper = threading.Thread(target=stop_and_capture)
    stopper.start()
    assert transcribing.wait(1)
    with pytest.raises(RecordingStateError):
        stt.start()
    release.set()
    stopper.join(1)
    assert len(outcome) == 1 and isinstance(outcome[0], RuntimeError)
    assert stt._state is _State.FAILED
    assert stt._recorder is None and stt._worker is None and stt._active_session is None
