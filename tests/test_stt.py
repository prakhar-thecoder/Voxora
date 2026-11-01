import logging

import pytest

from voxora import ConfigurationError, RecordingStateError, STT
from voxora.stt import _State


def test_public_import_and_defaults(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    stt = STT()
    assert stt.config.stt_model == "whisper-large-v3-turbo"
    assert stt.config.vad is True
    assert stt.config.api_key == "env-key"


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
    vad = STT()
    with pytest.raises(ValueError):
        vad.start()
    with pytest.raises(ValueError):
        vad.stop()
    with pytest.raises(NotImplementedError):
        vad.listen()

    manual = STT(vad=False)
    with pytest.raises(ValueError):
        manual.listen()
    with pytest.raises(RecordingStateError):
        manual.stop()


def test_start_does_not_change_state(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    stt = STT(vad=False)
    with pytest.raises(NotImplementedError):
        stt.start()
    assert stt._state is _State.IDLE
    with pytest.raises(RecordingStateError):
        stt.stop()


def test_active_state_rejects_repeated_start_and_stop_is_unimplemented(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    stt = STT(vad=False)
    stt._state = _State.RECORDING
    with pytest.raises(RecordingStateError):
        stt.start()
    with pytest.raises(NotImplementedError):
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
