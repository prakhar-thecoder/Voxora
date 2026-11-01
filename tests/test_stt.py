import logging

import pytest

from voxora import ConfigurationError, RecordingStateError, STT


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


def test_formatter_key_fallback_and_validation(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
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


def test_repeated_start_rejected(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    stt = STT(vad=False)
    with pytest.raises(NotImplementedError):
        stt.start()
    with pytest.raises(RecordingStateError):
        stt.start()


def test_named_logger_does_not_configure_root(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")
    root_handlers = list(logging.getLogger().handlers)
    STT(log_level=logging.INFO)
    assert logging.getLogger("voxora").level == logging.INFO
    assert list(logging.getLogger().handlers) == root_handlers
