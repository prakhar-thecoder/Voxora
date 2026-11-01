import io
import wave

import pytest

from voxora.audio import AudioResult
from voxora.exceptions import TranscriptionError
from voxora.groq import GroqTranscriber, audio_result_to_wav


def test_audio_result_to_wav_preserves_metadata_and_pcm():
    pcm = b"\x01\x02\x03\x04"
    output = audio_result_to_wav(AudioResult(pcm, sample_rate=8000, channels=1, sample_width=2))
    assert isinstance(output, io.BytesIO)
    assert output.name == "voxora-recording.wav"
    with wave.open(output, "rb") as wav_file:
        assert wav_file.getframerate() == 8000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(wav_file.getnframes()) == pcm


@pytest.mark.parametrize("audio", [
    AudioResult(b"", 16000),
    AudioResult(b"\x00", 16000, channels=1, sample_width=2),
])
def test_invalid_audio_is_rejected_without_disk_io(audio):
    with pytest.raises(TranscriptionError):
        audio_result_to_wav(audio)


@pytest.mark.parametrize("audio", [
    AudioResult(b"\0", 16000, channels=1, sample_width=5),
    AudioResult(b"\0\0", 0),
    AudioResult(b"\0\0", 16000, channels=70000),
])
def test_malformed_wav_metadata_is_rejected(audio):
    with pytest.raises(TranscriptionError):
        audio_result_to_wav(audio)


class FakeTranscriptions:
    def __init__(self, response):
        self.response = response
        self.request = None

    def create(self, **request):
        self.request = request
        return self.response


class FakeClient:
    def __init__(self, response):
        self.audio = type("Audio", (), {})()
        self.audio.transcriptions = FakeTranscriptions(response)


def test_groq_request_uses_model_and_optional_language():
    client = FakeClient(type("Response", (), {"text": "hello"})())
    transcriber = GroqTranscriber("secret", client=client)
    result = transcriber.transcribe(AudioResult(b"\0\0", 16000), model="whisper-test", language="en")
    request = client.audio.transcriptions.request
    assert result == "hello"
    assert request["model"] == "whisper-test"
    assert request["language"] == "en"
    assert request["file"].name == "voxora-recording.wav"


def test_groq_request_omits_language_when_not_configured():
    client = FakeClient({"text": "hello"})
    transcriber = GroqTranscriber("secret", client=client)
    assert transcriber.transcribe(AudioResult(b"\0\0", 16000), model="whisper-test") == "hello"
    assert "language" not in client.audio.transcriptions.request


@pytest.mark.parametrize("response_text, expected", [("  hello world  ", "hello world"), ("Hello, Voxora!", "Hello, Voxora!")])
def test_transcription_text_is_trimmed_without_other_changes(response_text, expected):
    client = FakeClient(type("Response", (), {"text": response_text})())
    result = GroqTranscriber("secret", client=client).transcribe(
        AudioResult(b"\0\0", 16000), model="whisper-test"
    )
    assert result == expected


@pytest.mark.parametrize("response", [None, {}, {"text": ""}, {"text": "   "}, type("Response", (), {})()])
def test_empty_or_malformed_response_raises(response):
    with pytest.raises(TranscriptionError):
        GroqTranscriber("secret", client=FakeClient(response)).transcribe(
            AudioResult(b"\0\0", 16000), model="whisper-test"
        )


def test_provider_exception_propagates():
    class FailingTranscriptions:
        def create(self, **request):
            raise RuntimeError("provider failure")

    client = FakeClient("unused")
    client.audio.transcriptions = FailingTranscriptions()
    with pytest.raises(RuntimeError, match="provider failure"):
        GroqTranscriber("secret", client=client).transcribe(AudioResult(b"\0\0", 16000), model="whisper-test")


def test_client_is_initialized_with_api_key(monkeypatch):
    calls = []

    class FakeGroq:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("voxora.groq.Groq", FakeGroq)
    GroqTranscriber("secret")
    assert calls == [{"api_key": "secret"}]
