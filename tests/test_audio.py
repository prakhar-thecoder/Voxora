from types import SimpleNamespace

import pytest

from voxora.audio import AudioRecorder
from voxora.exceptions import AudioCaptureError, NoSpeechError


class FakeStream:
    def __init__(self, frames=None, error=None):
        self.frames = iter(frames or [])
        self.error = error
        self.stopped = False
        self.closed = False

    def read(self, _size, exception_on_overflow=False):
        if self.error:
            raise self.error
        return next(self.frames)

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeAudio:
    def __init__(self, stream):
        self.stream = stream
        self.terminated = False
        self.open_args = None

    def open(self, **kwargs):
        self.open_args = kwargs
        return self.stream

    def terminate(self):
        self.terminated = True


class FakePyAudio:
    paInt16 = "paInt16"

    def __init__(self, stream):
        self.instance = FakeAudio(stream)

    def PyAudio(self):
        return self.instance


class FakeVAD:
    instances = []

    def __init__(self, mode, results):
        self.mode = mode
        self.results = iter(results)
        self.__class__.instances.append(self)

    def is_speech(self, _frame, _rate):
        return next(self.results)


def config(**overrides):
    values = dict(sample_rate=16000, frame_duration_ms=20, pre_speech_duration=0.04,
                  silence_duration=0.04, speech_start_timeout=None, max_recording_duration=None,
                  vad_mode=3)
    values.update(overrides)
    return SimpleNamespace(**values)


def recorder_with(stream, cfg, vad_results=None, clock=lambda: 0.0):
    audio = FakePyAudio(stream)
    vad_factory = lambda mode: FakeVAD(mode, vad_results or [])
    logger = SimpleNamespace(debug=lambda *_: None)
    return AudioRecorder(cfg, logger, pyaudio_module=audio, vad_factory=vad_factory, clock=clock), audio


def test_vad_capture_uses_correct_parameters_and_prespeech_buffer():
    frame = lambda value: bytes([value]) * 640
    stream = FakeStream([frame(1), frame(2), frame(3), frame(4), frame(5), frame(6), frame(7)])
    recorder, audio = recorder_with(stream, config(), [False, False, False, True, True, False, False])
    result = recorder.record_vad()
    assert result.pcm_data == b"".join(frame(i) for i in (2, 3, 4, 5, 6, 7))
    assert audio.instance.open_args == {"format": "paInt16", "channels": 1, "rate": 16000,
                                        "input": True, "frames_per_buffer": 320}
    assert stream.stopped and stream.closed and audio.instance.terminated
    assert FakeVAD.instances[-1].mode == 3


def test_initial_silence_does_not_end_recording():
    frame = lambda value: bytes([value]) * 640
    stream = FakeStream([frame(1), frame(2), frame(3), frame(4), frame(5)])
    recorder, _ = recorder_with(stream, config(), [False, False, True, False, False])
    result = recorder.record_vad()
    assert result.pcm_data == b"".join(frame(i) for i in (1, 2, 3, 4, 5))


def test_vad_maximum_duration_stops_after_speech():
    frame = b"m" * 640
    stream = FakeStream([frame, frame, frame])
    ticks = iter([0.0, 0.0, 0.0, 0.2])
    recorder, _ = recorder_with(
        stream,
        config(max_recording_duration=0.1),
        [False, True, True, True],
        clock=lambda: next(ticks),
    )
    result = recorder.record_vad()
    assert result.pcm_data == frame * 2


def test_no_speech_timeout_cleans_up():
    frame = b"x" * 640
    stream = FakeStream([frame, frame])
    ticks = iter([0.0, 0.0, 0.2])
    recorder, audio = recorder_with(stream, config(speech_start_timeout=0.1), [False, False, False], clock=lambda: next(ticks))
    with pytest.raises(NoSpeechError):
        recorder.record_vad()
    assert stream.closed and audio.instance.terminated


def test_speech_before_timeout_is_accepted():
    frame = lambda value: bytes([value]) * 640
    stream = FakeStream([frame(1), frame(2), frame(3), frame(4)])
    ticks = iter([0.0, 0.05, 0.09, 0.09, 0.09])
    recorder, _ = recorder_with(
        stream, config(speech_start_timeout=0.1), [False, True, False, False], clock=lambda: next(ticks)
    )
    result = recorder.record_vad()
    assert result.pcm_data == b"".join(frame(i) for i in (1, 2, 3, 4))


@pytest.mark.parametrize("speech_time", [0.1, 0.11])
def test_speech_at_or_after_timeout_is_rejected(speech_time):
    frame = b"t" * 640
    stream = FakeStream([frame, frame])
    ticks = iter([0.0, 0.0, speech_time])
    recorder, audio = recorder_with(
        stream, config(speech_start_timeout=0.1), [False, True], clock=lambda: next(ticks)
    )
    with pytest.raises(NoSpeechError):
        recorder.record_vad()
    assert stream.closed and audio.instance.terminated


def test_read_failure_and_open_failure_clean_up():
    stream = FakeStream(error=RuntimeError("read"))
    recorder, audio = recorder_with(stream, config(), [False])
    with pytest.raises(AudioCaptureError):
        recorder.record_vad()
    assert stream.closed and audio.instance.terminated

    class BrokenAudio(FakePyAudio):
        def PyAudio(self):
            raise RuntimeError("open")

    broken = AudioRecorder(config(), SimpleNamespace(debug=lambda *_: None), pyaudio_module=BrokenAudio(stream))
    with pytest.raises(AudioCaptureError):
        broken.open()


def test_manual_capture_with_stop_event_cleans_up():
    frame = b"z" * 640
    stream = FakeStream([frame, frame])
    recorder, audio = recorder_with(stream, config(), clock=lambda: 0.0)
    recorder.open()
    import threading
    event = threading.Event()
    event.set()
    with pytest.raises(NoSpeechError):
        recorder.record_manual(event)
    assert stream.closed and audio.instance.terminated


def test_manual_maximum_duration_returns_captured_audio():
    frame = b"q" * 640
    stream = FakeStream([frame])
    ticks = iter([0.0, 0.0, 0.2])
    recorder, audio = recorder_with(stream, config(max_recording_duration=0.1), clock=lambda: next(ticks))
    recorder.open()
    import threading
    result = recorder.record_manual(threading.Event())
    assert result.pcm_data == frame
    assert stream.closed and audio.instance.terminated
