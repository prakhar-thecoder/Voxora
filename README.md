# Voxora

Voxora is a standalone, microphone-based Python speech-to-text SDK. Its
intended public class is `voxora.STT`.

From a Voxora source checkout, install with `pip install .` and configure a
Groq API key explicitly or with the `GROQ_API_KEY` environment variable. This
is not a PyPI installation instruction; Voxora is not published on PyPI.

```bash
export GROQ_API_KEY="your-key"
```

Phase 2 implements microphone capture, in-memory PCM buffering, WebRTC VAD,
pre-speech buffering, silence detection, timeouts, and manual recording
concurrency. Groq Whisper transcription is now integrated; formatting remains
deferred.

VAD-enabled `listen()` waits for speech, retains the configured pre-speech
buffer, and stops after configured silence. `speech_start_timeout` raises
`NoSpeechError` when speech never starts. `max_recording_duration` stops a
recording without treating the duration limit itself as an error when usable
audio was captured. Manual `start()` runs capture in a background thread;
`stop()` signals and joins it before requesting transcription.

The default transcription model is `whisper-large-v3-turbo`. Configure a
different supported model and optional ISO-639-1 language hint with
`stt_model` and `language`. Audio is sent to Groq's cloud transcription API;
Voxora is not an offline transcription engine.

Empty or malformed provider response text raises `TranscriptionError`.
Authentication, network, rate-limit, timeout, and other provider exceptions
may propagate from the Groq SDK with their original details.

Audio is kept in memory as mono 16-bit PCM and is never exposed by the public
API or written to disk. PyAudio and WebRTC VAD are required system/runtime
dependencies; unit tests use fakes and do not require a physical microphone.
The internal WAV converter preserves PCM metadata and accepts standard
1-to-4-byte PCM sample widths; the microphone recorder currently emits mono,
16-bit PCM.
The `webrtcvad-wheels` distribution provides the `webrtcvad` import and its
current wheel release requires Python 3.10 or newer.

`log_level` applies only to the instance's `voxora.stt.<instance>` child
logger. Voxora does not configure the root logger, install handlers, or change
application logging configuration; multiple `STT` instances can therefore
use different levels without affecting one another.

```python
from voxora import STT

stt = STT(engine="groq", api_key="YOUR_GROQ_API_KEY", vad=True)
text = stt.listen()
```

For manually controlled recording:

```python
stt = STT(engine="groq", api_key="YOUR_GROQ_API_KEY", vad=False)
stt.start()
text = stt.stop()
```

Voxora does not include Wayland, GTK, clipboard, keyboard simulation, IPC, or
desktop notification functionality. It is not production-ready and is not
published on PyPI.
