# Voxora

Voxora is a standalone, microphone-based Python speech-to-text SDK. Its
intended public class is `voxora.STT`.

Phase 2 implements microphone capture, in-memory PCM buffering, WebRTC VAD,
pre-speech buffering, silence detection, timeouts, and manual recording
concurrency. Groq transcription and formatting are not implemented yet, so
recording methods reach an explicit internal `NotImplementedError` boundary
after audio preparation.

VAD-enabled `listen()` waits for speech, retains the configured pre-speech
buffer, and stops after configured silence. `speech_start_timeout` raises
`NoSpeechError` when speech never starts. `max_recording_duration` stops a
recording without treating the duration limit itself as an error when usable
audio was captured. Manual `start()` runs capture in a background thread;
`stop()` signals and joins it before reaching the transcription boundary.

Audio is kept in memory as mono 16-bit PCM and is never exposed by the public
API or written to disk. PyAudio and WebRTC VAD are required system/runtime
dependencies; unit tests use fakes and do not require a physical microphone.
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
