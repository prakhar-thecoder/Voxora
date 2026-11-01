# Voxora

Voxora is a standalone, microphone-based Python speech-to-text SDK. Its
intended public class is `voxora.STT`.

Phase 1 establishes the package, configuration validation, lifecycle API, and
tests. Actual microphone capture, VAD, Groq transcription, and formatting are
not implemented yet, so the recording methods currently fail explicitly.

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
