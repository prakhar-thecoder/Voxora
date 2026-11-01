# Voxora

**A lightweight, microphone-based speech-to-text SDK for Python.**

Voxora makes it easy to add speech recognition to Python applications with automatic voice detection, manual recording controls, and optional AI-powered transcript formatting.

## Features

- **Automatic recording:** Detect speech and automatically stop after silence.
- **Manual recording:** Control recording with `start()` and `stop()`.
- **Speech-to-text:** Transcribe audio using Groq Whisper.
- **Transcript formatting:** Optionally improve punctuation, capitalization, and paragraph structure.
- **In-memory processing:** Audio is processed in memory without writing recordings to disk.
- **Simple API:** A single `STT` class for both recording modes.

## Installation

```bash
pip install voxora
```

Set your Groq API key:

```bash
export GROQ_API_KEY="your-groq-api-key"
```

Voxora requires Python 3.10 or newer and microphone/audio dependencies. PyAudio may require platform-specific system libraries.

## Quick Start

### Automatic recording (VAD)

Voxora automatically detects speech and stops recording after the configured silence duration.

```python
from voxora import STT

stt = STT(
    api_key="YOUR_GROQ_API_KEY",
    vad=True
)

text = stt.listen()

print(text)
```

### Manual recording

For applications that need explicit recording controls:

```python
from voxora import STT

stt = STT(
    api_key="YOUR_GROQ_API_KEY",
    vad=False
)

stt.start()

input("Press Enter to stop recording...")

text = stt.stop()

print(text)
```

## Transcript Formatting

Voxora can optionally format transcribed text using a compatible chat-completions API.

Formatting improves punctuation, capitalization, and paragraph breaks while preserving the original spoken words and meaning.

```python
from voxora import STT

stt = STT(
    api_key="YOUR_GROQ_API_KEY",
    vad=True,
    format=True,
    format_model="openai/gpt-oss-120b",
    format_base_url="https://api.groq.com/openai/v1"
)

text = stt.listen()

print(text)
```

Formatting is disabled by default. The formatter receives the transcript after speech recognition; it does not receive the audio.

## Configuration

| Parameter | Description |
|---|---|
| `api_key` | Groq API key. Defaults to `GROQ_API_KEY`. |
| `stt_model` | Groq transcription model. |
| `language` | Optional ISO-639-1 language hint. |
| `vad` | Enables automatic voice activity detection. |
| `vad_mode` | WebRTC VAD aggressiveness (0–3). |
| `silence_duration` | Silence duration before automatic recording stops. |
| `pre_speech_duration` | Audio retained before speech detection. |
| `max_recording_duration` | Maximum recording duration. |
| `speech_start_timeout` | Maximum wait for speech to begin. |
| `format` | Enables optional transcript formatting. |
| `format_base_url` | Compatible formatting API endpoint. |
| `format_model` | Formatting model. |
| `format_api_key` | Formatting API key; defaults to `api_key`. |

## Error Handling

Voxora exposes exceptions for configuration, recording, transcription, and formatting failures.

Provider authentication, network, rate-limit, and other API errors may propagate from the underlying SDK.

## Requirements

- Python 3.10+
- PyAudio
- WebRTC VAD
- Groq API key
- Internet connection for transcription and optional formatting

Voxora uses Groq's cloud transcription API and is not an offline speech recognition engine.

## License

See [LICENSE](LICENSE).

---

**Voxora** — Speech recognition, made simple.