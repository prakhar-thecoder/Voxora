from voxora import STT


stt = STT(api_key="YOUR_GROQ_API_KEY", vad=False)
stt.start()  # Available after microphone recording is implemented.
text = stt.stop()
print(text)
