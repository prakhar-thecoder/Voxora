from voxora import STT


stt = STT(api_key="YOUR_GROQ_API_KEY", vad=True)
text = stt.listen()  # Available after VAD recording is implemented.
print(text)
