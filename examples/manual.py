from voxora import STT


stt = STT(api_key="YOUR_GROQ_API_KEY", vad=False)
stt.start()
text = stt.stop()
print(text)
