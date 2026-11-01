from voxora import STT


stt = STT(api_key="YOUR_GROQ_API_KEY", vad=True)
text = stt.listen()
print(text)

# To enable optional formatting, construct STT with format=True and configure
# format_model, format_base_url, and optionally format_api_key.
