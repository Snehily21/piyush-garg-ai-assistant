import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
import whisper
import asyncio
import edge_tts
import soundfile as sf
import io
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# ---------------- CONFIG ----------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

VOICE = "en-IN-PrabhatNeural"
whisper_model = whisper.load_model("base")

# ---------------- UI ----------------
st.set_page_config(page_title="Piyush Garg AI", page_icon="🎓")

st.title("Hi I am Piyush Garg")
st.subheader("You can ask your queries and I will solve that")

st.divider()

# Text input
text_query = st.text_input("Ask your question (Text)")

# Voice input
audio_file = st.audio_input("Or ask using your voice 🎤")

ask_btn = st.button("Ask")

# ---------------- TTS ----------------
async def speak(text):
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate="+28%",
        pitch="+15Hz",
        volume="+100%"
    )

    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    sd.play(data, samplerate)
    sd.wait()

# ---------------- MAIN LOGIC ----------------
if ask_btn:

    user_query = None

    # Case 1: Voice input
    if audio_file is not None:
        with st.spinner("Listening..."):
            audio_bytes = audio_file.read()
            with open("input.wav", "wb") as f:
                f.write(audio_bytes)

            result = whisper_model.transcribe("input.wav")
            user_query = result["text"]

        st.info(f"🎤 You said: {user_query}")

    # Case 2: Text input
    elif text_query.strip():
        user_query = text_query

    else:
        st.warning("Please provide text or voice input")

    if user_query:

        prompt = f"""
Act like you are piyush garg who is tutor of chai code channel and have expertise in GenAi.
you have to reply the user query in piyush garg's style.

Here are some information about piyush garg:

1. He uses "nice nice" word very often
2. He believes in hands-on practice
3. He always ends the lecture by saying "milte hai aapko next class ke andar"
4. He is graduated from chitkara University rajpura branch
5. Right now he is teaching in gen Ai cohort
6. he uses hinglish as a medium to talk
7. Output mein koi symbol ya emoji mat dena

Question:
{user_query}
"""

        with st.spinner("Thinking..."):
            response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt
            )

        answer = response.text

        st.success("Answer")
        st.write(answer)

        # Speak answer
        asyncio.run(speak(answer))
