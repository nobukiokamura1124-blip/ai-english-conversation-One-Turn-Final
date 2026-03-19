import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
import tempfile
import soundfile as sf
from openai import OpenAI
import queue

st.title("AI英会話（1ターン）")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
})

webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    audio_receiver_size=256,
)

# 音声フレームを受信してバッファにためる
if webrtc_ctx.audio_receiver:
    try:
        frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.2)
        for frame in frames:
            audio = frame.to_ndarray()

            # モノラル化
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            # int16化
            if audio.dtype != np.int16:
                if np.issubdtype(audio.dtype, np.floating):
                    audio = np.clip(audio, -1.0, 1.0)
                    audio = (audio * 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)

            st.session_state.audio_buffer.append(audio)

    except queue.Empty:
        pass

if st.button("録音終了 → AIに送る"):
    if len(st.session_state.audio_buffer) == 0:
        st.warning("音声が取れてない。マイクON確認")
    else:
        audio_data = st.session_state.audio_buffer.copy()
        st.session_state.audio_buffer = []

        audio_np = np.concatenate(audio_data, axis=0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, audio_np, 48000)
            filename = f.name

        try:
            with open(filename, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=f
                )

            user_text = transcript.text
            st.write("🧑‍💬", user_text)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly English conversation partner. Keep replies short and natural."
                    },
                    {
                        "role": "user",
                        "content": user_text
                    }
                ]
            )

            ai_text = response.choices[0].message.content
            st.write("🤖", ai_text)

            speech = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=ai_text
            )

            with open("response.mp3", "wb") as f:
                f.write(speech.content)

            st.audio("response.mp3")

        except Exception as e:
            st.error(f"エラー: {e}")