# Install necessary libraries if running in a new environment
!pip install openai-whisper gradio gtts groq pydub
!apt-get install ffmpeg  # For pydub to handle audio conversions

import os
import gradio as gr
import whisper
from gtts import gTTS
from pydub import AudioSegment
from groq import Groq

# Set up API key for Groq (replace with your actual key if not using environment variable)
GROQ_API_KEY = "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"  # Replace this with your actual Groq API key

# Initialize Whisper model for transcription
try:
    model = whisper.load_model("base")  # Choose model size based on speed and accuracy needed
    print("Whisper model loaded successfully.")
except Exception as e:
    print("Error loading Whisper model:", e)
    raise e

# Set up Groq client with API key
try:
    client = Groq(api_key=GROQ_API_KEY)
    print("Groq client initialized successfully.")
except Exception as e:
    print("Error setting up Groq client:", e)
    raise e

# Function to convert microphone audio to a compatible format
def convert_audio(audio_path):
    try:
        # Load the audio file and convert it to a standard .wav format
        audio = AudioSegment.from_file(audio_path)
        converted_path = "converted_audio.wav"
        audio.export(converted_path, format="wav", parameters=["-ar", "16000"])  # 16 kHz sample rate
        print("Audio successfully converted to .wav format.")
        return converted_path
    except Exception as e:
        print("Error converting audio:", e)
        return None

# Function for voice-to-voice chatbot
def voice_chat(audio):
    # Convert the audio to a compatible format
    converted_audio_path = convert_audio(audio)
    if not converted_audio_path:
        return "Error: Could not process audio. Please try again.", None

    # Step 1: Transcribe audio input
    try:
        print("Attempting to transcribe audio...")
        transcription_result = model.transcribe(converted_audio_path)  # Full output from Whisper
        transcription = transcription_result.get("text", None)

        if not transcription:
            print("No transcription found. Raw output:", transcription_result)
            return "Error: Transcription failed. No text found.", None

        print(f"Transcription successful: {transcription}")
    except Exception as e:
        print("Error during transcription:", e)
        return "Error during transcription. Please try again.", None

    # Step 2: Interact with Groq API for LLM response
    try:
        print("Sending transcription to Groq API...")
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": transcription}],
            model="llama3-8b-8192"
        )
        response = chat_completion.choices[0].message.content
        print(f"LLM Response: {response}")
    except Exception as e:
        print("Error during Groq API interaction:", e)
        return "Error during Groq API interaction. Please check API key or internet connection.", None

    # Step 3: Convert LLM response to speech using gTTS and save to temporary file
    try:
        print("Converting text to speech...")
        tts = gTTS(response, lang='en')
        audio_output_path = "response_audio.mp3"
        tts.save(audio_output_path)
        print("Audio response generated and saved.")
    except Exception as e:
        print("Error generating audio response:", e)
        return response, "Error generating audio response. Text output available."

    return response, audio_output_path

# Gradio interface setup with debug mode enabled
interface = gr.Interface(
    fn=voice_chat,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Textbox(label="Response"), gr.Audio(label="Audio Response")]
)

# Launch the Gradio app
interface.launch(share=True, debug=True)
