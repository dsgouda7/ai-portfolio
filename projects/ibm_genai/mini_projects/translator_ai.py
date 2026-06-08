"""
Translator AI: Speech-to-Text Translation Service
Uses local Hugging Face models:
- Whisper for speech-to-text
- Helsinki-NLP for translation
- Facebook MMS-TTS for text-to-speech
"""

import base64
import json
from flask import Flask, render_template, request
from flask_cors import CORS
import os
import torch
import io
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
import soundfile as sf

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize models globally for efficiency
print("Loading models... This may take a few minutes on first run.")

# Speech-to-Text: Whisper (CPU-friendly)
print("Loading Whisper for speech recognition...")
speech_recognizer = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device=-1  # CPU
)

# Translation: Helsinki-NLP models (English to Spanish)
print("Loading translation model...")
translation_model_name = "Helsinki-NLP/opus-mt-en-es"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

# Text-to-Speech: Using a simple TTS pipeline
print("Loading text-to-speech model...")
try:
    # Try using facebook/mms-tts-spa for Spanish TTS
    tts_pipeline = pipeline(
        "text-to-speech",
        model="facebook/mms-tts-spa",
        device=-1
    )
    tts_available = True
except Exception as e:
    print(f"TTS model not available: {e}")
    print("TTS will be disabled. Install additional dependencies if needed.")
    tts_available = False

print("All models loaded successfully!")


def speech_to_text(audio_binary):
    """
    Convert speech audio to text using Whisper.

    Args:
        audio_binary: Binary audio data

    Returns:
        str: Transcribed text
    """
    try:
        # Save audio to temporary file for processing
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_binary)

        # Transcribe using Whisper
        result = speech_recognizer(audio_path)
        text = result["text"]

        # Clean up temp file
        if os.path.exists(audio_path):
            os.remove(audio_path)

        print(f'Recognized text: {text}')
        return text

    except Exception as e:
        print(f"Error in speech_to_text: {e}")
        return "Error processing audio"


def translate_text(text, source_lang="en", target_lang="es"):
    """
    Translate text from source language to target language.

    Args:
        text: Text to translate
        source_lang: Source language code (default: "en")
        target_lang: Target language code (default: "es")

    Returns:
        str: Translated text
    """
    try:
        # Tokenize the input text
        inputs = translation_tokenizer(text, return_tensors="pt", padding=True)

        # Generate translation
        with torch.no_grad():
            outputs = translation_model.generate(**inputs, max_length=512)

        # Decode the output
        translated_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Translation: {text} -> {translated_text}")
        return translated_text.strip()

    except Exception as e:
        print(f"Error in translate_text: {e}")
        return f"Error translating text: {str(e)}"


def text_to_speech(text, voice=""):
    """
    Convert text to speech using TTS model.

    Args:
        text: Text to convert to speech
        voice: Voice preference (currently unused with MMS-TTS)

    Returns:
        bytes: Audio data in WAV format
    """
    if not tts_available:
        # Return empty audio if TTS is not available
        print("TTS not available, returning empty audio")
        # Generate 1 second of silence as placeholder
        sample_rate = 16000
        silence = np.zeros(sample_rate, dtype=np.float32)

        # Convert to WAV format
        buffer = io.BytesIO()
        sf.write(buffer, silence, sample_rate, format='WAV')
        return buffer.getvalue()

    try:
        # Generate speech
        speech = tts_pipeline(text)

        # Extract audio data
        audio_data = speech["audio"]
        sampling_rate = speech["sampling_rate"]

        # Convert to WAV format
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sampling_rate, format='WAV')
        audio_bytes = buffer.getvalue()

        print(f"Text-to-speech generated for: {text[:50]}...")
        return audio_bytes

    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        # Return silence on error
        sample_rate = 16000
        silence = np.zeros(sample_rate, dtype=np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, silence, sample_rate, format='WAV')
        return buffer.getvalue()


def process_translation(user_message):
    """
    Process user message and return translation.

    Args:
        user_message: Text to translate

    Returns:
        str: Translated text
    """
    # Translate from English to Spanish
    translated_text = translate_text(user_message, source_lang="en", target_lang="es")

    # Clean up any empty lines
    translated_text = os.linesep.join([s for s in translated_text.splitlines() if s])

    return translated_text


# Flask Routes

@app.route('/', methods=['GET'])
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    """Handle speech-to-text conversion requests"""
    print("Processing Speech-to-Text")

    # Get the audio data from request
    audio_binary = request.data

    # Transcribe the audio
    text = speech_to_text(audio_binary)

    # Return JSON response
    response = app.response_class(
        response=json.dumps({'text': text}),
        status=200,
        mimetype='application/json'
    )

    print(f"Response: {response.data}")
    return response


@app.route('/process-message', methods=['POST'])
def process_message_route():
    """Handle translation and TTS requests"""
    # Get user message from request
    user_message = request.json.get('userMessage', '')
    print(f'User message: {user_message}')

    # Get voice preference
    voice = request.json.get('voice', 'default')
    print(f'Voice: {voice}')

    # Translate the message
    translated_text = process_translation(user_message)

    # Convert translation to speech
    translated_speech = text_to_speech(translated_text, voice)

    # Encode speech to base64 for JSON transmission
    translated_speech_base64 = base64.b64encode(translated_speech).decode('utf-8')

    # Return JSON response with both text and speech
    response = app.response_class(
        response=json.dumps({
            "watsonxResponseText": translated_text,
            "watsonxResponseSpeech": translated_speech_base64
        }),
        status=200,
        mimetype='application/json'
    )

    print(f"Translation completed: {user_message} -> {translated_text}")
    return response


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Translator AI Server Starting...")
    print("="*50)
    print("\nEndpoints:")
    print("  - GET  /                : Main page")
    print("  - POST /speech-to-text  : Convert speech to text")
    print("  - POST /process-message : Translate text and convert to speech")
    print("\nServer running on http://0.0.0.0:8000")
    print("="*50 + "\n")

    app.run(port=8000, host='0.0.0.0', debug=False)
