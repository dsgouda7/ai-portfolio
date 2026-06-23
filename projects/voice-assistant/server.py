import base64
import json
import os

from dotenv import load_dotenv

load_dotenv()  # loads .env before config.py reads env vars

from flask import Flask, render_template, request
from flask_cors import CORS
from worker import process_message, speech_to_text, text_to_speech

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/speech-to-text", methods=["POST"])
def speech_to_text_route():
    print("Processing speech-to-text")
    audio_binary = request.data
    text = speech_to_text(audio_binary)
    return app.response_class(
        response=json.dumps({"text": text}),
        status=200,
        mimetype="application/json",
    )


@app.route("/process-message", methods=["POST"])
def process_message_route():
    user_message: str = request.json["userMessage"]
    voice: str = request.json.get("voice", "")
    print("user_message:", user_message)
    print("voice:", voice)

    response_text = process_message(user_message)
    # Strip blank lines
    response_text = os.linesep.join([s for s in response_text.splitlines() if s])

    response_speech = text_to_speech(response_text, voice)
    response_speech_b64 = base64.b64encode(response_speech).decode("utf-8")

    return app.response_class(
        response=json.dumps(
            {
                "responseText": response_text,
                "responseSpeech": response_speech_b64,
            }
        ),
        status=200,
        mimetype="application/json",
    )


if __name__ == "__main__":
    app.run(port=8000, host="0.0.0.0")
