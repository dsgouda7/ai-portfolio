from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import io
import base64
import threading
import soundfile as sf
from model_manager import get_model_manager

app = Flask(__name__)
CORS(app)

model_manager = get_model_manager()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200


@app.route('/api/status')
def get_status():
    return jsonify({'models': model_manager.get_model_status()})


@app.route('/api/text', methods=['POST'])
def process_text():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    input_text = data['text']
    print(f"{input_text}")

    response_text = model_manager.generate_text_response(input_text)
    print(f"{response_text}")

    return jsonify({'input': input_text, 'response': response_text})


@app.route('/api/speech', methods=['POST'])
def process_speech():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    try:
        audio_bytes = audio_file.read()
        print(f"Received {len(audio_bytes)} bytes")

        transcription = model_manager.speech_to_text(audio_bytes)
        print(f"Transcription: {transcription}")

        if not transcription:
            return jsonify({'error': 'Could not process audio'}), 400

        response_text = model_manager.generate_text_response(transcription)
        print(f"Response: {response_text}")

        audio_array, sample_rate = model_manager.text_to_speech(response_text)

        audio_base64 = None
        if audio_array is not None:
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='WAV')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return jsonify({
            'transcription': transcription,
            'response': response_text,
            'audio_base64': audio_base64
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/init-models', methods=['POST'])
def init_models():
    try:
        model_manager.download_all_models()
        return jsonify({'status': 'All models initialized'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting server...")
    print("Loading models...")
    print("Server started: http://localhost:5000")

    threading.Thread(target=model_manager.download_all_models, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
