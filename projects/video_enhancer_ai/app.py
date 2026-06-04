from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import threading
from pathlib import Path
from video_processor import get_video_processor
from audio_processor import get_audio_processor
import subprocess
import tempfile

app = Flask(__name__)
CORS(app)

video_processor = get_video_processor()
audio_processor = get_audio_processor()

UPLOAD_FOLDER = Path("/app/uploads")
OUTPUT_FOLDER = Path("/app/outputs")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)


@app.route('/')
def index():
    return jsonify({
        'service': 'Video Enhancer AI',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'Health check',
            '/api/status': 'Get processor status',
            '/api/enhance': 'POST video file for enhancement'
        }
    })


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200


@app.route('/api/status')
def get_status():
    return jsonify({
        'video': {
            'device': video_processor.device,
            'model_loaded': video_processor.model_config['gpu']['loaded'] or
                          video_processor.model_config['cpu']['loaded']
        },
        'audio': {
            'device': audio_processor.device,
            'model_loaded': audio_processor.model_config['gpu']['loaded'] or
                          audio_processor.model_config['cpu']['loaded']
        }
    })


@app.route('/api/enhance', methods=['POST'])
def enhance_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        input_path = UPLOAD_FOLDER / video_file.filename
        video_file.save(str(input_path))
        print(f"Saved input: {input_path}")

        output_filename = f"enhanced_{input_path.stem}.mp4"
        video_output = OUTPUT_FOLDER / f"video_{output_filename}"
        audio_output = OUTPUT_FOLDER / f"audio_{input_path.stem}.wav"
        final_output = OUTPUT_FOLDER / output_filename

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        print("Extracting audio...")
        subprocess.run([
            'ffmpeg', '-i', str(input_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            temp_audio_path, '-y'
        ], check=True, capture_output=True)

        print("Processing video and audio in parallel...")
        video_thread = threading.Thread(
            target=video_processor.process_video,
            args=(str(input_path), str(video_output))
        )
        audio_thread = threading.Thread(
            target=audio_processor.enhance_audio,
            args=(temp_audio_path, str(audio_output))
        )

        video_thread.start()
        audio_thread.start()
        video_thread.join()
        audio_thread.join()

        print("Merging video and audio...")
        subprocess.run([
            'ffmpeg', '-i', str(video_output),
            '-i', str(audio_output),
            '-c:v', 'libx264', '-c:a', 'aac',
            '-strict', 'experimental',
            str(final_output), '-y'
        ], check=True, capture_output=True)

        os.remove(temp_audio_path)
        if video_output.exists():
            os.remove(video_output)
        if audio_output.exists():
            os.remove(audio_output)

        return jsonify({
            'status': 'success',
            'output_file': output_filename,
            'message': 'Video enhanced to 4K with improved audio'
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    file_path = OUTPUT_FOLDER / filename
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    return send_file(str(file_path), as_attachment=True)


if __name__ == '__main__':
    print("Starting Video Enhancer AI server...")
    print("Preloading models...")

    def preload():
        video_processor.load_model()
        audio_processor.load_model()

    threading.Thread(target=preload, daemon=True).start()

    print("Server started: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
