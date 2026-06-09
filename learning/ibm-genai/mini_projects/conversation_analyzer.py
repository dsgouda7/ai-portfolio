import gradio as gr
from transformers import pipeline
import torch

# Local models - no credentials required
# Uses CPU-optimized models from HuggingFace

# Initialize summarization pipeline with FLAN-T5 (good for CPU, ~250MB)
summarizer = pipeline(
    "summarization",
    model="google/flan-t5-small",
    device=-1  # CPU
)

# Alternative: For better quality, use FLAN-T5-base (~900MB)
# summarizer = pipeline(
#     "summarization",
#     model="google/flan-t5-base",
#     device=-1
# )

def extract_key_points(text):
    """Extract key points from transcribed text using local model"""
    # FLAN-T5 works well with instruction-based prompts
    prompt = f"List the key points from this conversation: {text}"

    # Summarize in chunks if text is too long
    max_length = 512
    if len(text.split()) > max_length:
        # Split into chunks
        words = text.split()
        chunks = [' '.join(words[i:i+max_length]) for i in range(0, len(words), max_length)]
        summaries = []
        for chunk in chunks:
            result = summarizer(f"summarize: {chunk}", max_length=150, min_length=30, do_sample=False)
            summaries.append(result[0]['summary_text'])
        return ' '.join(summaries)
    else:
        result = summarizer(prompt, max_length=200, min_length=50, do_sample=False)
        return result[0]['summary_text']

# Speech-to-text pipeline (runs locally on CPU)
speech_to_text = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
    device=-1  # CPU
)

def transcribe_audio(audio_file):
    """Transcribe audio and extract key points using local models"""
    # Transcribe the audio
    transcribed_txt = speech_to_text(audio_file, batch_size=8)["text"]

    # Extract key points from transcription
    result = extract_key_points(transcribed_txt)

    # Return both transcription and key points
    return f"Transcription:\n{transcribed_txt}\n\nKey Points:\n{result}"

# init the gradio interface
audio_input=gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface =gr.Interface(fn=transcribe_audio, inputs=audio_input, outputs=output_text, title="Audio Transcription App", description="Upload the audio file")

iface.launch(server_name='0.0.0.0', server_port=7860)
