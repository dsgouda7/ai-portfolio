"""
Evaluation metrics for multimodal models
"""

import torch
import numpy as np
from typing import List, Dict, Union
from PIL import Image
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from .utils import setup_logger


logger = setup_logger(__name__)


def compute_clip_score(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    normalize: bool = True
) -> float:
    """
    Compute CLIP score (cosine similarity between text and image embeddings).
    
    Args:
        text_embeddings: Text embedding vectors (N, D)
        image_embeddings: Image embedding vectors (N, D)
        normalize: Whether to normalize embeddings
    
    Returns:
        Average CLIP score
    """
    if normalize:
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    
    # Compute cosine similarity
    similarities = np.sum(text_embeddings * image_embeddings, axis=1)
    
    return float(np.mean(similarities))


def compute_word_error_rate(
    references: List[str],
    hypotheses: List[str]
) -> float:
    """
    Compute Word Error Rate (WER) for speech recognition.
    
    Args:
        references: Ground truth transcriptions
        hypotheses: Predicted transcriptions
    
    Returns:
        Word Error Rate (lower is better)
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    # Compute WER for each pair
    wer_scores = []
    for ref, hyp in zip(references, hypotheses):
        score = wer(ref, hyp)
        wer_scores.append(score)
    
    avg_wer = float(np.mean(wer_scores))
    
    logger.info(f"Average WER: {avg_wer:.4f}")
    
    return avg_wer


def compute_bleu_score(
    references: List[str],
    hypotheses: List[str],
    n: int = 4
) -> float:
    """
    Compute BLEU score for caption generation.
    
    Args:
        references: Ground truth captions
        hypotheses: Generated captions
        n: Maximum n-gram order (default: 4 for BLEU-4)
    
    Returns:
        Average BLEU score
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        # Tokenize
        ref_tokens = [ref.lower().split()]
        hyp_tokens = hyp.lower().split()
        
        # Compute BLEU
        weights = tuple([1.0 / n] * n)
        score = sentence_bleu(ref_tokens, hyp_tokens, weights=weights, smoothing_function=smoothing)
        bleu_scores.append(score)
    
    avg_bleu = float(np.mean(bleu_scores))
    
    logger.info(f"Average BLEU-{n}: {avg_bleu:.4f}")
    
    return avg_bleu


def compute_embedding_similarity(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    metric: str = "cosine"
) -> float:
    """
    Compute similarity between two sets of embeddings.
    
    Args:
        embeddings1: First embedding matrix (N, D)
        embeddings2: Second embedding matrix (N, D)
        metric: Similarity metric ("cosine", "euclidean")
    
    Returns:
        Average similarity score
    """
    if metric == "cosine":
        # Cosine similarity
        similarities = cosine_similarity(embeddings1, embeddings2)
        # Get diagonal (pairwise similarities)
        scores = np.diag(similarities)
    elif metric == "euclidean":
        # Euclidean distance (convert to similarity)
        distances = np.sqrt(np.sum((embeddings1 - embeddings2) ** 2, axis=1))
        # Convert distance to similarity (0 = identical)
        scores = 1.0 / (1.0 + distances)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return float(np.mean(scores))


def evaluate_text_image_alignment(
    clip_model,
    texts: List[str],
    images: List[Union[str, Image.Image]],
    threshold: float = 0.7
) -> Dict[str, float]:
    """
    Evaluate text-image alignment using CLIP.
    
    Args:
        clip_model: CLIPModel instance
        texts: List of text descriptions
        images: List of images
        threshold: Threshold for considering alignment good
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Get embeddings
    text_embeds = []
    image_embeds = []
    
    for text, image in zip(texts, images):
        text_embed = clip_model.get_text_embedding(text)
        image_embed = clip_model.get_image_embedding(image)
        
        text_embeds.append(text_embed)
        image_embeds.append(image_embed)
    
    text_embeds = np.array(text_embeds)
    image_embeds = np.array(image_embeds)
    
    # Compute CLIP score
    clip_score = compute_clip_score(text_embeds, image_embeds)
    
    # Compute accuracy (above threshold)
    text_embeds_norm = text_embeds / np.linalg.norm(text_embeds, axis=1, keepdims=True)
    image_embeds_norm = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
    similarities = np.sum(text_embeds_norm * image_embeds_norm, axis=1)
    
    accuracy = float(np.mean(similarities >= threshold))
    
    return {
        "clip_score": clip_score,
        "accuracy": accuracy,
        "threshold": threshold,
        "num_samples": len(texts)
    }


def evaluate_speech_recognition(
    whisper_model,
    audio_files: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Evaluate speech recognition performance.
    
    Args:
        whisper_model: WhisperModel instance
        audio_files: List of audio file paths
        references: Ground truth transcriptions
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Generate transcriptions
    hypotheses = []
    
    for audio_file in audio_files:
        result = whisper_model.transcribe(audio_file)
        hypotheses.append(result["text"])
    
    # Compute WER
    wer_score = compute_word_error_rate(references, hypotheses)
    
    # Compute accuracy (WER < 0.15)
    accuracy = float(wer_score < 0.15)
    
    return {
        "wer": wer_score,
        "accuracy": accuracy,
        "num_samples": len(audio_files)
    }


def evaluate_image_captioning(
    captioner_model,
    images: List[Union[str, Image.Image]],
    references: List[str]
) -> Dict[str, float]:
    """
    Evaluate image captioning performance.
    
    Args:
        captioner_model: ImageCaptioner instance
        images: List of images
        references: Ground truth captions
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Generate captions
    hypotheses = []
    
    for image in images:
        caption = captioner_model.caption(image)
        hypotheses.append(caption)
    
    # Compute BLEU
    bleu_score = compute_bleu_score(references, hypotheses)
    
    return {
        "bleu_score": bleu_score,
        "num_samples": len(images)
    }


def compute_generation_quality_metrics(
    generated_images: List[Image.Image],
    prompts: List[str],
    clip_model
) -> Dict[str, float]:
    """
    Compute quality metrics for generated images.
    
    Args:
        generated_images: List of generated images
        prompts: Text prompts used for generation
        clip_model: CLIPModel instance
    
    Returns:
        Dictionary with quality metrics
    """
    # Compute CLIP score
    text_embeds = []
    image_embeds = []
    
    for prompt, image in zip(prompts, generated_images):
        text_embed = clip_model.get_text_embedding(prompt)
        image_embed = clip_model.get_image_embedding(image)
        
        text_embeds.append(text_embed)
        image_embeds.append(image_embed)
    
    text_embeds = np.array(text_embeds)
    image_embeds = np.array(image_embeds)
    
    clip_score = compute_clip_score(text_embeds, image_embeds)
    
    # Additional quality metrics could include:
    # - Image diversity (variance in embeddings)
    # - Perceptual quality scores
    # - FID (Fréchet Inception Distance) if reference dataset available
    
    return {
        "clip_score": clip_score,
        "num_images": len(generated_images)
    }
