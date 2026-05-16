"""
Tests for Image Generator (Stable Diffusion)
"""

import pytest
from PIL import Image
from src.models.image_gen import ImageGenerator


@pytest.fixture
def image_generator(device):
    """Initialize Image Generator for testing."""
    # Use smaller/faster model for testing if available
    return ImageGenerator(device=device, cache_dir="./models/stable-diffusion", safety_checker=False)


def test_generator_initialization(image_generator):
    """Test Image Generator initialization."""
    assert image_generator is not None
    assert image_generator.pipe is not None


@pytest.mark.slow
def test_generate_single_image(image_generator):
    """Test generating a single image."""
    prompt = "a red car on a highway"
    
    images = image_generator.generate(
        prompt=prompt,
        num_images=1,
        num_inference_steps=20,  # Fewer steps for testing
        width=256,  # Smaller size for testing
        height=256
    )
    
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (256, 256)


@pytest.mark.slow
def test_generate_with_negative_prompt(image_generator):
    """Test generation with negative prompt."""
    prompt = "a beautiful landscape"
    negative_prompt = "blurry, low quality"
    
    images = image_generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images=1,
        num_inference_steps=20,
        width=256,
        height=256
    )
    
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


@pytest.mark.slow
def test_generate_with_seed(image_generator):
    """Test generation with fixed seed for reproducibility."""
    prompt = "a cat"
    seed = 42
    
    # Generate twice with same seed
    images1 = image_generator.generate(
        prompt=prompt,
        num_images=1,
        num_inference_steps=20,
        width=256,
        height=256,
        seed=seed
    )
    
    images2 = image_generator.generate(
        prompt=prompt,
        num_images=1,
        num_inference_steps=20,
        width=256,
        height=256,
        seed=seed
    )
    
    # Images should be identical (or very similar)
    assert images1[0].size == images2[0].size


def test_generate_variations(image_generator):
    """Test generating variations of same prompt."""
    prompt = "a tree"
    
    # Mock the generate method to avoid slow generation
    def mock_generate(*args, **kwargs):
        return [Image.new("RGB", (256, 256), color="green")]
    
    original_generate = image_generator.generate
    image_generator.generate = mock_generate
    
    try:
        variations = image_generator.generate_variations(
            prompt=prompt,
            num_variations=2
        )
        
        assert len(variations) == 2
        assert all(isinstance(img, Image.Image) for img in variations)
    finally:
        image_generator.generate = original_generate


def test_generate_batch(image_generator):
    """Test batch generation for multiple prompts."""
    prompts = ["a cat", "a dog"]
    
    # Mock the generate method
    def mock_generate(prompt, *args, **kwargs):
        return [Image.new("RGB", (256, 256), color="blue")]
    
    original_generate = image_generator.generate
    image_generator.generate = mock_generate
    
    try:
        results = image_generator.generate_batch(prompts)
        
        assert len(results) == len(prompts)
        assert all(isinstance(imgs, list) for imgs in results)
    finally:
        image_generator.generate = original_generate
