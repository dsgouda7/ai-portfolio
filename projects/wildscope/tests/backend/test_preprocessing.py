from PIL import Image

from wildscope.preprocessing import prepare_image


def test_high_resolution_image_keeps_full_frame_without_upscaling(tmp_path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (1600, 1000), "#315c42").save(source, quality=95)

    result = prepare_image(source, tmp_path / "stages")

    assert result.source_size == (1600, 1000)
    assert result.model_input_size == (1600, 1000)
    assert result.enhancement_applied is False
    assert result.enhancement_method == "original-resolution-passthrough"
    assert Image.open(result.model_input_path).size == (1600, 1000)


def test_low_resolution_image_is_enhanced_for_model_input(tmp_path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (480, 320), "#756847").save(source, quality=80)

    result = prepare_image(source, tmp_path / "stages")

    assert result.source_size == (480, 320)
    assert result.model_input_size == (960, 640)
    assert result.enhancement_applied is True
    assert result.enhancement_method == "opencv-clahe-bicubic-x2"
    assert result.normalized_path.is_file()
    assert result.enhanced_path.is_file()
