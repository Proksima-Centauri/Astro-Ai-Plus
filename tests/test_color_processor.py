import numpy as np

from processing.color_processor import ColorProcessor


def test_to_float01_accepts_16bit_mono_and_expands_to_three_channels():
    mono = np.arange(25, dtype=np.uint16).reshape(5, 5) * 1024

    out = ColorProcessor._to_float01(mono)

    assert out.shape == (5, 5, 3)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert np.allclose(out[:, :, 0], out[:, :, 1])
    assert np.allclose(out[:, :, 1], out[:, :, 2])


def test_apply_camera_raw_and_hsl_identity_when_params_are_empty():
    image = np.array(
        [
            [[0, 64, 255], [32, 32, 32]],
            [[255, 128, 0], [16, 200, 16]],
        ],
        dtype=np.uint8,
    )

    out = ColorProcessor.apply_camera_raw_and_hsl(
        image=image,
        basic_params=None,
        hsl_params=None,
        ghs_params=None,
    )

    expected = image.astype(np.float32) / 255.0
    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert np.allclose(out, expected, atol=5 / 255)


def test_apply_camera_raw_and_hsl_changes_image_with_exposure_and_hsl():
    image = np.full((5, 5, 3), 64, dtype=np.uint8)
    basic_params = {
        "exposure": 1.0,
        "contrast": 0.2,
        "saturation": 0.3,
        "vibrance": 0.2,
        "texture": 0.0,
        "clarity": 0.0,
        "dehaze": 0.0,
        "noise_reduction": 0.0,
    }
    hsl_params = {
        "Hue": {"Blues": 15.0},
        "Saturation": {"Blues": 20.0},
        "Luminance": {"Blues": 5.0},
    }

    out = ColorProcessor.apply_camera_raw_and_hsl(
        image=image,
        basic_params=basic_params,
        hsl_params=hsl_params,
        ghs_params={"stretch_factor": 0.0},
    )

    src = image.astype(np.float32) / 255.0
    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert float(out.mean()) > float(src.mean())
    assert not np.allclose(out, src)
