"""Proste testy klasy ColorProcessor.

Uruchamianie (z glownego katalogu repo):

    pytest

Kazda funkcja zaczynajaca sie od "test_" to jeden test. Test przygotowuje
male sztuczne zdjecie, wywoluje funkcje i sprawdza, czy wynik ma sens.
"""

import numpy as np

from processing.color_processor import ColorProcessor


def make_test_image() -> np.ndarray:
    """Male kolorowe "zdjecie" 32x32 (BGR, uint8) do testow."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


NEUTRAL_PARAMS = {
    "temperature": 0.0,
    "tint": 0.0,
    "exposure": 0.0,
    "contrast": 0.0,
    "saturation": 0.0,
    "vibrance": 0.0,
}


def test_none_returns_none():
    # Gdy nie ma obrazu, funkcja nie powinna wybuchnac - tylko zwrocic None.
    assert ColorProcessor.apply_camera_raw_and_hsl(None, {}, {}) is None


def test_neutral_params_keep_image_unchanged():
    # Wszystkie suwaki na zero -> obraz powinien wyjsc (prawie) taki sam.
    # "Prawie", bo obraz przechodzi konwersje BGR -> HSV -> BGR na uint8,
    # a hue w OpenCV ma tylko 180 krokow - zmierzona maksymalna odchylka
    # dla tego obrazu to 6 na 255.
    img = make_test_image()
    out = ColorProcessor.apply_camera_raw_and_hsl(img, NEUTRAL_PARAMS, {})
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.max(np.abs(out.astype(int) - img.astype(int))) <= 6


def test_exposure_brightens_image():
    # exposure = +1 to podwojenie jasnosci. Testujemy na ciemnym obrazie,
    # bo na jasnym piksele obcinaja sie na bieli (255) i srednia rosnie
    # mniej niz dwukrotnie.
    img = (make_test_image() // 3).astype(np.uint8)
    params = dict(NEUTRAL_PARAMS, exposure=1.0)
    out = ColorProcessor.apply_camera_raw_and_hsl(img, params, {})
    assert out.astype(float).mean() > img.astype(float).mean() * 1.7


def test_negative_exposure_darkens_image():
    img = make_test_image()
    params = dict(NEUTRAL_PARAMS, exposure=-1.0)
    out = ColorProcessor.apply_camera_raw_and_hsl(img, params, {})
    assert out.astype(float).mean() < img.astype(float).mean() * 0.7


def test_temperature_warms_colors():
    # Dodatnia temperatura = wiecej czerwieni, mniej niebieskiego.
    # Uwaga: obraz jest w formacie BGR, wiec kanal 0 to Blue, kanal 2 to Red.
    img = make_test_image()
    params = dict(NEUTRAL_PARAMS, temperature=50.0)
    out = ColorProcessor.apply_camera_raw_and_hsl(img, params, {})
    assert out[:, :, 2].astype(float).mean() > img[:, :, 2].astype(float).mean()
    assert out[:, :, 0].astype(float).mean() < img[:, :, 0].astype(float).mean()


def test_noise_reduction_zero_is_noop():
    # amount = 0 -> funkcja ma oddac dokladnie ten sam obraz.
    img = make_test_image().astype(np.float32) / 255.0
    out = ColorProcessor._fast_noise_reduction(img, 0.0)
    assert out is img
