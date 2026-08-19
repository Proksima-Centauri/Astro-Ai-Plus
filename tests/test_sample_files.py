from pathlib import Path

import cv2
import numpy as np
import pytest
from astropy.io import fits


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_mono_16bit_tiff_fixture_is_small_and_readable():
    path = FIXTURES_DIR / "mono16_5x5.tiff"
    assert path.exists()

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None
    assert image.shape == (5, 5)
    assert image.dtype == np.uint16


def test_color_32bit_fit_fixture_is_small_and_readable():
    path = FIXTURES_DIR / "color32_5x5.fit"
    assert path.exists()

    data = fits.getdata(path)
    assert data is not None
    assert data.shape == (5, 5, 3)
    assert data.dtype.kind == "f"
    assert data.dtype.itemsize == np.dtype(np.float32).itemsize


@pytest.mark.parametrize(
    ("filename", "shape", "dtype"),
    [
        ("mono8_5x5.png", (5, 5), np.uint8),
        ("color8_5x5.jpg", (5, 5, 3), np.uint8),
        ("color16_5x5.tiff", (5, 5, 3), np.uint16),
    ],
)
def test_opencv_fixtures_are_small_and_readable(filename, shape, dtype):
    path = FIXTURES_DIR / filename
    assert path.exists()

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None
    assert image.shape == shape
    assert image.dtype == dtype


@pytest.mark.parametrize(
    ("filename", "shape", "dtype_kind", "itemsize"),
    [
        ("mono32_5x5.fit", (5, 5), "f", np.dtype(np.float32).itemsize),
        ("mono16_5x5.fit", (5, 5), "u", np.dtype(np.uint16).itemsize),
        ("color32_5x5.fit", (5, 5, 3), "f", np.dtype(np.float32).itemsize),
    ],
)
def test_fits_fixtures_are_small_and_readable(filename, shape, dtype_kind, itemsize):
    path = FIXTURES_DIR / filename
    assert path.exists()

    data = fits.getdata(path)
    assert data is not None
    assert data.shape == shape
    assert data.dtype.kind == dtype_kind
    assert data.dtype.itemsize == itemsize
