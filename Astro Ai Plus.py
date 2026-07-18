import sys
import os
import json
import html
import shutil
import subprocess
import tempfile
import time
import re
import shlex
import math
import importlib
from dataclasses import dataclass
from pathlib import Path
import socket
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta, timezone
import numpy as np
import cv2

try:
    sep = importlib.import_module("sep")
    SEP_AVAILABLE = True
except ImportError:
    sep = None
    SEP_AVAILABLE = False

try:
    from astropy.stats import sigma_clipped_stats
    ASTROPY_STATS_AVAILABLE = True
except ImportError:
    sigma_clipped_stats = None
    ASTROPY_STATS_AVAILABLE = False

try:
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import detect_sources, SourceCatalog
    PHOTUTILS_AVAILABLE = True
except ImportError:
    Background2D = None
    MedianBackground = None
    detect_sources = None
    SourceCatalog = None
    PHOTUTILS_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    sr = None
    SPEECH_RECOGNITION_AVAILABLE = False

FIXED_GEMINI_MODEL = "gemini-2.0-flash-lite"
AI_ASSISTANT_NAME = "Altair"
ALTAIR_3D_FLY_GUIDE_FILE = "3d_fly_help.md"


def neutralize_background(image: np.ndarray, roi: tuple) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if image.dtype != np.float32:
        raise ValueError("image must be float32")
    if not isinstance(roi, tuple) or len(roi) != 4:
        raise TypeError("roi must be a tuple: (y1, y2, x1, x2)")

    try:
        y1, y2, x1, x2 = (int(v) for v in roi)
    except Exception as exc:
        raise ValueError("roi values must be integers") from exc

    h, w, _ = image.shape
    if not (0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w):
        raise ValueError(f"roi {roi} is out of image bounds (H={h}, W={w})")

    roi_data = image[y1:y2, x1:x2, :]
    if roi_data.size == 0 or roi_data.shape[0] == 0 or roi_data.shape[1] == 0:
        raise ValueError("roi is empty")

    if not np.isfinite(roi_data).all():
        raise ValueError("roi contains non-finite values (NaN/Inf)")
    if np.any(roi_data < 0.0):
        raise ValueError("roi contains negative values")

    channel_means = roi_data.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    if channel_means.size != 3:
        raise ValueError("failed to compute channel means for ROI")

    eps = np.float32(1e-6)
    target = np.float32(channel_means.mean(dtype=np.float64))
    gains = target / (channel_means + eps)
    balanced = image * gains.reshape(1, 1, 3)
    return np.clip(balanced, 0.0, 1.0).astype(np.float32, copy=False)


def _safe_median_absolute_deviation(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return mad


def _estimate_background_and_noise(gray_f32: np.ndarray) -> tuple[float, float, str]:
    if ASTROPY_STATS_AVAILABLE and sigma_clipped_stats is not None:
        try:
            _mean, median, std = sigma_clipped_stats(gray_f32, sigma=3.0, maxiters=5)
            noise_sigma = max(1e-6, float(std))
            return float(median), noise_sigma, "astropy_sigma_clip"
        except Exception:
            pass

    median = float(np.median(gray_f32))
    noise_sigma = max(1e-6, 1.4826 * _safe_median_absolute_deviation(gray_f32))
    return median, noise_sigma, "mad"


def _build_empty_stars_payload(method: str) -> dict:
    return {
        "count": 0,
        "fwhm_px_median": None,
        "fwhm_px_mean": None,
        "fwhm_px_min": None,
        "fwhm_px_max": None,
        "snr_median": None,
        "snr_mean": None,
        "sample_size": 0,
        "top_stars": [],
        "method": method,
    }


def _build_stars_payload_from_samples(stars: list, method: str) -> dict:
    if not stars:
        return _build_empty_stars_payload(method)

    fwhm_vals = np.array([item["fwhm_px"] for item in stars], dtype=np.float32)
    snr_vals = np.array([item["snr"] for item in stars], dtype=np.float32)
    return {
        "count": int(len(stars)),
        "fwhm_px_median": float(np.median(fwhm_vals)),
        "fwhm_px_mean": float(np.mean(fwhm_vals)),
        "fwhm_px_min": float(np.min(fwhm_vals)),
        "fwhm_px_max": float(np.max(fwhm_vals)),
        "snr_median": float(np.median(snr_vals)),
        "snr_mean": float(np.mean(snr_vals)),
        "sample_size": int(len(stars)),
        "top_stars": stars[:10],
        "method": method,
    }


def _compute_star_metrics_sep(gray_u8: np.ndarray, max_stars: int = 120) -> dict:
    if not SEP_AVAILABLE or sep is None:
        return _build_empty_stars_payload("sep_unavailable")

    data = gray_u8.astype(np.float32)
    bkg = sep.Background(data)
    data_sub = data - bkg.back()
    threshold = max(4.0 * float(bkg.globalrms), 6.0)

    objects = sep.extract(
        data_sub,
        threshold,
        err=bkg.globalrms,
        minarea=5,
        deblend_cont=0.005,
        clean=True,
    )

    if objects is None or len(objects) == 0:
        return _build_empty_stars_payload("sep")

    fwhm_factor = 2.354820045
    stars = []
    for obj in objects:
        a = float(obj["a"])
        b = float(obj["b"])
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0 or b <= 0.0:
            continue

        sigma_eq = math.sqrt(max(1e-6, 0.5 * (a * a + b * b)))
        fwhm = float(fwhm_factor * sigma_eq)
        if not (0.7 <= fwhm <= 20.0):
            continue

        flux = float(obj["flux"])
        peak = float(obj["peak"])
        npix = float(obj["npix"]) if "npix" in obj.dtype.names else max(1.0, math.pi * a * b)
        noise_term = max(1e-6, math.sqrt(max(0.0, flux) + npix * (float(bkg.globalrms) ** 2)))
        snr = float(flux / noise_term)
        if not np.isfinite(snr) or snr <= 0.0:
            continue

        stars.append({
            "fwhm_px": fwhm,
            "snr": snr,
            "peak": peak,
            "x": int(round(float(obj["x"]))),
            "y": int(round(float(obj["y"]))),
            "area": int(round(npix)),
            "ellipticity": float(max(a, b) / max(1e-6, min(a, b))),
        })

    if not stars:
        return _build_empty_stars_payload("sep")

    stars.sort(key=lambda item: item.get("snr", 0.0), reverse=True)
    stars = stars[:max(1, int(max_stars))]
    return _build_stars_payload_from_samples(stars, "sep")


def _compute_star_metrics_photutils(gray_f32: np.ndarray, noise_sigma: float, max_stars: int = 120) -> dict:
    if not PHOTUTILS_AVAILABLE or detect_sources is None or SourceCatalog is None:
        return _build_empty_stars_payload("photutils_unavailable")

    try:
        box_size = (
            max(32, min(96, int(gray_f32.shape[0] // 8))),
            max(32, min(96, int(gray_f32.shape[1] // 8))),
        )
        bkg = Background2D(
            gray_f32,
            box_size=box_size,
            filter_size=(3, 3),
            bkg_estimator=MedianBackground(),
        )
        background = bkg.background
        rms = np.maximum(bkg.background_rms, 1e-6)
        rms_median = float(np.median(rms))
    except Exception:
        background = np.full_like(gray_f32, float(np.median(gray_f32)), dtype=np.float32)
        rms_median = max(1e-6, float(noise_sigma))
        rms = np.full_like(gray_f32, rms_median, dtype=np.float32)

    data_sub = gray_f32 - background
    threshold = max(4.0 * rms_median, 6.0)
    segmentation = detect_sources(data_sub, threshold, n_pixels=5)
    if segmentation is None:
        return _build_empty_stars_payload("photutils")

    catalog = SourceCatalog(data_sub, segmentation, error=rms)
    table = catalog.to_table(
        columns=(
            "x_centroid",
            "y_centroid",
            "area",
            "max_value",
            "segment_flux",
            "fwhm",
            "semimajor_axis",
            "semiminor_axis",
        )
    )
    if table is None or len(table) == 0:
        return _build_empty_stars_payload("photutils")

    stars = []
    fwhm_factor = 2.354820045
    for row in table:
        area = float(row["area"])
        flux = float(row["segment_flux"])
        peak = float(row["max_value"])
        x_val = float(row["x_centroid"])
        y_val = float(row["y_centroid"])
        fwhm = row["fwhm"]
        try:
            fwhm_candidate = float(fwhm)
        except Exception:
            fwhm_candidate = float("nan")

        if not np.isfinite(fwhm_candidate):
            major = float(row["semimajor_axis"])
            minor = float(row["semiminor_axis"])
            sigma_eq = math.sqrt(max(1e-6, 0.5 * (major * major + minor * minor)))
            fwhm_val = float(fwhm_factor * sigma_eq)
        else:
            fwhm_val = fwhm_candidate

        if not np.isfinite(fwhm_val) or not (0.7 <= fwhm_val <= 24.0):
            continue

        noise_term = max(1e-6, math.sqrt(max(0.0, flux) + max(1.0, area) * (rms_median ** 2)))
        snr = float(flux / noise_term)
        if not np.isfinite(snr) or snr <= 0.0:
            continue

        stars.append(
            {
                "fwhm_px": fwhm_val,
                "snr": snr,
                "peak": peak,
                "x": int(round(x_val)),
                "y": int(round(y_val)),
                "area": int(round(area)),
                "ellipticity": None,
            }
        )

    if not stars:
        return _build_empty_stars_payload("photutils")

    stars.sort(key=lambda item: item.get("snr", 0.0), reverse=True)
    stars = stars[:max(1, int(max_stars))]
    return _build_stars_payload_from_samples(stars, "photutils")


def _compute_star_metrics_localmax(gray_f32: np.ndarray, background_median: float, noise_sigma: float, max_stars: int = 120) -> dict:
    blurred = cv2.GaussianBlur(gray_f32, (0, 0), 1.0)
    height, width = blurred.shape[:2]
    star_threshold = background_median + max(8.0, noise_sigma * 4.0)

    dilated = cv2.dilate(blurred, np.ones((3, 3), dtype=np.uint8))
    local_max_mask = np.logical_and(blurred >= star_threshold, blurred >= (dilated - 1e-6))
    binary_candidates = local_max_mask.astype(np.uint8)
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary_candidates, connectivity=8)

    stars = []
    patch_radius = 4
    fwhm_factor = 2.354820045

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        cx, cy = centroids[idx]
        ix = int(round(float(cx)))
        iy = int(round(float(cy)))

        if ix < patch_radius or iy < patch_radius or ix >= (width - patch_radius) or iy >= (height - patch_radius):
            continue

        patch = blurred[iy - patch_radius:iy + patch_radius + 1, ix - patch_radius:ix + patch_radius + 1]
        if patch.shape != (2 * patch_radius + 1, 2 * patch_radius + 1):
            continue

        local_background = float(np.percentile(patch, 20))
        signal = np.clip(patch - local_background, 0.0, None)
        peak = float(signal[patch_radius, patch_radius])
        total_signal = float(signal.sum())
        if peak <= noise_sigma * 2.0 or total_signal <= 0.0:
            continue

        yy, xx = np.indices(signal.shape, dtype=np.float32)
        dx = xx - patch_radius
        dy = yy - patch_radius
        var_x = float((signal * (dx ** 2)).sum() / total_signal)
        var_y = float((signal * (dy ** 2)).sum() / total_signal)
        sigma = max(0.1, math.sqrt(max(0.0, 0.5 * (var_x + var_y))))

        stars.append({
            "fwhm_px": float(fwhm_factor * sigma),
            "snr": float(peak / max(noise_sigma, 1e-6)),
            "peak": peak,
            "x": ix,
            "y": iy,
            "area": area,
            "ellipticity": None,
        })

    if not stars:
        return _build_empty_stars_payload("localmax")

    stars.sort(key=lambda item: item.get("snr", 0.0), reverse=True)
    stars = stars[:max(1, int(max_stars))]
    return _build_stars_payload_from_samples(stars, "localmax")


def compute_image_analysis_metrics(image: np.ndarray, max_stars: int = 120) -> dict:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {}

    if image.ndim == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        return {}

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    gray_f32 = gray.astype(np.float32)
    height, width = gray_f32.shape[:2]

    background_median, noise_sigma, noise_method = _estimate_background_and_noise(gray_f32)

    stars_payload = _build_empty_stars_payload("none")
    if SEP_AVAILABLE:
        try:
            stars_payload = _compute_star_metrics_sep(gray, max_stars=max_stars)
        except Exception:
            stars_payload = _build_empty_stars_payload("sep_error")

    if int(stars_payload.get("count") or 0) == 0 and PHOTUTILS_AVAILABLE:
        try:
            stars_payload = _compute_star_metrics_photutils(
                gray_f32,
                noise_sigma=noise_sigma,
                max_stars=max_stars,
            )
        except Exception:
            stars_payload = _build_empty_stars_payload("photutils_error")

    if int(stars_payload.get("count") or 0) == 0:
        stars_payload = _compute_star_metrics_localmax(
            gray_f32,
            background_median=background_median,
            noise_sigma=noise_sigma,
            max_stars=max_stars,
        )

    p01 = float(np.percentile(gray_f32, 1.0))
    p99 = float(np.percentile(gray_f32, 99.0))
    clipped_black = float(np.mean(gray_f32 <= 1.0) * 100.0)
    clipped_white = float(np.mean(gray_f32 >= 254.0) * 100.0)

    return {
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "image_shape": {
            "width": int(width),
            "height": int(height),
        },
        "analysis_backend": {
            "stars_method": stars_payload.get("method"),
            "noise_method": noise_method,
            "sep_available": bool(SEP_AVAILABLE),
            "photutils_available": bool(PHOTUTILS_AVAILABLE),
            "astropy_stats_available": bool(ASTROPY_STATS_AVAILABLE),
        },
        "luminance": {
            "mean": float(np.mean(gray_f32)),
            "median": background_median,
            "std": float(np.std(gray_f32)),
            "background_sigma": float(noise_sigma),
            "p01": p01,
            "p99": p99,
            "dynamic_range_p99_p01": float(max(0.0, p99 - p01)),
            "black_clipping_pct": clipped_black,
            "white_clipping_pct": clipped_white,
        },
        "stars": stars_payload,
    }

if sys.platform.startswith("linux"):
    session_type = os.environ.get("XDG_SESSION_TYPE", "").strip().lower()
    if session_type == "wayland":
        os.environ["QT_QPA_PLATFORM"] = "wayland"
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
    qt_logging_rules = os.environ.get("QT_LOGGING_RULES", "")
    muted_rule = "qt.qpa.wayland.warning=false"
    if muted_rule not in qt_logging_rules:
        os.environ["QT_LOGGING_RULES"] = f"{qt_logging_rules};{muted_rule}".strip(";")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

ONNX_SESSIONS = {}

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("com.astro.magic")
    except Exception:
        pass

from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow,
    QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QGraphicsView,
    QGraphicsScene, QSlider, QLabel, QFrame, QTabWidget, QGridLayout,
    QMenuBar, QMenu, QAction, QDialog, QStyle, QSpinBox,
    QDoubleSpinBox, QTextEdit, QLineEdit, QComboBox, QCompleter, QSplitter,
    QScrollArea, QStackedLayout, QTabBar, QListWidget, QListWidgetItem, QStackedWidget
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon, QColor, QFont, QPolygon, QPolygonF, QPen, QPalette, QRegion, QLinearGradient, QDrag
from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF, QThread, pyqtSignal, QTimer, QSize, QByteArray, QEvent, qInstallMessageHandler, QMimeData
from PyQt5.QtWidgets import QSizePolicy, QCheckBox, QInputDialog, QMessageBox
from PyQt5.QtGui import QPainterPath


def _qt_message_handler(_msg_type, _context, message):
    text = str(message or "")
    if "QSocketNotifier: Can only be used with threads started with QThread" in text:
        return
    if "Wayland does not support QWindow::requestActivate()" in text:
        return
    sys.stderr.write(text + "\n")


qInstallMessageHandler(_qt_message_handler)
try:
    from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
    QT_SERIAL_AVAILABLE = True
except ImportError:
    QSerialPort = None
    QSerialPortInfo = None
    QT_SERIAL_AVAILABLE = False
try:
    from PyQt5.QtSvg import QSvgRenderer
    SVG_AVAILABLE = True
except ImportError:
    QSvgRenderer = None
    SVG_AVAILABLE = False
try:
    import serial
    from serial.tools import list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    list_ports = None
    SERIAL_AVAILABLE = False

from deep_sky_catalog import DEEP_SKY_CATALOG
from processing.color_processor import ColorProcessor

I18N = {
    "app_title": {"pl": "Astro Ai Plus v1.0.0", "en": "Astro Ai Plus v1.0.0"},
    "action_open": {"pl": "Otwórz", "en": "Open"},
    "action_save": {"pl": "Zapisz", "en": "Save"},
    "action_save_as": {"pl": "Zapisz jako...", "en": "Save As..."},
    "action_exit": {"pl": "Wyjście", "en": "Exit"},
    "action_undo": {"pl": "Cofnij", "en": "Undo"},
    "action_redo": {"pl": "Ponów", "en": "Redo"},
    "action_workspace": {"pl": "Przestrzeń robocza", "en": "Workspace"},
    "action_home": {"pl": "Folder domowy", "en": "Home Folder"},
    "action_new_workspace": {"pl": "Nowa przestrzeń...", "en": "New Workspace..."},
    "action_preferences": {"pl": "Preferencje", "en": "Preferences"},
    "action_histogram": {"pl": "Histogram", "en": "Histogram"},
    "action_console": {"pl": "Konsola", "en": "Console"},
    "action_menu": {"pl": "Menu", "en": "Menu"},
    "action_ai": {"pl": "Asystent AI", "en": "AI Assistant"},
    "action_new_layer": {"pl": "Nowa warstwa", "en": "New Layer"},
    "action_delete_layer": {"pl": "Usuń zaznaczoną warstwę", "en": "Delete Selected Layer"},
    "action_levels": {"pl": "Poziomy", "en": "Levels"},
    "action_curves": {"pl": "Krzywe (LUT)", "en": "Curves (LUT)"},
    "action_magic": {"pl": "Magic Filter", "en": "Magic Filter"},
    "action_shrink": {"pl": "Zmniejsz gwiazdy", "en": "Star Shrink"},
    "action_plate": {"pl": "Plate Solving", "en": "Plate Solving"},
    "action_starnet": {"pl": "Uruchom StarNet++", "en": "Run StarNet++"},
    "action_deepsnr": {"pl": "Uruchom deepSNR", "en": "Run deepSNR"},
    "action_correction": {"pl": "Korekcja", "en": "Correction"},
    "action_color_calibration": {"pl": "Kalibracja kolorów", "en": "Color Calibration"},
    "action_3d_fly": {"pl": "Filtr 3D FLY", "en": "3D FLY Filter"},
    "action_blur": {"pl": "Rozmycie Gaussa", "en": "Gaussian Blur"},
    "action_rotate": {"pl": "Obrót", "en": "Rotate"},
    "action_crop": {"pl": "Kadruj", "en": "Crop"},
    "top_open": {"pl": "Otwórz", "en": "Open"},
    "top_save": {"pl": "Zapisz", "en": "Save"},
    "top_save_as": {"pl": "Zapisz jako", "en": "Save As"},
    "top_undo": {"pl": "Cofnij", "en": "Undo"},
    "top_redo": {"pl": "Ponów", "en": "Redo"},
    "top_workspace": {"pl": "Workspace", "en": "Workspace"},
    "top_home": {"pl": "Home", "en": "Home"},
    "top_prefs": {"pl": "Ust.", "en": "Prefs"},
    "top_console": {"pl": "Konsola", "en": "Console"},
    "top_menu": {"pl": "Menu", "en": "Menu"},
    "top_histogram": {"pl": "Histogram", "en": "Histogram"},
    "top_ai": {"pl": "AI", "en": "AI"},
    "top_magic": {"pl": "Magic", "en": "Magic"},
    "top_shrink": {"pl": "Shrink", "en": "Shrink"},
    "top_plate": {"pl": "Plate", "en": "Plate"},
    "top_starnet": {"pl": "StarNet", "en": "StarNet"},
    "top_deepsnr": {"pl": "deepSNR", "en": "deepSNR"},
    "top_3d_fly": {"pl": "3D FLY", "en": "3D FLY"},
    "top_blur": {"pl": "Blur", "en": "Blur"},
    "top_rotate": {"pl": "Obrót", "en": "Rotate"},
    "top_crop": {"pl": "Kadr", "en": "Crop"},
    "top_correction": {"pl": "Korekcja", "en": "Correction"},
    "top_color_calibration": {"pl": "Kalibracja", "en": "Calibration"},
    "top_levels": {"pl": "Poziomy", "en": "Levels"},
    "top_curves": {"pl": "Krzywe", "en": "Curves"},
    "top_new_layer": {"pl": "Nowa warstwa", "en": "New Layer"},
    "top_delete_layer": {"pl": "Usuń warstwę", "en": "Delete Layer"},
    "top_adj_levels": {"pl": "Warstwa Poziomy", "en": "Adj Levels"},
    "top_adj_curves": {"pl": "Warstwa Krzywe", "en": "Adj Curves"},
    "top_exit": {"pl": "Wyjście", "en": "Exit"},
    "connect_joystick": {"pl": "Połącz joystick", "en": "Connect Joystick"},
    "disconnect_joystick": {"pl": "Rozłącz joystick", "en": "Disconnect Joystick"},
    "joystick_disconnected": {"pl": "Arduino joystick: rozłączony", "en": "Arduino joystick: disconnected"},
    "joystick_connecting": {"pl": "Arduino joystick: łączenie", "en": "Arduino joystick: connecting"},
    "plate_status_idle": {"pl": "Status plate solve: bezczynny", "en": "Plate solve status: idle"},
    "plate_details_none": {"pl": "Szczegóły plate solve: brak", "en": "Plate solve details: none"},
    "rotate_title": {"pl": "Obrót", "en": "Rotate"},
    "rotate_left_90": {"pl": "Obrót w lewo 90°", "en": "Rotate Left 90°"},
    "rotate_right_90": {"pl": "Obrót w prawo 90°", "en": "Rotate Right 90°"},
    "rotate_180": {"pl": "Obrót 180°", "en": "Rotate 180°"},
    "flip_horizontal": {"pl": "Przerzuć poziomo", "en": "Flip Horizontal"},
    "flip_vertical": {"pl": "Przerzuć pionowo", "en": "Flip Vertical"},
    "angle_degrees": {"pl": "Kąt (stopnie):", "en": "Angle (degrees):"},
    "apply": {"pl": "Zastosuj", "en": "Apply"},
    "cancel": {"pl": "Anuluj", "en": "Cancel"},
}


def tr_text(language: str, key: str, default: str = None) -> str:
    lang = (language or "pl").strip().lower()
    value = I18N.get(key, {}).get(lang)
    if value is not None:
        return value
    fallback = I18N.get(key, {}).get("pl")
    if fallback is not None:
        return fallback
    return default if default is not None else key


AUTO_TEXT_MAP = {
    "Open": {"pl": "Otwórz", "en": "Open"},
    "Save": {"pl": "Zapisz", "en": "Save"},
    "Save As": {"pl": "Zapisz jako", "en": "Save As"},
    "Save As...": {"pl": "Zapisz jako...", "en": "Save As..."},
    "Undo": {"pl": "Cofnij", "en": "Undo"},
    "Redo": {"pl": "Ponów", "en": "Redo"},
    "Workspace": {"pl": "Przestrzeń robocza", "en": "Workspace"},
    "Prefs": {"pl": "Ust.", "en": "Prefs"},
    "Preferences": {"pl": "Preferencje", "en": "Preferences"},
    "Console": {"pl": "Konsola", "en": "Console"},
    "Menu": {"pl": "Menu", "en": "Menu"},
    "Histogram": {"pl": "Histogram", "en": "Histogram"},
    "AI Assistant": {"pl": "Asystent AI", "en": "AI Assistant"},
    "Magic Filter": {"pl": "Magic Filter", "en": "Magic Filter"},
    "Star Shrink": {"pl": "Zmniejsz gwiazdy", "en": "Star Shrink"},
    "Plate Solving": {"pl": "Plate Solving", "en": "Plate Solving"},
    "Run StarNet++": {"pl": "Uruchom StarNet++", "en": "Run StarNet++"},
    "3D FLY Filter": {"pl": "Filtr 3D FLY", "en": "3D FLY Filter"},
    "Gaussian Blur": {"pl": "Rozmycie Gaussa", "en": "Gaussian Blur"},
    "Rotate": {"pl": "Obrót", "en": "Rotate"},
    "Crop": {"pl": "Kadruj", "en": "Crop"},
    "Correction": {"pl": "Korekcja", "en": "Correction"},
    "Levels": {"pl": "Poziomy", "en": "Levels"},
    "Curves": {"pl": "Krzywe", "en": "Curves"},
    "Curves (LUT)": {"pl": "Krzywe (LUT)", "en": "Curves (LUT)"},
    "New Layer": {"pl": "Nowa warstwa", "en": "New Layer"},
    "Delete Layer": {"pl": "Usuń warstwę", "en": "Delete Layer"},
    "Exit": {"pl": "Wyjście", "en": "Exit"},
    "Apply": {"pl": "Zastosuj", "en": "Apply"},
    "Cancel": {"pl": "Anuluj", "en": "Cancel"},
    "Connect Joystick": {"pl": "Połącz joystick", "en": "Connect Joystick"},
    "Disconnect Joystick": {"pl": "Rozłącz joystick", "en": "Disconnect Joystick"},
    "Arduino joystick: disconnected": {"pl": "Arduino joystick: rozłączony", "en": "Arduino joystick: disconnected"},
    "Plate solve status: idle": {"pl": "Status plate solve: bezczynny", "en": "Plate solve status: idle"},
    "Plate solve details: none": {"pl": "Szczegóły plate solve: brak", "en": "Plate solve details: none"},
}


def translate_literal(language: str, text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return text
    lang = (language or "pl").strip().lower()
    for mapping in AUTO_TEXT_MAP.values():
        if cleaned == mapping.get("pl") or cleaned == mapping.get("en"):
            return mapping.get(lang, text)
    return text





# ---------- NarzÄ™dzia ----------

def np_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, ch = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)

    return QPixmap.fromImage(qimg.copy())  # đź”Ą TO JEST KLUCZ


def qimage_to_bgr_array(image: QImage) -> np.ndarray | None:
    if image is None or image.isNull():
        return None

    converted = image.convertToFormat(QImage.Format_RGBA8888)
    width = converted.width()
    height = converted.height()
    if width <= 0 or height <= 0:
        return None

    ptr = converted.bits()
    ptr.setsize(converted.byteCount())
    rgba = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
    rgb = np.ascontiguousarray(rgba[..., :3])
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def svg_icon_from_text(svg_text: str, size: int = 18) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    if SVG_AVAILABLE and QSvgRenderer is not None:
        renderer = QSvgRenderer()
        renderer.load(QByteArray(svg_text.encode("utf-8")))
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
    return QIcon(pixmap)


def apply_dialog_window_flags(widget):
    widget.setWindowFlags(widget.windowFlags() & ~Qt.WindowContextHelpButtonHint)


def apply_standard_layout_margins(layout, margins=(20, 20, 20, 20), spacing=12):
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)


def get_safe_file_dialog_options():
    options = QFileDialog.Options()
    if sys.platform.startswith("linux"):
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.DontUseCustomDirectoryIcons
    return options


def get_dark_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#1e1e1e"))
    palette.setColor(QPalette.WindowText, QColor("#ffffff"))
    palette.setColor(QPalette.Base, QColor("#121212"))
    palette.setColor(QPalette.AlternateBase, QColor("#252526"))
    palette.setColor(QPalette.ToolTipBase, QColor("#252526"))
    palette.setColor(QPalette.ToolTipText, QColor("#ffffff"))
    palette.setColor(QPalette.Text, QColor("#ffffff"))
    palette.setColor(QPalette.Button, QColor("#3a3a3a"))
    palette.setColor(QPalette.ButtonText, QColor("#ffffff"))
    palette.setColor(QPalette.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.Highlight, QColor("#007acc"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Link, QColor("#4aa3ff"))
    return palette


def apply_theme_application(app, theme_name="Fusion Dark"):
    app.setStyle("Fusion")
    app.setPalette(get_theme_palette(theme_name))
    app.setStyleSheet(get_theme_stylesheet(theme_name))


def apply_dark_application_theme(app):
    apply_theme_application(app, "Fusion Dark")


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class BlurDialog(QDialog):
    def __init__(self, parent=None, preview_callback=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("Gaussian Blur")
        self.setMinimumWidth(300)
        self.preview_callback = preview_callback
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._emit_preview)
        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        self.lbl_sigma = QLabel("Sigma (SiĹ‚a): 1.0")
        self.sld_sigma = QSlider(Qt.Horizontal)
        self.sld_sigma.setRange(1, 100)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 10.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setDecimals(1)
        self.sld_sigma.blockSignals(True)
        self.spin_sigma.blockSignals(True)
        self.sld_sigma.setValue(10)
        self.spin_sigma.setValue(1.0)
        self.sld_sigma.blockSignals(False)
        self.spin_sigma.blockSignals(False)

        sigma_row = QGridLayout()
        sigma_row.setHorizontalSpacing(10)
        sigma_row.addWidget(QLabel("Sigma"), 0, 0)
        sigma_row.addWidget(self.sld_sigma, 0, 1)
        sigma_row.addWidget(self.spin_sigma, 0, 2)

        self.sld_sigma.valueChanged.connect(lambda v: self.spin_sigma.setValue(v / 10.0))
        self.spin_sigma.valueChanged.connect(lambda v: self.sld_sigma.setValue(int(round(v * 10))))
        self.sld_sigma.valueChanged.connect(lambda v: self.lbl_sigma.setText(f"Sigma (SiĹ‚a): {v/10:.1f}"))
        self.sld_sigma.valueChanged.connect(self._on_value_changed)
        self.spin_sigma.valueChanged.connect(self._on_value_changed)

        layout.addWidget(self.lbl_sigma)
        layout.addLayout(sigma_row)

        buttons = QHBoxLayout()
        self.btn_apply = QPushButton("Zastosuj")
        self.btn_apply.setProperty("accent", True)
        self.btn_cancel = QPushButton("Anuluj")
        buttons.addWidget(self.btn_apply)
        buttons.addWidget(self.btn_cancel)
        layout.addLayout(buttons)

        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def get_value(self):
        return self.sld_sigma.value() / 10.0

    def _on_value_changed(self, _value=None):
        if self.preview_callback is not None:
            self.preview_timer.start(80)

    def _emit_preview(self):
        if self.preview_callback is not None:
            self.preview_callback(self.get_value())

    def set_preview_callback(self, preview_callback):
        self.preview_callback = preview_callback

    def reset_defaults(self, emit_preview: bool = False):
        self.sld_sigma.blockSignals(True)
        self.spin_sigma.blockSignals(True)
        self.sld_sigma.setValue(10)
        self.spin_sigma.setValue(1.0)
        self.lbl_sigma.setText("Sigma (Si\u0142a): 1.0")
        self.sld_sigma.blockSignals(False)
        self.spin_sigma.blockSignals(False)
        if emit_preview and self.preview_callback is not None:
            self.preview_callback(self.get_value())

class StarShrinkDialog(QDialog):
    def __init__(self, parent=None, preview_callback=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("Star Shrink Parameters")
        self.setMinimumWidth(350)
        self.preview_callback = preview_callback
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._emit_preview)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        self.lbl_amount = QLabel("Amount (SiĹ‚a): 0.60")
        self.sld_amount = QSlider(Qt.Horizontal)
        self.sld_amount.setRange(0, 100)
        self.spin_amount = QDoubleSpinBox()
        self.spin_amount.setRange(0.0, 1.0)
        self.spin_amount.setSingleStep(0.01)
        self.spin_amount.setDecimals(2)

        self.lbl_selection = QLabel("Selection (Maskowanie): 0.50")
        self.sld_selection = QSlider(Qt.Horizontal)
        self.sld_selection.setRange(0, 100)
        self.spin_selection = QDoubleSpinBox()
        self.spin_selection.setRange(0.0, 1.0)
        self.spin_selection.setSingleStep(0.01)
        self.spin_selection.setDecimals(2)

        self.sld_amount.blockSignals(True)
        self.spin_amount.blockSignals(True)
        self.sld_selection.blockSignals(True)
        self.spin_selection.blockSignals(True)
        self.sld_amount.setValue(60)
        self.spin_amount.setValue(0.60)
        self.sld_selection.setValue(50)
        self.spin_selection.setValue(0.50)
        self.sld_amount.blockSignals(False)
        self.spin_amount.blockSignals(False)
        self.sld_selection.blockSignals(False)
        self.spin_selection.blockSignals(False)

        amount_row = QGridLayout()
        amount_row.setHorizontalSpacing(10)
        amount_row.addWidget(QLabel("Amount"), 0, 0)
        amount_row.addWidget(self.sld_amount, 0, 1)
        amount_row.addWidget(self.spin_amount, 0, 2)

        selection_row = QGridLayout()
        selection_row.setHorizontalSpacing(10)
        selection_row.addWidget(QLabel("Selection"), 0, 0)
        selection_row.addWidget(self.sld_selection, 0, 1)
        selection_row.addWidget(self.spin_selection, 0, 2)

        self.sld_selection.valueChanged.connect(lambda v: self.spin_selection.setValue(v / 100.0))
        self.spin_selection.valueChanged.connect(lambda v: self.sld_selection.setValue(int(round(v * 100))))

        layout.addWidget(self.lbl_amount)
        layout.addLayout(amount_row)
        layout.addWidget(self.lbl_selection)
        layout.addLayout(selection_row)

        buttons = QHBoxLayout()
        self.btn_apply = QPushButton("Zastosuj")
        self.btn_apply.setProperty("accent", True)
        self.btn_cancel = QPushButton("Anuluj")
        buttons.addWidget(self.btn_apply)
        buttons.addWidget(self.btn_cancel)
        layout.addLayout(buttons)

        self.sld_amount.valueChanged.connect(self._on_slider_changed)
        self.sld_selection.valueChanged.connect(self._on_slider_changed)
        self.sld_amount.valueChanged.connect(lambda v: self.spin_amount.setValue(v / 100.0))
        self.spin_amount.valueChanged.connect(lambda v: self.sld_amount.setValue(int(round(v * 100))))
        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def _update_labels(self):
        self.lbl_amount.setText(f"Amount (SiĹ‚a): {self.sld_amount.value()/100:.2f}")
        self.lbl_selection.setText(f"Selection (Maskowanie): {self.sld_selection.value()/100:.2f}")

    def _on_slider_changed(self):
        self._update_labels()
        if self.preview_callback is not None:
            self.preview_timer.start(80)

    def _emit_preview(self):
        if self.preview_callback is not None:
            amount, selection = self.get_values()
            self.preview_callback(amount, selection)

    def set_preview_callback(self, preview_callback):
        self.preview_callback = preview_callback

    def get_values(self):
        return self.sld_amount.value() / 100.0, self.sld_selection.value() / 100.0

    def reset_defaults(self, emit_preview: bool = False):
        self.sld_amount.blockSignals(True)
        self.spin_amount.blockSignals(True)
        self.sld_selection.blockSignals(True)
        self.spin_selection.blockSignals(True)
        self.sld_amount.setValue(0)
        self.spin_amount.setValue(0.0)
        self.sld_selection.setValue(0)
        self.spin_selection.setValue(0.0)
        self.sld_amount.blockSignals(False)
        self.spin_amount.blockSignals(False)
        self.sld_selection.blockSignals(False)
        self.spin_selection.blockSignals(False)
        self._update_labels()
        if emit_preview and self.preview_callback is not None:
            amount, selection = self.get_values()
            self.preview_callback(amount, selection)


class PlateSolveDialog(QDialog):
    def __init__(self, parent=None, pixel_size_um: float = 5.4,
                 focal_length_mm: float = 800.0, api_key: str = ""):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("Plate Solving")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        self.lbl_info = QLabel("Enter pixel size and focal length for plate solving.")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.addWidget(QLabel("Pixel size (Âµm):"), 0, 0)
        self.spin_pixel_size = QDoubleSpinBox()
        self.spin_pixel_size.setRange(0.1, 50.0)
        self.spin_pixel_size.setSingleStep(0.1)
        self.spin_pixel_size.setValue(pixel_size_um)
        self.spin_pixel_size.setDecimals(2)
        grid.addWidget(self.spin_pixel_size, 0, 1)

        grid.addWidget(QLabel("Focal length (mm):"), 1, 0)
        self.spin_focal_length = QDoubleSpinBox()
        self.spin_focal_length.setRange(10.0, 10000.0)
        self.spin_focal_length.setSingleStep(1.0)
        self.spin_focal_length.setValue(focal_length_mm)
        self.spin_focal_length.setDecimals(1)
        grid.addWidget(self.spin_focal_length, 1, 1)
        layout.addLayout(grid)

        self.lbl_api_key = QLabel("Astrometry.net API key (optional, fallback if local solver unavailable):")
        self.lbl_api_key.setWordWrap(True)
        layout.addWidget(self.lbl_api_key)
        self.edit_api_key = QLineEdit()
        self.edit_api_key.setText(api_key)
        self.edit_api_key.setPlaceholderText("Enter API key if you want Astrometry.net fallback")
        layout.addWidget(self.edit_api_key)

        self.chk_overlay = QCheckBox("Draw overlay after solve")
        self.chk_overlay.setChecked(True)
        layout.addWidget(self.chk_overlay)

        buttons = QHBoxLayout()
        self.btn_solve = QPushButton("Solve")
        self.btn_solve.setProperty("accent", True)
        self.btn_cancel = QPushButton("Cancel")
        buttons.addWidget(self.btn_solve)
        buttons.addWidget(self.btn_cancel)
        layout.addLayout(buttons)

        self.btn_solve.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def get_parameters(self):
        return {
            "pixel_size_um": float(self.spin_pixel_size.value()),
            "focal_length_mm": float(self.spin_focal_length.value()),
            "api_key": self.edit_api_key.text().strip(),
            "overlay_enabled": self.chk_overlay.isChecked(),
        }

    def set_parameters(self, pixel_size_um: float, focal_length_mm: float, api_key: str):
        self.spin_pixel_size.setValue(float(pixel_size_um))
        self.spin_focal_length.setValue(float(focal_length_mm))
        self.edit_api_key.setText(api_key or "")


class CircularProgressBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 100
        self._value = 0
        self._text_visible = True
        self._format = "%p%"
        self._indeterminate = False
        self._indeterminate_angle = 0
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(35)
        self._animation_timer.timeout.connect(self._rotate_indeterminate_arc)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def minimum(self) -> int:
        return self._minimum

    def maximum(self) -> int:
        return self._maximum

    def value(self) -> int:
        return self._value

    def setTextVisible(self, visible: bool):
        self._text_visible = bool(visible)
        self.update()

    def isTextVisible(self) -> bool:
        return self._text_visible

    def setFormat(self, fmt: str):
        self._format = str(fmt)
        self.update()

    def format(self) -> str:
        return self._format

    def resetFormat(self):
        self._format = "%p%"
        self.update()

    def setMinimum(self, minimum: int):
        self.setRange(int(minimum), self._maximum)

    def setMaximum(self, maximum: int):
        self.setRange(self._minimum, int(maximum))

    def setRange(self, minimum: int, maximum: int):
        minimum = int(minimum)
        maximum = int(maximum)
        if maximum < minimum:
            maximum = minimum

        self._minimum = minimum
        self._maximum = maximum
        self._indeterminate = minimum == 0 and maximum == 0

        if self._indeterminate:
            if not self._animation_timer.isActive():
                self._animation_timer.start()
        else:
            self._animation_timer.stop()
            self._value = max(self._minimum, min(self._maximum, self._value))

        self.update()

    def setValue(self, value: int):
        value = int(value)
        self._value = max(self._minimum, min(self._maximum, value))
        self.update()

    def sizeHint(self):
        return QSize(64, 64)

    def minimumSizeHint(self):
        return QSize(48, 48)

    def _rotate_indeterminate_arc(self):
        self._indeterminate_angle = (self._indeterminate_angle + 8) % 360
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        side = max(10, min(self.width(), self.height()) - 4)
        x0 = (self.width() - side) / 2.0
        y0 = (self.height() - side) / 2.0
        arc_rect = QRectF(x0, y0, side, side)
        ring_width = max(6.0, side * 0.2)

        track_pen = QPen(QColor("#ffffff"), ring_width)
        track_pen.setCapStyle(Qt.FlatCap)
        painter.setPen(track_pen)
        painter.drawArc(arc_rect, 0, 360 * 16)

        painter.setPen(QPen(QColor("#000000"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(arc_rect)

        inner_offset = ring_width / 2.0
        inner_rect = QRectF(
            arc_rect.left() + inner_offset,
            arc_rect.top() + inner_offset,
            max(1.0, arc_rect.width() - ring_width),
            max(1.0, arc_rect.height() - ring_width),
        )
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(inner_rect)

        progress_pen = QPen(QColor("#1bff00"), max(3.0, ring_width - 2.0))
        progress_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(progress_pen)

        if self._indeterminate:
            start_angle = (90 - self._indeterminate_angle) * 16
            span_angle = -95 * 16
        else:
            span = self._maximum - self._minimum
            if span <= 0:
                progress = 0.0
            else:
                progress = (self._value - self._minimum) / float(span)
            progress = max(0.0, min(1.0, progress))
            start_angle = 90 * 16
            span_angle = int(-progress * 360 * 16)

        painter.drawArc(arc_rect, start_angle, span_angle)

        if self._text_visible and not self._indeterminate:
            span = self._maximum - self._minimum
            percent = 0 if span <= 0 else int(round((self._value - self._minimum) * 100.0 / span))
            text = self._format
            text = text.replace("%p", str(percent)).replace("%v", str(self._value)).replace("%m", str(self._maximum))
            painter.setPen(QPen(QColor("#000000"), 1))
            font = painter.font()
            font.setPointSize(max(9, int(side * 0.24)))
            font.setBold(False)
            painter.setFont(font)
            painter.drawText(inner_rect, Qt.AlignCenter, text)


class PlateSolvingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("Plate solving dialog")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        self.lbl_plate_solve_status = QLabel("Plate solve status: idle")
        self.lbl_plate_solve_status.setWordWrap(True)
        layout.addWidget(self.lbl_plate_solve_status)

        self.plate_solve_progress = CircularProgressBar()
        self.plate_solve_progress.setRange(0, 1)
        self.plate_solve_progress.setValue(0)
        self.plate_solve_progress.setTextVisible(False)
        layout.addWidget(self.plate_solve_progress, 0, Qt.AlignHCenter)

        self.lbl_plate_solve_details = QLabel("Plate solve details: none")
        self.lbl_plate_solve_details.setWordWrap(True)
        layout.addWidget(self.lbl_plate_solve_details)

        self.lbl_plate_main_object = QLabel("Object: unknown")
        self.lbl_plate_main_object.setWordWrap(True)
        layout.addWidget(self.lbl_plate_main_object)

        self.lbl_plate_catalog = QLabel("Catalog: unknown")
        self.lbl_plate_catalog.setWordWrap(True)
        layout.addWidget(self.lbl_plate_catalog)

        self.lbl_plate_designation = QLabel("Designation: unknown")
        self.lbl_plate_designation.setWordWrap(True)
        layout.addWidget(self.lbl_plate_designation)

        self.lbl_plate_objects_in_field = QLabel("Objects in Field: none")
        self.lbl_plate_objects_in_field.setWordWrap(True)
        layout.addWidget(self.lbl_plate_objects_in_field)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        layout.addWidget(self.btn_close)


class RotatePreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(260)
        self._pixmap = None

    def set_image(self, img: np.ndarray):
        if img is None:
            self._pixmap = None
        else:
            self._pixmap = np_to_qpixmap(img)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1f1f1f"))
        if self._pixmap is None:
            return
        avail_w = max(1, self.width() - 12)
        avail_h = max(1, self.height() - 12)
        scaled = self._pixmap.scaled(avail_w, avail_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x0 = (self.width() - scaled.width()) // 2
        y0 = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x0, y0, scaled)


class RotateDialog(QDialog):
    def __init__(self, parent=None, source_img: np.ndarray = None, language: str = "pl"):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self._language = (language or "pl").strip().lower()
        self.setWindowTitle(tr_text(self._language, "rotate_title", "Rotate"))
        self.setMinimumWidth(560)
        self.setMinimumHeight(520)

        self._source_img = source_img.copy() if isinstance(source_img, np.ndarray) else None
        self._preview_img = self._prepare_preview_image(self._source_img)
        self._flip_h = False
        self._flip_v = False
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._refresh_preview)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        self.preview = RotatePreviewWidget(self)
        layout.addWidget(self.preview, 1)

        quick_row = QGridLayout()
        quick_row.setHorizontalSpacing(8)
        quick_row.setVerticalSpacing(8)
        self.btn_rot_left = QPushButton(tr_text(self._language, "rotate_left_90", "Rotate Left 90°"))
        self.btn_rot_right = QPushButton(tr_text(self._language, "rotate_right_90", "Rotate Right 90°"))
        self.btn_rot_180 = QPushButton(tr_text(self._language, "rotate_180", "Rotate 180°"))
        self.btn_flip_h = QPushButton(tr_text(self._language, "flip_horizontal", "Flip Horizontal"))
        self.btn_flip_v = QPushButton(tr_text(self._language, "flip_vertical", "Flip Vertical"))
        self.btn_flip_h.setCheckable(True)
        self.btn_flip_v.setCheckable(True)
        quick_row.addWidget(self.btn_rot_left, 0, 0)
        quick_row.addWidget(self.btn_rot_right, 0, 1)
        quick_row.addWidget(self.btn_rot_180, 0, 2)
        quick_row.addWidget(self.btn_flip_h, 1, 0, 1, 2)
        quick_row.addWidget(self.btn_flip_v, 1, 2)
        layout.addLayout(quick_row)

        rotate_icons_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets",
            "icons",
            "rotate_dialog",
        )
        rotate_icons = {
            "btn_rot_left": "rotate_left.svg",
            "btn_rot_right": "rotate_right.svg",
            "btn_rot_180": "rotate_180.svg",
            "btn_flip_h": "flip_horizontal.svg",
            "btn_flip_v": "flip_vertical.svg",
        }
        for button_name, icon_file in rotate_icons.items():
            button = getattr(self, button_name, None)
            if button is None:
                continue
            icon_path = os.path.join(rotate_icons_dir, icon_file)
            if os.path.exists(icon_path):
                button.setIcon(QIcon(icon_path))
                button.setIconSize(QSize(18, 18))

        row = QHBoxLayout()
        row.addWidget(QLabel(tr_text(self._language, "angle_degrees", "Angle (degrees):")))
        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(-360.0, 360.0)
        self.spin_angle.setDecimals(1)
        self.spin_angle.setSingleStep(1.0)
        self.spin_angle.setValue(0.0)
        self.sld_angle = QSlider(Qt.Horizontal)
        self.sld_angle.setRange(-3600, 3600)
        self.sld_angle.setSingleStep(1)
        self.sld_angle.setPageStep(10)
        self.sld_angle.setValue(0)
        row.addWidget(self.sld_angle, 1)
        row.addWidget(self.spin_angle)
        layout.addLayout(row)

        buttons = QHBoxLayout()
        self.btn_apply = QPushButton(tr_text(self._language, "apply", "Apply"))
        self.btn_apply.setProperty("accent", True)
        self.btn_cancel = QPushButton(tr_text(self._language, "cancel", "Cancel"))
        buttons.addWidget(self.btn_apply)
        buttons.addWidget(self.btn_cancel)
        layout.addLayout(buttons)

        self.btn_rot_left.clicked.connect(lambda: self.spin_angle.setValue(self.spin_angle.value() - 90.0))
        self.btn_rot_right.clicked.connect(lambda: self.spin_angle.setValue(self.spin_angle.value() + 90.0))
        self.btn_rot_180.clicked.connect(lambda: self.spin_angle.setValue(self.spin_angle.value() + 180.0))
        self.spin_angle.valueChanged.connect(self._schedule_preview)
        self.spin_angle.valueChanged.connect(self._sync_angle_slider_from_spin)
        self.sld_angle.valueChanged.connect(self._sync_angle_spin_from_slider)
        self.btn_flip_h.toggled.connect(self._on_flip_h_toggled)
        self.btn_flip_v.toggled.connect(self._on_flip_v_toggled)

        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self._refresh_preview()

    def _prepare_preview_image(self, img: np.ndarray):
        if img is None:
            return None
        h, w = img.shape[:2]
        max_side = max(h, w)
        if max_side <= 900:
            return img.copy()
        scale = 900.0 / float(max_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _apply_transform(self, img: np.ndarray) -> np.ndarray:
        out = img
        if self._flip_h and self._flip_v:
            out = cv2.flip(out, -1)
        elif self._flip_h:
            out = cv2.flip(out, 1)
        elif self._flip_v:
            out = cv2.flip(out, 0)

        angle = float(self.spin_angle.value())
        if abs(angle) >= 1e-6:
            h, w = out.shape[:2]
            center = (w / 2.0, h / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos = abs(matrix[0, 0])
            sin = abs(matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            matrix[0, 2] += (new_w / 2) - center[0]
            matrix[1, 2] += (new_h / 2) - center[1]
            border_value = 0 if out.ndim == 2 else (0, 0, 0)
            out = cv2.warpAffine(
                out,
                matrix,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value,
            )
        return out

    def _schedule_preview(self, _value=None):
        self.preview_timer.start(60)

    def _sync_angle_slider_from_spin(self, value: float):
        slider_value = int(round(float(value) * 10.0))
        if self.sld_angle.value() != slider_value:
            self.sld_angle.blockSignals(True)
            self.sld_angle.setValue(slider_value)
            self.sld_angle.blockSignals(False)

    def _sync_angle_spin_from_slider(self, value: int):
        spin_value = float(value) / 10.0
        if abs(self.spin_angle.value() - spin_value) > 1e-9:
            self.spin_angle.blockSignals(True)
            self.spin_angle.setValue(spin_value)
            self.spin_angle.blockSignals(False)
            self._schedule_preview()

    def _refresh_preview(self):
        if self._preview_img is None:
            self.preview.set_image(None)
            return
        self.preview.set_image(self._apply_transform(self._preview_img))

    def _on_flip_h_toggled(self, checked: bool):
        self._flip_h = bool(checked)
        self._schedule_preview()

    def _on_flip_v_toggled(self, checked: bool):
        self._flip_v = bool(checked)
        self._schedule_preview()

    def get_transform(self):
        return float(self.spin_angle.value()), bool(self._flip_h), bool(self._flip_v)


class CropPreviewWidget(QWidget):
    rectChanged = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(360)
        self.setMouseTracking(True)
        self._img = None
        self._pixmap = None
        self._display_rect = QRectF()
        self._crop = (0, 0, 1, 1)
        self._dragging = False
        self._drag_mode = None
        self._drag_dx = 0
        self._drag_dy = 0
        self._resize_handle = None
        self._resize_anchor = (0, 0)
        self._handle_radius = 6
        self._grid_mode = "Brak"
        self._aspect_ratio = None
        self._show_handles = True

    def set_image(self, img: np.ndarray):
        self._img = img.copy()
        self._pixmap = np_to_qpixmap(self._img)
        h, w = self._img.shape[:2]
        self._crop = (0, 0, max(1, w), max(1, h))
        self.update()

    def set_crop_rect(self, x: int, y: int, w: int, h: int):
        self._crop = (int(x), int(y), int(w), int(h))
        self.update()

    def set_aspect_ratio(self, ratio):
        self._aspect_ratio = ratio

    def set_grid_mode(self, mode: str):
        self._grid_mode = mode or "Brak"
        self.update()

    def set_show_handles(self, visible: bool):
        self._show_handles = bool(visible)
        self.update()

    def _image_to_widget(self, x: float, y: float):
        if self._img is None or self._display_rect.width() <= 0 or self._display_rect.height() <= 0:
            return 0.0, 0.0
        img_h, img_w = self._img.shape[:2]
        wx = self._display_rect.left() + (x / img_w) * self._display_rect.width()
        wy = self._display_rect.top() + (y / img_h) * self._display_rect.height()
        return wx, wy

    def _widget_to_image(self, wx: float, wy: float):
        if self._img is None or self._display_rect.width() <= 0 or self._display_rect.height() <= 0:
            return 0, 0
        img_h, img_w = self._img.shape[:2]
        nx = (wx - self._display_rect.left()) / max(1.0, self._display_rect.width())
        ny = (wy - self._display_rect.top()) / max(1.0, self._display_rect.height())
        x = int(round(nx * img_w))
        y = int(round(ny * img_h))
        x = max(0, min(img_w, x))
        y = max(0, min(img_h, y))
        return x, y

    def _crop_widget_rect(self) -> QRectF:
        cx, cy, cw, ch = self._crop
        x1, y1 = self._image_to_widget(cx, cy)
        x2, y2 = self._image_to_widget(cx + cw, cy + ch)
        return QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    def _handle_points(self, rect: QRectF):
        return {
            "nw": QPointF(rect.left(), rect.top()),
            "ne": QPointF(rect.right(), rect.top()),
            "sw": QPointF(rect.left(), rect.bottom()),
            "se": QPointF(rect.right(), rect.bottom()),
            "n": QPointF(rect.center().x(), rect.top()),
            "s": QPointF(rect.center().x(), rect.bottom()),
            "w": QPointF(rect.left(), rect.center().y()),
            "e": QPointF(rect.right(), rect.center().y()),
        }

    def _hit_test_handle(self, pos, rect: QRectF):
        px = float(pos.x())
        py = float(pos.y())
        threshold = float(self._handle_radius + 3)
        for name, p in self._handle_points(rect).items():
            dx = px - p.x()
            dy = py - p.y()
            if (dx * dx + dy * dy) <= (threshold * threshold):
                return name
        return None

    def _cursor_for_handle(self, handle_name: str):
        if handle_name in ("nw", "se"):
            return Qt.SizeFDiagCursor
        if handle_name in ("ne", "sw"):
            return Qt.SizeBDiagCursor
        if handle_name in ("n", "s"):
            return Qt.SizeVerCursor
        if handle_name in ("e", "w"):
            return Qt.SizeHorCursor
        return Qt.ArrowCursor

    def _build_resized_rect(self, handle: str, ix: int, iy: int):
        img_h, img_w = self._img.shape[:2]
        ax, ay = self._resize_anchor
        cx, cy, cw, ch = self._crop

        if handle == "n":
            top = max(0, min(cy + ch - 1, iy))
            left = cx
            right = cx + cw
            bottom = cy + ch
            if self._aspect_ratio is not None and self._aspect_ratio > 0:
                ratio = float(self._aspect_ratio)
                new_h = max(1, bottom - top)
                new_w = max(1, int(round(new_h * ratio)))
                center_x = cx + cw / 2.0
                left = int(round(center_x - new_w / 2.0))
                right = left + new_w
                if left < 0:
                    left = 0
                    right = min(img_w, new_w)
                if right > img_w:
                    right = img_w
                    left = max(0, right - new_w)
            left = max(0, min(img_w - 1, left))
            top = max(0, min(img_h - 1, top))
            right = max(left + 1, min(img_w, right))
            bottom = max(top + 1, min(img_h, bottom))
            return left, top, right - left, bottom - top

        if handle == "s":
            top = cy
            left = cx
            right = cx + cw
            bottom = max(cy + 1, min(img_h, iy))
            if self._aspect_ratio is not None and self._aspect_ratio > 0:
                ratio = float(self._aspect_ratio)
                new_h = max(1, bottom - top)
                new_w = max(1, int(round(new_h * ratio)))
                center_x = cx + cw / 2.0
                left = int(round(center_x - new_w / 2.0))
                right = left + new_w
                if left < 0:
                    left = 0
                    right = min(img_w, new_w)
                if right > img_w:
                    right = img_w
                    left = max(0, right - new_w)
            left = max(0, min(img_w - 1, left))
            top = max(0, min(img_h - 1, top))
            right = max(left + 1, min(img_w, right))
            bottom = max(top + 1, min(img_h, bottom))
            return left, top, right - left, bottom - top

        if handle == "w":
            left = max(0, min(cx + cw - 1, ix))
            top = cy
            right = cx + cw
            bottom = cy + ch
            if self._aspect_ratio is not None and self._aspect_ratio > 0:
                ratio = float(self._aspect_ratio)
                new_w = max(1, right - left)
                new_h = max(1, int(round(new_w / ratio)))
                center_y = cy + ch / 2.0
                top = int(round(center_y - new_h / 2.0))
                bottom = top + new_h
                if top < 0:
                    top = 0
                    bottom = min(img_h, new_h)
                if bottom > img_h:
                    bottom = img_h
                    top = max(0, bottom - new_h)
            left = max(0, min(img_w - 1, left))
            top = max(0, min(img_h - 1, top))
            right = max(left + 1, min(img_w, right))
            bottom = max(top + 1, min(img_h, bottom))
            return left, top, right - left, bottom - top

        if handle == "e":
            left = cx
            top = cy
            right = max(cx + 1, min(img_w, ix))
            bottom = cy + ch
            if self._aspect_ratio is not None and self._aspect_ratio > 0:
                ratio = float(self._aspect_ratio)
                new_w = max(1, right - left)
                new_h = max(1, int(round(new_w / ratio)))
                center_y = cy + ch / 2.0
                top = int(round(center_y - new_h / 2.0))
                bottom = top + new_h
                if top < 0:
                    top = 0
                    bottom = min(img_h, new_h)
                if bottom > img_h:
                    bottom = img_h
                    top = max(0, bottom - new_h)
            left = max(0, min(img_w - 1, left))
            top = max(0, min(img_h - 1, top))
            right = max(left + 1, min(img_w, right))
            bottom = max(top + 1, min(img_h, bottom))
            return left, top, right - left, bottom - top

        if handle == "nw":
            ix = max(0, min(ax - 1, ix))
            iy = max(0, min(ay - 1, iy))
            max_dx, max_dy = ax, ay
            sx, sy = -1.0, -1.0
        elif handle == "ne":
            ix = max(ax + 1, min(img_w, ix))
            iy = max(0, min(ay - 1, iy))
            max_dx, max_dy = img_w - ax, ay
            sx, sy = 1.0, -1.0
        elif handle == "sw":
            ix = max(0, min(ax - 1, ix))
            iy = max(ay + 1, min(img_h, iy))
            max_dx, max_dy = ax, img_h - ay
            sx, sy = -1.0, 1.0
        else:
            ix = max(ax + 1, min(img_w, ix))
            iy = max(ay + 1, min(img_h, iy))
            max_dx, max_dy = img_w - ax, img_h - ay
            sx, sy = 1.0, 1.0

        dx = max(1.0, min(float(max_dx), abs(float(ix - ax))))
        dy = max(1.0, min(float(max_dy), abs(float(iy - ay))))

        if self._aspect_ratio is not None and self._aspect_ratio > 0:
            ratio = float(self._aspect_ratio)
            if dx / max(1e-6, dy) > ratio:
                dx = dy * ratio
            else:
                dy = dx / ratio

            if dx > max_dx:
                dx = float(max_dx)
                dy = dx / ratio
            if dy > max_dy:
                dy = float(max_dy)
                dx = dy * ratio

            dx = max(1.0, min(float(max_dx), dx))
            dy = max(1.0, min(float(max_dy), dy))

        rx = int(round(ax + sx * dx))
        ry = int(round(ay + sy * dy))

        left = min(ax, rx)
        top = min(ay, ry)
        right = max(ax, rx)
        bottom = max(ay, ry)

        left = max(0, min(img_w - 1, left))
        top = max(0, min(img_h - 1, top))
        right = max(left + 1, min(img_w, right))
        bottom = max(top + 1, min(img_h, bottom))
        return left, top, right - left, bottom - top

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1f1f1f"))
        if self._pixmap is None:
            return

        avail_w = max(1, self.width() - 12)
        avail_h = max(1, self.height() - 12)
        scaled = self._pixmap.scaled(avail_w, avail_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x0 = (self.width() - scaled.width()) / 2.0
        y0 = (self.height() - scaled.height()) / 2.0
        self._display_rect = QRectF(x0, y0, scaled.width(), scaled.height())
        painter.drawPixmap(int(x0), int(y0), scaled)

        crop_rect = self._crop_widget_rect()

        shade = QColor(0, 0, 0, 120)
        painter.fillRect(QRectF(self._display_rect.left(), self._display_rect.top(), self._display_rect.width(), max(0.0, crop_rect.top() - self._display_rect.top())), shade)
        painter.fillRect(QRectF(self._display_rect.left(), crop_rect.bottom(), self._display_rect.width(), max(0.0, self._display_rect.bottom() - crop_rect.bottom())), shade)
        painter.fillRect(QRectF(self._display_rect.left(), crop_rect.top(), max(0.0, crop_rect.left() - self._display_rect.left()), crop_rect.height()), shade)
        painter.fillRect(QRectF(crop_rect.right(), crop_rect.top(), max(0.0, self._display_rect.right() - crop_rect.right()), crop_rect.height()), shade)

        painter.setPen(QPen(QColor("#00e5ff"), 2))
        painter.drawRect(crop_rect)

        self._draw_grid(painter, crop_rect)
        if self._show_handles:
            painter.setBrush(QColor("#00e5ff"))
            painter.setPen(QPen(QColor("#003b42"), 1))
            r = float(self._handle_radius)
            for name, p in self._handle_points(crop_rect).items():
                if name in ("n", "s", "e", "w"):
                    painter.drawRect(QRectF(p.x() - r, p.y() - r, 2 * r, 2 * r))
                else:
                    painter.drawEllipse(QRectF(p.x() - r, p.y() - r, 2 * r, 2 * r))

    def _draw_grid(self, painter: QPainter, rect: QRectF):
        if rect.width() < 4 or rect.height() < 4:
            return
        pen = QPen(QColor(255, 255, 255, 155), 1, Qt.DashLine)
        painter.setPen(pen)
        mode = self._grid_mode
        if mode == "TrĂłjpodziaĹ‚":
            for i in (1, 2):
                x = rect.left() + rect.width() * (i / 3.0)
                y = rect.top() + rect.height() * (i / 3.0)
                painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
                painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
        elif mode == "ZĹ‚oty podziaĹ‚":
            phi = 0.61803398875
            for r in (phi, 1.0 - phi):
                x = rect.left() + rect.width() * r
                y = rect.top() + rect.height() * r
                painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
                painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
        elif mode == "KrzyĹĽ":
            painter.drawLine(int(rect.center().x()), int(rect.top()), int(rect.center().x()), int(rect.bottom()))
            painter.drawLine(int(rect.left()), int(rect.center().y()), int(rect.right()), int(rect.center().y()))
        elif mode == "Diagonale":
            painter.drawLine(int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom()))
            painter.drawLine(int(rect.right()), int(rect.top()), int(rect.left()), int(rect.bottom()))

    def mousePressEvent(self, event):
        if self._img is None or event.button() != Qt.LeftButton:
            return
        crop_rect = self._crop_widget_rect()
        handle = self._hit_test_handle(event.pos(), crop_rect)
        if handle is not None:
            cx, cy, cw, ch = self._crop
            opposite = {
                "nw": (cx + cw, cy + ch),
                "ne": (cx, cy + ch),
                "sw": (cx + cw, cy),
                "se": (cx, cy),
                "n": (cx + cw // 2, cy + ch),
                "s": (cx + cw // 2, cy),
                "w": (cx + cw, cy + ch // 2),
                "e": (cx, cy + ch // 2),
            }
            self._dragging = True
            self._drag_mode = "resize"
            self._resize_handle = handle
            self._resize_anchor = opposite[handle]
            self.setCursor(self._cursor_for_handle(handle))
            super().mousePressEvent(event)
            return
        cx, cy, cw, ch = self._crop
        ix, iy = self._widget_to_image(event.pos().x(), event.pos().y())
        if cx <= ix <= cx + cw and cy <= iy <= cy + ch:
            self._dragging = True
            self._drag_mode = "move"
            self._drag_dx = ix - cx
            self._drag_dy = iy - cy
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._img is None:
            return
        if self._dragging:
            ix, iy = self._widget_to_image(event.pos().x(), event.pos().y())
            if self._drag_mode == "resize" and self._resize_handle is not None:
                x, y, w, h = self._build_resized_rect(self._resize_handle, ix, iy)
                self._crop = (int(x), int(y), int(w), int(h))
                self.rectChanged.emit(int(x), int(y), int(w), int(h))
            else:
                img_h, img_w = self._img.shape[:2]
                _, _, cw, ch = self._crop
                new_x = max(0, min(img_w - cw, ix - self._drag_dx))
                new_y = max(0, min(img_h - ch, iy - self._drag_dy))
                self._crop = (int(new_x), int(new_y), int(cw), int(ch))
                self.rectChanged.emit(int(new_x), int(new_y), int(cw), int(ch))
            self.update()
        else:
            crop_rect = self._crop_widget_rect()
            handle = self._hit_test_handle(event.pos(), crop_rect)
            if handle is not None:
                self.setCursor(self._cursor_for_handle(handle))
            elif crop_rect.contains(event.pos()):
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._dragging = False
        self._drag_mode = None
        self._resize_handle = None
        self.unsetCursor()
        super().mouseReleaseEvent(event)


class CropDialog(QDialog):
    previewRectChanged = pyqtSignal(int, int, int, int)
    gridChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("Crop")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("Grid:"))
        self.combo_grid = QComboBox()
        self.combo_grid.addItems([
            "Brak",
            "TrĂłjpodziaĹ‚",
            "ZĹ‚oty podziaĹ‚",
            "KrzyĹĽ",
            "Diagonale",
        ])
        options_row.addWidget(self.combo_grid)
        options_row.addSpacing(12)
        options_row.addWidget(QLabel("Proporcje:"))
        self.combo_ratio = QComboBox()
        self.combo_ratio.addItems([
            "Dowolny",
            "1:1",
            "3:2",
            "4:3",
            "16:9",
            "21:9",
            "9:16",
        ])
        options_row.addWidget(self.combo_ratio)
        options_row.addStretch(1)
        layout.addLayout(options_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.spin_x = QDoubleSpinBox()
        self.spin_y = QDoubleSpinBox()
        self.spin_w = QDoubleSpinBox()
        self.spin_h = QDoubleSpinBox()
        for spin in (self.spin_x, self.spin_y, self.spin_w, self.spin_h):
            spin.setRange(0, 100000)
            spin.setDecimals(0)
            spin.setSingleStep(1.0)

        grid.addWidget(QLabel("X:"), 0, 0)
        grid.addWidget(self.spin_x, 0, 1)
        grid.addWidget(QLabel("Y:"), 1, 0)
        grid.addWidget(self.spin_y, 1, 1)
        grid.addWidget(QLabel("Width:"), 2, 0)
        grid.addWidget(self.spin_w, 2, 1)
        grid.addWidget(QLabel("Height:"), 3, 0)
        grid.addWidget(self.spin_h, 3, 1)
        layout.addLayout(grid)

        self.spin_x.valueChanged.connect(self._on_spins_changed)
        self.spin_y.valueChanged.connect(self._on_spins_changed)
        self.spin_w.valueChanged.connect(self._on_spins_changed)
        self.spin_h.valueChanged.connect(self._on_spins_changed)
        self.combo_grid.currentIndexChanged.connect(self.gridChanged.emit)
        self.combo_ratio.currentTextChanged.connect(self._on_ratio_changed)

        buttons = QHBoxLayout()
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setProperty("accent", True)
        self.btn_cancel = QPushButton("Cancel")
        buttons.addWidget(self.btn_apply)
        buttons.addWidget(self.btn_cancel)
        layout.addLayout(buttons)

        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self._updating = False
        self._img_w = 1
        self._img_h = 1
        self._aspect_ratio = None

    def set_image(self, img: np.ndarray):
        if img is None:
            return
        self._img_h, self._img_w = img.shape[:2]
        self._set_rect(0, 0, self._img_w, self._img_h, emit_signal=False)

    def _set_rect(self, x: int, y: int, w: int, h: int, emit_signal: bool = True):
        x, y, w, h = self._normalize_rect(x, y, w, h)
        self._updating = True
        self.spin_x.setRange(0, max(0, self._img_w - 1))
        self.spin_y.setRange(0, max(0, self._img_h - 1))
        self.spin_x.setValue(float(x))
        self.spin_y.setValue(float(y))
        self.spin_w.setValue(float(w))
        self.spin_h.setValue(float(h))
        self._updating = False
        self._update_limits()
        if emit_signal:
            self.previewRectChanged.emit(int(x), int(y), int(w), int(h))

    def _normalize_rect(self, x: int, y: int, w: int, h: int):
        x = int(max(0, min(x, self._img_w - 1)))
        y = int(max(0, min(y, self._img_h - 1)))
        max_w = max(1, self._img_w - x)
        max_h = max(1, self._img_h - y)
        w = int(max(1, min(w, max_w)))
        h = int(max(1, min(h, max_h)))

        if self._aspect_ratio is not None:
            ratio = float(self._aspect_ratio)
            h_by_w = int(round(w / ratio))
            w_by_h = int(round(h * ratio))
            if 1 <= h_by_w <= max_h:
                h = h_by_w
            elif 1 <= w_by_h <= max_w:
                w = w_by_h
            else:
                if max_w / max_h > ratio:
                    h = max_h
                    w = max(1, min(max_w, int(round(h * ratio))))
                else:
                    w = max_w
                    h = max(1, min(max_h, int(round(w / ratio))))

        return x, y, w, h

    def _update_limits(self):
        max_w = max(1.0, float(self._img_w) - float(self.spin_x.value()))
        max_h = max(1.0, float(self._img_h) - float(self.spin_y.value()))
        self.spin_w.setRange(1.0, max_w)
        self.spin_h.setRange(1.0, max_h)

    def _ratio_value(self):
        text = self.combo_ratio.currentText().strip()
        if text == "Dowolny":
            return None
        if ":" in text:
            a, b = text.split(":", 1)
            try:
                a_val = float(a)
                b_val = float(b)
                if a_val > 0 and b_val > 0:
                    return a_val / b_val
            except Exception:
                return None
        return None

    def _on_ratio_changed(self, _text: str):
        self._aspect_ratio = self._ratio_value()
        self._on_spins_changed()

    def _on_spins_changed(self):
        if self._updating:
            return
        self._set_rect(
            int(self.spin_x.value()),
            int(self.spin_y.value()),
            int(self.spin_w.value()),
            int(self.spin_h.value()),
        )

    def set_rect_from_overlay(self, x: int, y: int, w: int, h: int):
        self._set_rect(int(x), int(y), int(w), int(h), emit_signal=False)

    def get_aspect_ratio(self):
        return self._aspect_ratio

    def get_rect(self):
        return (
            int(self.spin_x.value()),
            int(self.spin_y.value()),
            int(self.spin_w.value()),
            int(self.spin_h.value()),
        )

    def reset_defaults(self):
        self.combo_grid.blockSignals(True)
        self.combo_ratio.blockSignals(True)
        self.combo_grid.setCurrentIndex(0)
        self.combo_ratio.setCurrentIndex(0)
        self.combo_grid.blockSignals(False)
        self.combo_ratio.blockSignals(False)
        self._aspect_ratio = None
        self._set_rect(0, 0, self._img_w, self._img_h, emit_signal=False)


class StarNetDialog(QDialog):
    def __init__(self, parent=None, stride: int = 16, starnet_path: str = None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("StarNet++ Settings")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        self.lbl_info = QLabel("Configure StarNet++ parameters before running.")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        self.lbl_hint = QLabel("Hint: for StarNet2 CLI, lower stride (even 2) can improve quality but is slower.")
        self.lbl_hint.setWordWrap(True)
        layout.addWidget(self.lbl_hint)

        self.lbl_executable = QLabel(
            f"StarNet++ executable: {os.path.basename(starnet_path) if starnet_path else 'Not selected'}"
        )
        self.lbl_executable.setWordWrap(True)
        layout.addWidget(self.lbl_executable)

        stride_layout = QHBoxLayout()
        stride_layout.addWidget(QLabel("Stride:"))
        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(2, 512)
        self.spin_stride.setValue(int(stride))
        self.spin_stride.setSingleStep(2)
        stride_layout.addWidget(self.spin_stride)
        layout.addLayout(stride_layout)

        button_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_run.setProperty("accent", True)
        self.btn_cancel = QPushButton("Cancel")
        button_layout.addWidget(self.btn_run)
        button_layout.addWidget(self.btn_cancel)
        layout.addLayout(button_layout)

        self.btn_run.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def get_parameters(self):
        return {
            "stride": int(self.spin_stride.value()),
        }

    def set_parameters(self, stride: int, starnet_path: str):
        self.spin_stride.setValue(int(stride))
        self.lbl_executable.setText(
            f"StarNet++ executable: {os.path.basename(starnet_path) if starnet_path else 'Not selected'}"
        )


class DeepSNRDialog(QDialog):
    def __init__(self, parent=None, deepsnr_path: str = None, deepsnr_args: str = "{input}"):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("deepSNR Settings")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        self.lbl_info = QLabel("Configure deepSNR CLI parameters before running.")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        self.lbl_executable = QLabel(
            f"deepSNR executable: {os.path.basename(deepsnr_path) if deepsnr_path else 'Not selected'}"
        )
        self.lbl_executable.setWordWrap(True)
        layout.addWidget(self.lbl_executable)

        layout.addWidget(QLabel("Arguments:"))
        self.edit_args = QLineEdit(str(deepsnr_args or "{input}"))
        self.edit_args.setPlaceholderText("-i \"{input}\" -o \"{output}\" -q -m 2")
        layout.addWidget(self.edit_args)

        self.lbl_help = QLabel("Use {input} and optional {output} placeholders.")
        self.lbl_help.setWordWrap(True)
        layout.addWidget(self.lbl_help)

        button_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_run.setProperty("accent", True)
        self.btn_cancel = QPushButton("Cancel")
        button_layout.addWidget(self.btn_run)
        button_layout.addWidget(self.btn_cancel)
        layout.addLayout(button_layout)

        self.btn_run.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def get_parameters(self):
        args = str(self.edit_args.text() or "").strip() or "{input}"
        return {
            "args": args,
        }

    def set_parameters(self, deepsnr_path: str, deepsnr_args: str):
        self.edit_args.setText(str(deepsnr_args or "{input}"))
        self.lbl_executable.setText(
            f"deepSNR executable: {os.path.basename(deepsnr_path) if deepsnr_path else 'Not selected'}"
        )


class FlyZoneBrushWidget(QWidget):
    maskChanged = pyqtSignal()

    ZONE_COLORS = [
        QColor(255, 82, 82, 118),
        QColor(255, 193, 7, 118),
        QColor(76, 175, 80, 118),
        QColor(33, 150, 243, 118),
        QColor(156, 39, 176, 118),
        QColor(255, 152, 0, 118),
    ]

    def __init__(self, preview_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self.preview_bgr = normalize_to_uint8_bgr(preview_bgr)
        self.preview_rgb = cv2.cvtColor(self.preview_bgr, cv2.COLOR_BGR2RGB)
        self.preview_gray = cv2.cvtColor(self.preview_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.preview_hsv = cv2.cvtColor(self.preview_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        gx = cv2.Sobel(self.preview_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(self.preview_gray, cv2.CV_32F, 0, 1, ksize=3)
        self.preview_grad = cv2.magnitude(gx, gy)
        self.preview_lab = cv2.cvtColor(self.preview_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.zone_map = np.full(self.preview_gray.shape, -1, dtype=np.int16)
        self.locked_mask = np.zeros(self.preview_gray.shape, dtype=bool)

        self.active_zone = 0
        self.brush_radius = 24
        self.tolerance = 0.32
        self.aggressiveness = 0.45
        self.zoom_factor = 1.0
        self.zoom_min = 0.25
        self.zoom_max = 8.0
        self.ants_phase = 0.0
        self.painting = False
        self.erase_mode = False
        self.cursor_pos = None

        h, w = self.preview_gray.shape
        self._base_w = int(w)
        self._base_h = int(h)
        self._update_widget_size()
        self.setMouseTracking(True)
        self._ants_timer = QTimer(self)
        self._ants_timer.setInterval(80)
        self._ants_timer.timeout.connect(self._advance_ants)
        self._ants_timer.start()

    def set_active_zone(self, zone_index: int):
        self.active_zone = max(0, int(zone_index))

    def set_brush(self, radius: int, tolerance: float, aggressiveness: float = 0.45):
        self.brush_radius = max(2, int(radius))
        self.tolerance = float(np.clip(tolerance, 0.02, 0.95))
        self.aggressiveness = float(np.clip(aggressiveness, 0.0, 1.0))

    def clear_zones(self):
        self.zone_map.fill(-1)
        self.maskChanged.emit()
        self.update()

    def set_preview_image(self, preview_bgr: np.ndarray):
        incoming = normalize_to_uint8_bgr(preview_bgr)
        if incoming is None:
            return
        if incoming.shape[:2] != (self._base_h, self._base_w):
            incoming = cv2.resize(incoming, (self._base_w, self._base_h), interpolation=cv2.INTER_LINEAR)
        self.preview_bgr = incoming
        self.preview_rgb = cv2.cvtColor(self.preview_bgr, cv2.COLOR_BGR2RGB)
        self.preview_gray = cv2.cvtColor(self.preview_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.preview_hsv = cv2.cvtColor(self.preview_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        gx = cv2.Sobel(self.preview_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(self.preview_gray, cv2.CV_32F, 0, 1, ksize=3)
        self.preview_grad = cv2.magnitude(gx, gy)
        self.preview_lab = cv2.cvtColor(self.preview_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.update()

    def _update_widget_size(self):
        scaled_w = max(1, int(round(self._base_w * self.zoom_factor)))
        scaled_h = max(1, int(round(self._base_h * self.zoom_factor)))
        self.setMinimumSize(scaled_w, scaled_h)
        self.setMaximumSize(scaled_w, scaled_h)
        self.resize(scaled_w, scaled_h)

    def _advance_ants(self):
        self.ants_phase = (self.ants_phase + 1.0) % 8.0
        if np.any(self.zone_map >= 0):
            self.update()

    def get_zone_map(self) -> np.ndarray:
        return self.zone_map.copy()

    def _widget_to_image_coords(self, x: int, y: int):
        ix = int(round(float(x) / max(1e-6, self.zoom_factor)))
        iy = int(round(float(y) / max(1e-6, self.zoom_factor)))
        ix = int(np.clip(ix, 0, self._base_w - 1))
        iy = int(np.clip(iy, 0, self._base_h - 1))
        return ix, iy

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self.zoom_factor = float(np.clip(self.zoom_factor * factor, self.zoom_min, self.zoom_max))
        self._update_widget_size()
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return super().mousePressEvent(event)
        self.painting = True
        self.erase_mode = event.button() == Qt.RightButton
        ix, iy = self._widget_to_image_coords(event.pos().x(), event.pos().y())
        self._apply_stamp(ix, iy)
        event.accept()

    def mouseMoveEvent(self, event):
        self.cursor_pos = self._widget_to_image_coords(event.pos().x(), event.pos().y())
        if self.painting:
            self._apply_stamp(self.cursor_pos[0], self.cursor_pos[1])
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.cursor_pos = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self.painting = False
            self.erase_mode = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.scale(self.zoom_factor, self.zoom_factor)

        base = QImage(
            self.preview_rgb.data,
            self.preview_rgb.shape[1],
            self.preview_rgb.shape[0],
            self.preview_rgb.strides[0],
            QImage.Format_RGB888,
        )
        painter.drawImage(0, 0, base)

        zone_present = self.zone_map >= 0
        if np.any(zone_present):
            active_mask = zone_present.astype(np.uint8)
            edge_mask = cv2.morphologyEx(active_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
            edge_points = edge_mask > 0
            if np.any(edge_points):
                edge_luma = float(np.mean(self.preview_gray[edge_points]))
            else:
                edge_luma = 0.0

            ants_color = QColor(255, 255, 255) if edge_luma < 170.0 else QColor(0, 0, 0)
            ant_pen = QPen(ants_color, 1, Qt.CustomDashLine)
            ant_pen.setDashPattern([4.0, 4.0])
            ant_pen.setDashOffset(-float(self.ants_phase))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(ant_pen)

            unique_zones = np.unique(self.zone_map[zone_present])
            for zone_id in unique_zones:
                zone_u8 = (self.zone_map == int(zone_id)).astype(np.uint8)
                contours, _ = cv2.findContours(zone_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    if contour is None or len(contour) < 2:
                        continue
                    poly = QPolygon()
                    for point in contour[:, 0, :]:
                        poly.append(QPoint(int(point[0]), int(point[1])))
                    painter.drawPolyline(poly)

        if self.cursor_pos is not None:
            cx, cy = self.cursor_pos
            ring = QColor(255, 255, 255, 220)
            if self.erase_mode:
                ring = QColor(255, 120, 120, 230)
            painter.setPen(QPen(ring, 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            d = int(self.brush_radius * 2)
            painter.drawEllipse(int(cx - self.brush_radius), int(cy - self.brush_radius), d, d)

    def _apply_stamp(self, x: int, y: int):
        h, w = self.preview_gray.shape
        cx = int(np.clip(int(x), 0, w - 1))
        cy = int(np.clip(int(y), 0, h - 1))

        y0 = max(0, cy - self.brush_radius)
        y1 = min(h, cy + self.brush_radius + 1)
        x0 = max(0, cx - self.brush_radius)
        x1 = min(w, cx + self.brush_radius + 1)
        if y1 <= y0 or x1 <= x0:
            return

        yy, xx = np.ogrid[y0:y1, x0:x1]
        radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        circle = radial <= float(self.brush_radius)
        if not np.any(circle):
            return

        editable = np.logical_not(self.locked_mask[y0:y1, x0:x1])
        circle = np.logical_and(circle, editable)
        if not np.any(circle):
            return

        ref_luma = float(self.preview_gray[cy, cx])
        ref_grad = float(self.preview_grad[cy, cx])
        ref_lab = self.preview_lab[cy, cx, :]
        ref_hue = float(self.preview_hsv[cy, cx, 0])
        ref_sat = float(self.preview_hsv[cy, cx, 1])

        local_luma = self.preview_gray[y0:y1, x0:x1]
        local_grad = self.preview_grad[y0:y1, x0:x1]
        local_lab = self.preview_lab[y0:y1, x0:x1, :]
        local_hsv = self.preview_hsv[y0:y1, x0:x1, :]

        luma_delta = np.abs(local_luma - ref_luma) / 255.0
        grad_delta = np.abs(local_grad - ref_grad) / (np.max(local_grad) + 1e-6)
        lab_delta = np.linalg.norm(local_lab - ref_lab.reshape(1, 1, 3), axis=2) / 255.0
        hue_delta = np.abs(local_hsv[:, :, 0] - ref_hue)
        hue_delta = np.minimum(hue_delta, 180.0 - hue_delta) / 90.0
        sat_delta = np.abs(local_hsv[:, :, 1] - ref_sat) / 255.0
        local_score = 0.45 * luma_delta + 0.55 * grad_delta + 0.55 * lab_delta + 0.85 * hue_delta + 0.35 * sat_delta

        threshold = float(np.clip(self.tolerance + (0.30 * (1.0 - self.aggressiveness)), 0.02, 1.25))
        candidate = np.logical_and(circle, local_score <= threshold)
        seed_x = int(cx - x0)
        seed_y = int(cy - y0)

        smart_mask = candidate
        if 0 <= seed_x < candidate.shape[1] and 0 <= seed_y < candidate.shape[0] and candidate[seed_y, seed_x]:
            candidate_u8 = candidate.astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(candidate_u8, connectivity=8)
            if num_labels > 1:
                seed_label = int(labels[seed_y, seed_x])
                if seed_label > 0:
                    smart_mask = labels == seed_label

        edge_limit = ref_grad + (35.0 + 120.0 * threshold) + (110.0 * (1.0 - self.aggressiveness))
        edge_guard = local_grad <= edge_limit
        smart_mask = np.logical_and(smart_mask, edge_guard)

        if not np.any(smart_mask):
            smart_mask = circle

        sub = self.zone_map[y0:y1, x0:x1]
        sub[smart_mask] = -1 if self.erase_mode else int(self.active_zone)
        self.zone_map[y0:y1, x0:x1] = sub
        self.maskChanged.emit()
        self.update()


class HorizontalTextTabBar(QTabBar):
    def tabSizeHint(self, index):
        _ = index
        return QSize(220, 42)

    def sizeHint(self):
        base = super().sizeHint()
        min_height = max(base.height(), self.count() * 46 + 8)
        return QSize(max(base.width(), 228), min_height)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        for idx in range(self.count()):
            rect = self.tabRect(idx).adjusted(4, 2, -4, -2)
            selected = idx == self.currentIndex()

            bg = QColor(44, 64, 82) if selected else QColor(28, 36, 44)
            border = QColor(86, 126, 164) if selected else QColor(58, 74, 90)
            text = QColor(245, 248, 252) if selected else QColor(206, 216, 226)

            painter.setPen(QPen(border, 1))
            painter.setBrush(bg)
            painter.drawRoundedRect(rect, 7, 7)

            painter.setPen(text)
            painter.drawText(rect, Qt.AlignCenter, self.tabText(idx))


class Fly3DDialog(QDialog):
    def __init__(self, image: np.ndarray, default_path: str = "", parent=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("3D FLY Filter")
        self.setMinimumSize(980, 760)
        self.source_image = normalize_to_uint8_bgr(image)

        h, w = self.source_image.shape[:2]
        preview_scale = min(1.0, 900.0 / max(1.0, float(max(h, w))))
        preview_w = max(1, int(round(w * preview_scale)))
        preview_h = max(1, int(round(h * preview_scale)))
        self.preview_image = cv2.resize(self.source_image, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        self.cut_items = []
        self.edge_blur_by_layer = {}
        self.edge_blur_brush_by_layer = {}
        self.clip_timeline_entries = []
        self.clip_timeline_index = 0
        self._edge_blur_brush_painting = False
        self._edge_blur_brush_value = 255
        self._updating_stage_gate = False

        root = QVBoxLayout(self)
        apply_standard_layout_margins(root)

        self.stage_names = [
            "1 usun gwiazdy",
            "2 zaznacz sekcje",
            "3 wygladzanie krawedzi",
            "4 laczenie w klip",
            "5 dodaj gwiazdy",
            "6 definiowanie ruchu i pozycji",
            "7 dodaj muzyke",
        ]

        top_container = QWidget()
        top_container_layout = QVBoxLayout(top_container)
        top_container_layout.setContentsMargins(0, 0, 0, 0)

        stage_shell = QHBoxLayout()
        stage_shell.setContentsMargins(0, 0, 0, 0)
        stage_shell.setSpacing(0)
        self.stage_nav = QListWidget()
        self.stage_nav.setMinimumWidth(0)
        self.stage_nav.setMaximumWidth(420)
        self.stage_nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stage_nav.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.stage_nav.setSpacing(4)
        self.stage_nav.setStyleSheet(
            "QListWidget { border: 1px solid #3b4c5d; padding: 4px; }"
            "QListWidget::item { min-height: 40px; border: 1px solid #3a4d61; border-radius: 6px; padding: 4px 8px; }"
            "QListWidget::item:selected { background: #2b3f55; color: #f5f8fc; border-color: #6f96bf; }"
            "QListWidget::item:!selected { background: #1c242c; color: #c9d4df; }"
            "QListWidget::item:disabled { color: #6a737d; border-color: #2c333a; background: #161b20; }"
        )
        for name in self.stage_names:
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(236, 42))
            self.stage_nav.addItem(item)

        self.stage_stack = QStackedWidget()
        self.stage_stack.setStyleSheet("QStackedWidget { border: 1px solid #3b4c5d; }")

        self.left_splitter = QSplitter(Qt.Horizontal)
        self.left_splitter.addWidget(self.stage_nav)
        self.left_splitter.addWidget(self.stage_stack)
        self.left_splitter.setChildrenCollapsible(True)
        self.left_splitter.setStretchFactor(0, 0)
        self.left_splitter.setStretchFactor(1, 1)
        self.left_splitter.setCollapsible(0, True)
        self.left_splitter.setCollapsible(1, True)
        self.left_splitter.setSizes([260, 900])
        stage_shell.addWidget(self.left_splitter)
        top_container_layout.addLayout(stage_shell)

        self.bottom_panel = QWidget()
        bottom_panel_layout = QVBoxLayout(self.bottom_panel)
        bottom_panel_layout.setContentsMargins(6, 6, 6, 6)
        bottom_panel_layout.setSpacing(6)
        self.stage_nav.currentRowChanged.connect(self._on_stage_tab_changed)

        # Etap 1: usun gwiazdy
        stage0 = QWidget()
        stage0_layout = QVBoxLayout(stage0)
        stage0_layout.setContentsMargins(10, 10, 10, 10)
        stage0_layout.setSpacing(8)

        self.starless_preview_label = QLabel()
        self.starless_preview_label.setAlignment(Qt.AlignCenter)
        self.starless_preview_label.setMinimumHeight(280)
        self.starless_preview_label.setStyleSheet("border: 1px solid #3b4c5d; background: #12171c;")
        stage0_layout.addWidget(self.starless_preview_label, 1)

        controls_starless = QHBoxLayout()
        self.btn_remove_stars = QPushButton("Usun gwiazdy")
        self.btn_remove_stars.clicked.connect(self._remove_stars_now)
        controls_starless.addWidget(self.btn_remove_stars)
        controls_starless.addStretch(1)
        stage0_layout.addLayout(controls_starless)

        advanced_row = QHBoxLayout()
        self.chk_starless_advanced = QCheckBox("Zaawansowane ustawienia")
        self.chk_starless_advanced.toggled.connect(self._on_starless_advanced_toggled)
        advanced_row.addWidget(self.chk_starless_advanced)
        self.lbl_starless_stride = QLabel("Stride:")
        self.spin_starless_stride = QSpinBox()
        self.spin_starless_stride.setRange(2, 512)
        self.spin_starless_stride.setSingleStep(2)
        parent_stride = int(getattr(self.parent(), "starnet_stride", 16) or 16)
        if parent_stride < 2:
            parent_stride = 2
        if parent_stride > 512:
            parent_stride = 512
        if parent_stride % 2 != 0:
            parent_stride += 1
            if parent_stride > 512:
                parent_stride = 512
        self.spin_starless_stride.setValue(parent_stride)
        self.spin_starless_stride.valueChanged.connect(self._on_starless_stride_changed)
        advanced_row.addWidget(self.lbl_starless_stride)
        advanced_row.addWidget(self.spin_starless_stride)
        advanced_row.addStretch(1)
        stage0_layout.addLayout(advanced_row)
        self._on_starless_advanced_toggled(False)

        self.stage_stack.addWidget(stage0)

        # Etap 2: zaznacz sekcje
        stage1 = QWidget()
        stage1_layout = QVBoxLayout(stage1)
        stage1_layout.setContentsMargins(10, 10, 10, 10)
        stage1_layout.setSpacing(8)
        info_select = QLabel(
            "LPM: malowanie stref. PPM: gumka. Kolo myszy: zoom podgladu. Pedzel bierze pod uwage jasnosc, kontrast i barwe."
        )
        info_select.setWordWrap(True)
        stage1_layout.addWidget(info_select)

        self.zone_widget = FlyZoneBrushWidget(self.preview_image)
        self.zone_widget.maskChanged.connect(self._update_stage_gate)
        self.current_zone_index = 0
        self.zone_widget.set_active_zone(self.current_zone_index)
        self.zone_scroll = QScrollArea()
        self.zone_scroll.setWidgetResizable(False)
        self.zone_scroll.setWidget(self.zone_widget)
        stage1_layout.addWidget(self.zone_scroll, 1)

        zone_row = QHBoxLayout()
        zone_row.addStretch(1)
        self.combo_zone = None

        zone_buttons_col = QVBoxLayout()
        zone_buttons_col.setSpacing(6)
        self.btn_clear_zones = QPushButton("Wyczysc")
        self.btn_clear_zones.clicked.connect(self.zone_widget.clear_zones)
        zone_buttons_col.addWidget(self.btn_clear_zones)
        self.btn_cut = QPushButton("Wytnij (Ctrl+X)")
        self.btn_cut.clicked.connect(self._perform_cut_selection)
        zone_buttons_col.addWidget(self.btn_cut)
        self.btn_load_cut_layers = QPushButton("Wczytaj warstwy")
        self.btn_load_cut_layers.clicked.connect(self._import_cut_layers)
        zone_buttons_col.addWidget(self.btn_load_cut_layers)
        zone_buttons_col.addStretch(1)
        zone_row.addLayout(zone_buttons_col)
        stage1_layout.addLayout(zone_row)

        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Rozmiar:"))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(2, 120)
        self.spin_brush.setValue(24)
        self.spin_brush.valueChanged.connect(self._sync_brush)
        brush_row.addWidget(self.spin_brush)
        brush_row.addWidget(QLabel("Czulosc:"))
        self.spin_tolerance = QDoubleSpinBox()
        self.spin_tolerance.setRange(0.02, 0.95)
        self.spin_tolerance.setSingleStep(0.01)
        self.spin_tolerance.setValue(0.32)
        self.spin_tolerance.valueChanged.connect(self._sync_brush)
        brush_row.addWidget(self.spin_tolerance)
        brush_row.addWidget(QLabel("Agresywnosc:"))
        self.spin_aggressiveness = QDoubleSpinBox()
        self.spin_aggressiveness.setRange(0.0, 1.0)
        self.spin_aggressiveness.setSingleStep(0.05)
        self.spin_aggressiveness.setValue(0.45)
        self.spin_aggressiveness.valueChanged.connect(self._sync_brush)
        brush_row.addWidget(self.spin_aggressiveness)
        stage1_layout.addLayout(brush_row)

        hole_row = QHBoxLayout()
        hole_row.addWidget(QLabel("Akceptowalny rozmiar dziury (px):"))
        self.slider_hole_size_px = ZoomRangeSlider()
        self.slider_hole_size_px.set_bounds(0.0, 256.0)
        self.slider_hole_size_px.set_min_span(0.0)
        self.slider_hole_size_px.set_values(0.0, 24.0, emit_signal=False)
        self.slider_hole_size_px.setToolTip("Dziury w zaznaczeniu z tego zakresu pikseli zostana automatycznie wypelnione.")
        hole_row.addWidget(self.slider_hole_size_px, 1)
        self.lbl_hole_size_px = QLabel()
        self.lbl_hole_size_px.setMinimumWidth(84)
        self.lbl_hole_size_px.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hole_row.addWidget(self.lbl_hole_size_px)
        self.slider_hole_size_px.rangeChanged.connect(self._on_hole_size_range_changed)
        self._on_hole_size_range_changed(*self.slider_hole_size_px.values())
        stage1_layout.addLayout(hole_row)

        self.stage_stack.addWidget(stage1)

        # Etap 3: wygladzanie krawedzi
        stage2 = QWidget()
        stage2_layout = QVBoxLayout(stage2)
        stage2_layout.setContentsMargins(10, 10, 10, 10)
        stage2_layout.setSpacing(10)
        stage2_layout.addWidget(QLabel("Wygładzanie granic miedzy strefami (feather/blur):"))
        row_edge = QHBoxLayout()
        row_edge.addWidget(QLabel("Blur strefy:"))
        self.sld_edge_blur = QSlider(Qt.Horizontal)
        self.sld_edge_blur.setRange(0, 200)
        self.sld_edge_blur.setValue(22)
        self.sld_edge_blur.valueChanged.connect(self._on_edge_blur_slider_changed)
        row_edge.addWidget(self.sld_edge_blur, 1)
        self.lbl_edge_blur_value = QLabel("2.2")
        self.lbl_edge_blur_value.setMinimumWidth(44)
        self.lbl_edge_blur_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row_edge.addWidget(self.lbl_edge_blur_value)
        self.btn_apply_edge_blur = QPushButton("Zastosuj")
        self.btn_apply_edge_blur.clicked.connect(self._apply_selected_layer_blur)
        row_edge.addWidget(self.btn_apply_edge_blur)
        self.btn_apply_edge_blur_all = QPushButton("Zastosuj dla wszystkich")
        self.btn_apply_edge_blur_all.clicked.connect(self._apply_all_layers_blur)
        row_edge.addWidget(self.btn_apply_edge_blur_all)
        stage2_layout.addLayout(row_edge)

        auto_row = QHBoxLayout()
        auto_row.addWidget(QLabel("Sila auto:"))
        self.sld_auto_edge_blur = QSlider(Qt.Horizontal)
        self.sld_auto_edge_blur.setRange(0, 100)
        self.sld_auto_edge_blur.setValue(62)
        auto_row.addWidget(self.sld_auto_edge_blur, 1)
        self.lbl_auto_edge_blur_value = QLabel("62%")
        self.lbl_auto_edge_blur_value.setMinimumWidth(46)
        self.lbl_auto_edge_blur_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        auto_row.addWidget(self.lbl_auto_edge_blur_value)
        self.sld_auto_edge_blur.valueChanged.connect(lambda value: self.lbl_auto_edge_blur_value.setText(f"{int(value)}%"))
        self.btn_auto_edge_blur = QPushButton("Auto korekta blur")
        self.btn_auto_edge_blur.clicked.connect(self._run_auto_edge_blur_correction)
        auto_row.addWidget(self.btn_auto_edge_blur)
        stage2_layout.addLayout(auto_row)

        stage2_layout.addWidget(QLabel("Podglad sekcji (tlo + warstwy):"))
        self.clip_timeline_list = QListWidget()
        self.clip_timeline_list.setMinimumHeight(130)
        self.clip_timeline_list.currentRowChanged.connect(self._on_clip_timeline_row_changed)
        stage2_layout.addWidget(self.clip_timeline_list)

        self.clip_preview_label = QLabel("Podglad wybranej sekcji.")
        self.clip_preview_label.setAlignment(Qt.AlignCenter)
        self.clip_preview_label.setMinimumHeight(240)
        self.clip_preview_label.setStyleSheet("border: 1px solid #3b4c5d; background: #12171c;")
        self.clip_preview_label.setMouseTracking(True)
        self.clip_preview_label.mousePressEvent = self._on_clip_preview_mouse_press
        self.clip_preview_label.mouseMoveEvent = self._on_clip_preview_mouse_move
        self.clip_preview_label.mouseReleaseEvent = self._on_clip_preview_mouse_release
        stage2_layout.addWidget(self.clip_preview_label, 1)

        clip_nav_row = QHBoxLayout()
        self.btn_clip_prev = QPushButton("Poprzednia")
        self.btn_clip_prev.clicked.connect(self._show_prev_clip_layer)
        self.btn_clip_next = QPushButton("Nastepna")
        self.btn_clip_next.clicked.connect(self._show_next_clip_layer)
        clip_nav_row.addWidget(self.btn_clip_prev)
        clip_nav_row.addWidget(self.btn_clip_next)
        clip_nav_row.addStretch(1)
        stage2_layout.addLayout(clip_nav_row)

        brush_row = QHBoxLayout()
        self.chk_edge_blur_brush = QCheckBox("Pedzel korekty bluru")
        self.chk_edge_blur_brush.toggled.connect(self._on_edge_blur_brush_toggled)
        brush_row.addWidget(self.chk_edge_blur_brush)
        brush_row.addWidget(QLabel("Rozmiar:"))
        self.spin_edge_blur_brush_size = QSpinBox()
        self.spin_edge_blur_brush_size.setRange(2, 140)
        self.spin_edge_blur_brush_size.setValue(24)
        brush_row.addWidget(self.spin_edge_blur_brush_size)
        self.btn_edge_blur_brush_clear = QPushButton("Wyczysc pedzel")
        self.btn_edge_blur_brush_clear.clicked.connect(self._clear_edge_blur_brush_for_selected_layer)
        brush_row.addWidget(self.btn_edge_blur_brush_clear)
        brush_row.addWidget(QLabel("LPM: dodaj blur, PPM: usun."))
        brush_row.addStretch(1)
        stage2_layout.addLayout(brush_row)
        stage2_layout.addStretch(1)
        self.stage_stack.addWidget(stage2)
        self._on_edge_blur_slider_changed(self.sld_edge_blur.value())

        # Etap 4: laczenie w klip
        stage3 = QWidget()
        stage3_layout = QVBoxLayout(stage3)
        stage3_layout.setContentsMargins(10, 10, 10, 10)
        stage3_layout.setSpacing(10)
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(1.0, 20.0)
        self.spin_duration.setValue(6.0)
        self.spin_duration.setSingleStep(0.5)
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(12, 60)
        self.spin_fps.setValue(24)
        row_timing = QHBoxLayout()
        row_timing.addWidget(QLabel("Czas (s):"))
        row_timing.addWidget(self.spin_duration)
        row_timing.addWidget(QLabel("FPS:"))
        row_timing.addWidget(self.spin_fps)
        stage3_layout.addLayout(row_timing)
        stage3_layout.addStretch(1)
        self.stage_stack.addWidget(stage3)

        # Etap 5: dodaj gwiazdy
        stage4 = QWidget()
        stage4_layout = QVBoxLayout(stage4)
        stage4_layout.setContentsMargins(10, 10, 10, 10)
        stage4_layout.setSpacing(10)

        stars_info = QLabel(
            "Symulacja lotu gwiazd: jasniejsze punkty traktowane sa jako blizsze i poruszaja sie szybciej."
        )
        stars_info.setWordWrap(True)
        stage4_layout.addWidget(stars_info)

        self.chk_add_stars = QCheckBox("Dodaj gwiazdy do klipu")
        self.chk_add_stars.setChecked(True)
        stage4_layout.addWidget(self.chk_add_stars)

        stars_mask_row = QHBoxLayout()
        stars_mask_row.addWidget(QLabel("Plik star mask:"))
        self.edit_stars_mask_path = QLineEdit("")
        self.edit_stars_mask_path.setPlaceholderText("Wybierz obraz maski gwiazd (StarMask)")
        self.btn_stars_mask = QPushButton("Plik...")
        self.btn_stars_mask.clicked.connect(self._pick_stars_mask_path)
        stars_mask_row.addWidget(self.edit_stars_mask_path, 1)
        stars_mask_row.addWidget(self.btn_stars_mask)
        stage4_layout.addLayout(stars_mask_row)

        stars_count_row = QHBoxLayout()
        stars_count_row.addWidget(QLabel("Liczba gwiazd:"))
        self.spin_stars_count = QSpinBox()
        self.spin_stars_count.setRange(80, 3000)
        self.spin_stars_count.setSingleStep(20)
        self.spin_stars_count.setValue(900)
        stars_count_row.addWidget(self.spin_stars_count)
        stars_count_row.addStretch(1)
        stage4_layout.addLayout(stars_count_row)

        stars_speed_row = QHBoxLayout()
        stars_speed_row.addWidget(QLabel("Predkosc gwiazd:"))
        self.spin_stars_speed = QDoubleSpinBox()
        self.spin_stars_speed.setRange(0.2, 3.0)
        self.spin_stars_speed.setSingleStep(0.1)
        self.spin_stars_speed.setValue(1.2)
        stars_speed_row.addWidget(self.spin_stars_speed)
        stars_speed_row.addStretch(1)
        stage4_layout.addLayout(stars_speed_row)

        stars_near_row = QHBoxLayout()
        stars_near_row.addWidget(QLabel("Udzial jasnych (bliskich):"))
        self.spin_stars_near_ratio = QDoubleSpinBox()
        self.spin_stars_near_ratio.setRange(0.05, 0.80)
        self.spin_stars_near_ratio.setSingleStep(0.05)
        self.spin_stars_near_ratio.setValue(0.28)
        self.spin_stars_near_ratio.setDecimals(2)
        stars_near_row.addWidget(self.spin_stars_near_ratio)
        stars_near_row.addStretch(1)
        stage4_layout.addLayout(stars_near_row)

        stars_size_row = QHBoxLayout()
        stars_size_row.addWidget(QLabel("Rozmiar gwiazd:"))
        self.spin_stars_size = QDoubleSpinBox()
        self.spin_stars_size.setRange(0.6, 2.6)
        self.spin_stars_size.setSingleStep(0.1)
        self.spin_stars_size.setValue(1.0)
        self.spin_stars_size.setDecimals(1)
        stars_size_row.addWidget(self.spin_stars_size)
        stars_size_row.addStretch(1)
        stage4_layout.addLayout(stars_size_row)

        stage4_layout.addStretch(1)
        self.stage_stack.addWidget(stage4)

        # Etap 6: definiowanie ruchu i pozycji
        stage5 = QWidget()
        stage5_layout = QVBoxLayout(stage5)
        stage5_layout.setContentsMargins(10, 10, 10, 10)
        stage5_layout.setSpacing(8)
        motion_zoom_row = QHBoxLayout()
        motion_zoom_row.addWidget(QLabel("Main zoom speed:"))
        self.spin_main_zoom_speed = QDoubleSpinBox()
        self.spin_main_zoom_speed.setRange(-1.0, 1.0)
        self.spin_main_zoom_speed.setSingleStep(0.01)
        self.spin_main_zoom_speed.setValue(0.03)
        motion_zoom_row.addWidget(self.spin_main_zoom_speed)
        motion_zoom_row.addStretch(1)
        stage5_layout.addLayout(motion_zoom_row)

        self.motion_preview_label = QLabel("Podglad ruchu warstw.")
        self.motion_preview_label.setAlignment(Qt.AlignCenter)
        self.motion_preview_label.setMinimumHeight(320)
        self.motion_preview_label.setStyleSheet("border: 1px solid #3b4c5d; background: #12171c;")
        self.motion_preview_label.setMouseTracking(True)
        self.motion_preview_label.mousePressEvent = self._on_motion_preview_mouse_press
        self.motion_preview_label.mouseMoveEvent = self._on_motion_preview_mouse_move
        self.motion_preview_label.mouseReleaseEvent = self._on_motion_preview_mouse_release
        self.motion_preview_label.wheelEvent = self._on_motion_preview_wheel
        stage5_layout.addWidget(self.motion_preview_label, 1)

        motion_props_row = QHBoxLayout()
        motion_props_row.addWidget(QLabel("Warstwa:"))
        self.combo_motion_layer_props = QComboBox()
        self.combo_motion_layer_props.currentIndexChanged.connect(self._on_motion_props_layer_changed)
        motion_props_row.addWidget(self.combo_motion_layer_props)
        motion_props_row.addWidget(QLabel("Kierunek:"))
        self.combo_motion_direction = QComboBox()
        self.combo_motion_direction.addItem("Strzalka 2D", "vector")
        self.combo_motion_direction.addItem("Do widza (zoom)", "viewer")
        self.combo_motion_direction.currentIndexChanged.connect(self._on_motion_props_direction_changed)
        motion_props_row.addWidget(self.combo_motion_direction)
        motion_props_row.addWidget(QLabel("Predkosc:"))
        self.spin_motion_speed_props = QDoubleSpinBox()
        self.spin_motion_speed_props.setRange(0.0, 400.0)
        self.spin_motion_speed_props.setSingleStep(2.0)
        self.spin_motion_speed_props.valueChanged.connect(self._on_motion_props_speed_changed)
        motion_props_row.addWidget(self.spin_motion_speed_props)
        motion_props_row.addWidget(QLabel("Predkosc zoomu:"))
        self.spin_motion_zoom_speed = QDoubleSpinBox()
        self.spin_motion_zoom_speed.setRange(-1.0, 1.0)
        self.spin_motion_zoom_speed.setSingleStep(0.01)
        self.spin_motion_zoom_speed.valueChanged.connect(self._on_motion_props_zoom_changed)
        motion_props_row.addWidget(self.spin_motion_zoom_speed)
        self.btn_motion_preview_clip = QPushButton("Podglad klipu")
        self.btn_motion_preview_clip.clicked.connect(self._toggle_motion_clip_preview)
        motion_props_row.addWidget(self.btn_motion_preview_clip)
        motion_props_row.addStretch(1)
        stage5_layout.addLayout(motion_props_row)

        self.motion_settings = {}
        self.motion_zone_centers = {}
        self.selected_motion_zone = None
        self._motion_dragging = False
        self._cut_layer_preview_cache = {}
        self._motion_clip_frames = []
        self._motion_clip_frame_index = 0
        self._motion_clip_timer = QTimer(self)
        self._motion_clip_timer.timeout.connect(self._advance_motion_clip_preview_frame)
        stage5_layout.addStretch(1)
        self.stage_stack.addWidget(stage5)

        # Etap 7: zapis
        stage6 = QWidget()
        stage6_layout = QVBoxLayout(stage6)
        stage6_layout.setContentsMargins(10, 10, 10, 10)
        stage6_layout.setSpacing(8)

        stage6_layout.addWidget(QLabel("Dodaj muzyke na timeline z bazy lub z dysku:"))

        self.music_db_list = QListWidget()
        self.music_db_list.setMinimumHeight(140)
        stage6_layout.addWidget(self.music_db_list)

        music_controls = QHBoxLayout()
        self.btn_music_refresh = QPushButton("Odswiez baze")
        self.btn_music_refresh.clicked.connect(self._reload_music_database)
        self.btn_music_import = QPushButton("Import z dysku")
        self.btn_music_import.clicked.connect(self._import_music_from_disk)
        self.btn_music_add_db = QPushButton("Dodaj z bazy")
        self.btn_music_add_db.clicked.connect(self._add_selected_db_music_to_timeline)
        music_controls.addWidget(self.btn_music_refresh)
        music_controls.addWidget(self.btn_music_import)
        music_controls.addWidget(self.btn_music_add_db)
        stage6_layout.addLayout(music_controls)

        timeline_cfg = QHBoxLayout()
        timeline_cfg.addWidget(QLabel("Start (s):"))
        self.spin_music_start = QDoubleSpinBox()
        self.spin_music_start.setRange(0.0, 600.0)
        self.spin_music_start.setSingleStep(0.2)
        self.spin_music_start.setValue(0.0)
        timeline_cfg.addWidget(self.spin_music_start)
        timeline_cfg.addWidget(QLabel("Glosnosc:"))
        self.spin_music_volume = QDoubleSpinBox()
        self.spin_music_volume.setRange(0.0, 3.0)
        self.spin_music_volume.setSingleStep(0.1)
        self.spin_music_volume.setValue(1.0)
        timeline_cfg.addWidget(self.spin_music_volume)
        stage6_layout.addLayout(timeline_cfg)

        stage6_layout.addWidget(QLabel("Timeline muzyki:"))
        self.music_timeline_list = QListWidget()
        self.music_timeline_list.setMinimumHeight(120)
        stage6_layout.addWidget(self.music_timeline_list)

        timeline_buttons = QHBoxLayout()
        self.btn_music_remove = QPushButton("Usun zaznaczony")
        self.btn_music_remove.clicked.connect(self._remove_selected_timeline_clip)
        self.btn_music_clear = QPushButton("Wyczysc timeline")
        self.btn_music_clear.clicked.connect(self._clear_music_timeline)
        timeline_buttons.addWidget(self.btn_music_remove)
        timeline_buttons.addWidget(self.btn_music_clear)
        stage6_layout.addLayout(timeline_buttons)

        output_row = QHBoxLayout()
        self.edit_output = QLineEdit(default_path)
        self.btn_output = QPushButton("Plik...")
        self.btn_output.clicked.connect(self._pick_output_path)
        output_row.addWidget(self.edit_output, 1)
        output_row.addWidget(self.btn_output)
        stage6_layout.addLayout(output_row)
        stage6_layout.addStretch(1)
        self.stage_stack.addWidget(stage6)

        self.music_database_entries = []
        self.audio_timeline_entries = []
        self._reload_music_database()

        self.action_cut_selection = QAction("Cut selection", self)
        self.action_cut_selection.setShortcut("Ctrl+X")
        self.action_cut_selection.setShortcutContext(Qt.WindowShortcut)
        self.action_cut_selection.triggered.connect(self._perform_cut_selection)
        self.addAction(self.action_cut_selection)

        self.cuts_label = QLabel("Wycięte sekcje (PNG):")
        bottom_panel_layout.addWidget(self.cuts_label)
        self.cut_scroll = QScrollArea()
        self.cut_scroll.setWidgetResizable(True)
        self.cut_scroll.setMinimumHeight(140)
        self.cut_container = QWidget()
        self.cut_layout = QHBoxLayout(self.cut_container)
        self.cut_layout.setContentsMargins(4, 4, 4, 4)
        self.cut_layout.setSpacing(8)
        self.cut_layout.addStretch(1)
        self.cut_scroll.setWidget(self.cut_container)
        bottom_panel_layout.addWidget(self.cut_scroll, 1)

        buttons = QVBoxLayout()
        self.btn_run = QPushButton("Renderuj 3D FLY")
        self.btn_run.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(self.btn_run)
        buttons.addWidget(self.btn_cancel)
        buttons.addStretch(1)
        bottom_panel_layout.addLayout(buttons)

        self.bottom_splitter = QSplitter(Qt.Vertical)
        self.bottom_splitter.addWidget(top_container)
        self.bottom_splitter.addWidget(self.bottom_panel)
        self.bottom_splitter.setChildrenCollapsible(True)
        self.bottom_splitter.setStretchFactor(0, 1)
        self.bottom_splitter.setStretchFactor(1, 0)
        self.bottom_splitter.setCollapsible(0, True)
        self.bottom_splitter.setCollapsible(1, True)
        self.bottom_splitter.setSizes([760, 120])
        root.addWidget(self.bottom_splitter, 1)

        self._sync_brush()
        self._refresh_motion_layers(keep_selection=False)
        self._updating_stage_gate = False
        self.stage_nav.setCurrentRow(0)
        self.stage_stack.setCurrentIndex(0)
        self._update_stage_gate()
        self._update_run_button_visibility(0)
        self._update_cut_sections_visibility(0)
        self._update_starless_preview()
        self._update_edge_preview()
        self._refresh_clip_timeline_layers(keep_index=False)

    def _is_stage1_complete(self) -> bool:
        if len(getattr(self, "cut_items", [])) > 0:
            return True
        unlocked = np.logical_not(self.zone_widget.locked_mask)
        if not np.any(unlocked):
            return True
        assigned = self.zone_widget.zone_map >= 0
        return bool(np.all(np.logical_or(np.logical_not(unlocked), assigned)))

    def _update_stage_gate(self):
        complete = self._is_stage1_complete()
        for idx in range(2, self.stage_nav.count()):
            item = self.stage_nav.item(idx)
            if item is None:
                continue
            flags = item.flags()
            if complete:
                item.setFlags(flags | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            else:
                item.setFlags((flags | Qt.ItemIsSelectable) & ~Qt.ItemIsEnabled)
        if not complete and self.stage_nav.currentRow() > 1:
            self._updating_stage_gate = True
            self.stage_nav.setCurrentRow(1)
            self.stage_stack.setCurrentIndex(1)
            self._updating_stage_gate = False
            self._update_run_button_visibility(1)
            self._update_cut_sections_visibility(1)
        self._update_edge_preview()

    def _compute_zone_edge_mask(self) -> np.ndarray:
        zone_map = self.zone_widget.get_zone_map()
        if not isinstance(zone_map, np.ndarray) or zone_map.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        valid = zone_map >= 0
        h, w = zone_map.shape
        edges_u8 = np.zeros((h, w), dtype=np.uint8)

        unique_zones = np.unique(zone_map[valid]) if np.any(valid) else np.array([], dtype=np.int16)
        for zone_id in unique_zones:
            zone_mask = (zone_map == int(zone_id)).astype(np.uint8)
            if int(np.count_nonzero(zone_mask)) == 0:
                continue
            contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                cv2.drawContours(edges_u8, contours, -1, 255, 1)

        if np.any(self.zone_widget.locked_mask):
            locked_u8 = self.zone_widget.locked_mask.astype(np.uint8)
            locked_contours, _ = cv2.findContours(locked_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if locked_contours:
                cv2.drawContours(edges_u8, locked_contours, -1, 255, 1)

        return edges_u8

    def _update_edge_preview(self):
        if not hasattr(self, "edge_preview_label"):
            return
        if self.preview_image is None:
            self.edge_preview_label.setText("Brak obrazu do podgladu.")
            self.edge_preview_label.setPixmap(QPixmap())
            return

        edge_mask = self._compute_zone_edge_mask()
        if edge_mask.size == 0 or int(np.count_nonzero(edge_mask)) == 0:
            fallback = np_to_qpixmap(self.preview_image)
            target_size = self.edge_preview_label.size()
            if target_size.width() > 8 and target_size.height() > 8:
                fallback = fallback.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.edge_preview_label.setText("Brak granic miedzy strefami - dodaj co najmniej dwie strefy lub wyciecie.")
            self.edge_preview_label.setPixmap(fallback)
            return

        sigma = self._edge_blur_value()
        edge_f = edge_mask.astype(np.float32) / 255.0
        if sigma > 0.0:
            edge_f = cv2.GaussianBlur(edge_f, (0, 0), sigma)

        edge_f = np.clip(edge_f, 0.0, 1.0)
        alpha = np.clip(edge_f * 2.4, 0.0, 0.9)
        alpha3 = alpha[:, :, np.newaxis]

        base = self.preview_image.astype(np.float32)
        white = np.full_like(base, 255.0)
        preview = base * (1.0 - alpha3) + white * alpha3
        preview = np.clip(preview, 0, 255).astype(np.uint8)

        pix = np_to_qpixmap(preview)
        target_size = self.edge_preview_label.size()
        if target_size.width() > 8 and target_size.height() > 8:
            pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.edge_preview_label.setText("")
        self.edge_preview_label.setPixmap(pix)

    def _edge_blur_value(self) -> float:
        if hasattr(self, "sld_edge_blur"):
            return float(self.sld_edge_blur.value()) / 10.0
        if hasattr(self, "spin_edge_blur"):
            return float(self.spin_edge_blur.value())
        return 0.0

    def _on_edge_blur_slider_changed(self, value: int):
        if hasattr(self, "lbl_edge_blur_value"):
            self.lbl_edge_blur_value.setText(f"{float(value) / 10.0:.1f}")
        self._update_edge_preview()

    def _selected_clip_layer_key(self):
        idx = int(getattr(self, "clip_timeline_index", 0))
        if idx < 0 or idx >= len(getattr(self, "clip_timeline_entries", [])):
            return None
        entry = self.clip_timeline_entries[idx]
        return entry.get("layer_key")

    def _iter_clip_layer_entries(self):
        for entry in getattr(self, "clip_timeline_entries", []):
            layer_key = entry.get("layer_key")
            path = str(entry.get("path") or "")
            if not layer_key or not path:
                continue
            if not os.path.exists(path):
                continue
            yield str(layer_key), path

    def _build_auto_edge_blur_mask_for_layer(self, layer_path: str, strength: float) -> np.ndarray | None:
        if not layer_path or not os.path.exists(layer_path):
            return None
        cut = cv2.imread(layer_path, cv2.IMREAD_UNCHANGED)
        if cut is None or cut.ndim < 3:
            return None

        if cut.shape[2] >= 4:
            rgb = cut[:, :, :3]
            alpha_u8 = cut[:, :, 3]
        else:
            rgb = cut[:, :, :3]
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            alpha_u8 = np.where(gray < 250, 255, 0).astype(np.uint8)

        mask = alpha_u8 > 0
        if int(np.count_nonzero(mask)) == 0:
            return None

        mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
        ring_outer = cv2.dilate(mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
        ring_inner = cv2.erode(mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
        edge_ring = np.logical_xor(ring_outer > 0, ring_inner > 0)
        if int(np.count_nonzero(edge_ring)) == 0:
            return None

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mask_f = mask.astype(np.float32)
        inv_f = 1.0 - mask_f
        blur_kernel_sigma = 2.0

        inside_num = cv2.GaussianBlur(gray * mask_f, (0, 0), blur_kernel_sigma)
        inside_den = cv2.GaussianBlur(mask_f, (0, 0), blur_kernel_sigma)
        outside_num = cv2.GaussianBlur(gray * inv_f, (0, 0), blur_kernel_sigma)
        outside_den = cv2.GaussianBlur(inv_f, (0, 0), blur_kernel_sigma)

        inside_mean = inside_num / np.maximum(inside_den, 1e-5)
        outside_mean = outside_num / np.maximum(outside_den, 1e-5)
        contrast = np.abs(inside_mean - outside_mean)
        contrast = np.clip(contrast / 255.0, 0.0, 1.0)

        edge_values = contrast[edge_ring]
        if edge_values.size == 0:
            return None

        strength01 = float(np.clip(strength, 0.0, 1.0))
        quantile = float(np.clip(0.86 - strength01 * 0.62, 0.12, 0.92))
        threshold = float(np.quantile(edge_values, quantile))
        selected = np.logical_and(edge_ring, contrast >= threshold)
        if int(np.count_nonzero(selected)) == 0:
            selected = np.logical_and(edge_ring, contrast >= float(np.mean(edge_values)))
        if int(np.count_nonzero(selected)) == 0:
            return None

        grow_px = int(round(1 + 7 * strength01))
        if grow_px > 0:
            selected_u8 = np.where(selected, 255, 0).astype(np.uint8)
            selected_u8 = cv2.dilate(selected_u8, np.ones((grow_px * 2 + 1, grow_px * 2 + 1), dtype=np.uint8), iterations=1)
            selected = selected_u8 > 0

        feather_sigma = 0.9 + 2.2 * strength01
        selected_f = cv2.GaussianBlur(selected.astype(np.float32), (0, 0), feather_sigma)
        selected_f = np.clip(selected_f, 0.0, 1.0)
        selected_f *= mask.astype(np.float32)
        if float(np.max(selected_f)) < 1e-6:
            return None

        return np.clip(selected_f * 255.0, 0, 255).astype(np.uint8)

    def _run_auto_edge_blur_correction(self):
        layers = list(self._iter_clip_layer_entries())
        if not layers:
            QMessageBox.information(self, "Auto blur", "Brak warstw do automatycznej korekty bluru.")
            return

        strength = float(self.sld_auto_edge_blur.value()) / 100.0 if hasattr(self, "sld_auto_edge_blur") else 0.62
        applied = 0
        cleared = 0
        for layer_key, path in layers:
            mask = self._build_auto_edge_blur_mask_for_layer(path, strength)
            key = str(layer_key)
            if isinstance(mask, np.ndarray) and mask.size > 0 and int(np.count_nonzero(mask)) > 0:
                self.edge_blur_brush_by_layer[key] = mask
                applied += 1
            else:
                if key in self.edge_blur_brush_by_layer:
                    del self.edge_blur_brush_by_layer[key]
                    cleared += 1

        self._update_clip_timeline_preview()
        QMessageBox.information(
            self,
            "Auto blur",
            f"Automatyczna korekta zakonczona. Zaktualizowano: {applied} warstw." + (f" Wyczyszczono: {cleared}." if cleared > 0 else ""),
        )

    def _apply_selected_layer_blur(self):
        layer_key = self._selected_clip_layer_key()
        if not layer_key:
            return
        blur_value = self._edge_blur_value()
        if blur_value <= 0.0:
            self.edge_blur_by_layer.pop(str(layer_key), None)
        else:
            self.edge_blur_by_layer[str(layer_key)] = float(blur_value)
        self._update_clip_timeline_preview()

    def _apply_all_layers_blur(self):
        blur_value = self._edge_blur_value()
        changed = False
        for entry in getattr(self, "clip_timeline_entries", []):
            layer_key = entry.get("layer_key")
            if not layer_key:
                continue
            key = str(layer_key)
            if blur_value <= 0.0:
                if key in self.edge_blur_by_layer:
                    del self.edge_blur_by_layer[key]
                    changed = True
            else:
                if float(self.edge_blur_by_layer.get(key, -1.0)) != float(blur_value):
                    self.edge_blur_by_layer[key] = float(blur_value)
                    changed = True
        if changed:
            self._update_clip_timeline_preview()

    def _on_stage_tab_changed(self, index: int):
        if getattr(self, "_updating_stage_gate", False):
            return
        if index < 0:
            return
        if index != 5:
            self._stop_motion_clip_preview()
        if index > 1 and not self._is_stage1_complete():
            self._updating_stage_gate = True
            self.stage_nav.setCurrentRow(1)
            self.stage_stack.setCurrentIndex(1)
            self._updating_stage_gate = False
            self._update_run_button_visibility(1)
            self._update_cut_sections_visibility(1)
            QMessageBox.information(self, "Etapy", "Do kolejnych etapow przejdziesz dopiero po podziale calego obrazu na strefy.")
            return
        if index == 2:
            self._refresh_clip_timeline_layers(keep_index=True)
        if index == 5:
            self._refresh_motion_layers(keep_selection=True)
        self.stage_stack.setCurrentIndex(index)
        self._update_run_button_visibility(index)
        self._update_cut_sections_visibility(index)

    def _update_run_button_visibility(self, index: int):
        if not hasattr(self, "btn_run"):
            return
        last_stage = max(0, self.stage_stack.count() - 1) if hasattr(self, "stage_stack") else 0
        self.btn_run.setVisible(int(index) == int(last_stage))

    def _update_cut_sections_visibility(self, index: int):
        visible = int(index) == 1
        if hasattr(self, "cuts_label"):
            self.cuts_label.setVisible(visible)
        if hasattr(self, "cut_scroll"):
            self.cut_scroll.setVisible(visible)

    def accept(self):
        last_stage = max(0, self.stage_stack.count() - 1)
        current_stage = int(self.stage_stack.currentIndex())
        if current_stage != last_stage:
            self.stage_nav.setCurrentRow(last_stage)
            return
        if hasattr(self, "chk_add_stars") and self.chk_add_stars.isChecked():
            mask_path = str(self.edit_stars_mask_path.text() or "").strip() if hasattr(self, "edit_stars_mask_path") else ""
            if not mask_path or not os.path.exists(mask_path):
                QMessageBox.warning(self, "Dodaj gwiazdy", "Wybierz poprawny plik StarMask.")
                self.stage_nav.setCurrentRow(4)
                return
        if len(self._build_motion_behaviors()) == 0:
            QMessageBox.warning(self, "3D FLY", "Brak warstw do ruchu. Najpierw podziel obraz lub wczytaj pocięte warstwy.")
            self.stage_nav.setCurrentRow(1)
            return
        super().accept()

    def _on_zone_changed(self):
        if self.combo_zone is None:
            zone_index = int(getattr(self, "current_zone_index", 0))
        else:
            zone_index = int(self.combo_zone.currentData() or 0)
            self.current_zone_index = zone_index
        self.zone_widget.set_active_zone(zone_index)

    def _sync_brush(self):
        self.zone_widget.set_brush(
            int(self.spin_brush.value()),
            float(self.spin_tolerance.value()),
            float(self.spin_aggressiveness.value()),
        )

    def _update_starless_preview(self):
        if not hasattr(self, "starless_preview_label"):
            return
        if self.preview_image is None:
            self.starless_preview_label.setText("Brak obrazu do podgladu.")
            self.starless_preview_label.setPixmap(QPixmap())
            return

        pix = np_to_qpixmap(self.preview_image)
        target_size = self.starless_preview_label.size()
        if target_size.width() > 8 and target_size.height() > 8:
            pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.starless_preview_label.setText("")
        self.starless_preview_label.setPixmap(pix)

    def _compose_cut_preview_layer(self, path: str, blur_sigma: float = 0.0) -> np.ndarray | None:
        if not path or not os.path.exists(path):
            return None
        cut_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if cut_img is None:
            return None
        if cut_img.ndim < 3:
            return cv2.cvtColor(cut_img, cv2.COLOR_GRAY2BGR)
        if cut_img.shape[2] == 4:
            alpha = cut_img[:, :, 3].astype(np.float32) / 255.0
            if blur_sigma > 0.0:
                alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)
                alpha = np.clip(alpha, 0.0, 1.0)
            alpha = alpha[:, :, np.newaxis]
            rgb = cut_img[:, :, :3].astype(np.float32)
            white = np.full_like(rgb, 255.0)
            comp = rgb * alpha + white * (1.0 - alpha)
            return np.clip(comp, 0, 255).astype(np.uint8)
        return cut_img[:, :, :3]

    def _refresh_clip_timeline_layers(self, keep_index: bool = True):
        if not hasattr(self, "clip_timeline_list"):
            return
        old_index = int(getattr(self, "clip_timeline_index", 0))
        entries = [{"label": "Tlo", "image": self.source_image.copy()}]
        for idx, path in enumerate(self.cut_items):
            layer = self._compose_cut_preview_layer(path, blur_sigma=0.0)
            if layer is None:
                continue
            entries.append(
                {
                    "label": f"Warstwa {idx + 1}: {os.path.basename(path)}",
                    "path": path,
                    "layer_key": f"cut:{idx}",
                }
            )

        self.clip_timeline_entries = entries
        self.clip_timeline_list.blockSignals(True)
        self.clip_timeline_list.clear()
        for idx, entry in enumerate(self.clip_timeline_entries, start=1):
            self.clip_timeline_list.addItem(f"{idx}. {entry.get('label', 'Warstwa')}")
        self.clip_timeline_list.blockSignals(False)

        if keep_index:
            self.clip_timeline_index = max(0, min(old_index, len(self.clip_timeline_entries) - 1))
        else:
            self.clip_timeline_index = 0

        if self.clip_timeline_entries:
            self.clip_timeline_list.setCurrentRow(self.clip_timeline_index)
            self._update_clip_timeline_preview()
        else:
            self.clip_preview_label.setText("Brak warstw timeline.")
            self.clip_preview_label.setPixmap(QPixmap())
        self._update_clip_nav_buttons()

    def _update_clip_timeline_preview(self):
        if not hasattr(self, "clip_preview_label"):
            return
        if not self.clip_timeline_entries:
            self.clip_preview_label.setText("Brak warstw timeline.")
            self.clip_preview_label.setPixmap(QPixmap())
            return

        idx = max(0, min(int(self.clip_timeline_index), len(self.clip_timeline_entries) - 1))
        self.clip_timeline_index = idx
        entry = self.clip_timeline_entries[idx]
        image = entry.get("image")
        layer_key = entry.get("layer_key")
        has_layer = bool(layer_key)
        if hasattr(self, "btn_apply_edge_blur"):
            self.btn_apply_edge_blur.setEnabled(has_layer)
        if hasattr(self, "sld_edge_blur"):
            self.sld_edge_blur.setEnabled(has_layer)
        if hasattr(self, "chk_edge_blur_brush"):
            self.chk_edge_blur_brush.setEnabled(has_layer)
        if hasattr(self, "spin_edge_blur_brush_size"):
            self.spin_edge_blur_brush_size.setEnabled(has_layer and bool(getattr(self, "chk_edge_blur_brush", None) and self.chk_edge_blur_brush.isChecked()))
        if hasattr(self, "btn_edge_blur_brush_clear"):
            self.btn_edge_blur_brush_clear.setEnabled(has_layer)
        if hasattr(self, "btn_auto_edge_blur"):
            self.btn_auto_edge_blur.setEnabled(len(self.clip_timeline_entries) > 1)
        if hasattr(self, "sld_auto_edge_blur"):
            self.sld_auto_edge_blur.setEnabled(len(self.clip_timeline_entries) > 1)
        if has_layer and hasattr(self, "sld_edge_blur"):
            stored = float(self.edge_blur_by_layer.get(str(layer_key), 0.0))
            slider_value = int(round(max(0.0, min(20.0, stored)) * 10.0))
            self.sld_edge_blur.blockSignals(True)
            self.sld_edge_blur.setValue(slider_value)
            self.sld_edge_blur.blockSignals(False)
            if hasattr(self, "lbl_edge_blur_value"):
                self.lbl_edge_blur_value.setText(f"{stored:.1f}")
        elif hasattr(self, "lbl_edge_blur_value"):
            self.lbl_edge_blur_value.setText("0.0")
        if image is None and idx > 0:
            path = str(entry.get("path") or "")
            blur_sigma = float(self.edge_blur_by_layer.get(str(layer_key), 0.0)) if has_layer else 0.0
            image = self._compose_cut_preview_layer(path, blur_sigma=blur_sigma)
        if not isinstance(image, np.ndarray) or image.size == 0:
            self.clip_preview_label.setText("Brak podgladu warstwy.")
            self.clip_preview_label.setPixmap(QPixmap())
            return

        display = image.copy()
        if has_layer:
            brush_mask = self.edge_blur_brush_by_layer.get(str(layer_key))
            if isinstance(brush_mask, np.ndarray) and brush_mask.size > 0 and int(np.count_nonzero(brush_mask)) > 0:
                if brush_mask.shape[:2] != display.shape[:2]:
                    brush_mask = cv2.resize(brush_mask.astype(np.uint8), (display.shape[1], display.shape[0]), interpolation=cv2.INTER_NEAREST)
                blend = np.clip(brush_mask.astype(np.float32) / 255.0, 0.0, 1.0)
                blend3 = (blend * 0.46)[:, :, np.newaxis]
                tint = np.zeros_like(display, dtype=np.float32)
                tint[:, :, 0] = 40.0
                tint[:, :, 1] = 145.0
                tint[:, :, 2] = 255.0
                display = np.clip(display.astype(np.float32) * (1.0 - blend3) + tint * blend3, 0, 255).astype(np.uint8)

        pix = np_to_qpixmap(display)
        target_size = self.clip_preview_label.size()
        if target_size.width() > 8 and target_size.height() > 8:
            pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.clip_preview_label.setText("")
        self.clip_preview_label.setPixmap(pix)

    def _on_clip_timeline_row_changed(self, row: int):
        if row < 0:
            return
        self.clip_timeline_index = int(row)
        self._update_clip_timeline_preview()
        self._update_clip_nav_buttons()

    def _update_clip_nav_buttons(self):
        count = len(getattr(self, "clip_timeline_entries", []))
        if not hasattr(self, "btn_clip_prev") or not hasattr(self, "btn_clip_next"):
            return
        idx = int(getattr(self, "clip_timeline_index", 0))
        self.btn_clip_prev.setEnabled(count > 1 and idx > 0)
        self.btn_clip_next.setEnabled(count > 1 and idx < count - 1)

    def _show_prev_clip_layer(self):
        if not self.clip_timeline_entries:
            return
        idx = max(0, int(self.clip_timeline_index) - 1)
        self.clip_timeline_list.setCurrentRow(idx)

    def _show_next_clip_layer(self):
        if not self.clip_timeline_entries:
            return
        idx = min(len(self.clip_timeline_entries) - 1, int(self.clip_timeline_index) + 1)
        self.clip_timeline_list.setCurrentRow(idx)

    def _on_edge_blur_brush_toggled(self, _enabled: bool):
        if hasattr(self, "spin_edge_blur_brush_size"):
            has_layer = bool(self._selected_clip_layer_key())
            self.spin_edge_blur_brush_size.setEnabled(has_layer and self.chk_edge_blur_brush.isChecked())
        self._update_clip_timeline_preview()

    def _clear_edge_blur_brush_for_selected_layer(self):
        layer_key = self._selected_clip_layer_key()
        if not layer_key:
            return
        self.edge_blur_brush_by_layer.pop(str(layer_key), None)
        self._update_clip_timeline_preview()

    def _clip_preview_pos_to_image(self, pos) -> tuple[int, int] | None:
        if not self.clip_timeline_entries:
            return None
        idx = max(0, min(int(self.clip_timeline_index), len(self.clip_timeline_entries) - 1))
        entry = self.clip_timeline_entries[idx]
        layer_key = entry.get("layer_key")
        image = entry.get("image")
        if image is None and layer_key:
            image = self._compose_cut_preview_layer(str(entry.get("path") or ""), blur_sigma=0.0)
        if not isinstance(image, np.ndarray) or image.size == 0:
            return None

        img_h, img_w = image.shape[:2]
        rect = self.clip_preview_label.contentsRect()
        if rect.width() <= 1 or rect.height() <= 1:
            return None
        scale = min(rect.width() / float(img_w), rect.height() / float(img_h))
        disp_w = img_w * scale
        disp_h = img_h * scale
        x0 = rect.x() + (rect.width() - disp_w) * 0.5
        y0 = rect.y() + (rect.height() - disp_h) * 0.5
        px = float(pos.x())
        py = float(pos.y())
        if px < x0 or py < y0 or px > (x0 + disp_w) or py > (y0 + disp_h):
            return None
        ix = int(round((px - x0) * (img_w / max(1e-6, disp_w))))
        iy = int(round((py - y0) * (img_h / max(1e-6, disp_h))))
        ix = max(0, min(img_w - 1, ix))
        iy = max(0, min(img_h - 1, iy))
        return ix, iy

    def _paint_edge_blur_brush(self, ix: int, iy: int, paint_value: int):
        layer_key = self._selected_clip_layer_key()
        if not layer_key:
            return
        idx = max(0, min(int(self.clip_timeline_index), len(self.clip_timeline_entries) - 1))
        entry = self.clip_timeline_entries[idx]
        image = entry.get("image")
        if image is None:
            image = self._compose_cut_preview_layer(str(entry.get("path") or ""), blur_sigma=0.0)
        if not isinstance(image, np.ndarray) or image.size == 0:
            return

        h, w = image.shape[:2]
        key = str(layer_key)
        mask = self.edge_blur_brush_by_layer.get(key)
        if not isinstance(mask, np.ndarray) or mask.shape[:2] != (h, w):
            mask = np.zeros((h, w), dtype=np.uint8)
        brush_size = int(self.spin_edge_blur_brush_size.value()) if hasattr(self, "spin_edge_blur_brush_size") else 24
        brush_size = max(1, int(brush_size))
        cv2.circle(mask, (int(ix), int(iy)), brush_size, int(np.clip(paint_value, 0, 255)), -1, cv2.LINE_AA)
        if int(np.count_nonzero(mask)) == 0:
            self.edge_blur_brush_by_layer.pop(key, None)
        else:
            self.edge_blur_brush_by_layer[key] = mask
        self._update_clip_timeline_preview()

    def _on_clip_preview_mouse_press(self, event):
        if not hasattr(self, "chk_edge_blur_brush") or not self.chk_edge_blur_brush.isChecked():
            return QLabel.mousePressEvent(self.clip_preview_label, event)
        if not self._selected_clip_layer_key():
            return QLabel.mousePressEvent(self.clip_preview_label, event)
        if event.button() == Qt.LeftButton:
            self._edge_blur_brush_value = 255
        elif event.button() == Qt.RightButton:
            self._edge_blur_brush_value = 0
        else:
            return QLabel.mousePressEvent(self.clip_preview_label, event)

        pos = self._clip_preview_pos_to_image(event.pos())
        if pos is None:
            return
        self._edge_blur_brush_painting = True
        self._paint_edge_blur_brush(pos[0], pos[1], self._edge_blur_brush_value)
        event.accept()

    def _on_clip_preview_mouse_move(self, event):
        if not self._edge_blur_brush_painting:
            return QLabel.mouseMoveEvent(self.clip_preview_label, event)
        pos = self._clip_preview_pos_to_image(event.pos())
        if pos is None:
            return
        self._paint_edge_blur_brush(pos[0], pos[1], self._edge_blur_brush_value)
        event.accept()

    def _on_clip_preview_mouse_release(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._edge_blur_brush_painting = False
            event.accept()
            return
        return QLabel.mouseReleaseEvent(self.clip_preview_label, event)

    def _on_starless_advanced_toggled(self, enabled: bool):
        show = bool(enabled)
        if hasattr(self, "lbl_starless_stride"):
            self.lbl_starless_stride.setVisible(show)
        if hasattr(self, "spin_starless_stride"):
            self.spin_starless_stride.setVisible(show)

    def _on_starless_stride_changed(self, value: int):
        if not hasattr(self, "spin_starless_stride"):
            return
        even_value = int(max(2, min(512, value)))
        if even_value % 2 != 0:
            even_value += 1
            if even_value > 512:
                even_value = 512
        if even_value != int(value):
            self.spin_starless_stride.blockSignals(True)
            self.spin_starless_stride.setValue(even_value)
            self.spin_starless_stride.blockSignals(False)

    def _auto_starmask_output_path(self) -> str:
        out_dir = self._workspace_output_directory()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"starmask_auto_{stamp}"
        out_path = os.path.join(out_dir, base_name + ".png")
        suffix = 1
        while os.path.exists(out_path):
            out_path = os.path.join(out_dir, f"{base_name}_{suffix}.png")
            suffix += 1
        return out_path

    def _build_starmask_from_starless(self, original_bgr: np.ndarray, starless_bgr: np.ndarray) -> np.ndarray:
        original_u8 = normalize_to_uint8_bgr(original_bgr)
        starless_u8 = normalize_to_uint8_bgr(starless_bgr)
        if original_u8.shape[:2] != starless_u8.shape[:2]:
            starless_u8 = cv2.resize(starless_u8, (original_u8.shape[1], original_u8.shape[0]), interpolation=cv2.INTER_AREA)

        diff = cv2.subtract(original_u8, starless_u8)
        mask_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        if mask_gray.size == 0:
            return mask_gray

        mask_gray = cv2.GaussianBlur(mask_gray, (0, 0), 0.8)
        mask_gray = np.where(mask_gray > 3, mask_gray, 0).astype(np.uint8)
        if int(np.max(mask_gray)) > 0:
            mask_gray = cv2.normalize(mask_gray, None, 0, 255, cv2.NORM_MINMAX)
        return mask_gray

    def _remove_stars_now(self):
        app = self.parent()
        starnet_path = str(getattr(app, "starnet_path", "") or "").strip() if app is not None else ""

        original_before_starless = self.source_image.copy()

        stride = int(self.spin_starless_stride.value()) if hasattr(self, "spin_starless_stride") else 16
        stride = max(2, min(512, stride))
        if stride % 2 != 0:
            stride += 1
            if stride > 512:
                stride = 512

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            starless, err = run_starnet_sync_for_image(self.source_image.copy(), starnet_path, stride)
        finally:
            QApplication.restoreOverrideCursor()

        if err:
            QMessageBox.warning(self, "Usun gwiazdy", err)
            return
        if not isinstance(starless, np.ndarray):
            QMessageBox.warning(self, "Usun gwiazdy", "StarNet++ nie zwrocil poprawnego obrazu.")
            return

        starless_u8 = normalize_to_uint8_bgr(starless)
        auto_mask_path = ""
        if hasattr(self, "edit_stars_mask_path"):
            starmask_u8 = self._build_starmask_from_starless(original_before_starless, starless_u8)
            if isinstance(starmask_u8, np.ndarray) and starmask_u8.size > 0 and int(np.count_nonzero(starmask_u8)) > 0:
                candidate = self._auto_starmask_output_path()
                if cv2.imwrite(candidate, starmask_u8):
                    auto_mask_path = os.path.abspath(candidate)
                    self.edit_stars_mask_path.setText(auto_mask_path)

        self.source_image = starless_u8
        h, w = self.source_image.shape[:2]
        preview_h, preview_w = self.preview_image.shape[:2]
        self.preview_image = cv2.resize(self.source_image, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        self._cut_layer_preview_cache.clear()
        self.zone_widget.set_preview_image(self.preview_image)
        self.zone_widget.update()
        self._update_starless_preview()
        self._update_edge_preview()
        self._refresh_clip_timeline_layers(keep_index=True)
        self._refresh_motion_layers(keep_selection=True)
        message = "Gotowe. Gwiazdy zostaly usuniete."
        if auto_mask_path:
            message += f"\nWygenerowano StarMask: {auto_mask_path}"
        QMessageBox.information(self, "Usun gwiazdy", message)

    def _hole_size_limits(self) -> tuple[int, int]:
        if hasattr(self, "slider_hole_size_px"):
            left_value, right_value = self.slider_hole_size_px.values()
            hole_min = int(round(min(left_value, right_value)))
            hole_max = int(round(max(left_value, right_value)))
            return max(0, hole_min), min(256, hole_max)
        if hasattr(self, "spin_hole_size_px"):
            hole_max = int(self.spin_hole_size_px.value())
            return 0, min(256, max(0, hole_max))
        return 0, 24

    def _on_hole_size_range_changed(self, left_value: float, right_value: float):
        if not hasattr(self, "lbl_hole_size_px"):
            return
        hole_min = int(round(min(left_value, right_value)))
        hole_max = int(round(max(left_value, right_value)))
        self.lbl_hole_size_px.setText(f"{hole_min}-{hole_max} px")

    def _get_motion_zone_ids(self) -> list[int]:
        zone_map = self.zone_widget.get_zone_map() if hasattr(self, "zone_widget") else None
        if not isinstance(zone_map, np.ndarray) or zone_map.size == 0:
            return []
        ids = np.unique(zone_map[zone_map >= 0]).tolist() if np.any(zone_map >= 0) else []
        return sorted(int(zone_id) for zone_id in ids)

    def _get_motion_layer_keys(self):
        zone_ids = self._get_motion_zone_ids()
        if zone_ids:
            return zone_ids
        keys = []
        for idx, path in enumerate(self.cut_items):
            if os.path.exists(path):
                keys.append(f"cut:{idx}")
        return keys

    def _estimate_cut_layer_preview_placement(self, path: str):
        if not path or not os.path.exists(path):
            return None
        if not isinstance(self.preview_image, np.ndarray) or self.preview_image.size == 0:
            return None

        preview_shape = tuple(self.preview_image.shape[:2])
        cached = self._cut_layer_preview_cache.get(path)
        if isinstance(cached, dict) and cached.get("preview_shape") == preview_shape:
            return cached

        layer = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if layer is None or layer.ndim < 3:
            return None

        if layer.shape[2] >= 4:
            alpha = layer[:, :, 3]
            bgr = layer[:, :, :3]
        else:
            bgr = layer[:, :, :3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            alpha = np.where(gray < 250, 255, 0).astype(np.uint8)

        ys, xs = np.where(alpha > 0)
        if ys.size == 0:
            return None

        y0, y1 = int(np.min(ys)), int(np.max(ys)) + 1
        x0, x1 = int(np.min(xs)), int(np.max(xs)) + 1
        crop_bgr = bgr[y0:y1, x0:x1]
        crop_alpha = alpha[y0:y1, x0:x1]

        src_h, src_w = self.source_image.shape[:2]
        prev_h, prev_w = self.preview_image.shape[:2]
        scale_x = prev_w / max(1.0, float(src_w))
        scale_y = prev_h / max(1.0, float(src_h))
        draw_w = max(2, int(round(crop_bgr.shape[1] * scale_x)))
        draw_h = max(2, int(round(crop_bgr.shape[0] * scale_y)))
        if draw_w >= prev_w or draw_h >= prev_h:
            return None

        template_bgr = cv2.resize(crop_bgr, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
        template_mask = cv2.resize(crop_alpha, (draw_w, draw_h), interpolation=cv2.INTER_NEAREST)
        template_mask = np.where(template_mask > 0, 255, 0).astype(np.uint8)
        if int(np.count_nonzero(template_mask)) == 0:
            return None

        src_gray = cv2.cvtColor(self.preview_image, cv2.COLOR_BGR2GRAY)
        tpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

        top_left = (0, 0)
        try:
            match = cv2.matchTemplate(src_gray, tpl_gray, cv2.TM_SQDIFF, mask=template_mask)
            min_loc = cv2.minMaxLoc(match)[2]
            top_left = (int(min_loc[0]), int(min_loc[1]))
        except Exception:
            match = cv2.matchTemplate(src_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            max_loc = cv2.minMaxLoc(match)[3]
            top_left = (int(max_loc[0]), int(max_loc[1]))

        placement = {
            "preview_shape": preview_shape,
            "x": int(top_left[0]),
            "y": int(top_left[1]),
            "mask": template_mask,
            "rgb": template_bgr,
        }
        self._cut_layer_preview_cache[path] = placement
        return placement

    def _refresh_motion_layers(self, keep_selection: bool = True):
        self._stop_motion_clip_preview()
        previous_zone = self.selected_motion_zone if (keep_selection and self.selected_motion_zone is not None) else None
        layer_keys = self._get_motion_layer_keys()
        for idx, layer_key in enumerate(layer_keys, start=1):
            if layer_key not in self.motion_settings:
                self.motion_settings[layer_key] = {
                    "angle_deg": -35.0 + idx * 12.0,
                    "speed": 24.0 + idx * 6.0,
                    "zoom_speed": 0.0,
                    "direction_mode": "vector",
                }

        for key in list(self.motion_settings.keys()):
            if key not in layer_keys:
                del self.motion_settings[key]

        if not layer_keys:
            self.selected_motion_zone = None
            self._update_motion_preview()
            return

        if previous_zone in layer_keys:
            self.selected_motion_zone = previous_zone
        elif self.selected_motion_zone in layer_keys:
            pass
        else:
            self.selected_motion_zone = layer_keys[0]
        self._sync_motion_props_widgets()
        self._update_motion_preview()

    def _current_motion_zone_id(self):
        return self.selected_motion_zone if self.selected_motion_zone is not None else None

    def _sync_motion_props_widgets(self):
        if not hasattr(self, "combo_motion_layer_props"):
            return
        current = self._current_motion_zone_id()
        layer_keys = self._get_motion_layer_keys()

        self.combo_motion_layer_props.blockSignals(True)
        self.combo_motion_layer_props.clear()
        for idx, layer_key in enumerate(layer_keys, start=1):
            self.combo_motion_layer_props.addItem(f"Warstwa {idx}", str(layer_key))
        self.combo_motion_layer_props.blockSignals(False)

        if current is None or str(current) not in {str(k) for k in layer_keys}:
            if layer_keys:
                current = layer_keys[0]
                self.selected_motion_zone = current

        if current is not None:
            self.combo_motion_layer_props.blockSignals(True)
            for i in range(self.combo_motion_layer_props.count()):
                if self.combo_motion_layer_props.itemData(i) == str(current):
                    self.combo_motion_layer_props.setCurrentIndex(i)
                    break
            self.combo_motion_layer_props.blockSignals(False)

        cfg = self.motion_settings.get(current, {}) if current is not None else {}
        direction_mode = str(cfg.get("direction_mode") or "vector")
        speed = float(cfg.get("speed") or 0.0)
        zoom_speed = float(cfg.get("zoom_speed") or 0.0)
        enabled = current is not None

        if hasattr(self, "combo_motion_direction"):
            self.combo_motion_direction.blockSignals(True)
            self.combo_motion_direction.setCurrentIndex(0 if direction_mode != "viewer" else 1)
            self.combo_motion_direction.setEnabled(enabled)
            self.combo_motion_direction.blockSignals(False)

        if hasattr(self, "spin_motion_speed_props"):
            self.spin_motion_speed_props.blockSignals(True)
            self.spin_motion_speed_props.setValue(speed)
            self.spin_motion_speed_props.setEnabled(enabled and direction_mode != "viewer")
            self.spin_motion_speed_props.blockSignals(False)

        if hasattr(self, "spin_motion_zoom_speed"):
            self.spin_motion_zoom_speed.blockSignals(True)
            self.spin_motion_zoom_speed.setValue(zoom_speed)
            self.spin_motion_zoom_speed.setEnabled(enabled)
            self.spin_motion_zoom_speed.blockSignals(False)

    def _on_motion_props_layer_changed(self, index: int):
        self._stop_motion_clip_preview()
        if index < 0 or not hasattr(self, "combo_motion_layer_props"):
            return
        key = self.combo_motion_layer_props.itemData(index)
        if key is None:
            return
        self.selected_motion_zone = key if str(key).startswith("cut:") else int(key)
        self._sync_motion_props_widgets()
        self._update_motion_preview()

    def _on_motion_props_direction_changed(self, _index: int):
        self._stop_motion_clip_preview()
        zone_id = self._current_motion_zone_id()
        if zone_id is None:
            return
        cfg = self.motion_settings.setdefault(zone_id, {"angle_deg": 0.0, "speed": 24.0, "zoom_speed": 0.0, "direction_mode": "vector"})
        mode = self.combo_motion_direction.currentData() if hasattr(self, "combo_motion_direction") else "vector"
        cfg["direction_mode"] = "viewer" if str(mode) == "viewer" else "vector"
        self._sync_motion_props_widgets()
        self._update_motion_preview()

    def _on_motion_props_speed_changed(self, value: float):
        self._stop_motion_clip_preview()
        zone_id = self._current_motion_zone_id()
        if zone_id is None:
            return
        cfg = self.motion_settings.setdefault(zone_id, {"angle_deg": 0.0, "speed": 24.0, "zoom_speed": 0.0, "direction_mode": "vector"})
        cfg["speed"] = float(value)
        self._update_motion_preview()

    def _on_motion_props_zoom_changed(self, value: float):
        self._stop_motion_clip_preview()
        zone_id = self._current_motion_zone_id()
        if zone_id is None:
            return
        cfg = self.motion_settings.setdefault(zone_id, {"angle_deg": 0.0, "speed": 24.0, "zoom_speed": 0.0, "direction_mode": "vector"})
        cfg["zoom_speed"] = float(value)
        self._update_motion_preview()

    def _motion_label_pos_to_image(self, pos) -> tuple[int, int] | None:
        if not isinstance(self.preview_image, np.ndarray) or self.preview_image.size == 0:
            return None
        img_h, img_w = self.preview_image.shape[:2]
        rect = self.motion_preview_label.contentsRect()
        if rect.width() <= 1 or rect.height() <= 1:
            return None
        scale = min(rect.width() / float(img_w), rect.height() / float(img_h))
        disp_w = img_w * scale
        disp_h = img_h * scale
        x0 = rect.x() + (rect.width() - disp_w) * 0.5
        y0 = rect.y() + (rect.height() - disp_h) * 0.5
        px = float(pos.x())
        py = float(pos.y())
        if px < x0 or py < y0 or px > (x0 + disp_w) or py > (y0 + disp_h):
            return None
        ix = int(round((px - x0) * (img_w / max(1e-6, disp_w))))
        iy = int(round((py - y0) * (img_h / max(1e-6, disp_h))))
        ix = max(0, min(img_w - 1, ix))
        iy = max(0, min(img_h - 1, iy))
        return ix, iy

    def _pick_motion_zone_by_point(self, ix: int, iy: int) -> int | None:
        closest_zone = None
        closest_dist = None
        for zone_id, data in self.motion_zone_centers.items():
            cx, cy = data.get("center", (0, 0))
            dist = math.hypot(float(ix - cx), float(iy - cy))
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest_zone = zone_id
        if closest_zone is None:
            return None
        if closest_dist is not None and closest_dist > 120.0:
            return None
        return closest_zone

    def _set_motion_angle_from_point(self, zone_id, ix: int, iy: int):
        data = self.motion_zone_centers.get(zone_id)
        if not data:
            return
        cx, cy = data.get("center", (0, 0))
        dx = float(ix - cx)
        dy = float(iy - cy)
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return
        angle = math.degrees(math.atan2(dy, dx))
        cfg = self.motion_settings.setdefault(zone_id, {"angle_deg": 0.0, "speed": 24.0, "zoom_speed": 0.0, "direction_mode": "vector"})
        cfg["angle_deg"] = float(angle)

    def _mask_anchor_local(self, mask: np.ndarray) -> tuple[int, int] | None:
        if not isinstance(mask, np.ndarray) or mask.size == 0:
            return None
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
        cx_f = float(np.mean(xs))
        cy_f = float(np.mean(ys))
        # Ensure arrow tail is on the section: pick nearest pixel that belongs to the mask.
        d2 = (xs.astype(np.float32) - cx_f) ** 2 + (ys.astype(np.float32) - cy_f) ** 2
        idx = int(np.argmin(d2))
        return int(xs[idx]), int(ys[idx])

    def _on_motion_preview_mouse_press(self, event):
        self._stop_motion_clip_preview()
        if event.button() != Qt.LeftButton:
            return QLabel.mousePressEvent(self.motion_preview_label, event)
        pos = self._motion_label_pos_to_image(event.pos())
        if pos is None:
            return
        ix, iy = pos
        zone_id = self._pick_motion_zone_by_point(ix, iy)
        if zone_id is None:
            return
        self.selected_motion_zone = zone_id
        self._sync_motion_props_widgets()
        self._motion_dragging = True
        self._set_motion_angle_from_point(zone_id, ix, iy)
        self._update_motion_preview()
        event.accept()

    def _on_motion_preview_mouse_move(self, event):
        if not self._motion_dragging or self.selected_motion_zone is None:
            return QLabel.mouseMoveEvent(self.motion_preview_label, event)
        pos = self._motion_label_pos_to_image(event.pos())
        if pos is None:
            return
        ix, iy = pos
        self._set_motion_angle_from_point(self.selected_motion_zone, ix, iy)
        self._update_motion_preview()
        event.accept()

    def _on_motion_preview_mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            self._motion_dragging = False
            event.accept()
            return
        return QLabel.mouseReleaseEvent(self.motion_preview_label, event)

    def _on_motion_preview_wheel(self, event):
        self._stop_motion_clip_preview()
        pos = self._motion_label_pos_to_image(event.pos())
        if pos is None:
            return QLabel.wheelEvent(self.motion_preview_label, event)
        ix, iy = pos
        zone_id = self._pick_motion_zone_by_point(ix, iy)
        if zone_id is None:
            return
        self.selected_motion_zone = zone_id
        cfg = self.motion_settings.setdefault(zone_id, {"angle_deg": 0.0, "speed": 24.0, "zoom_speed": 0.0, "direction_mode": "vector"})
        delta = event.angleDelta().y()
        step = 2.0 if delta > 0 else -2.0
        cfg["speed"] = float(max(0.0, min(400.0, float(cfg.get("speed", 24.0)) + step)))
        self._sync_motion_props_widgets()
        self._update_motion_preview()
        event.accept()

    def _toggle_motion_clip_preview(self):
        if self._motion_clip_timer.isActive():
            self._stop_motion_clip_preview()
            return
        self._start_motion_clip_preview()

    def _build_motion_preview_layers(self) -> tuple[np.ndarray, list[ImageLayer], list[str]]:
        if not isinstance(self.preview_image, np.ndarray) or self.preview_image.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8), [], []

        base = self.preview_image
        h, w = base.shape[:2]
        zone_map = self.zone_widget.get_zone_map() if hasattr(self, "zone_widget") else None
        has_zone_map = isinstance(zone_map, np.ndarray) and zone_map.shape[:2] == (h, w) and np.any(zone_map >= 0)
        behaviors = sorted(self._build_motion_behaviors(), key=lambda item: float(item.get("depth", 0.0)))
        occupied = np.zeros((h, w), dtype=bool)
        layers: list[ImageLayer] = []
        warnings: list[str] = []

        if has_zone_map:
            for behavior in behaviors:
                zone_id = int(behavior.get("zone", -1))
                if zone_id < 0:
                    continue
                raw_mask = zone_map == zone_id
                if not np.any(raw_mask):
                    continue

                overlap_px = int(np.count_nonzero(np.logical_and(raw_mask, occupied)))
                if overlap_px > 0:
                    warnings.append(f"Mask overlap removed for zone {zone_id}: {overlap_px} px")

                unique_mask = np.logical_and(raw_mask, np.logical_not(occupied))
                if not np.any(unique_mask):
                    continue
                occupied[unique_mask] = True

                layer_key = str(behavior.get("layer_key") or zone_id)
                alpha = unique_mask.astype(np.float32)
                blur_sigma = max(0.0, float(self.edge_blur_by_layer.get(layer_key, 0.0)) * 1.8)
                if blur_sigma > 0.0:
                    alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)
                    # Keep alpha strictly inside the unique ownership mask.
                    alpha *= unique_mask.astype(np.float32)
                alpha = np.clip(alpha, 0.0, 1.0)
                if float(np.max(alpha)) < 1e-6:
                    continue

                layer_image = np.zeros_like(base)
                layer_image[unique_mask] = base[unique_mask]
                layers.append(
                    ImageLayer(
                        image=layer_image,
                        alpha=alpha,
                        mask=unique_mask,
                        depth=float(behavior.get("depth", 0.0)),
                        move_x=float(behavior.get("move_x", 0.0)),
                        move_y=float(behavior.get("move_y", 0.0)),
                        zoom=float(behavior.get("zoom", 0.0)),
                        layer_key=layer_key,
                    )
                )
        else:
            for behavior in behaviors:
                layer_key = str(behavior.get("layer_key") or "")
                if not layer_key.startswith("cut:"):
                    continue
                try:
                    cut_index = int(layer_key.split(":", 1)[1])
                except Exception:
                    continue
                if cut_index < 0 or cut_index >= len(self.cut_items):
                    continue

                placement = self._estimate_cut_layer_preview_placement(self.cut_items[cut_index])
                if not isinstance(placement, dict):
                    continue

                mask_u8 = placement.get("mask")
                rgb_u8 = placement.get("rgb")
                if not isinstance(mask_u8, np.ndarray) or mask_u8.size == 0:
                    continue
                if not isinstance(rgb_u8, np.ndarray) or rgb_u8.size == 0:
                    continue

                x0 = int(placement.get("x", 0))
                y0 = int(placement.get("y", 0))
                mh, mw = mask_u8.shape[:2]
                x1 = min(w, x0 + mw)
                y1 = min(h, y0 + mh)
                if x1 <= x0 or y1 <= y0:
                    continue

                src_w = x1 - x0
                src_h = y1 - y0
                raw_mask = np.zeros((h, w), dtype=bool)
                raw_mask[y0:y1, x0:x1] = mask_u8[:src_h, :src_w] > 0
                if not np.any(raw_mask):
                    continue

                overlap_px = int(np.count_nonzero(np.logical_and(raw_mask, occupied)))
                if overlap_px > 0:
                    warnings.append(f"Mask overlap removed for {layer_key}: {overlap_px} px")

                unique_mask = np.logical_and(raw_mask, np.logical_not(occupied))
                if not np.any(unique_mask):
                    continue
                occupied[unique_mask] = True

                alpha = unique_mask.astype(np.float32)
                blur_sigma = max(0.0, float(self.edge_blur_by_layer.get(layer_key, 0.0)) * 1.8)
                if blur_sigma > 0.0:
                    alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)
                    alpha *= unique_mask.astype(np.float32)

                brush_strength_full = None
                brush_local = self.edge_blur_brush_by_layer.get(layer_key)
                if isinstance(brush_local, np.ndarray) and brush_local.size > 0 and int(np.count_nonzero(brush_local)) > 0:
                    local_h, local_w = int(src_h), int(src_w)
                    if brush_local.shape[:2] != (local_h, local_w):
                        brush_local = cv2.resize(brush_local.astype(np.uint8), (local_w, local_h), interpolation=cv2.INTER_NEAREST)
                    brush_local_f = np.clip(brush_local.astype(np.float32) / 255.0, 0.0, 1.0)
                    brush_local_f *= unique_mask[y0:y1, x0:x1].astype(np.float32)
                    if float(np.max(brush_local_f)) > 1e-6:
                        brush_strength_full = np.zeros((h, w), dtype=np.float32)
                        brush_strength_full[y0:y1, x0:x1] = brush_local_f

                if isinstance(brush_strength_full, np.ndarray) and float(np.max(brush_strength_full)) > 1e-6:
                    extra_sigma = max(1.2, blur_sigma * 1.8)
                    alpha_extra = unique_mask.astype(np.float32)
                    alpha_extra = cv2.GaussianBlur(alpha_extra, (0, 0), extra_sigma)
                    alpha_extra *= unique_mask.astype(np.float32)
                    alpha = alpha * (1.0 - brush_strength_full) + alpha_extra * brush_strength_full

                alpha = np.clip(alpha, 0.0, 1.0)
                if float(np.max(alpha)) < 1e-6:
                    continue

                layer_image = np.zeros_like(base)
                layer_image[y0:y1, x0:x1][unique_mask[y0:y1, x0:x1]] = rgb_u8[:src_h, :src_w][unique_mask[y0:y1, x0:x1]]
                layers.append(
                    ImageLayer(
                        image=layer_image,
                        alpha=alpha,
                        mask=unique_mask,
                        depth=float(behavior.get("depth", 0.0)),
                        move_x=float(behavior.get("move_x", 0.0)),
                        move_y=float(behavior.get("move_y", 0.0)),
                        zoom=float(behavior.get("zoom", 0.0)),
                        layer_key=layer_key,
                    )
                )

        overlap_after = _count_mask_overlap_pixels([layer.mask.astype(np.uint8) for layer in layers])
        if overlap_after > 0:
            warnings.append(f"Layer mask overlap detected after build: {overlap_after} px")

        background = _inpaint_background_without_layers(base, occupied)
        return background, layers, warnings

    def _start_motion_clip_preview(self):
        background, layers, warnings = self._build_motion_preview_layers()
        if not layers:
            QMessageBox.information(self, "Podglad klipu", "Brak warstw do podgladu klipu.")
            return

        if warnings:
            # Preview keeps working, but we still expose data consistency issues.
            QMessageBox.warning(self, "3D FLY", "\n".join(warnings[:3]))

        h, w = background.shape[:2]
        fps = max(12, int(self.spin_fps.value()) if hasattr(self, "spin_fps") else 24)
        duration = max(1.0, float(self.spin_duration.value()) if hasattr(self, "spin_duration") else 5.0)
        frame_count = max(16, min(96, int(round(duration * fps))))
        main_zoom_speed = float(self.spin_main_zoom_speed.value()) if hasattr(self, "spin_main_zoom_speed") else 0.03

        progress_dialog = MagicProgressDialog(self)
        progress_dialog.setWindowTitle("Podglad klipu")
        progress_dialog.show()
        progress_dialog.update_progress("Generowanie podgladu klipu...", 0, 0)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            frames = []
            for frame_idx in range(frame_count):
                phase = frame_idx / float(max(1, frame_count - 1))
                ease = 0.5 - 0.5 * math.cos(math.pi * phase)
                global_zoom = main_zoom_speed * ease
                global_x = -8.0 * ease
                global_y = 5.0 * ease
                composed, _ = _warp_layer_with_motion(background, np.ones((h, w), dtype=np.float32), global_x, global_y, global_zoom)

                for layer in layers:
                    depth_gain = 1.0 + float(layer.depth) * 0.7
                    move_x = float(layer.move_x) * ease * depth_gain
                    move_y = float(layer.move_y) * ease * depth_gain
                    zoom = float(layer.zoom) * ease * (1.0 + 0.4 * float(layer.depth))
                    warped_img, warped_alpha = _warp_layer_with_motion(layer.image, layer.alpha, move_x, move_y, zoom)
                    alpha3 = warped_alpha[:, :, np.newaxis]
                    composed = (composed.astype(np.float32) * (1.0 - alpha3) + warped_img.astype(np.float32) * alpha3)
                    composed = np.clip(composed, 0, 255).astype(np.uint8)

                frames.append(composed)
                progress_value = int(round((frame_idx + 1) * 100.0 / max(1, frame_count)))
                progress_dialog.update_progress("Generowanie podgladu klipu...", progress_value, progress_value)
        finally:
            QApplication.restoreOverrideCursor()
            progress_dialog.close()

        self._motion_clip_frames = frames
        self._motion_clip_frame_index = 0
        interval_ms = max(16, int(round(1000.0 / max(1, fps))))
        self._motion_clip_timer.start(interval_ms)
        if hasattr(self, "btn_motion_preview_clip"):
            self.btn_motion_preview_clip.setText("Zatrzymaj podglad")
        self._advance_motion_clip_preview_frame()

    def _advance_motion_clip_preview_frame(self):
        if not self._motion_clip_frames:
            self._stop_motion_clip_preview()
            return
        idx = int(self._motion_clip_frame_index) % len(self._motion_clip_frames)
        frame = self._motion_clip_frames[idx]
        pix = np_to_qpixmap(frame)
        target_size = self.motion_preview_label.size()
        if target_size.width() > 8 and target_size.height() > 8:
            pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.motion_preview_label.setText("")
        self.motion_preview_label.setPixmap(pix)
        self._motion_clip_frame_index = idx + 1

    def _stop_motion_clip_preview(self):
        if self._motion_clip_timer.isActive():
            self._motion_clip_timer.stop()
        self._motion_clip_frames = []
        self._motion_clip_frame_index = 0
        if hasattr(self, "btn_motion_preview_clip"):
            self.btn_motion_preview_clip.setText("Podglad klipu")
        self._update_motion_preview()

    def _update_motion_preview(self):
        if not hasattr(self, "motion_preview_label"):
            return
        frame = self.preview_image.copy() if isinstance(self.preview_image, np.ndarray) else None
        if frame is None or frame.size == 0:
            self.motion_preview_label.setText("Brak podgladu ruchu.")
            self.motion_preview_label.setPixmap(QPixmap())
            return

        zone_map = self.zone_widget.get_zone_map() if hasattr(self, "zone_widget") else None
        self.motion_zone_centers = {}
        has_zone_map = isinstance(zone_map, np.ndarray) and zone_map.shape[:2] == frame.shape[:2] and np.any(zone_map >= 0)
        if has_zone_map:
            selected_zone = self._current_motion_zone_id()
            for zone_id in self._get_motion_zone_ids():
                mask = zone_map == int(zone_id)
                if not np.any(mask):
                    continue
                anchor = self._mask_anchor_local(mask)
                if anchor is None:
                    continue
                cx, cy = anchor
                cfg = self.motion_settings.get(zone_id, {"angle_deg": 0.0, "speed": 24.0, "zoom_speed": 0.0, "direction_mode": "vector"})
                angle_rad = math.radians(float(cfg.get("angle_deg", 0.0)))
                speed = float(cfg.get("speed", 24.0))
                zoom_speed = float(cfg.get("zoom_speed", 0.0))
                direction_mode = str(cfg.get("direction_mode") or "vector")
                arrow_len = max(34.0, min(140.0, speed * 0.45))
                tx = int(round(cx + math.cos(angle_rad) * arrow_len))
                ty = int(round(cy + math.sin(angle_rad) * arrow_len))
                color = (255, 220, 90) if zone_id == selected_zone else (110, 225, 255)
                contour_mask = mask.astype(np.uint8)
                contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contours:
                    cv2.drawContours(frame, contours, -1, color, 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)
                if direction_mode != "viewer":
                    cv2.arrowedLine(frame, (cx, cy), (tx, ty), color, 3, cv2.LINE_AA, tipLength=0.34)
                    cv2.putText(frame, f"{int(round(speed))}", (tx + 6, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f"zoom {zoom_speed:+.2f}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                self.motion_zone_centers[zone_id] = {"center": (cx, cy), "tip": (tx, ty)}
        else:
            valid_paths = [p for p in self.cut_items if os.path.exists(p)]
            if valid_paths:
                selected_zone = self._current_motion_zone_id()
                for idx, path in enumerate(valid_paths):
                    placement = self._estimate_cut_layer_preview_placement(path)
                    if not isinstance(placement, dict):
                        continue
                    mask_u8 = placement.get("mask")
                    if not isinstance(mask_u8, np.ndarray) or mask_u8.size == 0:
                        continue
                    x0 = int(placement.get("x", 0))
                    y0 = int(placement.get("y", 0))
                    mask = mask_u8 > 0
                    if not np.any(mask):
                        continue
                    contours, _ = cv2.findContours(mask_u8.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    key = f"cut:{idx}"
                    cfg = self.motion_settings.get(key, {"angle_deg": 0.0, "speed": 24.0, "zoom_speed": 0.0, "direction_mode": "vector"})
                    color = (255, 220, 90) if key == selected_zone else (110, 225, 255)
                    for contour in contours:
                        contour[:, 0, 0] = contour[:, 0, 0] + x0
                        contour[:, 0, 1] = contour[:, 0, 1] + y0
                    if contours:
                        cv2.drawContours(frame, contours, -1, color, 1, cv2.LINE_AA)

                    anchor = self._mask_anchor_local(mask)
                    if anchor is None:
                        continue
                    cx = int(x0 + anchor[0])
                    cy = int(y0 + anchor[1])
                    angle_rad = math.radians(float(cfg.get("angle_deg", 0.0)))
                    speed = float(cfg.get("speed", 24.0))
                    zoom_speed = float(cfg.get("zoom_speed", 0.0))
                    direction_mode = str(cfg.get("direction_mode") or "vector")
                    arrow_len = max(34.0, min(140.0, speed * 0.45))
                    tx = int(round(cx + math.cos(angle_rad) * arrow_len))
                    ty = int(round(cy + math.sin(angle_rad) * arrow_len))
                    cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)
                    if direction_mode != "viewer":
                        cv2.arrowedLine(frame, (cx, cy), (tx, ty), color, 3, cv2.LINE_AA, tipLength=0.34)
                        cv2.putText(frame, f"{int(round(speed))}", (tx + 6, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, f"zoom {zoom_speed:+.2f}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                    self.motion_zone_centers[key] = {"center": (cx, cy), "tip": (tx, ty)}

        pix = np_to_qpixmap(frame)
        target_size = self.motion_preview_label.size()
        if target_size.width() > 8 and target_size.height() > 8:
            pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.motion_preview_label.setText("")
        self.motion_preview_label.setPixmap(pix)

    def _build_motion_behaviors(self) -> list[dict]:
        behaviors = []
        layer_keys = self._get_motion_layer_keys()
        for idx, layer_key in enumerate(layer_keys):
            cfg = self.motion_settings.get(layer_key, {"angle_deg": 0.0, "speed": 24.0, "zoom_speed": 0.0, "direction_mode": "vector"})
            angle_rad = math.radians(float(cfg.get("angle_deg", 0.0)))
            speed = float(cfg.get("speed", 24.0))
            zoom_speed = float(cfg.get("zoom_speed", 0.0))
            direction_mode = str(cfg.get("direction_mode") or "vector")
            if direction_mode == "viewer":
                move_x = 0.0
                move_y = 0.0
            else:
                move_x = float(math.cos(angle_rad) * speed)
                move_y = float(math.sin(angle_rad) * speed)
            zone_id = int(layer_key) if isinstance(layer_key, (int, np.integer)) else -1
            behaviors.append(
                {
                    "zone": int(zone_id),
                    "layer_key": str(layer_key),
                    "depth": float(idx) * 0.28,
                    "move_x": move_x,
                    "move_y": move_y,
                    "zoom": zoom_speed,
                }
            )
        return behaviors

    def _pick_output_path(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save 3D FLY video",
            self.edit_output.text().strip() or "3d_fly.mp4",
            "Video (*.mp4 *.avi)",
            options=get_safe_file_dialog_options(),
        )
        if path:
            self.edit_output.setText(path)

    def _pick_stars_mask_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz plik StarMask",
            self._workspace_output_directory(),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)",
            options=get_safe_file_dialog_options(),
        )
        if path:
            self.edit_stars_mask_path.setText(os.path.abspath(path))

    def _music_db_file_path(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "astro_music_library.json")

    def _music_assets_dir(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "assets", "music")

    def _is_audio_path(self, path: str) -> bool:
        ext = os.path.splitext(str(path or "").lower())[1]
        return ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac")

    def _load_music_db_entries(self) -> list[dict]:
        entries = []

        assets_dir = self._music_assets_dir()
        if os.path.isdir(assets_dir):
            try:
                for name in sorted(os.listdir(assets_dir)):
                    path = os.path.join(assets_dir, name)
                    if os.path.isfile(path) and self._is_audio_path(path):
                        entries.append({
                            "name": os.path.splitext(name)[0],
                            "path": os.path.abspath(path),
                            "source": "assets",
                        })
            except Exception:
                pass

        db_path = self._music_db_file_path()
        if os.path.exists(db_path):
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        path = str(item.get("path") or "").strip()
                        if not path:
                            continue
                        if not os.path.isabs(path):
                            path = os.path.abspath(os.path.join(os.path.dirname(db_path), path))
                        if not os.path.exists(path) or not self._is_audio_path(path):
                            continue
                        entries.append(
                            {
                                "name": str(item.get("name") or os.path.splitext(os.path.basename(path))[0]),
                                "path": path,
                                "source": "db",
                            }
                        )
            except Exception:
                pass

        dedup = {}
        for entry in entries:
            dedup[os.path.abspath(entry["path"])] = entry
        return list(dedup.values())

    def _save_music_db_entries(self):
        db_path = self._music_db_file_path()
        serializable = []
        for entry in self.music_database_entries:
            path = str(entry.get("path") or "").strip()
            if not path:
                continue
            serializable.append(
                {
                    "name": str(entry.get("name") or os.path.splitext(os.path.basename(path))[0]),
                    "path": os.path.abspath(path),
                }
            )
        try:
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _refresh_music_db_list_widget(self):
        self.music_db_list.clear()
        for entry in self.music_database_entries:
            label = f"{entry.get('name', 'Track')} ({os.path.basename(entry.get('path', ''))})"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, dict(entry))
            self.music_db_list.addItem(item)

    def _refresh_timeline_list_widget(self):
        self.music_timeline_list.clear()
        for idx, entry in enumerate(self.audio_timeline_entries, start=1):
            label = (
                f"{idx}. {entry.get('label', 'Track')} | start={float(entry.get('start', 0.0)):.2f}s"
                f" | vol={float(entry.get('volume', 1.0)):.2f}"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, dict(entry))
            self.music_timeline_list.addItem(item)

    def _reload_music_database(self):
        self.music_database_entries = self._load_music_db_entries()
        self._refresh_music_db_list_widget()

    def _append_timeline_clip(self, path: str, label: str):
        clip = {
            "path": os.path.abspath(path),
            "label": str(label or os.path.splitext(os.path.basename(path))[0]),
            "start": float(self.spin_music_start.value()),
            "volume": float(self.spin_music_volume.value()),
        }
        self.audio_timeline_entries.append(clip)
        self._refresh_timeline_list_widget()

    def _add_selected_db_music_to_timeline(self):
        item = self.music_db_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Muzyka", "Wybierz utwor z bazy.")
            return
        entry = item.data(Qt.UserRole) or {}
        path = str(entry.get("path") or "").strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Muzyka", "Wybrany plik muzyczny nie istnieje.")
            return
        self._append_timeline_clip(path, str(entry.get("name") or os.path.basename(path)))

    def _import_music_from_disk(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import music",
            self._workspace_output_directory(),
            "Audio (*.mp3 *.wav *.flac *.ogg *.m4a *.aac)",
            options=get_safe_file_dialog_options(),
        )
        if not paths:
            return
        existing = {os.path.abspath(item.get("path", "")) for item in self.music_database_entries}
        for path in paths:
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path) or not self._is_audio_path(abs_path):
                continue
            if abs_path not in existing:
                self.music_database_entries.append(
                    {
                        "name": os.path.splitext(os.path.basename(abs_path))[0],
                        "path": abs_path,
                        "source": "import",
                    }
                )
                existing.add(abs_path)
            self._append_timeline_clip(abs_path, os.path.splitext(os.path.basename(abs_path))[0])

        self._save_music_db_entries()
        self._refresh_music_db_list_widget()

    def _remove_selected_timeline_clip(self):
        idx = self.music_timeline_list.currentRow()
        if idx < 0 or idx >= len(self.audio_timeline_entries):
            return
        del self.audio_timeline_entries[idx]
        self._refresh_timeline_list_widget()

    def _clear_music_timeline(self):
        self.audio_timeline_entries = []
        self._refresh_timeline_list_widget()

    def _workspace_output_directory(self) -> str:
        app = self.parent()
        home_folder = str(getattr(app, "home_folder", "") or "").strip() if app is not None else ""
        if home_folder and os.path.isdir(home_folder):
            return home_folder
        image_path = str(getattr(app, "current_image_path", "") or "").strip() if app is not None else ""
        if image_path:
            candidate = os.path.dirname(image_path)
            if candidate and os.path.isdir(candidate):
                return candidate
        return os.getcwd()

    def _refine_cut_mask(self, mask_bool: np.ndarray) -> np.ndarray:
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        if mask_u8.size == 0:
            return mask_u8

        h, w = mask_u8.shape[:2]
        base = max(1, int(round(min(h, w) / 1400.0)))
        kernel_size = max(3, 2 * base + 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        refined = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)

        mask_bin = refined > 0

        hole_min, hole_max = self._hole_size_limits()

        inv = np.logical_not(mask_bin).astype(np.uint8)
        num_holes, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        if num_holes > 1:
            border_labels = set()
            border_labels.update(np.unique(hole_labels[0, :]).tolist())
            border_labels.update(np.unique(hole_labels[-1, :]).tolist())
            border_labels.update(np.unique(hole_labels[:, 0]).tolist())
            border_labels.update(np.unique(hole_labels[:, -1]).tolist())

            for idx in range(1, num_holes):
                if idx in border_labels:
                    continue
                area = int(hole_stats[idx, cv2.CC_STAT_AREA])
                if hole_min <= area <= hole_max:
                    mask_bin[hole_labels == idx] = True

        refined = (mask_bin.astype(np.uint8) * 255)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((refined > 0).astype(np.uint8), connectivity=8)
        if num_labels > 1:
            min_area = max(24, int(mask_u8.size * 0.00001))
            filtered = np.zeros_like(refined)
            for idx in range(1, num_labels):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area >= min_area:
                    filtered[labels == idx] = 255
            if np.any(filtered):
                refined = filtered

        return refined

    def _perform_cut_selection(self):
        if self.combo_zone is None:
            zone_index = int(getattr(self, "current_zone_index", 0))
        else:
            zone_index = int(self.combo_zone.currentData() or 0)
        zone_map = self.zone_widget.get_zone_map()
        mask_preview = np.logical_and(zone_map == zone_index, np.logical_not(self.zone_widget.locked_mask))
        if not np.any(mask_preview):
            QMessageBox.information(self, "Wytnij", "Brak zaznaczenia w aktywnej strefie.")
            return

        src_h, src_w = self.source_image.shape[:2]
        mask_full = cv2.resize(mask_preview.astype(np.uint8), (src_w, src_h), interpolation=cv2.INTER_NEAREST) > 0
        refined_full_u8 = self._refine_cut_mask(mask_full)
        mask_full_refined = refined_full_u8 > 0

        ys, xs = np.where(mask_full_refined)
        if ys.size == 0:
            QMessageBox.information(self, "Wytnij", "Nie udało się wyciąć zaznaczenia.")
            return

        y0, y1 = int(np.min(ys)), int(np.max(ys)) + 1
        x0, x1 = int(np.min(xs)), int(np.max(xs)) + 1
        crop_bgr = self.source_image[y0:y1, x0:x1].copy()
        crop_mask = mask_full_refined[y0:y1, x0:x1]
        cut_rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
        cut_rgba[:, :, 3] = np.where(crop_mask, 255, 0).astype(np.uint8)

        out_dir = self._workspace_output_directory()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"cut_zone_{zone_index + 1}_{stamp}"
        out_path = os.path.join(out_dir, base_name + ".png")
        suffix = 1
        while os.path.exists(out_path):
            out_path = os.path.join(out_dir, f"{base_name}_{suffix}.png")
            suffix += 1

        if not cv2.imwrite(out_path, cut_rgba):
            QMessageBox.warning(self, "Wytnij", "Nie udało się zapisać PNG wycinka.")
            return

        self.source_image[mask_full_refined] = (255, 255, 255)

        preview_h, preview_w = mask_preview.shape[:2]
        refined_preview = cv2.resize(refined_full_u8, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST) > 0
        self.preview_image[refined_preview] = (255, 255, 255)
        self.zone_widget.locked_mask[refined_preview] = True
        self.zone_widget.zone_map[refined_preview] = -1
        self.zone_widget.set_preview_image(self.preview_image)
        self.zone_widget.maskChanged.emit()
        self.zone_widget.update()

        self._append_cut_thumbnail(out_path, cut_rgba)

    def _append_cut_thumbnail(self, out_path: str, cut_rgba: np.ndarray):
        item = QFrame()
        item.setFrameShape(QFrame.StyledPanel)
        item_layout = QVBoxLayout(item)
        item_layout.setContentsMargins(6, 6, 6, 6)
        item_layout.setSpacing(4)

        preview = cut_rgba.copy()
        if preview.shape[2] == 4:
            alpha = preview[:, :, 3:4].astype(np.float32) / 255.0
            white_bg = np.full(preview[:, :, :3].shape, 255, dtype=np.uint8)
            comp = (preview[:, :, :3].astype(np.float32) * alpha + white_bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        else:
            comp = preview[:, :, :3]
        thumb_pix = np_to_qpixmap(comp).scaled(120, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_label = QLabel()
        img_label.setPixmap(thumb_pix)
        name_label = QLabel(os.path.basename(out_path))
        name_label.setWordWrap(True)
        name_label.setMaximumWidth(150)

        item_layout.addWidget(img_label)
        item_layout.addWidget(name_label)
        self.cut_layout.insertWidget(max(0, self.cut_layout.count() - 1), item)
        self.cut_items.append(out_path)
        self._cut_layer_preview_cache.pop(out_path, None)
        self._refresh_clip_timeline_layers(keep_index=True)
        self._refresh_motion_layers(keep_selection=True)

    def _import_cut_layers(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Wczytaj pocięte warstwy",
            self._workspace_output_directory(),
            "Images (*.png *.tif *.tiff *.webp)",
            options=get_safe_file_dialog_options(),
        )
        if not paths:
            return

        existing = {os.path.abspath(p) for p in self.cut_items}
        imported_count = 0
        for path in paths:
            abs_path = os.path.abspath(path)
            if abs_path in existing:
                continue
            layer = cv2.imread(abs_path, cv2.IMREAD_UNCHANGED)
            if layer is None:
                continue
            self._append_cut_thumbnail(abs_path, layer)
            existing.add(abs_path)
            imported_count += 1

        if imported_count > 0:
            self._update_stage_gate()
            QMessageBox.information(self, "Warstwy", f"Wczytano warstwy: {imported_count}")
        else:
            QMessageBox.information(self, "Warstwy", "Nie wczytano nowych warstw.")

    def get_payload(self) -> dict:
        behaviors = self._build_motion_behaviors()
        zone_count = len(behaviors)
        edge_blur_brush_payload = {}
        for key, mask in self.edge_blur_brush_by_layer.items():
            if not isinstance(mask, np.ndarray) or mask.size == 0:
                continue
            if int(np.count_nonzero(mask)) == 0:
                continue
            edge_blur_brush_payload[str(key)] = mask.astype(np.uint8).copy()

        return {
            "zone_map": self.zone_widget.get_zone_map(),
            "zone_count": zone_count,
            "behaviors": behaviors,
            "main_zoom_speed": float(self.spin_main_zoom_speed.value()) if hasattr(self, "spin_main_zoom_speed") else 0.03,
            "edge_blur": self._edge_blur_value(),
            "edge_blur_map": dict(self.edge_blur_by_layer),
            "edge_blur_brush_map": edge_blur_brush_payload,
            "duration": float(self.spin_duration.value()),
            "fps": int(self.spin_fps.value()),
            "stars_overlay": {
                "enabled": bool(self.chk_add_stars.isChecked()) if hasattr(self, "chk_add_stars") else False,
                "count": int(self.spin_stars_count.value()) if hasattr(self, "spin_stars_count") else 900,
                "speed": float(self.spin_stars_speed.value()) if hasattr(self, "spin_stars_speed") else 1.2,
                "near_ratio": float(self.spin_stars_near_ratio.value()) if hasattr(self, "spin_stars_near_ratio") else 0.28,
                "size": float(self.spin_stars_size.value()) if hasattr(self, "spin_stars_size") else 1.0,
                "mask_path": str(self.edit_stars_mask_path.text() or "").strip() if hasattr(self, "edit_stars_mask_path") else "",
            },
            "starless": False,
            "starnet_stride": int(self.spin_starless_stride.value()) if hasattr(self, "spin_starless_stride") else 16,
            "output_path": str(self.edit_output.text() or "").strip(),
            "preview_shape": tuple(self.preview_image.shape[:2]),
            "edited_source_image": self.source_image.copy(),
            "cut_items": list(self.cut_items),
            "audio_timeline": [dict(item) for item in self.audio_timeline_entries],
        }


class PulsingCirclesWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._phase = 0.0
        self.setFixedSize(92, 28)
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

    def sizeHint(self):
        return QSize(92, 28)

    def start(self):
        if not self._timer.isActive():
            self._timer.start()

    def stop(self):
        self._timer.stop()
        self.update()

    def _tick(self):
        self._phase += 0.2
        self.update()

    def _build_star_path(self, cx: float, cy: float, r_outer: float, r_inner: float) -> QPainterPath:
        path = QPainterPath()
        for i in range(10):
            angle = -math.pi / 2.0 + i * (math.pi / 5.0)
            radius = r_outer if (i % 2 == 0) else r_inner
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        return path

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)

        cycle = 12.0
        t = (self._phase % cycle) / cycle

        pulse_base = 0.5 + 0.5 * math.sin(self._phase * 1.25)
        pulse = 0.18 + 0.24 * pulse_base

        orbit_x = 0.0
        orbit_y = 0.0
        if 0.36 <= t < 0.68:
            u = (t - 0.36) / 0.32
            orbit_radius = 4.2
            orbit_angle = 2.0 * math.pi * u
            orbit_x = orbit_radius * math.cos(orbit_angle)
            orbit_y = orbit_radius * math.sin(orbit_angle)
            pulse += 0.10 * math.sin(self._phase * 2.1)

        cx = 18.0 + orbit_x
        cy = 14.0 + orbit_y
        outer = 6.4 + pulse * 3.2
        inner = outer * 0.47

        star_color = QColor(232, 245, 255, int(round(170 + 80 * pulse_base)))
        painter.setBrush(star_color)
        star_path = self._build_star_path(cx, cy, outer, inner)
        painter.drawPath(star_path)


class AltairStartupWidget(QWidget):
    def __init__(self, image_path: str, star_point: tuple[float, float], parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap(image_path)
        self._star_x = float(star_point[0])
        self._star_y = float(star_point[1])
        self._logo_pixmap = None
        self._logo_star_x = self._star_x
        self._logo_star_y = self._star_y
        self._star_origin_x = 18
        self._star_origin_y = 14
        self._prepare_logo_layer(image_path)
        self.setStyleSheet("background: #0f1115;")
        self._star_widget = PulsingCirclesWidget(self)
        self._star_widget.start()
        self._star_widget.hide()
        self._update_star_position()

    def has_image(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull()

    def _prepare_logo_layer(self, image_path: str):
        try:
            raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if raw is None or raw.size == 0:
                return

            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)
            if int(np.count_nonzero(mask)) == 0:
                _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
            points = cv2.findNonZero(mask)
            if points is None:
                return

            x, y, w, h = cv2.boundingRect(points)
            if w <= 2 or h <= 2:
                return

            cropped_bgr = raw[y:y + h, x:x + w]
            cropped_gray = gray[y:y + h, x:x + w]

            rgba = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGBA)
            alpha = np.clip((cropped_gray.astype(np.float32) - 115.0) * 3.4, 0.0, 255.0).astype(np.uint8)
            rgba[:, :, 3] = alpha

            qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
            self._logo_pixmap = QPixmap.fromImage(qimg.copy())
            self._logo_star_x = self._star_x - float(x)
            self._logo_star_y = self._star_y - float(y)
        except Exception:
            self._logo_pixmap = None
            self._logo_star_x = self._star_x
            self._logo_star_y = self._star_y

    def _cover_source_rect(self) -> QRectF:
        if not self.has_image():
            return QRectF()
        src_w = float(self._pixmap.width())
        src_h = float(self._pixmap.height())
        dst_w = float(max(1, self.width()))
        dst_h = float(max(1, self.height()))
        if src_w <= 0.0 or src_h <= 0.0 or dst_w <= 0.0 or dst_h <= 0.0:
            return QRectF()

        src_ratio = src_w / src_h
        dst_ratio = dst_w / dst_h
        if src_ratio > dst_ratio:
            crop_h = src_h
            crop_w = crop_h * dst_ratio
        else:
            crop_w = src_w
            crop_h = crop_w / dst_ratio

        x0 = (src_w - crop_w) * 0.5
        y0 = (src_h - crop_h) * 0.5
        return QRectF(x0, y0, crop_w, crop_h)

    def _fit_target_rect(self) -> QRectF:
        logo = self._logo_pixmap if self._logo_pixmap is not None and not self._logo_pixmap.isNull() else None
        if logo is None:
            return QRectF()
        src_w = float(logo.width())
        src_h = float(logo.height())
        dst_w = float(max(1, self.width()))
        dst_h = float(max(1, self.height()))
        if src_w <= 0.0 or src_h <= 0.0 or dst_w <= 0.0 or dst_h <= 0.0:
            return QRectF()

        scale = min(dst_w / src_w, dst_h / src_h)
        scale *= 0.86
        draw_w = src_w * scale
        draw_h = src_h * scale
        x0 = (dst_w - draw_w) * 0.5
        y0 = (dst_h - draw_h) * 0.5
        return QRectF(x0, y0, draw_w, draw_h)

    def _update_star_position(self):
        if not self.has_image():
            self._star_widget.hide()
            return
        target = self._fit_target_rect()
        if target.isNull() or target.width() <= 0.0 or target.height() <= 0.0:
            self._star_widget.hide()
            return

        logo = self._logo_pixmap if self._logo_pixmap is not None and not self._logo_pixmap.isNull() else None
        if logo is None:
            self._star_widget.hide()
            return

        px = target.left() + (self._logo_star_x / float(max(1, logo.width()))) * target.width()
        py = target.top() + (self._logo_star_y / float(max(1, logo.height()))) * target.height()

        self._star_widget.move(int(round(px - self._star_origin_x)), int(round(py - self._star_origin_y)))
        self._star_widget.show()
        self._star_widget.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_star_position()

    def paintEvent(self, _event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, float(max(1, self.width())), float(max(1, self.height())))
        gradient.setColorAt(0.0, QColor("#141a21"))
        gradient.setColorAt(0.45, QColor("#0f141b"))
        gradient.setColorAt(1.0, QColor("#0a0e14"))
        painter.fillRect(self.rect(), gradient)
        if not self.has_image():
            return
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        logo = self._logo_pixmap if self._logo_pixmap is not None and not self._logo_pixmap.isNull() else self._pixmap
        target = self._fit_target_rect()
        if target.isNull() or target.width() <= 0.0 or target.height() <= 0.0:
            return
        source_logo = QRectF(0.0, 0.0, float(logo.width()), float(logo.height()))
        painter.setOpacity(1.0)
        painter.drawPixmap(target, logo, source_logo)


class AIAssistantPanel(QFrame):
    def __init__(self, parent=None, app=None):
        super().__init__(parent)
        self.app = app
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.startup_splash_duration_ms = 5000
        self.startup_splash_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets",
            "altair",
            "init.png",
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.stacked_layout = QStackedLayout()
        layout.addLayout(self.stacked_layout)

        self.startup_splash_widget = AltairStartupWidget(
            self.startup_splash_path,
            star_point=(708.5, 218.5),
        )
        self.stacked_layout.addWidget(self.startup_splash_widget)

        self.content_widget = QWidget()
        self.stacked_layout.addWidget(self.content_widget)

        content_layout = QVBoxLayout(self.content_widget)
        apply_standard_layout_margins(content_layout)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.chat_scroll.setStyleSheet("QScrollArea { background: #121212; border: 1px solid #2b2b2b; border-radius: 8px; }")
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_layout.setSpacing(8)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_container)
        content_layout.addWidget(self.chat_scroll, 1)

        self.input_text = QTextEdit()
        self.input_text.setFixedHeight(100)
        self.input_text.setPlaceholderText("Type your message here...")
        content_layout.addWidget(self.input_text)

        button_layout = QHBoxLayout()
        self.btn_voice = QPushButton("")
        self.btn_voice.setToolTip("Voice to Text")
        self.btn_voice.setFixedSize(34, 30)
        self.btn_voice.setIcon(svg_icon_from_text(self._microphone_svg(), 18))
        self.btn_voice.setIconSize(QSize(18, 18))
        self.btn_analyze = QPushButton("Analizuj")
        self.btn_send = QPushButton("Send")
        self.btn_send.setProperty("accent", True)
        self.btn_run_ase = QPushButton("Run ASE")

        self.token_usage_bar = CircularProgressBar()
        self.token_usage_bar.setRange(0, 100)
        self.token_usage_bar.setValue(0)
        self.token_usage_bar.setTextVisible(True)
        self.token_usage_bar.setFixedSize(40, 40)
        self.token_usage_bar.setFormat("%p%")
        self.token_usage_bar.setToolTip("Brak danych o limitach tokenów.")

        button_layout.addWidget(self.btn_voice)
        button_layout.addWidget(self.btn_analyze)
        button_layout.addWidget(self.btn_send)
        button_layout.addWidget(self.btn_run_ase)
        button_layout.addWidget(self.token_usage_bar)
        content_layout.addLayout(button_layout)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #a0a0a0;")
        content_layout.addWidget(self.status_label)

        settings_layout = QGridLayout()
        settings_layout.addWidget(QLabel("Gemini model:"), 0, 0)
        self.lbl_gemini_model = QLabel(FIXED_GEMINI_MODEL)
        self.lbl_gemini_model.setStyleSheet("color: #c7d3de;")
        settings_layout.addWidget(self.lbl_gemini_model, 0, 1)
        settings_layout.addWidget(QLabel("ASE:"), 1, 0)
        self.combo_ase_execution = QComboBox()
        self.combo_ase_execution.addItem("Nie wykonuj poleceń", "deny")
        self.combo_ase_execution.addItem("Pytaj o zgodę", "ask")
        self.combo_ase_execution.addItem(
            svg_icon_from_text(self._warning_svg(), 16),
            "Zezwalaj na wykonywanie poleceń",
            "allow",
        )
        self.combo_ase_execution.setCurrentIndex(1)
        settings_layout.addWidget(self.combo_ase_execution, 1, 1)
        content_layout.addLayout(settings_layout)

        self.btn_voice.clicked.connect(self.on_voice_to_text)
        self.btn_analyze.clicked.connect(self.on_analyze)
        self.btn_send.clicked.connect(self.on_send_message)
        self.btn_run_ase.clicked.connect(self.on_run_ase)

        self.last_generated_ase = ""
        self.last_assistant_message = ""
        self.pending_assistant_row = None
        self.pending_indicator_widget = None
        self.pending_analysis_row = None
        self.pending_analysis_indicator_widget = None
        self.stt_worker = None
        self.last_token_reset_at = None
        if not SPEECH_RECOGNITION_AVAILABLE:
            self.btn_voice.setToolTip("Voice to Text wymaga pakietu SpeechRecognition")
        self._init_startup_splash()

    def _init_startup_splash(self):
        if not os.path.exists(self.startup_splash_path):
            self.stacked_layout.setCurrentWidget(self.content_widget)
            return

        if not self.startup_splash_widget.has_image():
            self.stacked_layout.setCurrentWidget(self.content_widget)
            return

        self.stacked_layout.setCurrentWidget(self.startup_splash_widget)
        QTimer.singleShot(self.startup_splash_duration_ms, self._show_chat_panel)

    def _show_chat_panel(self):
        self.stacked_layout.setCurrentWidget(self.content_widget)


    def _microphone_svg(self) -> str:
        return """
        <svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\">
          <g fill=\"none\" stroke=\"#d8e8f7\" stroke-width=\"1.9\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
            <rect x=\"9\" y=\"3\" width=\"6\" height=\"11\" rx=\"3\"/>
            <path d=\"M6 10v1a6 6 0 0 0 12 0v-1\"/>
            <path d=\"M12 17v4\"/>
            <path d=\"M9 21h6\"/>
          </g>
        </svg>
        """

    def _warning_svg(self) -> str:
        return """
        <svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\">
          <path d=\"M12 3.2 2.7 19.3h18.6z\" fill=\"none\" stroke=\"#f8e36d\" stroke-width=\"1.7\" stroke-linejoin=\"round\"/>
          <path d=\"M12 8.2v6.3\" stroke=\"#f8e36d\" stroke-width=\"1.9\" stroke-linecap=\"round\"/>
          <circle cx=\"12\" cy=\"17.6\" r=\"1.1\" fill=\"#f8e36d\"/>
        </svg>
        """

    def _resolve_stt_language_code(self) -> str:
        lang = str(getattr(self.app, "language", "pl") or "pl").strip().lower()
        if lang.startswith("en"):
            return "en-US"
        return "pl-PL"

    def on_voice_to_text(self):
        if self.stt_worker is not None and self.stt_worker.isRunning():
            return

        self.btn_voice.setEnabled(False)
        self.status_label.setText("[Voice to text: listening...]")

        self.stt_worker = SpeechToTextWorker(language_code=self._resolve_stt_language_code())
        self.stt_worker.finished_signal.connect(self.on_voice_to_text_finished)
        self.stt_worker.start()

    def on_voice_to_text_finished(self, text: str, error: str):
        self.btn_voice.setEnabled(True)
        self.status_label.setText("")

        if error:
            self.append_system_message(f"Voice to text error: {error}")
            return

        recognized = str(text or "").strip()
        if not recognized:
            self.append_system_message("Voice to text: brak rozpoznanego tekstu.")
            return

        current = self.input_text.toPlainText().strip()
        merged = f"{current} {recognized}".strip() if current else recognized
        self.input_text.setPlainText(merged)
        cursor = self.input_text.textCursor()
        cursor.movePosition(cursor.End)
        self.input_text.setTextCursor(cursor)
        self.input_text.setFocus()

    def append_system_message(self, text: str):
        self._append_message("System", text)

    def append_user_message(self, text: str):
        self._append_message("User", text)

    def append_assistant_message(self, text: str):
        self.last_assistant_message = str(text or "")
        self._append_message(AI_ASSISTANT_NAME, text)

    def _append_message(self, role: str, text: str):
        role_key = (role or AI_ASSISTANT_NAME).strip().lower()
        is_user = role_key == "user"

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        bubble = QFrame()
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(10, 8, 10, 8)
        bubble_layout.setSpacing(4)

        role_label = QLabel(role)
        role_label.setStyleSheet("color: #a9b0b7; font-size: 10px; font-weight: 600;")
        bubble_layout.addWidget(role_label)

        message_label = QLabel(text)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble_layout.addWidget(message_label)

        if is_user:
            bubble.setStyleSheet("QFrame { background: #0d4f6b; border: 1px solid #16739b; border-radius: 12px; }")
            message_label.setStyleSheet("color: #eaf6ff; font-size: 12px;")
            row_layout.addStretch(1)
            row_layout.addWidget(bubble, 0)
        else:
            bubble.setStyleSheet("QFrame { background: #1f2228; border: 1px solid #343944; border-radius: 12px; }")
            message_label.setStyleSheet("color: #f0f2f4; font-size: 12px;")
            row_layout.addWidget(bubble, 0)
            row_layout.addStretch(1)

        self.chat_layout.insertWidget(max(0, self.chat_layout.count() - 1), row)
        self._scroll_chat_to_bottom()

    def _clear_chat_messages(self):
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _scroll_chat_to_bottom(self):
        scrollbar = self.chat_scroll.verticalScrollBar()
        QTimer.singleShot(0, lambda: scrollbar.setValue(scrollbar.maximum()))

    def _show_assistant_typing_indicator(self):
        if self.pending_assistant_row is not None:
            return

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        circles = PulsingCirclesWidget()
        row_layout.addWidget(circles, 0)
        row_layout.addStretch(1)

        self.chat_layout.insertWidget(max(0, self.chat_layout.count() - 1), row)
        self.pending_assistant_row = row
        self.pending_indicator_widget = circles
        circles.start()
        self._scroll_chat_to_bottom()

    def _hide_assistant_typing_indicator(self):
        if self.pending_indicator_widget is not None:
            self.pending_indicator_widget.stop()
            self.pending_indicator_widget = None

        row = self.pending_assistant_row
        self.pending_assistant_row = None
        if row is not None:
            row.deleteLater()

    def _show_analysis_chat_indicator(self):
        if self.pending_analysis_row is not None:
            return

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        star_indicator = PulsingCirclesWidget()
        row_layout.addWidget(star_indicator, 0)
        row_layout.addStretch(1)

        self.chat_layout.insertWidget(max(0, self.chat_layout.count() - 1), row)
        self.pending_analysis_row = row
        self.pending_analysis_indicator_widget = star_indicator
        star_indicator.start()
        self._scroll_chat_to_bottom()

    def _hide_analysis_chat_indicator(self):
        if self.pending_analysis_indicator_widget is not None:
            self.pending_analysis_indicator_widget.stop()
            self.pending_analysis_indicator_widget = None

        row = self.pending_analysis_row
        self.pending_analysis_row = None
        if row is not None:
            row.deleteLater()

    def _fallback_token_reset_datetime(self) -> datetime:
        now = datetime.now().astimezone()
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return next_midnight

    def _format_reset_datetime(self, dt: datetime) -> str:
        if dt is None:
            return "brak danych"
        try:
            return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _update_token_usage_progress(self, token_meta: dict = None, error_message: str = ""):
        token_meta = token_meta if isinstance(token_meta, dict) else {}
        budget = token_meta.get("token_budget") if isinstance(token_meta.get("token_budget"), dict) else {}
        usage = token_meta.get("usage") if isinstance(token_meta.get("usage"), dict) else {}
        model_used = str(token_meta.get("model_used") or budget.get("model") or FIXED_GEMINI_MODEL)
        quota_reached = _is_gemini_quota_error(error_message)

        input_limit = budget.get("input_token_limit")
        remaining = budget.get("remaining_input_tokens_estimate")
        prompt_est = budget.get("prompt_tokens_estimate")
        prompt_used = usage.get("prompt_token_count")
        output_used = usage.get("output_token_count")
        total_used = usage.get("total_token_count")

        used_for_progress = prompt_used if prompt_used is not None else prompt_est
        progress_value = 0
        if input_limit is not None and used_for_progress is not None and int(input_limit) > 0:
            progress_value = int(max(0, min(100, round(100.0 * float(used_for_progress) / float(input_limit)))))
        elif quota_reached:
            progress_value = 100
            if remaining is None:
                remaining = 0

        self.token_usage_bar.setValue(progress_value)
        self.token_usage_bar.setFormat("%p%")

        retry_seconds = _extract_retry_seconds_from_error(error_message) if error_message else None
        if retry_seconds is not None:
            self.last_token_reset_at = datetime.now().astimezone() + timedelta(seconds=float(retry_seconds))
        elif self.last_token_reset_at is None:
            self.last_token_reset_at = self._fallback_token_reset_datetime()

        reset_text = self._format_reset_datetime(self.last_token_reset_at)
        tooltip_lines = [f"Model: {model_used}"]
        if input_limit is not None:
            tooltip_lines.append(f"Limit wejściowy: {int(input_limit)}")
        if used_for_progress is not None:
            tooltip_lines.append(f"Użyte (prompt): {int(used_for_progress)}")
        if remaining is not None:
            tooltip_lines.append(f"Pozostało (szacunek): {int(remaining)}")
        elif quota_reached:
            tooltip_lines.append("Pozostało (szacunek): 0")
        if output_used is not None:
            tooltip_lines.append(f"Użyte (output): {int(output_used)}")
        if total_used is not None:
            tooltip_lines.append(f"Użyte (łącznie): {int(total_used)}")
        tooltip_lines.append(f"Reset limitu: {reset_text}")
        if retry_seconds is None:
            tooltip_lines.append("Reset to szacunek (Gemini API nie zwraca stałej daty resetu).")
        if error_message:
            tooltip_lines.append(f"Ostatni błąd: {error_message}")
        if quota_reached:
            tooltip_lines.append("Wykryto osiągnięcie limitu tokenów (quota reached).")
        self.token_usage_bar.setToolTip("\n".join(tooltip_lines))

    def on_send_message(self):
        message = self.input_text.toPlainText().strip()
        if not message:
            return
        self.append_user_message(message)
        self.input_text.clear()

        self.app.gemini_model = FIXED_GEMINI_MODEL
        save_config(
            self.app.denoise_model_path,
            self.app.bg_removal_model_path,
            self.app.dark_mode,
            self.app.starnet_path,
            self.app.plate_solve_api_key,
            self.app.plate_solve_pixel_size_um,
            self.app.plate_solve_focal_length_mm,
            self.app.starnet_stride,
            self.app.gemini_api_key,
            self.app.gemini_model,
        )

        payload = self.get_context_payload()
        if self.app.gemini_api_key:
            self.status_label.setText("")
            self._show_assistant_typing_indicator()
            self._update_token_usage_progress()
            self.btn_send.setEnabled(False)
            self.btn_run_ase.setEnabled(False)
            self.gemini_worker = GeminiWorker(
                message,
                payload,
                self.app.gemini_model,
                self.app.gemini_api_key,
            )
            self.gemini_worker.finished_signal.connect(self.on_gemini_finished)
            self.gemini_worker.start()
        else:
            response = self.generate_ai_response(message, payload)
            self.append_assistant_message(response)
            code = self._extract_ase_from_text(response)
            if code:
                self.last_generated_ase = code

    def on_analyze(self):
        if self.app is None:
            self.append_assistant_message("Brak aktywnej instancji aplikacji.")
            return

        image = self.app._get_image_for_analysis() if hasattr(self.app, "_get_image_for_analysis") else None
        if image is None:
            self.append_assistant_message("Brak obrazu do analizy.")
            return

        self.btn_analyze.setEnabled(False)
        self.status_label.setText("")
        self._show_analysis_chat_indicator()
        QApplication.processEvents()

        try:
            result = self.app.run_image_analysis(log_result=True)
        finally:
            self._hide_analysis_chat_indicator()
            self.status_label.setText("")
            self.btn_analyze.setEnabled(True)

        if not result:
            self.append_assistant_message("Brak obrazu do analizy.")
            return

        stars = result.get("stars", {}) or {}
        luminance = result.get("luminance", {}) or {}
        backend = result.get("analysis_backend", {}) or {}
        star_count = int(stars.get("count") or 0)
        fwhm_value = stars.get("fwhm_px_median")
        snr_value = stars.get("snr_median")
        noise = luminance.get("background_sigma")
        method = str(backend.get("stars_method") or "unknown")
        if fwhm_value is None:
            self.append_system_message(
                f"Analiza gotowa. Gwiazdy: {star_count}, FWHM: brak wiarygodnego pomiaru, SNR median: {float(snr_value or 0.0):.2f}, sigma tła: {float(noise or 0.0):.2f}, metoda: {method}."
            )
            return

        self.append_system_message(
            f"Analiza gotowa. Gwiazdy: {star_count}, FWHM median: {float(fwhm_value):.2f}px, SNR median: {float(snr_value or 0.0):.2f}, sigma tła: {float(noise or 0.0):.2f}, metoda: {method}."
        )

    def on_gemini_finished(self, response: str, error: str, token_meta: dict):
        self.btn_send.setEnabled(True)
        self.btn_run_ase.setEnabled(True)
        self.status_label.setText("")
        self._hide_assistant_typing_indicator()
        if error:
            self._update_token_usage_progress(token_meta=token_meta, error_message=error)
            self.append_assistant_message(f"Błąd Gemini: {error}")
        else:
            self._update_token_usage_progress(token_meta=token_meta)
            self.append_assistant_message(response)
            code = self._extract_ase_from_text(response)
            if code:
                self.last_generated_ase = code

    def on_clear(self):
        self._hide_assistant_typing_indicator()
        self._hide_analysis_chat_indicator()
        self._clear_chat_messages()
        self.status_label.setText("")
        self.last_generated_ase = ""
        self.last_assistant_message = ""

    def shutdown_workers(self):
        self._hide_assistant_typing_indicator()
        self._hide_analysis_chat_indicator()
        self.status_label.setText("")

        for attr_name in ("gemini_worker", "stt_worker"):
            worker = getattr(self, attr_name, None)
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    worker.requestInterruption()
                    if not worker.wait(1500):
                        worker.terminate()
                        worker.wait(1000)
            except Exception:
                pass
            setattr(self, attr_name, None)

    def on_run_ase(self):
        code = self._extract_ase_from_text(self.last_assistant_message)
        if not code:
            code = self.last_generated_ase.strip()
        if not code:
            self.append_assistant_message("Brak wygenerowanego ASE.")
            return

        ase_policy = self._get_ase_execution_policy()
        if ase_policy == "deny":
            self.append_assistant_message("ASE zablokowane przez ustawienie: 'Nie wykonuj poleceń'.")
            return

        valid, error = self.validate_ase_code(code)
        if not valid:
            self.append_assistant_message(f"ASE nieprawidĹ‚owy: {error}")
            return

        if ase_policy == "ask":
            decision = QMessageBox.question(
                self,
                "Potwierdzenie ASE",
                "Altair chce wykonać polecenia ASE. Czy zezwalasz?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if decision != QMessageBox.Yes:
                self.append_assistant_message("Wykonanie ASE anulowane przez użytkownika.")
                return

        self.status_label.setText("Uruchamiam ASE...")
        for raw_line in code.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            self.execute_ase_line(line)
        self.status_label.setText("")
        self.append_assistant_message("ASE wykonano.")

    def _get_ase_execution_policy(self) -> str:
        combo = getattr(self, "combo_ase_execution", None)
        if combo is None:
            return "ask"
        value = combo.currentData()
        if value in ("deny", "ask", "allow"):
            return str(value)
        return "ask"

    def _extract_ase_from_text(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""

        fenced = re.search(r"```(?:ase|txt|python)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
        candidate = fenced.group(1).strip() if fenced else raw

        lines = []
        for line in candidate.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if self._parse_ase_command(stripped) is not None:
                lines.append(stripped)

        return "\n".join(lines).strip()

    def execute_ase_line(self, line: str):
        parsed = self._parse_ase_command(line)
        if parsed is None:
            if hasattr(self.app, "_execute_script_line"):
                self.app._execute_script_line(line)
            else:
                self.app.execute_console_command(line)
            return

        cmd_name, args = parsed
        normalized = cmd_name.lower()

        ase_to_console = {
            "open.curves": "open.curves",
            "open.levels": "open.levels",
            "open.histogram": "open.histogram",
            "open.correction": "open.correction",
            "open.calibration": "open.calibration",
            "open.console": "open.console",
            "open.menu": "open.menu",
            "run.magic": "run.magic",
            "run.starnet": "run.starnet",
            "run.starnet++": "run.starnet++",
            "run.deepsnr": "run.deepsnr",
            "open.blur": "open.blur",
            "save": "save",
            "undo": "undo",
            "redo": "redo",
            "platesolve": "plate solve",
            "platesolving": "plate solve",
            "solveplate": "plate solve",
            "blur": "blur",
            "gaussianblur": "gaussian blur",
            "gaussian": "gaussian",
            "levels": "levels",
            "level": "level",
            "menu": "menu",
            "console": "console",
            "curves": "curves",
            "curve": "curve",
            "lut": "lut",
            "curveslut": "curves lut",
            "curvesreset": "curves reset",
            "lutreset": "lut reset",
            "histogram": "histogram",
            "hist": "hist",
            "correction": "correction",
            "correct": "correct",
            "calibration": "calibration",
            "bn": "bn",
            "backgroundneutralization": "background neutralization",
            "cameraraw": "camera raw",
            "reset": "reset",
            "resetsliders": "reset sliders",
            "darkon": "dark on",
            "darkoff": "dark off",
            "models": "models",
            "deepsnr": "deepsnr",
            "exit": "exit",
            "quit": "quit",
            "help": "help",
        }

        if normalized == "saveas":
            if args:
                path = self._parse_ase_string(args)
                if path is not None:
                    self.app.save_image_to_path(path)
                else:
                    self.append_assistant_message("ASE nieprawidĹ‚owy: niepoprawny argument SaveAs().")
            else:
                self.app.save_image_as()
            return

        if normalized in ase_to_console:
            self.app.execute_console_command(ase_to_console[normalized])
            return

        if hasattr(self.app, "_execute_script_line"):
            self.app._execute_script_line(line)
        else:
            self.app.execute_console_command(line)

    def _parse_ase_command(self, line: str):
        match = re.match(r"^\s*([A-Za-z0-9_.+]+)\((.*)\)\s*$", line)
        if not match:
            return None
        name = match.group(1).strip()
        args = match.group(2).strip()
        return name, args

    def _parse_ase_string(self, value: str):
        if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
            return value[1:-1]
        return None

    def generate_ase_code(self, message: str, payload: dict) -> str:
        text = message.lower()
        commands = []
        if "magic" in text or "denoise" in text or "background" in text:
            commands.append("Run.Magic()")
        if "star" in text and ("remove" in text or "removal" in text or "starnet" in text):
            commands.append("Run.StarNet()")
        if "deepsnr" in text or "deep snr" in text:
            commands.append("Run.DeepSNR()")
        if "blur" in text:
            commands.append("Open.Blur()")
        if "curves" in text or "lut" in text:
            commands.append("Open.Curves()")
        if "levels" in text:
            commands.append("Open.Levels()")
        if "histogram" in text:
            commands.append("Open.Histogram()")
        if "correction" in text or "camera raw" in text:
            commands.append("Open.Correction()")
        if "calibration" in text:
            commands.append("Open.Calibration()")
        if "background neutralization" in text or "background neutralisation" in text:
            commands.append("BN()")
        if "console" in text:
            commands.append("Open.Console()")
        if "menu" in text and "open" in text:
            commands.append("Open.Menu()")
        if "plate solve" in text or "solve" in text:
            commands.append("PlateSolve()")
        if "save" in text and "as" not in text:
            commands.append("Save()")
        if "save as" in text or "save_as" in text:
            commands.append("SaveAs(\"output.tif\")")
        if "reset" in text and "sliders" in text:
            commands.append("ResetSliders()")
        if "dark on" in text:
            commands.append("DarkOn()")
        if "dark off" in text:
            commands.append("DarkOff()")
        if "models" in text:
            commands.append("Models()")
        if not commands:
            return "# unavailable"
        return "\n".join(dict.fromkeys(commands))

    def validate_ase_code(self, code: str) -> tuple[bool, str]:
        allowed_commands = {
            "open.curves", "open.levels", "open.histogram", "open.correction", "open.calibration",
            "open.console", "open.menu", "run.magic", "run.starnet", "run.starnet++", "run.deepsnr",
            "open.blur", "save", "saveas", "undo", "redo", "platesolve",
            "solveplate", "blur", "gaussianblur", "gaussian", "levels", "level",
            "menu", "console", "curves", "curve", "lut", "curveslut",
            "curvesreset", "lutreset", "histogram", "hist", "correction", "correct", "calibration", "bn", "backgroundneutralization",
            "cameraraw", "reset", "resetsliders", "darkon", "darkoff", "models", "deepsnr",
            "exit", "quit", "help"
        }
        for raw_line in code.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = self._parse_ase_command(line)
            if parsed is None:
                return False, f"Niepoprawny format ASE: {line}"
            cmd_name, args = parsed
            lower_cmd = cmd_name.lower()
            if lower_cmd not in allowed_commands:
                return False, f"Nieznana komenda ASE: {cmd_name}"
            if lower_cmd == "saveas":
                if not args:
                    continue
                if self._parse_ase_string(args) is None:
                    return False, "NieprawidĹ‚owy argument SaveAs()."
                continue
            if args:
                return False, f"Komenda {cmd_name} nie przyjmuje argumentĂłw." 
        return True, ""

    def get_context_payload(self) -> dict:
        analysis_metrics = {}
        if self.app is not None and hasattr(self.app, "get_or_run_image_analysis"):
            analysis_metrics = self.app.get_or_run_image_analysis() or {}

        payload = {
            "assistant_name": AI_ASSISTANT_NAME,
            "current_file": self.app.current_image_path or self.app.current_save_path or "Untitled",
            "image_size": {
                "width": self.app.original_img.shape[1] if self.app.original_img is not None else None,
                "height": self.app.original_img.shape[0] if self.app.original_img is not None else None,
            },
            "active_layers": [layer for layer, visible in self.app.layer_visibility.items() if visible],
            "plate_solve_info": self.app.plate_solve_object_info or {},
            "plate_solve_result": self.app.latest_plate_solve_result or {},
            "analysis_metrics": analysis_metrics,
            "available_ase_commands": [
                "Open.Curves()", "Open.Levels()", "Open.Histogram()", "Open.Correction()",
                "Open.Console()", "Open.Menu()", "Run.Magic()", "Run.StarNet()", "Run.StarNet++()", "Run.DeepSNR()",
                "Open.Blur()", "Save()", "SaveAs(\"output.tif\")", "Undo()", "Redo()", "PlateSolve()",
                "DarkOn()", "DarkOff()", "Models()", "Exit()", "Quit()", "Help()",
                "Reset()", "ResetSliders()", "CurvesReset()", "LutReset()"
            ],
        }
        return payload

    def generate_ai_response(self, message: str, payload: dict) -> str:
        text = message.lower().strip()
        greetings = ("hej", "hi", "hello", "czeĹ›Ä‡")
        if text in greetings or re.fullmatch(r"^(hej|hi|hello|czeĹ›Ä‡)[\s!Âˇ]*$", text):
            return "CzeĹ›Ä‡. W czym mogÄ™ pomĂłc?"

        if any(keyword in text for keyword in ["kim jeste", "jak sie nazywasz", "jak masz na imie", "twoje imie", "who are you", "your name"]):
            return f"Jestem {AI_ASSISTANT_NAME}, asystent analizy astrofotografii."

        plate_info = payload.get("plate_solve_info", {}) or {}
        plate_result = payload.get("plate_solve_result", {}) or {}

        if "show payload" in text or "analysis payload" in text or "json" in text or "data" in text:
            return "Aktualny payload analizy:\n" + json.dumps(payload, indent=2)

        if any(keyword in text for keyword in ["object", "messier", "ngc", "ic", "designation"]):
            main = plate_info.get("main_object")
            catalog = plate_info.get("catalog")
            designation = plate_info.get("designation")
            if main:
                return f"Wykryty obiekt: {main} ({catalog or 'brak katalogu'}). Designacja: {designation or main}."
            return "Brak identyfikacji obiektu. Uruchom plate solve lub wczytaj metadane."

        if "ra" in text or "dec" in text:
            ra = plate_info.get("ra") if plate_info else plate_result.get("ra")
            dec = plate_info.get("dec") if plate_info else plate_result.get("dec")
            if ra is not None and dec is not None:
                return f"WspĂłĹ‚rzÄ™dne plate solve: RA = {ra}, Dec = {dec}."
            return "WspĂłĹ‚rzÄ™dne plate solve sÄ… niedostÄ™pne. Najpierw uruchom plate solving."

        if "plate solve" in text or "solve" in text:
            if plate_result:
                return "Dane plate solve sÄ… dostÄ™pne. Zapytaj o obiekt, wspĂłĹ‚rzÄ™dne lub dalsze kroki."
            return "Brak wyniku plate solve. UĹĽyj narzÄ™dzia Plate Solving i sprĂłbuj ponownie."

        if any(keyword in text for keyword in ["next", "recommend", "suggest", "advice"]):
            return self.build_recommendation(payload)

        if any(keyword in text for keyword in ["exposure", "gain", "temperature", "filter", "noise", "histogram", "fwhm", "fhwm", "star", "gwiazd"]):
            return self.build_metadata_hint(text, payload)

        return "Jestem gotowy do analizy dostÄ™pnych danych. Zadaj pytanie lub wygeneruj ASE, jeĹ›li potrzebujesz automatyzacji."

    def build_recommendation(self, payload: dict) -> str:
        plate_info = payload.get("plate_solve_info", {}) or {}
        suggestions = []
        if not plate_info:
            suggestions.append("Run plate solving first to identify the field and verify RA/DEC.")
        else:
            suggestions.append("Use identified object metadata to confirm the target and catalog designation.")
            suggestions.append("Check the plate solve overlay if available to verify star alignment.")
        if payload.get("image_size", {}).get("width") is None:
            suggestions.append("Load a valid image file to enable analysis.")
        return " ".join(suggestions)

    def build_metadata_hint(self, text: str, payload: dict) -> str:
        analysis = payload.get("analysis_metrics", {}) or {}
        stars = analysis.get("stars", {}) or {}
        luminance = analysis.get("luminance", {}) or {}

        if "exposure" in text:
            return "The current app does not yet expose explicit exposure metadata in the assistant payload. Use FITS header data or manual entry to supply exposure, gain, and temperature information."
        if "gain" in text:
            return "Gain information is not currently available in the context payload. If the image is FITS, ensure the header metadata is loaded into the analysis payload."
        if "temperature" in text:
            return "Temperature data is not currently present in the assistant payload. Add FITS header parsing to the analysis engine to include it."
        if "filter" in text:
            return "Filter metadata is not currently available here. Please ensure FITS header or manual filter data is included in the analysis summary."
        if "fwhm" in text or "fhwm" in text or "star" in text or "gwiazd" in text:
            if stars.get("fwhm_px_median") is None:
                return "FWHM nie jest obecnie dostępne. Użyj przycisku Analizuj i upewnij się, że obraz zawiera punktowe gwiazdy."
            return (
                f"Wynik analizy gwiazd: liczba={int(stars.get('count') or 0)}, "
                f"FWHM median={float(stars.get('fwhm_px_median') or 0.0):.2f}px, "
                f"SNR median={float(stars.get('snr_median') or 0.0):.2f}."
            )
        if "noise" in text or "histogram" in text:
            if not analysis:
                return "Brak danych o histogramie/szumie. Użyj przycisku Analizuj."
            return (
                f"Sigma tła={float(luminance.get('background_sigma') or 0.0):.2f}, "
                f"p01={float(luminance.get('p01') or 0.0):.2f}, p99={float(luminance.get('p99') or 0.0):.2f}, "
                f"black clip={float(luminance.get('black_clipping_pct') or 0.0):.2f}%, "
                f"white clip={float(luminance.get('white_clipping_pct') or 0.0):.2f}%."
            )
        return "I can answer general processing recommendations when the analysis payload includes more image metadata."


class PlateSolveWorker(QThread):
    progress_signal = pyqtSignal(str, int, int)
    finished_signal = pyqtSignal(object, str)

    def __init__(self, img, pixel_size_um, focal_length_mm, api_key=None):
        super().__init__()
        self.img = img
        self.pixel_size_um = pixel_size_um
        self.focal_length_mm = focal_length_mm
        self.api_key = api_key

    def run(self):
        try:
            if self.img is None:
                raise RuntimeError("No image provided for plate solving.")

            def progress(stage, overall, current):
                self.progress_signal.emit(stage, overall, current)

            self.progress_signal.emit("Preparing image...", 0, 0)
            scale = self._compute_scale()
            self.progress_signal.emit("Scale estimated...", 10, 10)

            local_solver = self._find_local_solver()
            if local_solver is not None:
                result = self._solve_with_local_solver(local_solver, scale)
            elif self.api_key:
                result = self._solve_with_astrometry_net(scale)
            else:
                raise RuntimeError("No local solver found and no Astrometry.net API key provided.")

            self.finished_signal.emit(result, "")
        except Exception as exc:
            self.finished_signal.emit(None, str(exc))

    def _compute_scale(self):
        if self.pixel_size_um <= 0 or self.focal_length_mm <= 0:
            raise RuntimeError("Pixel size and focal length must be positive values.")
        return 206.265 * self.pixel_size_um / self.focal_length_mm

    def _find_local_solver(self):
        for name in ("solve-field", "solve-field.exe"):
            path = shutil.which(name)
            if path:
                return path
        return None

    def _solve_with_local_solver(self, solver_path, scale_arcsec):
        temp_dir = tempfile.mkdtemp(prefix="astro_plate_solve_")
        try:
            input_path = os.path.join(temp_dir, "input.jpg")
            cv2.imwrite(input_path, self.img)
            scale_low = max(0.05, scale_arcsec * 0.7)
            scale_high = scale_arcsec * 1.3
            command = [
                solver_path,
                input_path,
                "--overwrite",
                "--no-plots",
                "--scale-units", "arcsecperpix",
                "--scale-low", f"{scale_low:.8f}",
                "--scale-high", f"{scale_high:.8f}",
                "--cpulimit", "60",
                "--dir", temp_dir,
                "--verbose",
            ]
            self.progress_signal.emit("Running local solver...", 20, 20)
            proc = subprocess.run(command, cwd=temp_dir, capture_output=True, text=True, timeout=900)
            if proc.returncode != 0:
                raise RuntimeError(f"Local solve-field failed: {proc.stderr.strip() or proc.stdout.strip()}")

            input_base = os.path.splitext(os.path.basename(input_path))[0]
            wcs_candidates = [
                os.path.join(temp_dir, f"{input_base}.wcs"),
                os.path.join(temp_dir, "plate_solve.wcs"),
            ]
            existing_wcs = [path for path in wcs_candidates if os.path.exists(path)]
            if not existing_wcs:
                existing_wcs = [
                    os.path.join(temp_dir, name)
                    for name in os.listdir(temp_dir)
                    if name.lower().endswith(".wcs")
                ]
            if not existing_wcs:
                raise RuntimeError("Local solver did not produce a WCS file.")
            wcs_path = max(existing_wcs, key=os.path.getmtime)

            axy_candidates = [
                os.path.join(temp_dir, f"{input_base}.axy"),
                os.path.join(temp_dir, "plate_solve.axy"),
            ]
            existing_axy = [path for path in axy_candidates if os.path.exists(path)]
            if not existing_axy:
                existing_axy = [
                    os.path.join(temp_dir, name)
                    for name in os.listdir(temp_dir)
                    if name.lower().endswith(".axy")
                ]
            axy_path = max(existing_axy, key=os.path.getmtime) if existing_axy else ""

            header = self._parse_wcs_header(wcs_path)
            points = self._parse_axy_points(axy_path)
            return self._build_result(header, points)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _solve_with_astrometry_net(self, scale_arcsec):
        self.progress_signal.emit("Login...", 0, 0)
        login_url = "https://nova.astrometry.net/api/login"
        session = self._post_json(login_url, {"apikey": self.api_key})

        print("LOGIN RESPONSE:", session)

        if session.get("status") != "success":
            raise Exception(
                f"Astrometry.net login failed: {session}"
            )

        session_key = session.get("session")
        if not session_key:
            raise Exception(
                "No session key returned"
            )

        self.progress_signal.emit("Uploading image...", 0, 0)
        upload_url = "https://nova.astrometry.net/api/upload"
        upload_temp_dir = tempfile.mkdtemp(prefix="astro_plate_upload_")
        temp_path = os.path.join(upload_temp_dir, "input.jpg")
        try:
            cv2.imwrite(temp_path, self.img)

            upload_fields = {
                "allow_commercial_use": "n",
                "allow_modifications": "n",
                "scale_units": "arcsecperpix",
                "scale_lower": f"{scale_arcsec * 0.7:.8f}",
                "scale_upper": f"{scale_arcsec * 1.3:.8f}",
            }
            if session_key:
                upload_fields["session"] = session_key
            else:
                upload_fields["apikey"] = self.api_key

            response = self._post_multipart(upload_url, upload_fields, "file", temp_path)
        finally:
            shutil.rmtree(upload_temp_dir, ignore_errors=True)
        print("SUBMISSION RESPONSE:", response)
        if response.get("status") != "success":
            raise RuntimeError(f"Astrometry.net upload failed: {response}")

        submission_id = response.get("submission_id") or response.get("subid")
        if submission_id is None:
            raise RuntimeError(f"Astrometry.net upload failed to return a submission id. Response: {response}")

        self.progress_signal.emit("✓ Job Created", 0, 0)
        job_id = None
        status = None
        submission_max_attempts = 60
        submission_poll_interval = 3
        for attempt in range(submission_max_attempts):
            time.sleep(submission_poll_interval)
            status = self._get_json(f"https://nova.astrometry.net/api/submissions/{submission_id}")
            self.progress_signal.emit(
                f"⏳ Waiting for job id... Attempt {attempt + 1} / {submission_max_attempts}",
                0,
                attempt + 1,
            )
            if attempt < 3 or (attempt + 1) % 10 == 0:
                print(f"SUBMISSION STATUS (attempt {attempt + 1}):", status)
            jobs = status.get("jobs") if isinstance(status, dict) else None
            if isinstance(jobs, list):
                for job in jobs:
                    if job is not None:
                        job_id = job
                        break
            if job_id is not None:
                break
        if job_id is None:
            raise RuntimeError(
                f"Astrometry.net did not return a job id in time. Submission status: {status}"
            )

        job_info = None
        info = {}
        job_status = None
        solve_max_attempts = 96
        solve_poll_interval = 5
        for attempt in range(solve_max_attempts):
            time.sleep(solve_poll_interval)
            info = self._get_json(f"https://nova.astrometry.net/api/jobs/{job_id}/info/")
            job_status = (info.get("status") or "").lower() if isinstance(info, dict) else ""
            self.progress_signal.emit(
                f"⏳ Solving image... Attempt {attempt + 1} / {solve_max_attempts}",
                0,
                attempt + 1,
            )
            if attempt < 3 or (attempt + 1) % 10 == 0:
                print(f"JOB INFO (attempt {attempt + 1}):", info)
            if job_status in ("success", "solved"):
                job_info = info
                break
            if job_status in ("failure", "failed", "error"):
                raise RuntimeError(f"Astrometry.net job failed. Info: {info}")
            try:
                job_probe = self._get_json(f"https://nova.astrometry.net/api/jobs/{job_id}")
            except Exception:
                job_probe = {}
            probe_status = (job_probe.get("status") or "").lower() if isinstance(job_probe, dict) else ""
            calibration_data = job_probe.get("calibration") if isinstance(job_probe, dict) else None
            if probe_status in ("success", "solved") or isinstance(calibration_data, dict):
                job_info = info if isinstance(info, dict) else {}
                break
            if probe_status in ("failure", "failed", "error"):
                raise RuntimeError(f"Astrometry.net job failed. Job response: {job_probe}")
        if job_info is None:
            raise RuntimeError(
                f"Astrometry.net job timed out after {solve_max_attempts * solve_poll_interval}s. Last info: {info}"
            )

        job = self._get_json(f"https://nova.astrometry.net/api/jobs/{job_id}")
        print("JOB RESPONSE:", job)
        calibration_response = None
        try:
            calibration_response = self._get_json(f"https://nova.astrometry.net/api/jobs/{job_id}/calibration")
            print("CALIBRATION RESPONSE:", calibration_response)
        except Exception as e:
            print(f"CALIBRATION RESPONSE ERROR: {e}")

        wcs_url = f"https://nova.astrometry.net/api/jobs/{job_id}/wcs_file"
        wcs_text = self._get_url_text(wcs_url)
        header = self._parse_wcs_header_text(wcs_text)
        points = []
        return self._build_result(
            header,
            points,
            job_info=job_info,
            job=job,
            calibration=calibration_response,
        )

    def _parse_wcs_header(self, path):
        try:
            from astropy.io import fits
            with fits.open(path, memmap=False) as hdul:
                if hdul and getattr(hdul[0], "header", None) is not None:
                    header = {}
                    for key in hdul[0].header.keys():
                        if key:
                            header[key] = hdul[0].header[key]
                    if header:
                        return header
        except Exception:
            pass

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return self._parse_wcs_header_text(f.read())

    def _parse_wcs_header_text(self, text):
        header = {}
        for line in text.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.split("/")[0].strip()
            if value.startswith("'") and value.endswith("'"):
                header[key] = value.strip("'")
                continue
            try:
                header[key] = float(value)
            except Exception:
                header[key] = value
        return header

    def _parse_axy_points(self, path):
        points = []
        if not path or not os.path.exists(path):
            return points

        try:
            from astropy.io import fits
        except ImportError:
            return points

        candidate_pairs = [
            ("X", "Y"),
            ("x", "y"),
            ("XIMAGE", "YIMAGE"),
            ("ximage", "yimage"),
            ("X_IMAGE", "Y_IMAGE"),
            ("field_x", "field_y"),
        ]

        try:
            with fits.open(path, memmap=False) as hdul:
                table_data = None
                for hdu in hdul:
                    data = getattr(hdu, "data", None)
                    names = list(getattr(data, "names", []) or [])
                    if data is not None and names:
                        table_data = data
                        break

                if table_data is None:
                    return points

                columns = list(getattr(table_data, "names", []) or [])
                lowered = {name.lower(): name for name in columns}
                x_key = None
                y_key = None

                for cand_x, cand_y in candidate_pairs:
                    if cand_x in columns and cand_y in columns:
                        x_key, y_key = cand_x, cand_y
                        break
                    lx = lowered.get(cand_x.lower())
                    ly = lowered.get(cand_y.lower())
                    if lx and ly:
                        x_key, y_key = lx, ly
                        break

                if not x_key or not y_key:
                    return points

                for x_val, y_val in zip(table_data[x_key], table_data[y_key]):
                    try:
                        x_float = float(x_val)
                        y_float = float(y_val)
                        if not (math.isfinite(x_float) and math.isfinite(y_float)):
                            continue
                        points.append((int(round(x_float)), int(round(y_float))))
                    except Exception:
                        continue
        except Exception as exc:
            print(f"Could not parse AXY points from {path}: {exc}")
        return points

    def _build_result(self, header, points, job_info=None, job=None, calibration=None):
        def first_value(source, keys):
            if not isinstance(source, dict):
                return None
            for key in keys:
                value = source.get(key)
                if value is not None:
                    return value
            return None

        calibration_data = {}
        if isinstance(job, dict) and isinstance(job.get("calibration"), dict):
            calibration_data = job.get("calibration", {}) or {}
        elif isinstance(calibration, dict):
            calibration_data = calibration
        elif isinstance(job_info, dict) and isinstance(job_info.get("calibration"), dict):
            calibration_data = job_info.get("calibration", {}) or {}

        ra = first_value(calibration_data, ["ra", "RA", "center_ra", "CRVAL1", "crval1"])
        if ra is None:
            ra = first_value(job_info, ["ra", "RA", "center_ra", "CRVAL1", "crval1"])
        if ra is None:
            ra = first_value(job, ["ra", "RA", "center_ra", "CRVAL1", "crval1"])
        if ra is None:
            ra = header.get("CRVAL1")

        dec = first_value(calibration_data, ["dec", "DEC", "center_dec", "CRVAL2", "crval2"])
        if dec is None:
            dec = first_value(job_info, ["dec", "DEC", "center_dec", "CRVAL2", "crval2"])
        if dec is None:
            dec = first_value(job, ["dec", "DEC", "center_dec", "CRVAL2", "crval2"])
        if dec is None:
            dec = header.get("CRVAL2")

        scale = first_value(calibration_data, ["pixscale", "pixel_scale", "scale", "xpixscale", "ypixscale"])
        if scale is None:
            scale = first_value(job_info, ["pixscale", "pixel_scale", "scale", "xpixscale", "ypixscale"])
        if scale is None:
            scale = first_value(job, ["pixscale", "pixel_scale", "scale", "xpixscale", "ypixscale"])

        rotation = first_value(calibration_data, ["orientation", "Orientation", "rotation", "Rotation", "posang", "pos_angle", "angle"])
        if rotation is None:
            rotation = first_value(job_info, ["orientation", "Orientation", "rotation", "Rotation", "posang", "pos_angle", "angle"])
        if rotation is None:
            rotation = first_value(job, ["orientation", "Orientation", "rotation", "Rotation", "posang", "pos_angle", "angle"])

        cd11 = header.get("CD1_1", header.get("CDELT1", 0.0))
        cd12 = header.get("CD1_2", 0.0)
        cd21 = header.get("CD2_1", 0.0)
        cd22 = header.get("CD2_2", header.get("CDELT2", 0.0))
        scale_x = math.hypot(cd11, cd12) * 3600.0
        scale_y = math.hypot(cd21, cd22) * 3600.0

        if scale is None:
            scale = (scale_x + scale_y) / 2.0 if scale_x > 0 and scale_y > 0 else self._compute_scale()
        else:
            try:
                scale = float(scale)
            except Exception:
                scale = (scale_x + scale_y) / 2.0 if scale_x > 0 and scale_y > 0 else self._compute_scale()

        if rotation is None:
            rotation = math.degrees(math.atan2(cd12, cd11)) if cd11 else 0.0
        else:
            try:
                rotation = float(rotation)
            except Exception:
                rotation = math.degrees(math.atan2(cd12, cd11)) if cd11 else 0.0

        objects = []
        if isinstance(job, dict):
            objects = job.get("objects_in_field") or job.get("objects") or []
            if not isinstance(objects, list):
                objects = []
        elif isinstance(job_info, dict):
            objects = job_info.get("objects_in_field") or job_info.get("objects") or []
            if not isinstance(objects, list):
                objects = []

        h, w = self.img.shape[:2]
        fov_x = w * scale / 3600.0
        fov_y = h * scale / 3600.0
        return {
            "ra": ra,
            "dec": dec,
            "rotation": rotation,
            "scale": scale,
            "fov_x": fov_x,
            "fov_y": fov_y,
            "points": points,
            "wcs_header": header,
            "objects_in_field": objects,
            "calibration_data": calibration_data,
        }

    def _post_json(self, url, data):
        payload = urllib.parse.urlencode({
            "request-json": json.dumps(data)
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Referer": "https://nova.astrometry.net/api/login",
                "Content-Type": "application/x-www-form-urlencoded"
            }
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            text = resp.read().decode("utf-8")

        print("ASTROMETRY RESPONSE:", text)

        return json.loads(text)

    def _post_multipart(self, url, fields, file_field, file_path):
        boundary = f"----AstroAiProBoundary{int(time.time() * 1000)}"
        body = bytearray()

        request_json = json.dumps(fields)
        body.extend((f"--{boundary}\r\n").encode("utf-8"))
        body.extend((f"Content-Disposition: form-data; name=\"request-json\"\r\n\r\n").encode("utf-8"))
        body.extend((request_json + "\r\n").encode("utf-8"))

        filename = os.path.basename(file_path)
        body.extend((f"--{boundary}\r\n").encode("utf-8"))
        body.extend((f"Content-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\n").encode("utf-8"))
        body.extend(("Content-Type: application/octet-stream\r\n\r\n").encode("utf-8"))
        with open(file_path, "rb") as f:
            body.extend(f.read())
        body.extend(("\r\n").encode("utf-8"))
        body.extend((f"--{boundary}--\r\n").encode("utf-8"))
        req = urllib.request.Request(url, data=bytes(body), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            response = json.load(resp)

        print("ASTROMETRY UPLOAD RESPONSE:", response)
        return response

    def _get_json(self, url):
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        try:
            return json.loads(text)
        except Exception as e:
            print(f"Astrometry JSON parse error for {url}: {e}")
            print(f"Response text: {text}")
            raise

    def _get_url_text(self, url):
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read().decode("utf-8", errors="ignore")


def _load_altair_3d_fly_guide_text(max_chars: int = 6000) -> str:
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, ALTAIR_3D_FLY_GUIDE_FILE)
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return ""
        if len(content) > max_chars:
            return content[:max_chars].rstrip() + "\n..."
        return content
    except Exception:
        return ""


def build_gemini_system_prompt() -> str:
    base_prompt = (
        f"Nazywasz się {AI_ASSISTANT_NAME}. "
        "Jesteś ekspertem od analizy zdjęć astronomicznych. Otrzymujesz tylko metadane i wyniki analizy w formacie JSON. "
        "Nie interpretuj surowych pikseli obrazu ani nie zakładaj dodatkowych danych poza tym, co jest w payload. "
        "Odpowiadaj krótko i naturalnie po polsku. "
        "Na powitanie daj krótki komunikat bez podsumowywania payload. "
        "Jeśli użytkownik pyta o filtr 3D FLY, najpierw opieraj odpowiedź o lokalną instrukcję 3d_fly_help.md. "
        "Jeśli użytkownik poprosi o ASE, wygeneruj tylko kod ASE w formie funkcji, używając wyłącznie istniejących poleceń aplikacji."
    )
    guide = _load_altair_3d_fly_guide_text()
    if not guide:
        return base_prompt
    return base_prompt + "\n\nInstrukcja 3D FLY (źródło lokalne):\n" + guide


def extract_gemini_text(response_data: dict) -> str:
    if not isinstance(response_data, dict):
        return ""

    if "candidates" in response_data and isinstance(response_data["candidates"], list):
        candidate = response_data["candidates"][0] if response_data["candidates"] else {}
    elif "outputs" in response_data and isinstance(response_data["outputs"], list):
        candidate = response_data["outputs"][0] if response_data["outputs"] else {}
    else:
        candidate = response_data

    if isinstance(candidate, dict):
        if "output" in candidate and isinstance(candidate["output"], str):
            return candidate["output"].strip()
        if "text" in candidate and isinstance(candidate["text"], str):
            return candidate["text"].strip()
        content = candidate.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip()

    if isinstance(response_data.get("outputText"), str):
        return response_data.get("outputText").strip()

    if isinstance(response_data.get("response"), str):
        return response_data.get("response").strip()

    return json.dumps(response_data, indent=2)


def _is_gemini_quota_error(message: str) -> bool:
    text = str(message or "").lower()
    return (
        "resource_exhausted" in text
        or "quota exceeded" in text
        or "quota reached" in text
        or "chat quota reached" in text
        or "qouta reached" in text
        or "429" in text
    )


def _extract_retry_seconds_from_error(message: str):
    text = str(message or "")
    match = re.search(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    match = re.search(r"retryDelay'?:\s*'([0-9]+)s'", text)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def _int_or_none(value):
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _extract_gemini_response_usage(response) -> dict:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}

    prompt_tokens = _int_or_none(getattr(usage, "prompt_token_count", None))
    output_tokens = _int_or_none(getattr(usage, "candidates_token_count", None))
    total_tokens = _int_or_none(getattr(usage, "total_token_count", None))
    out = {}
    if prompt_tokens is not None:
        out["prompt_token_count"] = prompt_tokens
    if output_tokens is not None:
        out["output_token_count"] = output_tokens
    if total_tokens is not None:
        out["total_token_count"] = total_tokens
    return out


def _estimate_gemini_token_budget(client, model_name: str, full_prompt: str) -> dict:
    info = {
        "model": model_name,
    }

    input_limit = None
    output_limit = None
    try:
        model_info = client.models.get(model=model_name)
        input_limit = _int_or_none(getattr(model_info, "input_token_limit", None))
        output_limit = _int_or_none(getattr(model_info, "output_token_limit", None))
    except Exception:
        model_info = None

    if input_limit is not None:
        info["input_token_limit"] = input_limit
    if output_limit is not None:
        info["output_token_limit"] = output_limit

    prompt_tokens = None
    try:
        count_result = client.models.count_tokens(model=model_name, contents=full_prompt)
        prompt_tokens = _int_or_none(getattr(count_result, "total_tokens", None))
    except Exception:
        pass

    if prompt_tokens is not None:
        info["prompt_tokens_estimate"] = prompt_tokens
        if input_limit is not None:
            info["remaining_input_tokens_estimate"] = max(0, int(input_limit - prompt_tokens))

    return info


def call_gemini_api(message: str, payload: dict, model: str, api_key: str) -> tuple[str, dict]:
    if not api_key:
        raise RuntimeError("Missing Gemini API key.")

    if not model:
        raise RuntimeError("Missing Gemini model.")

    system_prompt = build_gemini_system_prompt()

    user_prompt = (
        f"User request:\n{message}\n\n"
        "Analysis payload:\n"
        f"{json.dumps(payload, indent=2)}"
    )
    full_prompt = f"""
SYSTEM:
{system_prompt}

USER:
{user_prompt}
"""

    try:
        from google import genai as google_genai
        client = google_genai.Client(api_key=api_key)

        def _request(model_name: str) -> tuple[str, dict]:
            token_budget = _estimate_gemini_token_budget(client, model_name, full_prompt)
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
            )
            text = getattr(response, "text", "") if response is not None else ""
            if isinstance(text, str) and text.strip():
                return text.strip(), {
                    "model_used": model_name,
                    "token_budget": token_budget,
                    "usage": _extract_gemini_response_usage(response),
                }
            raise RuntimeError("Gemini returned empty response.")

        try:
            return _request(model)
        except Exception as primary_error:
            primary_message = str(primary_error)
            if _is_gemini_quota_error(primary_message):
                fallback_models = [
                    "gemini-2.5-flash",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                ]
                for fallback_model in fallback_models:
                    if fallback_model == model:
                        continue
                    try:
                        return _request(fallback_model)
                    except Exception:
                        continue

                retry_seconds = _extract_retry_seconds_from_error(primary_message)
                if retry_seconds is not None:
                    raise RuntimeError(
                        f"Gemini quota exceeded for model '{model}'. Retry in about {int(round(retry_seconds))}s."
                    ) from primary_error
                raise RuntimeError(
                    f"Gemini quota exceeded for model '{model}'."
                ) from primary_error

            raise RuntimeError(f"Gemini API error: {primary_message}") from primary_error

    except ImportError as e:
        raise RuntimeError(
            "Gemini SDK not found. Install 'google-genai'."
        ) from e
    except Exception:
        raise
    
class GeminiWorker(QThread):
    finished_signal = pyqtSignal(str, str, object)

    def __init__(self, message: str, payload: dict, model: str, api_key: str):
        super().__init__()
        self.message = message
        self.payload = payload
        self.model = model
        self.api_key = api_key

    def run(self):
        try:
            result, token_meta = call_gemini_api(self.message, self.payload, self.model, self.api_key)
            self.finished_signal.emit(result, "", token_meta)
        except Exception as e:
            self.finished_signal.emit("", str(e), {})


class SpeechToTextWorker(QThread):
    finished_signal = pyqtSignal(str, str)

    def __init__(self, language_code: str = "pl-PL", timeout_sec: float = 6.0, phrase_time_limit_sec: float = 20.0):
        super().__init__()
        self.language_code = str(language_code or "pl-PL")
        self.timeout_sec = float(max(1.0, timeout_sec))
        self.phrase_time_limit_sec = float(max(2.0, phrase_time_limit_sec))

    def run(self):
        if not SPEECH_RECOGNITION_AVAILABLE or sr is None:
            self.finished_signal.emit(
                "",
                "Pakiet SpeechRecognition nie jest zainstalowany.",
            )
            return

        try:
            recognizer = sr.Recognizer()
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.8

            with sr.Microphone(sample_rate=16000) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
                audio = recognizer.listen(
                    source,
                    timeout=self.timeout_sec,
                    phrase_time_limit=self.phrase_time_limit_sec,
                )

            text = recognizer.recognize_google(audio, language=self.language_code)
            self.finished_signal.emit(str(text or "").strip(), "")
        except sr.WaitTimeoutError:
            self.finished_signal.emit("", "Nie wykryto mowy w czasie nasłuchiwania.")
        except sr.UnknownValueError:
            self.finished_signal.emit("", "Nie udało się rozpoznać wypowiedzi.")
        except sr.RequestError as exc:
            self.finished_signal.emit("", f"Błąd usługi rozpoznawania mowy: {exc}")
        except OSError as exc:
            error_text = str(exc or "")
            if "Could not find PyAudio" in error_text:
                error_text = "Brak PyAudio (mikrofon). Zainstaluj pyaudio lub portaudio dla Voice to Text."
            elif "No Default Input Device Available" in error_text:
                error_text = "Brak domyślnego urządzenia wejściowego audio (mikrofon)."
            self.finished_signal.emit("", error_text)
        except Exception as exc:
            error_text = str(exc or "")
            if not error_text:
                error_text = "Nieznany błąd Voice to Text."
            self.finished_signal.emit("", f"{error_text}")


class MagicWorker(QThread):

    progress_signal = pyqtSignal(str, int, int)
    finished_signal = pyqtSignal(object)

    def __init__(self, img, denoise_path, bg_path):
        super().__init__()

        self.img = img
        self.denoise_path = denoise_path
        self.bg_path = bg_path

    def run(self):

        def callback(stage, overall, current):
            self.progress_signal.emit(stage, overall, current)

        result = magic_pipeline(
            self.img,
            self.denoise_path,
            self.bg_path,
            progress_callback=callback
        )

        self.finished_signal.emit(result)


def _resolve_starnet_command(starnet_path: str = None):
    explicit = str(starnet_path or "").strip()
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend([
        "starnet2",
        "starnet2.exe",
        "starnet++",
        "starnet++.exe",
        "starnet",
        "starnet.exe",
    ])

    seen = set()
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)

        if os.path.isabs(candidate) or os.path.sep in candidate:
            resolved = candidate if os.path.exists(candidate) else ""
        else:
            resolved = shutil.which(candidate) or ""
        if not resolved:
            continue

        name = os.path.basename(resolved).lower()
        mode = "starnet2" if "starnet2" in name else "legacy"
        if resolved.lower().endswith(".py"):
            base_command = [sys.executable, resolved]
        else:
            base_command = [resolved]
        command_dir = os.path.dirname(resolved) or None
        return base_command, command_dir, mode

    return None, None, "starnet2"


def _build_starnet_command(base_command: list, mode: str, input_path: str, output_path: str, stride: int) -> list:
    stride_value = int(stride)
    stride_value = max(2, min(512, stride_value))
    if stride_value % 2 != 0:
        stride_value += 1
        if stride_value > 512:
            stride_value = 512

    if str(mode) == "legacy":
        return list(base_command) + [input_path, output_path, "--stride", str(stride_value)]

    return list(base_command) + ["-i", input_path, "-o", output_path, "-s", str(stride_value), "-u", "-q"]


class StarNetWorker(QThread):
    progress_signal = pyqtSignal(str, int, int)
    finished_signal = pyqtSignal(object, str)

    def __init__(self, img, starnet_path, stride=16):
        super().__init__()
        self.img = img
        self.starnet_path = starnet_path
        self.stride = int(stride)

    @staticmethod
    def _extract_progress_percent(text: str):
        if not text:
            return None
        matches = re.findall(r"(\d{1,3})\s*%", str(text))
        if not matches:
            return None
        for raw in reversed(matches):
            try:
                value = int(raw)
            except Exception:
                continue
            if 0 <= value <= 100:
                return value
        return None

    def run(self):
        temp_dir = tempfile.mkdtemp(prefix="astro_starnet_")
        input_path = os.path.join(temp_dir, "input.tif")
        output_path = os.path.join(temp_dir, "starless.tif")

        try:
            self.progress_signal.emit("Preparing StarNet++ input...", 10, 20)
            if not cv2.imwrite(input_path, self._to_starnet_input(self.img)):
                self.finished_signal.emit(None, "Could not write temporary StarNet++ input.")
                return

            self.progress_signal.emit("Running StarNet++...", 35, 40)
            base_command, command_dir, mode = _resolve_starnet_command(self.starnet_path)
            if not base_command:
                self.finished_signal.emit(None, "StarNet not found. Install starnet2 in PATH or select executable path.")
                return
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            start_time = time.time()
            command = _build_starnet_command(base_command, mode, input_path, output_path, self.stride)
            output_lines = []
            live_percent = -1
            proc = subprocess.Popen(
                command,
                cwd=command_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            timeout_sec = 1800
            while True:
                if time.time() - start_time > timeout_sec:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    raise subprocess.TimeoutExpired(command, timeout_sec)

                line = ""
                if proc.stdout is not None:
                    line = proc.stdout.readline()

                if line:
                    output_lines.append(line)
                    percent = self._extract_progress_percent(line)
                    if percent is not None and percent > live_percent:
                        live_percent = percent
                        overall = int(round(35.0 + (90.0 - 35.0) * (float(percent) / 100.0)))
                        self.progress_signal.emit(f"Running StarNet++... {percent}%", overall, int(percent))
                    continue

                if proc.poll() is not None:
                    break
                time.sleep(0.03)

            result_stdout = "".join(output_lines)
            result = subprocess.CompletedProcess(command, int(proc.returncode or 0), stdout=result_stdout, stderr="")

            if result.returncode != 0:
                details = (result.stdout or "").strip()
                message = f"StarNet++ failed with code {result.returncode}."
                if details:
                    message += f" {details}"
                self.finished_signal.emit(None, message)
                return

            self.progress_signal.emit("Loading StarNet++ result...", 90, 80)
            output_candidates = self._find_output_candidates(temp_dir, output_path, start_time, command_dir)
            result_img = None
            for candidate in output_candidates:
                if os.path.exists(candidate):
                    result_img = normalize_to_uint8_bgr(safe_imread(candidate))
                    if result_img is not None:
                        break

            if result_img is None:
                details = (result.stdout or "").strip()
                message = "StarNet++ finished, but no output image was found."
                if details:
                    message += f" Output: {details}"
                self.finished_signal.emit(None, message)
                return

            self.progress_signal.emit("StarNet++ complete.", 100, 100)
            self.finished_signal.emit(result_img, "")

        except subprocess.TimeoutExpired:
            self.finished_signal.emit(None, "StarNet++ timed out after 30 minutes.")
        except Exception as e:
            self.finished_signal.emit(None, f"StarNet++ error: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _to_starnet_input(self, img):
        img = normalize_to_uint8_bgr(img)
        return img.astype(np.uint16) * 257

    def _find_output_candidates(self, temp_dir, output_path, start_time, command_dir=None):
        image_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        candidates = [
            output_path,
            os.path.join(temp_dir, "input_starless.tif"),
            os.path.join(temp_dir, "input_starnet.tif"),
            os.path.join(temp_dir, "input_s.tif"),
            os.path.join(temp_dir, "starless.tif"),
        ]

        search_dirs = [temp_dir, command_dir]
        for directory in search_dirs:
            if not directory or not os.path.isdir(directory):
                continue
            try:
                for name in os.listdir(directory):
                    path = os.path.join(directory, name)
                    if not os.path.isfile(path):
                        continue
                    if not name.lower().endswith(image_exts):
                        continue
                    if os.path.abspath(path) == os.path.abspath(os.path.join(temp_dir, "input.tif")):
                        continue
                    if os.path.getmtime(path) >= start_time - 2:
                        candidates.append(path)
            except Exception:
                pass

        seen = set()
        unique_candidates = []
        for candidate in candidates:
            key = os.path.abspath(candidate)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
        return unique_candidates


def run_starnet_sync_for_image(img: np.ndarray, starnet_path: str = None, stride: int = 16):
    temp_dir = tempfile.mkdtemp(prefix="astro_starnet_sync_")
    input_path = os.path.join(temp_dir, "input.tif")
    output_path = os.path.join(temp_dir, "starless.tif")
    try:
        data = normalize_to_uint8_bgr(img)
        if data is None:
            return None, "Invalid image for StarNet++."
        input_16 = data.astype(np.uint16) * 257
        if not cv2.imwrite(input_path, input_16):
            return None, "Could not write temporary StarNet++ input."

        base_command, command_dir, mode = _resolve_starnet_command(starnet_path)
        if not base_command:
            return None, "StarNet not found. Install starnet2 in PATH or select executable path."

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        command = _build_starnet_command(base_command, mode, input_path, output_path, int(stride))
        result = subprocess.run(
            command,
            cwd=command_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            creationflags=creationflags,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            msg = f"StarNet++ failed with code {result.returncode}."
            if details:
                msg += f" {details}"
            return None, msg

        worker = StarNetWorker(data, starnet_path, stride=int(stride))
        output_candidates = worker._find_output_candidates(temp_dir, output_path, time.time(), command_dir)
        for candidate in output_candidates:
            if os.path.exists(candidate):
                parsed = normalize_to_uint8_bgr(safe_imread(candidate))
                if parsed is not None:
                    return parsed, ""

        return None, "StarNet++ completed, but output image was not found."
    except subprocess.TimeoutExpired:
        return None, "StarNet++ timed out after 30 minutes."
    except Exception as exc:
        return None, f"StarNet++ error: {exc}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _warp_layer_with_motion(image: np.ndarray, alpha: np.ndarray, move_x: float, move_y: float, zoom_delta: float):
    h, w = image.shape[:2]
    center = (w * 0.5, h * 0.5)
    scale = max(0.2, 1.0 + float(zoom_delta))
    matrix = cv2.getRotationMatrix2D(center, 0.0, scale)
    matrix[0, 2] += float(move_x)
    matrix[1, 2] += float(move_y)

    warped_img = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    warped_alpha = cv2.warpAffine(
        alpha,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_alpha = np.clip(warped_alpha, 0.0, 1.0)
    return warped_img, warped_alpha


@dataclass
class ImageLayer:
    image: np.ndarray
    alpha: np.ndarray
    mask: np.ndarray
    depth: float
    move_x: float
    move_y: float
    zoom: float
    layer_key: str


def _inpaint_background_without_layers(base_image: np.ndarray, occupied_mask: np.ndarray) -> np.ndarray:
    if not isinstance(base_image, np.ndarray) or base_image.size == 0:
        return base_image
    if not isinstance(occupied_mask, np.ndarray) or occupied_mask.size == 0:
        return base_image.copy()

    remove = np.where(occupied_mask, 255, 0).astype(np.uint8)
    if int(np.count_nonzero(remove)) == 0:
        return base_image.copy()

    # Slightly dilate to avoid thin halos of removed objects.
    kernel = np.ones((3, 3), dtype=np.uint8)
    remove = cv2.dilate(remove, kernel, iterations=1)
    return cv2.inpaint(base_image, remove, 3, cv2.INPAINT_TELEA)


def _count_mask_overlap_pixels(masks: list[np.ndarray]) -> int:
    if not masks:
        return 0
    shape = masks[0].shape[:2]
    hits = np.zeros(shape, dtype=np.uint16)
    for mask in masks:
        if not isinstance(mask, np.ndarray) or mask.shape[:2] != shape:
            continue
        hits += (mask > 0)
    return int(np.count_nonzero(hits > 1))


def _normalize_audio_timeline(entries) -> list[dict]:
    out = []
    if not isinstance(entries, list):
        return out
    for item in entries:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path or not os.path.exists(path):
            continue
        out.append(
            {
                "path": path,
                "label": str(item.get("label") or os.path.basename(path)),
                "start": max(0.0, float(item.get("start") or 0.0)),
                "volume": float(np.clip(float(item.get("volume") or 1.0), 0.0, 3.0)),
            }
        )
    return out


def attach_audio_timeline_with_ffmpeg(video_path: str, timeline_entries: list[dict], duration: float) -> tuple[str, str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return video_path, "ffmpeg not found; video exported without audio timeline."

    entries = _normalize_audio_timeline(timeline_entries)
    if not entries:
        return video_path, ""

    tmp_output = os.path.splitext(video_path)[0] + "_audio_tmp.mp4"
    final_output = video_path

    cmd = [ffmpeg_bin, "-y", "-i", video_path]
    for entry in entries:
        cmd.extend(["-i", entry["path"]])

    filter_parts = []
    mix_inputs = []
    max_duration = max(0.1, float(duration))
    for idx, entry in enumerate(entries, start=1):
        start_ms = int(round(max(0.0, float(entry["start"])) * 1000.0))
        volume = float(entry["volume"])
        label = f"a{idx}"
        part = (
            f"[{idx}:a]aresample=44100,atrim=0:{max_duration:.3f},"
            f"asetpts=PTS-STARTPTS,volume={volume:.3f},adelay={start_ms}|{start_ms}[{label}]"
        )
        filter_parts.append(part)
        mix_inputs.append(f"[{label}]")

    if len(mix_inputs) == 1:
        filter_parts.append(f"{mix_inputs[0]}atrim=0:{max_duration:.3f}[aout]")
    else:
        filter_parts.append(
            f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest:dropout_transition=0,"
            f"atrim=0:{max_duration:.3f}[aout]"
        )

    filter_complex = ";".join(filter_parts)
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            tmp_output,
        ]
    )

    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1200,
            creationflags=creationflags,
        )
        if result.returncode != 0 or not os.path.exists(tmp_output):
            details = (result.stderr or result.stdout or "").strip()
            message = "Could not attach audio timeline via ffmpeg."
            if details:
                message += f" {details}"
            return video_path, message

        os.replace(tmp_output, final_output)
        return final_output, ""
    except Exception as exc:
        return video_path, f"Audio timeline mix failed: {exc}"
    finally:
        if os.path.exists(tmp_output) and tmp_output != final_output:
            try:
                os.remove(tmp_output)
            except Exception:
                pass


class Fly3DWorker(QThread):
    progress_signal = pyqtSignal(str, int, int)
    finished_signal = pyqtSignal(str, str)

    def __init__(self, source_img: np.ndarray, payload: dict, starnet_path: str = None, starnet_stride: int = 16):
        super().__init__()
        self.source_img = normalize_to_uint8_bgr(source_img)
        self.payload = dict(payload or {})
        self.starnet_path = starnet_path
        self.starnet_stride = int(starnet_stride)

    def _build_layer_from_cut(self, base: np.ndarray, path: str):
        if not path or not os.path.exists(path):
            return None
        cut = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if cut is None or cut.ndim < 3:
            return None

        if cut.shape[2] >= 4:
            cut_rgb = cut[:, :, :3]
            cut_alpha = cut[:, :, 3]
        else:
            cut_rgb = cut[:, :, :3]
            gray = cv2.cvtColor(cut_rgb, cv2.COLOR_BGR2GRAY)
            cut_alpha = np.where(gray < 250, 255, 0).astype(np.uint8)

        if int(np.count_nonzero(cut_alpha)) == 0:
            return None

        h, w = base.shape[:2]
        gray_base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        gray_cut = cv2.cvtColor(cut_rgb, cv2.COLOR_BGR2GRAY)
        mask_u8 = np.where(cut_alpha > 0, 255, 0).astype(np.uint8)

        top_left = (0, 0)
        try:
            match = cv2.matchTemplate(gray_base, gray_cut, cv2.TM_SQDIFF, mask=mask_u8)
            top_left = cv2.minMaxLoc(match)[2]
        except Exception:
            match = cv2.matchTemplate(gray_base, gray_cut, cv2.TM_CCOEFF_NORMED)
            top_left = cv2.minMaxLoc(match)[3]

        x0 = int(np.clip(top_left[0], 0, max(0, w - cut_rgb.shape[1])))
        y0 = int(np.clip(top_left[1], 0, max(0, h - cut_rgb.shape[0])))
        x1 = min(w, x0 + cut_rgb.shape[1])
        y1 = min(h, y0 + cut_rgb.shape[0])
        if x1 <= x0 or y1 <= y0:
            return None

        src_w = x1 - x0
        src_h = y1 - y0
        src_rgb = cut_rgb[:src_h, :src_w]
        src_mask = cut_alpha[:src_h, :src_w] > 0

        layer_image = np.zeros_like(base)
        layer_image[y0:y1, x0:x1][src_mask] = src_rgb[src_mask]

        layer_mask = np.zeros((h, w), dtype=bool)
        layer_mask[y0:y1, x0:x1] = src_mask
        placement = {
            "x0": int(x0),
            "y0": int(y0),
            "x1": int(x1),
            "y1": int(y1),
            "src_mask": src_mask.astype(np.uint8),
        }
        return layer_image, layer_mask, placement

    def _build_starfield_config(self, width: int, height: int) -> dict:
        stars_payload = self.payload.get("stars_overlay") or {}
        enabled = bool(stars_payload.get("enabled", False))
        if not enabled:
            return {"enabled": False}

        mask_path = str(stars_payload.get("mask_path") or "").strip()
        if not mask_path or not os.path.exists(mask_path):
            return {"enabled": False}

        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            return {"enabled": False}

        if mask_img.ndim == 2:
            mask_gray = mask_img
        elif mask_img.ndim == 3 and mask_img.shape[2] >= 4:
            mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGRA2GRAY)
        else:
            mask_gray = cv2.cvtColor(mask_img[:, :, :3], cv2.COLOR_BGR2GRAY)

        if mask_gray.shape[0] != int(height) or mask_gray.shape[1] != int(width):
            mask_gray = cv2.resize(mask_gray, (int(width), int(height)), interpolation=cv2.INTER_AREA)

        candidates = np.where(mask_gray > 8)
        if candidates[0].size == 0:
            return {"enabled": False}

        count = int(stars_payload.get("count") or 0)
        count = max(32, min(5000, count))
        speed = float(stars_payload.get("speed") or 1.0)
        speed = max(0.05, min(5.0, speed))
        near_ratio = float(stars_payload.get("near_ratio") or 0.25)
        near_ratio = max(0.05, min(0.9, near_ratio))
        size_gain = float(stars_payload.get("size") or 1.0)
        size_gain = max(0.5, min(3.0, size_gain))

        seed = int(self.payload.get("stars_seed") or 472911)
        rng = np.random.default_rng(seed)
        center_x = float(width) * 0.5
        center_y = float(height) * 0.5
        max_radius = float(np.hypot(center_x, center_y)) * 1.14

        y_all = candidates[0].astype(np.int32)
        x_all = candidates[1].astype(np.int32)
        intensities = mask_gray[y_all, x_all].astype(np.float32)

        if y_all.size > count:
            weights = np.clip(intensities, 1.0, None).astype(np.float64)
            weights_sum = float(np.sum(weights))
            if weights_sum > 0.0:
                prob = weights / weights_sum
                chosen_idx = rng.choice(y_all.size, size=count, replace=False, p=prob)
            else:
                chosen_idx = rng.choice(y_all.size, size=count, replace=False)
            y_sel = y_all[chosen_idx]
            x_sel = x_all[chosen_idx]
            intensities = intensities[chosen_idx]
        else:
            y_sel = y_all
            x_sel = x_all

        count = int(y_sel.size)
        if count <= 0:
            return {"enabled": False}

        x_float = x_sel.astype(np.float32)
        y_float = y_sel.astype(np.float32)
        dx = x_float - center_x
        dy = y_float - center_y
        start_radius = np.sqrt(dx * dx + dy * dy).astype(np.float32)

        tiny = start_radius < 1.0
        if np.any(tiny):
            random_angles = rng.uniform(0.0, math.pi * 2.0, int(np.count_nonzero(tiny))).astype(np.float32)
            dx[tiny] = np.cos(random_angles)
            dy[tiny] = np.sin(random_angles)
            start_radius[tiny] = 1.0

        angles = np.arctan2(dy, dx).astype(np.float32)
        phase_offsets = rng.random(count, dtype=np.float32)
        twinkle_phase = rng.uniform(0.0, math.pi * 2.0, count).astype(np.float32)

        intensity_norm = np.clip(intensities / 255.0, 0.0, 1.0)
        proximity = np.clip(0.75 * intensity_norm + 0.25 * (intensity_norm >= (1.0 - near_ratio)).astype(np.float32), 0.0, 1.0)

        speed_factor = (0.45 + 1.65 * proximity).astype(np.float32)
        brightness = np.clip(70.0 + 185.0 * intensity_norm, 70.0, 255.0).astype(np.float32)
        radius_px = (0.6 + 2.0 * proximity * size_gain).astype(np.float32)

        return {
            "enabled": True,
            "count": count,
            "angles": angles,
            "phase_offsets": phase_offsets,
            "twinkle_phase": twinkle_phase,
            "proximity": proximity,
            "start_radius": start_radius,
            "speed_factor": speed_factor,
            "brightness": brightness,
            "radius_px": radius_px,
            "center_x": center_x,
            "center_y": center_y,
            "max_radius": max_radius,
            "travel": float(speed),
            "near_threshold": float(max(0.45, 1.0 - near_ratio)),
        }

    def _draw_starfield_frame(self, frame: np.ndarray, starfield: dict, phase: float, prev_phase: float | None = None):
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return
        if not isinstance(starfield, dict) or not starfield.get("enabled"):
            return

        h, w = frame.shape[:2]
        cx = float(starfield["center_x"])
        cy = float(starfield["center_y"])
        max_radius = float(starfield["max_radius"])
        travel = float(starfield["travel"])

        angles = starfield["angles"]
        phase_offsets = starfield["phase_offsets"]
        twinkle_phase = starfield["twinkle_phase"]
        proximity = starfield["proximity"]
        start_radius = starfield["start_radius"]
        speed_factor = starfield["speed_factor"]
        brightness = starfield["brightness"]
        radius_px = starfield["radius_px"]
        near_threshold = float(starfield["near_threshold"])

        near_mask = proximity >= near_threshold

        local_phase = np.mod(phase_offsets + phase * travel * speed_factor, 1.0)
        radial_dist = start_radius + local_phase * max_radius
        x_pos = cx + np.cos(angles) * radial_dist
        y_pos = cy + np.sin(angles) * radial_dist
        valid = np.logical_and.reduce((x_pos >= 0.0, x_pos < float(w), y_pos >= 0.0, y_pos < float(h)))

        prev_valid = None
        prev_x_pos = None
        prev_y_pos = None
        prev_local_phase = None
        if prev_phase is not None:
            prev_local_phase = np.mod(phase_offsets + prev_phase * travel * speed_factor, 1.0)
            prev_radial = start_radius + prev_local_phase * max_radius
            prev_x_pos = cx + np.cos(angles) * prev_radial
            prev_y_pos = cy + np.sin(angles) * prev_radial
            prev_valid = np.logical_and.reduce((prev_x_pos >= 0.0, prev_x_pos < float(w), prev_y_pos >= 0.0, prev_y_pos < float(h)))

        for idx in np.where(valid)[0]:
            prox = float(proximity[idx])
            twinkle = 0.72 + 0.28 * math.sin(float(twinkle_phase[idx]) + phase * 24.0)
            value = int(np.clip(brightness[idx] * twinkle, 90.0, 255.0))
            radius = int(max(1, round(float(radius_px[idx]))))
            x = int(round(float(x_pos[idx])))
            y = int(round(float(y_pos[idx])))

            if prev_phase is not None and prev_valid is not None and prev_x_pos is not None and prev_y_pos is not None:
                if bool(near_mask[idx]) and bool(prev_valid[idx]) and prev_local_phase is not None and float(local_phase[idx]) >= float(prev_local_phase[idx]):
                    px = int(round(float(prev_x_pos[idx])))
                    py = int(round(float(prev_y_pos[idx])))
                    trail_val = int(np.clip(value * (0.65 + 0.35 * prox), 80, 255))
                    cv2.line(frame, (px, py), (x, y), (trail_val, trail_val, min(255, trail_val + 8)), 1, cv2.LINE_AA)

            cv2.circle(frame, (x, y), radius, (value, value, min(255, value + 10)), -1, cv2.LINE_AA)

    def run(self):
        if self.source_img is None:
            self.finished_signal.emit("", "3D FLY canceled: no image loaded.")
            return

        output_path = str(self.payload.get("output_path") or "").strip()
        if not output_path:
            self.finished_signal.emit("", "3D FLY canceled: output path is empty.")
            return

        root, ext = os.path.splitext(output_path)
        if ext.lower() not in (".mp4", ".avi"):
            output_path = root + ".mp4"

        self.progress_signal.emit("Preparing 3D FLY source...", 6, 10)
        base = self.source_img.copy()
        use_starless = bool(self.payload.get("starless"))
        if use_starless:
            self.progress_signal.emit("Running StarNet++ for starless frame...", 18, 20)
            starless, err = run_starnet_sync_for_image(base, self.starnet_path, self.starnet_stride)
            if err:
                self.finished_signal.emit("", err)
                return
            base = starless

        zone_map_small = self.payload.get("zone_map")
        cut_items = list(self.payload.get("cut_items") or [])
        h, w = base.shape[:2]
        zone_map = None
        if isinstance(zone_map_small, np.ndarray) and zone_map_small.size > 0:
            candidate = cv2.resize(zone_map_small.astype(np.int16), (w, h), interpolation=cv2.INTER_NEAREST)
            if np.any(candidate >= 0):
                zone_map = candidate

        behaviors = list(self.payload.get("behaviors") or [])
        if not behaviors:
            self.finished_signal.emit("", "No layer behavior was configured.")
            return

        edge_blur = float(self.payload.get("edge_blur") or 0.0)
        edge_blur_map_raw = self.payload.get("edge_blur_map")
        edge_blur_map = {}
        if isinstance(edge_blur_map_raw, dict):
            for key, value in edge_blur_map_raw.items():
                try:
                    edge_blur_map[str(key)] = float(value)
                except Exception:
                    continue
        edge_blur_brush_map_raw = self.payload.get("edge_blur_brush_map")
        edge_blur_brush_map = {}
        if isinstance(edge_blur_brush_map_raw, dict):
            for key, value in edge_blur_brush_map_raw.items():
                if isinstance(value, np.ndarray) and value.size > 0:
                    edge_blur_brush_map[str(key)] = value.astype(np.uint8)
        main_zoom_speed = float(self.payload.get("main_zoom_speed") if self.payload.get("main_zoom_speed") is not None else 0.03)
        main_zoom_speed = max(-1.0, min(1.0, main_zoom_speed))
        fps = max(12, int(self.payload.get("fps") or 24))
        duration = max(1.0, float(self.payload.get("duration") or 5.0))
        frame_count = max(2, int(round(duration * fps)))
        starfield = self._build_starfield_config(w, h)

        preview_shape = self.payload.get("preview_shape")
        motion_scale_x = 1.0
        motion_scale_y = 1.0
        if isinstance(preview_shape, (tuple, list)) and len(preview_shape) >= 2:
            try:
                preview_h = max(1.0, float(preview_shape[0]))
                preview_w = max(1.0, float(preview_shape[1]))
                motion_scale_x = float(w) / preview_w
                motion_scale_y = float(h) / preview_h
            except Exception:
                motion_scale_x = 1.0
                motion_scale_y = 1.0

        self.progress_signal.emit("Building zone masks...", 30, 35)
        layers: list[ImageLayer] = []
        warnings: list[str] = []
        occupied = np.zeros((h, w), dtype=bool)
        behaviors_sorted = sorted(behaviors, key=lambda item: float(item.get("depth") or 0.0))
        if zone_map is not None:
            for behavior in behaviors_sorted:
                zone_id = int(behavior.get("zone", -1))
                raw_mask = zone_map == zone_id
                if not np.any(raw_mask):
                    continue
                layer_key = str(behavior.get("layer_key") or zone_id)
                overlap_px = int(np.count_nonzero(np.logical_and(raw_mask, occupied)))
                if overlap_px > 0:
                    warnings.append(f"Layer overlap removed for zone {zone_id}: {overlap_px} px")

                unique_mask = np.logical_and(raw_mask, np.logical_not(occupied))
                if not np.any(unique_mask):
                    continue
                occupied[unique_mask] = True

                blur_value = float(edge_blur_map.get(layer_key, edge_blur))
                edge_blur_sigma = max(0.0, blur_value * 1.8)
                zone_alpha = unique_mask.astype(np.float32)
                if edge_blur_sigma > 0:
                    zone_alpha = cv2.GaussianBlur(zone_alpha, (0, 0), edge_blur_sigma)
                    zone_alpha *= unique_mask.astype(np.float32)
                zone_alpha = np.clip(zone_alpha, 0.0, 1.0)
                if float(np.max(zone_alpha)) < 1e-6:
                    continue

                layer_image = np.zeros_like(base)
                layer_image[unique_mask] = base[unique_mask]
                layers.append(
                    ImageLayer(
                        image=layer_image,
                        alpha=zone_alpha,
                        mask=unique_mask,
                        depth=float(behavior.get("depth") or 0.0),
                        move_x=float(behavior.get("move_x") or 0.0),
                        move_y=float(behavior.get("move_y") or 0.0),
                        zoom=float(behavior.get("zoom") or 0.0),
                        layer_key=layer_key,
                    )
                )
        else:
            for behavior in behaviors_sorted:
                layer_key = str(behavior.get("layer_key") or "")
                if not layer_key.startswith("cut:"):
                    continue
                try:
                    cut_index = int(layer_key.split(":", 1)[1])
                except Exception:
                    continue
                if cut_index < 0 or cut_index >= len(cut_items):
                    continue
                built = self._build_layer_from_cut(base, str(cut_items[cut_index]))
                if built is None:
                    continue
                layer_image_raw, raw_mask, placement = built
                layer_key = str(behavior.get("layer_key") or f"cut:{cut_index}")

                overlap_px = int(np.count_nonzero(np.logical_and(raw_mask, occupied)))
                if overlap_px > 0:
                    warnings.append(f"Layer overlap removed for {layer_key}: {overlap_px} px")

                unique_mask = np.logical_and(raw_mask, np.logical_not(occupied))
                if not np.any(unique_mask):
                    continue
                occupied[unique_mask] = True

                blur_value = float(edge_blur_map.get(layer_key, edge_blur))
                edge_blur_sigma = max(0.0, blur_value * 1.8)
                layer_alpha = unique_mask.astype(np.float32)
                if edge_blur_sigma > 0:
                    layer_alpha = cv2.GaussianBlur(layer_alpha, (0, 0), edge_blur_sigma)
                    layer_alpha *= unique_mask.astype(np.float32)

                brush_strength_full = None
                brush_local = edge_blur_brush_map.get(layer_key)
                if isinstance(brush_local, np.ndarray) and brush_local.size > 0 and int(np.count_nonzero(brush_local)) > 0:
                    x0 = int(placement.get("x0", 0)) if isinstance(placement, dict) else 0
                    y0 = int(placement.get("y0", 0)) if isinstance(placement, dict) else 0
                    x1 = int(placement.get("x1", x0)) if isinstance(placement, dict) else x0
                    y1 = int(placement.get("y1", y0)) if isinstance(placement, dict) else y0
                    src_h = max(0, y1 - y0)
                    src_w = max(0, x1 - x0)
                    src_mask = placement.get("src_mask") if isinstance(placement, dict) else None
                    if not isinstance(src_mask, np.ndarray) or src_mask.shape[:2] != (src_h, src_w):
                        src_mask = unique_mask[y0:y1, x0:x1].astype(np.uint8)
                    if src_h <= 0 or src_w <= 0:
                        src_mask = None

                if isinstance(brush_local, np.ndarray) and brush_local.size > 0 and int(np.count_nonzero(brush_local)) > 0 and isinstance(src_mask, np.ndarray):
                    local_h, local_w = int(src_h), int(src_w)
                    if brush_local.shape[:2] != (local_h, local_w):
                        brush_local = cv2.resize(brush_local.astype(np.uint8), (local_w, local_h), interpolation=cv2.INTER_NEAREST)
                    brush_local_f = np.clip(brush_local.astype(np.float32) / 255.0, 0.0, 1.0)
                    brush_local_f *= (src_mask > 0).astype(np.float32)
                    if float(np.max(brush_local_f)) > 1e-6:
                        brush_strength_full = np.zeros((h, w), dtype=np.float32)
                        brush_strength_full[y0:y1, x0:x1] = brush_local_f

                if isinstance(brush_strength_full, np.ndarray) and float(np.max(brush_strength_full)) > 1e-6:
                    extra_sigma = max(1.2, edge_blur_sigma * 1.8)
                    alpha_extra = unique_mask.astype(np.float32)
                    alpha_extra = cv2.GaussianBlur(alpha_extra, (0, 0), extra_sigma)
                    alpha_extra *= unique_mask.astype(np.float32)
                    layer_alpha = layer_alpha * (1.0 - brush_strength_full) + alpha_extra * brush_strength_full

                layer_alpha = np.clip(layer_alpha, 0.0, 1.0)
                if float(np.max(layer_alpha)) < 1e-6:
                    continue

                layer_image = np.zeros_like(base)
                layer_image[unique_mask] = layer_image_raw[unique_mask]
                layers.append(
                    ImageLayer(
                        image=layer_image,
                        alpha=layer_alpha,
                        mask=unique_mask,
                        depth=float(behavior.get("depth") or 0.0),
                        move_x=float(behavior.get("move_x") or 0.0),
                        move_y=float(behavior.get("move_y") or 0.0),
                        zoom=float(behavior.get("zoom") or 0.0),
                        layer_key=layer_key,
                    )
                )

        if not layers:
            self.finished_signal.emit("", "3D FLY failed: none of the configured zones contain pixels.")
            return

        overlap_after = _count_mask_overlap_pixels([layer.mask.astype(np.uint8) for layer in layers])
        if overlap_after > 0:
            warnings.append(f"Layer overlap detected after mask build: {overlap_after} px")

        # Remove all near-layer pixels from background and fill holes.
        background = _inpaint_background_without_layers(base, occupied)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
        if not writer.isOpened():
            output_path = os.path.splitext(output_path)[0] + ".avi"
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (w, h))
        if not writer.isOpened():
            self.finished_signal.emit("", "Could not create output video file.")
            return

        try:
            self.progress_signal.emit("Rendering 3D FLY clip...", 42, 40)
            for frame_idx in range(frame_count):
                phase = frame_idx / float(max(1, frame_count - 1))
                prev_phase = (frame_idx - 1) / float(max(1, frame_count - 1)) if frame_idx > 0 else None
                ease = 0.5 - 0.5 * math.cos(math.pi * phase)

                global_zoom = main_zoom_speed * ease
                global_x = -8.0 * motion_scale_x * ease
                global_y = 5.0 * motion_scale_y * ease
                composed, _ = _warp_layer_with_motion(
                    background,
                    np.ones((h, w), dtype=np.float32),
                    global_x,
                    global_y,
                    global_zoom,
                )

                for layer in layers:
                    depth_gain = 1.0 + layer.depth * 0.7
                    move_x = layer.move_x * motion_scale_x * ease * depth_gain
                    move_y = layer.move_y * motion_scale_y * ease * depth_gain
                    zoom = layer.zoom * ease * (1.0 + 0.4 * layer.depth)
                    warped_img, warped_alpha = _warp_layer_with_motion(layer.image, layer.alpha, move_x, move_y, zoom)
                    alpha3 = warped_alpha[:, :, np.newaxis]
                    composed = (composed.astype(np.float32) * (1.0 - alpha3) + warped_img.astype(np.float32) * alpha3)
                    composed = np.clip(composed, 0, 255).astype(np.uint8)

                self._draw_starfield_frame(composed, starfield, phase, prev_phase=prev_phase)

                writer.write(composed)
                progress = 42 + int(56 * (frame_idx + 1) / frame_count)
                self.progress_signal.emit(f"Rendering frame {frame_idx + 1}/{frame_count}...", progress, 75)

            self.progress_signal.emit("Merging audio timeline...", 98, 90)
            audio_timeline = self.payload.get("audio_timeline")
            output_path, audio_error = attach_audio_timeline_with_ffmpeg(output_path, audio_timeline, duration)
            warnings_text = "\n".join(warnings)
            if audio_error:
                self.progress_signal.emit("3D FLY completed (without audio timeline).", 100, 100)
                joined = audio_error if not warnings_text else f"{audio_error}\n{warnings_text}"
                self.finished_signal.emit(output_path, joined)
                return

            self.progress_signal.emit("3D FLY completed.", 100, 100)
            self.finished_signal.emit(output_path, warnings_text)
        except Exception as exc:
            self.finished_signal.emit("", f"3D FLY failed: {exc}")
        finally:
            writer.release()


class ArduinoJoystickWorker(QThread):
    pan_signal = pyqtSignal(int, int)
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, port: str, baud_rate: int = 115200, deadzone: int = 40, sensitivity: float = 1.6):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.deadzone = deadzone
        self.sensitivity = sensitivity
        self._running = True

    def stop(self):
        self._running = False

    def _parse_axes(self, line: str):
        numbers = [int(n) for n in re.findall(r"-?\d+", line)]
        if len(numbers) >= 2:
            return numbers[0], numbers[1]
        return None

    def _axis_to_delta(self, value: int) -> int:
        center = 512
        delta = value - center
        if abs(delta) <= self.deadzone:
            return 0

        span = max(1, 512 - self.deadzone)
        normalized = min(1.0, abs(delta) / float(span))
        step = max(1, int(round(normalized * 22.0 * self.sensitivity)))
        return step if delta > 0 else -step

    def run(self):
        if QT_SERIAL_AVAILABLE and QSerialPort is not None and QSerialPortInfo is not None:
            port = QSerialPort()
            port.setPortName(self.port)
            port.setBaudRate(self.baud_rate)
            port.setDataBits(QSerialPort.Data8)
            port.setParity(QSerialPort.NoParity)
            port.setStopBits(QSerialPort.OneStop)
            port.setFlowControl(QSerialPort.NoFlowControl)

            if not port.open(QSerialPort.ReadOnly):
                self.error_signal.emit(f"Could not open Arduino port {self.port}")
                return

            self.status_signal.emit(f"Arduino joystick connected on {self.port}")
            self.status_signal.emit("Waiting for joystick data...")
            buffer = bytearray()
            first_packet_reported = False
            try:
                while self._running:
                    if not port.waitForReadyRead(100):
                        continue

                    buffer.extend(bytes(port.readAll()))
                    while b"\n" in buffer:
                        line, _, buffer = buffer.partition(b"\n")
                        raw = line.decode("utf-8", errors="ignore").strip()
                        if not raw:
                            continue

                        axes = self._parse_axes(raw)
                        if axes is None:
                            continue

                        if not first_packet_reported:
                            self.status_signal.emit(f"First packet: {raw}")
                            first_packet_reported = True

                        x_val, y_val = axes
                        dx = self._axis_to_delta(y_val)
                        dy = -self._axis_to_delta(x_val)

                        if dx != 0 or dy != 0:
                            self.pan_signal.emit(dx, dy)
            finally:
                port.close()
                self.status_signal.emit("Arduino joystick disconnected")
            return

        if SERIAL_AVAILABLE and serial is not None:
            try:
                ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
                time.sleep(1.5)
            except Exception as e:
                self.error_signal.emit(f"Could not open Arduino port {self.port}: {e}")
                return

            self.status_signal.emit(f"Arduino joystick connected on {self.port}")
            self.status_signal.emit("Waiting for joystick data...")
            first_packet_reported = False
            try:
                while self._running:
                    try:
                        raw = ser.readline().decode("utf-8", errors="ignore").strip()
                    except Exception:
                        break

                    if not raw:
                        continue

                    axes = self._parse_axes(raw)
                    if axes is None:
                        continue

                    if not first_packet_reported:
                        self.status_signal.emit(f"First packet: {raw}")
                        first_packet_reported = True

                    x_val, y_val = axes
                    dx = self._axis_to_delta(y_val)
                    dy = -self._axis_to_delta(x_val)

                    if dx != 0 or dy != 0:
                        self.pan_signal.emit(dx, dy)
            finally:
                try:
                    ser.close()
                except Exception:
                    pass
                self.status_signal.emit("Arduino joystick disconnected")
            return

        if not QT_SERIAL_AVAILABLE or QSerialPort is None or QSerialPortInfo is None:
            self.error_signal.emit("Serial support is not available. Install pyserial or PyQt5 QtSerialPort.")
            return


class NewWorkspaceDialog(QDialog):
    DIALOG_OPTIONS = [
        ("console", "Console"),
        ("histogram", "Histogram"),
        ("menu", "Menu"),
        ("color_calibration", "Color Calibration"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.app = parent
        self._watched_dialogs = {}
        self._syncing_geometry = False
        self.setWindowTitle("New Workspace")
        self.setMinimumWidth(760)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        form.addWidget(QLabel("Workspace name:"), 0, 0)
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("e.g. Processing, Review, Plate Solve")
        form.addWidget(self.edit_name, 0, 1)

        form.addWidget(QLabel("Dialog layout:"), 1, 0)
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["Custom", "Cascade", "Tile", "Left stack", "Right stack"])
        self.combo_layout.currentTextChanged.connect(self._on_layout_changed)
        form.addWidget(self.combo_layout, 1, 1)
        layout.addLayout(form)

        layout.addWidget(QLabel("Dialogs:"))
        self.dialog_checks = {}
        self.geometry_spins = {}
        dialog_grid = QGridLayout()
        dialog_grid.setHorizontalSpacing(8)
        dialog_grid.setVerticalSpacing(8)
        dialog_grid.addWidget(QLabel("Window"), 0, 0)
        dialog_grid.addWidget(QLabel("X"), 0, 1)
        dialog_grid.addWidget(QLabel("Y"), 0, 2)
        dialog_grid.addWidget(QLabel("W"), 0, 3)
        dialog_grid.addWidget(QLabel("H"), 0, 4)
        for index, (key, label) in enumerate(self.DIALOG_OPTIONS, start=1):
            check = QCheckBox(label)
            check.stateChanged.connect(lambda state, dialog_key=key: self._on_dialog_toggled(dialog_key, state))
            self.dialog_checks[key] = check
            dialog_grid.addWidget(check, index, 0)

            self.geometry_spins[key] = {}
            for column, field, minimum, value in [
                (1, "x", -10000, 120 + index * 28),
                (2, "y", -10000, 120 + index * 28),
                (3, "width", 260, 440),
                (4, "height", 180, 320),
            ]:
                spin = QDoubleSpinBox()
                spin.setDecimals(0)
                spin.setRange(minimum, 20000)
                spin.setSingleStep(10)
                spin.setValue(value)
                spin.valueChanged.connect(lambda _value, dialog_key=key: self._apply_spin_geometry(dialog_key))
                self.geometry_spins[key][field] = spin
                dialog_grid.addWidget(spin, index, column)
        layout.addLayout(dialog_grid)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_create = QPushButton("Create")
        self.btn_create.setProperty("accent", True)
        self.btn_cancel = QPushButton("Cancel")
        buttons.addWidget(self.btn_create)
        buttons.addWidget(self.btn_cancel)
        layout.addLayout(buttons)

        self.btn_create.clicked.connect(self._accept_if_valid)
        self.btn_cancel.clicked.connect(self.reject)

    def _on_dialog_toggled(self, dialog_key, state):
        if self.app is None:
            return
        if state == Qt.Checked:
            dialog = self.app.open_workspace_dialog(dialog_key)
            self._watch_dialog(dialog_key, dialog)
            if self.combo_layout.currentText() == "Custom":
                self._apply_spin_geometry(dialog_key)
            else:
                self._apply_selected_layout()
            if dialog is not None:
                self.app.log(f"Workspace dialog opened: {dialog_key}")
        else:
            self._unwatch_dialog(dialog_key)
            self.app.close_workspace_dialog(dialog_key)

    def _on_layout_changed(self, layout_name):
        if layout_name == "Custom":
            return
        self._apply_selected_layout()

    def _apply_selected_layout(self):
        if self.app is None:
            return
        dialogs = self.app.get_workspace_open_dialogs(self.get_selected_dialogs())
        self._syncing_geometry = True
        try:
            self.app.arrange_workspace_dialogs(dialogs, self.combo_layout.currentText())
        finally:
            self._syncing_geometry = False
        for dialog_key in self.get_selected_dialogs():
            self._sync_spins_from_dialog(dialog_key, self.app.get_workspace_dialog(dialog_key))

    def _watch_dialog(self, dialog_key, dialog):
        if dialog is None:
            return
        current = self._watched_dialogs.get(dialog)
        if current is None:
            dialog.installEventFilter(self)
        self._watched_dialogs[dialog] = dialog_key

    def _unwatch_dialog(self, dialog_key):
        for dialog, watched_key in list(self._watched_dialogs.items()):
            if watched_key == dialog_key:
                dialog.removeEventFilter(self)
                self._watched_dialogs.pop(dialog, None)

    def eventFilter(self, watched, event):
        dialog_key = self._watched_dialogs.get(watched)
        if dialog_key and event.type() in (QEvent.Move, QEvent.Resize) and not self._syncing_geometry:
            if self.combo_layout.currentText() != "Custom":
                self.combo_layout.setCurrentText("Custom")
            QTimer.singleShot(0, lambda key=dialog_key, dialog=watched: self._sync_spins_from_dialog(key, dialog))
        return super().eventFilter(watched, event)

    def _sync_spins_from_dialog(self, dialog_key, dialog):
        if dialog is None or dialog_key not in self.geometry_spins:
            return
        rect = dialog.geometry()
        values = {
            "x": rect.x(),
            "y": rect.y(),
            "width": rect.width(),
            "height": rect.height(),
        }
        self._syncing_geometry = True
        try:
            for field, value in values.items():
                spin = self.geometry_spins[dialog_key][field]
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        finally:
            self._syncing_geometry = False

    def _apply_spin_geometry(self, dialog_key):
        if self.app is None or dialog_key not in self.geometry_spins:
            return
        if not self.dialog_checks.get(dialog_key) or not self.dialog_checks[dialog_key].isChecked():
            return
        dialog = self.app.get_workspace_dialog(dialog_key)
        if dialog is None:
            return
        geometry = self.get_dialog_geometry(dialog_key)
        self._syncing_geometry = True
        try:
            dialog.setGeometry(geometry["x"], geometry["y"], geometry["width"], geometry["height"])
        finally:
            self._syncing_geometry = False

    def _accept_if_valid(self):
        if not self.edit_name.text().strip():
            QMessageBox.warning(self, "New Workspace", "Enter a workspace name.")
            return
        if not self.get_selected_dialogs():
            QMessageBox.warning(self, "New Workspace", "Select at least one dialog.")
            return
        self.accept()

    def get_selected_dialogs(self):
        return [key for key, check in self.dialog_checks.items() if check.isChecked()]

    def get_dialog_geometry(self, dialog_key):
        spins = self.geometry_spins.get(dialog_key, {})
        return {
            "x": int(spins["x"].value()),
            "y": int(spins["y"].value()),
            "width": max(260, int(spins["width"].value())),
            "height": max(180, int(spins["height"].value())),
        }

    def get_workspace_geometry(self, selected_dialogs):
        return {dialog_key: self.get_dialog_geometry(dialog_key) for dialog_key in selected_dialogs}

    def get_workspace(self):
        selected_dialogs = self.get_selected_dialogs()
        geometry = self.get_workspace_geometry(selected_dialogs)
        return {
            "name": self.edit_name.text().strip(),
            "layout": self.combo_layout.currentText(),
            "dialogs": selected_dialogs,
            "geometry": geometry,
        }


class PreferencesDialog(QDialog):
    def __init__(self, app, parent=None):
        super().__init__(parent or app)
        apply_dialog_window_flags(self)
        self.app = app
        self.setWindowTitle("Preferencje")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        processor_tab = QWidget()
        processor_layout = QGridLayout(processor_tab)
        processor_layout.setContentsMargins(16, 16, 16, 16)
        processor_layout.setHorizontalSpacing(10)
        processor_layout.setVerticalSpacing(10)

        processor_layout.addWidget(QLabel("Liczba uĹĽywanych rdzeni:"), 0, 0)
        self.spin_cpu_cores = QSpinBox()
        self.spin_cpu_cores.setRange(1, max(1, os.cpu_count() or 1))
        self.spin_cpu_cores.setValue(int(getattr(self.app, "processor_cores", max(1, os.cpu_count() or 1))))
        processor_layout.addWidget(self.spin_cpu_cores, 0, 1)

        processor_layout.addWidget(QLabel("Akcelerator ONNX:"), 1, 0)
        self.combo_onnx_provider = QComboBox()
        self.combo_onnx_provider.addItems([
            "Auto",
            "CPU only",
            "NVIDIA / CUDA",
            "DirectML",
        ])
        self.combo_onnx_provider.setCurrentText(str(getattr(self.app, "onnx_provider", "Auto")))
        processor_layout.addWidget(self.combo_onnx_provider, 1, 1)

        processor_layout.addWidget(QLabel("Uwagi:"), 2, 0)
        self.processor_info = QLabel("WybĂłr wpĹ‚ywa na sesje ONNX i obciÄ…ĹĽenie CPU podczas filtrĂłw AI.")
        self.processor_info.setWordWrap(True)
        processor_layout.addWidget(self.processor_info, 2, 1)
        tabs.addTab(processor_tab, "Procesor")

        appearance_tab = QWidget()
        appearance_layout = QGridLayout(appearance_tab)
        appearance_layout.setContentsMargins(16, 16, 16, 16)
        appearance_layout.setHorizontalSpacing(10)
        appearance_layout.setVerticalSpacing(10)

        appearance_layout.addWidget(QLabel("Styl interfejsu:"), 0, 0)
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(["Fusion Dark", "Graphite", "Midnight", "Light"])
        self.combo_theme.setCurrentText(str(getattr(self.app, "theme_name", "Fusion Dark")))
        appearance_layout.addWidget(self.combo_theme, 0, 1)

        appearance_layout.addWidget(QLabel("JÄ™zyk:"), 1, 0)
        self.combo_language = QComboBox()
        self.combo_language.addItems(["pl", "en"])
        self.combo_language.setCurrentText(str(getattr(self.app, "language", "pl")))
        appearance_layout.addWidget(self.combo_language, 1, 1)

        appearance_layout.addWidget(QLabel("PodglÄ…d:"), 2, 0)
        self.appearance_info = QLabel(
            "Styl zastÄ™puje prosty przeĹ‚Ä…cznik dark mode i jest zapamiÄ™tywany miÄ™dzy uruchomieniami."
        )
        self.appearance_info.setWordWrap(True)
        appearance_layout.addWidget(self.appearance_info, 2, 1)
        tabs.addTab(appearance_tab, "WyglÄ…d")


        select_tab = QWidget()
        select_layout = QVBoxLayout(select_tab)
        select_layout.setContentsMargins(16, 16, 16, 16)
        select_layout.setSpacing(10)

        self.btn_select_models_pref = QPushButton("Select ONNX Models")
        self.btn_select_models_pref.clicked.connect(self._select_onnx_models_from_preferences)
        select_layout.addWidget(self.btn_select_models_pref)

        self.lbl_select_models_info = QLabel()
        self.lbl_select_models_info.setWordWrap(True)
        select_layout.addWidget(self.lbl_select_models_info)

        self.btn_select_starnet_pref = QPushButton("Select StarNet++")
        self.btn_select_starnet_pref.clicked.connect(self._select_starnet_from_preferences)
        select_layout.addWidget(self.btn_select_starnet_pref)

        self.lbl_select_starnet_info = QLabel()
        self.lbl_select_starnet_info.setWordWrap(True)
        select_layout.addWidget(self.lbl_select_starnet_info)

        self.btn_select_deepsnr_pref = QPushButton("Select deepSNR")
        self.btn_select_deepsnr_pref.clicked.connect(self._select_deepsnr_from_preferences)
        select_layout.addWidget(self.btn_select_deepsnr_pref)

        self.lbl_select_deepsnr_info = QLabel()
        self.lbl_select_deepsnr_info.setWordWrap(True)
        select_layout.addWidget(self.lbl_select_deepsnr_info)

        self.edit_deepsnr_args = QLineEdit(str(getattr(self.app, "deepsnr_args", "{input}") or "{input}"))
        self.edit_deepsnr_args.setPlaceholderText("np. --input \"{input}\" --output \"{output}\"")
        select_layout.addWidget(self.edit_deepsnr_args)

        self.lbl_deepsnr_args_help = QLabel("deepSNR CLI args: uzyj {input} i opcjonalnie {output}.")
        self.lbl_deepsnr_args_help.setWordWrap(True)
        select_layout.addWidget(self.lbl_deepsnr_args_help)

        self.btn_run_deepsnr_pref = QPushButton("Run deepSNR")
        self.btn_run_deepsnr_pref.clicked.connect(self.app.run_deepsnr)
        select_layout.addWidget(self.btn_run_deepsnr_pref)

        self.btn_arduino_joystick_pref = QPushButton("Connect Joystick")
        self.btn_arduino_joystick_pref.clicked.connect(self.app.connect_arduino_joystick)
        select_layout.addWidget(self.btn_arduino_joystick_pref)

        self.lbl_arduino_status_pref = QLabel("Arduino joystick: disconnected")
        self.lbl_arduino_status_pref.setWordWrap(True)
        select_layout.addWidget(self.lbl_arduino_status_pref)

        select_layout.addStretch(1)
        tabs.addTab(select_tab, "select")

        layers_tab = QWidget()
        layers_layout = QVBoxLayout(layers_tab)
        layers_layout.setContentsMargins(16, 16, 16, 16)
        layers_layout.setSpacing(10)
        self.mask_info = QLabel(
            "Maskowanie warstw jest dostÄ™pne z poziomu aktywnej warstwy i okna Curves."
        )
        self.mask_info.setWordWrap(True)
        layers_layout.addWidget(self.mask_info)
        tabs.addTab(layers_tab, "Warstwy")

        ai_chat_tab = QWidget()
        ai_chat_layout = QGridLayout(ai_chat_tab)
        ai_chat_layout.setContentsMargins(16, 16, 16, 16)
        ai_chat_layout.setHorizontalSpacing(10)
        ai_chat_layout.setVerticalSpacing(10)
        ai_chat_layout.addWidget(QLabel("Gemini API key:"), 0, 0)
        self.edit_gemini_api_pref = QLineEdit(str(getattr(self.app, "gemini_api_key", "") or ""))
        self.edit_gemini_api_pref.setEchoMode(QLineEdit.Password)
        self.edit_gemini_api_pref.setPlaceholderText("Wpisz klucz API")
        ai_chat_layout.addWidget(self.edit_gemini_api_pref, 0, 1)
        tabs.addTab(ai_chat_tab, "Ai Chat")

        buttons = QHBoxLayout()
        self.btn_apply = QPushButton("Zastosuj")
        self.btn_apply.setProperty("accent", True)
        self.btn_close = QPushButton("Zamknij")
        buttons.addWidget(self.btn_apply)
        buttons.addWidget(self.btn_close)
        layout.addLayout(buttons)

        self.btn_apply.clicked.connect(self.apply_changes)
        self.btn_close.clicked.connect(self.close)
        self.refresh_select_paths()
        self.refresh_joystick_controls()

    def refresh_select_paths(self):
        denoise_path = getattr(self.app, "denoise_model_path", None) or "None"
        bg_path = getattr(self.app, "bg_removal_model_path", None) or "None"
        starnet_path = getattr(self.app, "starnet_path", None) or "None"
        deepsnr_path = getattr(self.app, "deepsnr_path", None) or "None"
        self.lbl_select_models_info.setText(
            f"Denoise model: {denoise_path}\nBackground model: {bg_path}"
        )
        self.lbl_select_starnet_info.setText(f"StarNet++: {starnet_path}")
        self.lbl_select_deepsnr_info.setText(f"deepSNR: {deepsnr_path}")
        if hasattr(self, "edit_deepsnr_args"):
            self.edit_deepsnr_args.setText(str(getattr(self.app, "deepsnr_args", "{input}") or "{input}"))

    def _select_onnx_models_from_preferences(self):
        self.app.select_onnx_models()
        self.refresh_select_paths()

    def _select_starnet_from_preferences(self):
        self.app.select_starnet_path()
        self.refresh_select_paths()

    def _select_deepsnr_from_preferences(self):
        self.app.select_deepsnr_path()
        self.refresh_select_paths()

    def refresh_joystick_controls(self):
        if getattr(self.app, "arduino_joystick_worker", None) is None:
            self.btn_arduino_joystick_pref.setText(self.app.tr("connect_joystick", "Connect Joystick"))
            self.lbl_arduino_status_pref.setText(self.app.tr("joystick_disconnected", "Arduino joystick: disconnected"))
        else:
            self.btn_arduino_joystick_pref.setText(self.app.tr("disconnect_joystick", "Disconnect Joystick"))

    def apply_changes(self):
        theme_name = self.combo_theme.currentText().strip() or "Fusion Dark"
        language = self.combo_language.currentText().strip() or "pl"
        provider = self.combo_onnx_provider.currentText().strip() or "Auto"
        cpu_cores = int(self.spin_cpu_cores.value())
        gemini_api_key = self.edit_gemini_api_pref.text().strip()
        deepsnr_args = self.edit_deepsnr_args.text().strip() if hasattr(self, "edit_deepsnr_args") else "{input}"
        if not deepsnr_args:
            deepsnr_args = "{input}"

        self.app.gemini_api_key = gemini_api_key
        self.app.deepsnr_args = deepsnr_args
        APP_PREFERENCES["deepsnr_args"] = deepsnr_args

        self.app.apply_preferences(
            theme_name=theme_name,
            language=language,
            processor_cores=cpu_cores,
            onnx_provider=provider,
            gemini_api_key=gemini_api_key,
        )
        self.app.log("Preferences updated.", "success")
        self.accept()


class MagicProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("Magic Filter Progress")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)
        self.label_stage = QLabel("Starting...")
        self.progress_overall = CircularProgressBar()
        self.progress_overall.setRange(0, 100)
        self.progress_overall.setValue(0)

        self.label_current = QLabel("Current step...")
        self.progress_current = CircularProgressBar()
        self.progress_current.setRange(0, 100)
        self.progress_current.setValue(0)

        layout.addWidget(self.label_stage)
        layout.addWidget(self.progress_overall, 0, Qt.AlignHCenter)
        layout.addWidget(self.label_current)
        layout.addWidget(self.progress_current, 0, Qt.AlignHCenter)

    def update_progress(self, stage_name: str, overall_value: int, current_value: int):
        self.label_stage.setText(stage_name)
        self.progress_overall.setValue(max(0, min(100, overall_value)))
        self.progress_current.setValue(max(0, min(100, current_value)))
        QApplication.processEvents()


FITS_EXTENSIONS = (".fits", ".fit", ".fts")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff") + FITS_EXTENSIONS


def _is_fits_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in FITS_EXTENSIONS


def _is_supported_image_path(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def image_paths_from_mime_data(mime_data):
    if not mime_data.hasUrls():
        return []
    return [
        url.toLocalFile()
        for url in mime_data.urls()
        if _is_supported_image_path(url.toLocalFile())
    ]


def safe_fits_read(path: str):
    try:
        from astropy.io import fits
    except ImportError:
        raise RuntimeError("astropy is required to read FITS files.")

    with fits.open(path, memmap=False) as hdul:
        data = None
        for hdu in hdul:
            if getattr(hdu, 'data', None) is not None:
                data = hdu.data
                break
        if data is None:
            raise RuntimeError("No image data found in FITS file.")

        data = np.array(data, copy=False)

        if data.ndim == 3:
            if data.shape[0] in (3, 4):
                data = np.moveaxis(data, 0, -1)
            elif data.shape[-1] in (3, 4):
                data = np.array(data)
        elif data.ndim == 2:
            pass
        else:
            raise RuntimeError(f"Unsupported FITS image dimensions: {data.shape}")

        if data.ndim == 3 and data.shape[2] in (3, 4):
            data = data[..., ::-1]

        return data


def safe_imread(path: str):
    """Wczytywanie pliku z polskimi znakami."""
    if _is_fits_path(path):
        return safe_fits_read(path)

    with open(path, "rb") as f:
        data = f.read()
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    return img


def safe_fits_write(path: str, img: np.ndarray) -> bool:
    try:
        from astropy.io import fits
    except ImportError:
        raise RuntimeError("astropy is required to save FITS files.")

    if img is None:
        return False

    img = img.copy()
    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = img[..., ::-1]
        img = np.moveaxis(img, 2, 0)

    hdu = fits.PrimaryHDU(img)
    hdu.writeto(path, overwrite=True)
    return True


def normalize_to_uint8_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img -= img.min()
        if img.max() > 0:
            img /= img.max()
        img = (img * 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def get_onnx_session(model_path: str):
    """
    Cache ONNX sessions.
    Model Ĺ‚adowany tylko raz.
    """

    global ONNX_SESSIONS

    if model_path in ONNX_SESSIONS:
        return ONNX_SESSIONS[model_path]

    providers = ort.get_available_providers()
    pref = APP_PREFERENCES if "APP_PREFERENCES" in globals() else {}
    requested_provider = str(pref.get("onnx_provider", "Auto"))
    cpu_cores = int(pref.get("processor_cores") or max(1, os.cpu_count() or 1))

    if requested_provider == "CPU only":
        preferred = ["CPUExecutionProvider"]
    elif requested_provider in ("NVIDIA / CUDA", "CUDA"):
        preferred = ["CUDAExecutionProvider"]
        if "CPUExecutionProvider" in providers:
            preferred.append("CPUExecutionProvider")
    elif requested_provider in ("DirectML", "GPU / DirectML"):
        preferred = ["DmlExecutionProvider"]
        if "CPUExecutionProvider" in providers:
            preferred.append("CPUExecutionProvider")
    else:
        preferred = []
        if "CUDAExecutionProvider" in providers:
            preferred.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in providers:
            preferred.append("DmlExecutionProvider")
        preferred.append("CPUExecutionProvider")

    preferred = [provider for provider in preferred if provider in providers or provider == "CPUExecutionProvider"]
    if not preferred:
        preferred = ["CPUExecutionProvider"]

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = max(1, cpu_cores)
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=preferred
    )

    ONNX_SESSIONS[model_path] = session

    print(f"Loaded ONNX model once: {model_path}")
    print(f"Providers: {session.get_providers()}")

    return session


# ---------- Magic pipeline ----------

def process_onnx_tile(tile: np.ndarray, session, model_type: str = "denoise") -> np.ndarray:
    """Process a single tile with ONNX model"""
    try:
        tile_normalized = tile.astype(np.float32) / 255.0

        if tile_normalized.ndim == 3:
            # HWC -> CHW
            tile_input = np.transpose(tile_normalized, (2, 0, 1))
            tile_input = np.expand_dims(tile_input, axis=0).astype(np.float32)
        else:
            tile_input = np.expand_dims(
                np.expand_dims(tile_normalized, axis=0),
                axis=0
            ).astype(np.float32)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        output = session.run([output_name], {input_name: tile_input})[0]
        result = np.squeeze(output)
        if result.ndim == 3:
            # CHW -> HWC
            result = np.transpose(result, (1, 2, 0))

        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        return result
    except Exception as e:
        print(f"{model_type} error: {e}")
        return tile


def process_image_with_tiles(
    img: np.ndarray,
    model_path: str,
    tile_size: int = 256,
    overlap: int = 32,
    model_type: str = "denoise",
    progress_callback=None,
    overall_start: int = 0,
    overall_end: int = 100,
) -> np.ndarray:
    """Process image using tiling to handle large images with fixed-size models"""
    if model_path is None or not ONNX_AVAILABLE or not os.path.exists(model_path):
        if progress_callback is not None:
            progress_callback(model_type, overall_end, 100)
        return img
    session = get_onnx_session(model_path)

    h, w = img.shape[:2]

    if h <= tile_size and w <= tile_size:
        processed = process_onnx_tile(img, session, model_type)
        if progress_callback is not None:
            progress_callback(model_type, overall_end, 100)
        return processed

    result = np.zeros_like(img, dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    stride = tile_size - overlap

    y_positions = []
    y = 0
    while y < h:
        y_end = min(y + tile_size, h)
        y_positions.append((y, y_end))
        if y_end == h:
            break
        y += stride

    x_positions = []
    x = 0
    while x < w:
        x_end = min(x + tile_size, w)
        x_positions.append((x, x_end))
        if x_end == w:
            break
        x += stride

    total_tiles = len(y_positions) * len(x_positions)
    tile_index = 0

    for y, y_end in y_positions:
        for x, x_end in x_positions:
            tile = img[y:y_end, x:x_end]
            processed_tile = process_onnx_tile(tile, session, model_type).astype(np.float32)

            if processed_tile.shape[0] != (y_end - y) or processed_tile.shape[1] != (x_end - x):
                processed_tile = cv2.resize(processed_tile, (x_end - x, y_end - y))

            result[y:y_end, x:x_end] += processed_tile
            weight_map[y:y_end, x:x_end] += 1

            tile_index += 1
            if progress_callback is not None:
                current_value = int(tile_index / total_tiles * 100)
                overall_value = overall_start + int((overall_end - overall_start) * current_value / 100)
                progress_callback(model_type, overall_value, current_value)

    weight_map[weight_map == 0] = 1
    if result.ndim == 3:
        weight_map = np.stack([weight_map] * result.shape[2], axis=2)
    result = result / weight_map

    if progress_callback is not None:
        progress_callback(model_type, overall_end, 100)
    return np.clip(result, 0, 255).astype(np.uint8)


def magic_pipeline(img: np.ndarray, denoise_model_path: str = None, bg_removal_model_path: str = None, progress_callback=None) -> np.ndarray:
    """
    Pipeline with ONNX-based denoising and background extraction using tiling:
    - Optional ONNX denoising
    - Optional ONNX background extraction
    - normalizacja
    - asinh stretch
    - unsharp mask
    """
    if img is None:
        return None

    result = img.copy().astype(np.float32)

    stages = []
    if denoise_model_path and ONNX_AVAILABLE and os.path.exists(denoise_model_path):
        stages.append("Denoising")
    if bg_removal_model_path and ONNX_AVAILABLE and os.path.exists(bg_removal_model_path):
        stages.append("Background extraction")

    stage_ranges = {}
    if len(stages) == 2:
        stage_ranges["Denoising"] = (0, 50)
        stage_ranges["Background extraction"] = (50, 100)
    else:
        stage_ranges["Denoising"] = (0, 100)
        stage_ranges["Background extraction"] = (0, 100)

    if denoise_model_path and ONNX_AVAILABLE and os.path.exists(denoise_model_path):
        start, end = stage_ranges["Denoising"]
        result = process_image_with_tiles(
            result.astype(np.uint8),
            denoise_model_path,
            tile_size=256,
            overlap=32,
            model_type="Denoising",
            progress_callback=progress_callback,
            overall_start=start,
            overall_end=end,
        )

    if bg_removal_model_path and ONNX_AVAILABLE and os.path.exists(bg_removal_model_path):
        start, end = stage_ranges["Background extraction"]
        mask = process_image_with_tiles(
            result.astype(np.uint8),
            bg_removal_model_path,
            tile_size=256,
            overlap=32,
            model_type="Background extraction",
            progress_callback=progress_callback,
            overall_start=start,
            overall_end=end,
        )

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=2)
        mask = mask.astype(np.float32) / 255.0
        result = result * mask

    if progress_callback is not None:
        progress_callback("Finished", 100, 100)
    result = np.clip(result, 0, 255).astype(np.float32)

    img_f = result
    img_norm = img_f - img_f.min()
    if img_norm.max() > 0:
        img_norm /= img_norm.max()

    stretch_k = 5.0
    stretch = np.arcsinh(stretch_k * img_norm) / np.arcsinh(stretch_k)

    blur = cv2.GaussianBlur(stretch, (0, 0), 1.0)
    sharp = cv2.addWeighted(stretch, 1.5, blur, -0.5, 0)

    sharp = np.clip(sharp * 255, 0, 255).astype(np.uint8)
    return sharp


# ---------- Config Management ----------

def get_config_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "config")


def get_legacy_config_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "astro_magic_config.json")


def load_config() -> dict:
    config_path = get_config_path()
    legacy_config_path = get_legacy_config_path()
    default_config = {
        "denoise_model_path": None,
        "bg_removal_model_path": None,
        "starnet_path": None,
        "deepsnr_path": None,
        "deepsnr_args": "{input}",
        "starnet_stride": 16,
        "dark_mode": True,
        "theme_name": "Fusion Dark",
        "api_key": "",
        "pixel_size_um": 5.4,
        "focal_length_mm": 800.0,
        "gemini_api_key": "",
        "gemini_model": FIXED_GEMINI_MODEL,
        "language": "pl",
        "processor_cores": max(1, (os.cpu_count() or 4) // 2),
        "onnx_provider": "Auto",
        "workspaces": [],
        "home_folder": "",
        "topbar_button_order": [],
    }

    try:
        source_path = None
        if os.path.exists(config_path):
            source_path = config_path
        elif os.path.exists(legacy_config_path):
            source_path = legacy_config_path

        if source_path:
            with open(source_path, 'r') as f:
                loaded_config = json.load(f)
            config = default_config.copy()
            if isinstance(loaded_config, dict):
                config.update(loaded_config)
            if config.get("denoise_model_path") and not os.path.exists(config["denoise_model_path"]):
                config["denoise_model_path"] = None
            if config.get("bg_removal_model_path") and not os.path.exists(config["bg_removal_model_path"]):
                config["bg_removal_model_path"] = None
            if config.get("starnet_path") and not os.path.exists(config["starnet_path"]):
                config["starnet_path"] = None
            if config.get("deepsnr_path") and not os.path.exists(config["deepsnr_path"]):
                config["deepsnr_path"] = None
            config["deepsnr_args"] = str(config.get("deepsnr_args", "{input}") or "{input}")

            if not isinstance(config.get("topbar_button_order"), list):
                config["topbar_button_order"] = []
            else:
                config["topbar_button_order"] = [
                    str(name).strip()
                    for name in config["topbar_button_order"]
                    if str(name).strip()
                ]

            if source_path == legacy_config_path:
                try:
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                except Exception:
                    pass

            return config
    except Exception as e:
        print(f"Error loading config: {e}")

    return default_config


def save_config(denoise_path: str = None, bg_removal_path: str = None,
                dark_mode: bool = True, starnet_path: str = None,
                api_key: str = "", pixel_size_um: float = 5.4,
                focal_length_mm: float = 800.0,
                starnet_stride: int = 16,
                gemini_api_key: str = "",
                gemini_model: str = FIXED_GEMINI_MODEL,
                theme_name: str = None,
                language: str = None,
                processor_cores: int = None,
                onnx_provider: str = None,
                workspaces: list = None,
                home_folder: str = None,
                topbar_button_order: list = None,
                deepsnr_path: str = None,
                deepsnr_args: str = None):
    if theme_name is None:
        theme_name = APP_PREFERENCES.get("theme_name", "Fusion Dark" if dark_mode else "Light")
    if language is None:
        language = APP_PREFERENCES.get("language", "pl")
    if processor_cores is None:
        processor_cores = APP_PREFERENCES.get("processor_cores", max(1, os.cpu_count() or 4))
    if onnx_provider is None:
        onnx_provider = APP_PREFERENCES.get("onnx_provider", "Auto")
    if workspaces is None:
        workspaces = APP_PREFERENCES.get("workspaces", [])
    if home_folder is None:
        home_folder = APP_PREFERENCES.get("home_folder", "")
    if topbar_button_order is None:
        topbar_button_order = APP_PREFERENCES.get("topbar_button_order", [])
    if deepsnr_path is None:
        deepsnr_path = APP_PREFERENCES.get("deepsnr_path")
    if deepsnr_args is None:
        deepsnr_args = APP_PREFERENCES.get("deepsnr_args", "{input}")

    config = {
        "denoise_model_path": denoise_path,
        "bg_removal_model_path": bg_removal_path,
        "starnet_path": starnet_path,
        "deepsnr_path": deepsnr_path,
        "deepsnr_args": str(deepsnr_args or "{input}"),
        "starnet_stride": starnet_stride,
        "dark_mode": dark_mode,
        "theme_name": theme_name,
        "api_key": api_key,
        "pixel_size_um": pixel_size_um,
        "focal_length_mm": focal_length_mm,
        "gemini_api_key": gemini_api_key,
        "gemini_model": gemini_model,
        "language": language,
        "processor_cores": int(processor_cores or 1),
        "onnx_provider": onnx_provider,
        "workspaces": workspaces if isinstance(workspaces, list) else [],
        "home_folder": str(home_folder or ""),
        "topbar_button_order": [
            str(name).strip()
            for name in (topbar_button_order if isinstance(topbar_button_order, list) else [])
            if str(name).strip()
        ],
    }

    try:
        config_path = get_config_path()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")


APP_PREFERENCES = {
    "theme_name": "Fusion Dark",
    "language": "pl",
    "processor_cores": max(1, (os.cpu_count() or 4) // 2),
    "onnx_provider": "Auto",
    "workspaces": [],
    "home_folder": "",
    "topbar_button_order": [],
    "deepsnr_path": None,
    "deepsnr_args": "{input}",
}


def _replace_theme_tokens(base: str, replacements: dict) -> str:
    themed = base
    for old, new in replacements.items():
        themed = themed.replace(old, new)
    return themed


def get_graphite_stylesheet() -> str:
    return _replace_theme_tokens(
        get_dark_stylesheet(),
        {
            "#1e1e1e": "#1d1f21",
            "#252526": "#232629",
            "#2d2d2d": "#282b2f",
            "#3a3a3a": "#35393f",
            "#3d3d3d": "#40454d",
            "#007acc": "#5b86c5",
            "#008cff": "#76a3dc",
            "#0a86d9": "#6a96d6",
            "#0062a3": "#4b709f",
        },
    )


def get_midnight_stylesheet() -> str:
    return _replace_theme_tokens(
        get_dark_stylesheet(),
        {
            "#1e1e1e": "#121826",
            "#252526": "#182033",
            "#2d2d2d": "#1e2940",
            "#3a3a3a": "#2a3956",
            "#3d3d3d": "#344867",
            "#007acc": "#3ea6ff",
            "#008cff": "#58b7ff",
            "#0a86d9": "#4b9fe6",
            "#0062a3": "#2e78b8",
        },
    )


def get_theme_stylesheet(theme_name: str) -> str:
    normalized = (theme_name or "").strip().lower()
    if normalized in ("light", "jasny"):
        return get_light_stylesheet()
    if normalized in ("graphite", "graphite dark"):
        return get_graphite_stylesheet()
    if normalized in ("midnight", "midnight blue"):
        return get_midnight_stylesheet()
    return get_dark_stylesheet()


def get_theme_palette(theme_name: str) -> QPalette:
    if (theme_name or "").strip().lower() in ("light", "jasny"):
        return QPalette()
    return get_dark_palette()

def apply_levels(img: np.ndarray, black: int, gamma: float, white: int, channels=None) -> np.ndarray:
    if img is None:
        return None

    if channels is not None and len(channels) == 0:
        return img.copy()

    def apply_to_channel(channel_img):
        img_f = channel_img.astype(np.float32) / 255.0

        img_f = (img_f - black / 255.0) / max(1e-6, (white - black) / 255.0)
        img_f = np.clip(img_f, 0, 1)

        img_f = img_f ** (1.0 / max(0.01, gamma))

        return np.clip(img_f * 255, 0, 255).astype(np.uint8)

    if channels is None or img.ndim == 2:
        return apply_to_channel(img)

    out = img.copy()
    channel_map = {"b": 0, "g": 1, "r": 2}
    for channel_name in channels:
        idx = channel_map.get(channel_name.lower())
        if idx is not None and idx < out.shape[2]:
            out[:, :, idx] = apply_to_channel(out[:, :, idx])

    return out


    # Normalizacja blackâ€“white



def build_curve_lut(points, curve_mode="linear") -> np.ndarray:
    control_points = [(0, 0)] + list(points or []) + [(255, 255)]
    sorted_points = sorted((int(x), int(y)) for x, y in control_points)
    compact_points = []
    for x, y in sorted_points:
        x = int(np.clip(x, 0, 255))
        y = int(np.clip(y, 0, 255))
        if compact_points and compact_points[-1][0] == x:
            compact_points[-1] = (x, y)
        else:
            compact_points.append((x, y))

    if len(compact_points) < 2:
        return np.arange(256, dtype=np.uint8)

    sorted_points = compact_points
    xs = np.array([p[0] for p in sorted_points], dtype=np.float32)
    ys = np.array([p[1] for p in sorted_points], dtype=np.float32)

    if curve_mode == "cubic" and len(xs) >= 3:
        lut = natural_cubic_spline(xs, ys, np.arange(256, dtype=np.float32))
    else:
        lut = np.interp(np.arange(256, dtype=np.float32), xs, ys)

    return np.clip(lut, 0, 255).astype(np.uint8)


def natural_cubic_spline(xs: np.ndarray, ys: np.ndarray, query_xs: np.ndarray) -> np.ndarray:
    n = len(xs)
    h = np.diff(xs)
    if np.any(h <= 0):
        return np.interp(query_xs, xs, ys)

    alpha = np.zeros(n, dtype=np.float32)
    for i in range(1, n - 1):
        alpha[i] = (3.0 / h[i]) * (ys[i + 1] - ys[i]) - (3.0 / h[i - 1]) * (ys[i] - ys[i - 1])

    l = np.ones(n, dtype=np.float32)
    mu = np.zeros(n, dtype=np.float32)
    z = np.zeros(n, dtype=np.float32)

    for i in range(1, n - 1):
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        if abs(l[i]) < 1e-6:
            return np.interp(query_xs, xs, ys)
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    b = np.zeros(n - 1, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    d = np.zeros(n - 1, dtype=np.float32)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = ((ys[j + 1] - ys[j]) / h[j]) - (h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0)
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    indices = np.searchsorted(xs, query_xs, side="right") - 1
    indices = np.clip(indices, 0, n - 2)
    dx = query_xs - xs[indices]
    return ys[indices] + b[indices] * dx + c[indices] * dx ** 2 + d[indices] * dx ** 3


def apply_curves_lut(img: np.ndarray, points, channels=None, curve_mode="linear") -> np.ndarray:
    if img is None:
        return None

    if channels is not None and len(channels) == 0:
        return img.copy()

    lut = build_curve_lut(points, curve_mode)
    if img.ndim == 2:
        return cv2.LUT(img, lut)

    out = img.copy()
    channel_map = {"b": 0, "g": 1, "r": 2}
    selected_channels = channels if channels is not None else ("b", "g", "r")
    for channel_name in selected_channels:
        idx = channel_map.get(channel_name.lower())
        if idx is not None and idx < out.shape[2]:
            out[:, :, idx] = cv2.LUT(out[:, :, idx], lut)

    return out


class HistogramWidget(QFrame):
    zoomChanged = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()

        self.setMinimumHeight(350)
        self.setMouseTracking(True)

        self.hist_r = np.zeros(256)
        self.hist_g = np.zeros(256)
        self.hist_b = np.zeros(256)

        self.check_r = None
        self.check_g = None
        self.check_b = None

        self._zoom_factor = 1.0
        self._zoom_min = 1.0
        self._zoom_max = 16.0
        self._view_center = 127.5
        self._panning = False
        self._last_pan_x = 0

    def set_image(self, img):
        if img is None:
            return

        b, g, r = cv2.split(img)

        self.hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
        self.hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
        self.hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()

        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()

        p.fillRect(0, 0, w, h, QColor("#121212"))

        grid_pen = QPen(QColor("#282828"))
        grid_pen.setStyle(Qt.DashLine)
        grid_pen.setWidth(1)
        p.setPen(grid_pen)
        for i in range(1, 4):
            x = int(w * i / 4)
            y = int(h * i / 4)
            p.drawLine(x, 8, x, h - 8)
            p.drawLine(8, y, w - 8, y)

        channels = []
        if self.check_r is None or self.check_r.isChecked():
            channels.append((self.hist_r, QColor(255, 85, 85, 160)))
        if self.check_g is None or self.check_g.isChecked():
            channels.append((self.hist_g, QColor(85, 255, 85, 160)))
        if self.check_b is None or self.check_b.isChecked():
            channels.append((self.hist_b, QColor(85, 153, 255, 160)))

        view_min, view_max = self._visible_range()
        index_min = max(0, int(math.floor(view_min)))
        index_max = min(255, int(math.ceil(view_max)))

        max_candidates = [1.0]
        for hist, _ in channels:
            if index_max >= index_min:
                max_candidates.append(float(np.max(hist[index_min:index_max + 1])))
        max_val = max(max_candidates)

        baseline = h - 8
        top_pad = 8
        graph_height = max(1, baseline - top_pad)
        view_span = max(1e-6, view_max - view_min)

        for hist, color in channels:
            points = [QPointF(0.0, baseline)]
            for i in range(index_min, index_max + 1):
                x = (w - 1) * ((i - view_min) / view_span)
                y = baseline - float(hist[i] / max_val) * graph_height
                points.append(QPointF(x, y))
            points.append(QPointF(w - 1, baseline))
            poly = QPolygonF(points)
            p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 220), 1.2))
            p.setBrush(color)
            p.drawPolygon(poly)
            p.setBrush(Qt.NoBrush)

        p.setPen(QPen(QColor("#d0d0d0"), 1))
        p.drawText(10, 20, f"Zoom: {self._zoom_factor:.2f}x")

    def _visible_range(self):
        span = max(1.0, 255.0 / float(self._zoom_factor))
        half = span / 2.0
        left = self._view_center - half
        right = self._view_center + half
        if left < 0.0:
            right -= left
            left = 0.0
        if right > 255.0:
            left -= (right - 255.0)
            right = 255.0
        left = max(0.0, left)
        right = min(255.0, right)
        if right - left < 1.0:
            right = min(255.0, left + 1.0)
        return left, right

    def _clamp_center(self):
        span = max(1.0, 255.0 / float(self._zoom_factor))
        half = span / 2.0
        self._view_center = float(np.clip(self._view_center, half, 255.0 - half))

    def _set_zoom(self, zoom_value, anchor_ratio=0.5):
        anchor_ratio = float(np.clip(anchor_ratio, 0.0, 1.0))
        old_min, old_max = self._visible_range()
        old_span = max(1e-6, old_max - old_min)
        anchor_bin = old_min + anchor_ratio * old_span

        new_zoom = float(np.clip(zoom_value, self._zoom_min, self._zoom_max))
        if abs(new_zoom - self._zoom_factor) < 1e-6:
            return
        self._zoom_factor = new_zoom

        new_span = max(1.0, 255.0 / float(self._zoom_factor))
        self._view_center = anchor_bin - (anchor_ratio - 0.5) * new_span
        self._clamp_center()
        self.zoomChanged.emit(self._zoom_factor, self._view_center)
        self.update()

    def zoom_in(self, anchor_ratio=0.5):
        self._set_zoom(self._zoom_factor * 1.2, anchor_ratio=anchor_ratio)

    def zoom_out(self, anchor_ratio=0.5):
        self._set_zoom(self._zoom_factor / 1.2, anchor_ratio=anchor_ratio)

    def reset_zoom(self):
        changed = abs(self._zoom_factor - 1.0) > 1e-6
        self._zoom_factor = 1.0
        self._view_center = 127.5
        if changed:
            self.zoomChanged.emit(self._zoom_factor, self._view_center)
        self.update()

    def get_zoom_factor(self):
        return float(self._zoom_factor)

    def get_zoom_state(self):
        return float(self._zoom_factor), float(self._view_center)

    def get_visible_range(self):
        return self._visible_range()

    def set_visible_range(self, left_value, right_value, emit_signal=False):
        left_value = float(np.clip(left_value, 0.0, 254.0))
        right_value = float(np.clip(right_value, left_value + 1.0, 255.0))
        span = max(1.0, right_value - left_value)
        zoom_factor = float(np.clip(255.0 / span, self._zoom_min, self._zoom_max))
        view_center = (left_value + right_value) * 0.5
        previous_zoom, previous_center = self.get_zoom_state()
        self._zoom_factor = zoom_factor
        self._view_center = view_center
        self._clamp_center()
        changed = (
            abs(previous_zoom - self._zoom_factor) > 1e-6
            or abs(previous_center - self._view_center) > 1e-6
        )
        if emit_signal and changed:
            self.zoomChanged.emit(self._zoom_factor, self._view_center)
        self.update()

    def set_zoom_state(self, zoom_factor, view_center=None, emit_signal=False):
        self._zoom_factor = float(np.clip(zoom_factor, self._zoom_min, self._zoom_max))
        if view_center is not None:
            self._view_center = float(view_center)
        self._clamp_center()
        if emit_signal:
            self.zoomChanged.emit(self._zoom_factor, self._view_center)
        self.update()

    def wheelEvent(self, event):
        if self.width() <= 1:
            return
        anchor_ratio = float(np.clip(event.position().x() / max(1.0, self.width() - 1), 0.0, 1.0))
        if event.angleDelta().y() > 0:
            self.zoom_in(anchor_ratio=anchor_ratio)
        elif event.angleDelta().y() < 0:
            self.zoom_out(anchor_ratio=anchor_ratio)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._zoom_factor > 1.0:
            self._panning = True
            self._last_pan_x = int(event.x())
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            dx = int(event.x()) - self._last_pan_x
            self._last_pan_x = int(event.x())
            if self.width() > 1:
                view_min, view_max = self._visible_range()
                span = max(1e-6, view_max - view_min)
                bins_per_px = span / float(max(1, self.width() - 1))
                self._view_center -= dx * bins_per_px
                self._clamp_center()
                self.zoomChanged.emit(self._zoom_factor, self._view_center)
                self.update()
            event.accept()
            return
        if self._zoom_factor > 1.0:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() == Qt.LeftButton:
            self._panning = False
            if self._zoom_factor > 1.0:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if not self._panning:
            self.unsetCursor()
        super().leaveEvent(event)


class LevelsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)

        self.setWindowTitle("Levels")
        self.setMinimumWidth(520)
        self.setMinimumHeight(420)

        self.parent = parent
        self._updating_zoom_slider = False
        self._session_snapshot = None
        self._session_dirty = False

        main_layout = QVBoxLayout(self)
        apply_standard_layout_margins(main_layout)

        # ---------- TOP BAR ----------
        top_bar = QFrame()
        top_bar.setFixedHeight(42)

        top_layout = QHBoxLayout(top_bar)

        top_layout.setContentsMargins(10, 5, 10, 5)
        top_layout.setSpacing(15)

        self.check_r = QCheckBox("R")
        self.check_g = QCheckBox("G")
        self.check_b = QCheckBox("B")

        self.check_r.setChecked(True)
        self.check_g.setChecked(True)
        self.check_b.setChecked(True)

        self.check_r.setStyleSheet("color: #ff5555; font-weight: bold;")
        self.check_g.setStyleSheet("color: #55ff55; font-weight: bold;")
        self.check_b.setStyleSheet("color: #5599ff; font-weight: bold;")

        top_layout.addWidget(self.check_r)
        top_layout.addWidget(self.check_g)
        top_layout.addWidget(self.check_b)

        top_layout.addStretch(1)

        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(30)
        self.lbl_zoom = QLabel("1.00x")
        self.lbl_zoom.setMinimumWidth(64)
        self.lbl_zoom.setAlignment(Qt.AlignCenter)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(30)
        self.btn_zoom_reset = QPushButton("Reset Zoom")

        top_layout.addWidget(self.btn_zoom_out)
        top_layout.addWidget(self.lbl_zoom)
        top_layout.addWidget(self.btn_zoom_in)
        top_layout.addWidget(self.btn_zoom_reset)

        main_layout.addWidget(top_bar)

        self.zoom_slider = ZoomRangeSlider()
        main_layout.addWidget(self.zoom_slider)

        # ---------- LEVELS ----------
        self.levels_widget = LevelsWidget(
            on_change_callback=self._on_levels_changed
        )
        self.levels_widget.check_r = self.check_r
        self.levels_widget.check_g = self.check_g
        self.levels_widget.check_b = self.check_b

        main_layout.addWidget(self.levels_widget, 1)

        levels_spin_layout = QHBoxLayout()
        self.spin_black = QSpinBox()
        self.spin_black.setRange(0, 254)
        self.spin_black.setValue(self.levels_widget.black)
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.1, 5.0)
        self.spin_gamma.setDecimals(2)
        self.spin_gamma.setSingleStep(0.05)
        self.spin_gamma.setValue(self.levels_widget.gamma)
        self.spin_white = QSpinBox()
        self.spin_white.setRange(1, 255)
        self.spin_white.setValue(self.levels_widget.white)
        levels_spin_layout.addWidget(QLabel("Black"))
        levels_spin_layout.addWidget(self.spin_black)
        levels_spin_layout.addWidget(QLabel("Gray"))
        levels_spin_layout.addWidget(self.spin_gamma)
        levels_spin_layout.addWidget(QLabel("White"))
        levels_spin_layout.addWidget(self.spin_white)
        levels_spin_layout.addStretch(1)
        main_layout.addLayout(levels_spin_layout)

        self.spin_black.valueChanged.connect(self._set_black)
        self.spin_gamma.valueChanged.connect(self._set_gamma)
        self.spin_white.valueChanged.connect(self._set_white)

        # ---------- REFRESH ----------
        self.check_r.stateChanged.connect(lambda _state: self.levels_widget.update())
        self.check_g.stateChanged.connect(lambda _state: self.levels_widget.update())
        self.check_b.stateChanged.connect(lambda _state: self.levels_widget.update())
        self.check_r.stateChanged.connect(lambda _state: self._on_levels_changed())
        self.check_g.stateChanged.connect(lambda _state: self._on_levels_changed())
        self.check_b.stateChanged.connect(lambda _state: self._on_levels_changed())
        self.btn_zoom_in.clicked.connect(self.levels_widget.zoom_in)
        self.btn_zoom_out.clicked.connect(self.levels_widget.zoom_out)
        self.btn_zoom_reset.clicked.connect(self.levels_widget.reset_zoom)
        self.zoom_slider.rangeChanged.connect(self._on_zoom_slider_changed)
        self.levels_widget.zoomChanged.connect(self._update_zoom_controls)
        self._update_zoom_controls(*self.levels_widget.get_zoom_state())

        # ---------- BUTTONS ----------
        btn_layout = QHBoxLayout()

        self.btn_ok = QPushButton("OK")
        self.btn_apply = QPushButton("Apply")
        self.btn_close = QPushButton("Close")

        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_apply)
        btn_layout.addWidget(self.btn_close)

        main_layout.addLayout(btn_layout)

        # ---------- ACTIONS ----------
        self.btn_close.clicked.connect(self.close)

        self.btn_apply.clicked.connect(self._apply)

        self.btn_ok.clicked.connect(self._ok)

    # ---------- AUTO REFRESH ----------
    def _on_levels_changed(self):
        self._sync_level_spins()
        self._session_dirty = True

        if self.parent is not None:
            self.parent.apply_full_processing()

    def _capture_snapshot(self):
        return {
            "black": int(self.levels_widget.black),
            "gamma": float(self.levels_widget.gamma),
            "white": int(self.levels_widget.white),
            "r": bool(self.check_r.isChecked()),
            "g": bool(self.check_g.isChecked()),
            "b": bool(self.check_b.isChecked()),
        }

    def _apply_snapshot(self, snapshot):
        if not isinstance(snapshot, dict):
            return

        self.check_r.blockSignals(True)
        self.check_g.blockSignals(True)
        self.check_b.blockSignals(True)
        self.check_r.setChecked(bool(snapshot.get("r", True)))
        self.check_g.setChecked(bool(snapshot.get("g", True)))
        self.check_b.setChecked(bool(snapshot.get("b", True)))
        self.check_r.blockSignals(False)
        self.check_g.blockSignals(False)
        self.check_b.blockSignals(False)

        self.levels_widget.black = int(snapshot.get("black", 0))
        self.levels_widget.gamma = float(snapshot.get("gamma", 1.0))
        self.levels_widget.white = int(snapshot.get("white", 255))
        self.levels_widget.update()
        self._sync_level_spins()

    def showEvent(self, event):
        self._session_snapshot = self._capture_snapshot()
        self._session_dirty = False
        super().showEvent(event)

    def closeEvent(self, event):
        if self._session_dirty and self._session_snapshot is not None:
            self._apply_snapshot(self._session_snapshot)
            if self.parent is not None:
                self.parent.apply_full_processing()
                self.parent.log("Levels canceled.")
        self._session_dirty = False
        super().closeEvent(event)

    def _sync_level_spins(self):
        self.spin_black.blockSignals(True)
        self.spin_gamma.blockSignals(True)
        self.spin_white.blockSignals(True)
        self.spin_black.setValue(self.levels_widget.black)
        self.spin_gamma.setValue(self.levels_widget.gamma)
        self.spin_white.setValue(self.levels_widget.white)
        self.spin_black.blockSignals(False)
        self.spin_gamma.blockSignals(False)
        self.spin_white.blockSignals(False)

    def _update_zoom_label(self, zoom_value, _center_value):
        self.lbl_zoom.setText(f"{float(zoom_value):.2f}x")

    def _sync_zoom_slider_from_widget(self):
        left_value, right_value = self.levels_widget.get_visible_range()
        self._updating_zoom_slider = True
        try:
            self.zoom_slider.set_values(left_value, right_value, emit_signal=False)
        finally:
            self._updating_zoom_slider = False

    def _update_zoom_controls(self, zoom_value, center_value):
        self._update_zoom_label(zoom_value, center_value)
        self._sync_zoom_slider_from_widget()

    def _on_zoom_slider_changed(self, left_value, right_value):
        if self._updating_zoom_slider:
            return
        self.levels_widget.set_visible_range(left_value, right_value, emit_signal=True)

    def apply_external_zoom_state(self, zoom_factor, view_center):
        self.levels_widget.set_zoom_state(zoom_factor, view_center, emit_signal=False)
        self._update_zoom_controls(zoom_factor, view_center)

    def _set_black(self, value):
        self.levels_widget.black = min(value, self.levels_widget.white - 1)
        self.levels_widget.update()
        self._on_levels_changed()

    def _set_gamma(self, value):
        self.levels_widget.gamma = float(value)
        self.levels_widget.update()
        self._on_levels_changed()

    def _set_white(self, value):
        self.levels_widget.white = max(value, self.levels_widget.black + 1)
        self.levels_widget.update()
        self._on_levels_changed()

    # ---------- APPLY ----------
    def _apply(self):

        if self.parent is not None:
            self.parent.apply_full_processing()
            if self.parent.processed_img is not None:
                self.levels_widget.set_image(self.parent.processed_img)
            levels = self.levels_widget.get_params()
            level_str = f"B:{levels['black']} Îł:{levels['gamma']:.1f} W:{levels['white']}"
            self.parent.add_thumbnail(f"Levels ({level_str})", self.parent.processed_img)
            self.parent.log("Levels applied.")
        self._session_snapshot = self._capture_snapshot()
        self._session_dirty = False

    # ---------- OK ----------
    def _ok(self):

        if self.parent is not None:
            self.parent.apply_full_processing()
            if self.parent.processed_img is not None:
                self.levels_widget.set_image(self.parent.processed_img)
            levels = self.levels_widget.get_params()
            level_str = f"B:{levels['black']} Îł:{levels['gamma']:.1f} W:{levels['white']}"
            self.parent.add_thumbnail(f"Levels ({level_str})", self.parent.processed_img)
            self.parent.log("Levels applied.")

        self._session_snapshot = self._capture_snapshot()
        self._session_dirty = False

        self.close()

class LevelsWidget(QFrame):
    zoomChanged = pyqtSignal(float, float)

    def __init__(self, on_change_callback=None):
        super().__init__()

        self.on_change_callback = on_change_callback

        self.setMinimumHeight(350)

        self.black = 0
        self.gamma = 1.0
        self.white = 255

        self.dragging = None

        self.hist_r = np.zeros(256)
        self.hist_g = np.zeros(256)
        self.hist_b = np.zeros(256)

        # checkboxy przypinane z LevelsWindow
        self.check_r = None
        self.check_g = None
        self.check_b = None

        self._zoom_factor = 1.0
        self._zoom_min = 1.0
        self._zoom_max = 16.0
        self._view_center = 127.5
        self._panning = False
        self._last_pan_x = 0

    def set_image(self, img):
        if img is None:
            return

        b, g, r = cv2.split(img)

        self.hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
        self.hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
        self.hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()

        self.update()

    def _gamma_to_norm(self):
        return float(np.clip(0.5 - (np.log2(max(0.1, self.gamma)) / 4.0), 0.0, 1.0))

    def _norm_to_gamma(self, norm):
        return float(np.clip(2.0 ** ((0.5 - norm) * 4.0), 0.1, 5.0))

    def _plot_rect(self):
        left = 12
        top = 8
        right = max(left + 1, self.width() - 12)
        bottom = max(top + 1, self.height() - 40)
        return left, top, right, bottom

    def _visible_range(self):
        span = max(1.0, 255.0 / float(self._zoom_factor))
        half = span / 2.0
        left = self._view_center - half
        right = self._view_center + half
        if left < 0.0:
            right -= left
            left = 0.0
        if right > 255.0:
            left -= (right - 255.0)
            right = 255.0
        left = max(0.0, left)
        right = min(255.0, right)
        if right - left < 1.0:
            right = min(255.0, left + 1.0)
        return left, right

    def _clamp_center(self):
        span = max(1.0, 255.0 / float(self._zoom_factor))
        half = span / 2.0
        self._view_center = float(np.clip(self._view_center, half, 255.0 - half))

    def _set_zoom(self, zoom_value, anchor_ratio=0.5):
        anchor_ratio = float(np.clip(anchor_ratio, 0.0, 1.0))
        old_min, old_max = self._visible_range()
        old_span = max(1e-6, old_max - old_min)
        anchor_bin = old_min + anchor_ratio * old_span

        new_zoom = float(np.clip(zoom_value, self._zoom_min, self._zoom_max))
        if abs(new_zoom - self._zoom_factor) < 1e-6:
            return
        self._zoom_factor = new_zoom

        new_span = max(1.0, 255.0 / float(self._zoom_factor))
        self._view_center = anchor_bin - (anchor_ratio - 0.5) * new_span
        self._clamp_center()
        self.zoomChanged.emit(self._zoom_factor, self._view_center)
        self.update()

    def zoom_in(self, anchor_ratio=0.5):
        self._set_zoom(self._zoom_factor * 1.2, anchor_ratio=anchor_ratio)

    def zoom_out(self, anchor_ratio=0.5):
        self._set_zoom(self._zoom_factor / 1.2, anchor_ratio=anchor_ratio)

    def reset_zoom(self):
        changed = abs(self._zoom_factor - 1.0) > 1e-6
        self._zoom_factor = 1.0
        self._view_center = 127.5
        if changed:
            self.zoomChanged.emit(self._zoom_factor, self._view_center)
        self.update()

    def get_zoom_state(self):
        return float(self._zoom_factor), float(self._view_center)

    def get_visible_range(self):
        return self._visible_range()

    def set_visible_range(self, left_value, right_value, emit_signal=False):
        left_value = float(np.clip(left_value, 0.0, 254.0))
        right_value = float(np.clip(right_value, left_value + 1.0, 255.0))
        span = max(1.0, right_value - left_value)
        zoom_factor = float(np.clip(255.0 / span, self._zoom_min, self._zoom_max))
        view_center = (left_value + right_value) * 0.5
        previous_zoom, previous_center = self.get_zoom_state()
        self._zoom_factor = zoom_factor
        self._view_center = view_center
        self._clamp_center()
        changed = (
            abs(previous_zoom - self._zoom_factor) > 1e-6
            or abs(previous_center - self._view_center) > 1e-6
        )
        if emit_signal and changed:
            self.zoomChanged.emit(self._zoom_factor, self._view_center)
        self.update()

    def set_zoom_state(self, zoom_factor, view_center=None, emit_signal=False):
        self._zoom_factor = float(np.clip(zoom_factor, self._zoom_min, self._zoom_max))
        if view_center is not None:
            self._view_center = float(view_center)
        self._clamp_center()
        if emit_signal:
            self.zoomChanged.emit(self._zoom_factor, self._view_center)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()

        p.fillRect(0, 0, w, h, QColor("#121212"))

        left, top, right, bottom = self._plot_rect()
        histogram_bottom = h - 40

        grid_pen = QPen(QColor("#282828"))
        grid_pen.setStyle(Qt.DashLine)
        grid_pen.setWidth(1)
        p.setPen(grid_pen)
        for i in range(1, 4):
            x = left + int((right - left) * i / 4)
            y = top + int((histogram_bottom - top) * i / 4)
            p.drawLine(x, top, x, histogram_bottom)
            p.drawLine(left, y, right, y)

        channels = []
        if self.check_r is None or self.check_r.isChecked():
            channels.append((self.hist_r, QColor(255, 85, 85, 160)))
        if self.check_g is None or self.check_g.isChecked():
            channels.append((self.hist_g, QColor(85, 255, 85, 160)))
        if self.check_b is None or self.check_b.isChecked():
            channels.append((self.hist_b, QColor(85, 153, 255, 160)))

        view_min, view_max = self._visible_range()
        index_min = max(0, int(math.floor(view_min)))
        index_max = min(255, int(math.ceil(view_max)))
        view_span = max(1e-6, view_max - view_min)

        max_candidates = [1.0]
        for hist, _ in channels:
            if index_max >= index_min:
                max_candidates.append(float(np.max(hist[index_min:index_max + 1])))
        max_val = max(max_candidates)

        graph_height = max(1, histogram_bottom - top - 6)
        for hist, color in channels:
            points = [QPointF(left, histogram_bottom)]
            for i in range(index_min, index_max + 1):
                x = left + (right - left) * ((i - view_min) / view_span)
                y = histogram_bottom - float(hist[i] / max_val) * graph_height
                points.append(QPointF(x, y))
            points.append(QPointF(right, histogram_bottom))
            poly = QPolygonF(points)
            p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 220), 1.2))
            p.setBrush(color)
            p.drawPolygon(poly)
            p.setBrush(Qt.NoBrush)

        # ---------- photoshop triangles ----------
        def draw_triangle(x, mode="black"):
            size = 8
            y = h - 10

            poly = QPolygonF([
                QPointF(x - size, y - size),
                QPointF(x + size, y - size),
                QPointF(x, y)
            ])

            if mode == "black":
                brush = QColor(20, 20, 20)

            elif mode == "white":
                brush = QColor(245, 245, 245)

            else:
                brush = QColor(190, 190, 190)

            pen = QPen(QColor(0, 0, 0))
            pen.setWidth(1)

            p.setPen(pen)
            p.setBrush(brush)

            p.drawPolygon(poly)

        # ---------- positions ----------
        bx = int(left + (right - left) * ((self.black - view_min) / view_span))
        wx = int(left + (right - left) * ((self.white - view_min) / view_span))

        gamma_norm = self._gamma_to_norm()
        gamma_bin = float(np.clip(gamma_norm * 255.0, 0.0, 255.0))
        gx = int(left + (right - left) * ((gamma_bin - view_min) / view_span))

        p.setPen(QPen(QColor("#3d3d3d"), 1))
        p.drawLine(left, histogram_bottom, right, top)

        draw_triangle(bx, "black")
        draw_triangle(gx, "gamma")
        draw_triangle(wx, "white")

    def mousePressEvent(self, e):
        left, _top, right, _bottom = self._plot_rect()
        if e.button() == Qt.MiddleButton and self._zoom_factor > 1.0 and left <= e.x() <= right:
            self._panning = True
            self._last_pan_x = int(e.x())
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return

        x = e.x()
        view_min, view_max = self._visible_range()
        view_span = max(1e-6, view_max - view_min)

        bx = int(left + (right - left) * ((self.black - view_min) / view_span))
        wx = int(left + (right - left) * ((self.white - view_min) / view_span))

        gamma_norm = self._gamma_to_norm()
        gamma_bin = float(np.clip(gamma_norm * 255.0, 0.0, 255.0))
        gx = int(left + (right - left) * ((gamma_bin - view_min) / view_span))

        if abs(x - bx) < 12:
            self.dragging = "black"

        elif abs(x - gx) < 12:
            self.dragging = "gamma"

        elif abs(x - wx) < 12:
            self.dragging = "white"

    def mouseMoveEvent(self, e):
        left, _top, right, _bottom = self._plot_rect()
        if self._panning:
            dx = int(e.x()) - self._last_pan_x
            self._last_pan_x = int(e.x())
            if right > left:
                view_min, view_max = self._visible_range()
                span = max(1e-6, view_max - view_min)
                bins_per_px = span / float(max(1, right - left))
                self._view_center -= dx * bins_per_px
                self._clamp_center()
                self.zoomChanged.emit(self._zoom_factor, self._view_center)
                self.update()
            e.accept()
            return

        if not self.dragging:
            if self._zoom_factor > 1.0 and left <= e.x() <= right:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()
            return

        view_min, view_max = self._visible_range()
        view_span = max(1e-6, view_max - view_min)
        x = int(np.clip(e.x(), left, right))
        val = int(round(view_min + ((x - left) / max(1, right - left)) * view_span))
        val = max(0, min(255, val))

        if self.dragging == "black":
            self.black = min(val, self.white - 1)

        elif self.dragging == "white":
            self.white = max(val, self.black + 1)

        elif self.dragging == "gamma":
            gamma_norm = np.clip(val / 255.0, 0.0, 1.0)
            self.gamma = self._norm_to_gamma(gamma_norm)

        self.update()

        if self.on_change_callback:
            self.on_change_callback()

    def mouseReleaseEvent(self, e):
        if self._panning and e.button() == Qt.MiddleButton:
            self._panning = False
            self.unsetCursor()
            e.accept()
            return
        self.dragging = None

    def leaveEvent(self, event):
        if not self._panning:
            self.unsetCursor()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        left, _top, right, _bottom = self._plot_rect()
        if right <= left:
            return
        x = float(np.clip(event.position().x(), left, right))
        anchor_ratio = float(np.clip((x - left) / max(1.0, right - left), 0.0, 1.0))
        if event.angleDelta().y() > 0:
            self.zoom_in(anchor_ratio=anchor_ratio)
        elif event.angleDelta().y() < 0:
            self.zoom_out(anchor_ratio=anchor_ratio)
        event.accept()

    def get_params(self):
        channels = []
        if self.check_r is None or self.check_r.isChecked():
            channels.append("r")
        if self.check_g is None or self.check_g.isChecked():
            channels.append("g")
        if self.check_b is None or self.check_b.isChecked():
            channels.append("b")

        return {
            "black": self.black,
            "gamma": self.gamma,
            "white": self.white,
            "channels": channels
        }
        
class CurvesWidget(QFrame):
    zoomChanged = pyqtSignal(float, float)

    def __init__(self, on_change_callback=None):
        super().__init__()
        self.on_change_callback = on_change_callback
        self.points = []
        self.drag_index = None
        self.curve_mode = "linear"
        self.hist_r = np.zeros(256)
        self.hist_g = np.zeros(256)
        self.hist_b = np.zeros(256)
        self.check_r = None
        self.check_g = None
        self.check_b = None
        self.setMinimumHeight(320)
        self._zoom_factor = 1.0
        self._zoom_min = 1.0
        self._zoom_max = 16.0
        self._view_center = 127.5
        self._panning = False
        self._last_pan_x = 0

    def reset(self, notify=True):
        self.points = []
        self.drag_index = None
        self.update()
        if notify and self.on_change_callback:
            self.on_change_callback()

    def set_curve_mode(self, mode, notify=True):
        self.curve_mode = "cubic" if mode == "cubic" else "linear"
        self.update()
        if notify and self.on_change_callback:
            self.on_change_callback()

    def set_image(self, img):
        if img is None:
            return
        if img.ndim == 2:
            self.hist_r = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            self.hist_g = self.hist_r.copy()
            self.hist_b = self.hist_r.copy()
        else:
            b, g, r = cv2.split(img)
            self.hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
            self.hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
            self.hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()
        self.update()

    def set_point(self, index, x, y, notify=True):
        if 0 <= index < len(self.points):
            x = int(np.clip(x, 1, 254))
            y = int(np.clip(y, 0, 255))
            self.points[index] = (x, y)
            self.points.sort(key=lambda point: point[0])
            self.drag_index = self.points.index((x, y))
            self.update()
            if notify and self.on_change_callback:
                self.on_change_callback()

    def get_points(self):
        return list(self.points)

    def _plot_rect(self):
        left = 28
        top = 18
        right = max(left + 1, self.width() - 18)
        bottom = max(top + 1, self.height() - 28)
        return left, top, right, bottom

    def _visible_range(self):
        span = max(1.0, 255.0 / float(self._zoom_factor))
        half = span / 2.0
        left = self._view_center - half
        right = self._view_center + half
        if left < 0.0:
            right -= left
            left = 0.0
        if right > 255.0:
            left -= (right - 255.0)
            right = 255.0
        left = max(0.0, left)
        right = min(255.0, right)
        if right - left < 1.0:
            right = min(255.0, left + 1.0)
        return left, right

    def _clamp_center(self):
        span = max(1.0, 255.0 / float(self._zoom_factor))
        half = span / 2.0
        self._view_center = float(np.clip(self._view_center, half, 255.0 - half))

    def _set_zoom(self, zoom_value, anchor_ratio=0.5):
        anchor_ratio = float(np.clip(anchor_ratio, 0.0, 1.0))
        old_min, old_max = self._visible_range()
        old_span = max(1e-6, old_max - old_min)
        anchor_bin = old_min + anchor_ratio * old_span

        new_zoom = float(np.clip(zoom_value, self._zoom_min, self._zoom_max))
        if abs(new_zoom - self._zoom_factor) < 1e-6:
            return
        self._zoom_factor = new_zoom

        new_span = max(1.0, 255.0 / float(self._zoom_factor))
        self._view_center = anchor_bin - (anchor_ratio - 0.5) * new_span
        self._clamp_center()
        self.zoomChanged.emit(self._zoom_factor, self._view_center)
        self.update()

    def zoom_in(self, anchor_ratio=0.5):
        self._set_zoom(self._zoom_factor * 1.2, anchor_ratio=anchor_ratio)

    def zoom_out(self, anchor_ratio=0.5):
        self._set_zoom(self._zoom_factor / 1.2, anchor_ratio=anchor_ratio)

    def reset_zoom(self):
        changed = abs(self._zoom_factor - 1.0) > 1e-6
        self._zoom_factor = 1.0
        self._view_center = 127.5
        if changed:
            self.zoomChanged.emit(self._zoom_factor, self._view_center)
        self.update()

    def get_zoom_state(self):
        return float(self._zoom_factor), float(self._view_center)

    def get_visible_range(self):
        return self._visible_range()

    def set_visible_range(self, left_value, right_value, emit_signal=False):
        left_value = float(np.clip(left_value, 0.0, 254.0))
        right_value = float(np.clip(right_value, left_value + 1.0, 255.0))
        span = max(1.0, right_value - left_value)
        zoom_factor = float(np.clip(255.0 / span, self._zoom_min, self._zoom_max))
        view_center = (left_value + right_value) * 0.5
        previous_zoom, previous_center = self.get_zoom_state()
        self._zoom_factor = zoom_factor
        self._view_center = view_center
        self._clamp_center()
        changed = (
            abs(previous_zoom - self._zoom_factor) > 1e-6
            or abs(previous_center - self._view_center) > 1e-6
        )
        if emit_signal and changed:
            self.zoomChanged.emit(self._zoom_factor, self._view_center)
        self.update()

    def set_zoom_state(self, zoom_factor, view_center=None, emit_signal=False):
        self._zoom_factor = float(np.clip(zoom_factor, self._zoom_min, self._zoom_max))
        if view_center is not None:
            self._view_center = float(view_center)
        self._clamp_center()
        if emit_signal:
            self.zoomChanged.emit(self._zoom_factor, self._view_center)
        self.update()

    def _point_to_pos(self, point):
        left, top, right, bottom = self._plot_rect()
        view_min, view_max = self._visible_range()
        view_span = max(1e-6, view_max - view_min)
        x, y = point
        px = left + ((x - view_min) / view_span) * (right - left)
        py = bottom - (y / 255.0) * (bottom - top)
        return int(px), int(py)

    def _pos_to_value(self, y):
        _left, top, _right, bottom = self._plot_rect()
        value = 255.0 - ((y - top) / max(1, bottom - top)) * 255.0
        return int(np.clip(value, 0, 255))

    def _pos_to_input(self, x):
        left, _top, right, _bottom = self._plot_rect()
        view_min, view_max = self._visible_range()
        view_span = max(1e-6, view_max - view_min)
        clamped_x = float(np.clip(x, left, right))
        value = view_min + ((clamped_x - left) / max(1, right - left)) * view_span
        return int(np.clip(value, 1, 254))

    def _curve_value_at_x(self, x):
        lut = build_curve_lut(self.points, self.curve_mode)
        return int(lut[int(np.clip(x, 0, 255))])

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        left, top, right, bottom = self._plot_rect()

        p.fillRect(0, 0, self.width(), self.height(), QColor("#121212"))

        grid_pen = QPen(QColor("#282828"))
        grid_pen.setStyle(Qt.DashLine)
        grid_pen.setWidth(1)
        p.setPen(grid_pen)
        for i in range(5):
            x = left + int((right - left) * i / 4)
            y = top + int((bottom - top) * i / 4)
            p.drawLine(x, top, x, bottom)
            p.drawLine(left, y, right, y)

        channels = []
        if self.check_r is None or self.check_r.isChecked():
            channels.append((self.hist_r, QColor(255, 85, 85, 160)))
        if self.check_g is None or self.check_g.isChecked():
            channels.append((self.hist_g, QColor(85, 255, 85, 160)))
        if self.check_b is None or self.check_b.isChecked():
            channels.append((self.hist_b, QColor(85, 153, 255, 160)))

        view_min, view_max = self._visible_range()
        index_min = max(0, int(math.floor(view_min)))
        index_max = min(255, int(math.ceil(view_max)))
        view_span = max(1e-6, view_max - view_min)

        max_candidates = [1.0]
        for hist, _ in channels:
            if index_max >= index_min:
                max_candidates.append(float(np.max(hist[index_min:index_max + 1])))
        max_hist = max(max_candidates)
        graph_height = max(1, bottom - top - 4)

        for hist, color in channels:
            points = [QPointF(left, bottom)]
            for i in range(index_min, index_max + 1):
                x = left + (right - left) * ((i - view_min) / view_span)
                y = bottom - float(hist[i] / max_hist) * graph_height
                points.append(QPointF(x, y))
            points.append(QPointF(right, bottom))
            poly = QPolygonF(points)
            p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 220), 1.2))
            p.setBrush(color)
            p.drawPolygon(poly)
            p.setBrush(Qt.NoBrush)

        p.setPen(QPen(QColor("#3d3d3d"), 1))
        p.drawLine(left, bottom, right, top)

        lut = build_curve_lut(self.points, self.curve_mode)
        path = QPainterPath()
        x0 = left + int((right - left) * ((index_min - view_min) / view_span))
        y0 = bottom - int((lut[index_min] / 255.0) * (bottom - top))
        path.moveTo(x0, y0)
        for i in range(index_min + 1, index_max + 1):
            x = left + int((right - left) * ((i - view_min) / view_span))
            y = bottom - int((lut[i] / 255.0) * (bottom - top))
            path.lineTo(x, y)
        p.setPen(QPen(QColor(126, 211, 252), 2))
        p.drawPath(path)

        for index, point in enumerate(self.points):
            if not (view_min <= point[0] <= view_max):
                continue
            x, y = self._point_to_pos(point)
            color = QColor(255, 255, 255) if index != self.drag_index else QColor(248, 214, 109)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(color, 2))
            p.drawRect(x - 6, y - 6, 12, 12)

    def mousePressEvent(self, e):
        left, _top, right, _bottom = self._plot_rect()
        if e.button() == Qt.MiddleButton and self._zoom_factor > 1.0 and left <= e.x() <= right:
            self._panning = True
            self._last_pan_x = int(e.x())
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return

        if e.button() == Qt.RightButton:
            nearest = self._nearest_point_index(e.x(), e.y())
            if nearest is not None:
                self.points.pop(nearest)
                self.drag_index = None
                self.update()
                if self.on_change_callback:
                    self.on_change_callback()
            return

        nearest = None
        nearest_dist = 999
        for index, point in enumerate(self.points):
            x, y = self._point_to_pos(point)
            dist = abs(e.x() - x) + abs(e.y() - y)
            if dist < nearest_dist:
                nearest = index
                nearest_dist = dist
        if nearest is not None and nearest_dist <= 22:
            self.drag_index = nearest
            return

        left, top, right, bottom = self._plot_rect()
        if left <= e.x() <= right and top <= e.y() <= bottom:
            input_value = self._pos_to_input(e.x())
            output_value = self._curve_value_at_x(input_value)
            curve_y = self._point_to_pos((input_value, output_value))[1]
            if abs(e.y() - curve_y) > 18:
                return
            self.points.append((input_value, output_value))
            self.points.sort(key=lambda point: point[0])
            self.drag_index = self.points.index((input_value, output_value))
            self.update()
            if self.on_change_callback:
                self.on_change_callback()

    def mouseMoveEvent(self, e):
        left, _top, right, _bottom = self._plot_rect()
        if self._panning:
            dx = int(e.x()) - self._last_pan_x
            self._last_pan_x = int(e.x())
            if right > left:
                view_min, view_max = self._visible_range()
                span = max(1e-6, view_max - view_min)
                bins_per_px = span / float(max(1, right - left))
                self._view_center -= dx * bins_per_px
                self._clamp_center()
                self.zoomChanged.emit(self._zoom_factor, self._view_center)
                self.update()
            e.accept()
            return

        if self.drag_index is None:
            if self._zoom_factor > 1.0 and left <= e.x() <= right:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()
            return
        self.set_point(self.drag_index, self._pos_to_input(e.x()), self._pos_to_value(e.y()))

    def mouseReleaseEvent(self, e):
        if self._panning and e.button() == Qt.MiddleButton:
            self._panning = False
            self.unsetCursor()
            e.accept()
            return
        self.drag_index = None
        self.update()

    def leaveEvent(self, event):
        if not self._panning:
            self.unsetCursor()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        left, _top, right, _bottom = self._plot_rect()
        if right <= left:
            return
        x = float(np.clip(event.position().x(), left, right))
        anchor_ratio = float(np.clip((x - left) / max(1.0, right - left), 0.0, 1.0))
        if event.angleDelta().y() > 0:
            self.zoom_in(anchor_ratio=anchor_ratio)
        elif event.angleDelta().y() < 0:
            self.zoom_out(anchor_ratio=anchor_ratio)
        event.accept()

    def _nearest_point_index(self, x, y):
        nearest = None
        nearest_dist = 999
        for index, point in enumerate(self.points):
            px, py = self._point_to_pos(point)
            dist = abs(x - px) + abs(y - py)
            if dist < nearest_dist:
                nearest = index
                nearest_dist = dist
        return nearest if nearest_dist <= 22 else None


class CurvesWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("Curves (LUT)")
        self.setMinimumWidth(560)
        self.setMinimumHeight(410)
        self.parent = parent
        self._updating_zoom_slider = False
        self._session_snapshot = None
        self._session_dirty = False

        main_layout = QVBoxLayout(self)
        apply_standard_layout_margins(main_layout)

        top_bar = QFrame()
        top_bar.setFixedHeight(42)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 5, 10, 5)

        self.check_r = QCheckBox("R")
        self.check_g = QCheckBox("G")
        self.check_b = QCheckBox("B")
        self.check_r.setChecked(True)
        self.check_g.setChecked(True)
        self.check_b.setChecked(True)
        self.check_r.setStyleSheet("color: #ff5555; font-weight: bold;")
        self.check_g.setStyleSheet("color: #55ff55; font-weight: bold;")
        self.check_b.setStyleSheet("color: #5599ff; font-weight: bold;")
        top_layout.addWidget(self.check_r)
        top_layout.addWidget(self.check_g)
        top_layout.addWidget(self.check_b)
        top_layout.addStretch(1)
        top_layout.addWidget(QLabel("Curve"))
        self.btn_curve_linear = QPushButton("Linear")
        self.btn_curve_linear.setCheckable(True)
        self.btn_curve_cubic = QPushButton("Cubic")
        self.btn_curve_cubic.setCheckable(True)
        top_layout.addWidget(self.btn_curve_linear)
        top_layout.addWidget(self.btn_curve_cubic)
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(30)
        self.lbl_zoom = QLabel("1.00x")
        self.lbl_zoom.setMinimumWidth(64)
        self.lbl_zoom.setAlignment(Qt.AlignCenter)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(30)
        self.btn_zoom_reset = QPushButton("Reset Zoom")
        top_layout.addWidget(self.btn_zoom_out)
        top_layout.addWidget(self.lbl_zoom)
        top_layout.addWidget(self.btn_zoom_in)
        top_layout.addWidget(self.btn_zoom_reset)
        main_layout.addWidget(top_bar)

        self.zoom_slider = ZoomRangeSlider()
        main_layout.addWidget(self.zoom_slider)

        self.curves_widget = CurvesWidget(on_change_callback=self._on_curves_changed)
        self.curves_widget.check_r = self.check_r
        self.curves_widget.check_g = self.check_g
        self.curves_widget.check_b = self.check_b
        main_layout.addWidget(self.curves_widget, 1)

        self.check_r.stateChanged.connect(lambda _state: self._on_curves_changed())
        self.check_g.stateChanged.connect(lambda _state: self._on_curves_changed())
        self.check_b.stateChanged.connect(lambda _state: self._on_curves_changed())
        self.btn_curve_linear.clicked.connect(lambda: self._on_curve_mode_changed("linear"))
        self.btn_curve_cubic.clicked.connect(lambda: self._on_curve_mode_changed("cubic"))
        self.btn_zoom_in.clicked.connect(self.curves_widget.zoom_in)
        self.btn_zoom_out.clicked.connect(self.curves_widget.zoom_out)
        self.btn_zoom_reset.clicked.connect(self.curves_widget.reset_zoom)
        self.zoom_slider.rangeChanged.connect(self._on_zoom_slider_changed)
        self.curves_widget.zoomChanged.connect(self._update_zoom_controls)
        self._update_zoom_controls(*self.curves_widget.get_zoom_state())
        self._on_curve_mode_changed("linear")

        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Reset")
        self.btn_ok = QPushButton("OK")
        self.btn_apply = QPushButton("Apply")
        self.btn_close = QPushButton("Close")
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_apply)
        btn_layout.addWidget(self.btn_close)
        main_layout.addLayout(btn_layout)

        self.btn_reset.clicked.connect(self.reset_curves)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_ok.clicked.connect(self._ok)
        self.btn_close.clicked.connect(self.close)

    def _on_curve_mode_changed(self, mode=None):
        if mode is None:
            mode = self.curves_widget.curve_mode if self.curves_widget.curve_mode in ("linear", "cubic") else "linear"
        mode = "cubic" if mode == "cubic" else "linear"
        self.btn_curve_linear.setChecked(mode == "linear")
        self.btn_curve_cubic.setChecked(mode == "cubic")
        self.curves_widget.set_curve_mode(mode)

    def _on_curves_changed(self):
        self.curves_widget.update()
        self._session_dirty = True
        if self.parent is not None:
            self.parent.apply_full_processing()

    def _capture_snapshot(self):
        return {
            "points": list(self.curves_widget.get_points()),
            "curve_mode": str(self.curves_widget.curve_mode),
            "r": bool(self.check_r.isChecked()),
            "g": bool(self.check_g.isChecked()),
            "b": bool(self.check_b.isChecked()),
        }

    def _apply_snapshot(self, snapshot):
        if not isinstance(snapshot, dict):
            return

        self.check_r.blockSignals(True)
        self.check_g.blockSignals(True)
        self.check_b.blockSignals(True)
        self.check_r.setChecked(bool(snapshot.get("r", True)))
        self.check_g.setChecked(bool(snapshot.get("g", True)))
        self.check_b.setChecked(bool(snapshot.get("b", True)))
        self.check_r.blockSignals(False)
        self.check_g.blockSignals(False)
        self.check_b.blockSignals(False)

        mode = "cubic" if str(snapshot.get("curve_mode", "linear")) == "cubic" else "linear"
        self.btn_curve_linear.setChecked(mode == "linear")
        self.btn_curve_cubic.setChecked(mode == "cubic")
        self.curves_widget.curve_mode = mode
        self.curves_widget.points = [(int(x), int(y)) for x, y in snapshot.get("points", [])]
        self.curves_widget.points.sort(key=lambda point: point[0])
        self.curves_widget.drag_index = None
        self.curves_widget.update()

    def showEvent(self, event):
        self._session_snapshot = self._capture_snapshot()
        self._session_dirty = False
        super().showEvent(event)

    def closeEvent(self, event):
        if self._session_dirty and self._session_snapshot is not None:
            self._apply_snapshot(self._session_snapshot)
            if self.parent is not None:
                self.parent.apply_full_processing()
                self.parent.log("Curves canceled.")
        self._session_dirty = False
        super().closeEvent(event)

    def _update_zoom_label(self, zoom_value, _center_value):
        self.lbl_zoom.setText(f"{float(zoom_value):.2f}x")

    def _sync_zoom_slider_from_widget(self):
        left_value, right_value = self.curves_widget.get_visible_range()
        self._updating_zoom_slider = True
        try:
            self.zoom_slider.set_values(left_value, right_value, emit_signal=False)
        finally:
            self._updating_zoom_slider = False

    def _update_zoom_controls(self, zoom_value, center_value):
        self._update_zoom_label(zoom_value, center_value)
        self._sync_zoom_slider_from_widget()

    def _on_zoom_slider_changed(self, left_value, right_value):
        if self._updating_zoom_slider:
            return
        self.curves_widget.set_visible_range(left_value, right_value, emit_signal=True)

    def apply_external_zoom_state(self, zoom_factor, view_center):
        self.curves_widget.set_zoom_state(zoom_factor, view_center, emit_signal=False)
        self._update_zoom_controls(zoom_factor, view_center)

    def reset_curves(self, notify=True):
        self.curves_widget.reset(notify=notify)
        if notify and self.parent is not None:
            self.parent.log("Curves reset.")

    def set_image(self, img):
        self.curves_widget.set_image(img)

    def get_params(self):
        channels = []
        if self.check_r.isChecked():
            channels.append("r")
        if self.check_g.isChecked():
            channels.append("g")
        if self.check_b.isChecked():
            channels.append("b")
        return {
            "points": self.curves_widget.get_points(),
            "channels": channels,
            "curve_mode": self.curves_widget.curve_mode,
        }

    def _apply(self):
        if self.parent is not None:
            self.parent.apply_full_processing()
            if self.parent.processed_img is not None:
                self.curves_widget.set_image(self.parent.processed_img)
            self.parent.add_thumbnail("Curves Correction", self.parent.processed_img)
            self.parent.log("Curves applied.")
        self._session_snapshot = self._capture_snapshot()
        self._session_dirty = False

    def _ok(self):
        self._apply()
        self.close()


class ZoomRangeSlider(QFrame):
    rangeChanged = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value_min = 0.0
        self._value_max = 255.0
        self._min_span = 1.0
        self._left_value = 0.0
        self._right_value = 255.0
        self._drag_mode = None
        self._drag_start_x = 0
        self._drag_start_left = 0.0
        self._drag_start_right = 255.0
        self._handle_radius = 7
        self.setMinimumHeight(28)
        self.setMaximumHeight(34)
        self.setMouseTracking(True)

    def values(self):
        return float(self._left_value), float(self._right_value)

    def set_bounds(self, value_min, value_max, emit_signal=False):
        value_min = float(value_min)
        value_max = float(value_max)
        if value_max <= value_min:
            return
        self._value_min = value_min
        self._value_max = value_max
        self.set_values(self._left_value, self._right_value, emit_signal=emit_signal)

    def set_min_span(self, min_span, emit_signal=False):
        self._min_span = max(0.0, float(min_span))
        self.set_values(self._left_value, self._right_value, emit_signal=emit_signal)

    def set_values(self, left_value, right_value, emit_signal=False):
        min_span = max(0.0, float(self._min_span))
        left_value = float(np.clip(left_value, self._value_min, self._value_max - min_span))
        right_value = float(np.clip(right_value, left_value + min_span, self._value_max))
        if right_value < left_value:
            right_value = left_value
        changed = (
            abs(self._left_value - left_value) > 1e-6
            or abs(self._right_value - right_value) > 1e-6
        )
        self._left_value = left_value
        self._right_value = right_value
        if changed and emit_signal:
            self.rangeChanged.emit(self._left_value, self._right_value)
        self.update()

    def _track_bounds(self):
        pad = self._handle_radius + 2
        return pad, max(pad + 1, self.width() - pad)

    def _value_to_x(self, value):
        left_x, right_x = self._track_bounds()
        ratio = (value - self._value_min) / max(1e-6, self._value_max - self._value_min)
        return left_x + ratio * (right_x - left_x)

    def _x_to_value(self, x):
        left_x, right_x = self._track_bounds()
        x = float(np.clip(x, left_x, right_x))
        ratio = (x - left_x) / max(1e-6, right_x - left_x)
        return self._value_min + ratio * (self._value_max - self._value_min)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        y = self.height() * 0.5
        left_x, right_x = self._track_bounds()
        left_handle_x = self._value_to_x(self._left_value)
        right_handle_x = self._value_to_x(self._right_value)

        p.setPen(QPen(QColor("#353535"), 4))
        p.drawLine(int(left_x), int(y), int(right_x), int(y))

        p.setPen(QPen(QColor("#6aa9ff"), 4))
        p.drawLine(int(left_handle_x), int(y), int(right_handle_x), int(y))

        p.setPen(QPen(QColor("#d8d8d8"), 1))
        p.setBrush(QColor("#f2f2f2"))
        p.drawEllipse(QPointF(left_handle_x, y), self._handle_radius, self._handle_radius)
        p.drawEllipse(QPointF(right_handle_x, y), self._handle_radius, self._handle_radius)

    def _set_drag_mode(self, mode):
        self._drag_mode = mode
        if mode is None:
            self.unsetCursor()
        elif mode == "range":
            self.setCursor(Qt.ClosedHandCursor)
        else:
            self.setCursor(Qt.SizeHorCursor)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)

        x = float(event.x())
        y = float(event.y())
        center_y = self.height() * 0.5
        left_handle_x = self._value_to_x(self._left_value)
        right_handle_x = self._value_to_x(self._right_value)
        handle_hit = self._handle_radius + 4
        left_hit = (abs(x - left_handle_x) <= handle_hit) and (abs(y - center_y) <= handle_hit)
        right_hit = (abs(x - right_handle_x) <= handle_hit) and (abs(y - center_y) <= handle_hit)

        if left_hit and right_hit:
            self._set_drag_mode("left" if abs(x - left_handle_x) <= abs(x - right_handle_x) else "right")
        elif left_hit:
            self._set_drag_mode("left")
        elif right_hit:
            self._set_drag_mode("right")
        elif left_handle_x < x < right_handle_x:
            self._set_drag_mode("range")
            self._drag_start_x = int(x)
            self._drag_start_left = self._left_value
            self._drag_start_right = self._right_value
        else:
            clicked_value = self._x_to_value(x)
            width = self._right_value - self._left_value
            left_value = np.clip(clicked_value - width * 0.5, self._value_min, self._value_max - width)
            right_value = left_value + width
            self.set_values(left_value, right_value, emit_signal=True)
            self._set_drag_mode("range")
            self._drag_start_x = int(x)
            self._drag_start_left = self._left_value
            self._drag_start_right = self._right_value

        event.accept()

    def mouseMoveEvent(self, event):
        x = float(event.x())
        if self._drag_mode == "left":
            self.set_values(self._x_to_value(x), self._right_value, emit_signal=True)
            event.accept()
            return
        if self._drag_mode == "right":
            self.set_values(self._left_value, self._x_to_value(x), emit_signal=True)
            event.accept()
            return
        if self._drag_mode == "range":
            delta_value = self._x_to_value(x) - self._x_to_value(self._drag_start_x)
            width = self._drag_start_right - self._drag_start_left
            left_value = np.clip(self._drag_start_left + delta_value, self._value_min, self._value_max - width)
            right_value = left_value + width
            self.set_values(left_value, right_value, emit_signal=True)
            event.accept()
            return

        left_handle_x = self._value_to_x(self._left_value)
        right_handle_x = self._value_to_x(self._right_value)
        handle_hit = self._handle_radius + 4
        if abs(x - left_handle_x) <= handle_hit or abs(x - right_handle_x) <= handle_hit:
            self.setCursor(Qt.SizeHorCursor)
        elif left_handle_x < x < right_handle_x:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_mode is not None:
            self._set_drag_mode(None)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if self._drag_mode is None:
            self.unsetCursor()
        super().leaveEvent(event)


class HistogramWindow(QDialog):
    zoomSyncChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self._updating_zoom_slider = False

        self.setWindowTitle("Histogram")
        self.setMinimumSize(100, 100)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        # ---------- checkboxy NAD histogramem ----------
        top_bar = QHBoxLayout()

        self.check_r = QCheckBox("R")
        self.check_g = QCheckBox("G")
        self.check_b = QCheckBox("B")

        self.check_r.setChecked(True)
        self.check_g.setChecked(True)
        self.check_b.setChecked(True)

        self.check_r.setStyleSheet("color: #ff5555; font-weight: bold;")
        self.check_g.setStyleSheet("color: #55ff55; font-weight: bold;")
        self.check_b.setStyleSheet("color: #5599ff; font-weight: bold;")

        top_bar.addWidget(self.check_r)
        top_bar.addWidget(self.check_g)
        top_bar.addWidget(self.check_b)
        top_bar.addStretch(1)

        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(30)
        self.lbl_zoom = QLabel("1.00x")
        self.lbl_zoom.setMinimumWidth(64)
        self.lbl_zoom.setAlignment(Qt.AlignCenter)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(30)
        self.btn_zoom_reset = QPushButton("Reset Zoom")

        top_bar.addWidget(self.btn_zoom_out)
        top_bar.addWidget(self.lbl_zoom)
        top_bar.addWidget(self.btn_zoom_in)
        top_bar.addWidget(self.btn_zoom_reset)

        layout.addLayout(top_bar)

        self.zoom_slider = ZoomRangeSlider()
        layout.addWidget(self.zoom_slider)

        self.check_sync_zoom = QCheckBox("sunchronizuj zoom")
        self.check_sync_zoom.setChecked(True)
        layout.addWidget(self.check_sync_zoom)

        # ---------- histogram widget ----------
        self.histogram_widget = HistogramWidget()
        layout.addWidget(self.histogram_widget)

        # ---------- poĹ‚Ä…czenia ----------
        self.check_r.stateChanged.connect(lambda _state: self.histogram_widget.update())
        self.check_g.stateChanged.connect(lambda _state: self.histogram_widget.update())
        self.check_b.stateChanged.connect(lambda _state: self.histogram_widget.update())

        self.histogram_widget.check_r = self.check_r
        self.histogram_widget.check_g = self.check_g
        self.histogram_widget.check_b = self.check_b

        self.btn_zoom_in.clicked.connect(self.histogram_widget.zoom_in)
        self.btn_zoom_out.clicked.connect(self.histogram_widget.zoom_out)
        self.btn_zoom_reset.clicked.connect(self.histogram_widget.reset_zoom)
        self.zoom_slider.rangeChanged.connect(self._on_zoom_slider_changed)
        self.histogram_widget.zoomChanged.connect(self._update_zoom_controls)
        self.check_sync_zoom.toggled.connect(self._on_sync_zoom_changed)
        self._update_zoom_controls(*self.histogram_widget.get_zoom_state())

    def set_image(self, img):
        self.histogram_widget.set_image(img)

    def _update_zoom_label(self, zoom_value, _center_value):
        self.lbl_zoom.setText(f"{float(zoom_value):.2f}x")

    def _sync_zoom_slider_from_widget(self):
        left_value, right_value = self.histogram_widget.get_visible_range()
        self._updating_zoom_slider = True
        try:
            self.zoom_slider.set_values(left_value, right_value, emit_signal=False)
        finally:
            self._updating_zoom_slider = False

    def _update_zoom_controls(self, zoom_value, center_value):
        self._update_zoom_label(zoom_value, center_value)
        self._sync_zoom_slider_from_widget()

    def _on_zoom_slider_changed(self, left_value, right_value):
        if self._updating_zoom_slider:
            return
        self.histogram_widget.set_visible_range(left_value, right_value, emit_signal=True)

    def _on_sync_zoom_changed(self, _enabled):
        self.zoomSyncChanged.emit(self.is_zoom_sync_enabled())

    def is_zoom_sync_enabled(self):
        return self.check_sync_zoom.isChecked()

    def apply_external_zoom_state(self, zoom_factor, view_center):
        self.histogram_widget.set_zoom_state(zoom_factor, view_center, emit_signal=False)
        self._update_zoom_controls(zoom_factor, view_center)

    def closeEvent(self, event):
        if self.parent() is not None and hasattr(self.parent(), "log"):
            self.parent().log("Histogram window closed.")
        super().closeEvent(event)

class ConsoleCommandInput(QLineEdit):
    COMMANDS = [
        "save",
        "save as",
        "open",
        "curves",
        "levels",
        "histogram",
        "correction",
        "menu",
        "console",
        "magic",
        "star shrink",
        "starnet++",
        "blur",
        "open.blur",
        "undo",
        "redo",
        "help",
        "clear",
        "models",
        "exit",
    ]

    def __init__(self, console_window):
        super().__init__()
        self.console_window = console_window
        self.setAcceptDrops(True)
        self.completer = QCompleter(self.COMMANDS, self)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.completer.activated.connect(self.setText)
        self.setCompleter(self.completer)
        self.textEdited.connect(self._show_completion)

    def _show_completion(self, text):
        if not text:
            self.completer.popup().hide()
            return
        self.completer.setCompletionPrefix(text)
        if self.completer.completionCount() > 0:
            self.completer.complete()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Tab:
            completion = self._current_completion()
            if completion:
                self.setText(completion)
                self.completer.popup().hide()
            return
        super().keyPressEvent(event)

    def _current_completion(self):
        prefix = self.text().strip().lower()
        if not prefix:
            return None
        for command in self.COMMANDS:
            if command.lower().startswith(prefix):
                return command
        return None

    def dragEnterEvent(self, event):
        if self._has_txt_file(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".txt"):
                self.console_window.run_script_file(path)
                event.acceptProposedAction()
                return
        event.ignore()

    def _has_txt_file(self, mime_data):
        return any(url.toLocalFile().lower().endswith(".txt") for url in mime_data.urls())


class ConsoleWindow(QDialog):
    LOG_COLORS = {
        "command": "#5dbeeb",
        "error": "#de2b2b",
        "success": "#7ee787",
        "warning": "#f8e36d",
        "info": "#d6dde8",
        "help": "#c9b6ff",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("Console")
        self.setMinimumSize(700, 320)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QTextEdit.NoWrap)
        self.output.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        layout.addWidget(self.output)
        self.command_input = ConsoleCommandInput(self)
        self.command_input.setPlaceholderText("Enter command")
        self.command_input.returnPressed.connect(self._run_command)
        layout.addWidget(self.command_input)

    def log(self, message: str, level: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        level = level if level in self.LOG_COLORS else self._guess_level(message)
        color = self.LOG_COLORS[level]
        safe_line = html.escape(f"[{timestamp}] {message}").replace(" ", "&nbsp;").replace("\n", "<br>")
        self.output.append(f'<span style="color:{color};">{safe_line}</span>')

    def _guess_level(self, message: str) -> str:
        text = message.lower()
        if message.startswith(">"):
            return "command"
        if "error" in text or "unknown command" in text or "could not" in text or "failed" in text:
            return "error"
        if "skipped" in text or "canceled" in text:
            return "warning"
        if "saved" in text or "loaded" in text or "finished" in text or "applied" in text:
            return "success"
        return "info"

    def _run_command(self):
        command = self.command_input.text().strip()
        if not command:
            return
        self.command_input.clear()
        self.log(f"> {command}", "command")
        if command.lower() in ("clear", "cls"):
            self.output.clear()
            return
        if self.parent() is not None and hasattr(self.parent(), "execute_console_command"):
            self.parent().execute_console_command(command)

    def run_script_file(self, path):
        try:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    script_text = f.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="cp1250") as f:
                    script_text = f.read()
        except Exception as e:
            self.log(f"Script load failed: {path} ({e})", "error")
            return

        self.log(f"Script loaded: {os.path.basename(path)}", "success")
        self.run_script_text(script_text)

    def run_script_text(self, script_text):
        for raw_line in script_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            self.log(f"> {line}", "command")
            self._execute_script_line(line)

    def _execute_script_line(self, line):
        lower = line.lower()
        if lower.startswith("write(") and line.endswith(")"):
            text = line[line.find("(") + 1:-1].strip()
            if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                text = text[1:-1]
            self.log(text, "info")
            return

        script_map = {
            "open.curves": "curves",
            "open.levels": "levels",
            "open.histogram": "histogram",
            "open.correction": "correction",
            "open.calibration": "calibration",
            "open.console": "console",
            "open.menu": "menu",
            "run.magic": "magic",
            "run.starnet": "starnet++",
            "run.starnet++": "starnet++",
            "run.deepsnr": "deepsnr",
            "open.blur": "blur",
            "save": "save",
        }
        command = script_map.get(lower, line)
        if self.parent() is not None and hasattr(self.parent(), "execute_console_command"):
            self.parent().execute_console_command(command)


class PhotoshopMenuPanel(QFrame):
    LAYER_DEFS = [
        ("curves", "Curves"),
        ("stars", "Stars"),
        ("blur", "Gaussian Blur"),
        ("background", "Background"),
        ("plate_solve_overlay", "Plate Solve Overlay"),
        ("grid_overlay", "Grid Overlay"),
        ("object_labels", "Object Labels"),
        ("constellation_overlay", "Constellation Overlay"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.host = parent
        self.layer_thumbs = {}
        self.layer_names = {}
        self.layer_rows = {}
        self.layer_eye_buttons = {}
        self.channel_rows = {}
        self.channel_thumbs = {}
        self.channel_eye_buttons = {}
        self.channel_visibility = {"r": True, "g": True, "b": True}
        self.layers_layout = None
        self.setObjectName("photoshopPanel")
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QFrame()
        body.setObjectName("photoshopBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("photoshopTabs")
        self.tabs.addTab(self._build_layers_tab(), "Warstwy")
        self.tabs.addTab(self._build_channels_tab(), "KanaĹ‚y")
        self.tabs.addTab(self._build_history_tab(), "Historia")
        self.tabs.addTab(self._build_info_tab(), "Informacje")
        body_layout.addWidget(self.tabs)
        outer.addWidget(body)

        self.setStyleSheet("""
            QFrame#photoshopPanel {
                background: #252525;
                border: 1px solid #111111;
                border-radius: 8px;
            }
            QFrame#photoshopBody QLabel {
                color: #e6e6e6;
            }
            QFrame#photoshopBody {
                background: #2b2b2b;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
            }
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                background: #2b2b2b;
                top: -1px;
            }
            QTabBar::tab {
                background: #333333;
                color: #cfcfcf;
                min-height: 30px;
                padding: 6px 10px;
                margin-right: 2px;
                border: 1px solid #3c3c3c;
                border-bottom: 0;
            }
            QTabBar::tab:selected {
                background: #424242;
                color: #ffffff;
            }
            QFrame[layerRow="true"] {
                background: #303030;
                border: 1px solid transparent;
                border-radius: 6px;
            }
            QFrame[layerRow="true"][selected="true"] {
                background: #263341;
                border: 1px solid #007acc;
            }
            QLabel[layerName="true"] {
                color: #f2f2f2;
                font-weight: 600;
                font-size: 13px;
            }
            QLabel[smallInfo="true"] {
                color: #d0d0d0;
                font-size: 12px;
            }
            QPushButton[eyeButton="true"] {
                background: transparent;
                color: #d8d8d8;
                border: 0;
                font-size: 16px;
            }
            QPushButton[eyeButton="true"]:checked {
                color: #f4f4f4;
            }
            QComboBox, QSpinBox {
                background: #1f1f1f;
                color: #eeeeee;
                border: 1px solid #555555;
                min-height: 22px;
                border-radius: 4px;
            }
            QTextEdit {
                background: #1f1f1f;
                color: #eeeeee;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                font-family: Consolas, monospace;
                font-size: 10pt;
            }
        """)

    def _build_layers_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.setSpacing(6)

        controls = QGridLayout()
        controls.setHorizontalSpacing(6)
        controls.addWidget(QLabel("Blend"), 0, 0)
        self.blend_combo = QComboBox()
        self.blend_combo.addItems(["Normal", "Screen", "Multiply", "Overlay", "Soft Light"])
        controls.addWidget(self.blend_combo, 0, 1)
        controls.addWidget(QLabel("Opacity"), 1, 0)
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setValue(100)
        self.opacity_spin.setSuffix("%")
        controls.addWidget(self.opacity_spin, 1, 1)
        layout.addLayout(controls)

        self.layers_layout = layout
        for key, title in self.LAYER_DEFS:
            self._ensure_layer_row(key, title)

        layout.addStretch(1)
        return tab

    def _ensure_layer_row(self, key, title):
        row = self.layer_rows.get(key)
        if row is None:
            row = self._make_layer_row(key, title)
            if self.layers_layout is not None:
                if self.layers_layout.count() <= 1:
                    self.layers_layout.addWidget(row)
                else:
                    self.layers_layout.insertWidget(self.layers_layout.count() - 1, row)
        else:
            name = self.layer_names.get(key)
            if name is not None:
                name.setText(title)
        return row

    def _build_channels_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(6)
        self.channel_rows = {}
        self.channel_thumbs = {}
        self.channel_eye_buttons = {}
        self.channel_labels = {}

        for key, title, color in [
            ("r", "Red", "#ff5555"),
            ("g", "Green", "#55ff55"),
            ("b", "Blue", "#5599ff"),
        ]:
            row = self._make_channel_row(key, title, color)
            layout.addWidget(row)

        layout.addStretch(1)
        return tab

    def _build_history_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 8, 4, 4)
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        layout.addWidget(self.history_text)
        return tab

    def _build_info_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 10, 6, 6)
        self.info_label = QLabel("No image loaded")
        self.info_label.setProperty("smallInfo", True)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        layout.addStretch(1)
        return tab

    def _eye_svg(self, visible: bool, color: str) -> str: 
        if visible:
            return f"""
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
              <g fill="none" stroke="{color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
                <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z"/>
                <circle cx="12" cy="12" r="3"/>
              </g>
            </svg>
            """
        return f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
          <g fill="none" stroke="{color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 3l18 18"/>
            <path d="M2.4 12s3.5-6 9.6-6c1.1 0 2.2.2 3.2.5"/>
            <path d="M21.6 12s-3.5 6-9.6 6c-1.1 0-2.2-.2-3.2-.5"/>
          </g>
        </svg>
        """

    def _set_eye_icon(self, button: QPushButton, visible: bool):
        color = "#f2f2f2" if getattr(self.host, "dark_mode", True) else "#2d2d2d"
        button.setIcon(svg_icon_from_text(self._eye_svg(visible, color), 18))
        button.setIconSize(QSize(18, 18))
        button.setText("")

    def _channel_preview(self, source_img: np.ndarray, channel_key: str, visible: bool) -> QPixmap:
        if source_img is None:
            canvas = QPixmap(58, 38)
            canvas.fill(QColor(22, 22, 22))
            return canvas

        if source_img.ndim == 2:
            gray = source_img.copy()
            preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            b, g, r = cv2.split(source_img)
            zero = np.zeros_like(r)
            if channel_key == "r":
                preview = cv2.merge([zero, zero, r])
            elif channel_key == "g":
                preview = cv2.merge([zero, g, zero])
            else:
                preview = cv2.merge([b, zero, zero])

        if not visible:
            preview = cv2.addWeighted(preview, 0.2, np.zeros_like(preview), 0.8, 0)

        pix = np_to_qpixmap(preview)
        thumb = pix.scaled(58, 38, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        canvas = QPixmap(58, 38)
        canvas.fill(QColor(22, 22, 22))
        painter = QPainter(canvas)
        x = (58 - thumb.width()) // 2
        y = (38 - thumb.height()) // 2
        painter.drawPixmap(x, y, thumb)
        painter.end()
        return canvas

    def _make_channel_row(self, key: str, title: str, color: str) -> QFrame:
        row = QFrame()
        row.setProperty("layerRow", True)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(6, 5, 6, 5)
        layout.setSpacing(8)

        eye = QPushButton()
        eye.setProperty("eyeButton", True)
        eye.setCheckable(True)
        eye.setChecked(True)
        eye.setFixedSize(28, 28)
        eye.clicked.connect(lambda checked, channel_key=key: self._toggle_channel(channel_key, checked))
        self._set_eye_icon(eye, True)

        thumb = QLabel()
        thumb.setFixedSize(58, 38)
        thumb.setStyleSheet("background: #161616; border: 1px solid #555555; border-radius: 4px;")
        thumb.setPixmap(self._channel_preview(self.host.original_img if self.host is not None else None, key, True))

        name = QLabel(f"{title} channel")
        name.setProperty("layerName", True)

        layout.addWidget(eye)
        layout.addWidget(thumb)
        layout.addWidget(name)
        layout.addStretch(1)

        self.channel_rows[key] = row
        self.channel_thumbs[key] = thumb
        self.channel_eye_buttons[key] = eye
        self.channel_labels[key] = name
        return row

    def _toggle_channel(self, key, checked):
        self.channel_visibility[key] = bool(checked)
        if self.host is not None and hasattr(self.host, "set_channel_visibility"):
            self.host.set_channel_visibility(key, checked)
        elif self.host is not None:
            self.update_from_app(self.host)

    def _make_layer_row(self, key, title):
        row = QFrame()
        row.setProperty("layerRow", True)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(6, 5, 6, 5)
        layout.setSpacing(8)
        eye = QPushButton()
        eye.setProperty("eyeButton", True)
        eye.setCheckable(True)
        eye.setChecked(True)
        eye.setFixedSize(28, 28)
        eye.clicked.connect(lambda checked, layer_key=key: self._toggle_layer(layer_key, checked))
        self._set_eye_icon(eye, True)
        thumb = ClickableLabel()
        thumb.setFixedSize(58, 38)
        thumb.setStyleSheet("background: #161616; border: 1px solid #555555; border-radius: 4px;")
        name = QLabel(title)
        name.setProperty("layerName", True)
        thumb.clicked.connect(lambda layer_key=key: self._select_layer(layer_key))
        name.mousePressEvent = lambda event, layer_key=key: self._select_layer(layer_key)
        row.mousePressEvent = lambda event, layer_key=key: self._handle_layer_row_click(event, layer_key)
        for widget in (row, thumb, name):
            widget.setContextMenuPolicy(Qt.CustomContextMenu)
            widget.customContextMenuRequested.connect(
                lambda pos, layer_key=key, source_widget=widget: self._show_layer_context_menu(layer_key, source_widget, pos)
            )
        layout.addWidget(eye)
        layout.addWidget(thumb)
        layout.addWidget(name)
        layout.addStretch(1)
        self.layer_thumbs[key] = thumb
        self.layer_names[key] = name
        self.layer_rows[key] = row
        self.layer_eye_buttons[key] = eye
        return row

    def _handle_layer_row_click(self, event, key):
        if event.button() == Qt.LeftButton:
            self._select_layer(key)

    def _show_layer_context_menu(self, key, source_widget, pos):
        self._select_layer(key)
        menu = QMenu(self)
        delete_action = menu.addAction("UsuĹ„")
        can_delete = True
        if self.host is not None and hasattr(self.host, "can_delete_layer"):
            can_delete = self.host.can_delete_layer(key)
        delete_action.setEnabled(can_delete)
        selected_action = menu.exec_(source_widget.mapToGlobal(pos))
        if selected_action == delete_action and self.host is not None and hasattr(self.host, "delete_selected_layer"):
            self.host.delete_selected_layer()
            self.update_from_app(self.host)

    def _remove_layer_row(self, key):
        row = self.layer_rows.pop(key, None)
        self.layer_thumbs.pop(key, None)
        self.layer_names.pop(key, None)
        self.layer_eye_buttons.pop(key, None)
        if row is not None:
            row.setParent(None)
            row.deleteLater()

    def _toggle_layer(self, key, checked):
        button = self.layer_eye_buttons.get(key)
        if button is not None:
            self._set_eye_icon(button, checked)
        if self.host is not None and hasattr(self.host, "set_layer_visibility"):
            self.host.set_layer_visibility(key, checked)

    def _select_layer(self, key):
        if self.host is not None and hasattr(self.host, "select_layer"):
            self.host.select_layer(key)
        self.update_from_app(self.host)

    def update_from_app(self, app):
        active_layers = app.get_active_layers() if hasattr(app, "get_active_layers") else ["background"]
        for key, _title in self.LAYER_DEFS:
            row = self.layer_rows.get(key)
            if row is not None:
                row.setVisible(key in active_layers)
            eye = self.layer_eye_buttons.get(key)
            if eye is not None:
                is_visible = app.layer_visibility.get(key, True) if hasattr(app, "layer_visibility") else True
                eye.blockSignals(True)
                eye.setChecked(is_visible)
                eye.setText("đź‘" if is_visible else "â—‹")
                eye.blockSignals(False)

        self._set_thumbnail("background", app.original_img)
        self._set_thumbnail("blur", app.layer_images.get("blur") if hasattr(app, "layer_images") else app.magic_img)
        self._set_thumbnail("stars", app.layer_images.get("stars") if hasattr(app, "layer_images") else app.magic_img)
        self._set_thumbnail("curves", app.processed_img)

        if app.original_img is None:
            info = "No image loaded"
        else:
            h, w = app.original_img.shape[:2]
            channels = 1 if app.original_img.ndim == 2 else app.original_img.shape[2]
            save_path = app.current_save_path if app.current_save_path else "Not saved"
            starnet = app.starnet_path if app.starnet_path else "Not selected"
            info = (
                f"Size: {w} x {h}\n"
                f"Channels: {channels}\n"
                f"Undo: {len(app.undo_stack)}\n"
                f"Redo: {len(app.redo_stack)}\n"
                f"Save path: {save_path}\n"
                f"StarNet++: {starnet}"
            )
        self.info_label.setText(info)

        history = getattr(app, "history_items", [])
        self.history_text.setPlainText("\n".join(history[-18:]) if history else "No history yet")

    def _set_thumbnail(self, key, img):
        label = self.layer_thumbs.get(key)
        if label is None:
            return
        if img is None:
            label.clear()
            label.setStyleSheet("background: #000000; border: 2px solid #000000;")
            return

        if isinstance(img, QPixmap):
            pix = img
        else:
            pix = np_to_qpixmap(img)
        thumb = pix.scaled(54, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        canvas = QPixmap(58, 38)
        canvas.fill(QColor(22, 22, 22))
        painter = QPainter(canvas)
        x = (58 - thumb.width()) // 2
        y = (38 - thumb.height()) // 2
        painter.drawPixmap(x, y, thumb)
        painter.end()
        label.setPixmap(canvas)

    def update_from_app(self, app):
        active_layers = app.get_active_layers() if hasattr(app, "get_active_layers") else ["background"]
        layer_titles = getattr(app, "layer_titles", {})
        default_titles = dict(self.LAYER_DEFS)
        layer_keys = list(getattr(app, "layer_order", []))
        for key in default_titles:
            if key not in layer_keys:
                layer_keys.append(key)

        for stale_key in list(self.layer_rows):
            if stale_key not in layer_keys:
                self._remove_layer_row(stale_key)

        for key in layer_keys:
            title = layer_titles.get(key, default_titles.get(key, key.replace("_", " ").title()))
            self._ensure_layer_row(key, title)

            row = self.layer_rows.get(key)
            if row is not None:
                row.setVisible(key in active_layers)

            eye = self.layer_eye_buttons.get(key)
            if eye is not None:
                is_visible = app.layer_visibility.get(key, True) if hasattr(app, "layer_visibility") else True
                eye.blockSignals(True)
                eye.setChecked(is_visible)
                eye.setText("Ä‘Ĺşâ€Â" if is_visible else "Ă˘â€”â€ą")
                eye.blockSignals(False)

        self._set_thumbnail("background", app.original_img)
        self._set_thumbnail("blur", app.layer_images.get("blur") if hasattr(app, "layer_images") else app.magic_img)
        self._set_thumbnail("stars", app.layer_images.get("stars") if hasattr(app, "layer_images") else app.magic_img)
        self._set_thumbnail("curves", app.processed_img)

        if app.original_img is None:
            info = "No image loaded"
        else:
            h, w = app.original_img.shape[:2]
            channels = 1 if app.original_img.ndim == 2 else app.original_img.shape[2]
            save_path = app.current_save_path if app.current_save_path else "Not saved"
            starnet = app.starnet_path if app.starnet_path else "Not selected"
            info = (
                f"Size: {w} x {h}\n"
                f"Channels: {channels}\n"
                f"Undo: {len(app.undo_stack)}\n"
                f"Redo: {len(app.redo_stack)}\n"
                f"Save path: {save_path}\n"
                f"StarNet++: {starnet}"
            )
        self.info_label.setText(info)

        history = getattr(app, "history_items", [])
        self.history_text.setPlainText("\n".join(history[-18:]) if history else "No history yet")


    def update_from_app(self, app):
        active_layers = app.get_active_layers() if hasattr(app, "get_active_layers") else ["background"]
        layer_titles = getattr(app, "layer_titles", {})
        default_titles = dict(self.LAYER_DEFS)
        layer_keys = list(getattr(app, "layer_order", []))
        for key in default_titles:
            if key not in layer_keys:
                layer_keys.append(key)

        for stale_key in list(self.layer_rows):
            if stale_key not in layer_keys:
                self._remove_layer_row(stale_key)

        for key in layer_keys:
            title = layer_titles.get(key, default_titles.get(key, key.replace("_", " ").title()))
            self._ensure_layer_row(key, title)

            row = self.layer_rows.get(key)
            if row is not None:
                row.setVisible(key in active_layers)

            eye = self.layer_eye_buttons.get(key)
            if eye is not None:
                is_visible = app.layer_visibility.get(key, True) if hasattr(app, "layer_visibility") else True
                eye.blockSignals(True)
                eye.setChecked(is_visible)
                self._set_eye_icon(eye, is_visible)
                eye.blockSignals(False)

        self._set_thumbnail("background", app.original_img)
        self._set_thumbnail("blur", app.layer_images.get("blur") if hasattr(app, "layer_images") else app.magic_img)
        self._set_thumbnail("stars", app.layer_images.get("stars") if hasattr(app, "layer_images") else app.magic_img)
        self._set_thumbnail("curves", app.processed_img)

        source_img = app.original_img if app.original_img is not None else app.magic_img
        for key, title in (("r", "Red"), ("g", "Green"), ("b", "Blue")):
            row = self.channel_rows.get(key)
            if row is not None:
                row.setVisible(True)

            visible = app.channel_visibility.get(key, True) if hasattr(app, "channel_visibility") else True
            eye = self.channel_eye_buttons.get(key)
            if eye is not None:
                eye.blockSignals(True)
                eye.setChecked(visible)
                self._set_eye_icon(eye, visible)
                eye.blockSignals(False)

            thumb = self.channel_thumbs.get(key)
            if thumb is not None:
                thumb.setPixmap(self._channel_preview(source_img, key, visible))

            label = self.channel_labels.get(key)
            if label is not None:
                label.setText(f"{title} channel")
                label.setStyleSheet("" if visible else "color: #8a8a8a;")

        if app.original_img is None:
            info = "No image loaded"
        else:
            h, w = app.original_img.shape[:2]
            channels = 1 if app.original_img.ndim == 2 else app.original_img.shape[2]
            save_path = app.current_save_path if app.current_save_path else "Not saved"
            starnet = app.starnet_path if app.starnet_path else "Not selected"
            info = (
                f"Size: {w} x {h}\n"
                f"Channels: {channels}\n"
                f"Undo: {len(app.undo_stack)}\n"
                f"Redo: {len(app.redo_stack)}\n"
                f"Save path: {save_path}\n"
                f"StarNet++: {starnet}"
            )
        self.info_label.setText(info)

        history = getattr(app, "history_items", [])
        self.history_text.setPlainText("\n".join(history[-18:]) if history else "No history yet")


    def update_from_app(self, app):
        active_layers = app.get_active_layers() if hasattr(app, "get_active_layers") else ["background"]
        layer_titles = getattr(app, "layer_titles", {})
        default_titles = dict(self.LAYER_DEFS)
        layer_keys = list(getattr(app, "layer_order", []))
        selected_layer = getattr(app, "selected_layer_key", None)
        for key in default_titles:
            if key not in layer_keys:
                layer_keys.append(key)

        for stale_key in list(self.layer_rows):
            if stale_key not in layer_keys:
                self._remove_layer_row(stale_key)

        for key in layer_keys:
            title = layer_titles.get(key, default_titles.get(key, key.replace("_", " ").title()))
            self._ensure_layer_row(key, title)
            row = self.layer_rows.get(key)
            if row is not None:
                row.setVisible(key in active_layers)
                row.setProperty("selected", key == selected_layer)
                row.setStyleSheet(
                    "" if key != selected_layer else "background: #263341; border: 1px solid #007acc; border-radius: 6px;"
                )
            eye = self.layer_eye_buttons.get(key)
            if eye is not None:
                is_visible = app.layer_visibility.get(key, True) if hasattr(app, "layer_visibility") else True
                eye.blockSignals(True)
                eye.setChecked(is_visible)
                self._set_eye_icon(eye, is_visible)
                eye.blockSignals(False)

        self._set_thumbnail("background", app.original_img)
        self._set_thumbnail("blur", app.layer_images.get("blur") if hasattr(app, "layer_images") else app.magic_img)
        self._set_thumbnail("stars", app.layer_images.get("stars") if hasattr(app, "layer_images") else app.magic_img)
        self._set_thumbnail("curves", app.processed_img)

        source_img = app.original_img if app.original_img is not None else app.magic_img
        for key, title in (("r", "Red"), ("g", "Green"), ("b", "Blue")):
            row = self.channel_rows.get(key)
            if row is not None:
                row.setVisible(True)
            visible = app.channel_visibility.get(key, True) if hasattr(app, "channel_visibility") else True
            eye = self.channel_eye_buttons.get(key)
            if eye is not None:
                eye.blockSignals(True)
                eye.setChecked(visible)
                self._set_eye_icon(eye, visible)
                eye.blockSignals(False)
            thumb = self.channel_thumbs.get(key)
            if thumb is not None:
                thumb.setPixmap(self._channel_preview(source_img, key, visible))
            label = self.channel_labels.get(key)
            if label is not None:
                label.setText(f"{title} channel")
                label.setStyleSheet("" if visible else "color: #8a8a8a;")

        if app.original_img is None:
            info = "No image loaded"
        else:
            h, w = app.original_img.shape[:2]
            channels = 1 if app.original_img.ndim == 2 else app.original_img.shape[2]
            save_path = app.current_save_path if app.current_save_path else "Not saved"
            starnet = app.starnet_path if app.starnet_path else "Not selected"
            info = (
                f"Size: {w} x {h}\n"
                f"Channels: {channels}\n"
                f"Undo: {len(app.undo_stack)}\n"
                f"Redo: {len(app.redo_stack)}\n"
                f"Save path: {save_path}\n"
                f"StarNet++: {starnet}"
            )
        self.info_label.setText(info)

        history = getattr(app, "history_items", [])
        self.history_text.setPlainText("\n".join(history[-18:]) if history else "No history yet")


class PhotoshopMenuDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("Menu")
        self.setMinimumSize(390, 430)
        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)
        self.panel = PhotoshopMenuPanel(parent)
        layout.addWidget(self.panel)
        self.setStyleSheet("""
            QDialog {
                background: #1e1e1e;
            }
        """)

    def update_from_app(self, app):
        self.panel.update_from_app(app)


# ---------- Viewer ----------

class SingleViewer(QGraphicsView):
    zoom_changed_signal = pyqtSignal(float)
    imageClicked = pyqtSignal(int, int)
    imageDropped = pyqtSignal(object)
    roiDragStarted = pyqtSignal(int, int)
    roiDragUpdated = pyqtSignal(int, int)
    roiDragFinished = pyqtSignal(int, int)
    roiHoverMoved = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAcceptDrops(True)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.scale_factor = 1.0
        self._pixmap_item = None
        self._pan_margin = 800
        self._centered_once = False
        self.pick_mode = False
        self.roi_pick_mode = False
        self._roi_dragging = False

    def set_pixmap(self, pix: QPixmap):
        if pix is None:
            self._scene.clear()
            self._pixmap_item = None
            return

        current_center = self.mapToScene(self.viewport().rect().center())
        first_update = self._pixmap_item is None

        if first_update:
            self._pixmap_item = self._scene.addPixmap(pix)
        else:
            self._pixmap_item.setPixmap(pix)

        margin = self._pan_margin
        self._pixmap_item.setPos(margin, margin)
        self._scene.setSceneRect(0, 0, pix.width() + margin * 2, pix.height() + margin * 2)

        if not self._centered_once:
            self.centerOn(margin + pix.width() / 2, margin + pix.height() / 2)
            self._centered_once = True
            return

        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        old_h = hbar.value()
        old_v = vbar.value()
        old_h_max = hbar.maximum()
        old_v_max = vbar.maximum()

        self.centerOn(current_center)

        if old_h_max > 0 and hbar.maximum() > 0:
            hbar.setValue(int(old_h / old_h_max * hbar.maximum()))
        else:
            hbar.setValue(old_h)

        if old_v_max > 0 and vbar.maximum() > 0:
            vbar.setValue(int(old_v / old_v_max * vbar.maximum()))
        else:
            vbar.setValue(old_v)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.roi_pick_mode:
            point = self._event_to_image_coords(event, clamp=False)
            if point is not None:
                self._roi_dragging = True
                self.roiDragStarted.emit(point[0], point[1])
                event.accept()
                return
        if event.button() == Qt.LeftButton:
            # Reagujemy na klikniÄ™cie tylko wtedy, gdy tryb wyboru jest wĹ‚Ä…czony (tak jak w PS/Pix)
            if hasattr(self, 'pick_mode') and self.pick_mode:
                pos = event.pos()
                self.imageClicked.emit(pos.x(), pos.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.roi_pick_mode and not self._roi_dragging:
            self.roiHoverMoved.emit(self._event_to_image_coords(event, clamp=False))
        if self.roi_pick_mode and self._roi_dragging:
            point = self._event_to_image_coords(event, clamp=True)
            if point is not None:
                self.roiDragUpdated.emit(point[0], point[1])
                event.accept()
                return
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self.roi_pick_mode and not self._roi_dragging:
            self.roiHoverMoved.emit(None)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.roi_pick_mode and self._roi_dragging and event.button() == Qt.LeftButton:
            point = self._event_to_image_coords(event, clamp=True)
            self._roi_dragging = False
            if point is not None:
                self.roiDragFinished.emit(point[0], point[1])
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def set_pick_mode(self, enabled):
        """WĹ‚Ä…cza lub wyĹ‚Ä…cza tryb wyboru punktĂłw (Photoshop Picker / PixInsight DBE)"""
        self.pick_mode = enabled
        self._sync_interaction_mode()

    def set_roi_pick_mode(self, enabled):
        self.roi_pick_mode = bool(enabled)
        if not self.roi_pick_mode:
            self._roi_dragging = False
        self._sync_interaction_mode()

    def _sync_interaction_mode(self):
        if self.roi_pick_mode:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
            return
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        if self.pick_mode:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _event_to_image_coords(self, event, clamp=False):
        if self._pixmap_item is None:
            return None

        pix = self._pixmap_item.pixmap()
        if pix.isNull():
            return None

        scene_pos = self.mapToScene(event.pos())
        local_x = float(scene_pos.x() - self._pixmap_item.pos().x())
        local_y = float(scene_pos.y() - self._pixmap_item.pos().y())
        width = pix.width()
        height = pix.height()

        if clamp:
            x = int(round(max(0.0, min(float(width - 1), local_x))))
            y = int(round(max(0.0, min(float(height - 1), local_y))))
            return (x, y)

        if local_x < 0.0 or local_y < 0.0 or local_x > float(width - 1) or local_y > float(height - 1):
            return None
        return (int(round(local_x)), int(round(local_y)))

    def on_viewer_image_clicked(self, x, y):
     """ObsĹ‚uga klikniÄ™cia na obraz - alternatywa dla PixInsight i Photoshopa"""
     self.log(f"KlikniÄ™to na obraz w punkcie: X={x}, Y={y}")

     # --- FUNKCJA STYL PIXINSIGHT (Dynamic Background Extraction - PrĂłbki tĹ‚a) ---
     if hasattr(self, 'dbe_mode_active') and self.dbe_mode_active:
         if not hasattr(self, 'bg_samples'):
             self.bg_samples = []
         self.bg_samples.append((x, y))
         self.log(f"PixInsight DBE: Dodano prĂłbkÄ™ tĹ‚a ({x}, {y}). ĹÄ…cznie prĂłbki: {len(self.bg_samples)}")
         return

     # --- FUNKCJA STYL PHOTOSHOP (Color Picker - PrĂłbnik koloru dla krzywych/warstw) ---
     if hasattr(self, 'picker_mode_active') and self.picker_mode_active:
         if self.original_img is not None:
             try:
                 # OpenCV przechowuje obrazy w formacie BGR
                 color_bgr = self.original_img[y, x]
                 b, g, r = color_bgr[0], color_bgr[1], color_bgr[2]
                 self.log(f"Photoshop Picker: R={r}, G={g}, B={b}")
                 # Tutaj program w przyszĹ‚oĹ›ci moĹĽe postawiÄ‡ punkt na wykresie Curves (Krzywe)
             except IndexError:
                 pass
             return
             
    def pan_by(self, dx: int, dy: int):
        if self._pixmap_item is None:
            return

        scale_x = self.transform().m11() or 1.0
        scale_y = self.transform().m22() or 1.0
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        hbar.setValue(hbar.value() - int(round(dx / scale_x)))
        vbar.setValue(vbar.value() - int(round(dy / scale_y)))

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale_factor *= factor
        self.scale(factor, factor)
        self.zoom_changed_signal.emit(self.scale_factor)

    def dragEnterEvent(self, event):
        if image_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if image_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = image_paths_from_mime_data(event.mimeData())
        if paths:
            self.imageDropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class BlendViewer(QWidget):
    """
    Jeden viewer + slider Before/After.
    BEFORE = obraz po magic_pipeline
    AFTER  = obraz po Camera RAW + HSL
    """
    imageDropped = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)

        self.view = SingleViewer()
        self.view.imageDropped.connect(self.imageDropped)
        layout.addWidget(self.view)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.setMinimumWidth(300)
        self.slider.valueChanged.connect(self.update_blend)
        self.slider.setVisible(False)
        layout.addWidget(self.slider)

        self.pix_before = None
        self.pix_after = None
        self.compare_enabled = False
        self.overlay_pix = None

    def set_before(self, pix: QPixmap):
        self.pix_before = pix
        self.pix_after = None
        self.update_blend()

    def set_after(self, pix: QPixmap):
        self.pix_after = pix
        self.update_blend()

    def set_overlay_pixmap(self, overlay: QPixmap):
        self.overlay_pix = overlay
        self.update_blend()

    def clear_overlay(self):
        self.overlay_pix = None
        self.update_blend()

    def set_compare_enabled(self, enabled: bool):
        self.compare_enabled = enabled
        self.slider.setVisible(enabled)
        self.update_blend()

    def update_blend(self):
        if self.pix_before is None and self.pix_after is None:
            return

        def draw_overlay(base_pix):
            if self.overlay_pix is None:
                return base_pix
            if self.overlay_pix.size() != base_pix.size():
                return base_pix
            result = QPixmap(base_pix)
            painter = QPainter(result)
            painter.drawPixmap(0, 0, self.overlay_pix)
            painter.end()
            return result

        if self.compare_enabled and self.pix_before is not None and self.pix_after is not None:
            w = self.pix_before.width()
            h = self.pix_before.height()

            p = self.slider.value() / 100.0
            cut_x = int(w * p)

            blended = QPixmap(w, h)
            blended.fill(Qt.transparent)

            painter = QPainter(blended)
            painter.drawPixmap(0, 0, self.pix_before)
            painter.drawPixmap(0, 0, self.pix_after.copy(0, 0, cut_x, h))
            painter.end()

            self.view.set_pixmap(draw_overlay(blended))
            return

        if self.pix_after is not None:
            self.view.set_pixmap(draw_overlay(self.pix_after))
            return

        if self.pix_before is not None:
            self.view.set_pixmap(draw_overlay(self.pix_before))
            return

    def dragEnterEvent(self, event):
        if image_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if image_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = image_paths_from_mime_data(event.mimeData())
        if paths:
            self.imageDropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)



# ---------- Panel Camera RAW ----------

class CameraRawPanel(QFrame):
    def __init__(self, on_change_callback):
        super().__init__()
        self.on_change_callback = on_change_callback

        self.temperature = 0.0
        self.tint = 0.0
        self.exposure = 0.0
        self.contrast = 0.0
        self.saturation = 0.0
        self.vibrance = 0.0
        self.texture = 0.0
        self.dehaze = 0.0
        self.clarity = 0.0
        self.noise_reduction = 0.0

        self._build_ui()

    def _make_slider(self, minv, maxv, init, gradient_css):
        s = QSlider(Qt.Horizontal)
        s.setRange(minv, maxv)
        s.setValue(init)
        s.setMinimumWidth(300)
        s.setSingleStep(1)

        s.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid #444;
                height: 10px;
                border-radius: 5px;
                background: {gradient_css};
            }}
            QSlider::sub-page:horizontal {{
                background: transparent;
                border: none;
            }}
            QSlider::add-page:horizontal {{
                background: transparent;
                border: none;
            }}
            QSlider::handle:horizontal {{
                background: #d0d0d0;
                border: 1px solid #333;
                width: 16px;
                margin: -3px 0;
                border-radius: 8px;
            }}
        """)

        return s

    def _make_spinbox(self, slider, scale=1.0, decimals=0):
        spin = QSpinBox()
        spin.setRange(slider.minimum(), slider.maximum())
        spin.setValue(slider.value())
        spin.setFixedWidth(80)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        return spin

    def _build_ui(self):
        layout = QVBoxLayout(self)
        grid = QGridLayout()

        # Temperature (blue â†’ orange)
        self.lbl_temp = QLabel("Temperatura: 0")
        self.sld_temp = self._make_slider(
            -100, 100, 0,
            "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0099ff, stop:1 #ff8800)"
        )
        self.sld_temp.valueChanged.connect(self._on_temp_changed)
        self.spin_temp = self._make_spinbox(self.sld_temp)
        grid.addWidget(self.lbl_temp, 0, 0)
        grid.addWidget(self.sld_temp, 0, 1)
        grid.addWidget(self.spin_temp, 0, 2)

        # Tint (magenta â†’ green)
        self.lbl_tint = QLabel("Tint: 0")
        self.sld_tint = self._make_slider(
            -100, 100, 0, "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff00ff, stop:1 #00ff66)"
        )
        self.sld_tint.valueChanged.connect(self._on_tint_changed)
        self.spin_tint = self._make_spinbox(self.sld_tint)
        grid.addWidget(self.lbl_tint, 1, 0)
        grid.addWidget(self.sld_tint, 1, 1)
        grid.addWidget(self.spin_tint, 1, 2)

        # Exposure (black â†’ white)
        self.lbl_exp = QLabel("Ekspozycja: 0")
        self.sld_exp = self._make_slider(
            -100, 100, 0,
            "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #ffffff)"
        )
        self.sld_exp.valueChanged.connect(self._on_exp_changed)
        self.spin_exp = self._make_spinbox(self.sld_exp)
        grid.addWidget(self.lbl_exp, 2, 0)
        grid.addWidget(self.sld_exp, 2, 1)
        grid.addWidget(self.spin_exp, 2, 2)

        # Contrast (dark gray â†’ light gray)
        self.lbl_contrast = QLabel("Kontrast: 0")
        self.sld_contrast = self._make_slider(
            -100, 100, 0,
            "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #222222, stop:1 #dddddd)"
        )
        self.sld_contrast.valueChanged.connect(self._on_contrast_changed)
        self.spin_contrast = self._make_spinbox(self.sld_contrast)
        grid.addWidget(self.lbl_contrast, 3, 0)
        grid.addWidget(self.sld_contrast, 3, 1)
        grid.addWidget(self.spin_contrast, 3, 2)

        # Saturation (gray â†’ red)
        self.lbl_sat = QLabel("Saturacja: 0")
        self.sld_sat = self._make_slider(
            -100, 100, 0,
            "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #777777, stop:1 #ff4444)"
        )
        self.sld_sat.valueChanged.connect(self._on_sat_changed)
        self.spin_sat = self._make_spinbox(self.sld_sat)
        grid.addWidget(self.lbl_sat, 4, 0)
        grid.addWidget(self.sld_sat, 4, 1)
        grid.addWidget(self.spin_sat, 4, 2)

        # Vibrance (soft blue â†’ strong blue)
        self.lbl_vib = QLabel("Vibrance: 0")
        self.sld_vib = self._make_slider(
            -100, 100, 0,
            "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9999ff, stop:1 #0000ff)"
        )
        self.sld_vib.valueChanged.connect(self._on_vib_changed)
        self.spin_vib = self._make_spinbox(self.sld_vib)
        grid.addWidget(self.lbl_vib, 5, 0)
        grid.addWidget(self.sld_vib, 5, 1)
        grid.addWidget(self.spin_vib, 5, 2)

        # Texture (micro-detail enhancement)
        self.lbl_texture = QLabel("Texture: 0")
        self.sld_texture = self._make_slider(
            -100, 100, 0,
            "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #555555, stop:1 #ffffff)"
        )
        self.sld_texture.valueChanged.connect(self._on_texture_changed)
        self.spin_texture = self._make_spinbox(self.sld_texture)
        grid.addWidget(self.lbl_texture, 6, 0)
        grid.addWidget(self.sld_texture, 6, 1)
        grid.addWidget(self.spin_texture, 6, 2)

        # Dehaze (fog/haze reduction)
        self.lbl_dehaze = QLabel("Dehaze: 0")
        self.sld_dehaze = self._make_slider(
            -100, 100, 0,
            "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #bbbbbb, stop:1 #777777)"
        )
        self.sld_dehaze.valueChanged.connect(self._on_dehaze_changed)
        self.spin_dehaze = self._make_spinbox(self.sld_dehaze)
        grid.addWidget(self.lbl_dehaze, 7, 0)
        grid.addWidget(self.sld_dehaze, 7, 1)
        grid.addWidget(self.spin_dehaze, 7, 2)

        # Clarity (midtone contrast)
        self.lbl_clarity = QLabel("Clarity: 0")
        self.sld_clarity = self._make_slider(
            -100, 100, 0,
            "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #444444, stop:1 #dddddd)"
        )
        self.sld_clarity.valueChanged.connect(self._on_clarity_changed)
        self.spin_clarity = self._make_spinbox(self.sld_clarity)
        grid.addWidget(self.lbl_clarity, 8, 0)
        grid.addWidget(self.sld_clarity, 8, 1)
        grid.addWidget(self.spin_clarity, 8, 2)

        # Noise reduction
        self.lbl_noise = QLabel("Noise reduction: 0")
        self.sld_noise = self._make_slider(
            0, 100, 0,
            "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #222222, stop:1 #666666)"
        )
        self.sld_noise.valueChanged.connect(self._on_noise_changed)
        self.spin_noise = self._make_spinbox(self.sld_noise)
        grid.addWidget(self.lbl_noise, 9, 0)
        grid.addWidget(self.sld_noise, 9, 1)
        grid.addWidget(self.spin_noise, 9, 2)

        layout.addLayout(grid)
        layout.addStretch(1)

    def _on_temp_changed(self, v):
        self.temperature = float(v)
        self.lbl_temp.setText(f"Temperatura: {int(self.temperature)}")
        self.on_change_callback()

    def _on_tint_changed(self, v):
        self.tint = float(v)
        self.lbl_tint.setText(f"Tint: {int(self.tint)}")
        self.on_change_callback()

    def _on_exp_changed(self, v):
        self.exposure = v / 25.0
        self.lbl_exp.setText(f"Ekspozycja: {int(v)}")
        self.on_change_callback()

    def _on_contrast_changed(self, v):
        self.contrast = v / 100.0
        self.lbl_contrast.setText(f"Kontrast: {int(v)}")
        self.on_change_callback()

    def _on_sat_changed(self, v):
        self.saturation = v / 100.0
        self.lbl_sat.setText(f"Saturacja: {int(v)}")
        self.on_change_callback()

    def _on_vib_changed(self, v):
        self.vibrance = v / 100.0
        self.lbl_vib.setText(f"Vibrance: {int(v)}")
        self.on_change_callback()

    def _on_texture_changed(self, v):
        self.texture = v / 100.0
        self.lbl_texture.setText(f"Texture: {int(v)}")
        self.on_change_callback()

    def _on_dehaze_changed(self, v):
        self.dehaze = v / 100.0
        self.lbl_dehaze.setText(f"Dehaze: {int(v)}")
        self.on_change_callback()

    def _on_clarity_changed(self, v):
        self.clarity = v / 100.0
        self.lbl_clarity.setText(f"Clarity: {int(v)}")
        self.on_change_callback()

    def _on_noise_changed(self, v):
        self.noise_reduction = v / 100.0
        self.lbl_noise.setText(f"Noise reduction: {int(v)}")
        self.on_change_callback()

    def get_params(self) -> dict:
        return {
            "temperature": self.temperature,
            "tint": self.tint,
            "exposure": self.exposure,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "vibrance": self.vibrance,
            "texture": self.texture,
            "dehaze": self.dehaze,
            "clarity": self.clarity,
            "noise_reduction": self.noise_reduction,
        }


# ---------- Panel HSL ----------

class HSLPanel(QFrame):
    COLORS = ["Reds", "Oranges", "Yellows", "Greens", "Aquas", "Blues", "Purples", "Magentas"]
    COLOR_HEX = {
        "Reds": "#d32f2f",
        "Oranges": "#fb8c00",
        "Yellows": "#fdd835",
        "Greens": "#388e3c",
        "Aquas": "#00acc1",
        "Blues": "#1976d2",
        "Purples": "#8e24aa",
        "Magentas": "#c2185b",
    }
    MODES = ["Hue", "Saturation", "Luminance"]

    def __init__(self, on_change_callback):
        super().__init__()
        self.on_change_callback = on_change_callback

        self.values = {
            mode: {c: 0 for c in self.COLORS}
            for mode in self.MODES
        }
        self.sliders = {mode: {} for mode in self.MODES}
        self.spinboxes = {mode: {} for mode in self.MODES}

        self._build_ui()

    def _make_slider(self, color: str = None):
        s = QSlider(Qt.Horizontal)
        s.setRange(-100, 100)
        s.setValue(0)
        s.setMinimumWidth(300)
        s.setSingleStep(1)
        if color is not None:
            color_value = self.COLOR_HEX.get(color, "#999999")
            s.setStyleSheet(f"""
                QSlider::groove:horizontal {{
                    border: 1px solid #444;
                    height: 10px;
                    border-radius: 5px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {color_value}, stop:1 #222);
                }}
                QSlider::sub-page:horizontal {{
                    background: transparent;
                    border: none;
                }}
                QSlider::add-page:horizontal {{
                    background: transparent;
                    border: none;
                }}
                QSlider::handle:horizontal {{
                    background: {color_value};
                    border: 1px solid #333;
                    width: 16px;
                    margin: -3px 0;
                    border-radius: 8px;
                }}
            """)
        return s

    def _make_spinbox(self, slider):
        spin = QSpinBox()
        spin.setRange(slider.minimum(), slider.maximum())
        spin.setValue(slider.value())
        spin.setFixedWidth(80)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        return spin

    def _build_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        for mode in self.MODES:
            tab = QWidget()
            grid = QGridLayout(tab)

            for row, color in enumerate(self.COLORS):
                label = QLabel(f"{color} ({mode[0]}): 0")
                label.setStyleSheet(f"color: {self.COLOR_HEX.get(color, '#ffffff')};")
                slider = self._make_slider(color)
                spin = self._make_spinbox(slider)

                def make_cb(m=mode, c=color, lab=label):
                    def cb(v):
                        self.values[m][c] = int(v)
                        lab.setText(f"{c} ({m[0]}): {v}")
                        self.on_change_callback()
                    return cb

                slider.valueChanged.connect(make_cb())
                grid.addWidget(label, row, 0)
                grid.addWidget(slider, row, 1)
                grid.addWidget(spin, row, 2)

                self.sliders[mode][color] = slider
                self.spinboxes[mode][color] = spin

            tabs.addTab(tab, mode)

        layout.addStretch(1)

    def get_params(self) -> dict:
        return self.values


# ---------- Star tools ----------

# ---------- Dark Mode Stylesheet ----------

def get_dark_stylesheet() -> str:
    return """
    QWidget, QMainWindow, QDialog {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }

    QMenuBar {
        background-color: #252526;
        color: #f0f0f0;
        border-bottom: 1px solid #3d3d3d;
    }

    QMenuBar::item {
        background: transparent;
        padding: 4px 10px;
    }

    QMenuBar::item:selected {
        background-color: #3a3a3a;
    }

    QMenu {
        background-color: #252526;
        color: #f0f0f0;
        border: 1px solid #3d3d3d;
    }

    QMenu::item {
        padding: 6px 24px;
    }

    QMenu::item:selected {
        background-color: #007acc;
        color: #ffffff;
    }

    QFrame {
        background-color: #1e1e1e;
        color: #f0f0f0;
        border: 1px solid #3a3a3a;
    }

    QFrame#photoshopPanel,
    QFrame#photoshopBody {
        background-color: #252526;
        border: 1px solid #3d3d3d;
        border-radius: 8px;
    }

    QLabel {
        color: #f0f0f0;
    }

    QLineEdit,
    QTextEdit,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox {
        background-color: #2d2d2d;
        color: #f0f0f0;
        border: 1px solid #3d3d3d;
        border-radius: 4px;
        padding: 4px 6px;
        selection-background-color: #007acc;
        selection-color: #ffffff;
    }

    QLineEdit:focus,
    QTextEdit:focus,
    QPlainTextEdit:focus,
    QSpinBox:focus,
    QDoubleSpinBox:focus,
    QComboBox:focus {
        border: 1px solid #007acc;
    }

    QComboBox::drop-down {
        border: none;
        width: 24px;
    }

    QComboBox QAbstractItemView {
        background-color: #252526;
        selection-background-color: #007acc;
        selection-color: #ffffff;
    }

    QPushButton {
        background-color: #3a3a3a;
        color: #f0f0f0;
        border: 1px solid #4a4a4a;
        border-radius: 4px;
        padding: 6px 12px;
    }

    QPushButton:hover {
        background-color: #4a4a4a;
    }

    QPushButton:pressed {
        background-color: #2d2d2d;
    }

    QPushButton:disabled {
        background-color: #2a2a2a;
        color: #7a7a7a;
        border-color: #333333;
    }

    QPushButton[accent="true"] {
        background-color: #007acc;
        color: #ffffff;
        border: 1px solid #008cff;
    }

    QPushButton[accent="true"]:hover {
        background-color: #0a86d9;
    }

    QPushButton[accent="true"]:pressed {
        background-color: #0062a3;
    }

    QSlider::groove:horizontal {
        background: #333333;
        height: 4px;
        border-radius: 2px;
    }

    QSlider::sub-page:horizontal {
        background: #007acc;
        border-radius: 2px;
    }

    QSlider::add-page:horizontal {
        background: #333333;
        border-radius: 2px;
    }

    QSlider::handle:horizontal {
        background: #ffffff;
        border: 1px solid #cfcfcf;
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }

    QSlider::handle:horizontal:hover {
        background: #f4f4f4;
        border-color: #007acc;
    }

    QTabWidget::pane {
        border: 1px solid #3d3d3d;
        background: #1e1e1e;
    }

    QTabBar::tab {
        background: #252526;
        color: #dcdcdc;
        padding: 7px 14px;
        border: 1px solid #3d3d3d;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        margin-right: 2px;
    }

    QTabBar::tab:selected {
        background: #3a3a3a;
        color: #ffffff;
    }

    QProgressBar {
        background-color: #252526;
        color: #f0f0f0;
        border: 1px solid #3d3d3d;
        border-radius: 4px;
        text-align: center;
    }

    QProgressBar::chunk {
        background-color: #007acc;
        border-radius: 3px;
    }

    QCheckBox {
        color: #f0f0f0;
        spacing: 6px;
    }

    QToolTip {
        background-color: #252526;
        color: #f0f0f0;
        border: 1px solid #3d3d3d;
    }

    """

def get_light_stylesheet() -> str:
    return """
    QApplication, QWidget, QMainWindow, QDialog {
        background-color: #f4f4f4;
        color: #1f1f1f;
    }

    QMenuBar {
        background-color: #f8f8f8;
        color: #1f1f1f;
        border-bottom: 1px solid #cfcfcf;
    }

    QMenuBar::item {
        background: transparent;
        padding: 4px 10px;
    }

    QMenuBar::item:selected {
        background-color: #e6e6e6;
    }

    QMenu {
        background-color: #ffffff;
        color: #1f1f1f;
        border: 1px solid #cfcfcf;
    }

    QMenu::item {
        padding: 6px 24px 6px 24px;
    }

    QMenu::item:selected {
        background-color: #d9e8fb;
        color: #111111;
    }

    QToolBar {
        background-color: #f3f3f3;
        border-bottom: 1px solid #d6d6d6;
    }
    """

def star_shrink_pixinsight(img: np.ndarray, amount: float = 0.6, selection: float = 0.5) -> np.ndarray:
    if img is None:
        return None

    img_f = img.astype(np.float32) / 255.0

    blur = cv2.GaussianBlur(img_f, (0, 0), 2.0)
    highpass = cv2.subtract(img_f, blur)

    gray = cv2.cvtColor((highpass * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    protected = cv2.addWeighted(mask.astype(np.float32), selection,
                                eroded_mask.astype(np.float32), 1 - selection, 0)
    protected = np.clip(protected, 0, 255).astype(np.uint8)

    mask_f = protected.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (0, 0), 1.0)
    mask_f = np.repeat(mask_f[:, :, np.newaxis], 3, axis=2)

    eroded_img = cv2.erode(img_f, kernel, iterations=1)

    shrink = img_f * (1 - amount * mask_f) + eroded_img * (amount * mask_f)

    return np.clip(shrink * 255, 0, 255).astype(np.uint8)

class CorrectionDialog(QDialog):
    def __init__(self, camera_raw_panel, hsl_panel,
                 apply_callback, reset_callback, parent=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.setWindowTitle("Correction")
        self.setMinimumWidth(500)
        self._camera_raw_panel = camera_raw_panel
        self._hsl_panel = hsl_panel
        self._apply_callback = apply_callback
        self._reset_callback = reset_callback
        self._committed_snapshot = self._capture_snapshot()
        self._skip_close_restore_once = False

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        layout.addWidget(QLabel("Camera FITS"))
        layout.addWidget(camera_raw_panel)

        layout.addWidget(QLabel("Color Mixer (HSL)"))
        layout.addWidget(hsl_panel)

        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("OK")
        self.btn_apply = QPushButton("Apply")
        self.btn_reset = QPushButton("Reset")
        self.btn_close = QPushButton("Close")
        

        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_apply)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addWidget(self.btn_close)
        

        layout.addLayout(btn_layout)
        

        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_close.clicked.connect(self._on_cancel_and_close)
        self.btn_ok.clicked.connect(self._on_ok)

    def _capture_snapshot(self):
        snapshot = {
            "camera_raw": {},
            "hsl": {},
        }
        if hasattr(self._camera_raw_panel, "sld_temp"):
            snapshot["camera_raw"] = {
                "temp": int(self._camera_raw_panel.sld_temp.value()),
                "tint": int(self._camera_raw_panel.sld_tint.value()),
                "exp": int(self._camera_raw_panel.sld_exp.value()),
                "contrast": int(self._camera_raw_panel.sld_contrast.value()),
                "sat": int(self._camera_raw_panel.sld_sat.value()),
                "vib": int(self._camera_raw_panel.sld_vib.value()),
                "texture": int(self._camera_raw_panel.sld_texture.value()),
                "dehaze": int(self._camera_raw_panel.sld_dehaze.value()),
                "clarity": int(self._camera_raw_panel.sld_clarity.value()),
                "noise": int(self._camera_raw_panel.sld_noise.value()),
            }

        if hasattr(self._hsl_panel, "MODES") and hasattr(self._hsl_panel, "COLORS"):
            for mode in self._hsl_panel.MODES:
                snapshot["hsl"][mode] = {}
                for color in self._hsl_panel.COLORS:
                    slider = self._hsl_panel.sliders.get(mode, {}).get(color)
                    snapshot["hsl"][mode][color] = int(slider.value()) if slider is not None else 0
        return snapshot

    def _restore_snapshot(self, snapshot):
        if not isinstance(snapshot, dict):
            return

        camera_raw = snapshot.get("camera_raw", {})
        for attr_name, key in (
            ("sld_temp", "temp"),
            ("sld_tint", "tint"),
            ("sld_exp", "exp"),
            ("sld_contrast", "contrast"),
            ("sld_sat", "sat"),
            ("sld_vib", "vib"),
            ("sld_texture", "texture"),
            ("sld_dehaze", "dehaze"),
            ("sld_clarity", "clarity"),
            ("sld_noise", "noise"),
        ):
            slider = getattr(self._camera_raw_panel, attr_name, None)
            if slider is not None:
                slider.setValue(int(camera_raw.get(key, 0)))

        hsl = snapshot.get("hsl", {})
        if hasattr(self._hsl_panel, "MODES") and hasattr(self._hsl_panel, "COLORS"):
            for mode in self._hsl_panel.MODES:
                mode_values = hsl.get(mode, {}) if isinstance(hsl.get(mode, {}), dict) else {}
                for color in self._hsl_panel.COLORS:
                    slider = self._hsl_panel.sliders.get(mode, {}).get(color)
                    if slider is not None:
                        slider.setValue(int(mode_values.get(color, 0)))

    def _on_apply(self):
        if callable(self._apply_callback):
            self._apply_callback()
        self._committed_snapshot = self._capture_snapshot()

    def _on_reset(self):
        if callable(self._reset_callback):
            self._reset_callback()

    def _on_ok(self):
        self._on_apply()
        self._skip_close_restore_once = True
        self.close()

    def _on_cancel_and_close(self):
        self._restore_snapshot(self._committed_snapshot)
        parent = self.parent()
        if parent is not None and hasattr(parent, "apply_full_processing"):
            parent.apply_full_processing()
        self._skip_close_restore_once = True
        self.close()

    def closeEvent(self, event):
        if self._skip_close_restore_once:
            self._skip_close_restore_once = False
            super().closeEvent(event)
            return
        self._restore_snapshot(self._committed_snapshot)
        parent = self.parent()
        if parent is not None and hasattr(parent, "apply_full_processing"):
            parent.apply_full_processing()
        super().closeEvent(event)

class ColorCalibrationDialog(QDialog):
    requestRoiPick = pyqtSignal()
    roiChanged = pyqtSignal(int, int, int, int)

    def __init__(self, bn_callback, parent=None):
        super().__init__(parent)
        apply_dialog_window_flags(self)
        self.bn_callback = bn_callback
        self.setWindowTitle("Background Neutralization")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        apply_standard_layout_margins(layout)

        info = QLabel(
            "Background Neutralization uses ROI drawn manually on the image."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        roi_frame = QFrame()
        roi_layout = QGridLayout(roi_frame)
        roi_layout.addWidget(QLabel("ROI y1:"), 0, 0)
        roi_layout.addWidget(QLabel("ROI y2:"), 0, 2)
        roi_layout.addWidget(QLabel("ROI x1:"), 1, 0)
        roi_layout.addWidget(QLabel("ROI x2:"), 1, 2)
        self.spin_y1 = QSpinBox()
        self.spin_y2 = QSpinBox()
        self.spin_x1 = QSpinBox()
        self.spin_x2 = QSpinBox()
        for spin in (self.spin_y1, self.spin_y2, self.spin_x1, self.spin_x2):
            spin.setRange(0, 99999)
            spin.valueChanged.connect(self._on_roi_spin_changed)
        roi_layout.addWidget(self.spin_y1, 0, 1)
        roi_layout.addWidget(self.spin_y2, 0, 3)
        roi_layout.addWidget(self.spin_x1, 1, 1)
        roi_layout.addWidget(self.spin_x2, 1, 3)
        layout.addWidget(roi_frame)

        self.lbl_roi_status = QLabel("Draw ROI on image to enable Background Neutralization.")
        self.lbl_roi_status.setWordWrap(True)
        layout.addWidget(self.lbl_roi_status)

        buttons = QHBoxLayout()
        self.btn_pick_roi = QPushButton("Draw ROI on image")
        self.btn_bn = QPushButton("Run Background Neutralization")
        self.btn_close = QPushButton("Close")
        self.btn_pick_roi.clicked.connect(self.requestRoiPick.emit)
        self.btn_bn.clicked.connect(self._run_bn)
        self.btn_close.clicked.connect(self.close)
        buttons.addWidget(self.btn_pick_roi)
        buttons.addWidget(self.btn_bn)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_close)
        layout.addLayout(buttons)
        self.set_manual_roi_ready(False)

    def set_image_shape(self, height: int, width: int):
        h = max(1, int(height))
        w = max(1, int(width))
        self.spin_y1.setRange(0, h - 1)
        self.spin_y2.setRange(1, h)
        self.spin_x1.setRange(0, w - 1)
        self.spin_x2.setRange(1, w)

        y1 = max(0, int(round(h * 0.35)))
        y2 = min(h, max(y1 + 1, int(round(h * 0.65))))
        x1 = max(0, int(round(w * 0.35)))
        x2 = min(w, max(x1 + 1, int(round(w * 0.65))))
        self.spin_y1.setValue(y1)
        self.spin_y2.setValue(y2)
        self.spin_x1.setValue(x1)
        self.spin_x2.setValue(x2)

    def get_roi(self):
        y1 = int(self.spin_y1.value())
        y2 = int(self.spin_y2.value())
        x1 = int(self.spin_x1.value())
        x2 = int(self.spin_x2.value())
        return (y1, y2, x1, x2)

    def set_roi_values(self, y1: int, y2: int, x1: int, x2: int):
        for spin, value in (
            (self.spin_y1, int(y1)),
            (self.spin_y2, int(y2)),
            (self.spin_x1, int(x1)),
            (self.spin_x2, int(x2)),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._on_roi_spin_changed()

    def set_selection_active(self, active: bool):
        if active:
            self.btn_pick_roi.setText("ROI edit mode: ON (click to stop)")
            self.lbl_roi_status.setText("Drag ROI edges/corners to resize or drag inside ROI to move.")
        else:
            self.btn_pick_roi.setText("Draw ROI on image")

    def set_manual_roi_ready(self, ready: bool):
        self.btn_bn.setEnabled(bool(ready))
        if ready:
            self.lbl_roi_status.setText("ROI selected manually. You can run Background Neutralization.")
        else:
            self.lbl_roi_status.setText("Draw ROI on image to enable Background Neutralization.")

    def _on_roi_spin_changed(self):
        y1, y2, x1, x2 = self.get_roi()
        self.roiChanged.emit(int(y1), int(y2), int(x1), int(x2))

    def _run_bn(self):
        if callable(self.bn_callback):
            self.bn_callback(self.get_roi())


# ---------- GĹ‚Ăłwna aplikacja ----------

class DraggableTopActionButton(QPushButton):
    MIME_TYPE = "application/x-astro-top-action-button"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drag_start_pos = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            super().mouseMoveEvent(event)
            return

        if (event.pos() - self._drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            super().mouseMoveEvent(event)
            return

        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(self.MIME_TYPE, b"top-action")
        drag.setMimeData(mime)
        drag.setPixmap(self.grab())
        drag.setHotSpot(event.pos())
        drag.exec_(Qt.MoveAction)


class ReorderableTopActionsBar(QFrame):
    MIME_TYPE = DraggableTopActionButton.MIME_TYPE
    orderChanged = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(self.MIME_TYPE) and isinstance(event.source(), DraggableTopActionButton):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(self.MIME_TYPE) and isinstance(event.source(), DraggableTopActionButton):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        source_button = event.source()
        if not event.mimeData().hasFormat(self.MIME_TYPE) or not isinstance(source_button, DraggableTopActionButton):
            super().dropEvent(event)
            return

        layout = self.layout()
        if layout is None:
            event.ignore()
            return

        source_index = layout.indexOf(source_button)
        if source_index < 0:
            event.ignore()
            return

        target_index = self._target_layout_index(event.pos().x(), source_button)
        if target_index > source_index:
            target_index -= 1

        if target_index != source_index and target_index >= 0:
            layout.removeWidget(source_button)
            layout.insertWidget(target_index, source_button)
            self.orderChanged.emit()

        event.acceptProposedAction()

    def _target_layout_index(self, cursor_x: int, source_button: QPushButton) -> int:
        layout = self.layout()
        if layout is None:
            return -1

        button_widgets = []
        for idx in range(layout.count()):
            item = layout.itemAt(idx)
            widget = item.widget() if item is not None else None
            if isinstance(widget, DraggableTopActionButton) and widget is not source_button:
                button_widgets.append((idx, widget))

        button_widgets.sort(key=lambda item: item[0])

        for layout_index, widget in button_widgets:
            if cursor_x < widget.geometry().center().x():
                return layout_index

        return self._index_before_stretch(layout)

    @staticmethod
    def _index_before_stretch(layout) -> int:
        for idx in range(layout.count()):
            item = layout.itemAt(idx)
            spacer = item.spacerItem() if item is not None else None
            if spacer is None:
                continue
            if bool(spacer.expandingDirections() & Qt.Horizontal):
                return idx
        return layout.count()

class AstroApp(QMainWindow):
    def _inject_background_calibration(self, basic_params: dict) -> dict:
        params = dict(basic_params or {})
        gains = getattr(self, "background_calibration_rgb", {}) or {}
        params["calibration_red"] = float(gains.get("r", 0.0))
        params["calibration_green"] = float(gains.get("g", 0.0))
        params["calibration_blue"] = float(gains.get("b", 0.0))
        return params

    def apply_full_processing(self):
        if self.magic_img is None:
            return

        self.cancel_params_preview()

        base_img = self.preview_override_img.copy() if getattr(self, "preview_override_img", None) is not None else self.get_effective_magic_img()
        if base_img is None:
            return
        display_base_img = self.apply_channel_visibility(base_img)
        self.viewer.set_before(np_to_qpixmap(display_base_img))

        basic_params = self._inject_background_calibration(self.camera_raw_panel.get_params())
        hsl_params = self.hsl_panel.get_params()

        img = ColorProcessor.apply_camera_raw_and_hsl(
            base_img,
            basic_params,
            hsl_params,
        )

        # đź”Ą DODAJ LEVELS
        levels = self.levels_window.levels_widget.get_params()
        img = apply_levels(
            img,
            levels["black"],
            levels["gamma"],
            levels["white"],
            levels["channels"]
        )

        if hasattr(self, "curves_window"):
            curves = self.curves_window.get_params()
            if curves["points"]:
                self.add_layer("curves")
            if self.layer_visibility.get("curves", True):
                img = apply_curves_lut(img, curves["points"], curves["channels"], curves["curve_mode"])

        self.processed_img = img
        self.analysis_dirty = True
        display_img = self.apply_channel_visibility(img)

        if self.processed_img is not None:
            pix_after = np_to_qpixmap(display_img)
            self.viewer.set_after(pix_after)
        if self.histogram_window.isVisible():
            self.histogram_window.set_image(display_img)
        self.update_photoshop_panel()
        self.update_viewer_overlay()
    
    
    def get_preview_image(self, img: np.ndarray, max_side: int = 1200) -> np.ndarray:
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(1.0, max_side / float(max(h, w)))
        if scale >= 1.0:
            return img
        preview = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return preview

    def apply_preview_processing(self):
        if self.magic_img is None:
            return

        base_img = self.preview_override_img.copy() if getattr(self, "preview_override_img", None) is not None else self.get_effective_magic_img()
        if base_img is None:
            return

        preview_img = self.get_preview_image(base_img)
        display_base_img = self.apply_channel_visibility(preview_img)
        self.viewer.set_before(np_to_qpixmap(display_base_img))

        basic_params = self._inject_background_calibration(self.camera_raw_panel.get_params())
        hsl_params = self.hsl_panel.get_params()

        img = ColorProcessor.apply_camera_raw_and_hsl(
            preview_img,
            basic_params,
            hsl_params,
        )

        levels = self.levels_window.levels_widget.get_params()
        img = apply_levels(
            img,
            levels["black"],
            levels["gamma"],
            levels["white"],
            levels["channels"]
        )

        if hasattr(self, "curves_window"):
            curves = self.curves_window.get_params()
            if self.layer_visibility.get("curves", True):
                img = apply_curves_lut(img, curves["points"], curves["channels"], curves["curve_mode"])

        preview_processed = img
        display_img = self.apply_channel_visibility(preview_processed)
        if preview_processed is not None:
            self.viewer.set_after(np_to_qpixmap(display_img))

    def show_histogram_window(self):

        if self.processed_img is not None:
            self.histogram_window.set_image(self.apply_channel_visibility(self.processed_img))

        elif self.magic_img is not None:
            self.histogram_window.set_image(self.apply_channel_visibility(self.magic_img))

        elif self.original_img is not None:
            self.histogram_window.set_image(self.apply_channel_visibility(self.original_img))

        self.histogram_window.show()
        self.histogram_window.raise_()
        self.histogram_window.activateWindow()
        self.log("Histogram window opened.")

    def _apply_channel_calibration_scales(self, scales: dict, mode: str):
        if not isinstance(scales, dict):
            return

        def _scale_to_slider(scale_value: float) -> int:
            raw = int(np.clip(round((scale_value - 1.0) * 200.0), -100, 100))
            # Preserve subtle but non-zero calibration deltas instead of rounding to 0.
            if raw == 0:
                delta = float(scale_value - 1.0)
                if abs(delta) >= 0.001:
                    raw = 1 if delta > 0 else -1
            return raw

        r_scale = float(scales.get("r", 1.0))
        g_scale = float(scales.get("g", 1.0))
        b_scale = float(scales.get("b", 1.0))
        r_value = _scale_to_slider(r_scale)
        g_value = _scale_to_slider(g_scale)
        b_value = _scale_to_slider(b_scale)

        self.background_calibration_rgb = {
            "r": float(r_value),
            "g": float(g_value),
            "b": float(b_value),
        }

        self.last_color_calibration_method = mode
        quality = scales.get("quality", "unknown")
        self.log(
            f"{mode} applied. R:{r_value:+d} G:{g_value:+d} B:{b_value:+d} ({quality})",
            "success",
        )
        self.apply_full_processing()

    def apply_background_neutralization(self, roi=None):
        base_img = self._get_bn_source_image()
        if base_img is None:
            self.log("Background Neutralization skipped: no image loaded.", "warning")
            return

        h, w = base_img.shape[:2]
        if roi is None:
            y1 = max(0, int(round(h * 0.35)))
            y2 = min(h, max(y1 + 1, int(round(h * 0.65))))
            x1 = max(0, int(round(w * 0.35)))
            x2 = min(w, max(x1 + 1, int(round(w * 0.65))))
            roi = (y1, y2, x1, x2)

        try:
            y1, y2, x1, x2 = (int(v) for v in roi)
        except Exception:
            self.log(f"Background Neutralization skipped: invalid ROI {roi}.", "warning")
            return
        y1, y2, x1, x2 = self._normalize_roi(y1, y2, x1, x2, h, w)

        try:
            rgb_f32 = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            _ = neutralize_background(rgb_f32, (y1, y2, x1, x2))
            roi_data = rgb_f32[y1:y2, x1:x2, :]
            means = roi_data.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
            target = float(np.mean(means, dtype=np.float64))
            eps = 1e-6
            scales = {
                "r": float(target / (float(means[0]) + eps)),
                "g": float(target / (float(means[1]) + eps)),
                "b": float(target / (float(means[2]) + eps)),
                "quality": f"roi=({y1},{y2},{x1},{x2})",
            }
            self._apply_channel_calibration_scales(scales, "BN")
            if self.processed_img is not None:
                self.add_thumbnail("Background Neutralization", self.processed_img)
        except Exception as exc:
            self.log(f"Background Neutralization failed: {exc}", "error")

    def reset_all_sliders(self):
        self.last_color_calibration_method = None
        self.background_calibration_rgb = {"r": 0.0, "g": 0.0, "b": 0.0}
        self.camera_raw_panel.sld_temp.setValue(0)
        self.camera_raw_panel.sld_tint.setValue(0)
        self.camera_raw_panel.sld_exp.setValue(0)
        self.camera_raw_panel.sld_contrast.setValue(0)
        self.camera_raw_panel.sld_sat.setValue(0)
        self.camera_raw_panel.sld_vib.setValue(0)
        self.camera_raw_panel.sld_texture.setValue(0)
        self.camera_raw_panel.sld_dehaze.setValue(0)
        self.camera_raw_panel.sld_clarity.setValue(0)
        self.camera_raw_panel.sld_noise.setValue(0)

        for mode in self.hsl_panel.MODES:
            for color in self.hsl_panel.COLORS:
                self.hsl_panel.sliders[mode][color].setValue(0)

        if hasattr(self, "curves_window"):
            self.curves_window.reset_curves(notify=False)

        self.apply_full_processing()
    
    def select_onnx_models(self):
        denoise_path, _ = self._show_open_file_dialog(
            "Select Denoise ONNX Model",
            "ONNX Models (*.onnx);;All Files (*)",
            "",
        )
        if denoise_path:
            self.denoise_model_path = denoise_path

        bg_path, _ = self._show_open_file_dialog(
            "Select Background Removal ONNX Model",
            "ONNX Models (*.onnx);;All Files (*)",
            "",
        )
        if bg_path:
            self.bg_removal_model_path = bg_path

        save_config(
            self.denoise_model_path,
            self.bg_removal_model_path,
            self.dark_mode,
            self.starnet_path,
            self.plate_solve_api_key,
            self.plate_solve_pixel_size_um,
            self.plate_solve_focal_length_mm,
            self.starnet_stride,
            self.gemini_api_key,
            self.gemini_model,
        )
        self._update_models_label()
        self.log("ONNX model paths updated.")

    def select_starnet_path(self):
        starnet_path, _ = self._show_open_file_dialog(
            "Select StarNet++ Executable",
            "Executables (*.exe);;All Files (*)",
            "",
        )
        if not starnet_path:
            self.log("StarNet++ selection canceled.", "warning")
            return

        self.starnet_path = starnet_path
        save_config(
            self.denoise_model_path,
            self.bg_removal_model_path,
            self.dark_mode,
            self.starnet_path,
            self.plate_solve_api_key,
            self.plate_solve_pixel_size_um,
            self.plate_solve_focal_length_mm,
            self.starnet_stride,
            self.gemini_api_key,
            self.gemini_model,
        )
        self._update_models_label()
        self.log(f"StarNet++ path updated: {self.starnet_path}", "success")

    def select_deepsnr_path(self):
        deepsnr_path, _ = self._show_open_file_dialog(
            "Select deepSNR Executable",
            "Executables (*.exe *.bin *.sh *.py);;All Files (*)",
            "",
        )
        if not deepsnr_path:
            self.log("deepSNR selection canceled.", "warning")
            return

        self.deepsnr_path = deepsnr_path
        APP_PREFERENCES["deepsnr_path"] = self.deepsnr_path
        save_config(
            self.denoise_model_path,
            self.bg_removal_model_path,
            self.dark_mode,
            self.starnet_path,
            self.plate_solve_api_key,
            self.plate_solve_pixel_size_um,
            self.plate_solve_focal_length_mm,
            self.starnet_stride,
            self.gemini_api_key,
            self.gemini_model,
            deepsnr_path=self.deepsnr_path,
            deepsnr_args=self.deepsnr_args,
        )
        self.log(f"deepSNR path updated: {self.deepsnr_path}", "success")

    def _export_temp_image_for_external_tool(self, prefix: str) -> str:
        source_img = None
        if isinstance(self.processed_img, np.ndarray):
            source_img = self.processed_img
        elif isinstance(self.magic_img, np.ndarray):
            source_img = self.magic_img
        elif isinstance(self.original_img, np.ndarray):
            source_img = self.original_img
        if source_img is None:
            return ""

        temp_dir = tempfile.mkdtemp(prefix=f"astro_{prefix}_")
        temp_path = os.path.join(temp_dir, "input.tif")
        if not cv2.imwrite(temp_path, source_img):
            shutil.rmtree(temp_dir, ignore_errors=True)
            return ""
        return temp_path

    def run_deepsnr(self):
        if not self.deepsnr_path or not os.path.exists(self.deepsnr_path):
            self.log("deepSNR path is not set. Select deepSNR in Preferences first.", "warning")
            return

        dialog = self._get_deepsnr_dialog()
        if dialog.exec_() != QDialog.Accepted:
            self.log("deepSNR canceled.", "warning")
            return

        params = dialog.get_parameters()
        self.deepsnr_args = str(params.get("args") or "{input}").strip() or "{input}"
        APP_PREFERENCES["deepsnr_args"] = self.deepsnr_args
        save_config(
            self.denoise_model_path,
            self.bg_removal_model_path,
            self.dark_mode,
            self.starnet_path,
            self.plate_solve_api_key,
            self.plate_solve_pixel_size_um,
            self.plate_solve_focal_length_mm,
            self.starnet_stride,
            self.gemini_api_key,
            self.gemini_model,
            deepsnr_path=self.deepsnr_path,
            deepsnr_args=self.deepsnr_args,
        )

        input_path = self._export_temp_image_for_external_tool("deepsnr")
        args_template = str(getattr(self, "deepsnr_args", "{input}") or "{input}").strip()
        if not args_template:
            args_template = "{input}"
        if "{input}" in args_template and not input_path:
            self.log("deepSNR skipped: no image loaded for {input} argument.", "warning")
            return

        output_path = ""
        if "{output}" in args_template:
            output_dir = tempfile.mkdtemp(prefix="astro_deepsnr_output_")
            output_path = os.path.join(output_dir, "deepsnr_output.tif")

        try:
            rendered_args = args_template.format(input=input_path, output=output_path)
        except Exception as e:
            self.log(f"deepSNR args template error: {e}", "error")
            return

        try:
            args_list = shlex.split(rendered_args, posix=(sys.platform != "win32")) if rendered_args.strip() else []
        except Exception as e:
            self.log(f"deepSNR args parse error: {e}", "error")
            return

        command = [self.deepsnr_path] + args_list
        if self.deepsnr_path.lower().endswith(".py"):
            command = [sys.executable, self.deepsnr_path] + args_list

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            if output_path:
                result = subprocess.run(
                    command,
                    cwd=os.path.dirname(self.deepsnr_path) or None,
                    capture_output=True,
                    text=True,
                    timeout=7200,
                    creationflags=creationflags,
                )
                if result.returncode != 0:
                    stderr = (result.stderr or "").strip()
                    self.log(f"deepSNR failed (code {result.returncode}): {stderr or 'no stderr'}", "error")
                    return
                if os.path.exists(output_path):
                    self.load_image_from_path(output_path)
                    self.log(f"deepSNR finished. Loaded output: {output_path}", "success")
                else:
                    self.log("deepSNR finished but output file was not created.", "warning")
            else:
                subprocess.Popen(command, cwd=os.path.dirname(self.deepsnr_path) or None, creationflags=creationflags)
                self.log("deepSNR started (background).", "success")
        except subprocess.TimeoutExpired:
            self.log("deepSNR timed out after 2 hours.", "error")
        except Exception as e:
            self.log(f"deepSNR failed to start: {e}", "error")

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

        self.setWindowTitle("Astro Ai Plus v1.0.0")

        icon_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "main_icon.ico"
        )

        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.resize(1600, 900)
        self.compare_dialogs = []
        self.history_items = []
        self.arduino_joystick_worker = None
        self.params_preview_timer = QTimer(self)
        self.params_preview_timer.setSingleShot(True)
        self.params_preview_timer.setInterval(200)
        self.params_preview_timer.timeout.connect(self.apply_full_processing)

        # ---------- obrazy ----------
        self.original_img = None
        self.magic_img = None
        self.processed_img = None
        self.preview_override_img = None
        self.layer_images = {}
        self.layer_titles = {
            "background": "Background",
            "blur": "Gaussian Blur",
            "stars": "Stars",
            "curves": "Curves",
            "plate_solve_overlay": "Plate Solve Overlay",
            "grid_overlay": "Grid Overlay",
            "object_labels": "Object Labels",
            "constellation_overlay": "Constellation Overlay",
        }
        self.channel_visibility = {"r": True, "g": True, "b": True}
        self.layer_visibility = {
            "background": True,
            "blur": True,
            "stars": True,
            "curves": True,
            "plate_solve_overlay": False,
            "grid_overlay": False,
            "object_labels": False,
            "constellation_overlay": False,
        }
        self.layer_order = [
            "background",
            "plate_solve_overlay",
            "grid_overlay",
            "object_labels",
            "constellation_overlay",
        ]
        self.overlay_layer_pixmaps = {}
        self.constellation_lines = []
        self.catalog_sources = [
            "Messier",
            "NGC",
            "IC",
            "Sharpless",
            "Gaia",
            "Hipparcos",
        ]
        self.layer_counters = {
            "layer": 0,
            "levels": 0,
            "curves": 0,
        }
        self.layer_types = {
            "background": "raster",
            "blur": "effect",
            "stars": "effect",
            "curves": "adjustment",
            "plate_solve_overlay": "overlay",
            "grid_overlay": "overlay",
            "object_labels": "overlay",
            "constellation_overlay": "overlay",
        }
        self.layer_masks = {}
        self.selected_layer_key = "background"
        self.mask_pick_target = None
        self.mask_pick_tolerance = 28

        # ---------- save ----------
        self.current_save_path = None
        self.current_image_path = None
        self.latest_image_analysis = {}
        self.analysis_dirty = True
        self.plate_solve_object_info = {}
        self.ai_assistant_panel = None
        self.blur_dialog = None
        self.crop_dialog = None
        self.color_calibration_dialog = None
        self.plate_solve_dialog = None
        self.starnet_dialog = None
        self.deepsnr_dialog = None
        self.fly3d_dialog = None
        self.preferences_dialog = None
        self.active_image_window = None

        # ---------- undo / redo ----------
        self.undo_stack = []
        self.redo_stack = []

        # ---------- processing history & thumbnails ----------
        self.processing_history = []  # Lista (nazwa_operacji, obraz)
        self.max_thumbnails = 15  # Maksymalnie 15 miniatur
        self.thumbnail_size = 120  # Rozmiar miniatur w pikselach
        self.selected_thumbnail_index = -1
        self.current_history_node_id = None
        self.thumbnail_next_id = 1
        self._restoring_thumbnail = False
        self._crop_overlay_active = False
        self._crop_overlay_rect = (0, 0, 1, 1)
        self._crop_overlay_grid_index = 0
        self._crop_pick_active = False
        self._crop_pick_anchor = None
        self._crop_pick_operation = None
        self._crop_pick_start_rect = None
        self._crop_pick_last_point = None
        self._bn_overlay_active = False
        self._bn_overlay_rect = (0, 0, 1, 1)
        self._bn_pick_active = False
        self._bn_pick_anchor = None
        self._bn_pick_operation = None
        self._bn_pick_start_rect = None
        self._bn_pick_last_point = None
        self.corrections_timer = QTimer()  # Timer dla finalizacji Camera Raw/HSL
        self.corrections_timer.setSingleShot(True)
        self.corrections_timer.setInterval(1000)  # 1 sekunda opĂłĹşnienia
        self.corrections_timer.timeout.connect(self._finalize_camera_raw_hsl)
        self.last_corrections_snapshot = None  # Snapshot poprzednich wartoĹ›ci
        self.last_color_calibration_method = None
        self.background_calibration_rgb = {"r": 0.0, "g": 0.0, "b": 0.0}

        # ---------- modele ----------
        self.denoise_model_path = None
        self.bg_removal_model_path = None
        self.starnet_path = None
        self.deepsnr_path = None
        self.deepsnr_args = "{input}"
        self.starnet_stride = 16

        # ---------- theme ----------
        self.theme_name = "Fusion Dark"
        self.language = "pl"
        self.processor_cores = max(1, (os.cpu_count() or 4) // 2)
        self.onnx_provider = "Auto"
        self.dark_mode = True

        # ---------- config ----------
        config = load_config()

        self.denoise_model_path = config.get("denoise_model_path")
        self.bg_removal_model_path = config.get("bg_removal_model_path")
        self.starnet_path = config.get("starnet_path")
        self.deepsnr_path = config.get("deepsnr_path")
        self.deepsnr_args = str(config.get("deepsnr_args", "{input}") or "{input}")
        self.starnet_stride = int(config.get("starnet_stride") or 16)
        self.theme_name = config.get("theme_name") or ("Fusion Dark" if config.get("dark_mode", True) else "Light")
        self.language = config.get("language", "pl") or "pl"
        self.processor_cores = int(config.get("processor_cores") or self.processor_cores)
        self.onnx_provider = config.get("onnx_provider", "Auto") or "Auto"
        self.dark_mode = self.theme_name.lower() != "light"

        self.plate_solve_api_key = config.get("api_key", "") or ""
        self.plate_solve_pixel_size_um = float(config.get("pixel_size_um") or 5.4)
        self.plate_solve_focal_length_mm = float(config.get("focal_length_mm") or 800.0)
        self.gemini_api_key = config.get("gemini_api_key", "") or ""
        self.gemini_model = FIXED_GEMINI_MODEL
        self.workspaces = config.get("workspaces", []) if isinstance(config.get("workspaces", []), list) else []
        self.home_folder = str(config.get("home_folder", "") or "").strip()
        self.topbar_button_order = config.get("topbar_button_order", []) if isinstance(config.get("topbar_button_order", []), list) else []
        if self.home_folder and not os.path.isdir(self.home_folder):
            self.home_folder = ""
        APP_PREFERENCES["workspaces"] = self.workspaces
        APP_PREFERENCES["home_folder"] = self.home_folder
        APP_PREFERENCES["topbar_button_order"] = list(self.topbar_button_order)
        APP_PREFERENCES["deepsnr_path"] = self.deepsnr_path
        APP_PREFERENCES["deepsnr_args"] = self.deepsnr_args

        # ---------- windows ----------
        self.levels_window = LevelsWindow(parent=self)
        self.curves_window = CurvesWindow(parent=self)

        self.histogram_window = HistogramWindow(parent=self)
        self._zoom_sync_in_progress = False
        self.levels_window.levels_widget.zoomChanged.connect(
            lambda zoom, center: self._sync_histogram_zoom_state("levels", zoom, center)
        )
        self.curves_window.curves_widget.zoomChanged.connect(
            lambda zoom, center: self._sync_histogram_zoom_state("curves", zoom, center)
        )
        self.histogram_window.histogram_widget.zoomChanged.connect(
            lambda zoom, center: self._sync_histogram_zoom_state("histogram", zoom, center)
        )
        self.histogram_window.zoomSyncChanged.connect(self._on_histogram_zoom_sync_toggled)
        self.console_window = ConsoleWindow(parent=self)
        self.photoshop_menu_dialog = PhotoshopMenuDialog(parent=self)
        self.star_shrink_dialog = None
        self._star_shrink_context = None

        # ---------- apply theme ----------
        self.apply_dark_mode(self.dark_mode)

        # ---------- UI ----------
        self.init_ui()
        self._update_ui_language()

        # ---------- correction dialog ----------
        self.correction_dialog = CorrectionDialog(
            self.camera_raw_panel,
            self.hsl_panel,
            self.apply_correction_from_dialog,
            self.reset_all_sliders,
            parent=self
        )

        self._begin_dialog_compare(self.correction_dialog)
        self.correction_dialog.show()
        self.correction_dialog.raise_()
        self.correction_dialog.activateWindow()
        self.log("Application started.")

    def _on_histogram_zoom_sync_toggled(self, enabled):
        if bool(enabled):
            zoom, center = self.histogram_window.histogram_widget.get_zoom_state()
            self._sync_histogram_zoom_state("histogram", zoom, center)

    def _sync_histogram_zoom_state(self, source, zoom_factor, view_center):
        if self._zoom_sync_in_progress:
            return
        if not self.histogram_window.is_zoom_sync_enabled():
            return

        self._zoom_sync_in_progress = True
        try:
            if source != "histogram":
                self.histogram_window.apply_external_zoom_state(zoom_factor, view_center)
            if source != "levels":
                self.levels_window.apply_external_zoom_state(zoom_factor, view_center)
            if source != "curves":
                self.curves_window.apply_external_zoom_state(zoom_factor, view_center)
        finally:
            self._zoom_sync_in_progress = False

    def _get_blur_dialog(self):
        if self.blur_dialog is None:
            self.blur_dialog = BlurDialog(self)
        return self.blur_dialog

    def _get_crop_dialog(self):
        if self.crop_dialog is None:
            self.crop_dialog = CropDialog(self)
            self.crop_dialog.previewRectChanged.connect(self._on_crop_rect_from_dialog_changed)
            self.crop_dialog.gridChanged.connect(self._set_crop_overlay_grid)
            self.crop_dialog.finished.connect(self._on_crop_dialog_closed)
        return self.crop_dialog

    def _get_color_calibration_dialog(self):
        if self.color_calibration_dialog is None:
            self.color_calibration_dialog = ColorCalibrationDialog(
                self.apply_background_neutralization,
                parent=self,
            )
            self.color_calibration_dialog.requestRoiPick.connect(self.start_bn_roi_pick)
            self.color_calibration_dialog.roiChanged.connect(self._on_bn_roi_from_dialog_changed)
            self.color_calibration_dialog.finished.connect(self._on_color_calibration_dialog_closed)
        return self.color_calibration_dialog

    def _get_plate_solve_dialog(self):
        if self.plate_solve_dialog is None:
            self.plate_solve_dialog = PlateSolveDialog(
                self,
                pixel_size_um=self.plate_solve_pixel_size_um,
                focal_length_mm=self.plate_solve_focal_length_mm,
                api_key=self.plate_solve_api_key,
            )
        else:
            self.plate_solve_dialog.set_parameters(
                self.plate_solve_pixel_size_um,
                self.plate_solve_focal_length_mm,
                self.plate_solve_api_key,
            )
        return self.plate_solve_dialog

    def _get_starnet_dialog(self):
        if self.starnet_dialog is None:
            self.starnet_dialog = StarNetDialog(
                self,
                stride=self.starnet_stride,
                starnet_path=self.starnet_path,
            )
        else:
            self.starnet_dialog.set_parameters(self.starnet_stride, self.starnet_path)
        return self.starnet_dialog

    def _get_deepsnr_dialog(self):
        if self.deepsnr_dialog is None:
            self.deepsnr_dialog = DeepSNRDialog(
                self,
                deepsnr_path=self.deepsnr_path,
                deepsnr_args=self.deepsnr_args,
            )
        else:
            self.deepsnr_dialog.set_parameters(self.deepsnr_path, self.deepsnr_args)
        return self.deepsnr_dialog

    def _get_fly3d_dialog(self):
        source = self.get_effective_magic_img() if self.magic_img is not None else None
        if source is None:
            source = self.magic_img
        if source is None:
            return None

        default_dir = self._get_default_dialog_directory()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = os.path.join(default_dir, f"3d_fly_{stamp}.mp4")
        self.fly3d_dialog = Fly3DDialog(source, default_path=default_path, parent=self)
        return self.fly3d_dialog

    def _get_star_shrink_dialog(self):
        if self.star_shrink_dialog is None:
            self.star_shrink_dialog = StarShrinkDialog(self)
            self.star_shrink_dialog.finished.connect(self._finish_star_shrink_dialog)
        return self.star_shrink_dialog

    def _finish_star_shrink_dialog(self, result):
        context = self._star_shrink_context
        dialog = self.star_shrink_dialog
        if context is None or dialog is None:
            return

        dialog.preview_timer.stop()
        source_img = context["source_img"]
        source_pix = context["source_pix"]
        previous_processed_img = context["previous_processed_img"]
        previous_after_pix = context["previous_after_pix"]
        last_preview = context["last_preview"]

        if result == QDialog.Accepted:
            amount, selection = dialog.get_values()
            params = (amount, selection)
            if last_preview["params"] != params:
                context["preview_star_shrink"](amount, selection)
            else:
                self.magic_img = last_preview["magic_img"]
                self.processed_img = last_preview["processed_img"]
            self.undo_stack.append(source_img)
            self.redo_stack.clear()
            self.preview_override_img = None
            self.add_layer("stars", self.magic_img)
            self.log(f"Star Shrink applied. Amount: {amount:.2f}, Selection: {selection:.2f}")
        else:
            self.preview_override_img = None
            self.magic_img = source_img
            self.processed_img = previous_processed_img
            self.log("Star Shrink canceled.")

        self._end_dialog_compare(dialog)
        self.levels_window.levels_widget.set_image(self.magic_img)
        self.viewer.set_before(np_to_qpixmap(self.magic_img))
        if self.processed_img is not None:
            self.viewer.set_after(np_to_qpixmap(self.processed_img))
        elif previous_after_pix is not None:
            self.viewer.set_after(previous_after_pix)
        if self.histogram_window.isVisible() and self.processed_img is not None:
            self.histogram_window.set_image(self.processed_img)
        if result == QDialog.Accepted:
            self.apply_full_processing()
            self.add_thumbnail(
                f"Star Shrink (A:{amount:.2f}, S:{selection:.2f})",
                self.processed_img,
            )
        self.update_menu_actions()
        dialog.reset_defaults(emit_preview=False)
        self._star_shrink_context = None

    def _begin_dialog_compare(self, dialog, auto_end=True):
        if dialog not in self.compare_dialogs:
            self.compare_dialogs.append(dialog)
        if auto_end and not dialog.property("compare_auto_end_connected"):
            dialog.finished.connect(lambda _result, d=dialog: self._end_dialog_compare(d))
            dialog.setProperty("compare_auto_end_connected", True)
        self.viewer.set_compare_enabled(True)

    def _end_dialog_compare(self, dialog):
        if dialog in self.compare_dialogs:
            self.compare_dialogs.remove(dialog)
        self.viewer.set_compare_enabled(len(self.compare_dialogs) > 0)

    def log(self, message: str, level: str = "info"):
        if not hasattr(self, "history_items"):
            self.history_items = []
        self.history_items.append(message)
        self.history_items = self.history_items[-40:]
        if hasattr(self, "console_window") and self.console_window is not None:
            self.console_window.log(message, level)
        self.update_photoshop_panel()

    def update_photoshop_panel(self):
        if hasattr(self, "photoshop_menu_dialog") and self.photoshop_menu_dialog is not None:
            self.photoshop_menu_dialog.update_from_app(self)

    def add_layer(self, key, img=None, title=None):
        if key not in self.layer_order:
            self.layer_order.append(key)
        self.layer_visibility.setdefault(key, True)
        if title is not None:
            self.layer_titles[key] = title
        else:
            self.layer_titles.setdefault(key, key.replace("_", " ").title())
        if img is not None:
            self.layer_images[key] = img.copy()
        self.update_photoshop_panel()

    def get_active_layers(self):
        if not self.layer_order:
            return ["background"]
        return list(self.layer_order)

    def _next_layer_key(self, prefix: str) -> str:
        index = self.layer_counters.get(prefix, 0) + 1
        self.layer_counters[prefix] = index
        base_key = f"{prefix}_{index}"
        while base_key in self.layer_order or base_key in self.layer_visibility:
            index += 1
            self.layer_counters[prefix] = index
            base_key = f"{prefix}_{index}"
        return base_key

    def create_new_layer(self):
        if self.magic_img is None and self.original_img is None:
            self.log("New Layer skipped: no image loaded.", "warning")
            return

        source = self.get_effective_magic_img()
        if source is None:
            source = self.original_img

        key = self._next_layer_key("layer")
        title = f"Layer {self.layer_counters['layer']}"
        self.add_layer(key, np.zeros_like(source), title=title)
        self.apply_full_processing()
        self.add_thumbnail(f"New Layer: {title}", self.processed_img)
        self.log(f"New Layer created: {title}", "success")

    def create_adjustment_layer(self, kind: str):
        if self.magic_img is None and self.original_img is None:
            self.log(f"New Adjustment Layer > {kind.title()} skipped: no image loaded.", "warning")
            return

        prefix = "levels" if kind == "levels" else "curves"
        key = self._next_layer_key(prefix)
        index = self.layer_counters[prefix]
        title = f"{prefix.title()} {index}"
        self.add_layer(key, title=title)

        if kind == "levels":
            self.show_levels_window()
        elif kind == "curves":
            self.show_curves_window()

        self.log(f"New Adjustment Layer created: {title}", "success")

    def select_layer(self, key):
        if key not in self.layer_order:
            return
        self.selected_layer_key = key
        self.update_photoshop_panel()

    def can_delete_layer(self, key):
        return key in self.layer_order and key != "background"

    def delete_selected_layer(self):
        key = getattr(self, "selected_layer_key", None)
        if not self.can_delete_layer(key):
            self.log("Delete Layer skipped: select a removable layer first.", "warning")
            return

        title = self.layer_titles.get(key, key)
        if self.magic_img is not None:
            self.undo_stack.append(self.magic_img.copy())
            self.redo_stack.clear()

        if key in self.layer_order:
            index = self.layer_order.index(key)
            self.layer_order.remove(key)
        else:
            index = 0

        self.layer_images.pop(key, None)
        self.layer_titles.pop(key, None)
        self.layer_visibility.pop(key, None)
        self.layer_masks.pop(key, None)
        if not key.startswith(("blur", "stars", "curves", "levels", "plate_solve", "grid", "object_labels", "constellation")):
            self.layer_types.pop(key, None)

        if self.layer_order:
            self.selected_layer_key = self.layer_order[min(index, len(self.layer_order) - 1)]
        else:
            self.selected_layer_key = "background"
            self.layer_order.append("background")
            self.layer_visibility.setdefault("background", True)

        self.apply_full_processing()
        self.add_thumbnail(f"Delete Layer: {title}", self.processed_img)
        self.update_viewer_overlay()
        self.update_menu_actions()
        self.update_photoshop_panel()
        self.log(f"Layer deleted: {title}", "success")

    def is_paste_target_layer(self, key):
        return key in self.layer_order and key != "background"

    def set_layer_visibility(self, key, visible):
        self.layer_visibility[key] = bool(visible)
        self.apply_full_processing()
        self.add_thumbnail(f"Layer {'shown' if visible else 'hidden'}: {key}", self.processed_img)
        self.update_viewer_overlay()
        self.log(f"Layer {'shown' if visible else 'hidden'}: {key}", "info")

    def set_channel_visibility(self, key, visible):
        self.channel_visibility[key] = bool(visible)
        self.apply_full_processing()
        self.add_thumbnail(
            f"Channel {'shown' if visible else 'hidden'}: {key.upper()}",
            self.processed_img,
        )

    def _get_image_for_analysis(self):
        if isinstance(self.processed_img, np.ndarray):
            return self.processed_img
        if isinstance(self.magic_img, np.ndarray):
            return self.magic_img
        if isinstance(self.original_img, np.ndarray):
            return self.original_img
        return None

    def run_image_analysis(self, log_result: bool = True):
        image = self._get_image_for_analysis()
        if image is None:
            self.latest_image_analysis = {}
            self.analysis_dirty = True
            if log_result:
                self.log("Image analysis skipped: no image loaded.", "warning")
            return {}

        try:
            result = compute_image_analysis_metrics(image)
        except Exception as exc:
            if log_result:
                self.log(f"Image analysis failed: {exc}", "error")
            return {}

        return self.cache_image_analysis_result(result, log_result=log_result)

    def cache_image_analysis_result(self, result: dict, log_result: bool = True):
        if isinstance(result, dict):
            self.latest_image_analysis = result
        else:
            self.latest_image_analysis = {}
        self.analysis_dirty = False

        if log_result:
            stars = (self.latest_image_analysis.get("stars", {}) or {})
            luminance = (self.latest_image_analysis.get("luminance", {}) or {})
            backend = (self.latest_image_analysis.get("analysis_backend", {}) or {})
            star_count = int(stars.get("count") or 0)
            fwhm = stars.get("fwhm_px_median")
            snr = stars.get("snr_median")
            noise = float(luminance.get("background_sigma") or 0.0)
            stars_method = str(backend.get("stars_method") or "unknown")
            if fwhm is None:
                self.log(
                    f"Image analysis completed. Stars: {star_count}, FWHM: unavailable, SNR median: {float(snr or 0.0):.2f}, noise sigma: {noise:.2f}, method: {stars_method}.",
                    "success",
                )
            else:
                self.log(
                    f"Image analysis completed. Stars: {star_count}, FWHM median: {float(fwhm):.2f}px, SNR median: {float(snr or 0.0):.2f}, noise sigma: {noise:.2f}, method: {stars_method}.",
                    "success",
                )
        return self.latest_image_analysis

    def get_or_run_image_analysis(self):
        if self.analysis_dirty or not isinstance(self.latest_image_analysis, dict) or not self.latest_image_analysis:
            return self.run_image_analysis(log_result=False)
        return self.latest_image_analysis

    def get_effective_magic_img(self):
        if self.original_img is None:
            return None
        if not self.layer_visibility.get("background", True):
            return np.zeros_like(self.original_img)

        has_stack_layers = any(key in self.layer_order for key in ("blur", "stars"))
        img = self.original_img.copy() if has_stack_layers else self.magic_img.copy()
        for key in ("blur", "stars"):
            if key in self.layer_order and self.layer_visibility.get(key, True):
                layer_img = self.layer_images.get(key)
                if layer_img is not None:
                    layer_mask = self.layer_masks.get(key)
                    img = self._apply_mask_blend(img, layer_img, layer_mask) if layer_mask is not None else layer_img.copy()
        return img

    def apply_channel_visibility(self, img):
        if img is None or img.ndim != 3 or img.shape[2] < 3:
            return img

        out = img.copy()
        if not self.channel_visibility.get("b", True):
            out[:, :, 0] = 0
        if not self.channel_visibility.get("g", True):
            out[:, :, 1] = 0
        if not self.channel_visibility.get("r", True):
            out[:, :, 2] = 0
        return out

    def execute_console_command(self, command: str):
        normalized = command.strip()
        lower = normalized.lower()

        if lower in ("help", "?"):
            self.show_console_help()
            return
        if lower in ("open", "load"):
            self.load_image()
            return
        if lower.startswith("open ") or lower.startswith("load "):
            path = normalized.split(" ", 1)[1].strip().strip('"')
            if not path:
                self.log("Open skipped: path is empty.", "warning")
                return
            self.load_image_from_path(path)
            return
        if lower == "save":
            self.save_image()
            return
        if lower in ("save as", "save_as"):
            self.save_image_as()
            return
        if lower.startswith("save as ") or lower.startswith("save_as "):
            path = normalized.split(" ", 2)[2].strip().strip('"') if lower.startswith("save as ") else normalized.split(" ", 1)[1].strip().strip('"')
            self.save_image_to_path(path)
            return
        if lower in ("undo", "u"):
            self.undo()
            return
        if lower in ("redo", "r"):
            self.redo()
            return
        if lower in ("magic", "magic filter"):
            self.run_magic()
            return
        if lower in ("star", "star shrink", "star_shrink"):
            self.run_star_shrink()
            return
        if lower in ("plate solve", "platesolve", "plate solving", "solve plate"):
            self.run_plate_solve()
            return
        if lower in ("analyze", "analyse", "analysis", "analizuj"):
            self.run_image_analysis(log_result=True)
            return
        if lower in ("starnet", "starnet++", "run starnet", "run starnet++"):
            self.run_starnet()
            return
        if lower in ("deepsnr", "deep snr", "run deepsnr"):
            self.run_deepsnr()
            return
        if lower in ("3d fly", "fly", "3dfly", "fly3d"):
            self.run_3d_fly_filter()
            return
        if lower in ("blur", "gaussian blur", "gaussian"):
            self.apply_gaussian_blur_filter()
            return
        if lower in ("rotate", "rotation"):
            self.rotate_image_dialog()
            return
        if lower in ("crop", "kadruj", "kadrowanie"):
            self.crop_image_dialog()
            return
        if lower in ("levels", "level"):
            self.show_levels_window()
            return
        if lower == "menu":
            self.show_photoshop_menu()
            return
        if lower == "console":
            self.show_console_window()
            return
        if lower in ("curves", "curve", "lut", "curves lut", "curves (lut)"):
            self.show_curves_window()
            return
        if lower in ("curves reset", "curve reset", "lut reset"):
            self.curves_window.reset_curves()
            return
        if lower in ("histogram", "hist"):
            self.show_histogram_window()
            return
        if lower in ("correction", "correct", "camera raw"):
            self.show_correction_panels()
            return
        if lower in ("bn", "background neutralization", "background neutralisation"):
            self.apply_background_neutralization()
            return
        if lower in ("calibration", "color calibration"):
            self.show_color_calibration_dialog()
            return
        if lower in ("reset", "reset sliders"):
            self.reset_all_sliders()
            self.apply_full_processing()
            self.log("Sliders reset.", "success")
            return
        if lower in ("preferences", "preferencje", "prefs"):
            self.show_preferences_dialog()
            return
        if lower in ("dark", "dark toggle", "theme"):
            self.toggle_dark_mode()
            return
        if lower == "dark on":
            self.apply_dark_mode(True)
            self.apply_preferences(theme_name=self.theme_name)
            self.log("Dark Mode enabled.", "success")
            return
        if lower == "dark off":
            self.apply_dark_mode(False)
            self.apply_preferences(theme_name=self.theme_name)
            self.log("Dark Mode disabled.", "success")
            return
        if lower in ("models", "onnx models"):
            self.select_onnx_models()
            return
        if lower in ("exit", "quit"):
            self.close()
            return
        self.log(f"Unknown command: {command}. Type help to see available commands.", "error")

    def show_console_help(self):
        commands = [
            "help / ?                 show this command list",
            "clear / cls              clear console output",
            "open [path]              open an image or show file picker",
            "save                     save to the current save path",
            "save as [path]           save with a new path or show save dialog",
            "undo / redo              move through edit history",
            "magic                    run Magic Filter",
            "star shrink              open Star Shrink dialog",
            "starnet++                run StarNet++ star removal",
            "deepsnr                  run deepSNR external app",
            "3d fly                   render starless 3D fly-through clip",
            "analyze / analizuj       compute FWHM and image metrics",
            "blur                     open Gaussian Blur dialog",
            "rotate                   open Rotate dialog",
            "crop                     open Crop dialog",
            "levels                   open Levels window",
            "menu                     open Menu dialog",
            "curves / lut             open Curves (LUT) window",
            "curves reset             reset the Curves LUT",
            "histogram                open Histogram window",
            "correction               open correction panels",
            "bn                       run Background Neutralization",
            "calibration              open Background Neutralization dialog",
            "reset                    reset correction sliders",
            "preferences              open preferences dialog",
            "models                   select ONNX model paths",
            "exit / quit              close the app",
        ]
        self.log("Available commands:\n" + "\n".join(commands), "help")

    def apply_correction_from_dialog(self):
        self.cancel_params_preview()
        self.apply_full_processing()
        label = self._build_color_corrections_label()
        if label and self.processed_img is not None:
            self.add_thumbnail(label, self.processed_img)
        self.log("Correction applied.")

    def show_console_window(self):
        self.console_window.show()
        self.console_window.raise_()
        self.console_window.activateWindow()
        self.log("Console opened.")

    def show_plate_solving_dialog(self):
        if hasattr(self, "plate_solving_dialog") and self.plate_solving_dialog is not None:
            self.plate_solving_dialog.show()
            self.plate_solving_dialog.raise_()
            self.plate_solving_dialog.activateWindow()

    def open_script_editor(self):
        editor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Astro Script Editor.py")
        if not os.path.exists(editor_path):
            self.log(f"Script editor not found: {editor_path}", "error")
            return
        try:
            subprocess.Popen([sys.executable, editor_path], cwd=os.path.dirname(editor_path))
            self.log("Script editor opened.", "success")
        except Exception as e:
            self.log(f"Script editor failed: {e}", "error")

    def show_photoshop_menu(self):
        if hasattr(self, "photoshop_menu_dialog"):
            self.update_photoshop_panel()
            self.photoshop_menu_dialog.show()
            self.photoshop_menu_dialog.raise_()
            self.photoshop_menu_dialog.activateWindow()

    def show_preferences_dialog(self):
        if self.preferences_dialog is None:
            self.preferences_dialog = PreferencesDialog(self, parent=self)
        else:
            self.preferences_dialog.app = self
            self.preferences_dialog.combo_theme.setCurrentText(self.theme_name)
            self.preferences_dialog.combo_language.setCurrentText(self.language)
            self.preferences_dialog.spin_cpu_cores.setValue(int(self.processor_cores))
            self.preferences_dialog.combo_onnx_provider.setCurrentText(self.onnx_provider)
            if hasattr(self.preferences_dialog, "edit_gemini_api_pref"):
                self.preferences_dialog.edit_gemini_api_pref.setText(str(getattr(self, "gemini_api_key", "") or ""))
            if hasattr(self.preferences_dialog, "edit_deepsnr_args"):
                self.preferences_dialog.edit_deepsnr_args.setText(str(getattr(self, "deepsnr_args", "{input}") or "{input}"))
        self.preferences_dialog.refresh_select_paths()
        self.preferences_dialog.refresh_joystick_controls()
        self.preferences_dialog.show()
        self.preferences_dialog.raise_()
        self.preferences_dialog.activateWindow()
        self.log("Preferences opened.")

    def paste_image_into_selected_layer(self):
        target = getattr(self, "selected_layer_key", None)
        if not target:
            self.log("No active layer selected.", "warning")
            return
        if not self.is_paste_target_layer(target):
            self.log("This layer cannot receive pasted images.", "warning")
            return

        clipboard = QApplication.clipboard()
        image = clipboard.image()
        if image is None or image.isNull():
            pixmap = clipboard.pixmap()
            if pixmap is None or pixmap.isNull():
                self.log("Clipboard does not contain an image.", "warning")
                return
            image = pixmap.toImage()

        bgr = qimage_to_bgr_array(image)
        if bgr is None:
            self.log("Clipboard image could not be converted.", "error")
            return

        self.layer_images[target] = bgr.copy()
        self.layer_masks.setdefault(target, None)
        self.update_photoshop_panel()
        self.apply_full_processing()
        self.add_thumbnail(f"Paste Into Layer: {target}", self.processed_img)
        self.log(f"Image pasted into layer: {target}", "success")

    def show_ai_assistant_dialog(self):
        if self.ai_assistant_panel is not None:
            self.ai_assistant_panel.app = self
            self.ai_assistant_panel.show()
            self.ai_assistant_panel.input_text.setFocus()
            self.log("AI panel focused.")

    def show_levels_window(self):
        if self.magic_img is not None:
            self.levels_window.levels_widget.set_image(self.magic_img)
        self._begin_dialog_compare(self.levels_window)
        self.levels_window.show()
        self.levels_window.raise_()
        self.levels_window.activateWindow()
        self.log("Levels window opened.")

    def show_curves_window(self):
        if self.processed_img is not None:
            self.curves_window.set_image(self.processed_img)
        elif self.magic_img is not None:
            self.curves_window.set_image(self.magic_img)
        self.curves_window.show()
        self.curves_window.raise_()
        self.curves_window.activateWindow()
        self.log("Curves window opened.")

    def apply_gaussian_blur_filter(self):
        if self.magic_img is None:
            self.log("Gaussian Blur skipped: no image loaded.")
            return

        source_img = self.magic_img.copy()
        source_pix = np_to_qpixmap(source_img)
        previous_processed_img = None if self.processed_img is None else self.processed_img.copy()
        previous_after_pix = None if previous_processed_img is None else np_to_qpixmap(previous_processed_img)
        last_preview = {"sigma": None, "magic_img": None, "processed_img": None}

        def preview_blur(sigma):
            blurred = cv2.GaussianBlur(source_img, (0, 0), sigma)
            blur_mask = self.layer_masks.get("blur")
            self.magic_img = self._apply_mask_blend(source_img, blurred, blur_mask) if blur_mask is not None else blurred
            self.levels_window.levels_widget.set_image(self.magic_img)
            self.preview_override_img = self.magic_img.copy()
            self.viewer.set_before(source_pix)
            self.apply_full_processing()
            last_preview["sigma"] = sigma
            last_preview["magic_img"] = self.magic_img.copy()
            last_preview["processed_img"] = None if self.processed_img is None else self.processed_img.copy()
            
        dialog = self._get_blur_dialog()
        dialog.set_preview_callback(preview_blur)
        self._begin_dialog_compare(dialog)
        self.log("Gaussian Blur dialog opened.")
        preview_blur(dialog.get_value())
        if dialog.exec_() == QDialog.Accepted:
            dialog.preview_timer.stop()
            sigma = dialog.get_value()
            
            # Dodaj do historii (Undo)
            if last_preview["sigma"] != sigma:
                preview_blur(sigma)
            else:
                self.magic_img = last_preview["magic_img"]
                self.processed_img = last_preview["processed_img"]
            self.undo_stack.append(source_img)
            self.redo_stack.clear()
            
            # NaĹ‚ĂłĹĽ filtr
            # (0, 0) oznacza, ĹĽe ksize zostanie wyliczony z sigmy automatycznie
            blurred = cv2.GaussianBlur(source_img, (0, 0), sigma)
            blur_mask = self.layer_masks.get("blur")
            self.magic_img = self._apply_mask_blend(source_img, blurred, blur_mask) if blur_mask is not None else blurred
            self.levels_window.levels_widget.set_image(self.magic_img)
            self.preview_override_img = None
            self.add_layer("blur", self.magic_img)

            # Aktualizacja podglÄ…du
            pix_before = np_to_qpixmap(self.magic_img)
            self.viewer.set_before(pix_before)
            self.apply_full_processing()
            self.add_thumbnail(f"Gaussian Blur Ď={sigma:.1f}", self.processed_img)
            self.update_menu_actions()
            self.log(f"Gaussian Blur applied. Sigma: {sigma:.1f}")
            dialog.reset_defaults(emit_preview=False)
        else:
            dialog.preview_timer.stop()
            self.preview_override_img = None
            self.magic_img = source_img
            self.processed_img = previous_processed_img
            self.levels_window.levels_widget.set_image(self.magic_img)
            self.viewer.set_before(source_pix)
            if previous_after_pix is not None:
                self.viewer.set_after(previous_after_pix)
            else:
                self.apply_full_processing()
            self.update_menu_actions()
            self.log("Gaussian Blur canceled.")
            dialog.reset_defaults(emit_preview=False)
        self._end_dialog_compare(dialog)

    def _rotate_ndarray(self, img: np.ndarray, angle: float, is_mask: bool = False) -> np.ndarray:
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        border_value = 0 if img.ndim == 2 else (0, 0, 0)
        return cv2.warpAffine(
            img,
            matrix,
            (new_w, new_h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

    def _transform_ndarray(
        self,
        img: np.ndarray,
        angle: float,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        is_mask: bool = False,
    ) -> np.ndarray:
        out = img
        if flip_horizontal and flip_vertical:
            out = cv2.flip(out, -1)
        elif flip_horizontal:
            out = cv2.flip(out, 1)
        elif flip_vertical:
            out = cv2.flip(out, 0)
        if abs(float(angle)) >= 1e-6:
            out = self._rotate_ndarray(out, float(angle), is_mask=is_mask)
        return out

    def _clear_plate_solving_layers(self):
        for key in ["grid_overlay", "plate_solve_overlay", "object_labels", "constellation_overlay"]:
            self.layer_images.pop(key, None)
            self.layer_visibility[key] = False
        self.latest_plate_solve_result = None
        self.plate_solve_object_info = {}
        self._refresh_plate_solve_object_panel()
        self.lbl_plate_solve_status.setText(self.tr("plate_status_idle", "Plate solve status: idle"))
        self.lbl_plate_solve_details.setText(self.tr("plate_details_none", "Plate solve details: none"))

    def rotate_image_dialog(self):
        if self.magic_img is None:
            self.log("Rotate skipped: no image loaded.", "warning")
            return
        dialog = RotateDialog(self, source_img=self.magic_img, language=self.language)
        if dialog.exec_() != QDialog.Accepted:
            self.log("Rotate canceled.", "warning")
            return
        angle, flip_h, flip_v = dialog.get_transform()
        if abs(angle) < 1e-6 and not flip_h and not flip_v:
            self.log("Rotate skipped: no transform selected.", "warning")
            return

        base_h, base_w = self.magic_img.shape[:2]
        source_magic = self.magic_img.copy()
        self.undo_stack.append(source_magic)
        self.redo_stack.clear()

        for attr in ["original_img", "magic_img", "processed_img", "preview_override_img"]:
            val = getattr(self, attr, None)
            if isinstance(val, np.ndarray) and val.shape[:2] == (base_h, base_w):
                setattr(
                    self,
                    attr,
                    self._transform_ndarray(
                        val,
                        angle,
                        flip_horizontal=flip_h,
                        flip_vertical=flip_v,
                        is_mask=False,
                    ),
                )

        for key, value in list(self.layer_images.items()):
            if isinstance(value, np.ndarray) and value.shape[:2] == (base_h, base_w):
                self.layer_images[key] = self._transform_ndarray(
                    value,
                    angle,
                    flip_horizontal=flip_h,
                    flip_vertical=flip_v,
                    is_mask=False,
                )
        for key, value in list(self.layer_masks.items()):
            if isinstance(value, np.ndarray) and value.shape[:2] == (base_h, base_w):
                self.layer_masks[key] = self._transform_ndarray(
                    value,
                    angle,
                    flip_horizontal=flip_h,
                    flip_vertical=flip_v,
                    is_mask=True,
                )

        self._clear_plate_solving_layers()
        self.levels_window.levels_widget.set_image(self.magic_img)
        self.apply_full_processing()
        ops = []
        if flip_h:
            ops.append("Flip H")
        if flip_v:
            ops.append("Flip V")
        if abs(angle) >= 1e-6:
            ops.append(f"Rotate {angle:.1f}°")
        thumb_label = " + ".join(ops) if ops else "Rotate"
        self.add_thumbnail(thumb_label, self.processed_img)
        self.update_menu_actions()
        self.update_viewer_overlay()
        self.log(f"Transform applied: {thumb_label}", "success")

    def crop_image_dialog(self):
        if self.magic_img is None:
            self.log("Crop skipped: no image loaded.", "warning")
            return

        dialog = self._get_crop_dialog()
        if dialog.isVisible():
            dialog.raise_()
            dialog.activateWindow()
            self.log("Crop dialog is already open.", "info")
            return

        h, w = self.magic_img.shape[:2]
        dialog.set_image(self.magic_img)
        self._crop_overlay_active = True
        self._crop_pick_active = True
        self._crop_pick_anchor = None
        self._crop_pick_operation = None
        self._crop_pick_start_rect = None
        self._crop_pick_last_point = None
        self._set_crop_overlay_state(0, 0, w, h)
        self._set_crop_overlay_grid(dialog.combo_grid.currentIndex())
        self._sync_roi_pick_mode()
        self._begin_dialog_compare(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self.log("Crop edit mode enabled. Drag handles on image to resize crop.", "info")

    def _apply_crop_rect(self, x: int, y: int, cw: int, ch: int):
        if cw <= 0 or ch <= 0:
            self.log("Crop skipped: invalid rectangle.", "warning")
            return

        base_h, base_w = self.magic_img.shape[:2]
        source_magic = self.magic_img.copy()
        self.undo_stack.append(source_magic)
        self.redo_stack.clear()

        for attr in ["original_img", "magic_img", "processed_img", "preview_override_img"]:
            val = getattr(self, attr, None)
            if isinstance(val, np.ndarray) and val.shape[:2] == (base_h, base_w):
                setattr(self, attr, val[y:y + ch, x:x + cw].copy())

        for key, value in list(self.layer_images.items()):
            if isinstance(value, np.ndarray) and value.shape[:2] == (base_h, base_w):
                self.layer_images[key] = value[y:y + ch, x:x + cw].copy()
        for key, value in list(self.layer_masks.items()):
            if isinstance(value, np.ndarray) and value.shape[:2] == (base_h, base_w):
                self.layer_masks[key] = value[y:y + ch, x:x + cw].copy()

        self._clear_plate_solving_layers()
        self.levels_window.levels_widget.set_image(self.magic_img)
        self.apply_full_processing()
        self.add_thumbnail(f"Crop x:{x} y:{y} w:{cw} h:{ch}", self.processed_img)
        self.update_menu_actions()
        self.update_viewer_overlay()
        self.log(f"Crop applied: x={x}, y={y}, w={cw}, h={ch}", "success")

    def _on_crop_dialog_closed(self, result):
        dialog = getattr(self, "crop_dialog", None)
        if dialog is None:
            return

        x, y, cw, ch = dialog.get_rect()
        self._crop_pick_active = False
        self._crop_pick_anchor = None
        self._crop_pick_operation = None
        self._crop_pick_start_rect = None
        self._crop_pick_last_point = None
        self._sync_roi_pick_mode()
        self._crop_overlay_active = False
        self.update_viewer_overlay()
        self._end_dialog_compare(dialog)

        if result == QDialog.Accepted:
            self._apply_crop_rect(int(x), int(y), int(cw), int(ch))
        else:
            self.log("Crop canceled.", "warning")

        if self.magic_img is not None:
            dialog.set_image(self.magic_img)
        dialog.reset_defaults()

    def _on_crop_rect_from_dialog_changed(self, x: int, y: int, w: int, h: int):
        if self.magic_img is None:
            return
        img_h, img_w = self.magic_img.shape[:2]
        x = max(0, min(img_w - 1, int(x)))
        y = max(0, min(img_h - 1, int(y)))
        w = max(1, min(img_w - x, int(w)))
        h = max(1, min(img_h - y, int(h)))
        self._set_crop_overlay_state(x, y, w, h)

    def tr(self, key: str, default: str = None) -> str:
        return tr_text(getattr(self, "language", "pl"), key, default)

    def _translate_widgets_recursive(self, root_widget):
        if root_widget is None:
            return
        lang = getattr(self, "language", "pl")
        queue = [root_widget]
        seen = set()
        while queue:
            widget = queue.pop(0)
            if widget is None:
                continue
            widget_id = id(widget)
            if widget_id in seen:
                continue
            seen.add(widget_id)

            if hasattr(widget, "windowTitle") and hasattr(widget, "setWindowTitle"):
                title = widget.windowTitle()
                translated = translate_literal(lang, title)
                if translated != title:
                    widget.setWindowTitle(translated)

            if hasattr(widget, "text") and hasattr(widget, "setText"):
                try:
                    text = widget.text()
                    translated = translate_literal(lang, text)
                    if translated != text:
                        widget.setText(translated)
                except Exception:
                    pass

            if hasattr(widget, "actions"):
                try:
                    for action in widget.actions():
                        if action is None:
                            continue
                        text = action.text()
                        translated = translate_literal(lang, text)
                        if translated != text:
                            action.setText(translated)
                except Exception:
                    pass

            if hasattr(widget, "children"):
                try:
                    for child in widget.children():
                        queue.append(child)
                except Exception:
                    pass

    def _update_ui_language(self):
        self.setWindowTitle(self.tr("app_title", "Astro Ai Plus v1.0.0"))

        action_map = [
            ("action_open", "action_open", "Open"),
            ("action_save", "action_save", "Save"),
            ("action_save_as", "action_save_as", "Save As..."),
            ("action_exit", "action_exit", "Exit"),
            ("action_undo", "action_undo", "Undo"),
            ("action_redo", "action_redo", "Redo"),
            ("action_new_workspace", "action_new_workspace", "New Workspace..."),
            ("action_set_home_folder", "action_home", "Home Folder"),
            ("action_preferences", "action_preferences", "Preferences"),
            ("action_histogram", "action_histogram", "Histogram"),
            ("action_console", "action_console", "Console"),
            ("action_photoshop_menu", "action_menu", "Menu"),
            ("action_ai_assistant", "action_ai", "AI Assistant"),
            ("action_new_layer", "action_new_layer", "New Layer"),
            ("action_delete_layer", "action_delete_layer", "Delete Selected Layer"),
            ("action_adjustment_levels", "action_levels", "Levels"),
            ("action_adjustment_curves", "action_curves", "Curves (LUT)"),
            ("action_magic_filter", "action_magic", "Magic Filter"),
            ("action_star_shrink", "action_shrink", "Star Shrink"),
            ("action_plate_solve", "action_plate", "Plate Solving"),
            ("action_starnet", "action_starnet", "Run StarNet++"),
            ("action_deepsnr", "action_deepsnr", "Run deepSNR"),
            ("action_3d_fly", "action_3d_fly", "3D FLY Filter"),
            ("action_correction", "action_correction", "Correction"),
            ("action_color_calibration", "action_color_calibration", "Color Calibration"),
            ("action_levels", "action_levels", "Levels"),
            ("action_curves", "action_curves", "Curves (LUT)"),
            ("action_blur", "action_blur", "Gaussian Blur"),
            ("action_rotate", "action_rotate", "Rotate"),
            ("action_crop", "action_crop", "Crop"),
        ]
        for attr, key, default in action_map:
            action = getattr(self, attr, None)
            if action is not None:
                action.setText(self.tr(key, default))

        if hasattr(self, "menu_workspace") and self.menu_workspace is not None:
            self.menu_workspace.setTitle(self.tr("action_workspace", "Workspace"))

        button_map = [
            ("btn_open", "top_open", "Open"),
            ("btn_save", "top_save", "Save"),
            ("btn_save_as", "top_save_as", "Save As"),
            ("btn_undo", "top_undo", "Undo"),
            ("btn_redo", "top_redo", "Redo"),
            ("btn_workspace", "top_workspace", "Workspace"),
            ("btn_home", "top_home", "Home"),
            ("btn_prefs", "top_prefs", "Prefs"),
            ("btn_console_top", "top_console", "Console"),
            ("btn_menu_top", "top_menu", "Menu"),
            ("btn_hist_top", "top_histogram", "Histogram"),
            ("btn_ai_top", "top_ai", "AI"),
            ("btn_magic_top", "top_magic", "Magic"),
            ("btn_shrink_top", "top_shrink", "Shrink"),
            ("btn_plate_top", "top_plate", "Plate"),
            ("btn_starnet_top", "top_starnet", "StarNet"),
            ("btn_deepsnr_top", "top_deepsnr", "deepSNR"),
            ("btn_3d_fly_top", "top_3d_fly", "3D FLY"),
            ("btn_blur_top", "top_blur", "Blur"),
            ("btn_rotate_top", "top_rotate", "Rotate"),
            ("btn_crop_top", "top_crop", "Crop"),
            ("btn_corr_top", "top_correction", "Correction"),
            ("btn_calib_top", "top_color_calibration", "Calibration"),
            ("btn_levels_top", "top_levels", "Levels"),
            ("btn_curves_top", "top_curves", "Curves"),
            ("btn_new_layer_top", "top_new_layer", "New Layer"),
            ("btn_delete_layer_top", "top_delete_layer", "Delete Layer"),
            ("btn_adj_levels_top", "top_adj_levels", "Adj Levels"),
            ("btn_adj_curves_top", "top_adj_curves", "Adj Curves"),
            ("btn_exit_top", "top_exit", "Exit"),
        ]
        for attr, key, default in button_map:
            button = getattr(self, attr, None)
            if button is not None:
                translated = self.tr(key, default)
                button.setText("")
                button.setToolTip(translated)
                button.setStatusTip(translated)
                button.setAccessibleName(translated)

        if self.preferences_dialog is not None and hasattr(self.preferences_dialog, "refresh_joystick_controls"):
            self.preferences_dialog.refresh_joystick_controls()

        if hasattr(self, "lbl_plate_solve_status"):
            current = self.lbl_plate_solve_status.text().strip().lower()
            if not current or "idle" in current or "bezczynny" in current:
                self.lbl_plate_solve_status.setText(self.tr("plate_status_idle", "Plate solve status: idle"))
        if hasattr(self, "lbl_plate_solve_details"):
            current = self.lbl_plate_solve_details.text().strip().lower()
            if not current or "none" in current or "brak" in current:
                self.lbl_plate_solve_details.setText(self.tr("plate_details_none", "Plate solve details: none"))

        self._translate_widgets_recursive(self)
        for dialog_name in (
            "correction_dialog",
            "color_calibration_dialog",
            "histogram_window",
            "console_window",
            "photoshop_menu_dialog",
            "preferences_dialog",
            "plate_solving_dialog",
            "blur_dialog",
            "plate_solve_dialog",
            "starnet_dialog",
            "fly3d_dialog",
            "star_shrink_dialog",
        ):
            dialog = getattr(self, dialog_name, None)
            if dialog is not None:
                self._translate_widgets_recursive(dialog)

    def apply_dark_mode(self, enabled: bool):
        """Apply or remove dark mode stylesheet"""
        app = QApplication.instance()
        if app is None:
            return
        self.theme_name = "Fusion Dark" if enabled else "Light"
        self.dark_mode = enabled
        APP_PREFERENCES["theme_name"] = self.theme_name
        apply_theme_application(app, self.theme_name)

    def toggle_dark_mode(self):
        """Toggle dark mode on/off"""
        self.apply_dark_mode(not self.dark_mode)
        self.apply_preferences(theme_name=self.theme_name)
        self.log(f"Theme switched to {self.theme_name}.")

    def apply_preferences(self, theme_name=None, language=None, processor_cores=None, onnx_provider=None, gemini_api_key=None):
        if theme_name is not None:
            self.theme_name = theme_name
        if language is not None:
            self.language = language
        if processor_cores is not None:
            self.processor_cores = max(1, int(processor_cores))
        if onnx_provider is not None:
            self.onnx_provider = onnx_provider
        if gemini_api_key is not None:
            self.gemini_api_key = str(gemini_api_key)

        self.dark_mode = (self.theme_name or "").strip().lower() != "light"
        APP_PREFERENCES.update({
            "theme_name": self.theme_name,
            "language": self.language,
            "processor_cores": self.processor_cores,
            "onnx_provider": self.onnx_provider,
        })

        app = QApplication.instance()
        if app is not None:
            apply_theme_application(app, self.theme_name)
        self._update_ui_language()

        save_config(
            self.denoise_model_path,
            self.bg_removal_model_path,
            self.dark_mode,
            self.starnet_path,
            self.plate_solve_api_key,
            self.plate_solve_pixel_size_um,
            self.plate_solve_focal_length_mm,
            self.starnet_stride,
            self.gemini_api_key,
            self.gemini_model,
            theme_name=self.theme_name,
            language=self.language,
            processor_cores=self.processor_cores,
            onnx_provider=self.onnx_provider,
            workspaces=getattr(self, "workspaces", []),
        )

    def save_workspace_config(self):
        APP_PREFERENCES["workspaces"] = getattr(self, "workspaces", [])
        save_config(
            self.denoise_model_path,
            self.bg_removal_model_path,
            self.dark_mode,
            self.starnet_path,
            self.plate_solve_api_key,
            self.plate_solve_pixel_size_um,
            self.plate_solve_focal_length_mm,
            self.starnet_stride,
            self.gemini_api_key,
            self.gemini_model,
            theme_name=self.theme_name,
            language=self.language,
            processor_cores=self.processor_cores,
            onnx_provider=self.onnx_provider,
            workspaces=getattr(self, "workspaces", []),
        )

    def show_new_workspace_dialog(self):
        dialog = NewWorkspaceDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            self.log("New workspace canceled.")
            return
        workspace = dialog.get_workspace()
        self.add_workspace(workspace)
        self.restore_workspace_geometry(workspace)

    def add_workspace(self, workspace):
        if not hasattr(self, "workspaces") or not isinstance(self.workspaces, list):
            self.workspaces = []
        name = workspace.get("name", "Workspace")
        self.workspaces = [item for item in self.workspaces if item.get("name") != name]
        self.workspaces.append(workspace)
        self.save_workspace_config()
        self.refresh_workspace_menu()
        self.log(f"Workspace saved: {name}", "success")

    def refresh_workspace_menu(self):
        if not hasattr(self, "menu_workspace"):
            return
        self.menu_workspace.clear()
        self.menu_workspace.addAction(self.action_new_workspace)
        workspaces = getattr(self, "workspaces", [])
        if not workspaces:
            return
        self.menu_workspace.addSeparator()
        for workspace in workspaces:
            name = str(workspace.get("name", "Workspace"))
            action = QAction(name, self)
            action.triggered.connect(lambda _checked=False, saved_workspace=workspace: self.apply_workspace(saved_workspace))
            self.menu_workspace.addAction(action)

    def apply_workspace(self, workspace):
        opened = []
        for dialog_key in workspace.get("dialogs", []):
            opened_dialog = self.open_workspace_dialog(dialog_key)
            if opened_dialog is not None:
                opened.append(opened_dialog)
        if workspace.get("geometry"):
            self.restore_workspace_geometry(workspace)
        else:
            self.arrange_workspace_dialogs(opened, workspace.get("layout", "Cascade"))
        self.log(f"Workspace opened: {workspace.get('name', 'Workspace')}", "success")

    def open_workspace_dialog(self, dialog_key):
        openers = {
            "histogram": lambda: self.histogram_window,
            "console": lambda: self.console_window,
            "menu": lambda: self.photoshop_menu_dialog,
            "color_calibration": lambda: self._get_color_calibration_dialog(),
        }
        opener = openers.get(dialog_key)
        if opener is None:
            return None
        dialog = opener()
        if dialog_key == "histogram" and self.processed_img is not None:
            self.histogram_window.set_image(self.apply_channel_visibility(self.processed_img))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def get_workspace_dialog(self, dialog_key):
        dialogs = {
            "histogram": self.histogram_window,
            "console": self.console_window,
            "menu": self.photoshop_menu_dialog,
            "color_calibration": self.color_calibration_dialog,
        }
        return dialogs.get(dialog_key)

    def get_workspace_open_dialogs(self, dialog_keys):
        dialogs = []
        for dialog_key in dialog_keys:
            dialog = self.get_workspace_dialog(dialog_key)
            if dialog is not None and dialog.isVisible():
                dialogs.append(dialog)
        return dialogs

    def close_workspace_dialog(self, dialog_key):
        dialog = self.get_workspace_dialog(dialog_key)
        if dialog is not None:
            dialog.hide()
            self.log(f"Workspace dialog closed: {dialog_key}")

    def capture_workspace_geometry(self, dialog_keys):
        geometry = {}
        for dialog_key in dialog_keys:
            dialog = self.get_workspace_dialog(dialog_key)
            if dialog is None or not dialog.isVisible():
                continue
            rect = dialog.geometry()
            geometry[dialog_key] = {
                "x": int(rect.x()),
                "y": int(rect.y()),
                "width": int(rect.width()),
                "height": int(rect.height()),
            }
        return geometry

    def restore_workspace_geometry(self, workspace):
        geometry = workspace.get("geometry", {})
        if not isinstance(geometry, dict):
            return
        for dialog_key in workspace.get("dialogs", []):
            dialog = self.get_workspace_dialog(dialog_key)
            rect = geometry.get(dialog_key)
            if dialog is None or not isinstance(rect, dict):
                continue
            try:
                x = int(rect.get("x", dialog.x()))
                y = int(rect.get("y", dialog.y()))
                width = max(260, int(rect.get("width", dialog.width())))
                height = max(180, int(rect.get("height", dialog.height())))
            except (TypeError, ValueError):
                continue
            dialog.setGeometry(x, y, width, height)

    def arrange_workspace_dialogs(self, dialogs, layout_name):
        visible_dialogs = [dialog for dialog in dialogs if dialog is not None]
        if not visible_dialogs:
            return

        base = self.geometry()
        margin = 24
        title_offset = 32
        layout_name = (layout_name or "Cascade").lower()

        if layout_name == "tile":
            columns = max(1, math.ceil(math.sqrt(len(visible_dialogs))))
            rows = max(1, math.ceil(len(visible_dialogs) / columns))
            width = max(320, (base.width() - margin * (columns + 1)) // columns)
            height = max(260, (base.height() - margin * (rows + 1)) // rows)
            for index, dialog in enumerate(visible_dialogs):
                row = index // columns
                col = index % columns
                dialog.setGeometry(base.x() + margin + col * (width + margin), base.y() + title_offset + margin + row * (height + margin), width, height)
            return

        if layout_name in {"left stack", "right stack"}:
            width = max(360, min(520, base.width() // 3))
            height = max(260, min(420, base.height() // 3))
            x = base.x() + margin if layout_name == "left stack" else base.right() - width - margin
            for index, dialog in enumerate(visible_dialogs):
                y = base.y() + title_offset + margin + index * 34
                dialog.setGeometry(x, y, width, height)
            return

        width = max(380, min(560, base.width() // 3))
        height = max(280, min(460, base.height() // 3))
        for index, dialog in enumerate(visible_dialogs):
            offset = index * 34
            dialog.setGeometry(base.x() + margin + offset, base.y() + title_offset + margin + offset, width, height)

    def show_correction_panels(self):
        self._begin_dialog_compare(self.correction_dialog)
        self.correction_dialog.show()
        self.correction_dialog.raise_()
        self.correction_dialog.activateWindow()
        self.log("Correction window opened.")

    def show_color_calibration_dialog(self):
        dialog = self._get_color_calibration_dialog()
        base_img = self._get_bn_source_image()
        if isinstance(base_img, np.ndarray) and base_img.ndim >= 2:
            h, w = base_img.shape[:2]
            dialog.set_image_shape(h, w)
        dialog.set_manual_roi_ready(False)
        self._stop_bn_roi_pick()
        self._bn_overlay_active = False
        self.update_viewer_overlay()
        self._begin_dialog_compare(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self.log("Color calibration dialog opened.")

    def _get_bn_source_image(self):
        base_img = self.preview_override_img if getattr(self, "preview_override_img", None) is not None else self.get_effective_magic_img()
        if base_img is None and self.original_img is not None:
            base_img = self.original_img
        return base_img

    def _normalize_roi(self, y1: int, y2: int, x1: int, x2: int, height: int, width: int):
        width = max(1, int(width))
        height = max(1, int(height))
        x1 = max(0, min(width - 1, int(x1)))
        x2 = max(1, min(width, int(x2)))
        y1 = max(0, min(height - 1, int(y1)))
        y2 = max(1, min(height, int(y2)))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        return (y1, y2, x1, x2)

    def _sync_roi_pick_mode(self):
        self.viewer.view.set_roi_pick_mode(bool(self._bn_pick_active or self._crop_pick_active))

    def _cursor_for_roi_hit(self, hit: str):
        if hit in ("top_left", "bottom_right"):
            return Qt.SizeFDiagCursor
        if hit in ("top_right", "bottom_left"):
            return Qt.SizeBDiagCursor
        if hit in ("left", "right"):
            return Qt.SizeHorCursor
        if hit in ("top", "bottom"):
            return Qt.SizeVerCursor
        if hit == "inside":
            return Qt.OpenHandCursor
        return Qt.CrossCursor

    def _on_roi_hover_changed(self, point):
        view = self.viewer.view
        if not view.roi_pick_mode:
            return
        if point is None:
            view.setCursor(Qt.CrossCursor)
            return

        px, py = int(point[0]), int(point[1])
        if self._crop_pick_active:
            view.setCursor(self._cursor_for_roi_hit(self._crop_hit_test(px, py)))
            return
        if self._bn_pick_active:
            view.setCursor(self._cursor_for_roi_hit(self._bn_hit_test(px, py)))
            return
        view.setCursor(Qt.CrossCursor)

    def _on_roi_drag_started(self, x: int, y: int):
        if self._crop_pick_active:
            self._on_crop_roi_drag_started(x, y)
            return
        self._on_bn_roi_drag_started(x, y)

    def _on_roi_drag_updated(self, x: int, y: int):
        if self._crop_pick_active:
            self._on_crop_roi_drag_updated(x, y)
            return
        self._on_bn_roi_drag_updated(x, y)

    def _on_roi_drag_finished(self, x: int, y: int):
        if self._crop_pick_active:
            self._on_crop_roi_drag_finished(x, y)
            return
        self._on_bn_roi_drag_finished(x, y)

    def _normalize_crop_rect(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int):
        img_w = max(1, int(img_w))
        img_h = max(1, int(img_h))
        x = max(0, min(img_w - 1, int(x)))
        y = max(0, min(img_h - 1, int(y)))
        w = max(1, min(img_w - x, int(w)))
        h = max(1, min(img_h - y, int(h)))
        return (x, y, w, h)

    def _crop_hit_test(self, x: int, y: int, threshold: int = 10):
        if not self._crop_overlay_active:
            return "outside"
        rx, ry, rw, rh = (int(v) for v in self._crop_overlay_rect)
        left = rx
        top = ry
        right = rx + max(1, rw)
        bottom = ry + max(1, rh)

        near_left = abs(x - left) <= threshold
        near_right = abs(x - right) <= threshold
        near_top = abs(y - top) <= threshold
        near_bottom = abs(y - bottom) <= threshold
        inside_x = left <= x <= right
        inside_y = top <= y <= bottom

        if near_left and near_top:
            return "top_left"
        if near_right and near_top:
            return "top_right"
        if near_left and near_bottom:
            return "bottom_left"
        if near_right and near_bottom:
            return "bottom_right"
        if near_left and inside_y:
            return "left"
        if near_right and inside_y:
            return "right"
        if near_top and inside_x:
            return "top"
        if near_bottom and inside_x:
            return "bottom"
        if inside_x and inside_y:
            return "inside"
        return "outside"

    def _sync_crop_overlay_to_dialog(self):
        dialog = getattr(self, "crop_dialog", None)
        if dialog is None:
            return
        x, y, w, h = (int(v) for v in self._crop_overlay_rect)
        dialog.set_rect_from_overlay(x, y, w, h)
        nx, ny, nw, nh = dialog.get_rect()
        self._set_crop_overlay_state(nx, ny, nw, nh)

    def _set_crop_overlay_rect_interactive(self, x: int, y: int, w: int, h: int):
        if self.magic_img is None:
            return
        img_h, img_w = self.magic_img.shape[:2]
        x, y, w, h = self._normalize_crop_rect(x, y, w, h, img_w, img_h)
        self._crop_overlay_rect = (int(x), int(y), int(w), int(h))
        self._sync_crop_overlay_to_dialog()

    def _on_crop_roi_drag_started(self, x: int, y: int):
        if not self._crop_pick_active:
            return
        if self.magic_img is None:
            return
        px = int(x)
        py = int(y)
        hit = self._crop_hit_test(px, py)
        self._crop_pick_anchor = (px, py)
        self._crop_pick_start_rect = tuple(int(v) for v in self._crop_overlay_rect)
        self._crop_pick_last_point = (px, py)

        if hit == "inside":
            self._crop_pick_operation = "move"
            self.viewer.view.setCursor(Qt.ClosedHandCursor)
            return
        if hit != "outside":
            self._crop_pick_operation = f"resize:{hit}"
            self.viewer.view.setCursor(self._cursor_for_roi_hit(hit))
            return

        self._crop_pick_operation = "draw"
        self.viewer.view.setCursor(Qt.CrossCursor)
        self._set_crop_overlay_rect_interactive(px, py, 1, 1)

    def _on_crop_roi_drag_updated(self, x: int, y: int):
        if not self._crop_pick_active or self._crop_pick_anchor is None or not self._crop_pick_operation:
            return
        if self.magic_img is None:
            return
        img_h, img_w = self.magic_img.shape[:2]
        px = int(x)
        py = int(y)
        ax, ay = self._crop_pick_anchor
        op = self._crop_pick_operation

        if op == "draw":
            left = min(ax, px)
            top = min(ay, py)
            right = max(ax, px)
            bottom = max(ay, py)
            self._set_crop_overlay_rect_interactive(left, top, right - left + 1, bottom - top + 1)
            self._crop_pick_last_point = (px, py)
            return

        if op == "move":
            if self._crop_pick_last_point is None:
                self._crop_pick_last_point = (px, py)
            lx, ly = self._crop_pick_last_point
            dx = px - lx
            dy = py - ly
            rx, ry, rw, rh = (int(v) for v in self._crop_overlay_rect)
            new_x = max(0, min(img_w - max(1, rw), rx + dx))
            new_y = max(0, min(img_h - max(1, rh), ry + dy))
            self._set_crop_overlay_rect_interactive(new_x, new_y, rw, rh)
            self._crop_pick_last_point = (px, py)
            return

        if op.startswith("resize:"):
            handle = op.split(":", 1)[1]
            sx, sy, sw, sh = self._crop_pick_start_rect if self._crop_pick_start_rect is not None else self._crop_overlay_rect
            left = int(sx)
            top = int(sy)
            right = int(sx + max(1, sw))
            bottom = int(sy + max(1, sh))

            if "left" in handle:
                left = px
            if "right" in handle:
                right = px
            if "top" in handle:
                top = py
            if "bottom" in handle:
                bottom = py

            nx = min(left, right)
            ny = min(top, bottom)
            nw = max(1, abs(right - left))
            nh = max(1, abs(bottom - top))
            self._set_crop_overlay_rect_interactive(nx, ny, nw, nh)
            self._crop_pick_last_point = (px, py)
            return

    def _on_crop_roi_drag_finished(self, x: int, y: int):
        if not self._crop_pick_active or self._crop_pick_anchor is None or not self._crop_pick_operation:
            return

        self._on_crop_roi_drag_updated(int(x), int(y))
        rx, ry, rw, rh = (int(v) for v in self._crop_overlay_rect)
        self.log(f"Crop ROI updated: x={rx}, y={ry}, w={rw}, h={rh}.", "success")
        self._crop_pick_anchor = None
        self._crop_pick_operation = None
        self._crop_pick_start_rect = None
        self._crop_pick_last_point = None
        self._on_roi_hover_changed((int(x), int(y)))

    def start_bn_roi_pick(self):
        base_img = self._get_bn_source_image()
        if base_img is None:
            self.log("Background Neutralization ROI selection skipped: no image loaded.", "warning")
            return

        if self._bn_pick_active:
            self._stop_bn_roi_pick()
            self.log("BN ROI edit mode disabled.", "info")
            return

        dialog = self._get_color_calibration_dialog()
        self._bn_pick_active = True
        self._bn_pick_anchor = None
        self._bn_pick_operation = None
        self._bn_pick_start_rect = None
        self._bn_pick_last_point = None
        dialog.set_selection_active(True)
        self._sync_roi_pick_mode()
        self.log("BN ROI edit mode enabled. Draw, move, or resize ROI directly on image.", "info")

    def _stop_bn_roi_pick(self):
        self._bn_pick_active = False
        self._bn_pick_anchor = None
        self._bn_pick_operation = None
        self._bn_pick_start_rect = None
        self._bn_pick_last_point = None
        self._sync_roi_pick_mode()
        dialog = getattr(self, "color_calibration_dialog", None)
        if dialog is not None:
            dialog.set_selection_active(False)

    def _on_color_calibration_dialog_closed(self, _result):
        self._stop_bn_roi_pick()
        self._bn_overlay_active = False
        self.update_viewer_overlay()

    def _set_bn_overlay_from_roi(self, y1: int, y2: int, x1: int, x2: int):
        self._bn_overlay_rect = (
            int(x1),
            int(y1),
            max(1, int(x2) - int(x1)),
            max(1, int(y2) - int(y1)),
        )
        self._bn_overlay_active = True
        self.update_viewer_overlay()

    def _bn_overlay_rect_to_roi(self):
        x, y, w, h = (int(v) for v in self._bn_overlay_rect)
        return (y, y + max(1, h), x, x + max(1, w))

    def _bn_hit_test(self, x: int, y: int, threshold: int = 10):
        if not self._bn_overlay_active:
            return "outside"
        rx, ry, rw, rh = (int(v) for v in self._bn_overlay_rect)
        left = rx
        top = ry
        right = rx + max(1, rw)
        bottom = ry + max(1, rh)

        near_left = abs(x - left) <= threshold
        near_right = abs(x - right) <= threshold
        near_top = abs(y - top) <= threshold
        near_bottom = abs(y - bottom) <= threshold
        inside_x = left <= x <= right
        inside_y = top <= y <= bottom

        if near_left and near_top:
            return "top_left"
        if near_right and near_top:
            return "top_right"
        if near_left and near_bottom:
            return "bottom_left"
        if near_right and near_bottom:
            return "bottom_right"
        if near_left and inside_y:
            return "left"
        if near_right and inside_y:
            return "right"
        if near_top and inside_x:
            return "top"
        if near_bottom and inside_x:
            return "bottom"
        if inside_x and inside_y:
            return "inside"
        return "outside"

    def _sync_bn_roi_to_dialog(self, y1: int, y2: int, x1: int, x2: int):
        dialog = self._get_color_calibration_dialog()
        dialog.set_roi_values(y1, y2, x1, x2)
        dialog.set_manual_roi_ready(True)

    def _on_bn_roi_from_dialog_changed(self, y1: int, y2: int, x1: int, x2: int):
        base_img = self._get_bn_source_image()
        if base_img is None:
            return
        h, w = base_img.shape[:2]
        y1, y2, x1, x2 = self._normalize_roi(y1, y2, x1, x2, h, w)
        self._set_bn_overlay_from_roi(y1, y2, x1, x2)

    def _on_bn_roi_drag_started(self, x: int, y: int):
        if not self._bn_pick_active:
            return
        px = int(x)
        py = int(y)
        hit = self._bn_hit_test(px, py)
        self._bn_pick_anchor = (px, py)
        self._bn_pick_start_rect = tuple(int(v) for v in self._bn_overlay_rect)
        self._bn_pick_last_point = (px, py)
        if hit == "inside":
            self._bn_pick_operation = "move"
            self.viewer.view.setCursor(Qt.ClosedHandCursor)
            return
        if hit != "outside":
            self._bn_pick_operation = f"resize:{hit}"
            self.viewer.view.setCursor(self._cursor_for_roi_hit(hit))
            return
        self._bn_pick_operation = "draw"
        self.viewer.view.setCursor(Qt.CrossCursor)
        self._set_bn_overlay_from_roi(py, py + 1, px, px + 1)

    def _on_bn_roi_drag_updated(self, x: int, y: int):
        if not self._bn_pick_active or self._bn_pick_anchor is None or not self._bn_pick_operation:
            return
        px = int(x)
        py = int(y)
        base_img = self._get_bn_source_image()
        if base_img is None:
            return
        h, w = base_img.shape[:2]
        ax, ay = self._bn_pick_anchor
        op = self._bn_pick_operation

        if op == "draw":
            y1, y2, x1, x2 = self._normalize_roi(min(ay, py), max(ay, py) + 1, min(ax, px), max(ax, px) + 1, h, w)
            self._set_bn_overlay_from_roi(y1, y2, x1, x2)
            self._bn_pick_last_point = (px, py)
            return

        if op == "move":
            if self._bn_pick_last_point is None:
                self._bn_pick_last_point = (px, py)
            lx, ly = self._bn_pick_last_point
            dx = px - lx
            dy = py - ly
            rx, ry, rw, rh = (int(v) for v in self._bn_overlay_rect)
            new_x = max(0, min(w - max(1, rw), rx + dx))
            new_y = max(0, min(h - max(1, rh), ry + dy))
            self._bn_overlay_rect = (int(new_x), int(new_y), int(max(1, rw)), int(max(1, rh)))
            self._bn_overlay_active = True
            self.update_viewer_overlay()
            self._bn_pick_last_point = (px, py)
            return

        if op.startswith("resize:"):
            handle = op.split(":", 1)[1]
            sx, sy, sw, sh = self._bn_pick_start_rect if self._bn_pick_start_rect is not None else self._bn_overlay_rect
            left = int(sx)
            top = int(sy)
            right = int(sx + max(1, sw))
            bottom = int(sy + max(1, sh))

            if "left" in handle:
                left = px
            if "right" in handle:
                right = px
            if "top" in handle:
                top = py
            if "bottom" in handle:
                bottom = py

            y1, y2, x1, x2 = self._normalize_roi(min(top, bottom), max(top, bottom), min(left, right), max(left, right), h, w)
            self._set_bn_overlay_from_roi(y1, y2, x1, x2)
            self._bn_pick_last_point = (px, py)
            return

    def _on_bn_roi_drag_finished(self, x: int, y: int):
        if not self._bn_pick_active or self._bn_pick_anchor is None or not self._bn_pick_operation:
            return

        base_img = self._get_bn_source_image()
        if base_img is None:
            self._stop_bn_roi_pick()
            return

        self._on_bn_roi_drag_updated(int(x), int(y))
        y1, y2, x1, x2 = self._bn_overlay_rect_to_roi()
        self._sync_bn_roi_to_dialog(y1, y2, x1, x2)
        self.log(f"BN ROI updated: ({y1},{y2},{x1},{x2}).", "success")
        self._bn_pick_anchor = None
        self._bn_pick_operation = None
        self._bn_pick_start_rect = None
        self._bn_pick_last_point = None
        self._on_roi_hover_changed((int(x), int(y)))

    def _available_arduino_ports(self):
        ports = []
        seen_devices = set()

        if SERIAL_AVAILABLE and list_ports is not None:
            for port in list_ports.comports():
                device = port.device
                label = f"{device}"
                if port.description:
                    label += f" - {port.description}"
                if device not in seen_devices:
                    ports.append((label, device))
                    seen_devices.add(device)

        if not ports and QT_SERIAL_AVAILABLE and QSerialPortInfo is not None:
            for port in QSerialPortInfo.availablePorts():
                device = port.portName()
                label = device
                desc = port.description()
                if desc:
                    label += f" - {desc}"
                if device not in seen_devices:
                    ports.append((label, device))
                    seen_devices.add(device)

        return ports

    def connect_arduino_joystick(self):
        if self.arduino_joystick_worker is not None:
            self.disconnect_arduino_joystick()
            return

        ports = self._available_arduino_ports()
        if not ports:
            QMessageBox.warning(
                self,
                "Arduino joystick",
                "Nie znaleziono portĂłw szeregowych. PodĹ‚Ä…cz Arduino z HW-504 albo zainstaluj pyserial.",
            )
            return

        if len(ports) == 1:
            device = ports[0][1]
        else:
            items = [label for label, _device in ports]
            selected, ok = QInputDialog.getItem(
                self,
                "Arduino joystick",
                "Wybierz port Arduino:",
                items,
                0,
                False,
            )
            if not ok or not selected:
                return

            device = next((dev for label, dev in ports if label == selected), None)
        if not device:
            return

        self.arduino_joystick_worker = ArduinoJoystickWorker(device)
        self.arduino_joystick_worker.pan_signal.connect(self.on_joystick_pan)
        self.arduino_joystick_worker.status_signal.connect(self.on_joystick_status)
        self.arduino_joystick_worker.error_signal.connect(self.on_joystick_error)
        self.arduino_joystick_worker.finished.connect(self.on_joystick_finished)
        self.arduino_joystick_worker.start()
        self._update_joystick_widgets(
            button_text=self.tr("disconnect_joystick", "Disconnect Joystick"),
            status_text=f"{self.tr('joystick_connecting', 'Arduino joystick: connecting')} {device}...",
        )

    def _update_joystick_widgets(self, button_text=None, status_text=None):
        if hasattr(self, "btn_arduino_joystick") and button_text is not None:
            self.btn_arduino_joystick.setText(button_text)
        if hasattr(self, "lbl_arduino_status") and status_text is not None:
            self.lbl_arduino_status.setText(status_text)

        prefs = getattr(self, "preferences_dialog", None)
        if prefs is not None:
            if hasattr(prefs, "btn_arduino_joystick_pref") and button_text is not None:
                prefs.btn_arduino_joystick_pref.setText(button_text)
            if hasattr(prefs, "lbl_arduino_status_pref") and status_text is not None:
                prefs.lbl_arduino_status_pref.setText(status_text)

    def disconnect_arduino_joystick(self):
        worker = self.arduino_joystick_worker
        if worker is None:
            return
        worker.stop()
        worker.wait(1000)
        self.arduino_joystick_worker = None
        self._update_joystick_widgets(
            button_text=self.tr("connect_joystick", "Connect Joystick"),
            status_text=self.tr("joystick_disconnected", "Arduino joystick: disconnected"),
        )

    def on_joystick_pan(self, dx: int, dy: int):
        if hasattr(self, "viewer") and hasattr(self.viewer, "view"):
            self.viewer.view.pan_by(dx, dy)

    def on_joystick_status(self, message: str):
        self._update_joystick_widgets(status_text=f"Arduino joystick: {message}")
        self.log(message, "info")

    def on_joystick_error(self, message: str):
        self._update_joystick_widgets(status_text="Arduino joystick: error")
        self.log(message, "error")

    def on_joystick_finished(self):
        if self.arduino_joystick_worker is not None:
            self.arduino_joystick_worker = None
        self._update_joystick_widgets(
            button_text=self.tr("connect_joystick", "Connect Joystick"),
            status_text=self.tr("joystick_disconnected", "Arduino joystick: disconnected"),
        )

    def _stop_qthread(self, thread_obj, name: str = "worker"):
        if thread_obj is None:
            return
        try:
            if thread_obj.isRunning():
                thread_obj.requestInterruption()
                if not thread_obj.wait(1500):
                    self.log(f"Stopping {name}: forcing thread termination.", "warning")
                    thread_obj.terminate()
                    thread_obj.wait(1000)
        except Exception as exc:
            self.log(f"Stopping {name} failed: {exc}", "warning")

    def _shutdown_background_workers(self):
        if self.ai_assistant_panel is not None and hasattr(self.ai_assistant_panel, "shutdown_workers"):
            try:
                self.ai_assistant_panel.shutdown_workers()
            except Exception as exc:
                self.log(f"Assistant worker cleanup failed: {exc}", "warning")

        self._stop_qthread(getattr(self, "worker", None), "magic worker")
        self._stop_qthread(getattr(self, "starnet_worker", None), "starnet worker")
        self._stop_qthread(getattr(self, "fly3d_worker", None), "3d fly worker")
        self._stop_qthread(getattr(self, "plate_solve_worker", None), "plate solve worker")

    def _current_topbar_button_order(self) -> list:
        layout = getattr(self, "top_actions_layout", None)
        if layout is None:
            return []

        order = []
        for idx in range(layout.count()):
            item = layout.itemAt(idx)
            widget = item.widget() if item is not None else None
            if not isinstance(widget, DraggableTopActionButton):
                continue
            name = str(widget.objectName() or "").strip()
            if name:
                order.append(name)
        return order

    def _apply_saved_topbar_button_order(self):
        layout = getattr(self, "top_actions_layout", None)
        if layout is None:
            return

        saved_order = []
        seen = set()
        for raw_name in getattr(self, "topbar_button_order", []) or []:
            name = str(raw_name or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            saved_order.append(name)
        if not saved_order:
            return

        buttons_by_name = {}
        for idx in range(layout.count()):
            item = layout.itemAt(idx)
            widget = item.widget() if item is not None else None
            if isinstance(widget, DraggableTopActionButton):
                buttons_by_name[str(widget.objectName() or "").strip()] = widget

        insert_index = ReorderableTopActionsBar._index_before_stretch(layout)
        for button_name in saved_order:
            button = buttons_by_name.get(button_name)
            if button is None:
                continue
            layout.removeWidget(button)
            layout.insertWidget(insert_index, button)
            insert_index += 1

    def _ensure_topbar_single_trailing_stretch(self):
        layout = getattr(self, "top_actions_layout", None)
        if layout is None:
            return

        for idx in reversed(range(layout.count())):
            item = layout.itemAt(idx)
            spacer = item.spacerItem() if item is not None else None
            if spacer is None:
                continue
            if bool(spacer.expandingDirections() & Qt.Horizontal):
                layout.takeAt(idx)

        layout.addStretch(1)

    def _save_topbar_button_order(self):
        order = self._current_topbar_button_order()
        if not order:
            return

        self.topbar_button_order = list(order)
        APP_PREFERENCES["topbar_button_order"] = list(order)
        save_config(
            self.denoise_model_path,
            self.bg_removal_model_path,
            self.dark_mode,
            self.starnet_path,
            self.plate_solve_api_key,
            self.plate_solve_pixel_size_um,
            self.plate_solve_focal_length_mm,
            self.starnet_stride,
            self.gemini_api_key,
            self.gemini_model,
            theme_name=self.theme_name,
            language=self.language,
            processor_cores=self.processor_cores,
            onnx_provider=self.onnx_provider,
            workspaces=getattr(self, "workspaces", []),
            home_folder=getattr(self, "home_folder", ""),
            topbar_button_order=order,
        )

    def closeEvent(self, event):
        self._save_topbar_button_order()
        self._shutdown_background_workers()
        self.disconnect_arduino_joystick()
        super().closeEvent(event)

    def init_ui(self):
        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # GĂłrny pasek akcji (zamiast klasycznego MenuBar)
        top_actions_bar = ReorderableTopActionsBar()
        self.top_actions_bar = top_actions_bar
        top_actions_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_actions_layout = QHBoxLayout(top_actions_bar)
        self.top_actions_layout = top_actions_layout
        top_actions_layout.setContentsMargins(8, 6, 8, 6)
        top_actions_layout.setSpacing(6)
        top_actions_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        top_actions_bar.orderChanged.connect(self._save_topbar_button_order)

        # --- Akcje ---
        self.action_open = QAction("Open", self)
        self.action_open.triggered.connect(self.load_image)

        self.action_save = QAction("Save", self)
        self.action_save.triggered.connect(self.save_image)

        self.action_save_as = QAction("Save As...", self)
        self.action_save_as.triggered.connect(self.save_image_as)

        self.action_exit = QAction("Exit", self)
        self.action_exit.triggered.connect(self.close)

        self.action_undo = QAction("Undo", self)
        self.action_undo.setShortcut("Ctrl+Z")
        self.action_undo.triggered.connect(self.undo)

        self.action_redo = QAction("Redo", self)
        self.action_redo.setShortcut("Ctrl+Y")
        self.action_redo.triggered.connect(self.redo)

        self.action_paste_layer = QAction("Paste Into Selected Layer", self)
        self.action_paste_layer.triggered.connect(self.paste_image_into_selected_layer)

        self.menu_workspace = QMenu("Workspace", self)
        self.action_new_workspace = QAction("New Workspace...", self)
        self.action_new_workspace.triggered.connect(self.show_new_workspace_dialog)
        self.refresh_workspace_menu()

        self.action_set_home_folder = QAction("Home Folder", self)
        self.action_set_home_folder.triggered.connect(self.choose_home_folder)

        self.action_preferences = QAction("Preferencje", self)
        self.action_preferences.triggered.connect(self.show_preferences_dialog)

        self.action_histogram = QAction("Histogram", self)
        self.action_histogram.triggered.connect(self.show_histogram_window)

        self.action_console = QAction("Console", self)
        self.action_console.triggered.connect(self.show_console_window)

        self.action_photoshop_menu = QAction("Menu", self)
        self.action_photoshop_menu.triggered.connect(self.show_photoshop_menu)

        self.action_script_editor = QAction("Script Editor", self)
        self.action_script_editor.triggered.connect(self.open_script_editor)

        self.action_ai_assistant = QAction("AI Assistant", self)
        self.action_ai_assistant.triggered.connect(self.show_ai_assistant_dialog)

        self.action_new_layer = QAction("New Layer", self)
        self.action_new_layer.triggered.connect(self.create_new_layer)

        self.action_delete_layer = QAction("UsuĹ„ zaznaczonÄ… warstwÄ™", self)
        self.action_delete_layer.triggered.connect(self.delete_selected_layer)

        self.action_adjustment_levels = QAction("Levels", self)
        self.action_adjustment_levels.triggered.connect(lambda: self.create_adjustment_layer("levels"))

        self.action_adjustment_curves = QAction("Curves", self)
        self.action_adjustment_curves.triggered.connect(lambda: self.create_adjustment_layer("curves"))

        self.action_magic_filter = QAction("Magic Filter", self)
        self.action_magic_filter.triggered.connect(self.run_magic)

        self.action_star_shrink = QAction("Star Shrink", self)
        self.action_star_shrink.triggered.connect(self.run_star_shrink)

        self.action_plate_solve = QAction("Plate Solving", self)
        self.action_plate_solve.triggered.connect(self.run_plate_solve)

        self.action_starnet = QAction("Run StarNet++", self)
        self.action_starnet.triggered.connect(self.run_starnet)

        self.action_deepsnr = QAction("Run deepSNR", self)
        self.action_deepsnr.triggered.connect(self.run_deepsnr)

        self.action_3d_fly = QAction("3D FLY Filter", self)
        self.action_3d_fly.triggered.connect(self.run_3d_fly_filter)

        self.action_correction = QAction("Correction", self)
        self.action_correction.triggered.connect(self.show_correction_panels)

        self.action_color_calibration = QAction("Color Calibration", self)
        self.action_color_calibration.triggered.connect(self.show_color_calibration_dialog)

        self.action_levels = QAction("Levels", self)
        self.action_levels.triggered.connect(self.show_levels_window)

        self.action_curves = QAction("Curves (LUT)", self)
        self.action_curves.triggered.connect(self.show_curves_window)

        self.action_blur = QAction("Gaussian Blur", self)
        self.action_blur.triggered.connect(self.apply_gaussian_blur_filter)
        self.action_blur.setEnabled(False)

        self.action_rotate = QAction("Rotate", self)
        self.action_rotate.triggered.connect(self.rotate_image_dialog)
        self.action_rotate.setEnabled(False)

        self.action_crop = QAction("Crop", self)
        self.action_crop.triggered.connect(self.crop_image_dialog)
        self.action_crop.setEnabled(False)

        # SkrĂłty klawiszowe dziaĹ‚ajÄ…ce bez MenuBar
        self.addAction(self.action_undo)
        self.addAction(self.action_redo)

        def _add_top_btn(text, action=None, callback=None):
            btn = DraggableTopActionButton("")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(text)
            btn.setStatusTip(text)
            btn.setAccessibleName(text)
            btn.setFixedSize(34, 30)
            if action is not None:
                btn.clicked.connect(action.trigger)
            elif callback is not None:
                btn.clicked.connect(callback)
            top_actions_layout.addWidget(btn)
            return btn

        def _show_workspace_menu():
            if self.menu_workspace is None:
                return
            pos = self.btn_workspace.mapToGlobal(self.btn_workspace.rect().bottomLeft())
            self.menu_workspace.exec_(pos)

        self.btn_open = _add_top_btn("Open", action=self.action_open)
        self.btn_save = _add_top_btn("Save", action=self.action_save)
        self.btn_save_as = _add_top_btn("Save As", action=self.action_save_as)
        self.btn_undo = _add_top_btn("Undo", action=self.action_undo)
        self.btn_redo = _add_top_btn("Redo", action=self.action_redo)
        self.btn_workspace = _add_top_btn("Workspace", callback=_show_workspace_menu)
        self.btn_home = _add_top_btn("Home", action=self.action_set_home_folder)
        self.btn_prefs = _add_top_btn("Prefs", action=self.action_preferences)

        top_actions_layout.addSpacing(8)

        self.btn_console_top = _add_top_btn("Console", action=self.action_console)
        self.btn_menu_top = _add_top_btn("Menu", action=self.action_photoshop_menu)
        self.btn_hist_top = _add_top_btn("Histogram", action=self.action_histogram)
        self.btn_ai_top = _add_top_btn("AI", action=self.action_ai_assistant)

        top_actions_layout.addSpacing(8)

        self.btn_magic_top = _add_top_btn("Magic", action=self.action_magic_filter)
        self.btn_shrink_top = _add_top_btn("Shrink", action=self.action_star_shrink)
        self.btn_plate_top = _add_top_btn("Plate", action=self.action_plate_solve)
        self.btn_starnet_top = _add_top_btn("StarNet", action=self.action_starnet)
        self.btn_deepsnr_top = _add_top_btn("deepSNR", action=self.action_deepsnr)
        self.btn_3d_fly_top = _add_top_btn("3D FLY", action=self.action_3d_fly)
        self.btn_blur_top = _add_top_btn("Blur", action=self.action_blur)
        self.btn_rotate_top = _add_top_btn("Rotate", action=self.action_rotate)
        self.btn_crop_top = _add_top_btn("Crop", action=self.action_crop)
        self.btn_corr_top = _add_top_btn("Correction", action=self.action_correction)
        self.btn_calib_top = _add_top_btn("Calibration", action=self.action_color_calibration)
        self.btn_levels_top = _add_top_btn("Levels", action=self.action_levels)
        self.btn_curves_top = _add_top_btn("Curves", action=self.action_curves)

        top_actions_layout.addSpacing(8)

        self.btn_new_layer_top = _add_top_btn("New Layer", action=self.action_new_layer)
        self.btn_delete_layer_top = _add_top_btn("Delete Layer", action=self.action_delete_layer)
        self.btn_adj_levels_top = _add_top_btn("Adj Levels", action=self.action_adjustment_levels)
        self.btn_adj_curves_top = _add_top_btn("Adj Curves", action=self.action_adjustment_curves)

        top_button_attrs = [
            "btn_open",
            "btn_save",
            "btn_save_as",
            "btn_undo",
            "btn_redo",
            "btn_workspace",
            "btn_home",
            "btn_prefs",
            "btn_console_top",
            "btn_menu_top",
            "btn_hist_top",
            "btn_ai_top",
            "btn_magic_top",
            "btn_shrink_top",
            "btn_plate_top",
            "btn_starnet_top",
            "btn_deepsnr_top",
            "btn_3d_fly_top",
            "btn_blur_top",
            "btn_rotate_top",
            "btn_crop_top",
            "btn_corr_top",
            "btn_calib_top",
            "btn_levels_top",
            "btn_curves_top",
            "btn_new_layer_top",
            "btn_delete_layer_top",
            "btn_adj_levels_top",
            "btn_adj_curves_top",
        ]
        for attr_name in top_button_attrs:
            button = getattr(self, attr_name, None)
            if isinstance(button, DraggableTopActionButton):
                button.setObjectName(attr_name)

        top_actions_layout.addStretch(1)
        self._apply_saved_topbar_button_order()
        self._ensure_topbar_single_trailing_stretch()

        # Ikony SVG dla gornego paska
        icons_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets",
            "icons",
            "top bar",
        )
        top_button_icons = {
            "btn_open": "open.svg",
            "btn_save": "save.svg",
            "btn_save_as": "save_as.svg",
            "btn_undo": "undo.svg",
            "btn_redo": "redo.svg",
            "btn_workspace": "workspace.svg",
            "btn_home": "home.svg",
            "btn_prefs": "prefs.svg",
            "btn_console_top": "console.svg",
            "btn_menu_top": "menu.svg",
            "btn_hist_top": "histogram.svg",
            "btn_ai_top": "ai.svg",
            "btn_magic_top": "magic.svg",
            "btn_shrink_top": "shrink.svg",
            "btn_plate_top": "plate.svg",
            "btn_starnet_top": "starnet.svg",
            "btn_deepsnr_top": "starnet.svg",
            "btn_3d_fly_top": "fly3d.svg",
            "btn_blur_top": "blur.svg",
            "btn_rotate_top": "rotate.svg",
            "btn_crop_top": "crop.svg",
            "btn_corr_top": "correction.svg",
            "btn_calib_top": "correction.svg",
            "btn_levels_top": "levels.svg",
            "btn_curves_top": "curves.svg",
            "btn_new_layer_top": "new_layer.svg",
            "btn_delete_layer_top": "delete_layer.svg",
            "btn_adj_levels_top": "levels.svg",
            "btn_adj_curves_top": "curves.svg",
        }
        for button_name, icon_file in top_button_icons.items():
            button = getattr(self, button_name, None)
            if button is None:
                continue
            icon_path = os.path.join(icons_dir, icon_file)
            if os.path.exists(icon_path):
                button.setIcon(QIcon(icon_path))
                button.setIconSize(QSize(16, 16))

        main_layout.addWidget(top_actions_bar)



        center_layout = QHBoxLayout()

        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        self.camera_raw_panel = CameraRawPanel(self.on_params_changed)
        self.hsl_panel = HSLPanel(self.on_params_changed)

        # Plate solving widgets moved to dedicated dialog
        self.plate_solving_dialog = PlateSolvingDialog(parent=self)
        self.lbl_plate_solve_status = self.plate_solving_dialog.lbl_plate_solve_status
        self.plate_solve_progress = self.plate_solving_dialog.plate_solve_progress
        self.lbl_plate_solve_details = self.plate_solving_dialog.lbl_plate_solve_details
        self.lbl_plate_main_object = self.plate_solving_dialog.lbl_plate_main_object
        self.lbl_plate_catalog = self.plate_solving_dialog.lbl_plate_catalog
        self.lbl_plate_designation = self.plate_solving_dialog.lbl_plate_designation
        self.lbl_plate_objects_in_field = self.plate_solving_dialog.lbl_plate_objects_in_field

        self.ai_assistant_panel = AIAssistantPanel(parent=left_panel, app=self)
        left_layout.addWidget(self.ai_assistant_panel, 1)

        # Ustawienie minimalnej szerokoĹ›ci panelu bocznego
        left_panel.setMinimumWidth(280)
        left_panel.setMaximumWidth(600)

        self.viewer = BlendViewer()
        self.viewer.view.zoom_changed_signal.connect(self.update_viewer_overlay)
        self.viewer.view.imageClicked.connect(self.on_viewer_image_clicked)
        self.viewer.view.roiDragStarted.connect(self._on_roi_drag_started)
        self.viewer.view.roiDragUpdated.connect(self._on_roi_drag_updated)
        self.viewer.view.roiDragFinished.connect(self._on_roi_drag_finished)
        self.viewer.view.roiHoverMoved.connect(self._on_roi_hover_changed)
        self.viewer.imageDropped.connect(self._handle_dropped_images)
        self.viewer.view.set_pick_mode(False)
        self.latest_plate_solve_result = None

        # Splitter umoĹĽliwia zmianÄ™ rozmiaru paneli poprzez przeciÄ…gniÄ™cie
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.viewer)
        splitter.addWidget(left_panel)
        splitter.setStretchFactor(0, 1)  # Viewer rozciÄ…gniÄ™ty
        splitter.setStretchFactor(1, 0)  # Prawy panel nie rozkĹ‚adaÄ‡
        splitter.setCollapsible(0, False)  # Viewer nie moĹĽe byÄ‡ schowany
        splitter.setCollapsible(1, True)  # MoĹĽliwoĹ›Ä‡ schowania prawego panelu
        splitter.setSizes([1250, 350])  # DomyĹ›lne rozmiary

        center_layout.addWidget(splitter)

        # --- Dolny panel z progresem i miniaturkami ---
        bottom_panel = QFrame()
        bottom_panel.setMinimumHeight(90)
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(5, 5, 5, 5)

        # Progress bar
        progress_label = QLabel("Processing:")
        bottom_layout.addWidget(progress_label)
        self.processing_progress = CircularProgressBar()
        self.processing_progress.setRange(0, 100)
        self.processing_progress.setValue(0)
        self.processing_progress.setVisible(False)
        bottom_layout.addWidget(self.processing_progress, 0, Qt.AlignHCenter)

        # Scroll area z miniaturkami
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.thumbnails_container = ThumbnailTreeContainer()
        self.thumbnails_layout = QGridLayout(self.thumbnails_container)
        self.thumbnails_layout.setContentsMargins(0, 0, 0, 0)
        self.thumbnails_layout.setSpacing(5)
        self.thumbnails_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        scroll.setWidget(self.thumbnails_container)
        bottom_layout.addWidget(scroll)

        center_container = QWidget()
        center_container.setLayout(center_layout)

        # Pionowy splitter: gĂłrna sekcja robocza + dolny panel historii/progresu
        vertical_splitter = QSplitter(Qt.Vertical)
        vertical_splitter.addWidget(center_container)
        vertical_splitter.addWidget(bottom_panel)
        vertical_splitter.setStretchFactor(0, 1)
        vertical_splitter.setStretchFactor(1, 0)
        vertical_splitter.setCollapsible(0, False)
        vertical_splitter.setCollapsible(1, True)
        vertical_splitter.setSizes([900, 170])

        main_layout.addWidget(vertical_splitter, 1)
        self.setCentralWidget(main_widget)

        self.update_menu_actions()
    def on_viewer_image_clicked(self, x, y):
     """ObsĹ‚uga klikniÄ™cia na obraz - alternatywa dla PixInsight i Photoshopa"""
     self.log(f"KlikniÄ™to na obraz w punkcie: X={x}, Y={y}")

     # --- FUNKCJA STYL PIXINSIGHT (Dynamic Background Extraction - PrĂłbki tĹ‚a) ---
     if hasattr(self, 'dbe_mode_active') and self.dbe_mode_active:
         if not hasattr(self, 'bg_samples'):
             self.bg_samples = []
         self.bg_samples.append((x, y))
         self.log(f"PixInsight DBE: Dodano prĂłbkÄ™ tĹ‚a ({x}, {y}). ĹÄ…cznie prĂłbki: {len(self.bg_samples)}")
         return

     # --- FUNKCJA STYL PHOTOSHOP (Color Picker - PrĂłbnik koloru dla krzywych/warstw) ---
     if hasattr(self, 'picker_mode_active') and self.picker_mode_active:
         if self.original_img is not None:
             try:
                 # OpenCV przechowuje obrazy w formacie BGR
                 color_bgr = self.original_img[y, x]
                 b, g, r = color_bgr[0], color_bgr[1], color_bgr[2]
                 self.log(f"Photoshop Picker: R={r}, G={g}, B={b}")
                 # Tutaj program w przyszĹ‚oĹ›ci moĹĽe postawiÄ‡ punkt na wykresie Curves (Krzywe)
             except IndexError:
                 pass
             return


    # ---------- Handlery ----------

    def _get_default_dialog_directory(self) -> str:
        if self.home_folder and os.path.isdir(self.home_folder):
            return self.home_folder
        return os.path.expanduser("~")

    def choose_home_folder(self):
        start_dir = self._get_default_dialog_directory()
        folder_path = QFileDialog.getExistingDirectory(
            self,
            self.tr("action_home", "Home Folder"),
            start_dir,
            get_safe_file_dialog_options(),
        )
        folder_path = str(folder_path or "").strip()
        if not folder_path:
            self.log("Home folder selection canceled.", "warning")
            return
        if not os.path.isdir(folder_path):
            self.log(f"Home folder is invalid: {folder_path}", "error")
            return

        self.home_folder = folder_path
        APP_PREFERENCES["home_folder"] = self.home_folder
        save_config(
            self.denoise_model_path,
            self.bg_removal_model_path,
            self.dark_mode,
            self.starnet_path,
            self.plate_solve_api_key,
            self.plate_solve_pixel_size_um,
            self.plate_solve_focal_length_mm,
            self.starnet_stride,
            self.gemini_api_key,
            self.gemini_model,
        )
        self.log(f"Home folder set: {self.home_folder}", "success")

    def _show_open_file_dialog(self, title: str, name_filter: str, directory: str = ""):
        if not directory:
            directory = self._get_default_dialog_directory()
        dialog = QFileDialog(self, title, directory, name_filter)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setOptions(get_safe_file_dialog_options())
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_() != QDialog.Accepted:
            return "", dialog.selectedNameFilter()
        selected_files = dialog.selectedFiles()
        selected_path = selected_files[0] if selected_files else ""
        return selected_path, dialog.selectedNameFilter()

    def _show_save_file_dialog(self, title: str, name_filter: str, directory: str = ""):
        if not directory:
            directory = self._get_default_dialog_directory()
        dialog = QFileDialog(self, title, directory, name_filter)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setOptions(get_safe_file_dialog_options())
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_() != QDialog.Accepted:
            return "", dialog.selectedNameFilter()
        selected_files = dialog.selectedFiles()
        selected_path = selected_files[0] if selected_files else ""
        return selected_path, dialog.selectedNameFilter()

    def load_image(self):
        path, _ = self._show_open_file_dialog(
            "Wybierz obraz",
            "Obrazy (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.fts);;FITS (*.fits *.fit *.fts);;Wszystkie pliki (*)",
            "",
        )
        if not path:
            self.log("Open image canceled.")
            return

        self.load_image_from_path(path)

    def _handle_dropped_images(self, paths):
        if not paths:
            self.log("Drop skipped: no valid image files.", "warning")
            return

        first_path = paths[0]
        self.load_image_from_path(first_path)
        if len(paths) > 1:
            self.log(
                f"Loaded first dropped image: {os.path.basename(first_path)}. Ignored {len(paths) - 1} additional file(s).",
                "warning",
            )

    def dragEnterEvent(self, event):
        if image_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if image_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = image_paths_from_mime_data(event.mimeData())
        if paths:
            self._handle_dropped_images(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def load_image_from_path(self, path):
        if not path:
            self.log("Open skipped: path is empty.", "warning")
            return

        try:
            # Bezpieczne odczytanie pliku bajt po bajcie (obsĹ‚uga polskich znakĂłw)
            img_array = np.fromfile(path, np.uint8)
            loaded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if loaded_img is None:
                raise ValueError("Nie udaĹ‚o siÄ™ zdekodowaÄ‡ obrazu. Plik moĹĽe byÄ‡ uszkodzony.")

            # Przypisanie obrazu
            self.magic_img = loaded_img
            self.original_img = loaded_img.copy()
            self.processed_img = loaded_img.copy()
            self.current_image_path = path
            self.current_save_path = path
            self.latest_image_analysis = {}
            self.analysis_dirty = True

            # Czyszczenie historii operacji dla nowego pliku
            if hasattr(self, 'undo_stack'): self.undo_stack.clear()
            if hasattr(self, 'redo_stack'): self.redo_stack.clear()
            if hasattr(self, 'processing_history'): self.processing_history.clear()
            self.thumbnail_next_id = 1
            self.selected_thumbnail_index = -1
            self.current_history_node_id = None

            # Aktualizacja widoku
            self.viewer.set_before(np_to_qpixmap(self.processed_img))
            self.update_menu_actions()
            self.apply_full_processing()
            
            # Dodaj miniaturÄ™ z nazwÄ… pliku
            filename = os.path.basename(path)
            self.add_thumbnail(filename, self.processed_img)
            
            self.log(f"Successfully loaded: {filename}", "success")

        except Exception as e:
            self.magic_img = None
            self.original_img = None
            self.processed_img = None
            self.latest_image_analysis = {}
            self.analysis_dirty = True
            self.log(f"Error loading image: {str(e)}", "error")
            
            QMessageBox.critical(
                self, 
                "Błąd ładowania obrazu", 
                f"Nie udało się otworzyć pliku.\n\nSzczegóły: {e}\nŚcieżka: {path}"
            )
    def run_magic(self):

        if self.original_img is None:
            self.log("Magic Filter skipped: no image loaded.")
            return

        self.log("Magic Filter started.")
        self.undo_stack.append(self.magic_img.copy())
        self.redo_stack.clear()

        self.progress_dialog = MagicProgressDialog(self)
        self.progress_dialog.show()

        self.worker = MagicWorker(
            self.original_img,
            self.denoise_model_path,
            self.bg_removal_model_path
        )

        self.worker.progress_signal.connect(
            self.progress_dialog.update_progress
        )

        self.worker.finished_signal.connect(
            self.on_magic_finished
        )

        self.worker.start()

    def run_starnet(self):
        if self.magic_img is None:
            self.log("StarNet++ skipped: no image loaded.", "warning")
            QMessageBox.warning(self, "StarNet++", "Brak obrazu. Najpierw wczytaj zdjęcie.")
            return

        dialog = self._get_starnet_dialog()
        if dialog.exec_() != QDialog.Accepted:
            self.log("StarNet++ canceled.", "warning")
            return

        params = dialog.get_parameters()
        self.starnet_stride = params["stride"]
        save_config(
            self.denoise_model_path,
            self.bg_removal_model_path,
            self.dark_mode,
            self.starnet_path,
            self.plate_solve_api_key,
            self.plate_solve_pixel_size_um,
            self.plate_solve_focal_length_mm,
            self.starnet_stride,
            self.gemini_api_key,
            self.gemini_model,
        )

        self.log("StarNet++ started.")
        self.starnet_source_img = self.magic_img.copy()

        self.progress_dialog = MagicProgressDialog(self)
        self.progress_dialog.show()
        self.progress_dialog.update_progress("Starting StarNet++...", 0, 0)

        self.starnet_worker = StarNetWorker(self.magic_img.copy(), self.starnet_path, self.starnet_stride)
        self.starnet_worker.progress_signal.connect(self.progress_dialog.update_progress)
        self.starnet_worker.finished_signal.connect(self.on_starnet_finished)
        self.starnet_worker.start()

    def run_3d_fly_filter(self):
        if self.magic_img is None:
            self.log("3D FLY skipped: no image loaded.", "warning")
            return

        dialog = self._get_fly3d_dialog()
        if dialog is None:
            self.log("3D FLY skipped: no valid source image.", "warning")
            return
        if dialog.exec_() != QDialog.Accepted:
            self.log("3D FLY canceled.", "warning")
            return

        payload = dialog.get_payload()
        if not payload.get("output_path"):
            self.log("3D FLY canceled: output path is empty.", "warning")
            return

        source_img = payload.get("edited_source_image")
        if not isinstance(source_img, np.ndarray):
            source_img = self.get_effective_magic_img()
        if source_img is None:
            source_img = self.magic_img

        self.progress_dialog = MagicProgressDialog(self)
        self.progress_dialog.show()
        self.progress_dialog.update_progress("Starting 3D FLY...", 0, 0)

        self.fly3d_worker = Fly3DWorker(
            source_img,
            payload,
            starnet_path=self.starnet_path,
            starnet_stride=self.starnet_stride,
        )
        self.fly3d_worker.progress_signal.connect(self.progress_dialog.update_progress)
        self.fly3d_worker.finished_signal.connect(self.on_3d_fly_finished)
        self.fly3d_worker.start()

    def on_3d_fly_finished(self, output_path: str, error_message: str):
        if hasattr(self, "progress_dialog") and self.progress_dialog is not None:
            self.progress_dialog.close()

        if error_message:
            if output_path:
                self.log(error_message, "warning")
            else:
                self.log(error_message, "error")
                return
        self.log(f"3D FLY finished. Video saved: {output_path}", "success")

    def run_plate_solve(self):
        if self.original_img is None:
            self.log("Plate Solving skipped: no image loaded.", "warning")
            return

        dialog = self._get_plate_solve_dialog()
        if dialog.exec_() != QDialog.Accepted:
            self.log("Plate Solving canceled.", "warning")
            return

        params = dialog.get_parameters()
        self.plate_solve_pixel_size_um = params["pixel_size_um"]
        self.plate_solve_focal_length_mm = params["focal_length_mm"]
        self.plate_solve_api_key = params["api_key"]
        save_config(
            self.denoise_model_path,
            self.bg_removal_model_path,
            self.dark_mode,
            self.starnet_path,
            self.plate_solve_api_key,
            self.plate_solve_pixel_size_um,
            self.plate_solve_focal_length_mm,
            self.starnet_stride,
            self.gemini_api_key,
            self.gemini_model,
        )
        self.log("Plate Solving started.")
        self.show_plate_solving_dialog()
        self.lbl_plate_solve_status.setText("Plate solve status: solving...")
        self.plate_solve_start_time = time.time()

        self.plate_solve_worker = PlateSolveWorker(
            self.original_img.copy(),
            params["pixel_size_um"],
            params["focal_length_mm"],
            api_key=params.get("api_key") or None,
        )
        self.plate_solve_worker.progress_signal.connect(self.on_plate_solve_progress)
        self.plate_solve_worker.finished_signal.connect(lambda result, error: self.on_plate_solve_finished(result, error, params["overlay_enabled"]))
        self.plate_solve_worker.start()

    def on_plate_solve_progress(self, stage: str, overall: int, current: int):
        elapsed = ""
        if hasattr(self, "plate_solve_start_time") and self.plate_solve_start_time is not None:
            elapsed_seconds = int(time.time() - self.plate_solve_start_time)
            elapsed = f" | Elapsed: {elapsed_seconds}s"

        self.lbl_plate_solve_status.setText(f"{stage}{elapsed}")
        self.plate_solve_progress.setRange(0, 0)
        QApplication.processEvents()

    def on_plate_solve_finished(self, result, error_message, overlay_enabled: bool):
        self.plate_solve_progress.setRange(0, 1)
        self.plate_solve_progress.setValue(1)
        self.plate_solve_start_time = None

        if error_message:
            self.lbl_plate_solve_status.setText("Plate solve status: failed")
            self.lbl_plate_solve_details.setText("Plate solve details: failed")
            self.log(f"Plate Solving failed: {error_message}", "error")
            self.viewer.clear_overlay()
            return

        if result is None:
            self.lbl_plate_solve_status.setText("Plate solve status: failed")
            self.lbl_plate_solve_details.setText("Plate solve details: no result")
            self.log("Plate Solving failed: no result.", "error")
            self.viewer.clear_overlay()
            return

        ra = result.get("ra")
        dec = result.get("dec")
        rotation = result.get("rotation")
        scale = result.get("scale")
        fov_x = result.get("fov_x")
        fov_y = result.get("fov_y")

        object_info = self._build_plate_solve_object_info(result)
        result["plate_solve_object_info"] = object_info
        self.plate_solve_object_info = object_info

        details = [
            f"RA: {ra:.5f}Â°" if ra is not None else "RA: unknown",
            f"Dec: {dec:.5f}Â°" if dec is not None else "Dec: unknown",
            f"Rotation: {rotation:.2f}Â°" if rotation is not None else "Rotation: unknown",
            f"Scale: {scale:.3f} arcsec/pixel" if scale is not None else "Scale: unknown",
        ]

        self.lbl_plate_solve_status.setText("Plate solve status: solved")
        self.lbl_plate_solve_details.setText("\n".join(details))
        self._refresh_plate_solve_object_panel()
        self.add_thumbnail("Plate Solve", self.processed_img)
        self.log("Plate Solving succeeded. " + " | ".join(details), "success")

        self._update_plate_solve_layers(result)
        if overlay_enabled:
            self.layer_visibility["plate_solve_overlay"] = True
            self.layer_visibility["grid_overlay"] = True
            self.layer_visibility["object_labels"] = True
            self.update_viewer_overlay()
            self.update_photoshop_panel()
        else:
            self.viewer.clear_overlay()

    def _build_plate_overlay(self, width: int, height: int, points):
        overlay = QPixmap(width, height)
        overlay.fill(Qt.transparent)
        painter = QPainter(overlay)
        pen = QPen(QColor(255, 255, 0, 180))
        pen.setWidth(1)
        painter.setPen(pen)

        step_x = max(1, width // 6)
        step_y = max(1, height // 6)
        for x in range(step_x, width, step_x):
            painter.drawLine(x, 0, x, height)
        for y in range(step_y, height, step_y):
            painter.drawLine(0, y, width, y)

        pen.setWidth(2)
        pen.setColor(QColor(255, 100, 100, 220))
        painter.setPen(pen)
        painter.drawLine(width // 2, 0, width // 2, height)
        painter.drawLine(0, height // 2, width, height // 2)

        if points:
            pen.setWidth(3)
            pen.setColor(QColor(0, 255, 0, 220))
            painter.setPen(pen)
            for x, y in points[:200]:
                painter.drawPoint(int(x), int(y))
        painter.end()
        return overlay

    def _normalize_object_name(self, label):
        value = str(label or "").strip()
        value = re.sub(r"\s+", " ", value)
        if not value:
            return None
        compact = value.replace(" ", "")
        if m := re.match(r"(?i)^M\s*(\d+)$", value):
            return f"M{int(m.group(1))}"
        if m := re.match(r"(?i)^(NGC|IC)\s*(\d+)$", value):
            return f"{m.group(1).upper()}{int(m.group(2))}"
        if m := re.match(r"(?i)^(?:SH2|SH-2|SH\s*2|SHARPLESS\s*2?|SH)\s*-?\s*(\d+)$", value):
            return f"Sh2-{int(m.group(1))}"
        if m := re.match(r"(?i)^(?:B|BARNARD)\s*-?\s*(\d+)$", value):
            return f"B{int(m.group(1))}"
        if m := re.match(r"(?i)^(LDN|LBN)\s*-?\s*(\d+)$", value):
            return f"{m.group(1).upper()}{int(m.group(2))}"
        if m := re.match(r"(?i)^(?:C|CALDWELL)\s*-?\s*(\d+)$", value):
            return f"C{int(m.group(1))}"
        if m := re.match(r"(?i)^SH2-(\d+)$", compact):
            return f"Sh2-{int(m.group(1))}"
        return value

    def _catalog_object_label(self, obj):
        if not obj:
            return None
        display_name = obj.get("display_name")
        if display_name and display_name != obj.get("name"):
            return f"{obj['name']} {display_name}"
        return obj.get("name")

    def _catalog_object_designation(self, obj):
        aliases = obj.get("aliases") or []
        return f"{obj['name']} / {aliases[0]}" if aliases else obj.get("name")

    def _format_plate_object_label(self, objects, ra=None, dec=None):
        object_names = [self._normalize_object_name(o) for o in (objects or [])]
        object_names = [o for o in object_names if o]
        if object_names:
            primary = None
            for pattern in [r"^M\d+$", r"^NGC\d+$", r"^IC\d+$", r"^Sh2-\d+$", r"^B\d+$", r"^LDN\d+$", r"^LBN\d+$", r"^C\d+$"]:
                for name in object_names:
                    if re.match(pattern, name, re.I):
                        primary = name
                        break
                if primary:
                    break
            if not primary:
                primary = min(object_names, key=lambda value: len(value.replace(" ", "")))
            aliases = [o for o in object_names if o != primary]
            if aliases:
                return f"{primary} {aliases[0]}"
            return primary
        nearest = self._find_nearest_messier(ra, dec)
        return self._catalog_object_label(nearest)

    def _find_nearest_messier(self, ra, dec):
        if ra is None or dec is None:
            return None
        nearest = self._find_nearest_simbad(ra, dec)
        if nearest is not None:
            return nearest
        return self._find_nearest_local_catalog(ra, dec)

    def _simbad_tap_query(self, adql_query):
        now = time.time()
        if getattr(self, "_simbad_disabled_until", 0) > now:
            return None
        url = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
        payload = urllib.parse.urlencode({
            "request": "doQuery",
            "lang": "adql",
            "format": "json",
            "query": adql_query,
        }).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "AstroAiPlus/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=2.5) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except Exception:
            self._simbad_disabled_until = time.time() + 120.0
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        rows = parsed.get("data")
        if not isinstance(rows, list):
            return None
        return rows

    def _catalog_from_designation(self, designation):
        norm = self._normalize_object_name(designation) or str(designation or "")
        upper = norm.upper()
        if re.match(r"^M\d+$", upper):
            return "Messier"
        if re.match(r"^NGC\d+$", upper):
            return "NGC"
        if re.match(r"^IC\d+$", upper):
            return "IC"
        if re.match(r"^SH2-\d+$", upper):
            return "Sharpless"
        if re.match(r"^B\d+$", upper):
            return "Barnard"
        if re.match(r"^LDN\d+$", upper):
            return "LDN"
        if re.match(r"^LBN\d+$", upper):
            return "LBN"
        if re.match(r"^C\d+$", upper):
            return "Caldwell"
        return "SIMBAD"

    def _build_catalog_obj(self, name, ra, dec, display_name=None, aliases=None, catalog=None):
        normalized_name = self._normalize_object_name(name) or str(name or "").strip()
        if not normalized_name:
            return None
        normalized_aliases = []
        for alias in aliases or []:
            alias_value = str(alias or "").strip()
            if alias_value and alias_value not in normalized_aliases:
                normalized_aliases.append(alias_value)
        return {
            "name": normalized_name,
            "display_name": str(display_name or normalized_name).strip() or normalized_name,
            "aliases": normalized_aliases,
            "ra": ra,
            "dec": dec,
            "catalog": catalog or self._catalog_from_designation(normalized_name),
        }

    def _parse_simbad_row(self, row):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            return None
        try:
            main_id = str(row[0] or "").strip()
            ra = float(row[1])
            dec = float(row[2])
        except Exception:
            return None
        otype = ""
        if len(row) >= 4 and row[3] is not None:
            otype = str(row[3]).strip()
        aliases = [main_id] if main_id else []
        return self._build_catalog_obj(
            name=main_id,
            display_name=main_id,
            aliases=aliases,
            ra=ra,
            dec=dec,
            catalog=self._catalog_from_designation(main_id) if main_id else (otype or "SIMBAD"),
        )

    def _simbad_name_variants(self, normalized_name):
        variants = []
        raw = str(normalized_name or "").strip()
        if raw:
            variants.append(raw)
        if m := re.match(r"^M(\d+)$", raw, re.I):
            variants.append(f"M {int(m.group(1))}")
        if m := re.match(r"^(NGC|IC)\s*(\d+)$", raw, re.I):
            prefix = m.group(1).upper()
            value = int(m.group(2))
            variants.append(f"{prefix} {value}")
        if m := re.match(r"^C(\d+)$", raw, re.I):
            value = int(m.group(1))
            variants.append(f"C {value}")
            variants.append(f"Caldwell {value}")
        if m := re.match(r"^Sh2-(\d+)$", raw, re.I):
            value = int(m.group(1))
            variants.append(f"Sh2-{value}")
            variants.append(f"Sh 2-{value}")
            variants.append(f"Sh2 {value}")
        if m := re.match(r"^B(\d+)$", raw, re.I):
            variants.append(f"Barnard {int(m.group(1))}")
        if m := re.match(r"^(LDN|LBN)(\d+)$", raw, re.I):
            variants.append(f"{m.group(1).upper()} {int(m.group(2))}")
        unique = []
        for variant in variants:
            item = str(variant or "").strip()
            if item and item.lower() not in [v.lower() for v in unique]:
                unique.append(item)
        return unique

    def _find_object_by_name_simbad(self, name):
        normalized = self._normalize_object_name(name)
        if not normalized:
            return None
        if not hasattr(self, "_simbad_name_cache"):
            self._simbad_name_cache = {}
        cache_key = normalized.upper()
        if cache_key in self._simbad_name_cache:
            return self._simbad_name_cache[cache_key]
        for variant in self._simbad_name_variants(normalized):
            escaped = variant.replace("'", "''")
            query = (
                "SELECT TOP 1 basic.main_id, basic.ra, basic.dec, basic.otype_txt "
                "FROM ident JOIN basic ON ident.oidref = basic.oid "
                f"WHERE ident.id = '{escaped}'"
            )
            rows = self._simbad_tap_query(query)
            if rows:
                obj = self._parse_simbad_row(rows[0])
                if obj is not None:
                    obj["aliases"] = list(dict.fromkeys((obj.get("aliases") or []) + [variant, normalized]))
                    self._simbad_name_cache[cache_key] = obj
                    return obj
        self._simbad_name_cache[cache_key] = None
        return None

    def _find_nearest_simbad(self, ra, dec):
        if ra is None or dec is None:
            return None
        if not hasattr(self, "_simbad_nearest_cache"):
            self._simbad_nearest_cache = {}
        cache_key = (round(float(ra), 4), round(float(dec), 4))
        if cache_key in self._simbad_nearest_cache:
            return self._simbad_nearest_cache[cache_key]
        radius_deg = 2.0
        query = (
            "SELECT TOP 1 main_id, ra, dec, otype_txt FROM basic "
            "WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {float(ra):.8f}, {float(dec):.8f}, {radius_deg:.8f})) "
            "ORDER BY DISTANCE(POINT('ICRS', ra, dec), "
            f"POINT('ICRS', {float(ra):.8f}, {float(dec):.8f})) ASC"
        )
        rows = self._simbad_tap_query(query)
        if rows:
            parsed = self._parse_simbad_row(rows[0])
            self._simbad_nearest_cache[cache_key] = parsed
            return parsed
        self._simbad_nearest_cache[cache_key] = None
        return None

    def _find_nearest_local_catalog(self, ra, dec):
        if ra is None or dec is None:
            return None
        best = None
        best_dist = None
        mean_dec = math.radians(dec)
        cos_dec = math.cos(mean_dec)
        for obj in DEEP_SKY_CATALOG:
            obj_ra = obj.get("ra")
            obj_dec = obj.get("dec")
            if obj_ra is None or obj_dec is None:
                continue
            dra = math.radians((obj_ra - ra + 180.0) % 360.0 - 180.0)
            ddec = math.radians(obj_dec - dec)
            dist = math.hypot(dra * cos_dec, ddec)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = obj
        return best

    def _find_messier_by_name(self, name):
        by_api = self._find_object_by_name_simbad(name)
        if by_api is not None:
            return by_api
        if not name:
            return None
        normalized = self._normalize_object_name(name)
        if not normalized:
            return None
        for obj in DEEP_SKY_CATALOG:
            candidates = [obj.get("name", ""), obj.get("display_name", "")] + obj.get("aliases", [])
            for candidate in candidates:
                if not candidate:
                    continue
                if candidate.upper() == normalized.upper():
                    return obj
                normalized_candidate = self._normalize_object_name(candidate)
                if normalized_candidate and normalized_candidate.upper() == normalized.upper():
                    return obj
        return None

    def _build_plate_solve_object_info(self, result):
        objects = result.get("objects_in_field") or []
        ra = result.get("ra")
        dec = result.get("dec")

        normalized = []
        for o in objects:
            name = None
            if isinstance(o, str):
                name = self._normalize_object_name(o)
            elif isinstance(o, dict):
                name = self._normalize_object_name(o.get("name") or o.get("id") or "")
            if name and name not in normalized:
                normalized.append(name)

        candidate_info = []
        for name in normalized:
            candidate_info.append({
                "label": name,
                "catalog_obj": self._find_messier_by_name(name),
            })

        best_candidate = None
        if ra is not None and dec is not None:
            best_dist = None
            for entry in candidate_info:
                if entry["catalog_obj"] is None:
                    continue
                obj = entry["catalog_obj"]
                dra = math.radians((obj["ra"] - ra + 180.0) % 360.0 - 180.0)
                ddec = math.radians(obj["dec"] - dec)
                dist = math.hypot(dra * math.cos(math.radians(dec)), ddec)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_candidate = entry

        if best_candidate is None:
            for pattern in [r"^M\d+$", r"^NGC\d+$", r"^IC\d+$", r"^Sh2-\d+$", r"^B\d+$", r"^LDN\d+$", r"^LBN\d+$", r"^C\d+$"]:
                for entry in candidate_info:
                    if re.match(pattern, entry["label"], re.I):
                        best_candidate = entry
                        break
                if best_candidate is not None:
                    break
        if best_candidate is None and candidate_info:
            best_candidate = candidate_info[0]

        object_info = {
            "main_object": None,
            "catalog": "Unknown",
            "designation": None,
            "objects_in_field": [],
        }

        if best_candidate is not None:
            if best_candidate["catalog_obj"] is not None:
                obj = best_candidate["catalog_obj"]
                object_info["main_object"] = self._catalog_object_label(obj)
                object_info["catalog"] = obj.get("catalog", "Deep Sky")
                object_info["designation"] = self._catalog_object_designation(obj)
            else:
                object_info["main_object"] = best_candidate["label"]
                object_info["designation"] = best_candidate["label"]
                object_info["catalog"] = "Astrometry"

        if best_candidate is not None:
            object_info["objects_in_field"] = [entry["label"] for entry in candidate_info if entry["label"] != best_candidate["label"]]
        else:
            if ra is not None and dec is not None:
                nearest = self._find_nearest_messier(ra, dec)
                if nearest is not None:
                    object_info["main_object"] = self._catalog_object_label(nearest)
                    object_info["catalog"] = nearest.get("catalog", "Deep Sky")
                    object_info["designation"] = self._catalog_object_designation(nearest)

        return object_info

    def _refresh_plate_solve_object_panel(self):
        info = self.plate_solve_object_info or {}
        object_label = info.get("main_object") or "unknown"
        catalog = info.get("catalog") or "unknown"
        designation = info.get("designation") or "unknown"
        objects_in_field = info.get("objects_in_field") or []
        objects_text = ", ".join(objects_in_field) if objects_in_field else "none"

        self.lbl_plate_main_object.setText(f"Object: {object_label}")
        self.lbl_plate_catalog.setText(f"Catalog: {catalog}")
        self.lbl_plate_designation.setText(f"Designation: {designation}")
        self.lbl_plate_objects_in_field.setText(f"Objects in Field: {objects_text}")

    def _project_pixel_to_radec(self, x, y, header):
        crpix1 = header.get("CRPIX1", header.get("CRPIX", 0.0))
        crpix2 = header.get("CRPIX2", header.get("CRPIX", 0.0))
        crval1 = header.get("CRVAL1", 0.0)
        crval2 = header.get("CRVAL2", 0.0)
        cd11 = header.get("CD1_1", header.get("CDELT1", 0.0))
        cd12 = header.get("CD1_2", 0.0)
        cd21 = header.get("CD2_1", 0.0)
        cd22 = header.get("CD2_2", header.get("CDELT2", 0.0))
        dx = x - crpix1
        dy = y - crpix2
        ra = crval1 + dx * cd11 + dy * cd12
        dec = crval2 + dx * cd21 + dy * cd22
        return ra, dec

    def _nice_grid_step(self, degrees):
        if degrees <= 0:
            return 1.0
        for step in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            if degrees / step <= 10:
                return step
        return 10.0

    def _nice_coord_label(self, value, is_ra=False):
        if value is None:
            return "?"
        if is_ra:
            hours = value / 15.0
            h = int(hours) % 24
            m = int((hours - h) * 60)
            s = (hours - h - m / 60.0) * 3600.0
            return f"{h:02d}h{m:02d}m{int(s):02d}s"
        return f"{value:.2f}Â°"

    def _build_grid_overlay(self, width: int, height: int, header):
        overlay = QPixmap(width, height)
        overlay.fill(Qt.transparent)
        painter = QPainter(overlay)
        pen = QPen(QColor(0, 200, 255, 140))
        pen.setWidth(1)
        painter.setPen(pen)

        crval1 = header.get("CRVAL1")
        crval2 = header.get("CRVAL2")
        if crval1 is None or crval2 is None:
            painter.end()
            return overlay

        cd11 = header.get("CD1_1", header.get("CDELT1", 0.0))
        cd12 = header.get("CD1_2", 0.0)
        cd21 = header.get("CD2_1", 0.0)
        cd22 = header.get("CD2_2", header.get("CDELT2", 0.0))
        scale_x = math.hypot(cd11, cd12) * 3600.0
        scale_y = math.hypot(cd21, cd22) * 3600.0

        ra_center, dec_center = crval1, crval2
        ra_range = abs(width * scale_x / 3600.0) if scale_x else 1.0
        dec_range = abs(height * scale_y / 3600.0) if scale_y else 1.0
        ra_step = self._nice_grid_step(ra_range / 4.0)
        dec_step = self._nice_grid_step(dec_range / 4.0)

        crpix1 = header.get("CRPIX1", header.get("CRPIX", width / 2.0))
        crpix2 = header.get("CRPIX2", header.get("CRPIX", height / 2.0))

        ra_lines = []
        ra_start = ra_center - ra_range / 2.0
        ra_end = ra_center + ra_range / 2.0
        ra_value = math.floor(ra_start / ra_step) * ra_step
        while ra_value <= ra_end:
            ra_lines.append(ra_value)
            ra_value += ra_step

        dec_lines = []
        dec_start = dec_center - dec_range / 2.0
        dec_end = dec_center + dec_range / 2.0
        dec_value = math.floor(dec_start / dec_step) * dec_step
        while dec_value <= dec_end:
            dec_lines.append(dec_value)
            dec_value += dec_step

        def compute_line_points(const_value, is_ra):
            points = []
            if is_ra:
                a, b, c = cd11, cd12, const_value - ra_center
            else:
                a, b, c = cd21, cd22, const_value - dec_center
            if abs(b) > 1e-9:
                y0 = crpix2 + (c - a * (0.0 - crpix1)) / b
                y1 = crpix2 + (c - a * (width - crpix1)) / b
                points.append((0.0, y0))
                points.append((width, y1))
            elif abs(a) > 1e-9:
                x0 = crpix1 + (c - b * (0.0 - crpix2)) / a
                x1 = crpix1 + (c - b * (height - crpix2)) / a
                points.append((x0, 0.0))
                points.append((x1, height))
            return points

        for ra_value in ra_lines:
            pts = compute_line_points(ra_value, True)
            visible = []
            for px, py in pts:
                if 0 <= px <= width and 0 <= py <= height:
                    visible.append((px, py))
            if len(visible) >= 2:
                painter.drawLine(int(visible[0][0]), int(visible[0][1]), int(visible[1][0]), int(visible[1][1]))
                label = self._nice_coord_label(ra_value, is_ra=True)
                painter.drawText(int(visible[0][0] + 4), int(visible[0][1] + 12), label)

        for dec_value in dec_lines:
            pts = compute_line_points(dec_value, False)
            visible = []
            for px, py in pts:
                if 0 <= px <= width and 0 <= py <= height:
                    visible.append((px, py))
            if len(visible) >= 2:
                painter.drawLine(int(visible[0][0]), int(visible[0][1]), int(visible[1][0]), int(visible[1][1]))
                label = self._nice_coord_label(dec_value, is_ra=False)
                painter.drawText(int(visible[0][0] + 4), int(visible[0][1] + 14), label)

        painter.end()
        return overlay

    def _build_plate_solve_overlay(self, width: int, height: int, result):
        overlay = QPixmap(width, height)
        overlay.fill(Qt.transparent)
        painter = QPainter(overlay)
        pen = QPen(QColor(255, 255, 100, 190))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(width // 2, 0, width // 2, height)
        painter.drawLine(0, height // 2, width, height // 2)
        painter.drawRect(8, 8, width - 16, height - 16)
        info = []
        ra = result.get("ra")
        dec = result.get("dec")
        rotation = result.get("rotation")
        scale = result.get("scale")
        if ra is not None:
            info.append(f"RA: {ra:.5f}Â°")
        if dec is not None:
            info.append(f"Dec: {dec:.5f}Â°")
        if rotation is not None:
            info.append(f"Rot: {rotation:.2f}Â°")
        if scale is not None:
            info.append(f"Scale: {scale:.3f}â€ł/px")
        painter.drawText(12, height - 36, " | ".join(info))
        painter.end()
        return overlay

    def _project_radec_to_pixel(self, ra, dec, header):
        crpix1 = header.get('CRPIX1', header.get('CRPIX', 0.0))
        crpix2 = header.get('CRPIX2', header.get('CRPIX', 0.0))
        crval1 = header.get('CRVAL1', 0.0)
        crval2 = header.get('CRVAL2', 0.0)
        cd11 = header.get('CD1_1', header.get('CDELT1', 0.0))
        cd12 = header.get('CD1_2', 0.0)
        cd21 = header.get('CD2_1', 0.0)
        cd22 = header.get('CD2_2', header.get('CDELT2', 0.0))
        det = cd11 * cd22 - cd12 * cd21
        if abs(det) < 1e-12:
            return None
        dx = ra - crval1
        dy = dec - crval2
        dpix = (cd22 * dx - cd12 * dy) / det
        dpiy = (-cd21 * dx + cd11 * dy) / det
        return crpix1 + dpix, crpix2 + dpiy

    def _build_object_labels_overlay(self, width: int, height: int, result, current_zoom: float = 1.0):
        overlay = QPixmap(width, height)
        overlay.fill(Qt.transparent)
        painter = QPainter(overlay)
        base_pen = QPen(QColor(255, 200, 120, 220))
        base_pen.setWidth(2)
        painter.setPen(base_pen)

        header = result.get('wcs_header', {}) if isinstance(result, dict) else {}
        ra_center = result.get('ra') if isinstance(result, dict) else None
        dec_center = result.get('dec') if isinstance(result, dict) else None
        objects = result.get('objects_in_field', []) if isinstance(result, dict) else []

        normalized = [self._normalize_object_name(o) for o in (objects or [])]
        normalized = [o for o in normalized if o]
        object_label = None
        target_obj = None

        if normalized:
            object_label = self._format_plate_object_label(normalized, ra_center, dec_center)
            for label in normalized:
                if mo := self._find_messier_by_name(label):
                    target_obj = mo
                    break
        else:
            target_obj = self._find_nearest_messier(ra_center, dec_center)
            if target_obj:
                object_label = self._format_plate_object_label([], ra_center, dec_center)

        font_size = max(12, int(12 * max(1.0, current_zoom)))
        font = QFont()
        font.setPointSize(font_size)
        font.setBold(True)
        painter.setFont(font)

        label_text = f"Object: {object_label}" if object_label else "Object: unknown"
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(label_text)
        text_height = metrics.height()

        marker_pos = None
        if target_obj is not None:
            proj = self._project_radec_to_pixel(target_obj['ra'], target_obj['dec'], header)
            if proj is not None:
                px, py = proj
                if 0 <= px <= width and 0 <= py <= height:
                    marker_pos = (int(px), int(py))

        if marker_pos is not None:
            px, py = marker_pos
            painter.setPen(base_pen)
            painter.drawEllipse(px - 6, py - 6, 12, 12)
            text_x = px + 12
            text_y = py - 12
            if text_x + text_width + 12 > width:
                text_x = px - text_width - 16
            if text_x < 8:
                text_x = 8
            if text_y < text_height + 8:
                text_y = min(height - 8, py + text_height + 12)
        else:
            text_x = max(8, width // 2 - text_width // 2)
            text_y = max(text_height + 8, height // 2)

        bg_rect_x = int(text_x - 6)
        bg_rect_y = int(text_y - text_height - 4)
        bg_rect_w = int(text_width + 12)
        bg_rect_h = int(text_height + 8)
        painter.fillRect(bg_rect_x, bg_rect_y, bg_rect_w, bg_rect_h, QColor(0, 0, 0, 180))

        painter.setPen(QPen(QColor(255, 255, 255, 255)))
        painter.drawText(text_x, text_y, label_text)

        painter.end()
        return overlay

    def _build_constellation_overlay(self, width: int, height: int):
        overlay = QPixmap(width, height)
        overlay.fill(Qt.transparent)
        painter = QPainter(overlay)
        pen = QPen(QColor(180, 180, 255, 170))
        pen.setWidth(2)
        painter.setPen(pen)
        if self.constellation_lines:
            for line in self.constellation_lines:
                if len(line) >= 2:
                    x1, y1 = line[0]
                    x2, y2 = line[1]
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        painter.end()
        return overlay

    def _update_plate_solve_layers(self, result):
        if self.original_img is None:
            return
        self.latest_plate_solve_result = result
        width = self.original_img.shape[1]
        height = self.original_img.shape[0]
        self.add_layer("plate_solve_overlay", self._build_plate_solve_overlay(width, height, result), title="Plate Solve Overlay")
        self.add_layer("grid_overlay", self._build_grid_overlay(width, height, result.get("wcs_header", {})), title="Grid Overlay")
        self.add_layer("object_labels", self._build_object_labels_overlay(width, height, result, current_zoom=self.viewer.view.scale_factor), title="Object Labels")
        self.add_layer("constellation_overlay", self._build_constellation_overlay(width, height), title="Constellation Overlay")

        self.update_viewer_overlay()

    def update_viewer_overlay(self):
        if self.original_img is None:
            return
        width = self.original_img.shape[1]
        height = self.original_img.shape[0]
        overlay = QPixmap(width, height)
        overlay.fill(Qt.transparent)
        painter = QPainter(overlay)
        has_content = False
        for key in ["grid_overlay", "plate_solve_overlay", "object_labels", "constellation_overlay"]:
            if not self.layer_visibility.get(key, False):
                continue
            if key == "object_labels" and self.latest_plate_solve_result is not None:
                pix = self._build_object_labels_overlay(
                    width,
                    height,
                    self.latest_plate_solve_result,
                    current_zoom=getattr(self.viewer.view, "scale_factor", 1.0),
                )
                self.layer_images[key] = pix
            else:
                pix = self.layer_images.get(key)
            if isinstance(pix, QPixmap) and pix.size() == overlay.size():
                painter.drawPixmap(0, 0, pix)
                has_content = True
        if self._crop_overlay_active:
            painter.drawPixmap(0, 0, self._build_crop_overlay(width, height))
            has_content = True
        if self._bn_overlay_active:
            painter.drawPixmap(0, 0, self._build_bn_overlay(width, height))
            has_content = True
        painter.end()
        if has_content:
            self.viewer.set_overlay_pixmap(overlay)
        else:
            self.viewer.clear_overlay()

    def _set_crop_overlay_state(self, x: int, y: int, w: int, h: int):
        self._crop_overlay_rect = (int(x), int(y), int(w), int(h))
        if self._crop_overlay_active:
            self.update_viewer_overlay()

    def _set_crop_overlay_grid(self, grid_index: int):
        self._crop_overlay_grid_index = int(grid_index)
        if self._crop_overlay_active:
            self.update_viewer_overlay()

    def _build_crop_overlay(self, width: int, height: int) -> QPixmap:
        overlay = QPixmap(width, height)
        overlay.fill(Qt.transparent)
        painter = QPainter(overlay)
        x, y, w, h = self._crop_overlay_rect
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        w = max(1, min(width - x, int(w)))
        h = max(1, min(height - y, int(h)))
        rect = QRectF(float(x), float(y), float(w), float(h))

        shade = QColor(0, 0, 0, 120)
        painter.fillRect(QRectF(0.0, 0.0, float(width), max(0.0, rect.top())), shade)
        painter.fillRect(QRectF(0.0, rect.bottom(), float(width), max(0.0, float(height) - rect.bottom())), shade)
        painter.fillRect(QRectF(0.0, rect.top(), max(0.0, rect.left()), rect.height()), shade)
        painter.fillRect(QRectF(rect.right(), rect.top(), max(0.0, float(width) - rect.right()), rect.height()), shade)

        painter.setPen(QPen(QColor("#00e5ff"), 2))
        painter.drawRect(rect)
        self._draw_crop_grid_overlay(painter, rect)

        painter.setBrush(QColor("#00e5ff"))
        painter.setPen(QPen(QColor("#003b42"), 1))
        handle_size = 12
        handle_half = handle_size / 2.0
        handle_points = {
            "nw": QPointF(rect.left(), rect.top()),
            "ne": QPointF(rect.right(), rect.top()),
            "sw": QPointF(rect.left(), rect.bottom()),
            "se": QPointF(rect.right(), rect.bottom()),
            "n": QPointF(rect.center().x(), rect.top()),
            "s": QPointF(rect.center().x(), rect.bottom()),
            "w": QPointF(rect.left(), rect.center().y()),
            "e": QPointF(rect.right(), rect.center().y()),
        }
        for point in handle_points.values():
            painter.drawRect(
                QRectF(
                    point.x() - handle_half,
                    point.y() - handle_half,
                    handle_size,
                    handle_size,
                )
            )

        painter.end()
        return overlay

    def _draw_crop_grid_overlay(self, painter: QPainter, rect: QRectF):
        grid_index = int(getattr(self, "_crop_overlay_grid_index", 0))
        if grid_index <= 0:
            return

        painter.save()
        painter.setPen(QPen(QColor(255, 255, 255, 150), 1, Qt.DashLine))

        if grid_index == 1:
            thirds_x = [rect.left() + rect.width() / 3.0, rect.left() + 2.0 * rect.width() / 3.0]
            thirds_y = [rect.top() + rect.height() / 3.0, rect.top() + 2.0 * rect.height() / 3.0]
            for x_val in thirds_x:
                painter.drawLine(QPointF(x_val, rect.top()), QPointF(x_val, rect.bottom()))
            for y_val in thirds_y:
                painter.drawLine(QPointF(rect.left(), y_val), QPointF(rect.right(), y_val))
        elif grid_index == 2:
            phi = 0.61803398875
            gx = [rect.left() + rect.width() * (1.0 - phi), rect.left() + rect.width() * phi]
            gy = [rect.top() + rect.height() * (1.0 - phi), rect.top() + rect.height() * phi]
            for x_val in gx:
                painter.drawLine(QPointF(x_val, rect.top()), QPointF(x_val, rect.bottom()))
            for y_val in gy:
                painter.drawLine(QPointF(rect.left(), y_val), QPointF(rect.right(), y_val))

        painter.restore()

    def _build_bn_overlay(self, width: int, height: int) -> QPixmap:
        overlay = QPixmap(width, height)
        overlay.fill(Qt.transparent)
        painter = QPainter(overlay)
        x, y, w, h = self._bn_overlay_rect
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        w = max(1, min(width - x, int(w)))
        h = max(1, min(height - y, int(h)))
        rect = QRectF(float(x), float(y), float(w), float(h))

        shade = QColor(0, 0, 0, 120)
        painter.fillRect(QRectF(0.0, 0.0, float(width), max(0.0, rect.top())), shade)
        painter.fillRect(QRectF(0.0, rect.bottom(), float(width), max(0.0, float(height) - rect.bottom())), shade)
        painter.fillRect(QRectF(0.0, rect.top(), max(0.0, rect.left()), rect.height()), shade)
        painter.fillRect(QRectF(rect.right(), rect.top(), max(0.0, float(width) - rect.right()), rect.height()), shade)

        painter.setPen(QPen(QColor("#4cff6f"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        painter.end()
        return overlay

    def _apply_mask_blend(self, base_img, layer_img, layer_mask):
        try:
            if base_img is None or layer_img is None:
                return base_img
            if layer_mask is None:
                return layer_img.copy()
            mask = layer_mask.astype(np.float32)
            if mask.ndim == 2:
                mask = mask[:, :, None]
            mask = np.clip(mask, 0.0, 1.0)
            return np.clip(base_img.astype(np.float32) * (1.0 - mask) + layer_img.astype(np.float32) * mask, 0, 255).astype(np.uint8)
        except Exception:
            return layer_img.copy() if isinstance(layer_img, np.ndarray) else base_img

    def _build_color_corrections_label(self):
        return "Corrections"

    def _update_models_label(self):
        return

    def cancel_params_preview(self):
        self.preview_override_img = None
        if hasattr(self, "params_preview_timer") and self.params_preview_timer.isActive():
            self.params_preview_timer.stop()
        if hasattr(self, "corrections_timer") and self.corrections_timer.isActive():
            self.corrections_timer.stop()

    def _finalize_camera_raw_hsl(self):
        self.preview_override_img = None
        self.apply_full_processing()

    def add_thumbnail(self, label, image):
        if image is None:
            return
        if not hasattr(self, "processing_history"):
            self.processing_history = []
        self.processing_history.append((str(label), image.copy() if isinstance(image, np.ndarray) else image))
        max_items = int(getattr(self, "max_thumbnails", 15) or 15)
        if len(self.processing_history) > max_items:
            self.processing_history = self.processing_history[-max_items:]

    def update_menu_actions(self):
        has_image = isinstance(getattr(self, "magic_img", None), np.ndarray)
        for action_name in ("action_save", "action_save_as", "action_undo", "action_redo"):
            action = getattr(self, action_name, None)
            if action is not None:
                action.setEnabled(has_image)

    def undo(self):
        if not hasattr(self, "undo_stack") or not self.undo_stack:
            return
        if isinstance(self.magic_img, np.ndarray):
            self.redo_stack.append(self.magic_img.copy())
        self.magic_img = self.undo_stack.pop()
        self.apply_full_processing()

    def redo(self):
        if not hasattr(self, "redo_stack") or not self.redo_stack:
            return
        if isinstance(self.magic_img, np.ndarray):
            self.undo_stack.append(self.magic_img.copy())
        self.magic_img = self.redo_stack.pop()
        self.apply_full_processing()

    def save_image_to_path(self, path):
        if self.processed_img is None or not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            path = f"{path}.png"
        ok, encoded = cv2.imencode(os.path.splitext(path)[1] or ".png", self.processed_img)
        if not ok:
            self.log("Save failed: encoder error.", "error")
            return
        encoded.tofile(path)
        self.current_save_path = path
        self.log(f"Saved: {os.path.basename(path)}", "success")

    def save_image_as(self):
        if self.processed_img is None:
            self.log("Save As skipped: no image loaded.", "warning")
            return
        start_dir = self._get_default_dialog_directory()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
            options=get_safe_file_dialog_options(),
        )
        self.save_image_to_path(str(path or "").strip())

    def save_image(self):
        if self.processed_img is None:
            self.log("Save skipped: no image loaded.", "warning")
            return
        if getattr(self, "current_save_path", None):
            self.save_image_to_path(self.current_save_path)
        else:
            self.save_image_as()

    def run_star_shrink(self):
        if self.magic_img is None:
            self.log("Star Shrink skipped: no image loaded.")
            return

        source_img = self.magic_img.copy()
        source_pix = np_to_qpixmap(source_img)
        previous_processed_img = None if self.processed_img is None else self.processed_img.copy()
        previous_after_pix = None if previous_processed_img is None else np_to_qpixmap(previous_processed_img)
        last_preview = {"params": None, "magic_img": None, "processed_img": None}

        def preview_star_shrink(amount, selection):
            shrunk = star_shrink_pixinsight(source_img, amount=amount, selection=selection)
            stars_mask = self.layer_masks.get("stars")
            self.magic_img = self._apply_mask_blend(source_img, shrunk, stars_mask) if stars_mask is not None else shrunk
            self.levels_window.levels_widget.set_image(self.magic_img)
            self.preview_override_img = self.magic_img.copy()
            self.viewer.set_before(source_pix)
            self.apply_full_processing()
            last_preview["params"] = (amount, selection)
            last_preview["magic_img"] = self.magic_img.copy()
            last_preview["processed_img"] = None if self.processed_img is None else self.processed_img.copy()

        dialog = self._get_star_shrink_dialog()
        dialog.set_preview_callback(preview_star_shrink)
        self._star_shrink_context = {
            "source_img": source_img,
            "source_pix": source_pix,
            "previous_processed_img": previous_processed_img,
            "previous_after_pix": previous_after_pix,
            "last_preview": last_preview,
            "preview_star_shrink": preview_star_shrink,
        }
        self._begin_dialog_compare(dialog, auto_end=False)
        self.log("Star Shrink dialog opened.")
        amount, selection = dialog.get_values()
        preview_star_shrink(amount, selection)
        dialog.exec_()

    def _apply_mask_blend(self, base_img: np.ndarray, layer_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if base_img is None:
            return layer_img.copy() if isinstance(layer_img, np.ndarray) else base_img
        if layer_img is None:
            return base_img.copy()
        if mask is None:
            return layer_img.copy()

        h, w = base_img.shape[:2]
        layer = layer_img
        if layer.shape[:2] != (h, w):
            layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_LINEAR)

        alpha = mask.astype(np.float32)
        if alpha.shape[:2] != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        if alpha.ndim == 3:
            alpha = alpha[:, :, 0]
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha3 = alpha[:, :, np.newaxis]
        blended = base_img.astype(np.float32) * (1.0 - alpha3) + layer.astype(np.float32) * alpha3
        return np.clip(blended, 0, 255).astype(np.uint8)

    def on_magic_finished(self, result):
        if hasattr(self, "progress_dialog") and self.progress_dialog is not None:
            self.progress_dialog.close()
        if result is None:
            self.log("Magic Filter failed: empty result.", "error")
            return
        self.magic_img = result
        self.levels_window.levels_widget.set_image(self.magic_img)
        self.viewer.set_before(np_to_qpixmap(self.magic_img))
        self.apply_full_processing()
        self.add_thumbnail("Magic Filter", self.processed_img)
        self.update_menu_actions()
        self.log("Magic Filter finished.")

    def on_starnet_finished(self, result, error_message):
        if hasattr(self, "progress_dialog") and self.progress_dialog is not None:
            self.progress_dialog.close()
        if error_message:
            self.log(error_message, "error")
            return
        if result is None:
            self.log("StarNet++ failed: empty result.", "error")
            return
        if hasattr(self, "starnet_source_img") and self.starnet_source_img is not None:
            self.undo_stack.append(self.starnet_source_img.copy())
            self.starnet_source_img = None
        self.redo_stack.clear()
        self.magic_img = result
        self.levels_window.levels_widget.set_image(self.magic_img)
        self.add_layer("stars", self.magic_img)
        self.viewer.set_before(np_to_qpixmap(self.magic_img))
        self.apply_full_processing()
        self.add_thumbnail("StarNet++", self.processed_img)
        self.update_menu_actions()
        self.log("StarNet++ finished. Starless image applied.", "success")

    def on_params_changed(self):
        if self.magic_img is None:
            return
        self.apply_preview_processing()
        self.params_preview_timer.start()
        self.corrections_timer.stop()
        self.corrections_timer.start()

    def cancel_params_preview(self):
        if hasattr(self, "params_preview_timer") and self.params_preview_timer.isActive():
            self.params_preview_timer.stop()

    def _update_models_label(self):
        if self.preferences_dialog is not None and hasattr(self.preferences_dialog, "refresh_select_paths"):
            self.preferences_dialog.refresh_select_paths()

    def save_image(self):
        if self.processed_img is None:
            self.log("Save skipped: no processed image.", "warning")
            return
        if self.current_save_path is None:
            self.save_image_as()
            return
        self.save_image_to_path(self.current_save_path)

    def save_image_as(self):
        if self.processed_img is None:
            self.log("Save As skipped: no processed image.", "warning")
            return
        path, selected_filter = self._show_save_file_dialog(
            "Save Image As",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;FITS (*.fits *.fit *.fts);;All Files (*)",
            "",
        )
        path = (path or "").strip()
        if not path:
            self.log("Save As canceled.", "warning")
            return

        path_obj = Path(path)
        if path_obj.suffix == "":
            selected_filter_lower = (selected_filter or "").lower()
            ext = ".png"
            if "fits" in selected_filter_lower or "fit" in selected_filter_lower or "fts" in selected_filter_lower:
                ext = ".fits"
            elif "tif" in selected_filter_lower:
                ext = ".tif"
            elif "jpeg" in selected_filter_lower or "jpg" in selected_filter_lower:
                ext = ".jpg"
            path = str(path_obj.with_suffix(ext))

        self.save_image_to_path(path)

    def save_image_to_path(self, path: str):
        if self.processed_img is None:
            self.log("Save skipped: no processed image.", "warning")
            return
        normalized_path = os.path.expanduser((path or "").strip())
        if not normalized_path:
            self.log("Save skipped: path is empty.", "warning")
            return

        self.current_save_path = normalized_path
        try:
            if _is_fits_path(self.current_save_path):
                if not safe_fits_write(self.current_save_path, self.processed_img):
                    raise RuntimeError("Failed to write FITS image.")
            else:
                if not cv2.imwrite(self.current_save_path, self.processed_img):
                    raise RuntimeError("Failed to write image file.")
        except Exception as exc:
            self.log(f"Save failed: {self.current_save_path} ({exc})", "error")
            return
        self.log(f"Image saved as: {self.current_save_path}", "success")

    def _restore_magic_state(self, img):
        self.magic_img = img.copy()
        self.levels_window.levels_widget.set_image(self.magic_img)
        self.viewer.set_before(np_to_qpixmap(self.magic_img))
        self.apply_full_processing()
        self._sync_selected_thumbnail_with_current_state()
        self.update_menu_actions()

    def _history_depth_for_node(self, node, nodes_by_id, cache):
        node_id = node.get("id") if isinstance(node, dict) else None
        if node_id in cache:
            return cache[node_id]
        parent_id = node.get("parent_id") if isinstance(node, dict) else None
        if parent_id is None or parent_id not in nodes_by_id:
            cache[node_id] = 0
            return 0
        depth = 1 + self._history_depth_for_node(nodes_by_id[parent_id], nodes_by_id, cache)
        cache[node_id] = depth
        return depth

    def undo(self):
        if not self.undo_stack:
            self.log("Undo skipped: history is empty.")
            return
        if self.magic_img is not None:
            self.redo_stack.append(self.magic_img.copy())
        self._restore_magic_state(self.undo_stack.pop())
        self.log("Undo", "success")

    def redo(self):
        if not self.redo_stack:
            self.log("Redo skipped: history is empty.")
            return
        if self.magic_img is not None:
            self.undo_stack.append(self.magic_img.copy())
        self._restore_magic_state(self.redo_stack.pop())
        self.log("Redo", "success")

    def _sync_selected_thumbnail_with_current_state(self):
        if not self.processing_history:
            self.selected_thumbnail_index = -1
            self.current_history_node_id = None
            return

        selected_index = None
        for idx in range(len(self.processing_history) - 1, -1, -1):
            entry = self.processing_history[idx]
            magic = entry.get("magic_img") if isinstance(entry, dict) else None
            if not isinstance(magic, np.ndarray) or not isinstance(self.magic_img, np.ndarray):
                continue
            if magic.shape != self.magic_img.shape:
                continue
            if np.array_equal(magic, self.magic_img):
                selected_index = idx
                break

        if selected_index is None:
            selected_index = len(self.processing_history) - 1

        self.selected_thumbnail_index = selected_index
        selected_entry = self.processing_history[selected_index]
        self.current_history_node_id = selected_entry.get("id") if isinstance(selected_entry, dict) else None
        self._update_thumbnails_view()

    def add_thumbnail(self, operation_name: str, img: np.ndarray = None):
        if img is None:
            img = self.processed_img if self.processed_img is not None else self.magic_img
        if img is None:
            return
        if not hasattr(self, "processing_history"):
            self.processing_history = []

        valid_ids = {
            entry.get("id")
            for entry in self.processing_history
            if isinstance(entry, dict) and entry.get("id") is not None
        }

        parent_id = None

        selected_index = int(getattr(self, "selected_thumbnail_index", -1))
        if 0 <= selected_index < len(self.processing_history):
            selected_entry = self.processing_history[selected_index]
            if isinstance(selected_entry, dict):
                selected_id = selected_entry.get("id")
                if selected_id in valid_ids:
                    parent_id = selected_id

        if parent_id is None:
            current_id = getattr(self, "current_history_node_id", None)
            if current_id in valid_ids:
                parent_id = current_id

        if parent_id is None and self.processing_history:
            tail = self.processing_history[-1]
            parent_id = tail.get("id") if isinstance(tail, dict) else None

        node_id = int(getattr(self, "thumbnail_next_id", 1))
        self.processing_history.append(
            {
                "id": node_id,
                "parent_id": parent_id,
                "name": operation_name,
                "img": img.copy(),
                "magic_img": self.magic_img.copy() if self.magic_img is not None else None,
            }
        )
        self.thumbnail_next_id = int(getattr(self, "thumbnail_next_id", 1)) + 1
        self.processing_history = self.processing_history[-self.max_thumbnails :]
        valid_ids = {entry.get("id") for entry in self.processing_history if isinstance(entry, dict)}
        for entry in self.processing_history:
            if not isinstance(entry, dict):
                continue
            parent = entry.get("parent_id")
            if parent is not None and parent not in valid_ids:
                entry["parent_id"] = None
        self.current_history_node_id = node_id
        self.selected_thumbnail_index = len(self.processing_history) - 1
        self._update_thumbnails_view()

    def _on_thumbnail_clicked(self, index: int):
        if index < 0 or index >= len(self.processing_history):
            return
        item = self.processing_history[index]
        magic = item.get("magic_img")
        if isinstance(magic, np.ndarray):
            self.current_history_node_id = item.get("id")
            self.magic_img = magic.copy()
            self.levels_window.levels_widget.set_image(self.magic_img)
            self.apply_full_processing()
            self.update_menu_actions()
            self.selected_thumbnail_index = index
            self._update_thumbnails_view()

    def _on_thumbnail_action_requested(self, index: int, global_pos: QPoint):
        if index < 0 or index >= len(self.processing_history):
            return

        self.selected_thumbnail_index = int(index)
        item = self.processing_history[index]
        if isinstance(item, dict):
            self.current_history_node_id = item.get("id")

        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        remove_action = menu.addAction("Remove")
        selected_action = menu.exec_(global_pos)

        if selected_action == rename_action:
            self.rename_thumbnail(index)
        elif selected_action == remove_action:
            self.remove_thumbnail(index)

    def rename_thumbnail(self, index: int):
        if index < 0 or index >= len(self.processing_history):
            return
        entry = self.processing_history[index]
        if not isinstance(entry, dict):
            return

        current_name = str(entry.get("name") or "Step")
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Thumbnail",
            "Name:",
            QLineEdit.Normal,
            current_name,
        )
        if not ok:
            return

        new_name = str(new_name or "").strip()
        if not new_name:
            return

        entry["name"] = new_name
        self._update_thumbnails_view()
        self.log(f"Thumbnail renamed: {new_name}", "success")

    def remove_thumbnail(self, index: int):
        if index < 0 or index >= len(self.processing_history):
            return
        entry = self.processing_history[index]
        if not isinstance(entry, dict):
            return

        root_id = entry.get("id")
        if root_id is None:
            return

        nodes_by_id = {
            item.get("id"): item
            for item in self.processing_history
            if isinstance(item, dict) and item.get("id") is not None
        }

        def _has_ancestor(node_id, ancestor_id):
            seen = set()
            current_id = node_id
            while current_id is not None and current_id in nodes_by_id and current_id not in seen:
                seen.add(current_id)
                if current_id == ancestor_id:
                    return True
                current_node = nodes_by_id.get(current_id)
                if not isinstance(current_node, dict):
                    return False
                current_id = current_node.get("parent_id")
            return False

        ids_to_remove = {
            node_id
            for node_id in nodes_by_id
            if _has_ancestor(node_id, root_id)
        }

        self.processing_history = [
            item
            for item in self.processing_history
            if not (isinstance(item, dict) and item.get("id") in ids_to_remove)
        ]

        if not self.processing_history:
            self.selected_thumbnail_index = -1
            self.current_history_node_id = None
            self._update_thumbnails_view()
            self.log("Thumbnail removed.", "success")
            return

        self.selected_thumbnail_index = max(0, min(index - 1, len(self.processing_history) - 1))
        selected_entry = self.processing_history[self.selected_thumbnail_index]
        self.current_history_node_id = selected_entry.get("id") if isinstance(selected_entry, dict) else None
        selected_magic = selected_entry.get("magic_img") if isinstance(selected_entry, dict) else None
        if isinstance(selected_magic, np.ndarray):
            self.magic_img = selected_magic.copy()
            self.levels_window.levels_widget.set_image(self.magic_img)
            self.apply_full_processing()
            self.update_menu_actions()
        self._update_thumbnails_view()
        self.log("Thumbnail removed.", "success")

    def _update_thumbnails_view(self):
        while self.thumbnails_layout.count() > 0:
            item = self.thumbnails_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        widgets_by_id = {}
        nodes_by_id = {item.get("id"): item for item in self.processing_history if isinstance(item, dict)}

        children_by_parent = {}
        for history_item in self.processing_history:
            if not isinstance(history_item, dict):
                continue
            parent_id = history_item.get("parent_id")
            node_id = history_item.get("id")
            if node_id is None:
                continue
            children_by_parent.setdefault(parent_id, []).append(node_id)

        lane_by_id = {}
        next_lane = 0

        for idx, history_item in enumerate(self.processing_history):
            if isinstance(history_item, dict):
                node_id = history_item.get("id")
                parent_id = history_item.get("parent_id")
            else:
                node_id = None
                parent_id = None

            lane = 0
            if node_id is not None:
                parent_lane = lane_by_id.get(parent_id)
                if parent_lane is None:
                    lane = next_lane
                    next_lane += 1
                else:
                    siblings = children_by_parent.get(parent_id, [])
                    sibling_index = siblings.index(node_id) if node_id in siblings else 0
                    if sibling_index == 0:
                        lane = parent_lane
                    else:
                        lane = next_lane
                        next_lane += 1
                lane_by_id[node_id] = lane

            thumb_widget = ClickableThumbnailWidget(
                idx,
                history_item,
                selected=(idx == self.selected_thumbnail_index),
            )
            thumb_widget.thumbnail_action_requested.connect(self._on_thumbnail_action_requested)
            if node_id is not None:
                widgets_by_id[node_id] = thumb_widget
            col = idx
            self.thumbnails_layout.addWidget(thumb_widget, lane, col)

        self.thumbnails_container.set_branch_data(self.processing_history, widgets_by_id)

    def update_menu_actions(self):
        has_img = self.original_img is not None
        if hasattr(self, "action_save"):
            self.action_save.setEnabled(self.processed_img is not None)
        if hasattr(self, "action_save_as"):
            self.action_save_as.setEnabled(self.processed_img is not None)
        for action_name in (
            "action_magic_filter",
            "action_star_shrink",
            "action_plate_solve",
            "action_starnet",
            "action_deepsnr",
            "action_3d_fly",
            "action_color_calibration",
            "action_blur",
            "action_rotate",
            "action_crop",
            "action_paste_layer",
        ):
            action = getattr(self, action_name, None)
            if action is not None:
                action.setEnabled(has_img)
        if hasattr(self, "action_delete_layer"):
            self.action_delete_layer.setEnabled(self.can_delete_layer(getattr(self, "selected_layer_key", None)))
        if hasattr(self, "action_undo"):
            self.action_undo.setEnabled(len(self.undo_stack) > 0)
        if hasattr(self, "action_redo"):
            self.action_redo.setEnabled(len(self.redo_stack) > 0)


class ThumbnailTreeContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._branch_nodes = []
        self._widget_by_id = {}

    def set_branch_data(self, nodes, widget_by_id):
        self._branch_nodes = list(nodes or [])
        self._widget_by_id = dict(widget_by_id or {})
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._branch_nodes or not self._widget_by_id:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(70, 130, 180, 170))
        pen.setWidth(2)
        painter.setPen(pen)

        nodes_by_id = {
            item.get("id"): item
            for item in self._branch_nodes
            if isinstance(item, dict)
        }
        for node_id, node in nodes_by_id.items():
            parent_id = node.get("parent_id")
            if parent_id is None or parent_id not in nodes_by_id:
                continue

            parent_widget = self._widget_by_id.get(parent_id)
            child_widget = self._widget_by_id.get(node_id)
            if parent_widget is None or child_widget is None:
                continue

            parent_rect = parent_widget.geometry()
            child_rect = child_widget.geometry()
            start = QPoint(parent_rect.right(), parent_rect.center().y())
            end = QPoint(child_rect.left(), child_rect.center().y())
            mid_x = (start.x() + end.x()) // 2
            painter.drawLine(start, QPoint(mid_x, start.y()))
            painter.drawLine(QPoint(mid_x, start.y()), QPoint(mid_x, end.y()))
            painter.drawLine(QPoint(mid_x, end.y()), end)


class ClickableThumbnailWidget(QWidget):
    thumbnail_clicked = pyqtSignal(int)
    thumbnail_action_requested = pyqtSignal(int, QPoint)

    def __init__(self, index: int, history_item: dict, selected: bool = False, parent=None):
        super().__init__(parent)
        self.index = int(index)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(120, 74)
        image = history_item.get("img") if isinstance(history_item, dict) else None
        if isinstance(image, np.ndarray):
            pix = np_to_qpixmap(image).scaled(120, 74, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.thumb_label.setPixmap(pix)
        title = str((history_item or {}).get("name", "Step"))
        self.text_label = QLabel(title)
        self.text_label.setWordWrap(True)
        self.text_label.setMaximumWidth(120)
        layout.addWidget(self.thumb_label)
        layout.addWidget(self.text_label)
        self.setStyleSheet(
            "QWidget { background: #1f2933; border: 1px solid #00bcd4; border-radius: 4px; }"
            if selected
            else "QWidget { background: transparent; border: 1px solid #333; border-radius: 4px; }"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.thumbnail_action_requested.emit(self.index, self.mapToGlobal(event.pos()))
            event.accept()
            return
        super().mousePressEvent(event)


def main():
    if sys.platform.startswith("linux"):
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, True)
    app = QApplication(sys.argv)
    apply_theme_application(app, "Fusion Dark")
    window = AstroApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
