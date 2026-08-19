from __future__ import annotations

import math

import cv2
import numpy as np


class ColorProcessor:
    _HSL_COLOR_BANDS = {
        "Reds": (0.0, 22.5),
        "Oranges": (30.0, 22.5),
        "Yellows": (60.0, 22.5),
        "Greens": (120.0, 35.0),
        "Aquas": (180.0, 30.0),
        "Blues": (240.0, 35.0),
        "Purples": (280.0, 22.5),
        "Magentas": (320.0, 22.5),
    }

    @staticmethod
    def _to_float01(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("image cannot be None")
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("image must have shape (H, W, 3)")

        arr = arr[:, :, :3]
        if np.issubdtype(arr.dtype, np.integer):
            maxv = float(np.iinfo(arr.dtype).max)
            out = arr.astype(np.float32) / max(1.0, maxv)
        else:
            out = arr.astype(np.float32, copy=False)

        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _clip01(image: np.ndarray) -> np.ndarray:
        return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _gaussian_mask_for_hue(hue_deg: np.ndarray, center_deg: float, sigma_deg: float) -> np.ndarray:
        delta = np.abs(((hue_deg - center_deg + 180.0) % 360.0) - 180.0)
        sigma = max(1.0, float(sigma_deg))
        return np.exp(-0.5 * (delta / sigma) ** 2).astype(np.float32)

    @staticmethod
    def _ghs_curve_values(
        x: np.ndarray,
        stretch_factor: float,
        symmetry_point: float,
        highlight_protection_point: float,
        shadow_protection_point: float,
    ) -> np.ndarray:
        xv = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
        s = float(np.clip(stretch_factor, 0.0, 20.0))
        sp = float(np.clip(symmetry_point, 0.0, 1.0))
        hp = float(np.clip(highlight_protection_point, 0.0, 1.0))
        sh = float(np.clip(shadow_protection_point, 0.0, 1.0))
        if sh > hp:
            sh, hp = hp, sh

        if s <= 1e-6:
            base = xv
        else:
            k = 1.0 + 8.0 * s
            centered = xv - sp
            base = 1.0 / (1.0 + np.exp(-k * centered))
            spv = 1.0 / (1.0 + math.exp(0.0))
            base = (base - spv) / max(1e-6, (1.0 - spv))
            base = np.clip(base, 0.0, 1.0)

        if sh > 0.0:
            t = np.clip(xv / max(1e-6, sh), 0.0, 1.0)
            shadow_blend = t * t * (3.0 - 2.0 * t)
            base = xv * (1.0 - shadow_blend) + base * shadow_blend

        if hp < 1.0:
            t = np.clip((xv - hp) / max(1e-6, 1.0 - hp), 0.0, 1.0)
            high_blend = t * t * (3.0 - 2.0 * t)
            base = base * (1.0 - high_blend) + xv * high_blend

        return np.clip(base, 0.0, 1.0).astype(np.float32)

    @classmethod
    def _apply_ghs(cls, image: np.ndarray, ghs_params: dict | None) -> np.ndarray:
        if not isinstance(ghs_params, dict):
            return image

        stretch_factor = float(max(0.0, ghs_params.get("stretch_factor", 0.0)))
        if stretch_factor <= 1e-6:
            return image

        symmetry_point = float(np.clip(ghs_params.get("symmetry_point", 0.5), 0.0, 1.0))
        highlight_point = float(np.clip(ghs_params.get("highlight_protection_point", 0.95), 0.0, 1.0))
        shadow_point = float(np.clip(ghs_params.get("shadow_protection_point", 0.05), 0.0, 1.0))
        model = str(ghs_params.get("colour_stretch_model", "independent_channel_values"))

        if model == "luminance_only":
            gray = (
                image[:, :, 0] * 0.114
                + image[:, :, 1] * 0.587
                + image[:, :, 2] * 0.299
            ).astype(np.float32)
            mapped = cls._ghs_curve_values(
                gray,
                stretch_factor,
                symmetry_point,
                highlight_point,
                shadow_point,
            )
            ratio = mapped / np.maximum(gray, 1e-5)
            return cls._clip01(image * ratio[:, :, None])

        return cls._ghs_curve_values(
            image,
            stretch_factor,
            symmetry_point,
            highlight_point,
            shadow_point,
        )

    @staticmethod
    def _apply_hsl(image: np.ndarray, hsl_params: dict | None) -> np.ndarray:
        if not isinstance(hsl_params, dict):
            return image

        hsv = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hue = hsv[:, :, 0] * 2.0
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0

        hue_cfg = hsl_params.get("Hue", {}) if isinstance(hsl_params.get("Hue", {}), dict) else {}
        sat_cfg = hsl_params.get("Saturation", {}) if isinstance(hsl_params.get("Saturation", {}), dict) else {}
        lum_cfg = hsl_params.get("Luminance", {}) if isinstance(hsl_params.get("Luminance", {}), dict) else {}

        for name, (center, spread) in ColorProcessor._HSL_COLOR_BANDS.items():
            mask = ColorProcessor._gaussian_mask_for_hue(hue, center, spread)

            hue_shift = float(hue_cfg.get(name, 0.0))
            if abs(hue_shift) > 1e-6:
                hue = (hue + mask * (hue_shift * 0.5)) % 360.0

            sat_delta = float(sat_cfg.get(name, 0.0)) / 100.0
            if abs(sat_delta) > 1e-6:
                sat = np.clip(sat * (1.0 + sat_delta * mask), 0.0, 1.0)

            lum_delta = float(lum_cfg.get(name, 0.0)) / 100.0
            if abs(lum_delta) > 1e-6:
                val = np.clip(val * (1.0 + lum_delta * mask), 0.0, 1.0)

        hsv[:, :, 0] = np.clip(hue / 2.0, 0.0, 179.0)
        hsv[:, :, 1] = sat * 255.0
        hsv[:, :, 2] = val * 255.0
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        return ColorProcessor._clip01(bgr)

    @classmethod
    def apply_camera_raw_and_hsl(
        cls,
        image: np.ndarray,
        basic_params: dict | None,
        hsl_params: dict | None,
        ghs_params: dict | None = None,
    ) -> np.ndarray:
        img = cls._to_float01(image)
        params = basic_params if isinstance(basic_params, dict) else {}

        exposure = float(params.get("exposure", 0.0))
        if abs(exposure) > 1e-6:
            img = img * (2.0 ** exposure)

        contrast = float(params.get("contrast", 0.0))
        if abs(contrast) > 1e-6:
            k = 1.0 + contrast
            img = (img - 0.5) * k + 0.5

        temperature = float(params.get("temperature", 0.0)) / 100.0
        tint = float(params.get("tint", 0.0)) / 100.0
        if abs(temperature) > 1e-6 or abs(tint) > 1e-6:
            img[:, :, 2] *= 1.0 + 0.10 * temperature
            img[:, :, 0] *= 1.0 - 0.10 * temperature
            img[:, :, 1] *= 1.0 + 0.08 * tint

        for key, ch in (("calibration_blue", 0), ("calibration_green", 1), ("calibration_red", 2)):
            delta = float(params.get(key, 0.0))
            if abs(delta) > 1e-6:
                img[:, :, ch] *= 1.0 + delta

        img = cls._clip01(img)

        hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = hsv[:, :, 1] / 255.0

        saturation = float(params.get("saturation", 0.0))
        vibrance = float(params.get("vibrance", 0.0))
        if abs(saturation) > 1e-6:
            sat = np.clip(sat * (1.0 + saturation), 0.0, 1.0)
        if abs(vibrance) > 1e-6:
            sat = np.clip(sat + (1.0 - sat) * vibrance * 0.7, 0.0, 1.0)

        hsv[:, :, 1] = sat * 255.0
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

        texture = float(params.get("texture", 0.0))
        clarity = float(params.get("clarity", 0.0))
        if abs(texture) > 1e-6 or abs(clarity) > 1e-6:
            blur_small = cv2.GaussianBlur(img, (0, 0), 1.2)
            blur_large = cv2.GaussianBlur(img, (0, 0), 5.0)
            detail = img - blur_small
            midtone = img - blur_large
            img = img + detail * (0.6 * texture) + midtone * (0.4 * clarity)

        dehaze = float(params.get("dehaze", 0.0))
        if abs(dehaze) > 1e-6:
            haze = cv2.GaussianBlur(img, (0, 0), 12.0)
            img = img + (img - haze) * (0.9 * dehaze)

        noise_reduction = float(params.get("noise_reduction", 0.0))
        if noise_reduction > 1e-6:
            denoise_strength = int(np.clip(round(3 + noise_reduction * 17), 3, 20))
            img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            img_u8 = cv2.fastNlMeansDenoisingColored(
                img_u8,
                None,
                h=denoise_strength,
                hColor=max(3, denoise_strength // 2),
                templateWindowSize=7,
                searchWindowSize=21,
            )
            img = img_u8.astype(np.float32) / 255.0

        img = cls._apply_hsl(cls._clip01(img), hsl_params)
        img = cls._apply_ghs(cls._clip01(img), ghs_params)
        return cls._clip01(img)
