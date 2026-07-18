"""Przetwarzanie kolorow obrazu (Camera Raw + HSL).

Ten modul zostal wyciagniety z pliku "Astro Ai Plus.py" jako przyklad
refactoru - patrz TODO.md.

Klasa ColorProcessor nie zalezy od Qt ani od reszty aplikacji - potrzebuje
tylko numpy i OpenCV. Dzieki temu mozna ja latwo testowac (patrz
tests/test_color_processor.py).
"""

import numpy as np
import cv2


class ColorProcessor:
    COLOR_RANGES = {
        "Reds":      [(0, 10), (170, 179)],
        "Oranges":   [(11, 25)],
        "Yellows":   [(26, 35)],
        "Greens":    [(36, 85)],
        "Aquas":     [(86, 100)],
        "Blues":     [(101, 130)],
        "Purples":   [(131, 150)],
        "Magentas":  [(151, 169)],
    }

    @staticmethod
    def _fast_noise_reduction(img: np.ndarray, amount: float) -> np.ndarray:
        if amount <= 0.0:
            return img

        # Keep denoising interactive by filtering only the luma channel and
        # optionally working on a smaller intermediate when the source is large.
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        max_dim = max(img_u8.shape[:2])
        scale = 1.0
        if max_dim >= 2600:
            scale = 0.5
        elif max_dim >= 1600:
            scale = 0.75

        diameter = 5 if amount < 0.5 else 7
        sigma = 10.0 + amount * 35.0

        if scale < 1.0:
            small_l = cv2.resize(l, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            filtered_small = cv2.bilateralFilter(small_l, diameter, sigma, sigma)
            l_filtered = cv2.resize(filtered_small, (l.shape[1], l.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            l_filtered = cv2.bilateralFilter(l, diameter, sigma, sigma)

        if amount > 0.6:
            chroma_sigma = 0.5 + amount * 0.8
            a = cv2.GaussianBlur(a, (0, 0), chroma_sigma)
            b = cv2.GaussianBlur(b, (0, 0), chroma_sigma)

        denoised = cv2.cvtColor(cv2.merge([l_filtered, a, b]), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
        blend = 0.35 + amount * 0.65
        return np.clip(img * (1.0 - blend) + denoised * blend, 0, 1)

    @staticmethod
    def apply_camera_raw_and_hsl(
        base_img: np.ndarray,
        basic_params: dict,
        hsl_params: dict,
    ) -> np.ndarray:
        if base_img is None:
            return None

        img = base_img.astype(np.float32) / 255.0

        temp = basic_params.get("temperature", 0.0)
        tint = basic_params.get("tint", 0.0)
        calibration_red = basic_params.get("calibration_red", 0.0)
        calibration_green = basic_params.get("calibration_green", 0.0)
        calibration_blue = basic_params.get("calibration_blue", 0.0)
        exposure = basic_params.get("exposure", 0.0)
        contrast = basic_params.get("contrast", 0.0)
        saturation = basic_params.get("saturation", 0.0)
        vibrance = basic_params.get("vibrance", 0.0)
        texture = basic_params.get("texture", 0.0)
        dehaze = basic_params.get("dehaze", 0.0)
        clarity = basic_params.get("clarity", 0.0)
        noise_reduction = basic_params.get("noise_reduction", 0.0)

        b, g, r = cv2.split(img)

        t_scale = temp / 100.0
        r *= (1.0 + 0.5 * t_scale)
        b *= (1.0 - 0.5 * t_scale)

        ti_scale = tint / 100.0
        g *= (1.0 + 0.5 * ti_scale)
        r *= (1.0 - 0.25 * ti_scale)
        b *= (1.0 - 0.25 * ti_scale)

        r *= (1.0 + calibration_red / 200.0)
        g *= (1.0 + calibration_green / 200.0)
        b *= (1.0 + calibration_blue / 200.0)

        img = cv2.merge([b, g, r])
        img = np.clip(img, 0, 1)

        if noise_reduction > 0.0:
            img = ColorProcessor._fast_noise_reduction(img, noise_reduction)

        if dehaze != 0.0:
            haze_gain = 1.0 + dehaze * 0.45
            img = np.clip((img - 0.5) * haze_gain + 0.5, 0, 1)

        img *= 2.0 ** exposure
        img = (img - 0.5) * (1.0 + contrast) + 0.5
        img = np.clip(img, 0, 1)

        if texture != 0.0 or clarity != 0.0:
            local_blur = cv2.GaussianBlur(img, (0, 0), 3.0)
            detail = img - local_blur
            img = img + detail * texture
            img = img + detail * clarity * 0.5
            img = np.clip(img, 0, 1)

        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        s = s.astype(np.float32)
        s *= (1.0 + saturation)
        s = np.clip(s, 0, 255)

        vibr_factor = 1.0 + vibrance
        s = s + (255.0 - s) * (vibr_factor - 1.0)
        s = np.clip(s, 0, 255)

        hue_dict = hsl_params.get("Hue", {})
        sat_dict = hsl_params.get("Saturation", {})
        lum_dict = hsl_params.get("Luminance", {})

        h = h.astype(np.int16)
        v = v.astype(np.int16)

        for color_name, ranges in ColorProcessor.COLOR_RANGES.items():
            hue_shift_val = hue_dict.get(color_name, 0)
            sat_shift_val = sat_dict.get(color_name, 0)
            lum_shift_val = lum_dict.get(color_name, 0)

            if hue_shift_val == 0 and sat_shift_val == 0 and lum_shift_val == 0:
                continue

            mask = np.zeros_like(h, dtype=bool)
            for (hmin, hmax) in ranges:
                if hmin <= hmax:
                    mask |= (h >= hmin) & (h <= hmax)
                else:
                    mask |= (h >= hmin) | (h <= hmax)

            if not np.any(mask):
                continue

            hue_shift = int(hue_shift_val / 3)
            h[mask] = (h[mask] + hue_shift) % 180

            sat_scale = 1.0 + (sat_shift_val / 100.0)
            s[mask] = np.clip(s[mask] * sat_scale, 0, 255)

            lum_shift = int(lum_shift_val * 0.8)
            v[mask] = np.clip(v[mask] + lum_shift, 0, 255)

        h = np.clip(h, 0, 179).astype(np.uint8)
        s = np.clip(s, 0, 255).astype(np.uint8)
        v = np.clip(v, 0, 255).astype(np.uint8)

        hsv = cv2.merge([h, s, v])
        out_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return out_bgr
