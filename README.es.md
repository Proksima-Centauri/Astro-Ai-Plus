# Astro Ai Plus

Aplicacion de escritorio para procesado de astrofotografia, creada con Python (PyQt5 + OpenCV + NumPy).

El proyecto combina herramientas clasicas (Levels, Curves, Histogram, Blur, Crop, Rotate) con funciones astronomicas: analisis de estrellas (FWHM/SNR), plate solving, eliminacion de estrellas (StarNet++), denoise (deepSNR), mosaico de cuadros y filtro de animacion 3D FLY.

## Funciones principales

- Entrada/salida de imagenes: PNG, JPG, TIFF y FITS (`.fits`, `.fit`, `.fts`).
- Capas e historial: undo/redo, miniaturas, operaciones basicas de capas.
- Herramientas de tono y color: Levels, Curves (LUT), Histogram, GHS, correccion RGB/HSL.
- Herramientas astro:
  - analisis de imagen (FWHM, SNR, ruido de fondo),
  - plate solving (`solve-field` local o Astrometry.net como respaldo),
  - StarNet++ para quitar estrellas,
  - deepSNR externo para denoise,
  - mosaico de cuadros,
  - render de clip 3D FLY (audio opcional).
- Asistente AI Altair en modo 100% offline (`llama-cpp-python` + GGUF) y entrada por voz opcional.
- Interfaz PL/EN con preferencias y workspaces configurables.

## Requisitos

- Python 3.10+ (recomendado 3.11-3.13).
- Linux o Windows (en Linux la app activa software OpenGL).
- Paquetes de `requirements.txt`: `numpy`, `opencv-python`, `sep`, `astropy`, `photutils`, `SpeechRecognition`, `onnxruntime`, `PyQt5`, `pyserial`.

Herramientas externas opcionales:

- `solve-field` (Astrometry.net) para plate solving local.
- StarNet++ CLI.
- deepSNR CLI.
- `ffmpeg` para adjuntar audio en el resultado 3D FLY.

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecucion

```bash
python3 "Astro Ai Plus.py"
```

## Configuracion

La aplicacion guarda ajustes en `config` (JSON sin extension) en la raiz del proyecto.

Campos de ejemplo:

- rutas de herramientas (`starnet_path`, `deepsnr_path`),
- argumentos deepSNR (`deepsnr_args`),
- plate solving (`api_key`, `pixel_size_um`, `focal_length_mm`),
- ajustes AI (`local_ai_model_file`),
- idioma, tema, numero de nucleos, workspace.

Nota de seguridad: no publiques rutas privadas ni datos de workspace en repositorios publicos.

## Flujo rapido

1. Abre una imagen (`Open`) o arrastra un archivo.
2. Aplica correcciones base (Levels/Curves/Histogram/Correction).
3. Ejecuta `Analyze` para calcular metricas (FWHM, SNR).
4. Opcional: `StarNet++`, `deepSNR`, `Mosaic`, `Plate Solve`.
5. Para animacion, usa `3D FLY`.
6. Guarda el resultado (`Save` / `Save As`).

## Guia rapida de 3D FLY

1. Etapa 1 `quitar estrellas` (opcional): ejecuta StarNet++.
2. Etapa 2 `marcar secciones`: corta zonas por capas (`Cut`) o usa `Load layers`.
3. Etapa 3 `suavizado de bordes`: selecciona capa, define blur y pulsa `Apply`.
4. Etapa 4 `configurar clip`: define duracion y FPS.
5. Etapa 5/6 `movimiento y posicion`: define direccion, velocidad y zoom por capa.
6. Etapa final `agregar musica`: elige audio y pulsa `Render 3D FLY`.

Problemas comunes:

- `No layers for motion` -> agrega capas en la etapa 2.
- Blur no visible -> en etapa 3 selecciona capa y pulsa `Apply`.
- No se genera render -> abre la ultima pestana y define una ruta de salida valida.

## Comandos utiles de la consola interna

- `help`
- `open [path]`
- `save` / `save as [path]`
- `magic`
- `starnet++`
- `deepsnr`
- `3d fly`
- `analyze`
- `mosaic`
- `levels`, `curves`, `histogram`, `ghs`

## Estructura del proyecto

- `Astro Ai Plus.py` - archivo principal de la aplicacion.
- `deep_sky_catalog.py` - catalogo local de objetos de cielo profundo.
- `3d_fly_help.md` - manual detallado de 3D FLY.
- `requirements.txt` - dependencias Python.
- `assets/` - iconos y recursos de UI.
