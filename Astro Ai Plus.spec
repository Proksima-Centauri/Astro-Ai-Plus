# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Astro Ai Plus.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('assets/favicon.ico', 'assets'),
        ('assets/favicon.ico', '.'),
        ('3d_fly_help.md', '.'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Astro Ai Plus',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    icon='assets/favicon.ico',
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Astro Ai Plus',
)
