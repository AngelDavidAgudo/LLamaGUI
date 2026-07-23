# -*- mode: python ; coding: utf-8 -*-
import sys
import os

a = Analysis(
    ['llamagui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('app_icon.ico', '.'),
        ('icono.png', '.'),
    ],
    hiddenimports=[
        'PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',
        'pynvml', 'psutil',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', '_tkinter', 'matplotlib', 'numpy', 'scipy', 'pandas',
        'PIL', 'cv2', 'torch', 'tensorflow',
        'PySide6.QtBluetooth', 'PySide6.QtDBus', 'PySide6.QtDesigner',
        'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets',
        'PySide6.QtNetwork', 'PySide6.QtNfc',
        'PySide6.QtOpenGL', 'PySide6.QtOpenGLWidgets',
        'PySide6.QtPdf', 'PySide6.QtPdfWidgets',
        'PySide6.QtPositioning', 'PySide6.QtPrintSupport',
        'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.QtQuick3D',
        'PySide6.QtQuickWidgets', 'PySide6.QtRemoteObjects',
        'PySide6.QtSensors', 'PySide6.QtSerialPort',
        'PySide6.QtSql', 'PySide6.QtSvg', 'PySide6.QtSvgWidgets',
        'PySide6.QtTest', 'PySide6.QtWebChannel',
        'PySide6.QtWebSockets', 'PySide6.QtXml',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='LLamaGUI_DEBUG',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico',
    uac_admin=True,
)
