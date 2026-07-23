@echo off
echo ========================================
echo   LLama GUI v3.3 - Build Script
echo ========================================

echo.
echo [1/2] Compilando version DEBUG...
pyinstaller llamagui_debug.spec --clean --noconfirm
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en build debug
    pause
    exit /b 1
)

echo.
echo [2/2] Compilando version PRODUCCION...
pyinstaller llamagui.spec --clean --noconfirm
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en build produccion
    pause
    exit /b 1
)

echo.
echo ========================================
echo   BUILD COMPLETADO
echo   Debug:      dist\LLamaGUI_DEBUG.exe
echo   Produccion: dist\LLamaGUI_*.exe
echo ========================================
pause
