# LLamaGUI

Panel de control para inferencia de modelos LLM con llama.cpp + monitor de hardware.


<img width="2204" height="1560" alt="image" src="https://github.com/user-attachments/assets/6372fba7-9d10-4f4f-ab3d-5e2eedc44dee" />


## Descripción

Aplicación de escritorio que combina:

- **Monitor de hardware en tiempo real**: GPU (NVML), CPU, RAM, disco, red y temperaturas
- **Gestión de servidores llama.cpp**: UI interactiva con parser dinámico de parámetros desde `--help`
- **Modos de hardware automáticos**: Auto, Manual, Solo GPU, Solo CPU, Híbrido
- **Soporte multi-GPU**: Detección y distribución automática de carga en múltiples GPUs NVIDIA
- **Perfiles de servidor**: Guarda y carga configuraciones de parámetros

## Requisitos

- **Windows 10/11** (requiere privilegios de administrador para acceso NVML)
- **Python 3.11+** (recomendado 3.13)
- **GPU NVIDIA** con driver CUDA (opcional, modo CPU disponible)
- **llama-server.exe** — binario de llama.cpp con soporte CUDA

## Instalación

```bash
cd D:/Python/llamagui
pip install -r requirements.txt
```

## Ejecución

```bash
python llamagui.py
```

## Compilación (PyInstaller)

### Producción (sin consola)
```bash
pyinstaller llamagui.spec --clean
```

### Debug (con consola)
```bash
pyinstaller llamagui_debug.spec --clean
```

### Script automático
```bash
build.bat
```

## Estructura del proyecto

```
llamagui/
├── llamagui.py           # Aplicación principal
├── llamagui.spec         # Spec PyInstaller (producción)
├── llamagui_debug.spec   # Spec PyInstaller (debug)
├── build.bat             # Script de compilación
├── app_icon.ico          # Icono de la aplicación
├── requirements.txt      # Dependencias de Python
├── .gitignore            # Archivos a excluir del repo
└── README.md             # Este archivo
```

## Características

- **Parser dinámico**: Lee los parámetros de `llama-server --help` y construye la UI automáticamente
- **Modo Auto**: Detecta VRAM, cantidad de GPUs y tamaño del modelo para sugerir la mejor configuración
- **Perfiles múltiples**: Crea diferentes configuraciones para distintos modelos
- **Monitoreo en tiempo real**: Métricas de hardware actualizadas cada 2 segundos
- **Cero terminales**: Sin ventanas de consola ni logs visibles al usuario
- **Elevación UAC**: Se ejecuta como administrador automáticamente al hacer doble clic

## Parámetros de hardware

| Modo | Descripción |
|------|-------------|
| **Auto** | Detección automática según VRAM y modelo |
| **Manual** | Control total de cada parámetro |
| **Solo GPU** | Todo el modelo en VRAM (máxima velocidad) |
| **Solo CPU** | Modelo en CPU + RAM, sin GPU |
| **Híbrido** | Modelo dividido entre GPU y CPU |

## Licencia

MIT
