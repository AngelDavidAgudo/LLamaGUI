# -*- coding: utf-8 -*-
"""
AI Control Panel v3.3.1 - Monitor hardware + Motor inferencia llama.cpp
Parser dinamico + Modos (Auto/GPU/CPU/Hibrido) + Multi-GPU P100
Fix: defaults robustos + conversiones protegidas
"""
__version__ = "v3.3.1"

import sys
import os
import re
import subprocess
import psutil
import pynvml
import json
from typing import List, Dict, Any, Optional, Tuple

try:
    _devnull = open(os.devnull, 'w')
    sys.stdout = _devnull
    sys.stderr = _devnull
except Exception:
    pass

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog,
    QSpinBox, QComboBox, QTextEdit, QGroupBox,
    QTabWidget, QInputDialog,
    QDoubleSpinBox, QCheckBox, QScrollArea,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon, QTextCursor

_original_excepthook = sys.excepthook
def _global_excepthook(exc_type, exc_value, exc_tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_tb)
    _original_excepthook(exc_type, exc_value, exc_tb)
sys.excepthook = _global_excepthook

CONFIG_FILE = "ai_dashboard_config.json"
PROFILES_FILE = "server_profiles.json"
PARAMS_CACHE_FILE = "llama_params_cache.json"
DEFAULT_SERVER_PATH = r"C:\llamacuda\llama-server.exe"

# ============================================================
# GPU DETECTION
# ============================================================
def detect_gpu_vram():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = info.total / (1024 ** 3)
        pynvml.nvmlShutdown()
        return vram_gb
    except Exception:
        return 0.0

def detect_gpu_count():
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return count
    except Exception:
        return 0

def detect_gpu_arch():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        pynvml.nvmlShutdown()
        return (major, minor)
    except Exception:
        return None

GPU_VRAM_GB = detect_gpu_vram()
GPU_COUNT = detect_gpu_count()
GPU_ARCH = detect_gpu_arch()

def has_tensor_cores():
    return GPU_ARCH[0] >= 7 if GPU_ARCH else False

def estimate_kv_reserve_gb(ctx):
    return max(0.5, (ctx / 4096) * 1.0)

# ============================================================
# LLAMA PARAM PARSER
# ============================================================
class LlamaParam:
    def __init__(self, short_names, long_names, arg_type, default,
                 description, category, choices=None, env_var=None, section=None):
        self.short_names = short_names or []
        self.long_names = long_names or []
        self.arg_type = arg_type
        self.default = default
        self.description = description
        self.category = category
        self.choices = choices or []
        self.env_var = env_var
        self.section = section

    @property
    def key(self):
        if self.long_names:
            return self.long_names[0].lstrip("-")
        if self.short_names:
            return self.short_names[0].lstrip("-")
        return ""

    @property
    def primary_long(self):
        return self.long_names[0] if self.long_names else (self.short_names[0] if self.short_names else "")

    @property
    def display_name(self):
        name = self.key.replace("-", " ").title()
        shorts = ", ".join(self.short_names)
        if shorts:
            return f"{name} ({shorts})"
        return name

    @property
    def full_description(self):
        desc = self.description
        if self.env_var:
            desc += f"\n[env: {self.env_var}]"
        if self.choices:
            desc += f"\nOpciones: {', '.join(self.choices)}"
        return desc


def _smart_split_commas(s):
    parts = []
    current = ""
    depth = 0
    for char in s:
        if char in '{[<(':
            depth += 1
            current += char
        elif char in '}])>':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current:
        parts.append(current)
    return parts


def _parse_options_string(options_str):
    short_names = []
    long_names = []
    placeholder = None
    parts = _smart_split_commas(options_str)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith('-'):
            match = re.match(r'(-{1,2}[\w-]+)\s*(.*)', part)
            if match:
                flag = match.group(1)
                rest = match.group(2).strip()
                if flag.startswith('--'):
                    long_names.append(flag)
                else:
                    short_names.append(flag)
                if rest:
                    if placeholder:
                        placeholder += "," + rest
                    else:
                        placeholder = rest
        else:
            if placeholder:
                placeholder += "," + part
            else:
                placeholder = part
    return short_names, long_names, placeholder


def _extract_choices(placeholder, description):
    if placeholder is None:
        return "bool", []
    ph = placeholder.strip()
    m = re.match(r'^\{(.+)\}$', ph)
    if m:
        return "enum", [c.strip() for c in m.group(1).split(',')]
    m = re.match(r'^\[(.+)\]$', ph)
    if m:
        return "enum", [c.strip() for c in m.group(1).split('|')]
    m = re.match(r'^<(.+)>$', ph)
    if m:
        inner = m.group(1)
        if '|' in inner:
            return "enum", [c.strip() for c in inner.split('|')]
        if '...' in inner:
            return "int", []
        return "string", []
    m = re.search(r'allowed values:\s*(.+?)(?:\n|$)', description, re.IGNORECASE)
    if m:
        choices = [c.strip() for c in m.group(1).split(',')]
        return "enum", choices
    if ph in ('N', 'PORT', 'SEED', 'INT'):
        return "int", []
    if ph in ('FNAME', 'PATH', 'FILE', 'STRING', 'GRAMMAR', 'SCHEMA',
              'JSON', 'KEY', 'TOKEN', 'URL', 'HOST', 'ORIGINS',
              'METHODS', 'HEADERS', 'PREFIX', 'FORMAT', 'MESSAGE',
              'SAMPLERS', 'SEQUENCE', 'SERVERS', 'SIMILARITY',
              'SECONDS', 'M', 'TYPE'):
        return "string", []
    if re.match(r'^[\w]+[0-9],', ph):
        return "string", []
    if re.match(r'^\w+-\w+$', ph):
        return "string", []
    try:
        int(ph)
        return "int", []
    except ValueError:
        pass
    try:
        float(ph)
        return "float", []
    except ValueError:
        pass
    return "string", []


def _extract_default(description, arg_type):
    """FIX v3.3.1: extrae solo el numero/valor valido, ignora parentesis y basura."""
    m = re.search(r'\(default:\s*([^)]*)\)', description, re.IGNORECASE)
    if not m:
        m = re.search(r'default:\s*([^,()\n]+)', description, re.IGNORECASE)
    if not m:
        return False if arg_type == "bool" else None
    raw = m.group(1).strip().strip("'\"() ").strip()
    if not raw:
        return False if arg_type == "bool" else None
    if ',' in raw:
        raw = raw.split(',')[0].strip().strip("'\"() ")
    if raw.lower() == 'enabled':
        return True
    if raw.lower() == 'disabled':
        return False
    if raw.lower().startswith('same as'):
        return None
    if raw.lower() in ('true', 'false'):
        return raw.lower() == 'true'
    # Extraer SOLO el numero valido del inicio (ignora ')' u otros caracteres)
    num_match = re.match(r'^-?\d+(\.\d+)?', raw)
    if num_match:
        num_str = num_match.group(0)
        try:
            return int(num_str)
        except ValueError:
            try:
                return float(num_str)
            except ValueError:
                pass
    return raw


def _extract_env_var(description):
    m = re.search(r'\(env:\s*([\w]+)\)', description)
    return m.group(1) if m else None


def _categorize_param(long_names, description, section):
    text = (" ".join(long_names) + " " + description).lower()
    section_map = {
        "sampling params": "Sampling",
        "speculative params": "Speculative Decoding",
    }
    for sec_key, cat in section_map.items():
        if sec_key in section.lower():
            return cat
    categories = [
        ("Modelo", ["model", "gguf", "lora", "adapter", "mmproj", "vocoder",
                    "hf-repo", "hf-file", "docker-repo", "model-url", "alias"]),
        ("Contexto", ["ctx", "context", "predict", "keep", "swa",
                      "checkpoint", "cache-ram"]),
        ("GPU / Offload", ["ngl", "gpu", "layer", "vram", "offload",
                           "split-mode", "tensor-split", "main-gpu",
                           "device", "fit", "override-tensor",
                           "cpu-moe", "n-cpu-moe", "no-host",
                           "op-offload", "repack"]),
        ("KV Cache", ["kv", "cache-type", "ctk", "ctv", "defrag",
                      "kv-offload", "kv-unified", "cache-idle"]),
        ("CPU / Threads", ["thread", "cpu-mask", "cpu-range", "cpu-strict",
                           "prio", "poll", "batch-size", "ubatch",
                           "parallel", "cont-batching"]),
        ("Atencion / RoPE", ["flash", "attn", "rope", "yarn"]),
        ("Memoria", ["mlock", "mmap", "direct-io", "numa"]),
        ("Razonamiento", ["reasoning", "reason", "think"]),
        ("Servidor HTTP", ["port", "host", "http", "server", "timeout",
                           "cors", "api", "ssl", "sse", "metrics",
                           "props", "slots", "webui", "ui", "jinja",
                           "chat-template", "embedding",
                           "rerank", "tags", "path",
                           "reuse-port", "sleep-idle", "warmup",
                           "cache-prompt", "cache-reuse", "slot",
                           "media-path", "models-dir", "models-preset",
                           "models-max", "models-autoload", "agent",
                           "tools", "skip-chat", "prefill",
                           "log-prompts", "pooling", "special",
                           "reverse-prompt", "spm-infill",
                           "context-shift", "image-", "mtmd"]),
        ("Logging", ["log", "verbose", "verbosity", "offline"]),
        ("RPC", ["rpc"]),
    ]
    for cat, keywords in categories:
        if any(kw in text for kw in keywords):
            return cat
    return "Otros"


def _is_continuation_line(line):
    line = line.rstrip('\r')
    if not line.strip():
        return False
    stripped = line.lstrip()
    indent = len(line) - len(stripped)
    if indent < 20:
        return False
    if stripped.startswith('- '):
        return True
    if stripped.startswith('-'):
        return False
    return True


def _is_section_header(line):
    stripped = line.strip()
    return stripped.startswith('-----') and stripped.endswith('-----')


def parse_llama_help(help_text):
    params = []
    lines = help_text.split('\n')
    current_section = "common params"
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\r')
        if _is_section_header(line):
            current_section = line.strip().strip('-').strip()
            i += 1
            continue
        stripped = line.lstrip()
        is_param_line = (
            stripped.startswith('-')
            and not stripped.startswith('- ')
            and not _is_section_header(line)
        )
        if is_param_line:
            match = re.match(r'^(\s*)(.*?)(\s{2,})([^-\s].*)$', line)
            if match:
                options_part = match.group(2).strip()
                description = match.group(4).strip()
            else:
                options_part = stripped
                description = ""
            while i + 1 < len(lines) and _is_continuation_line(lines[i + 1]):
                i += 1
                cont = lines[i].rstrip('\r').strip()
                description = (description + " " + cont) if description else cont
            short_names, long_names, placeholder = _parse_options_string(options_part)
            if not long_names and not short_names:
                i += 1
                continue
            arg_type, choices = _extract_choices(placeholder, description)
            default = _extract_default(description, arg_type)
            env_var = _extract_env_var(description)
            clean_desc = description
            clean_desc = re.sub(r'\(env:\s*[\w]+\)', '', clean_desc).strip()
            clean_desc = re.sub(r'\(default:\s*[^)]+\)', '', clean_desc).strip()
            clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
            category = _categorize_param(long_names, description, current_section)
            params.append(LlamaParam(
                short_names=short_names, long_names=long_names,
                arg_type=arg_type, default=default,
                description=clean_desc, category=category,
                choices=choices, env_var=env_var, section=current_section,
            ))
        i += 1
    return params


def load_llama_params(server_path, force_refresh=False):
    if not force_refresh and os.path.exists(PARAMS_CACHE_FILE):
        try:
            with open(PARAMS_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            if cache.get("server_path") == server_path and cache.get("params"):
                params = []
                for p in cache.get("params", []):
                    params.append(LlamaParam(
                        short_names=p.get("short_names", []),
                        long_names=p.get("long_names", []),
                        arg_type=p.get("arg_type", "string"),
                        default=p.get("default"),
                        description=p.get("description", ""),
                        category=p.get("category", "Otros"),
                        choices=p.get("choices", []),
                        env_var=p.get("env_var"),
                        section=p.get("section"),
                    ))
                if params:
                    return params
        except Exception:
            pass
    if not server_path or not os.path.exists(server_path):
        return []
    try:
        result = subprocess.run(
            [server_path, "--help"],
            capture_output=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        out = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
        err = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""
        help_text = out + err
        params = parse_llama_help(help_text)
        cache = {
            "server_path": server_path,
            "timestamp": os.path.getmtime(server_path),
            "param_count": len(params),
            "params": [
                {
                    "short_names": p.short_names, "long_names": p.long_names,
                    "arg_type": p.arg_type, "default": p.default,
                    "description": p.description, "category": p.category,
                    "choices": p.choices, "env_var": p.env_var,
                    "section": p.section,
                }
                for p in params
            ]
        }
        with open(PARAMS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        return params
    except Exception as e:
        print(f"Error cargando parametros: {e}")
        return []


# ============================================================
# HARDWARE MODES
# ============================================================
def _find_param_exact(params, key):
    for p in params:
        if p.key == key:
            return p
    return None

def _find_param_contains(params, pattern, exclude=None):
    for p in params:
        text = (p.key + " " + " ".join(p.long_names)).lower()
        if pattern in text:
            if exclude and any(ex in text for ex in exclude):
                continue
            return p
    return None

def estimate_model_layers(model_size_gb):
    if not model_size_gb or model_size_gb <= 0:
        return 40
    if model_size_gb < 3:
        return 32
    elif model_size_gb < 8:
        return 36
    elif model_size_gb < 20:
        return 40
    elif model_size_gb < 45:
        return 64
    else:
        return 80

def estimate_layers_for_vram(vram_gb, model_size_gb, reserve_gb=2.0):
    if not vram_gb or vram_gb <= 0 or not model_size_gb or model_size_gb <= 0:
        return None
    available = vram_gb - reserve_gb
    if available <= 0:
        return 0
    total_layers = estimate_model_layers(model_size_gb)
    gb_per_layer = model_size_gb / total_layers
    if gb_per_layer <= 0:
        return 0
    layers = int(available / gb_per_layer)
    return max(0, min(layers, total_layers))

def detect_best_mode(vram_gb, gpu_count, model_size_gb, ctx=4096):
    vram_total = (vram_gb or 0) * (gpu_count or 1)
    if vram_total < 1.0:
        return "cpu_only", "Sin GPU usable detectada"
    if not model_size_gb or model_size_gb <= 0:
        if vram_total >= 8:
            return "gpu_only", f"GPU disponible ({vram_total:.1f}GB). Selecciona modelo para ajustar."
        return "hybrid", f"VRAM limitada ({vram_total:.1f}GB)"
    kv_reserve = estimate_kv_reserve_gb(ctx)
    overhead = 1.5
    needed = model_size_gb + kv_reserve + overhead
    if needed <= vram_total:
        return "gpu_only", f"Modelo+KV+overhead ({needed:.1f}GB) cabe en VRAM ({vram_total:.1f}GB). Expertos en VRAM = max velocidad."
    if vram_total >= model_size_gb * 0.2 + overhead:
        return "hybrid", f"Modelo ({model_size_gb:.1f}GB) no cabe entero en VRAM ({vram_total:.1f}GB). Offload parcial."
    return "cpu_only", f"VRAM ({vram_total:.1f}GB) insuficiente para offload util del modelo ({model_size_gb:.1f}GB)."

def apply_hardware_mode(mode, panel, params, vram_gb, gpu_count, model_size_gb):
    messages = []
    current_ctx = 4096
    ctx_widget = panel.param_widgets.get("ctx-size")
    if ctx_widget and ctx_widget.is_enabled():
        try:
            current_ctx = ctx_widget.get_value()
        except Exception:
            pass

    for widget in panel.param_widgets.values():
        widget.set_enabled(False)

    if mode == "manual":
        messages.append("[MODO] Manual: controlas cada parametro manualmente.")
        panel._on_param_changed()
        return messages

    def set_param(pattern, value, exclude=None):
        p = _find_param_exact(params, pattern)
        if not p:
            p = _find_param_contains(params, pattern, exclude)
        if p and p.key in panel.param_widgets:
            widget = panel.param_widgets[p.key]
            widget.set_enabled(True)
            widget.set_value(value)
            return p.key
        return None

    vram_total = (vram_gb or 0) * (gpu_count or 1)

    if mode == "gpu_only":
        messages.append(f"[MODO] Solo GPU: todo el modelo en VRAM ({vram_total:.1f}GB total, {gpu_count} GPU).")
        if set_param("gpu-layers", 99, exclude=["draft"]):
            messages.append("  - Capas GPU: 99 (todas en VRAM)")
        if set_param("kv-offload", True, exclude=["draft"]):
            messages.append("  - KV cache: en GPU")
        set_param("mmap", False)
        set_param("mlock", True)
        fa_value = "on" if has_tensor_cores() else "auto"
        if set_param("flash-attn", fa_value):
            arch_note = "tensor cores" if has_tensor_cores() else "Pascal/P100"
            messages.append(f"  - Flash Attention: {fa_value} ({arch_note})")
        if set_param("cache-type-k", "q8_0", exclude=["draft"]):
            messages.append("  - KV Key: q8_0")
        if set_param("cache-type-v", "q8_0", exclude=["draft"]):
            messages.append("  - KV Value: q8_0")
        if gpu_count > 1:
            split = ",".join(["1"] * gpu_count)
            if set_param("tensor-split", split):
                messages.append(f"  - Tensor split: {split} ({gpu_count} GPUs equitativo)")
            if set_param("split-mode", "layer"):
                messages.append("  - Split mode: layer (prueba 'row' si tienes NVLink)")
        if model_size_gb and vram_total > 0:
            kv_reserve = estimate_kv_reserve_gb(current_ctx)
            overhead = 1.5
            needed = model_size_gb + kv_reserve + overhead
            if needed <= vram_total:
                messages.append(f"  [OK] Necesario ~{needed:.1f}GB cabe en {vram_total:.1f}GB.")
                messages.append(f"  [OK] Expertos MoE en VRAM = maxima velocidad.")
            else:
                messages.append(f"  [WARN] Necesario ~{needed:.1f}GB > VRAM {vram_total:.1f}GB.")
                messages.append(f"  [WARN] Reduce contexto o usa modo Hibrido.")

    elif mode == "cpu_only":
        messages.append("[MODO] Solo CPU: modelo en CPU + RAM, sin grafica.")
        if set_param("gpu-layers", 0, exclude=["draft"]):
            messages.append("  - Capas GPU: 0 (todo en CPU)")
        if set_param("kv-offload", False, exclude=["draft"]):
            messages.append("  - KV cache: en RAM")
        set_param("mmap", True)
        set_param("mlock", True)
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 4
        if set_param("threads", cores):
            messages.append(f"  - Threads: {cores}")
        if set_param("numa", "distribute"):
            messages.append("  - NUMA: distribute")
        set_param("flash-attn", "auto")

    elif mode == "hybrid":
        messages.append(f"[MODO] Hibrido: GPU + CPU (usar solo si el modelo NO cabe en {vram_total:.1f}GB).")
        layers = estimate_layers_for_vram(vram_total, model_size_gb)
        if layers is not None:
            if set_param("gpu-layers", layers, exclude=["draft"]):
                messages.append(f"  - Capas GPU: {layers} (estimado para {vram_total:.1f}GB VRAM)")
        else:
            if set_param("gpu-layers", 33, exclude=["draft"]):
                messages.append("  - Capas GPU: 33 (ajusta segun tu VRAM)")
        if set_param("kv-offload", True, exclude=["draft"]):
            messages.append("  - KV cache: en GPU")
        set_param("mmap", True)
        set_param("mlock", True)
        fa_value = "on" if has_tensor_cores() else "auto"
        set_param("flash-attn", fa_value)
        if gpu_count > 1:
            split = ",".join(["1"] * gpu_count)
            set_param("tensor-split", split)
            set_param("split-mode", "layer")
        set_param("cache-type-k", "q8_0", exclude=["draft"])
        set_param("cache-type-v", "q8_0", exclude=["draft"])
        if model_size_gb and vram_total > 0 and model_size_gb > vram_total - 2:
            if set_param("cpu-moe", True, exclude=["n-cpu-moe"]):
                messages.append("  - CPU MoE: expertos en CPU (modelo no cabe entero en VRAM)")
        if set_param("fit", "on"):
            messages.append("  - Fit: on (auto-ajuste a memoria)")
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 4
        set_param("threads", cores)

    panel._on_param_changed()
    return messages


# ============================================================
# CONFIG & PROFILES
# ============================================================
def load_config():
    default_cfg = {"last_model": "", "server_path": DEFAULT_SERVER_PATH, "port": 8080}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                default_cfg.update(json.load(f))
        except Exception:
            pass
    return default_cfg

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            return raw if isinstance(raw, list) else [raw]
        except Exception:
            pass
    return [{
        "name": "Default",
        "model_path": "",
        "server_path": DEFAULT_SERVER_PATH,
        "mode": "auto",
        "params": {}
    }]

def save_profiles(profiles):
    try:
        with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ============================================================
# TEMPERATURES
# ============================================================
def get_cpu_temp_vbs():
    try:
        script = r'''
On Error Resume Next
Set objWMIService = GetObject("winmgmts:\.\root\WMI")
Set colItems = objWMIService.ExecQuery("SELECT * FROM MSAcpi_ThermalZoneTemperature")
For Each objItem in colItems
If objItem.CurrentTemperature <> 0 Then
TempCelsius = (objItem.CurrentTemperature - 2732) / 10
WScript.Echo FormatNumber(TempCelsius, 1)
Exit For
End If
Next
'''
        cmd = ["cscript.exe", "//Nologo", "//E:vbscript", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5,
                                input=script, creationflags=subprocess.CREATE_NO_WINDOW)
        out = result.stdout.strip()
        if out and out not in ("0.0", "-273.2", "N/A"):
            return float(out)
    except Exception:
        pass
    return None

def get_disk_temps():
    try:
        cmd = [
            "powershell.exe", "-NoProfile", "-Command",
            "$disks = Get-PhysicalDisk -ErrorAction SilentlyContinue; "
            "if ($disks) { "
            "$out = @(); foreach($d in $disks) { $name = $d.FriendlyName; $t = $d.HealthStatus; "
            "$out += \"$name:$t\" } Write-Output ($out -join '|') } "
            "else { Write-Output 'N/A' }"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        out = result.stdout.strip()
        if not out or out in ("N/A", "False"):
            return {}
        disk_temps = {}
        for item in out.split('|'):
            if ':' in item:
                name, status = item.split(':', 1)
                disk_temps[name.strip()] = status.strip()
        return disk_temps
    except Exception:
        return {}


# ============================================================
# THREADS
# ============================================================
class HardwareMonitorThread(QThread):
    stats_updated = Signal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.nvml_available = False
        self.gpu_count = 0
        self._init_nvml()
        self._cpu_temp = get_cpu_temp_vbs()

    def _init_nvml(self):
        try:
            pynvml.nvmlInit()
            self.nvml_available = True
            self.gpu_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            self.nvml_available = False
            self.gpu_count = 0

    def run(self):
        psutil.cpu_percent(interval=None)
        while self.running:
            data = {
                "cpu": psutil.cpu_percent(interval=None),
                "ram_pct": psutil.virtual_memory().percent,
                "ram_used_gb": psutil.virtual_memory().used / (1024**3),
                "ram_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_pct": psutil.disk_usage('C:\\').percent,
                "net_sent": 0, "net_recv": 0,
                "gpus": [], "cpu_temp": self._cpu_temp, "disk_temps": {},
            }
            if self.nvml_available:
                try:
                    for i in range(self.gpu_count):
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                        try:
                            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                        except Exception:
                            util = 0
                        data["gpus"].append({
                            "name": pynvml.nvmlDeviceGetName(h),
                            "temp": temp,
                            "vram_used": mem.used / (1024**3),
                            "vram_total": mem.total / (1024**3),
                            "power": power, "util": util,
                        })
                except pynvml.NVMLError:
                    pass
            try:
                data["disk_temps"] = get_disk_temps()
            except Exception:
                pass
            try:
                net = psutil.net_io_counters()
                data["net_sent"] = net.bytes_sent / (1024**2)
                data["net_recv"] = net.bytes_recv / (1024**2)
            except Exception:
                pass
            self.stats_updated.emit(data)
            self.msleep(2000)

    def stop(self):
        self.running = False
        self.wait()
        if self.nvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class WorkerThread(QThread):
    output_signal = Signal(str)
    finished_signal = Signal(bool)

    def __init__(self, command, env=None):
        super().__init__()
        self.command = command
        self.env = env or os.environ.copy()
        self.process = None

    def run(self):
        self.env["PYTHONUNBUFFERED"] = "1"
        try:
            self.process = subprocess.Popen(
                self.command, env=self.env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for line in iter(self.process.stdout.readline, ''):
                self.output_signal.emit(line.strip())
            self.process.stdout.close()
            self.process.wait()
            self.finished_signal.emit(self.process.returncode == 0)
        except Exception as e:
            self.output_signal.emit(f"Error fatal: {str(e)}")
            self.finished_signal.emit(False)

    def stop(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None


# ============================================================
# PARAM WIDGET (FIX v3.3.1: conversiones int/float protegidas)
# ============================================================
class ParamWidget(QWidget):
    value_changed = Signal()

    def __init__(self, param, parent=None):
        super().__init__(parent)
        self.param = param
        self.control = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        self.enable_chk = QCheckBox()
        self.enable_chk.setChecked(False)
        self.enable_chk.setToolTip("Activar este parametro")
        self.enable_chk.stateChanged.connect(self._on_enable_changed)
        layout.addWidget(self.enable_chk)

        label = QLabel(self.param.display_name)
        label.setFixedWidth(240)
        label.setToolTip(self.param.full_description)
        label.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        layout.addWidget(label)

        if self.param.arg_type == "bool":
            self.control = QCheckBox()
            self.control.setChecked(bool(self.param.default))
            self.control.setEnabled(False)
            self.control.setToolTip(self.param.full_description)
            layout.addWidget(self.control)
        elif self.param.arg_type == "int":
            self.control = QSpinBox()
            self.control.setRange(-999999, 999999)
            try:
                _val = int(self.param.default) if self.param.default is not None else 0
            except (ValueError, TypeError):
                _val = 0
            self.control.setValue(_val)
            self.control.setEnabled(False)
            self.control.setToolTip(self.param.full_description)
            layout.addWidget(self.control, stretch=1)
        elif self.param.arg_type == "float":
            self.control = QDoubleSpinBox()
            self.control.setRange(-999999.0, 999999.0)
            self.control.setDecimals(3)
            try:
                _val = float(self.param.default) if self.param.default is not None else 0.0
            except (ValueError, TypeError):
                _val = 0.0
            self.control.setValue(_val)
            self.control.setEnabled(False)
            self.control.setToolTip(self.param.full_description)
            layout.addWidget(self.control, stretch=1)
        elif self.param.arg_type == "enum":
            self.control = QComboBox()
            self.control.addItems(self.param.choices)
            if self.param.default in self.param.choices:
                self.control.setCurrentText(str(self.param.default))
            self.control.setEnabled(False)
            self.control.setToolTip(self.param.full_description)
            layout.addWidget(self.control, stretch=1)
        else:
            self.control = QLineEdit()
            self.control.setText(str(self.param.default) if self.param.default else "")
            self.control.setEnabled(False)
            self.control.setToolTip(self.param.full_description)
            layout.addWidget(self.control, stretch=1)

        reset_btn = QPushButton("R")
        reset_btn.setFixedSize(25, 25)
        reset_btn.setToolTip("Restaurar valor por defecto")
        reset_btn.clicked.connect(self.reset_to_default)
        layout.addWidget(reset_btn)

        self.setLayout(layout)

    def _on_enable_changed(self, state):
        enabled = state == Qt.CheckState.Checked.value
        self.control.setEnabled(enabled)
        self.value_changed.emit()

    def reset_to_default(self):
        if self.param.arg_type == "bool":
            self.control.setChecked(bool(self.param.default))
        elif self.param.arg_type in ("int", "float"):
            try:
                self.control.setValue(self.param.default or 0)
            except (ValueError, TypeError):
                self.control.setValue(0)
        elif self.param.arg_type == "enum":
            if self.param.default in self.param.choices:
                self.control.setCurrentText(str(self.param.default))
        else:
            self.control.setText(str(self.param.default) if self.param.default else "")

    def is_enabled(self):
        return self.enable_chk.isChecked()

    def set_enabled(self, enabled):
        self.enable_chk.setChecked(enabled)
        self.control.setEnabled(enabled)

    def get_value(self):
        if self.param.arg_type == "bool":
            return self.control.isChecked()
        elif self.param.arg_type == "int":
            return self.control.value()
        elif self.param.arg_type == "float":
            return self.control.value()
        elif self.param.arg_type == "enum":
            return self.control.currentText()
        else:
            return self.control.text()

    def set_value(self, value):
        try:
            if self.param.arg_type == "bool":
                self.control.setChecked(bool(value))
            elif self.param.arg_type == "int":
                self.control.setValue(int(value))
            elif self.param.arg_type == "float":
                self.control.setValue(float(value))
            elif self.param.arg_type == "enum":
                if value in self.param.choices:
                    self.control.setCurrentText(str(value))
            else:
                self.control.setText(str(value))
        except Exception:
            pass


# ============================================================
# PARAMS PANEL
# ============================================================
class ParamsPanel(QWidget):
    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.all_params = params
        self.param_widgets = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        info_bar = QHBoxLayout()
        self.info_label = QLabel(f"{len(self.all_params)} parametros disponibles del servidor")
        self.info_label.setStyleSheet("color: #4CAF50; font-size: 12px; font-weight: bold;")
        info_bar.addWidget(self.info_label)
        info_bar.addStretch()

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Todas las categorias", "Solo activados"])
        self.filter_combo.currentTextChanged.connect(self._filter_params)
        self.filter_combo.setFixedWidth(180)
        info_bar.addWidget(self.filter_combo)

        layout.addLayout(info_bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self.params_container = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_layout.setContentsMargins(8, 8, 8, 8)
        self.params_layout.setSpacing(4)
        self.params_container.setLayout(self.params_layout)

        scroll.setWidget(self.params_container)
        layout.addWidget(scroll, stretch=1)

        self.setLayout(layout)
        self._populate_all_params()

    def _populate_all_params(self, filter_mode="Todas las categorias"):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        categories = {}
        for param in self.all_params:
            if filter_mode == "Solo activados":
                widget = self.param_widgets.get(param.key)
                if not widget or not widget.is_enabled():
                    continue
            if param.category not in categories:
                categories[param.category] = []
            categories[param.category].append(param)

        for cat in sorted(categories.keys()):
            params = categories[cat]
            if not params:
                continue
            header = QLabel(cat)
            header.setStyleSheet(
                "color: #4CAF50; font-size: 13px; font-weight: bold; "
                "padding: 10px 4px 4px 4px; border-bottom: 1px solid #333;"
            )
            self.params_layout.addWidget(header)
            for param in params:
                widget = self._get_or_create_widget(param)
                self.params_layout.addWidget(widget)

        self.params_layout.addStretch()

    def _get_or_create_widget(self, param):
        if param.key not in self.param_widgets:
            widget = ParamWidget(param)
            widget.value_changed.connect(self._on_param_changed)
            self.param_widgets[param.key] = widget
        return self.param_widgets[param.key]

    def _on_param_changed(self):
        enabled_count = sum(1 for w in self.param_widgets.values() if w.is_enabled())
        self.info_label.setText(
            f"{len(self.all_params)} parametros disponibles | {enabled_count} activados"
        )

    def _filter_params(self):
        self._populate_all_params(self.filter_combo.currentText())

    def get_enabled_params(self):
        values = {}
        for key, widget in self.param_widgets.items():
            if widget.is_enabled():
                values[key] = widget.get_value()
        return values

    def set_enabled_params(self, values):
        for widget in self.param_widgets.values():
            widget.set_enabled(False)
        for key, value in values.items():
            if key in self.param_widgets:
                self.param_widgets[key].set_enabled(True)
                self.param_widgets[key].set_value(value)
        self._on_param_changed()

    def reset_all(self):
        for widget in self.param_widgets.values():
            widget.set_enabled(False)
            widget.reset_to_default()
        self._on_param_changed()


# ============================================================
# MAIN DASHBOARD
# ============================================================
class UnifiedDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.profiles = load_profiles()
        self.current_index = 0
        self.server_thread = None
        self.monitor_thread = HardwareMonitorThread()
        self.llama_params = []

        self.initUI()
        self.monitor_thread.stats_updated.connect(self.update_telemetry)
        self.monitor_thread.start()

        if self.profiles:
            server_path = self.profiles[0].get("server_path", DEFAULT_SERVER_PATH)
            self._load_params_for_server(server_path)

        if self.profile_tabs.count() > 0:
            self.on_profile_changed(0)

    def initUI(self):
        self.setWindowTitle("LLamaGUI v3.3.1")
        self.setMinimumSize(1100, 750)

        try:
            _icon_path = "icono.png"
            if getattr(sys, "frozen", False):
                _icon_path = os.path.join(sys._MEIPASS, "icono.png")
            elif not os.path.exists(_icon_path):
                _icon_path = None
            if _icon_path:
                self.setWindowIcon(QIcon(_icon_path))
        except Exception:
            pass

        self.setStyleSheet("""
            QWidget { background-color: #1a1a1a; color: #e0e0e0;
                      font-family: 'Consolas', monospace; }
            QGroupBox { border: 1px solid #444; border-radius: 6px;
                        margin-top: 12px; padding-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin;
                               left: 10px; padding: 0 5px; color: #4CAF50; }
            QPushButton { background-color: #333; border: 1px solid #555;
                          padding: 6px; border-radius: 4px; }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:pressed { background-color: #222; }
            QLineEdit, QSpinBox, QComboBox, QDoubleSpinBox { background-color: #2a2a2a;
                border: 1px solid #555; padding: 5px; border-radius: 3px; }
            QTextEdit { background-color: #121212; border: 1px solid #333; border-radius: 4px; }
            QCheckBox { color: #e0e0e0; spacing: 8px; }
            QScrollArea { border: none; }
            QScrollBar:vertical { background: #2a2a2a; width: 10px; border-radius: 5px; }
            QScrollBar::handle:vertical { background: #555; border-radius: 5px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background: #777; }
        """)

        main = QVBoxLayout()
        main.setSpacing(10)

        hw = QGroupBox("Monitor de Hardware")
        self.hw_layout = QVBoxLayout()

        self.lbl_cpu = QLabel("[CPU] CPU: Cargando...")
        self.lbl_cpu.setStyleSheet("font-size: 14px;")
        self.hw_layout.addWidget(self.lbl_cpu)

        self.lbl_ram = QLabel("[RAM] Mem: Cargando...")
        self.lbl_ram.setStyleSheet("font-size: 14px;")
        self.hw_layout.addWidget(self.lbl_ram)

        self.lbl_disk = QLabel("[DISK] Disco: Cargando...")
        self.lbl_disk.setStyleSheet("font-size: 14px;")
        self.hw_layout.addWidget(self.lbl_disk)

        self.lbl_net = QLabel("[NET] Red: Cargando...")
        self.lbl_net.setStyleSheet("font-size: 14px; color: #888;")
        self.hw_layout.addWidget(self.lbl_net)

        self.gpu_containers = []
        self.lbl_gpu_placeholder = QLabel("[GPU] GPUs: Cargando...")
        self.lbl_gpu_placeholder.setStyleSheet("font-size: 13px; color: #888;")
        self.hw_layout.addWidget(self.lbl_gpu_placeholder)

        hw.setLayout(self.hw_layout)
        main.addWidget(hw)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(140)
        self._console_max_lines = 500
        main.addWidget(self.console, stretch=0)

        self.profile_tabs = QTabWidget()
        self.profile_tabs.currentChanged.connect(self.on_profile_changed)
        main.addWidget(self.profile_tabs, stretch=1)

        for idx, prof in enumerate(self.profiles):
            self.create_profile_tab(prof.get("name", f"Perfil {idx+1}"), prof)

        # Tab "+" siempre al final para crear nuevo perfil
        self._add_plus_tab()

        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("INICIAR SERVIDOR")
        self.btn_start.setStyleSheet(
            "background-color: #2e7d32; font-weight: bold; "
            "font-size: 14px; padding: 10px;"
        )
        self.btn_start.clicked.connect(self.start_server)

        self.btn_stop = QPushButton("DETENER")
        self.btn_stop.setStyleSheet(
            "background-color: #c62828; font-weight: bold; "
            "font-size: 14px; padding: 10px;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_server)

        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        main.addLayout(ctrl)

        self.setLayout(main)

        arch_txt = f"SM {GPU_ARCH[0]}.{GPU_ARCH[1]}" if GPU_ARCH else "desconocida"
        tc_txt = "con tensor cores" if has_tensor_cores() else "SIN tensor cores (Pascal)"
        self.log(f"[INFO] GPU: {GPU_VRAM_GB:.1f}GB x {GPU_COUNT} = {GPU_VRAM_GB*GPU_COUNT:.1f}GB VRAM total")
        self.log(f"[INFO] Arquitectura: {arch_txt} ({tc_txt})")

    def _load_params_for_server(self, server_path):
        if not server_path or not os.path.exists(server_path):
            self.log(f"[WARN] Servidor no encontrado: {server_path}")
            self.llama_params = []
            return
        self.log(f"[INFO] Cargando parametros de: {os.path.basename(server_path)}")
        self.llama_params = load_llama_params(server_path, force_refresh=True)
        if self.llama_params:
            self.log(f"[OK] {len(self.llama_params)} parametros detectados y disponibles")
            for i in range(self.profile_tabs.count()):
                tab = self.profile_tabs.widget(i)
                if tab and hasattr(tab, 'params_panel'):
                    current_values = tab.params_panel.get_enabled_params()
                    old_panel = tab.params_panel
                    tab.layout().removeWidget(old_panel)
                    old_panel.deleteLater()
                    new_panel = ParamsPanel(self.llama_params)
                    new_panel.set_enabled_params(current_values)
                    tab.params_panel = new_panel
                    tab.layout().insertWidget(3, new_panel, stretch=1)
        else:
            self.log("[ERROR] No se detectaron parametros. Verifica la ruta del servidor.")

    def _add_plus_tab(self):
        plus_tab = QWidget()
        plus_tab.setObjectName("plus_tab")
        layout = QVBoxLayout(plus_tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plus_btn = QPushButton("+")
        plus_btn.setFixedSize(50, 50)
        plus_btn.setStyleSheet(
            "font-size: 28px; font-weight: bold; "
            "background-color: transparent; "
            "color: #2e7d32; border: none; "
            "border-radius: 25px;"
        )
        plus_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        plus_btn.clicked.connect(self._on_plus_tab_clicked)
        layout.addWidget(plus_btn)
        self.profile_tabs.addTab(plus_tab, "Nueva")

    def _on_plus_tab_clicked(self):
        if self.profile_tabs.currentWidget() is None:
            return
        name, ok = QInputDialog.getText(
            self, "Nuevo Perfil", "Nombre del perfil:"
        )
        if not (ok and name):
            # Volver a la ultima pestaña de perfil
            last_idx = self.profile_tabs.count() - 1
            if last_idx >= 0:
                self.profile_tabs.setCurrentIndex(last_idx)
            return
        prof = {
            "name": name,
            "model_path": self.config.get("last_model", ""),
            "server_path": self.config.get("server_path", DEFAULT_SERVER_PATH),
            "mode": "auto",
            "params": {}
        }
        self.profiles.append(prof)
        save_profiles(self.profiles)
        self.create_profile_tab(name, prof)
        # Quitar el tab "+" actual y agregarlo al final
        plus_idx = self.profile_tabs.indexOf(
            self.profile_tabs.currentWidget()
        )
        if plus_idx >= 0:
            self.profile_tabs.removeTab(plus_idx)
        self._add_plus_tab()
        self.profile_tabs.setCurrentIndex(
            self.profile_tabs.count() - 2
        )
        self.log(f"Perfil '{name}' creado.")

    def add_new_profile(self):
        name, ok = QInputDialog.getText(self, "Nuevo Perfil", "Nombre del perfil:")
        if not (ok and name):
            return
        prof = {
            "name": name,
            "model_path": self.config.get("last_model", ""),
            "server_path": self.config.get("server_path", DEFAULT_SERVER_PATH),
            "mode": "auto",
            "params": {}
        }
        self.profiles.append(prof)
        save_profiles(self.profiles)
        self.create_profile_tab(name, prof)
        # Reponer "+" al final
        plus_idx = self.profile_tabs.count() - 1
        if plus_idx >= 0:
            self.profile_tabs.removeTab(plus_idx)
        self._add_plus_tab()
        self.profile_tabs.setCurrentIndex(self.profile_tabs.count() - 2)
        self.log(f"Perfil '{name}' creado.")

    def on_profile_changed(self, index):
        # Ignorar si se clickeo en el tab "+"
        if index == self.profile_tabs.count() - 1:
            return
        if hasattr(self, 'current_index') and self.current_index != index:
            old = self.profile_tabs.widget(self.current_index)
            if old and hasattr(old, 'params_panel'):
                self._extract_profile_data(old, self.current_index)
        self.current_index = index
        if 0 <= index < len(self.profiles):
            tab = self.profile_tabs.widget(index)
            if tab and hasattr(tab, 'params_panel'):
                prof = self.profiles[index]
                if hasattr(tab, 'model_edit'):
                    tab.model_edit.setText(prof.get("model_path", ""))
                if hasattr(tab, 'server_edit'):
                    tab.server_edit.setText(prof.get("server_path", DEFAULT_SERVER_PATH))
                tab.params_panel.set_enabled_params(prof.get("params", {}))
                mode = prof.get("mode", "auto")
                tab.current_mode = mode
                if mode == "auto":
                    self.apply_mode_to_tab("auto", tab)
                else:
                    self._highlight_mode_button(tab, mode)

    def _extract_profile_data(self, tab, idx):
        if idx < 0 or idx >= len(self.profiles):
            return
        if hasattr(tab, 'model_edit'):
            self.profiles[idx]["model_path"] = tab.model_edit.text()
        if hasattr(tab, 'server_edit'):
            self.profiles[idx]["server_path"] = tab.server_edit.text()
        if hasattr(tab, 'params_panel'):
            self.profiles[idx]["params"] = tab.params_panel.get_enabled_params()
        if hasattr(tab, 'current_mode'):
            self.profiles[idx]["mode"] = tab.current_mode
        save_profiles(self.profiles)

    def apply_mode_to_tab(self, mode, tab):
        if not hasattr(tab, 'params_panel'):
            return
        model_size = None
        if hasattr(tab, 'model_edit'):
            model_path = tab.model_edit.text()
            if model_path:
                model_size = self._estimate_model_size(model_path)

        effective_mode = mode
        if mode == "auto":
            effective_mode, reason = detect_best_mode(GPU_VRAM_GB, GPU_COUNT, model_size)
            self.log(f"[AUTO] Detectado: {effective_mode.upper()} -> {reason}")

        messages = apply_hardware_mode(
            effective_mode, tab.params_panel, self.llama_params,
            GPU_VRAM_GB, GPU_COUNT, model_size
        )
        for msg in messages:
            self.log(msg)

        tab.current_mode = mode
        self._highlight_mode_button(tab, mode)

    def _highlight_mode_button(self, tab, mode):
        if not hasattr(tab, 'mode_buttons'):
            return
        mode_order = ["auto", "manual", "gpu_only", "cpu_only", "hybrid"]
        idx = mode_order.index(mode) if mode in mode_order else -1
        for i, btn in enumerate(tab.mode_buttons):
            if i == idx:
                color = "#2e7d32" if mode == "auto" else "#1565c0"
                border = "#4CAF50" if mode == "auto" else "#1976d2"
                btn.setStyleSheet(
                    f"background-color: {color}; color: #fff; "
                    f"font-weight: bold; border: 1px solid {border};"
                )
            else:
                btn.setStyleSheet("")

    def create_profile_tab(self, profile_name, profile_data=None):
        w = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

        server_row = QHBoxLayout()
        lbl_server = QLabel("Servidor:")
        lbl_server.setFixedWidth(90)
        lbl_server.setStyleSheet("color: #4CAF50; font-weight: bold;")
        txt_server = QLineEdit()
        txt_server.setPlaceholderText("Ruta a llama-server.exe...")
        txt_server.setMinimumHeight(30)
        w.server_edit = txt_server
        btn_server = QPushButton("Examinar...")
        btn_server.setFixedWidth(90)
        btn_server.setFixedHeight(30)
        btn_server.clicked.connect(lambda: self.browse_server_for_widget(txt_server))
        server_row.addWidget(lbl_server)
        server_row.addWidget(txt_server, stretch=1)
        server_row.addWidget(btn_server)
        main_layout.addLayout(server_row)

        model_row = QHBoxLayout()
        lbl_model = QLabel("Modelo:")
        lbl_model.setFixedWidth(90)
        lbl_model.setStyleSheet("color: #4CAF50; font-weight: bold;")
        txt_model = QLineEdit()
        txt_model.setPlaceholderText("Ruta al modelo .gguf...")
        txt_model.setMinimumHeight(30)
        w.model_edit = txt_model
        btn_browse = QPushButton("Examinar...")
        btn_browse.setFixedWidth(90)
        btn_browse.setFixedHeight(30)
        btn_browse.clicked.connect(lambda: self.browse_model_for_widget(txt_model))
        model_row.addWidget(lbl_model)
        model_row.addWidget(txt_model, stretch=1)
        model_row.addWidget(btn_browse)
        main_layout.addLayout(model_row)

        mode_row = QHBoxLayout()
        lbl_mode = QLabel("Modo:")
        lbl_mode.setFixedWidth(90)
        lbl_mode.setStyleSheet("color: #4CAF50; font-weight: bold;")
        mode_row.addWidget(lbl_mode)
        mode_buttons = []
        for mode_key, mode_label in [
            ("auto", "Auto"),
            ("manual", "Manual"),
            ("gpu_only", "Solo GPU"),
            ("cpu_only", "Solo CPU"),
            ("hybrid", "Hibrido"),
        ]:
            btn = QPushButton(mode_label)
            btn.setFixedHeight(30)
            btn.setMinimumWidth(100)
            btn.clicked.connect(lambda checked, m=mode_key, w=w: self.apply_mode_to_tab(m, w))
            mode_buttons.append(btn)
            mode_row.addWidget(btn)
        w.mode_buttons = mode_buttons
        w.current_mode = "auto"
        mode_row.addStretch()
        main_layout.addLayout(mode_row)

        params_panel = ParamsPanel(self.llama_params)
        w.params_panel = params_panel
        main_layout.addWidget(params_panel, stretch=1)

        w.setLayout(main_layout)
        self.profile_tabs.addTab(w, profile_name)

        if profile_data:
            if hasattr(w, 'server_edit'):
                w.server_edit.setText(profile_data.get("server_path", DEFAULT_SERVER_PATH))
            if hasattr(w, 'model_edit'):
                w.model_edit.setText(profile_data.get("model_path", ""))
            if hasattr(w, 'params_panel'):
                w.params_panel.set_enabled_params(profile_data.get("params", {}))
            mode = profile_data.get("mode", "auto")
            w.current_mode = mode
            self._highlight_mode_button(w, mode)

    def browse_server_for_widget(self, line_edit):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar llama-server.exe", "",
            "Ejecutables (*.exe);;All Files (*)"
        )
        if fn:
            line_edit.setText(fn)
            self.config["server_path"] = fn
            save_config(self.config)
            self.log(f"[INFO] Servidor seleccionado: {fn}")
            self._load_params_for_server(fn)

    def browse_model_for_widget(self, line_edit):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Modelo", "",
            "GGUF Files (*.gguf);;All Files (*)"
        )
        if fn:
            line_edit.setText(fn)
            self.config["last_model"] = fn
            save_config(self.config)
            tab = self.profile_tabs.widget(self.current_index)
            if tab and getattr(tab, 'current_mode', 'auto') == "auto":
                self.log(f"[INFO] Modelo seleccionado: {os.path.basename(fn)}")
                self.apply_mode_to_tab("auto", tab)

    def start_server(self):
        if self.current_index is None or self.current_index < 0:
            self.log("[WARN] Sin perfil seleccionado.")
            return
        tab = self.profile_tabs.widget(self.current_index)
        if not tab:
            return
        model_path = tab.model_edit.text() if hasattr(tab, 'model_edit') else ""
        server_path = tab.server_edit.text() if hasattr(tab, 'server_edit') else DEFAULT_SERVER_PATH
        params = tab.params_panel.get_enabled_params() if hasattr(tab, 'params_panel') else {}

        if not server_path or not os.path.exists(server_path):
            self.log(f"[WARN] Servidor no encontrado: '{server_path}'")
            return
        if not model_path or not os.path.exists(model_path):
            self.log("[WARN] Modelo no encontrado.")
            return

        cmd = [server_path, "-m", model_path]
        param_map = {p.key: p for p in self.llama_params}
        for key, value in params.items():
            if key not in param_map or key == "model":
                continue
            param = param_map[key]
            flag = param.primary_long
            if param.arg_type == "bool":
                pos = next((ln for ln in param.long_names if not ln.startswith("--no-")), None)
                neg = next((ln for ln in param.long_names if ln.startswith("--no-")), None)
                if value:
                    cmd.append(pos if pos else flag)
                else:
                    if neg:
                        cmd.append(neg)
            else:
                cmd.append(flag)
                cmd.append(str(value))

        vram_total = GPU_VRAM_GB * GPU_COUNT
        if vram_total > 0:
            self.log(f"[INFO] VRAM total: {vram_total:.1f}GB ({GPU_COUNT} GPU)")
        model_size = self._estimate_model_size(model_path)
        if model_size:
            self.log(f"[INFO] Modelo: {model_size:.1f}GB")

        self.log(f"\n[CMD] {' '.join(cmd)}")
        port = params.get("port", 8080)
        self.log(f"[INFO] Iniciando en puerto {port}...")

        self._extract_profile_data(tab, self.current_index)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.server_thread = WorkerThread(cmd)
        self.server_thread.output_signal.connect(self.log)
        self.server_thread.finished_signal.connect(self.on_server_stopped)
        self.server_thread.start()

    def _estimate_model_size(self, model_path):
        try:
            if os.path.exists(model_path):
                return os.path.getsize(model_path) / (1024 ** 3)
        except Exception:
            pass
        return None

    def stop_server(self):
        if self.server_thread:
            self.log("Deteniendo servidor...")
            self.server_thread.stop()

    def on_server_stopped(self, success):
        self.log("[STOP] Servidor detenido.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def closeEvent(self, event):
        if self.current_index is not None:
            tab = self.profile_tabs.widget(self.current_index)
            if tab:
                self._extract_profile_data(tab, self.current_index)
        save_profiles(self.profiles)
        save_config(self.config)
        if self.server_thread:
            self.server_thread.stop()
        self.monitor_thread.stop()
        super().closeEvent(event)

    def update_telemetry(self, data):
        cpu = data["cpu"]
        c = "#FF5252" if cpu >= 80 else ("#FFB74D" if cpu >= 60 else "#4CAF50")
        if data["cpu_temp"] is not None:
            self.lbl_cpu.setText(f"[CPU] {cpu:.1f}%  |  {data['cpu_temp']}C  |  {psutil.cpu_count(logical=True)} cores")
        else:
            self.lbl_cpu.setText(f"[CPU] {cpu:.1f}%  |  {psutil.cpu_count(logical=True)} cores")
        self.lbl_cpu.setStyleSheet(f"font-size: 14px; color: {c};")

        r = data
        rc = "#FF5252" if r["ram_pct"] >= 85 else ("#FFB74D" if r["ram_pct"] >= 70 else "#4CAF50")
        self.lbl_ram.setText(f"[RAM] {r['ram_used_gb']:.1f}/{r['ram_total_gb']:.1f}GB ({r['ram_pct']:.0f}%)")
        self.lbl_ram.setStyleSheet(f"font-size: 14px; color: {rc};")

        dc = "#FF5252" if data["disk_pct"] >= 90 else ("#FFB74D" if data["disk_pct"] >= 80 else "#4CAF50")
        disk_parts = [f"[DISK] {data['disk_pct']:.0f}%"]
        if data.get("disk_temps"):
            temps = ", ".join(f"{k}: {v}" for k, v in data["disk_temps"].items())
            disk_parts.append(temps)
        self.lbl_disk.setText("  |  ".join(disk_parts))
        self.lbl_disk.setStyleSheet(f"font-size: 14px; color: {dc};")

        self.lbl_net.setText(f"[NET] Up {data['net_sent']:.1f}MB  Down {data['net_recv']:.1f}MB")

        self.lbl_gpu_placeholder.setVisible(not data["gpus"])
        for i, g in enumerate(data["gpus"]):
            tc = "#FF5252" if g["temp"] >= 85 else ("#FFB74D" if g["temp"] >= 70 else "#4CAF50")
            label_text = (
                f"[GPU {i}] {g['name']}  |  {g['temp']}C | "
                f"VRAM: {g['vram_used']:.1f}/{g['vram_total']:.1f}GB | "
                f"{g['power']:.1f}W | {g['util']}%"
            )
            if i >= len(self.gpu_containers):
                lbl = QLabel(label_text)
                lbl.setStyleSheet(f"font-size: 13px; color: {tc};")
                self.gpu_containers.append(lbl)
                self.hw_layout.addWidget(lbl)
            else:
                self.gpu_containers[i].setText(label_text)
                self.gpu_containers[i].setStyleSheet(f"font-size: 13px; color: {tc};")

        while len(self.gpu_containers) > len(data["gpus"]):
            removed = self.gpu_containers.pop()
            self.hw_layout.removeWidget(removed)
            removed.deleteLater()

    def log(self, msg):
        self.console.append(msg)
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console.setTextCursor(cursor)
        doc = self.console.document()
        while doc.blockCount() > self._console_max_lines:
            cursor = QTextCursor(doc.begin())
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()


def main():
    app = QApplication(sys.argv)
    dashboard = UnifiedDashboard()
    dashboard.show()
    return app.exec()

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            input()
        except Exception:
            pass
        sys.exit(1)

