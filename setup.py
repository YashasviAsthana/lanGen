#!/usr/bin/env python3
"""
lanGen setup wizard — Linux first-time installer.

Guides the user through:
  1. Detecting GPU and CUDA availability
  2. Downloading the right llama.cpp llama-server binary
  3. Choosing and downloading a GGUF language model
  4. Writing lanGen.ini so agent.py spawns llama-server automatically

Run:  python setup.py
"""

import configparser
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
BIN_DIR = ROOT / "bin"
MODELS_DIR = ROOT / "models"
CONFIG_FILE = ROOT / "lanGen.ini"

LLAMA_CPP_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

# ── Terminal helpers ──────────────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


def _ok(msg: str) -> None:
    print(f"  {_c(GREEN, '✓')} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_c(YELLOW, '!')} {msg}")


def _err(msg: str) -> None:
    print(f"  {_c(RED, '✗')} {msg}", file=sys.stderr)


def _step(n: int, msg: str) -> None:
    print(f"\n{_c(BOLD, f'[{n}]')} {_c(BOLD, msg)}")


def _ask(prompt: str) -> str:
    return input(f"  {_c(CYAN, '?')} {prompt} ").strip()


def _header() -> None:
    print(f"""
{_c(BOLD, "╔══════════════════════════════════════════╗")}
{_c(BOLD, "║")}         {_c(CYAN + BOLD, "lanGen  setup wizard")}            {_c(BOLD, " ║")}
{_c(BOLD, "╚══════════════════════════════════════════╝")}
Sets up llama.cpp and downloads a language model.
Run this once before starting agent.py for the first time.
""")


# ── GPU detection ─────────────────────────────────────────────────────────────


def _run(cmd: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _detect_nvidia_gpu() -> str | None:
    """Return GPU name if an NVIDIA GPU is present in hardware, else None."""
    r = _run(["lspci"])
    if r and r.returncode == 0:
        for line in r.stdout.splitlines():
            if re.search(r"nvidia|geforce|quadro|tesla", line, re.I):
                m = re.search(r":\s*(.+)$", line)
                return m.group(1).strip() if m else "NVIDIA GPU"
    # Fallback: kernel driver directory or device files
    if Path("/proc/driver/nvidia").is_dir():
        return "NVIDIA GPU"
    nvidia_devs = list(Path("/dev").glob("nvidia[0-9]*"))
    if nvidia_devs:
        return "NVIDIA GPU"
    return None


def _detect_cuda_driver() -> tuple[bool, str | None]:
    """
    Return (driver_ok, cuda_version_string).
    driver_ok means nvidia-smi is present and working.
    cuda_version is e.g. "12.4" from the nvidia-smi header line.
    """
    r = _run(["nvidia-smi"])
    if r is None or r.returncode != 0:
        return False, None
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", r.stdout)
    return True, (m.group(1) if m else None)


def _detect_amd_gpu() -> str | None:
    r = _run(["lspci"])
    if r and r.returncode == 0:
        for line in r.stdout.splitlines():
            if re.search(r"\b(amd|radeon)\b", line, re.I) and re.search(
                r"vga|display|3d", line, re.I
            ):
                m = re.search(r":\s*(.+)$", line)
                return m.group(1).strip() if m else "AMD GPU"
    return None


def _gpu_survey() -> dict:
    nvidia = _detect_nvidia_gpu()
    amd = _detect_amd_gpu()
    driver_ok, cuda_ver = _detect_cuda_driver() if nvidia else (False, None)
    return {
        "nvidia_name": nvidia,
        "amd_name": amd,
        "cuda_driver_ok": driver_ok,
        "cuda_version": cuda_ver,
    }


# ── llama.cpp download ────────────────────────────────────────────────────────


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "lanGen-setup/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


def _pick_asset(assets: list, build: str, cuda_ver: str | None) -> dict | None:
    """
    Pick the best matching release asset for Linux.
    build: "cuda" | "vulkan" | "cpu"
    """
    candidates = []
    for asset in assets:
        name = asset["name"].lower()
        if build == "cuda":
            if "linux" not in name or "cuda" not in name:
                continue
            # Prefer matching major CUDA version; fall back to any cuda build
            if cuda_ver:
                major = cuda_ver.split(".")[0]
                priority = 2 if f"cu{major}" in name else 1
            else:
                priority = 1
            candidates.append((priority, asset))
        elif build == "vulkan":
            if "linux" in name and "vulkan" in name:
                candidates.append((1, asset))
        else:  # cpu
            is_linux = "ubuntu" in name or "linux" in name
            is_cpu = "cuda" not in name and "vulkan" not in name and "rocm" not in name
            is_x64 = "x64" in name or "x86_64" in name
            if is_linux and is_cpu and is_x64:
                candidates.append((1, asset))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _download_with_progress(url: str, dest: Path, label: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "lanGen-setup/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with dest.open("wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    mb = downloaded / 1_048_576
                    print(f"\r  {label}  {pct:3d}%  {mb:6.0f} MB", end="", flush=True)
    print()  # newline after progress


def _install_llama_cpp(build: str, cuda_ver: str | None) -> Path:
    print("  Fetching latest llama.cpp release info...")
    try:
        data = _fetch_json(LLAMA_CPP_API)
    except Exception as e:
        raise RuntimeError(f"GitHub API request failed: {e}") from e

    version = data["tag_name"]
    assets = data.get("assets", [])
    asset = _pick_asset(assets, build, cuda_ver)
    if not asset:
        raise RuntimeError(
            f"No {build} binary found in llama.cpp release {version}.\n"
            "  Check https://github.com/ggml-org/llama.cpp/releases manually."
        )

    size_mb = asset["size"] / 1_048_576
    print(f"  Release: {version}  →  {asset['name']}  ({size_mb:.0f} MB)")
    choice = _ask("Download? [Y/n]")
    if choice.lower() in ("n", "no"):
        raise RuntimeError("Download cancelled.")

    BIN_DIR.mkdir(exist_ok=True)
    zip_path = BIN_DIR / asset["name"]
    _download_with_progress(asset["browser_download_url"], zip_path, asset["name"])

    # Extract llama-server from the zip
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if Path(member).name == "llama-server":
                target = BIN_DIR / "llama-server"
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                target.chmod(0o755)
                _ok(f"Extracted llama-server → {target}")
                break
        else:
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(f"llama-server not found inside {asset['name']}")

    zip_path.unlink(missing_ok=True)
    return BIN_DIR / "llama-server"


# ── Model selection & download ─────────────────────────────────────────────────

# Filenames verified against official Qwen/Qwen3-*-GGUF repos on HuggingFace.
MODELS = [
    dict(
        idx=1,
        label="Qwen3-1.7B  Q8_0   ~2 GB  (fast, small)",
        repo="Qwen/Qwen3-1.7B-GGUF",
        file="Qwen3-1.7B-Q8_0.gguf",
        size_gb=2.0,
    ),
    dict(
        idx=2,
        label="Qwen3-4B   Q6_K   ~3.5 GB (balanced)",
        repo="Qwen/Qwen3-4B-GGUF",
        file="Qwen3-4B-Q6_K.gguf",
        size_gb=3.5,
    ),
    dict(
        idx=3,
        label="Qwen3-8B   Q4_K_M ~5 GB   (recommended ★)",
        repo="Qwen/Qwen3-8B-GGUF",
        file="Qwen3-8B-Q4_K_M.gguf",
        size_gb=5.0,
    ),
    dict(
        idx=4,
        label="Qwen3-14B  Q4_K_M ~9 GB",
        repo="Qwen/Qwen3-14B-GGUF",
        file="Qwen3-14B-Q4_K_M.gguf",
        size_gb=9.0,
    ),
    dict(
        idx=5,
        label="Qwen3-32B  Q4_K_M ~20 GB  (best quality)",
        repo="Qwen/Qwen3-32B-GGUF",
        file="Qwen3-32B-Q4_K_M.gguf",
        size_gb=20.0,
    ),
]


def _choose_model() -> dict:
    print()
    for m in MODELS:
        print(f"  {_c(BOLD, str(m['idx']))}. {m['label']}")
    print()
    while True:
        raw = _ask("Select a model [1-5, default 3]:")
        choice = raw or "3"
        if choice.isdigit() and 1 <= int(choice) <= 5:
            return MODELS[int(choice) - 1]
        _err("Enter a number between 1 and 5.")


def _download_model(model: dict) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / model["file"]
    if dest.exists():
        _ok(f"Already downloaded: {dest.name}")
        return dest

    print(f"  Downloading {model['file']}  (~{model['size_gb']:.0f} GB)...")

    # Prefer huggingface_hub (shows a nice tqdm bar); fall back to urllib.
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=model["repo"],
            filename=model["file"],
            local_dir=str(MODELS_DIR),
        )
        return Path(path)
    except ImportError:
        pass

    url = f"https://huggingface.co/{model['repo']}/resolve/main/{model['file']}"
    try:
        _download_with_progress(url, dest, model["file"])
    except urllib.error.HTTPError as e:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download failed ({e.code}). Check that the filename is correct:\n  https://huggingface.co/{model['repo']}"
        ) from e
    return dest


# ── Existing-install resolvers ────────────────────────────────────────────────


def _resolve_llama_binary(
    build: str, cuda_ver: str | None, existing_bin: str = ""
) -> Path:
    """
    Find llama-server: check system PATH and local bin/ first.
    Let the user point to an existing install before offering to download.
    """
    candidates: list[tuple[str, Path]] = []

    # Existing config value
    if existing_bin and Path(existing_bin).exists():
        candidates.append(("existing config", Path(existing_bin)))

    system_bin = shutil.which("llama-server")
    if system_bin:
        candidates.append(("system PATH", Path(system_bin)))

    local_bin = BIN_DIR / "llama-server"
    if local_bin.exists() and local_bin not in [p for _, p in candidates]:
        candidates.append(("local bin/", local_bin))

    if candidates:
        print()
        for i, (label, p) in enumerate(candidates, 1):
            tag = " (current)" if label == "existing config" else ""
            print(f"  {_c(BOLD, str(i))}. {p}{tag} ({label})")
        print()
        print(
            f"  {_c(BOLD, str(len(candidates) + 1))}. Enter a different path manually"
        )
        print(
            f"  {_c(BOLD, str(len(candidates) + 2))}. Download the latest release from GitHub"
        )
        print()
        while True:
            choice = _ask(f"Choose [1–{len(candidates) + 2}, default 1]:")
            choice = choice or "1"
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                chosen = candidates[int(choice) - 1][1]
                _ok(f"Using: {chosen}")
                return chosen
            if choice == str(len(candidates) + 1):
                break
            if choice == str(len(candidates) + 2):
                return _install_llama_cpp(build, cuda_ver)
            _err(f"Enter 1–{len(candidates) + 2}.")
    else:
        print("  llama-server not found in PATH or local bin/.")
        print()
        print(f"  {_c(BOLD, '1')}. Enter path to existing llama-server binary")
        print(f"  {_c(BOLD, '2')}. Download the latest release from GitHub")
        print()
        while True:
            choice = _ask("Choose [1/2, default 2]:")
            choice = choice or "2"
            if choice == "1":
                break
            if choice == "2":
                return _install_llama_cpp(build, cuda_ver)
            _err("Enter 1 or 2.")

    # Manual path entry
    while True:
        raw = _ask("Path to llama-server binary:")
        p = Path(raw).expanduser()
        if p.exists() and os.access(p, os.X_OK):
            _ok(f"Using: {p}")
            return p
        _err(f"Not found or not executable: {p}")


def _resolve_model(existing_model: str = "") -> Path:
    """
    Find a .gguf model: check ./models/ first, then let the user point
    to any path on the system, or download a fresh one.
    """
    local_models = sorted(MODELS_DIR.glob("*.gguf")) if MODELS_DIR.exists() else []

    # Build candidate list: existing model first, then local models
    candidates: list[tuple[Path, str]] = []  # (path, label)
    ep = Path(existing_model).expanduser() if existing_model else None
    if ep and ep.exists() and ep.suffix == ".gguf":
        candidates.append((ep, " (current)"))
    for p in local_models:
        tag = " (current)" if ep == p else ""
        candidates.append((p, tag))

    if candidates:
        print()
        print("  Models found:")
        for i, (p, tag) in enumerate(candidates, 1):
            size = p.stat().st_size / 1_073_741_824
            print(f"  {_c(BOLD, str(i))}. {p.name}  ({size:.1f} GB){tag}")
        print()

    print(f"  {_c(BOLD, 'p')}. Enter path to a model already on this machine")
    print(f"  {_c(BOLD, 'd')}. Download a model from HuggingFace")
    if candidates:
        print(
            f"  {_c(BOLD, '1')}–{_c(BOLD, str(len(candidates)))}. Use one of the models above"
        )
    print()

    while True:
        prompt = (
            f"Choose [1–{len(candidates)}/p/d, default 1]:"
            if candidates
            else "Choose [p/d, default d]:"
        )
        raw = _ask(prompt) or ("1" if candidates else "d")

        if raw.lower() == "d":
            model = _choose_model()
            return _download_model(model)

        if raw.lower() == "p":
            while True:
                path_raw = _ask("Full path to .gguf file:")
                p = Path(path_raw).expanduser()
                if p.exists() and p.suffix == ".gguf":
                    _ok(f"Using: {p}")
                    return p
                _err(f"File not found or not a .gguf: {p}")

        if raw.isdigit() and candidates and 1 <= int(raw) <= len(candidates):
            chosen = candidates[int(raw) - 1][0]
            _ok(f"Using: {chosen}")
            return chosen

        _err("Invalid choice.")


_COMFY_SEARCH_PATHS = [
    Path.home() / "ComfyUI",
    Path.home() / "comfyui",
    Path.home() / "stable-diffusion-webui",
    Path("/opt/ComfyUI"),
    Path("/opt/comfyui"),
]


def _resolve_comfyui(
    existing_workspace: str = "", existing_port: str = "7860"
) -> tuple[str, str]:
    """
    Find ComfyUI workspace and port.
    Returns (workspace_path_str, port_str).
    """
    candidates: list[Path] = []

    # Existing config value
    if existing_workspace and (Path(existing_workspace) / "main.py").exists():
        candidates.append(Path(existing_workspace))

    # Check common install locations
    for p in _COMFY_SEARCH_PATHS:
        if (p / "main.py").exists() and p not in candidates:
            candidates.append(p)

    # Check local ./comfyui
    local = ROOT / "comfyui"
    if (local / "main.py").exists() and local not in candidates:
        candidates.append(local)

    if candidates:
        print()
        for i, p in enumerate(candidates, 1):
            tag = (
                " (current)"
                if existing_workspace and Path(existing_workspace).expanduser() == p
                else ""
            )
            print(f"  {_c(BOLD, str(i))}. {p}{tag}")
        print()
    else:
        print("  ComfyUI not found in common locations.")
        print()

    print(f"  {_c(BOLD, 'p')}. Enter path to existing ComfyUI install")
    print(f"  {_c(BOLD, 's')}. Skip (ComfyUI features will be unavailable)")
    if candidates:
        print(
            f"  {_c(BOLD, '1')}–{_c(BOLD, str(len(candidates)))}. Use a detected path above"
        )
    print()

    workspace = ""
    while True:
        prompt = (
            f"Choose [1–{len(candidates)}/p/s, default 1]:"
            if candidates
            else "Choose [p/s, default s]:"
        )
        raw = _ask(prompt) or ("1" if candidates else "s")

        if raw.lower() == "s":
            _warn("Skipping ComfyUI — image/video tools will not be available.")
            workspace = ""
            break

        if raw.lower() == "p":
            while True:
                path_raw = _ask("Full path to ComfyUI directory:")
                p = Path(path_raw).expanduser()
                if (p / "main.py").exists():
                    workspace = str(p)
                    _ok(f"Using: {p}")
                    break
                _err(f"main.py not found in {p} — is this a ComfyUI directory?")
            break

        if raw.isdigit() and candidates and 1 <= int(raw) <= len(candidates):
            workspace = str(candidates[int(raw) - 1])
            _ok(f"Using: {workspace}")
            break

        _err("Invalid choice.")

    port = existing_port
    if workspace:
        raw_port = _ask(f"ComfyUI port [default {port}]:") or port
        port = raw_port

    return workspace, port


# ── Config writer ─────────────────────────────────────────────────────────────


def _ask_mmproj(existing_mmproj: str = "") -> str:
    """Optionally ask for a multimodal projection file (enables vision)."""
    if existing_mmproj and Path(existing_mmproj).exists():
        print()
        print(f"  Current mmproj: {_c(BOLD, existing_mmproj)}")
        choice = _ask("Keep this mmproj? [Y/n]")
        if choice.lower() in ("n", "no"):
            while True:
                path = _ask("Path to mmproj .gguf file:")
                if Path(path).exists():
                    return path
                _err(f"File not found: {path}")
            return ""  # unreachable but satisfies type checker
        print(f"  {_c(GREEN, '✓')} Keeping: {existing_mmproj}")
        return existing_mmproj

    print()
    print("  Multimodal (vision) models require a separate --mmproj file.")
    choice = _ask("Do you have an mmproj file for your model? [y/N]")
    if choice.lower() not in ("y", "yes"):
        return ""
    while True:
        path = _ask("Path to mmproj .gguf file:")
        if Path(path).exists():
            return path
        _err(f"File not found: {path}")


def _write_config(
    llama_bin: Path,
    model_path: Path,
    gpu_layers: int,
    build: str,
    mmproj: str = "",
    comfy_workspace: str = "",
    comfy_port: str = "7860",
) -> None:
    cfg = configparser.ConfigParser()
    # Preserve hand-edited sections such as the selected search provider.
    cfg.read(CONFIG_FILE)
    cfg["llama"] = {
        "binary": str(llama_bin),
        "model": str(model_path),
        "mmproj": mmproj,
        "gpu_layers": str(gpu_layers),
        "ctx_size": "8000",
        "n_parallel": "1",
        "flash_attn": "true",
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "temp": "0.6",
        "top_p": "0.95",
        "top_k": "20",
        "min_p": "0.0",
        "presence_penalty": "1.0",
        "repeat_penalty": "1.0",
        "reasoning_budget": "-1",
        "image_max_tokens": "1120",
        "port": "8080",
        "build": build,
        "managed": "true",
    }
    cfg["comfyui"] = {
        "workspace": comfy_workspace or str(ROOT / "comfyui"),
        "port": comfy_port,
        "python": "python3",
        "managed": "true",
    }
    cfg["huggingface"] = {
        "token": "",
    }
    with CONFIG_FILE.open("w") as f:
        cfg.write(f)
    _ok(f"Config written → {CONFIG_FILE}")


# ── Main flow ─────────────────────────────────────────────────────────────────


def _cuda_install_instructions() -> None:
    print(f"""
  To enable GPU acceleration, install the NVIDIA CUDA driver:

  {_c(DIM, "# Ubuntu / Debian")}
  {_c(BOLD, "sudo apt-get install -y nvidia-driver-570 nvidia-cuda-toolkit")}

  Or use the official installer wizard (recommended for non-Ubuntu distros):
  {_c(CYAN, "https://developer.nvidia.com/cuda-downloads")}

  After installing, {_c(BOLD, "reboot")} and re-run this setup to enable GPU acceleration.
""")


def _install_workflow_models(comfy_workspace: str) -> None:
    """Download missing ComfyUI workflow models declared in comfy_workflows/*.yaml."""
    import yaml

    workflows_dir = ROOT / "comfy_workflows"
    models_root = Path(comfy_workspace) / "models"

    yaml_files = sorted(workflows_dir.glob("*.yaml"))
    if not yaml_files:
        print("  No workflow YAML files found.")
        return

    # Collect all missing files across all workflows
    to_download: list[dict] = []
    for yf in yaml_files:
        try:
            with open(yf) as f:
                sidecar = yaml.safe_load(f)
        except Exception:
            continue
        wf_name = sidecar.get("name", yf.stem)
        for spec in sidecar.get("downloads", []):
            filename = spec.get("filename", "")
            dest = spec.get("dest", "")
            if not (filename and dest):
                continue
            dest_file = models_root / dest / filename
            if dest_file.exists():
                continue
            # Handle files previously downloaded to the wrong split_files path
            hf_path = spec.get("hf_path", filename)
            misplaced = models_root / dest / hf_path
            if misplaced.exists() and misplaced.resolve() != dest_file.resolve():
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                misplaced.rename(dest_file)
                for parent in list(misplaced.parents):
                    if parent.resolve() == (models_root / dest).resolve():
                        break
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                _ok(f"Moved {filename} to correct location")
                continue
            to_download.append(
                {
                    "workflow": wf_name,
                    "hf_repo": spec.get("hf_repo", ""),
                    "hf_path": spec.get("hf_path", filename),
                    "filename": filename,
                    "dest": dest,
                    "size": spec.get("size", "?"),
                    "dest_file": dest_file,
                }
            )

    if not to_download:
        _ok("All workflow models are already present.")
        return

    total_by_workflow: dict[str, list] = {}
    for spec in to_download:
        total_by_workflow.setdefault(spec["workflow"], []).append(spec)

    print(f"\n  {len(to_download)} model file(s) need to be downloaded:\n")
    print("  Select which files to download:")
    for i, spec in enumerate(to_download, 1):
        wf_tag = f" [{spec['workflow']}]" if len(total_by_workflow) > 1 else ""
        print(f"    {_c(BOLD, str(i))}. {spec['filename']}  ({spec['size']}){wf_tag}")
    print(f"\n    {_c(BOLD, 'a')}. All of the above")
    print(f"    {_c(BOLD, 'q')}. Quit (skip download)")
    print()

    answer = _ask("Enter choice(s) — comma-separated, e.g. `1,3,a`:")
    if answer.strip().lower() == "q":
        print("  Skipped. Run `python setup.py --models` again when ready.")
        return

    if answer.strip().lower() == "a":
        selected = list(range(1, len(to_download) + 1))
    else:
        selected = []
        for token in answer.split(","):
            token = token.strip()
            if token.isdigit() and 1 <= int(token) <= len(to_download):
                selected.append(int(token))
        if not selected:
            print("  No valid selections — skipped.")
            return

    chosen = [to_download[i - 1] for i in selected]

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        _err("huggingface_hub is not installed. Run: pip install huggingface_hub")
        return

    hf_token = None
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_FILE)
    if cfg.has_option("huggingface", "token"):
        hf_token = cfg["huggingface"]["token"] or None

    for i, spec in enumerate(chosen, 1):
        print(
            f"  [{i}/{len(chosen)}] Downloading {spec['filename']}  ({spec['size']})..."
        )
        dest_dir = models_root / spec["dest"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id=spec["hf_repo"],
                filename=spec["hf_path"],
                local_dir=str(dest_dir),
                token=hf_token,
            )
            # hf_hub_download preserves the full hf_path structure under local_dir.
            # Move the file flat into dest_dir so ComfyUI can find it.
            downloaded = dest_dir / spec["hf_path"]
            if (
                downloaded.exists()
                and downloaded.resolve() != spec["dest_file"].resolve()
            ):
                downloaded.rename(spec["dest_file"])
                # Remove any leftover empty intermediate directories (e.g. split_files/vae/)
                for parent in list(downloaded.parents):
                    if parent.resolve() == dest_dir.resolve():
                        break
                    try:
                        parent.rmdir()
                    except OSError:
                        break
            _ok(f"  {spec['filename']}")
        except Exception as e:
            _err(f"  Failed to download {spec['filename']}: {e}")

    print()
    _ok("Model installation complete.")


def main() -> None:
    if platform.system() != "Linux":
        print("This setup script currently supports Linux only.")
        sys.exit(1)

    _header()

    # ── Load existing config (if any) ───────────────────────────────────────
    _cfg = configparser.ConfigParser()
    _cfg.read(CONFIG_FILE)
    existing_bin = _cfg["llama"].get("binary", "") if "llama" in _cfg else ""
    existing_model = _cfg["llama"].get("model", "") if "llama" in _cfg else ""
    existing_mmproj = _cfg["llama"].get("mmproj", "") if "llama" in _cfg else ""
    existing_comfy_workspace = (
        _cfg["comfyui"].get("workspace", "") if "comfyui" in _cfg else ""
    )
    existing_comfy_port = (
        _cfg["comfyui"].get("port", "7860") if "comfyui" in _cfg else "7860"
    )

    if _cfg.has_section("llama"):
        print(
            f"  {_c(GREEN, '✓')} Found existing lanGen.ini — showing current values as defaults."
        )
        print()

    # ── Step 1: GPU detection ───────────────────────────────────────────────
    _step(1, "Detecting GPU...")
    gpu = _gpu_survey()

    build = "cpu"
    gpu_layers = 0

    if gpu["nvidia_name"]:
        print(f"  Found: {_c(GREEN, gpu['nvidia_name'])}")

        if gpu["cuda_driver_ok"]:
            _ok(f"CUDA drivers installed  (CUDA {gpu['cuda_version']})")
            build = "cuda"
            gpu_layers = 99

        else:
            _warn("CUDA GPU detected but CUDA drivers are NOT installed.")
            _cuda_install_instructions()
            print("  Options:")
            print(f"  {_c(BOLD, '1')}. Show CUDA install instructions and exit")
            print(f"  {_c(BOLD, '2')}. Continue with CPU-only build (GPU unused)")
            print()
            while True:
                choice = _ask("Choose [1/2]:")
                if choice == "1":
                    _cuda_install_instructions()
                    print("  Re-run this setup after installing CUDA drivers.")
                    sys.exit(0)
                elif choice == "2":
                    build = "cpu"
                    gpu_layers = 0
                    _warn("Continuing with CPU-only build — inference will be slower.")
                    break
                _err("Enter 1 or 2.")

    elif gpu["amd_name"]:
        print(f"  Found: {_c(GREEN, gpu['amd_name'])}")
        print("  Using Vulkan build (works with AMD and Intel Arc GPUs).")
        build = "vulkan"
        gpu_layers = 99

    else:
        _warn("No discrete GPU detected — using CPU-only build.")
        build = "cpu"
        gpu_layers = 0

    # ── Step 2: llama-server binary ─────────────────────────────────────────
    _step(2, "Locating llama-server...")
    llama_bin = _resolve_llama_binary(build, gpu["cuda_version"], existing_bin)

    # ── Step 3: Language model ──────────────────────────────────────────────
    _step(3, "Locating language model (.gguf)...")
    model_path = _resolve_model(existing_model)

    # ── Step 4: Multimodal projection file (optional) ──────────────────────
    _step(4, "Multimodal / vision support (optional)...")
    mmproj = _ask_mmproj(existing_mmproj)
    if mmproj:
        _ok(f"mmproj: {mmproj}")
    else:
        print("  Skipping — text-only mode.")

    # ── Step 5: ComfyUI (image / video generation) ──────────────────────────
    _step(5, "Locating ComfyUI...")
    comfy_workspace, comfy_port = _resolve_comfyui(
        existing_comfy_workspace, existing_comfy_port
    )

    # ── Step 6: Write config ────────────────────────────────────────────────
    _step(6, "Writing lanGen.ini...")
    _write_config(
        llama_bin, model_path, gpu_layers, build, mmproj, comfy_workspace, comfy_port
    )

    # ── Step 7: ComfyUI workflow models (optional) ─────────────────────────
    if comfy_workspace:
        _step(7, "ComfyUI workflow models (optional)...")
        answer = (
            _ask(
                "Download missing workflow models now? (large files — can skip and run later) [y/N]:"
            )
            .strip()
            .lower()
        )
        if answer in ("y", "yes"):
            _install_workflow_models(comfy_workspace)
        else:
            print(
                f"  Skipped. Run  {_c(BOLD, 'python setup.py --models')}  whenever you're ready."
            )

    # ── Done ────────────────────────────────────────────────────────────────
    lan_url = "https://<your-LAN-ip>:7000"
    print(f"""
{_c(GREEN + BOLD, "Setup complete!")}

Start lanGen (llama-server launches automatically):
  {_c(BOLD, "python agent.py")}

Open in a browser:
  {_c(CYAN, "https://localhost:7000")}  or  {_c(CYAN, lan_url)}
  {_c(DIM, "(self-signed cert — click Advanced → Proceed)")}
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="lanGen setup wizard")
    parser.add_argument(
        "--models",
        action="store_true",
        help="Download missing ComfyUI workflow models and exit",
    )
    args = parser.parse_args()

    if args.models:
        cfg = configparser.ConfigParser()
        cfg.read(CONFIG_FILE)
        workspace = (
            cfg["comfyui"].get("workspace", "") if cfg.has_section("comfyui") else ""
        )
        if not workspace:
            _err(
                "comfyui.workspace not set in lanGen.ini. Run `python setup.py` first."
            )
            sys.exit(1)
        try:
            _install_workflow_models(workspace)
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            sys.exit(0)
        sys.exit(0)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)
    except RuntimeError as e:
        _err(str(e))
        sys.exit(1)
