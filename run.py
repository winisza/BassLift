"""BassLift one-click launcher.

Uruchom dwuklikiem BassLift.command (macOS/Linux) lub BassLift.bat (Windows),
albo ręcznie: python run.py [--no-browser]

- Pierwsze uruchomienie: tworzy .venv obok tego pliku i instaluje requirements.txt
  (kilka minut, PyTorch jest duży). Kolejne starty są natychmiastowe.
- Startuje serwer, otwiera przeglądarkę i kończy pracę sam, gdy zamkniesz
  ostatnią kartę z BassLift (i nic się już nie przetwarza).
"""
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
REQUIREMENTS = ROOT_DIR / "requirements.txt"
INSTALL_MARKER = VENV_DIR / ".basslift-requirements.sha256"

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MIN_PYTHON = (3, 10)

# Moduły, bez których serwer nie ruszy (nazwy importów, nie pakietów pip)
REQUIRED_MODULES = ["fastapi", "uvicorn", "demucs", "librosa", "midiutil", "torchcrepe", "beat_this"]

STARTUP_GRACE = 300   # s — ile czekamy na pierwszą kartę przeglądarki
WATCHDOG_TICK = 2     # s


# ────────────────────────────────────────────────
# Środowisko: .venv + zależności
# ────────────────────────────────────────────────
def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def in_managed_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV_DIR.resolve()


def requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def deps_importable() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in REQUIRED_MODULES)


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def install_requirements(python: Path):
    print("\n=== BassLift: instaluję zależności (za pierwszym razem kilka minut) ===\n", flush=True)
    pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    subprocess.check_call(pip + ["--upgrade", "pip"])
    if sys.platform != "darwin" and has_nvidia_gpu():
        # Domyślne koła PyTorch z PyPI na Windows są CPU-only — dla NVIDIA bierzemy build CUDA
        print("Wykryto kartę NVIDIA — instaluję PyTorch z obsługą CUDA", flush=True)
        subprocess.check_call(pip + ["torch", "torchaudio",
                                     "--index-url", "https://download.pytorch.org/whl/cu126"])
    subprocess.check_call(pip + ["-r", str(REQUIREMENTS)])
    INSTALL_MARKER.write_text(requirements_hash())
    print("\n=== Zależności zainstalowane ===\n", flush=True)


def ensure_environment():
    """Zapewnia działające środowisko; w razie potrzeby przełącza się na .venv."""
    if in_managed_venv():
        if not INSTALL_MARKER.exists() or INSTALL_MARKER.read_text().strip() != requirements_hash():
            install_requirements(Path(sys.executable))
        return

    # Użytkownik uruchomił nas z własnego środowiska, w którym wszystko już jest
    if deps_importable():
        return

    if sys.version_info < MIN_PYTHON:
        print(f"BassLift wymaga Pythona {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ "
              f"(masz {sys.version.split()[0]}).")
        print("Pobierz aktualny Python ze strony https://www.python.org/downloads/ i uruchom ponownie.")
        sys.exit(1)

    if not venv_python().exists():
        # --clear naprawia .venv osierocone po aktualizacji/usunięciu systemowego Pythona
        print(f"Tworzę środowisko wirtualne w {VENV_DIR} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "venv", "--clear", str(VENV_DIR)])

    # Dalej działamy już interpreterem z .venv (on sam doinstaluje brakujące pakiety)
    sys.exit(subprocess.call([str(venv_python()), str(Path(__file__).resolve()), *sys.argv[1:]]))


# ────────────────────────────────────────────────
# Serwer
# ────────────────────────────────────────────────
def running_instance_url(port: int):
    """Adres już działającej instancji BassLift na danym porcie (albo None)."""
    url = f"http://{HOST}:{port}/"
    try:
        with urllib.request.urlopen(url + "health", timeout=1) as r:
            if json.load(r).get("app") == "basslift":
                return url
    except Exception:
        pass
    return None


def pick_port() -> int:
    for port in (DEFAULT_PORT, 0):  # 0 = dowolny wolny port od systemu
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((HOST, port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Brak wolnego portu")


def watchdog(server, presence):
    """Zatrzymuje serwer, gdy nikt nie ma otwartej karty i nic się nie liczy."""
    started = time.monotonic()
    while not server.should_exit:
        time.sleep(WATCHDOG_TICK)
        if not presence.seen_any():
            if time.monotonic() - started > STARTUP_GRACE:
                print("\nNikt nie otworzył BassLift w przeglądarce — zamykam.", flush=True)
                server.should_exit = True
            continue
        if presence.idle():
            print("\nKarta BassLift zamknięta — zamykam serwer.", flush=True)
            server.should_exit = True


def open_browser_when_ready(server, url):
    while not server.started and not server.should_exit:
        time.sleep(0.1)
    if server.started:
        webbrowser.open(url)


def main():
    ensure_environment()

    existing = running_instance_url(DEFAULT_PORT)
    if existing:
        print(f"BassLift już działa — otwieram {existing}")
        if "--no-browser" not in sys.argv:
            webbrowser.open(existing)
        return

    import uvicorn
    os.chdir(ROOT_DIR)
    sys.path.insert(0, str(ROOT_DIR))
    import server as basslift_server

    port = pick_port()
    url = f"http://{HOST}:{port}/"
    config = uvicorn.Config(basslift_server.app, host=HOST, port=port, log_level="info")
    server = uvicorn.Server(config)

    print(f"\nBassLift działa pod {url}")
    print("Zamknij kartę w przeglądarce, żeby wyłączyć aplikację (albo Ctrl+C tutaj).\n", flush=True)
    if "--no-browser" not in sys.argv:
        threading.Thread(target=open_browser_when_ready, args=(server, url), daemon=True).start()
    threading.Thread(target=watchdog, args=(server, basslift_server.presence), daemon=True).start()
    try:
        server.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
