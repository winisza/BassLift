"""BassLift jako natywna aplikacja macOS: serwer w wątku + okno WKWebView (pywebview).

Uruchamiane z BassLift.app (Briefcase) albo ręcznie: python -m basslift
Zamknięcie okna (lub Cmd+Q) kończy serwer i cały proces.

Modele trafiają do tych samych katalogów co przy starcie z BassLift.command
(~/.cache/...), więc wersja z launchera i aplikacja nie pobierają ich dwa razy.
Nic nie jest zapisywane do wnętrza podpisanej aplikacji.
"""
import base64
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

LOG_DIR = Path.home() / "Library" / "Logs" / "BassLift"
CACHE_DIR = Path.home() / "Library" / "Caches" / "BassLift"
# Stały port: localStorage (język, motyw, silnik) jest przypisany do origin razem z portem,
# więc przy losowym porcie ustawienia znikałyby po każdym uruchomieniu
APP_PORT = 47831

SPLASH = """<!doctype html><html><body style="margin:0;height:100vh;display:flex;align-items:center;
justify-content:center;background:#111520;color:#918e88;font:15px -apple-system,sans-serif">
<div style="text-align:center"><div style="font:600 30px -apple-system,sans-serif;color:#c0956c">Bass/Lift</div>
<div style="margin-top:12px">Uruchamianie…</div></div></body></html>"""

log = logging.getLogger("basslift")


def _configure():
    # .pyc są prekompilowane przy budowie; zapis w działającej aplikacji łamałby jej podpis
    sys.dont_write_bytecode = True
    os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR / "numba"))  # librosa: @jit(cache=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        handlers=[logging.FileHandler(LOG_DIR / "basslift.log"), logging.StreamHandler()])


def _pick_port() -> int:
    for port in (APP_PORT, 0):  # 0 = dowolny wolny, gdy stały jest zajęty
        with socket.socket() as s:
            # jak uvicorn: połączenia w TIME_WAIT po poprzednim uruchomieniu nie blokują portu
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Brak wolnego portu")


class Bridge:
    """Funkcje wołane z JS jako window.pywebview.api.* — WKWebView nie obsługuje
    pobierania plików jak przeglądarka, więc zapis idzie przez natywne okno „Zapisz jako”."""

    def __init__(self, base_url: str):
        self._base_url = base_url
        self._window = None  # podkreślnik: pywebview nie wystawia tego do JS

    def _target(self, filename: str) -> Optional[Path]:
        forced = os.environ.get("BASSLIFT_SAVE_DIR")  # testy: bez okna dialogowego
        if forced:
            return Path(forced) / filename
        import webview
        chosen = self._window.create_file_dialog(webview.FileDialog.SAVE, save_filename=filename)
        if isinstance(chosen, (list, tuple)):
            chosen = chosen[0] if chosen else None
        return Path(chosen) if chosen else None

    def save_file(self, filename: str, data_b64: str) -> bool:
        target = self._target(filename)
        if not target:
            return False
        target.write_bytes(base64.b64decode(data_b64))
        log.info("Zapisano %s", target)
        return True

    def save_download(self, filename: str, path: str) -> bool:
        """Plik z lokalnego serwera (np. /bass/<id>) — bez przesyłania go przez JS."""
        if not path.startswith("/"):
            return False
        target = self._target(filename)
        if not target:
            return False
        with urllib.request.urlopen(self._base_url + path) as r, open(target, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        log.info("Zapisano %s", target)
        return True

    def copy_text(self, text: str) -> bool:
        from AppKit import NSPasteboard, NSPasteboardTypeString
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        return bool(pb.setString_forType_(text, NSPasteboardTypeString))


def main(on_ready: Optional[Callable] = None):
    """`on_ready(window)` — wołane w osobnym wątku po załadowaniu strony.
    BASSLIFT_SMOKETEST=<plik.json> uruchamia test dymny (basslift/smoketest.py)."""
    _configure()
    if os.environ.get("BASSLIFT_SMOKETEST") and on_ready is None:
        import tempfile
        from basslift import smoketest
        save_dir = Path(tempfile.mkdtemp(prefix="basslift_save_"))
        os.environ["BASSLIFT_SAVE_DIR"] = str(save_dir)
        out = Path(os.environ["BASSLIFT_SMOKETEST"])
        on_ready = lambda window: smoketest.run(window, out, save_dir)  # noqa: E731
    import uvicorn
    import webview

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import server as basslift_server

    port = _pick_port()
    base_url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(basslift_server.app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="uvicorn").start()

    bridge = Bridge(base_url)
    window = webview.create_window("BassLift", html=SPLASH, js_api=bridge, width=1100, height=920,
                                   min_size=(720, 600), background_color="#111520")
    bridge._window = window

    def load_when_ready():
        deadline = time.monotonic() + 60
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            window.load_html(SPLASH.replace("Uruchamianie…", "Nie udało się uruchomić serwera — "
                                            f"szczegóły w {LOG_DIR / 'basslift.log'}"))
            return
        window.load_url(base_url + "/?app=1")
        if on_ready:
            threading.Thread(target=on_ready, args=(window,), daemon=True).start()

    log.info("BassLift (okno) na %s", base_url)
    # private_mode=False: dane strony w ~/Library/WebKit/<bundle id> — ustawienia przetrwają restart
    webview.start(load_when_ready, private_mode=False)

    # Okno zamknięte: wyłącz serwer; wątki torcha/numby nie mogą trzymać procesu przy życiu
    server.should_exit = True
    log.info("Okno zamknięte — koniec")
    logging.shutdown()
    os._exit(0)
