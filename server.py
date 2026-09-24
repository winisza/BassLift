"""
BassLift Backend — server.py  v3.0
Uruchomienie: dwuklik BassLift.command / BassLift.bat (albo python run.py).
Tryb deweloperski: uvicorn server:app --reload --port 8000

Warstwa HTTP. Separacja, transkrypcja i zapis nut są w pakiecie basslift/.
"""

import base64, logging, uuid, time, tempfile, threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from basslift import notation, separation, transcribe as tr

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("basslift")

app = FastAPI(title="BassLift", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ROOT_DIR = Path(__file__).parent


@app.get("/")
def index():
    return FileResponse(ROOT_DIR / "web_gui.html")


# Serwuj katalog logo/ jeśli istnieje
if (ROOT_DIR / "logo").is_dir():
    app.mount("/logo", StaticFiles(directory=ROOT_DIR / "logo"), name="logo")

VERSION = "0.3.0"

ALLOWED_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

BASS_CACHE_DIR = Path(tempfile.gettempdir()) / "basslift_cache"
BASS_CACHE_DIR.mkdir(exist_ok=True)
BASS_CACHE_MAX_AGE = 600  # 10 min


def _cleanup_cache():
    """Usuń pliki starsze niż BASS_CACHE_MAX_AGE."""
    now = time.time()
    for f in BASS_CACHE_DIR.glob("*.wav"):
        if now - f.stat().st_mtime > BASS_CACHE_MAX_AGE:
            f.unlink(missing_ok=True)


# ────────────────────────────────────────────────
# Obecność kart przeglądarki — run.py zamyka serwer, gdy nikt go nie używa
# ────────────────────────────────────────────────
class Presence:
    HEARTBEAT_TTL = 600  # s — karty w tle przeglądarka dławi (heartbeat nawet raz na minutę)
    BYE_TTL = 10         # s — po zamknięciu karty; odświeżenie zdąży się ponownie zgłosić

    def __init__(self):
        self._lock = threading.Lock()
        self._deadlines: Dict[str, float] = {}
        self._busy = 0
        self._seen_any = False

    def heartbeat(self, client: str):
        with self._lock:
            self._deadlines[client] = time.monotonic() + self.HEARTBEAT_TTL
            self._seen_any = True

    def bye(self, client: str):
        with self._lock:
            if client in self._deadlines:
                self._deadlines[client] = time.monotonic() + self.BYE_TTL

    @contextmanager
    def job(self):
        with self._lock:
            self._busy += 1
        try:
            yield
        finally:
            with self._lock:
                self._busy -= 1

    def seen_any(self) -> bool:
        return self._seen_any

    def idle(self) -> bool:
        now = time.monotonic()
        with self._lock:
            self._deadlines = {c: d for c, d in self._deadlines.items() if d > now}
            return not self._deadlines and self._busy == 0


presence = Presence()


# ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "app": "basslift"}


@app.post("/api/heartbeat")
def heartbeat(client: str):
    presence.heartbeat(client)
    return {"ok": True}


@app.post("/api/bye")
def bye(client: str):
    presence.bye(client)
    return {"ok": True}


@app.get("/download/{file_id}")
def download_file(file_id: str):
    """Pobierz wyizolowany stem WAV po ID."""
    safe_id = "".join(c for c in file_id if c.isalnum() or c in "-_")
    path = BASS_CACHE_DIR / f"{safe_id}.wav"
    if not path.exists():
        raise HTTPException(404, "Plik wygasł lub nie istnieje")
    # Wyłuskaj typ z nazwy pliku jeśli zakodowany
    fname = "stem.wav"
    if "_" in safe_id:
        suffix = safe_id.rsplit("_", 1)[1]
        fname = f"{suffix}.wav"
    return FileResponse(path, media_type="audio/wav", filename=fname)


# Alias dla kompatybilności wstecznej
@app.get("/bass/{file_id}")
def download_bass(file_id: str):
    return download_file(file_id)


def _save_upload(file: UploadFile, tmpdir: Path) -> Path:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Nieobsługiwany format: {suffix}")
    src = tmpdir / ("input" + suffix)
    src.write_bytes(file.file.read())
    return src


def _separate(src: Path, model: str):
    try:
        return separation.separate(src, model)
    except Exception as e:  # demucs.api.LoadAudioError, brak modelu itp.
        log.exception("Separacja nie powiodła się")
        hint = " (m4a/aac wymaga zainstalowanego ffmpeg)" if src.suffix in {".m4a", ".aac"} else ""
        raise HTTPException(500, f"Demucs nie przetworzył pliku{hint}: {e}")


# ────────────────────────────────────────────────
# Separacja wokal/instrumental — bez transkrypcji
# ────────────────────────────────────────────────
@app.post("/separate")
def separate(
    file: UploadFile = File(...),
    demucs_model: str = Form("htdemucs"),
    stem: str = Form("vocals"),  # vocals | bass | drums | other
):
    # Zwykłe `def` — FastAPI puszcza to w wątku, więc /health i heartbeat
    # odpowiadają także w trakcie kilkuminutowej separacji
    if stem not in {"vocals", "bass", "drums", "other"}:
        raise HTTPException(400, f"Nieobsługiwany stem: {stem}")

    with presence.job(), tempfile.TemporaryDirectory() as _tmp:
        src = _save_upload(file, Path(_tmp))
        log.info("Separacja Demucs (%s) — stem=%s", demucs_model, stem)
        sr, stems = _separate(src, demucs_model)
        target, accomp = separation.two_stems(stems, stem)

        _cleanup_cache()
        base_id = uuid.uuid4().hex[:12]
        accomp_label = {
            "vocals": "instrumental",
            "bass":   "no_bass",
            "drums":  "no_drums",
            "other":  "no_other",
        }[stem]
        target_id = f"{base_id}_{stem}"
        accomp_id = f"{base_id}_{accomp_label}"
        separation.save_wav(BASS_CACHE_DIR / f"{target_id}.wav", target, sr)
        separation.save_wav(BASS_CACHE_DIR / f"{accomp_id}.wav", accomp, sr)

        log.info("Gotowe! IDs: %s, %s", target_id, accomp_id)
        return JSONResponse({
            "target_id":      target_id,
            "target_label":   stem,
            "accomp_id":      accomp_id,
            "accomp_label":   accomp_label,
        })


# ────────────────────────────────────────────────
@app.post("/extract")
def extract(
    file: UploadFile = File(...),
    demucs_model: str = Form("htdemucs"),
    note_threshold: int = Form(40),
    tuning: str = Form("E,A,D,G"),
    export_midi: str = Form("no"),
    export_bass: str = Form("no"),
    quantize: str = Form("auto"),  # auto | 8 | 8t | 16 | 16t
    transcription_engine: str = Form("crepe"),  # "crepe" | "pyin"
):
    with presence.job(), tempfile.TemporaryDirectory() as _tmp:
        src = _save_upload(file, Path(_tmp))

        log.info("Krok 1/3 — Demucs separacja basu (%s)", demucs_model)
        sr, stems = _separate(src, demucs_model)
        bass, rest = separation.two_stems(stems, "bass")
        source, _ = separation.bass_for_transcription(stems, sr)

        engine = "pyin" if transcription_engine == "pyin" else "crepe"
        log.info("Krok 2/3 — Beaty i transkrypcja (%s)", engine)
        try:
            result = tr.transcribe(source, sr, mix=bass + rest, engine=engine,
                                   slider=note_threshold, quantize_mode=quantize)
        except RuntimeError as e:
            raise HTTPException(500, str(e))
        grid = result.grid
        log.info("  Wykryto %d nut, BPM=%.1f, metrum=%d/4, strój %+.0f centów",
                 len(result.notes), grid.bpm, grid.beats_per_bar, result.tuning_cents)

        log.info("Krok 3/3 — Tabulatura, MIDI, MusicXML")
        tuning_list = [s.strip().upper() for s in tuning.split(",")]
        tab_str = notation.generate_tab(result.notes, grid, tuning_list, result.quantize_mode,
                                        result.tuning_cents)
        midi_b64 = notation.notes_to_midi_b64(result.notes, grid) if export_midi == "yes" else None
        xml_str = notation.notes_to_musicxml(result.notes, grid)
        musicxml_b64 = base64.b64encode(xml_str.encode("utf-8")).decode() if xml_str else None

        bass_download_id = None
        if export_bass == "yes":
            _cleanup_cache()
            bass_download_id = uuid.uuid4().hex[:12]
            separation.save_wav(BASS_CACHE_DIR / f"{bass_download_id}.wav", bass, sr)

        log.info("Gotowe!")
        return JSONResponse({
            "tab": tab_str,
            "bpm": grid.bpm,
            "note_count": len(result.notes),
            "duration": round(result.duration, 1),
            "tuning": ",".join(tuning_list),
            "time_sig": grid.beats_per_bar,
            "tuning_cents": result.tuning_cents,
            "grid": notation.QUANT_LABELS.get(result.quantize_mode, "1/16"),
            "midi_b64": midi_b64,
            "musicxml_b64": musicxml_b64,
            "bass_download_id": bass_download_id,
        })


# ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
