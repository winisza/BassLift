"""
BassLift Backend — server.py  v3.0
Uruchomienie: dwuklik BassLift.command / BassLift.bat (albo python run.py).
Tryb deweloperski: uvicorn server:app --reload --port 8000

Warstwa HTTP. Separacja, transkrypcja i zapis nut są w pakiecie basslift/.
"""

import logging, shutil, time, tempfile, threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from basslift import jobs

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
BASS_CACHE_MAX_AGE = jobs.JOB_TTL  # ścieżki muszą żyć tak długo jak zadania, które je podają


def _cleanup_cache():
    """Usuń pliki i katalogi zadań starsze niż BASS_CACHE_MAX_AGE (także po restarcie serwera)."""
    now = time.time()
    for f in BASS_CACHE_DIR.glob("*.wav"):
        if now - f.stat().st_mtime > BASS_CACHE_MAX_AGE:
            f.unlink(missing_ok=True)
    for d in (BASS_CACHE_DIR / "jobs").glob("*/"):
        if now - d.stat().st_mtime > BASS_CACHE_MAX_AGE and not jobs_store.in_use(d.name):
            shutil.rmtree(d, ignore_errors=True)


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
jobs_store = jobs.JobStore(BASS_CACHE_DIR, busy=presence.job)


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


def _submit(file: UploadFile, kind: str, params: Dict, wait: bool) -> jobs.Job:
    _cleanup_cache()
    with tempfile.TemporaryDirectory() as tmp:
        return jobs_store.submit(kind, _save_upload(file, Path(tmp)), params, wait=wait)


def _job_or_404(job_id: str) -> jobs.Job:
    job = jobs_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Zadanie wygasło — uruchom je ponownie")
    return job


def _sync_result(job: jobs.Job) -> JSONResponse:
    if job.state == "error":
        raise HTTPException(500, job.error)
    return JSONResponse(job.result)


# ────────────────────────────────────────────────
# Zadania w tle: postęp + ponowna transkrypcja bez separacji
# ────────────────────────────────────────────────
@app.post("/api/jobs")
def create_job(
    file: UploadFile = File(...),
    kind: str = Form("extract"),                # extract | separate
    demucs_model: str = Form(jobs.separation.DEFAULT_MODEL),
    stem: str = Form("vocals"),                 # separate: vocals | bass | drums | other
    note_threshold: int = Form(40),
    tuning: str = Form("E,A,D,G"),
    export_midi: str = Form("no"),
    export_bass: str = Form("no"),
    quantize: str = Form("auto"),               # auto | 8 | 8t | 16 | 16t
    transcription_engine: str = Form("crepe"),  # crepe | pyin
):
    if kind not in {"extract", "separate"}:
        raise HTTPException(400, f"Nieznany rodzaj zadania: {kind}")
    if kind == "separate" and stem not in {"vocals", "bass", "drums", "other"}:
        raise HTTPException(400, f"Nieobsługiwany stem: {stem}")
    params = jobs.extract_params(demucs_model, note_threshold, tuning, export_midi, export_bass,
                                 quantize, transcription_engine)
    params["stem"] = stem
    return {"job_id": _submit(file, kind, params, wait=False).id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job_or_404(job_id).public()


@app.post("/api/jobs/{job_id}/retranscribe")
def retranscribe(
    job_id: str,
    note_threshold: int = Form(40),
    tuning: str = Form("E,A,D,G"),
    export_midi: str = Form("no"),
    export_bass: str = Form("no"),
    quantize: str = Form("auto"),
    transcription_engine: str = Form("crepe"),
):
    parent = _job_or_404(job_id)
    if parent.kind != "extract" or not parent.session.analyses:
        raise HTTPException(409, "To zadanie nie ma gotowej analizy do ponownej transkrypcji")
    params = jobs.extract_params(parent.params["demucs_model"], note_threshold, tuning,
                                 export_midi, export_bass, quantize, transcription_engine)
    return {"job_id": jobs_store.retranscribe(parent, params).id}


# ────────────────────────────────────────────────
# Synchroniczne API (zgodność wstecz) — te same zadania, odpowiedź po zakończeniu
# ────────────────────────────────────────────────
@app.post("/separate")
def separate(
    file: UploadFile = File(...),
    demucs_model: str = Form(jobs.separation.DEFAULT_MODEL),
    stem: str = Form("vocals"),  # vocals | bass | drums | other
):
    # Zwykłe `def` — FastAPI puszcza to w wątku, więc /health i heartbeat
    # odpowiadają także w trakcie kilkuminutowej separacji
    if stem not in {"vocals", "bass", "drums", "other"}:
        raise HTTPException(400, f"Nieobsługiwany stem: {stem}")
    return _sync_result(_submit(file, "separate", {"demucs_model": demucs_model, "stem": stem}, wait=True))


@app.post("/extract")
def extract(
    file: UploadFile = File(...),
    demucs_model: str = Form(jobs.separation.DEFAULT_MODEL),
    note_threshold: int = Form(40),
    tuning: str = Form("E,A,D,G"),
    export_midi: str = Form("no"),
    export_bass: str = Form("no"),
    quantize: str = Form("auto"),  # auto | 8 | 8t | 16 | 16t
    transcription_engine: str = Form("crepe"),  # "crepe" | "pyin"
):
    params = jobs.extract_params(demucs_model, note_threshold, tuning, export_midi, export_bass,
                                 quantize, transcription_engine)
    return _sync_result(_submit(file, "extract", params, wait=True))


# ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
