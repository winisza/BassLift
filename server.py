"""
BassLift Backend — server.py  v3.0
Uruchomienie: uvicorn server:app --reload --port 8000

Transkrypcja: librosa.pyin (probabilistic YIN) — zero dodatkowych zależności
"""

import io, tempfile, base64, subprocess, logging, uuid, time, shutil
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("basslift")

try:
    from midiutil import MIDIFile
    HAS_MIDI = True
except ImportError:
    HAS_MIDI = False

app = FastAPI(title="BassLift", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

VERSION = "3.2.0"

BASS_FREQ_MIN = 30.0
BASS_FREQ_MAX = 262.0   # do C4 — łapie grę wysoko na gryfie
BASS_MIDI_MIN = 28       # E1
BASS_MIDI_MAX = 60       # C4 (17. próg struny G)

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


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
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


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


# ────────────────────────────────────────────────
# Separacja wokal/instrumental — bez transkrypcji
# ────────────────────────────────────────────────
@app.post("/separate")
async def separate(
    file: UploadFile = File(...),
    demucs_model: str = Form("htdemucs"),
    stem: str = Form("vocals"),  # vocals | bass | drums | other
):
    allowed = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Nieobsługiwany format: {suffix}")
    if stem not in {"vocals", "bass", "drums", "other"}:
        raise HTTPException(400, f"Nieobsługiwany stem: {stem}")

    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)
        src = tmpdir / ("input" + suffix)
        src.write_bytes(await file.read())

        log.info("Separacja Demucs (%s) — stem=%s", demucs_model, stem)
        out_dir = tmpdir / "demucs_out"
        cmd = [
            "python", "-m", "demucs",
            "--two-stems", stem,
            "-n", demucs_model,
            "-o", str(out_dir),
            str(src),
        ]
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise HTTPException(500, "Demucs zakończył się błędem.")

        target = next(iter(out_dir.rglob(f"{stem}.wav")), None)
        accomp = next(iter(out_dir.rglob(f"no_{stem}.wav")), None)
        if not target or not accomp:
            raise HTTPException(500, "Demucs nie wygenerował plików.")

        _cleanup_cache()
        base_id = uuid.uuid4().hex[:12]

        # Etykiety dla nazw plików
        accomp_label = {
            "vocals": "instrumental",
            "bass":   "no_bass",
            "drums":  "no_drums",
            "other":  "no_other",
        }[stem]

        target_id = f"{base_id}_{stem}"
        accomp_id = f"{base_id}_{accomp_label}"
        shutil.copy2(target, BASS_CACHE_DIR / f"{target_id}.wav")
        shutil.copy2(accomp, BASS_CACHE_DIR / f"{accomp_id}.wav")

        log.info("Gotowe! IDs: %s, %s", target_id, accomp_id)
        return JSONResponse({
            "target_id":      target_id,
            "target_label":   stem,
            "accomp_id":      accomp_id,
            "accomp_label":   accomp_label,
        })


# ────────────────────────────────────────────────
@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    demucs_model: str = Form("htdemucs"),
    note_threshold: int = Form(40),
    tuning: str = Form("E,A,D,G"),
    export_midi: str = Form("no"),
    export_bass: str = Form("no"),
    quantize: str = Form("16"),
):
    allowed = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Nieobslugiwany format: {suffix}")

    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)
        src = tmpdir / ("input" + suffix)
        src.write_bytes(await file.read())

        log.info("Krok 1/3 — Demucs separacja basu (%s)", demucs_model)
        bass_path = run_demucs(src, tmpdir, demucs_model)

        log.info("Krok 2/3 — Transkrypcja (librosa pyin)")
        notes, bpm, duration = transcribe_bass(bass_path, note_threshold, quantize)
        log.info("  Wykryto %d nut, BPM=%.1f, dlugosc=%.1fs", len(notes), bpm, duration)

        log.info("Krok 3/3 — Generowanie tabulatury")
        tuning_list = [s.strip().upper() for s in tuning.split(",")]
        tab_str = generate_tab(notes, bpm, tuning_list, quantize)

        midi_b64 = None
        if export_midi == "yes" and HAS_MIDI:
            midi_b64 = notes_to_midi_b64(notes, bpm)

        bass_download_id = None
        if export_bass == "yes":
            _cleanup_cache()
            bass_id = uuid.uuid4().hex[:12]
            cached = BASS_CACHE_DIR / f"{bass_id}.wav"
            shutil.copy2(bass_path, cached)
            bass_download_id = bass_id

        log.info("Gotowe!")
        return JSONResponse({
            "tab": tab_str,
            "bpm": round(bpm, 1),
            "note_count": len(notes),
            "duration": round(duration, 1),
            "tuning": ",".join(tuning_list),
            "midi_b64": midi_b64,
            "bass_download_id": bass_download_id,
        })


# ────────────────────────────────────────────────
# KROK 1 — Demucs
# ────────────────────────────────────────────────
def run_demucs(src: Path, tmpdir: Path, model: str) -> Path:
    out_dir = tmpdir / "demucs_out"
    cmd = [
        "python", "-m", "demucs",
        "--two-stems", "bass",
        "-n", model,
        "-o", str(out_dir),
        str(src),
    ]
    log.info("  Uruchamiam: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise HTTPException(500, "Demucs zakonczyl sie bledem. Sprawdz konsole.")

    candidates = list(out_dir.rglob("bass.wav"))
    if not candidates:
        raise HTTPException(500, "Demucs nie wygenenowal bass.wav")
    log.info("  Bass wyizolowany: %s", candidates[0])
    return candidates[0]


# ────────────────────────────────────────────────
# KROK 2 — Transkrypcja: librosa pyin
# ────────────────────────────────────────────────
def transcribe_bass(bass_path: Path, threshold_velocity: int, quantize_mode: str = "16") -> tuple:
    import librosa

    # Wczytaj audio — 22050 Hz wystarczy dla basu
    y, sr = librosa.load(str(bass_path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # ── BPM ──
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    if bpm < 40 or bpm > 300:
        bpm = 120.0
    log.info("  BPM=%.1f", bpm)

    # ── pyin: probabilistyczny YIN, świetny dla monofonicznych instrumentów ──
    # frame_length dobrane pod bas (duże okno = lepsza rozdzielczość dla niskich F0)
    hop = 512
    frame_length = 4096  # ~185ms przy 22050 Hz — wystarczy dla E1 (41 Hz)

    log.info("  Uruchamiam pyin...")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=BASS_FREQ_MIN,
        fmax=BASS_FREQ_MAX,
        sr=sr,
        hop_length=hop,
        frame_length=frame_length,
    )

    # Czasy ramek
    times = librosa.times_like(f0, sr=sr, hop_length=hop)

    # ── Filtruj: tylko voiced ramki w zakresie basu ──
    # Próg pewności detekcji pitcha:
    # threshold_velocity 0→127 mapuje się na conf_threshold 0.10→0.60
    # Niższy próg = więcej nut (w tym cichych/niepewnych), wyższy = tylko pewne nuty
    conf_threshold = 0.10 + (threshold_velocity / 127.0) * 0.50
    mask = (
        voiced_flag &
        (voiced_probs >= conf_threshold) &
        np.isfinite(f0) &
        (f0 >= BASS_FREQ_MIN) &
        (f0 <= BASS_FREQ_MAX)
    )

    times_f = times[mask]
    f0_f    = f0[mask]
    probs_f = voiced_probs[mask]

    if len(times_f) == 0:
        log.warning("  Brak wykrytych nut — sprobuj obnizyc prog")
        return [], bpm, duration

    # F0 → MIDI pitch
    midi_pitches = freq_to_midi(f0_f)

    # Segmentuj w nuty
    notes = segment_notes(times_f, midi_pitches, probs_f, hop_sec=hop/sr)

    # Kwantyzuj do siatki metrycznej
    notes = quantize_to_grid(notes, bpm, quantize_mode)

    # Filtruj zbyt krótkie / duplikaty
    notes = filter_notes(notes)

    return notes, bpm, duration


def freq_to_midi(freqs: np.ndarray) -> np.ndarray:
    return np.round(12.0 * np.log2(np.maximum(freqs, 1e-6) / 440.0) + 69).astype(int)


def segment_notes(times, pitches, probs, hop_sec=0.023) -> List[Dict]:
    """Łącz kolejne ramki tego samego pitcha w nuty."""
    if len(times) == 0:
        return []

    notes = []
    seg_start = times[0]
    seg_pitch = pitches[0]
    seg_probs = [probs[0]]
    max_gap   = hop_sec * 4   # max przerwa w ramach tej samej nuty (~92ms)

    for i in range(1, len(times)):
        gap        = times[i] - times[i-1]
        same_pitch = (pitches[i] == seg_pitch)

        if same_pitch and gap <= max_gap:
            seg_probs.append(probs[i])
        else:
            _flush_note(notes, seg_start, times[i-1], seg_pitch, seg_probs)
            seg_start = times[i]
            seg_pitch = pitches[i]
            seg_probs = [probs[i]]

    _flush_note(notes, seg_start, times[-1], seg_pitch, seg_probs)
    return notes


def _flush_note(notes, start, end, pitch, probs, min_dur=0.04):
    dur = end - start
    if dur < min_dur:
        return
    if not (BASS_MIDI_MIN <= pitch <= BASS_MIDI_MAX):
        return
    vel = int(min(127, float(np.mean(probs)) * 127 * 1.3))
    notes.append({
        "start":    round(float(start), 4),
        "end":      round(float(end) + 0.02, 4),
        "pitch":    int(pitch),
        "velocity": max(30, vel),
    })


def quantize_to_grid(notes: List[Dict], bpm: float, mode: str = "16") -> List[Dict]:
    if not notes:
        return notes
    beat_dur = 60.0 / bpm
    # Siatka kwantyzacji wg trybu
    if mode == "8":
        grid = beat_dur / 2.0        # ósemki
    elif mode == "8t":
        grid = beat_dur / 3.0        # triole ósemkowe
    elif mode == "16t":
        grid = beat_dur / 6.0        # triole szesnastkowe
    else:  # "16" — domyślnie
        grid = beat_dur / 4.0        # szesnastki
    out = []
    for n in notes:
        qs = round(n["start"] / grid) * grid
        qe = round(n["end"]   / grid) * grid
        if qe <= qs:
            qe = qs + grid
        out.append({**n, "start": round(qs, 4), "end": round(qe, 4)})
    return out


def filter_notes(notes: List[Dict], min_dur=0.04) -> List[Dict]:
    filtered = [n for n in notes if (n["end"] - n["start"]) >= min_dur]
    seen: Dict[float, Dict] = {}
    for n in filtered:
        key = round(n["start"], 2)
        if key not in seen or n["velocity"] > seen[key]["velocity"]:
            seen[key] = n
    return sorted(seen.values(), key=lambda x: x["start"])


# ────────────────────────────────────────────────
# KROK 3 — Tabulatura
# ────────────────────────────────────────────────
def tuning_to_midi(tuning_list: List[str]) -> List[int]:
    base_octaves = [1, 1, 2, 2]
    result = []
    for i, name in enumerate(tuning_list[:4]):
        n = name.strip().upper()
        try:
            idx = NOTE_NAMES.index(n)
        except ValueError:
            idx = 4  # fallback E
        result.append(12 * (base_octaves[i] + 1) + idx)
    return result


def pitch_to_tab(pitch: int, open_pitches: List[int], prev_string: Optional[int] = None) -> Optional[tuple]:
    candidates = []
    for si, open_p in enumerate(open_pitches):
        fret = pitch - open_p
        if 0 <= fret <= 17:
            score = fret * 1.5
            if prev_string is not None and si != prev_string:
                score += 3
            if fret > 9:
                score += 5
            candidates.append((score, si, fret))
    if not candidates:
        return None
    return (sorted(candidates)[0][1], sorted(candidates)[0][2])


def generate_tab(notes: List[Dict], bpm: float, tuning_list: List[str], quantize_mode: str = "16") -> str:
    if not notes:
        return "(brak wykrytych nut — spróbuj obniżyć próg pewności)"

    open_pitches    = tuning_to_midi(tuning_list)
    strings_display = list(reversed(tuning_list[:4]))  # G D A E od gory
    open_midi_rev   = list(reversed(open_pitches))

    beat_dur  = 60.0 / bpm
    bar_dur   = beat_dur * 4

    # Ilość kolumn w takcie zależy od siatki kwantyzacji
    if quantize_mode == "8":
        grid = beat_dur / 2.0; COLS = 8
    elif quantize_mode == "8t":
        grid = beat_dur / 3.0; COLS = 12
    elif quantize_mode == "16t":
        grid = beat_dur / 6.0; COLS = 24
    else:
        grid = beat_dur / 4.0; COLS = 16

    CW = 3

    max_time = notes[-1]["end"]
    num_bars = max(1, int(np.ceil(max_time / bar_dur)))

    tab_grid: Dict[tuple, tuple] = {}
    prev_string = None

    for n in notes:
        bar = int(n["start"] / bar_dur)
        col = min(int((n["start"] % bar_dur) / grid), COLS - 1)
        pos = pitch_to_tab(n["pitch"], open_midi_rev, prev_string)
        if pos:
            key = (bar, col)
            if key not in tab_grid:
                tab_grid[key] = pos
                prev_string = pos[0]

    lines = ["" for _ in range(4)]
    for bar in range(num_bars):
        for si in range(4):
            seg = ""
            for col in range(COLS):
                pos = tab_grid.get((bar, col))
                if pos and pos[0] == si:
                    seg += str(pos[1]).ljust(CW, "-")
                else:
                    seg += "-" * CW
            lines[si] += seg + "|"

    # Numery taktow co 4
    bar_numbers = "    "
    for bar in range(num_bars):
        label = str(bar + 1) if bar % 4 == 0 else ""
        bar_numbers += label.ljust(COLS * CW + 1)

    result = [bar_numbers]
    for si, label in enumerate(strings_display):
        result.append(f"{label.ljust(2)}|{lines[si]}")
    quant_labels = {"8": "1/8", "8t": "1/8T", "16": "1/16", "16t": "1/16T"}
    quant_label = quant_labels.get(quantize_mode, "1/16")
    result += ["", f"Stroj: {'-'.join(strings_display[::-1])}   BPM: {bpm:.0f}   Nuty: {len(notes)}   Siatka: {quant_label}"]

    return "\n".join(result)


# ────────────────────────────────────────────────
# MIDI export
# ────────────────────────────────────────────────
def notes_to_midi_b64(notes: List[Dict], bpm: float) -> Optional[str]:
    if not HAS_MIDI or not notes:
        return None
    midi = MIDIFile(1)
    midi.addTempo(0, 0, bpm)
    midi.addProgramChange(0, 0, 0, 33)  # Electric Bass (finger)
    beat_dur = 60.0 / bpm
    for n in notes:
        start_beat = n["start"] / beat_dur
        dur_beat   = max((n["end"] - n["start"]) / beat_dur, 0.1)
        midi.addNote(0, 0, n["pitch"], start_beat, dur_beat, max(1, min(127, n["velocity"])))
    buf = io.BytesIO()
    midi.writeFile(buf)
    return base64.b64encode(buf.getvalue()).decode()


# ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
