"""
BassLift Backend — server.py  v3.0
Uruchomienie: uvicorn server:app --reload --port 8000

Transkrypcja: librosa.pyin (probabilistic YIN) — zero dodatkowych zależności
"""

import io, tempfile, base64, subprocess, logging, uuid, time, shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Dict
from collections import Counter

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("basslift")

try:
    from midiutil import MIDIFile
    HAS_MIDI = True
except ImportError:
    HAS_MIDI = False

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
    transcription_engine: str = Form("pyin"),  # "pyin" | "crepe"
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

        engine_label = "CREPE (torchcrepe)" if transcription_engine == "crepe" else "librosa pyin"
        log.info("Krok 2/3 — Transkrypcja (%s)", engine_label)
        if transcription_engine == "crepe":
            notes, bpm, duration, time_sig = transcribe_bass_crepe(bass_path, note_threshold, quantize)
        else:
            notes, bpm, duration, time_sig = transcribe_bass(bass_path, note_threshold, quantize)
        log.info("  Wykryto %d nut, BPM=%.1f, dlugosc=%.1fs, metrum=%d/4",
                 len(notes), bpm, duration, time_sig)

        log.info("Krok 3/3 — Generowanie tabulatury")
        tuning_list = [s.strip().upper() for s in tuning.split(",")]
        tab_str = generate_tab(notes, bpm, tuning_list, quantize, time_sig)

        midi_b64 = None
        if export_midi == "yes" and HAS_MIDI:
            midi_b64 = notes_to_midi_b64(notes, bpm)

        musicxml_b64 = None
        xml_str = notes_to_musicxml(notes, bpm, tuning_list, time_sig, quantize)
        if xml_str:
            musicxml_b64 = base64.b64encode(xml_str.encode('utf-8')).decode()

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
            "time_sig": time_sig,
            "midi_b64": midi_b64,
            "musicxml_b64": musicxml_b64,
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

    # ── BPM (robust multi-method) ──
    bpm = detect_bpm(y, sr)
    log.info("  BPM=%.1f", bpm)

    # ── Time signature detection ──
    time_sig = detect_time_signature(y, sr)
    log.info("  Time signature: %d/4", time_sig)

    # ── pyin: probabilistyczny YIN, świetny dla monofonicznych instrumentów ──
    hop = 512
    frame_length = 4096  # ~185ms przy 22050 Hz — wystarczy dla E1 (41 Hz)

    # ── Onset detection (for note boundary refinement) ──
    onset_times = detect_onsets(y, sr, hop)
    log.info("  Detected %d onsets", len(onset_times))

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

    # ── Adaptive confidence threshold ──
    base_threshold = 0.10 + (threshold_velocity / 127.0) * 0.50
    f0_std = rolling_std(f0, window=7)
    max_std = np.nanmax(f0_std) if np.any(f0_std > 0) else 1.0
    adaptive_threshold = base_threshold + 0.15 * (f0_std / max(max_std, 1e-6))

    mask = (
        voiced_flag &
        (voiced_probs >= adaptive_threshold) &
        np.isfinite(f0) &
        (f0 >= BASS_FREQ_MIN) &
        (f0 <= BASS_FREQ_MAX)
    )

    times_f = times[mask]
    f0_f    = f0[mask]
    probs_f = voiced_probs[mask]

    if len(times_f) == 0:
        log.warning("  Brak wykrytych nut — sprobuj obnizyc prog")
        return [], bpm, duration, time_sig

    # F0 → MIDI pitch
    midi_pitches = freq_to_midi(f0_f)

    # Pitch smoothing: median filter + octave correction
    midi_pitches = median_filter_pitches(midi_pitches, kernel_size=5)
    midi_pitches = correct_octave_errors(midi_pitches, max_short_frames=2)

    # Segmentuj w nuty (onset-assisted boundaries)
    notes = segment_notes(times_f, midi_pitches, probs_f,
                          hop_sec=hop/sr, onset_times=onset_times)

    # Kwantyzuj do siatki metrycznej (onset-aware)
    notes = quantize_to_grid(notes, bpm, quantize_mode, onset_times=onset_times)

    # Merge close same-pitch notes split by quantization
    grid_dur = _grid_duration(bpm, quantize_mode)
    notes = merge_close_notes(notes, min_gap=grid_dur)

    # Filtruj zbyt krótkie / duplikaty
    notes = filter_notes(notes)

    return notes, bpm, duration, time_sig


# ────────────────────────────────────────────────
# KROK 2 (alternatywa) — Transkrypcja: torchcrepe (neural)
# ────────────────────────────────────────────────
def transcribe_bass_crepe(bass_path: Path, threshold_velocity: int, quantize_mode: str = "16") -> tuple:
    """Transkrypcja basu przy pomocy CREPE (torchcrepe) — neuronowy pitch tracker.

    Znacznie celniejszy niż librosa.pyin (mniej skoków oktawowych w niskim rejestrze,
    lepsze granice nut), kosztem dłuższego czasu inferencji (~10-30s/min audio na CPU).
    Wykorzystuje GPU jeśli dostępne (CUDA).
    """
    import librosa
    try:
        import torch
        import torchcrepe
    except ImportError:
        raise HTTPException(
            500,
            "Silnik CREPE wymaga pakietów torch i torchcrepe. Zainstaluj: pip install torchcrepe"
        )

    # Wczytaj audio w natywnej sr, potem resampling do 16 kHz (CREPE wymaga 16 kHz)
    y, sr_native = librosa.load(str(bass_path), sr=None, mono=True)
    target_sr = 16000
    if sr_native != target_sr:
        y_resampled = librosa.resample(y, orig_sr=sr_native, target_sr=target_sr)
    else:
        y_resampled = y
    duration = librosa.get_duration(y=y, sr=sr_native)

    # BPM + metrum (na oryginalnej sr — lepsza detekcja onsetów rytmu)
    bpm = detect_bpm(y, sr_native)
    log.info("  BPM=%.1f", bpm)
    time_sig = detect_time_signature(y, sr_native)
    log.info("  Time signature: %d/4", time_sig)

    # CREPE: hop = 160 sampli przy 16 kHz = 10 ms
    hop = 160
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info("  Uruchamiam torchcrepe (model=full, device=%s)...", device)

    audio_tensor = torch.tensor(y_resampled, dtype=torch.float32).unsqueeze(0)
    f0_t, periodicity_t = torchcrepe.predict(
        audio_tensor,
        target_sr,
        hop_length=hop,
        fmin=BASS_FREQ_MIN,
        fmax=BASS_FREQ_MAX,
        model='full',
        return_periodicity=True,
        device=device,
        batch_size=2048,
        pad=True,
    )

    # Wygładź wyniki zgodnie z rekomendacjami torchcrepe
    periodicity_t = torchcrepe.filter.median(periodicity_t, 3)
    f0_t = torchcrepe.filter.mean(f0_t, 3)

    f0 = f0_t.squeeze(0).cpu().numpy()
    periodicity = periodicity_t.squeeze(0).cpu().numpy()

    times = np.arange(len(f0)) * (hop / target_sr)

    # Onset detection na audio przekazanym do CREPE (16 kHz)
    onset_times = detect_onsets(y_resampled, target_sr, hop)
    log.info("  Detected %d onsets", len(onset_times))

    # Adaptive threshold na periodicity (zakres ~0.2-0.95)
    base_threshold = 0.21 + (threshold_velocity / 127.0) * 0.30

    mask = (
        np.isfinite(f0) &
        (f0 >= BASS_FREQ_MIN) &
        (f0 <= BASS_FREQ_MAX) &
        (periodicity >= base_threshold)
    )

    times_f = times[mask]
    f0_f = f0[mask]
    probs_f = periodicity[mask]

    if len(times_f) == 0:
        log.warning("  Brak wykrytych nut — sprobuj obnizyc prog")
        return [], bpm, duration, time_sig

    # F0 → MIDI pitch
    midi_pitches = freq_to_midi(f0_f)

    # Pitch smoothing — CREPE rzadziej halffa, więc octave correction łagodniejsza
    midi_pitches = median_filter_pitches(midi_pitches, kernel_size=5)
    midi_pitches = correct_octave_errors(midi_pitches, max_short_frames=1)

    # Segmentacja w nuty (ten sam helper co dla pyin)
    notes = segment_notes(times_f, midi_pitches, probs_f,
                          hop_sec=hop/target_sr, onset_times=onset_times)

    # Kwantyzacja do siatki metrycznej (onset-aware)
    notes = quantize_to_grid(notes, bpm, quantize_mode, onset_times=onset_times)

    # Merge close same-pitch notes split by quantization
    grid_dur = _grid_duration(bpm, quantize_mode)
    notes = merge_close_notes(notes, min_gap=grid_dur)

    # Filtruj zbyt krótkie / duplikaty
    notes = filter_notes(notes)

    return notes, bpm, duration, time_sig


def freq_to_midi(freqs: np.ndarray) -> np.ndarray:
    return np.round(12.0 * np.log2(np.maximum(freqs, 1e-6) / 440.0) + 69).astype(int)


def median_filter_pitches(midi_pitches: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Smooth 1-2 frame pitch outliers using rolling median."""
    n = len(midi_pitches)
    if n < 3:
        return midi_pitches.copy()
    ks = min(kernel_size, n if n % 2 == 1 else n - 1)
    if ks < 3:
        return midi_pitches.copy()
    pad = ks // 2
    padded = np.pad(midi_pitches, pad, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, ks)
    return np.median(windows, axis=1).astype(int)


def correct_octave_errors(midi_pitches: np.ndarray, max_short_frames: int = 2) -> np.ndarray:
    """Fix pYIN octave jumps (±12 semitones lasting only 1-2 frames)."""
    result = midi_pitches.copy()
    n = len(result)
    if n < 3:
        return result
    i = 0
    while i < n:
        base = result[i]
        j = i + 1
        # Find run of frames that are exactly ±12 from base
        while j < n and abs(int(result[j]) - int(base)) == 12:
            j += 1
        run_len = j - i - 1
        if 0 < run_len <= max_short_frames:
            for k in range(i + 1, j):
                corrected = int(base)
                if BASS_MIDI_MIN <= corrected <= BASS_MIDI_MAX:
                    result[k] = corrected
        i = j if run_len > 0 else i + 1
    return result


def rolling_std(arr: np.ndarray, window: int = 7) -> np.ndarray:
    """Compute rolling standard deviation, handling NaN values."""
    n = len(arr)
    out = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = arr[lo:hi]
        finite = chunk[np.isfinite(chunk)]
        out[i] = np.std(finite) if len(finite) > 1 else 0.0
    return out


def detect_bpm(y, sr) -> float:
    """Robust BPM detection using multiple estimation methods."""
    import librosa
    candidates = []
    # Method 1: librosa.feature.tempo with multiple start hints
    for start_bpm in [80, 100, 120, 140, 160]:
        try:
            tempo_arr = librosa.feature.tempo(y=y, sr=sr, start_bpm=start_bpm)
            if len(tempo_arr) > 0:
                candidates.append(float(tempo_arr[0]))
        except Exception:
            pass
    # Method 2: beat_track
    try:
        tempo_bt, _ = librosa.beat.beat_track(y=y, sr=sr)
        bt_val = float(tempo_bt) if np.isscalar(tempo_bt) else float(tempo_bt[0])
        candidates.append(bt_val)
    except Exception:
        pass
    candidates = [c for c in candidates if 40 <= c <= 300]
    if not candidates:
        return 120.0
    rounded = [round(c) for c in candidates]
    bpm = Counter(rounded).most_common(1)[0][0]
    return float(bpm)


def detect_onsets(y, sr, hop_length: int) -> np.ndarray:
    """Detect note onset times in the bass audio."""
    import librosa
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, backtrack=True
    )
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)


def detect_time_signature(y, sr) -> int:
    """Detect whether the song is in 3/4 or 4/4 time."""
    import librosa
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if len(beat_frames) < 8:
            return 4
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        valid = beat_frames[beat_frames < len(onset_env)]
        if len(valid) < 8:
            return 4
        beat_strengths = onset_env[valid]

        def downbeat_ratio(group_size):
            if len(beat_strengths) < group_size * 2:
                return 0.0
            downbeats = beat_strengths[::group_size]
            total = beat_strengths[:len(downbeats) * group_size]
            avg = np.mean(total)
            return np.mean(downbeats) / avg if avg > 0 else 0.0

        r3, r4 = downbeat_ratio(3), downbeat_ratio(4)
        return 3 if r3 > r4 * 1.1 else 4
    except Exception:
        return 4


def merge_close_notes(notes: List[Dict], min_gap: float) -> List[Dict]:
    """Merge consecutive same-pitch notes separated by less than min_gap."""
    if len(notes) < 2:
        return notes
    merged = [notes[0].copy()]
    for n in notes[1:]:
        prev = merged[-1]
        if n["pitch"] == prev["pitch"] and (n["start"] - prev["end"]) < min_gap:
            prev["end"] = n["end"]
            prev["velocity"] = max(prev["velocity"], n["velocity"])
        else:
            merged.append(n.copy())
    return merged


def _grid_duration(bpm: float, mode: str) -> float:
    """Compute quantization grid duration in seconds."""
    beat_dur = 60.0 / bpm
    if mode == "8":
        return beat_dur / 2.0
    elif mode == "8t":
        return beat_dur / 3.0
    elif mode == "16t":
        return beat_dur / 6.0
    return beat_dur / 4.0


def segment_notes(times, pitches, probs, hop_sec=0.023, onset_times=None) -> List[Dict]:
    """Łącz kolejne ramki tego samego pitcha w nuty. Split at onsets."""
    if len(times) == 0:
        return []

    notes = []
    seg_start = times[0]
    seg_pitch = pitches[0]
    seg_probs = [probs[0]]
    max_gap   = hop_sec * 4   # max przerwa w ramach tej samej nuty (~92ms)
    onset_thr = hop_sec * 1.5

    for i in range(1, len(times)):
        gap        = times[i] - times[i-1]
        same_pitch = (pitches[i] == seg_pitch)

        # Check if current frame coincides with an onset
        near_onset = False
        if onset_times is not None and len(onset_times) > 0:
            near_onset = bool(np.any(np.abs(onset_times - times[i]) < onset_thr))

        if same_pitch and gap <= max_gap and not near_onset:
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


def quantize_to_grid(notes: List[Dict], bpm: float, mode: str = "16",
                     onset_times: Optional[np.ndarray] = None) -> List[Dict]:
    if not notes:
        return notes
    grid = _grid_duration(bpm, mode)
    half_grid = grid / 2.0
    out = []
    for n in notes:
        start = n["start"]
        # Prefer onset time if one is close
        if onset_times is not None and len(onset_times) > 0:
            diffs = np.abs(onset_times - start)
            nearest_idx = int(np.argmin(diffs))
            if diffs[nearest_idx] < half_grid:
                start = float(onset_times[nearest_idx])
        qs = round(start / grid) * grid
        qe = round(n["end"] / grid) * grid
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


def generate_tab(notes: List[Dict], bpm: float, tuning_list: List[str],
                 quantize_mode: str = "16", time_sig: int = 4) -> str:
    if not notes:
        return "(brak wykrytych nut — spróbuj obniżyć próg pewności)"

    open_pitches    = tuning_to_midi(tuning_list)
    strings_display = list(reversed(tuning_list[:4]))  # G D A E od gory
    open_midi_rev   = list(reversed(open_pitches))

    beat_dur  = 60.0 / bpm
    bar_dur   = beat_dur * time_sig

    # Ilość kolumn w takcie zależy od siatki kwantyzacji × time_sig
    if quantize_mode == "8":
        grid = beat_dur / 2.0; COLS = 2 * time_sig
    elif quantize_mode == "8t":
        grid = beat_dur / 3.0; COLS = 3 * time_sig
    elif quantize_mode == "16t":
        grid = beat_dur / 6.0; COLS = 6 * time_sig
    else:
        grid = beat_dur / 4.0; COLS = 4 * time_sig

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
    result += ["", f"Stroj: {'-'.join(strings_display[::-1])}   BPM: {bpm:.0f}   Nuty: {len(notes)}   Siatka: {quant_label}   Metrum: {time_sig}/4"]

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
# MusicXML export
# ────────────────────────────────────────────────
MUSICXML_STEPS = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B']
MUSICXML_ALTER = [0,   1,   0,   1,   0,   0,   1,   0,   1,   0,   1,   0]


def midi_to_musicxml_pitch(midi_pitch: int) -> tuple:
    """Convert MIDI pitch to (step, alter, octave)."""
    octave = (midi_pitch // 12) - 1
    step = MUSICXML_STEPS[midi_pitch % 12]
    alter = MUSICXML_ALTER[midi_pitch % 12]
    return step, alter, octave


def duration_to_type(dur_divs: int, divisions: int = 12) -> tuple:
    """Map duration in divisions to MusicXML type string and canonical duration."""
    type_map = {
        divisions * 4: 'whole',
        divisions * 2: 'half',
        divisions:     'quarter',
        divisions // 2: 'eighth',
        divisions // 4: '16th',
    }
    # Find closest valid duration
    valid_durs = sorted(type_map.keys())
    best = min(valid_durs, key=lambda k: abs(k - dur_divs))
    return type_map[best], best


def _add_note_el(parent, midi_pitch: int, dur_divs: int, divisions: int = 12):
    """Add a <note> element to a MusicXML measure."""
    step, alter, octave = midi_to_musicxml_pitch(midi_pitch)
    note_type, canonical = duration_to_type(dur_divs, divisions)

    note_el = ET.SubElement(parent, 'note')
    pitch_el = ET.SubElement(note_el, 'pitch')
    ET.SubElement(pitch_el, 'step').text = step
    if alter != 0:
        ET.SubElement(pitch_el, 'alter').text = str(alter)
    ET.SubElement(pitch_el, 'octave').text = str(octave)
    ET.SubElement(note_el, 'duration').text = str(max(1, dur_divs))
    ET.SubElement(note_el, 'type').text = note_type


def _add_rest_els(parent, total_dur: int, divisions: int = 12):
    """Add rest elements for total_dur divisions, decomposed into valid rest sizes."""
    # Greedy decomposition: largest valid rest first
    valid_durs = sorted([divisions * 4, divisions * 2, divisions,
                         divisions // 2, max(1, divisions // 4)], reverse=True)
    remaining = total_dur
    while remaining > 0:
        chosen = 1
        for vd in valid_durs:
            if vd <= remaining and vd > 0:
                chosen = vd
                break
        note_type, _ = duration_to_type(chosen, divisions)
        note_el = ET.SubElement(parent, 'note')
        ET.SubElement(note_el, 'rest')
        ET.SubElement(note_el, 'duration').text = str(chosen)
        ET.SubElement(note_el, 'type').text = note_type
        remaining -= chosen


def notes_to_musicxml(notes: List[Dict], bpm: float, tuning_list: List[str],
                      time_sig: int = 4, quantize_mode: str = "16") -> Optional[str]:
    """Generate MusicXML string from note data."""
    if not notes:
        return None

    divisions = 12  # per quarter note — supports 16ths (3) and triplets (4)
    beat_dur = 60.0 / bpm
    measure_capacity = time_sig * divisions

    # Convert notes to division-based timing
    events = []
    for n in notes:
        start_divs = round((n["start"] / beat_dur) * divisions)
        dur_divs = max(1, round(((n["end"] - n["start"]) / beat_dur) * divisions))
        events.append((start_divs, dur_divs, n["pitch"]))

    max_divs = max(s + d for s, d, _ in events) if events else measure_capacity
    num_measures = max(1, (max_divs + measure_capacity - 1) // measure_capacity)

    # Build XML
    root = ET.Element('score-partwise', version='4.0')

    part_list = ET.SubElement(root, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', id='P1')
    ET.SubElement(score_part, 'part-name').text = 'Bass'

    part = ET.SubElement(root, 'part', id='P1')

    for m in range(num_measures):
        measure_el = ET.SubElement(part, 'measure', number=str(m + 1))

        if m == 0:
            # Attributes: divisions, time signature, bass clef
            attrs = ET.SubElement(measure_el, 'attributes')
            ET.SubElement(attrs, 'divisions').text = str(divisions)
            time_el = ET.SubElement(attrs, 'time')
            ET.SubElement(time_el, 'beats').text = str(time_sig)
            ET.SubElement(time_el, 'beat-type').text = '4'
            clef = ET.SubElement(attrs, 'clef')
            ET.SubElement(clef, 'sign').text = 'F'
            ET.SubElement(clef, 'line').text = '4'

            # Tempo marking
            direction = ET.SubElement(measure_el, 'direction', placement='above')
            dt = ET.SubElement(direction, 'direction-type')
            metro = ET.SubElement(dt, 'metronome')
            ET.SubElement(metro, 'beat-unit').text = 'quarter'
            ET.SubElement(metro, 'per-minute').text = str(int(bpm))
            ET.SubElement(direction, 'sound', tempo=str(int(bpm)))

        measure_start = m * measure_capacity
        measure_end = measure_start + measure_capacity
        cursor = measure_start

        # Notes in this measure (sorted by start)
        measure_notes = sorted(
            [(s, d, p) for s, d, p in events if s < measure_end and s + d > measure_start],
            key=lambda x: x[0]
        )

        for s, d, p in measure_notes:
            note_start = max(s, measure_start)
            note_end = min(s + d, measure_end)
            note_dur = note_end - note_start
            if note_dur <= 0:
                continue

            # Insert rest for gap
            if note_start > cursor:
                _add_rest_els(measure_el, note_start - cursor, divisions)

            _add_note_el(measure_el, p, note_dur, divisions)
            cursor = note_start + note_dur

        # Fill remainder with rests
        if cursor < measure_end:
            _add_rest_els(measure_el, measure_end - cursor, divisions)

    # Serialize
    xml_decl = '<?xml version="1.0" encoding="UTF-8"?>\n'
    doctype = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
    body = ET.tostring(root, encoding='unicode')
    return xml_decl + doctype + body


# ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
