"""Transkrypcja basu: tor wysokości (CREPE / pyin) -> nuty zakotwiczone w onsetach.

Wysokość dźwięku z trackera jest pewna w środku nuty, ale jego okno analizy spóźnia
początek (pyin ~90 ms) i nie widzi powtórzeń tego samego dźwięku. Dlatego:
  1. segmenty = ciągłe klatki z tą samą wysokością,
  2. segment dzielony na onsetach wewnątrz (powtórzenia),
  3. początek przyciągany do najbliższego onsetu przed nim (kompensacja opóźnienia),
  4. wysokość = mediana po ataku, z poprawką na odstrojenie całego nagrania.
"""
import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .rhythm import BeatGrid, choose_grid, quantize, track_beats

log = logging.getLogger("basslift")

PYIN_FREQ_MIN = 30.0
# Najniższy bin CREPE to ~31.7 Hz — przy fmin poniżej torchcrepe maskuje wszystkie
# biny (periodicity = NaN, zero nut). C1 z zapasem, E1 (41 Hz) nadal w zakresie.
CREPE_FREQ_MIN = 32.71
BASS_FREQ_MAX = 262.0     # C4 — 17. próg struny G
BASS_MIDI_MIN = 23        # B0 (bas 5-strunowy)
BASS_MIDI_MAX = 60        # C4

MIN_NOTE = 0.05           # s — krótsze segmenty to artefakty
MAX_GAP = 0.06            # s — dziura bez tonu, którą jeszcze sklejamy w jedną nutę
SETTLE = 0.02             # s — atak pomijany przy liczeniu wysokości
ONSET_HOP = 256           # przy 22050 Hz = 11.6 ms


@dataclass
class PitchTrack:
    times: np.ndarray        # środek klatki [s]
    midi: np.ndarray         # ułamkowe MIDI, NaN gdy brak tonu
    confidence: np.ndarray   # voiced prob (pyin) / periodicity (CREPE), 0..1
    threshold: float         # próg pewności wybrany suwakiem
    latency: float           # jak bardzo tracker spóźnia początek nuty [s]


@dataclass
class Transcription:
    notes: List[Dict]        # skwantyzowane (z polami beat/beats)
    raw_notes: List[Dict]    # przed kwantyzacją — do ewaluacji
    grid: BeatGrid
    duration: float
    tuning_cents: float
    quantize_mode: str       # faktycznie użyta siatka (po rozstrzygnięciu „auto”)


# ────────────────────────────────────────────────
# Tory wysokości
# ────────────────────────────────────────────────
def track_pitch_crepe(y: np.ndarray, sr: int, slider: int) -> PitchTrack:
    import librosa
    try:
        import torch
        import torchcrepe
    except ImportError as e:
        raise RuntimeError("Silnik CREPE wymaga pakietów torch i torchcrepe "
                           "(pip install torchcrepe)") from e
    from .separation import free_gpu_memory, torch_device

    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    hop = 160  # 10 ms
    # ~37 MB RAM na klatkę w batchu (im2col drugiej warstwy): 32 ≈ 1.6 GB, 2048 ≈ 76 GB
    try:
        f0, per = torchcrepe.predict(
            torch.tensor(y16, dtype=torch.float32).unsqueeze(0), 16000,
            hop_length=hop, fmin=CREPE_FREQ_MIN, fmax=BASS_FREQ_MAX, model="full",
            return_periodicity=True, device=torch_device(), batch_size=32, pad=True)
    finally:
        free_gpu_memory()
    per = torchcrepe.filter.median(per, 3)
    f0 = f0.squeeze(0).cpu().numpy()
    per = np.nan_to_num(per.squeeze(0).cpu().numpy())
    return PitchTrack(times=np.arange(len(f0)) * hop / 16000, midi=_hz_to_midi(f0),
                      confidence=per, threshold=0.21 + slider / 127 * 0.30, latency=0.05)


def track_pitch_pyin(y: np.ndarray, sr: int, slider: int) -> PitchTrack:
    import librosa
    y22 = librosa.resample(y, orig_sr=sr, target_sr=22050) if sr != 22050 else y
    hop = 512
    f0, voiced, prob = librosa.pyin(y22, fmin=PYIN_FREQ_MIN, fmax=BASS_FREQ_MAX, sr=22050,
                                    hop_length=hop, frame_length=4096)
    prob = np.where(voiced, np.nan_to_num(prob), 0.0)
    return PitchTrack(times=librosa.times_like(f0, sr=22050, hop_length=hop),
                      midi=_hz_to_midi(f0), confidence=prob,
                      threshold=0.10 + slider / 127 * 0.50, latency=0.14)


def _hz_to_midi(f0):
    f0 = np.asarray(f0, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 12.0 * np.log2(f0 / 440.0) + 69.0
    return np.where(np.isfinite(m) & (f0 > 0), m, np.nan)


def estimate_tuning(track: PitchTrack, voiced: np.ndarray) -> float:
    """Odchyłka stroju całego nagrania od A=440 w półtonach (-0.5..0.5), ważona pewnością."""
    if voiced.sum() < 20:
        return 0.0
    dev = track.midi[voiced] - np.round(track.midi[voiced])
    w = track.confidence[voiced]
    # średnia kołowa: odchyłki -0.49 i +0.49 półtonu leżą obok siebie
    z = np.sum(w * np.exp(2j * np.pi * dev)) / np.sum(w)
    return float(np.angle(z) / (2 * np.pi))


# ────────────────────────────────────────────────
# Onsety i energia
# ────────────────────────────────────────────────
def detect_onsets(y22: np.ndarray):
    """Onsety basu (22050 Hz) z przyrostu log-energii. Zwraca (czasy, siła 0..1).

    Spectral flux (domyślny w librosa) gubił ponowne uderzenia tej samej nuty i spóźniał
    się o ~23 ms; przyrost log-RMS łapie je z błędem ~5 ms (eval: F nut 0.82 -> 0.88
    syntetyka, 0.52 -> 0.56 Slakh po Demucs)."""
    import librosa
    rms = librosa.feature.rms(y=y22, frame_length=512, hop_length=ONSET_HOP)[0]
    log_rms = np.log(rms + 1e-4)
    env = np.maximum(0.0, np.diff(log_rms, prepend=log_rms[0]))
    env = env / (np.max(env) + 1e-9)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=22050, hop_length=ONSET_HOP, delta=0.07)
    return librosa.frames_to_time(frames, sr=22050, hop_length=ONSET_HOP), env[frames]


def rms_envelope(y22: np.ndarray):
    import librosa
    rms = librosa.feature.rms(y=y22, frame_length=1024, hop_length=ONSET_HOP)[0]
    return librosa.times_like(rms, sr=22050, hop_length=ONSET_HOP), rms


# ────────────────────────────────────────────────
# Segmentacja
# ────────────────────────────────────────────────
def segment_notes(track: PitchTrack, onsets: np.ndarray, onset_strength: np.ndarray,
                  rms_t: np.ndarray, rms: np.ndarray, tuning: float) -> List[Dict]:
    t = track.times
    hop = float(t[1] - t[0])
    midi = track.midi - tuning
    voiced = (track.confidence >= track.threshold) & np.isfinite(midi) \
        & (midi >= BASS_MIDI_MIN - 0.5) & (midi <= BASS_MIDI_MAX + 0.5)
    q = np.where(voiced, np.round(np.nan_to_num(midi)), -1).astype(int)
    q = _absorb_short_runs(q, min_run=max(2, int(round(0.03 / hop))))

    # 1) segmenty o stałej wysokości (krótkie dziury bez tonu sklejane)
    segs = []
    for i in np.flatnonzero(q >= 0):
        if segs and segs[-1]["pitch"] == q[i] and t[i] - t[segs[-1]["last"]] <= MAX_GAP + hop:
            segs[-1]["last"] = i
        else:
            segs.append({"first": i, "last": i, "pitch": int(q[i])})
    notes = [{"start": float(t[s["first"]]), "end": float(t[s["last"]] + hop), "pitch": s["pitch"]}
             for s in segs]

    # 2) powtórzenia: podział segmentu na onsetach wewnątrz niego (próg siły onsetu
    #    nie pomaga: Slakh +0.01, syntetyka -0.05 — realne linie basu to dużo powtórzeń)
    split = []
    for n in notes:
        cuts = [o for o in onsets if n["start"] + MIN_NOTE < o < n["end"] - MIN_NOTE]
        bounds = [n["start"], *cuts, n["end"]]
        split += [{**n, "start": a, "end": b} for a, b in zip(bounds, bounds[1:])]

    # 3) początek -> najmocniejszy onset w oknie opóźnienia trackera
    anchored = []
    for n in split:
        lo = n["start"] - track.latency
        if anchored:
            lo = max(lo, anchored[-1]["start"] + MIN_NOTE)
        win = (onsets >= lo) & (onsets <= n["start"] + 0.03)
        if np.any(win):
            n["start"] = float(onsets[win][np.argmax(onset_strength[win])])
        if anchored and anchored[-1]["end"] > n["start"]:
            anchored[-1]["end"] = n["start"]
        anchored.append(n)

    # 4) wysokość po ataku, pewność, głośność
    loud_ref = np.percentile(rms, 98) + 1e-9
    out = []
    for n in anchored:
        if n["end"] - n["start"] < MIN_NOTE:
            continue
        body = (t >= n["start"] + SETTLE) & (t < n["end"]) & voiced
        if np.any(body):
            n["pitch"] = int(np.round(np.median(midi[body])))
            conf = float(np.mean(track.confidence[body]))
        else:
            conf = track.threshold
        if not BASS_MIDI_MIN <= n["pitch"] <= BASS_MIDI_MAX:
            continue
        attack = (rms_t >= n["start"]) & (rms_t <= n["start"] + 0.08)
        peak = float(np.max(rms[attack])) if np.any(attack) else 0.0
        out.append({
            "start": round(n["start"], 4),
            "end": round(n["end"], 4),
            "pitch": n["pitch"],
            "velocity": int(np.clip(40 + 87 * peak / loud_ref, 40, 127)),
            "confidence": round(conf, 3),
        })
    return out


def _absorb_short_runs(q: np.ndarray, min_run: int) -> np.ndarray:
    """Krótkie wtrącenia innej wysokości (skoki oktawowe, ślizgi) wchłania sąsiedni dźwięk."""
    q = q.copy()
    n = len(q)
    i = 0
    while i < n:
        j = i
        while j < n and q[j] == q[i]:
            j += 1
        if q[i] >= 0 and j - i < min_run:
            prev_p = q[i - 1] if i > 0 else -1
            next_p = q[j] if j < n else -1
            fill = prev_p if prev_p >= 0 else next_p
            if prev_p >= 0 and next_p >= 0 and prev_p != next_p:
                fill = prev_p  # wtrącenie między dwoma dźwiękami: przedłuż poprzedni
            if fill >= 0:
                q[i:j] = fill
        i = j
    return q


# ────────────────────────────────────────────────
# Cały proces
# ────────────────────────────────────────────────
def to_mono(wav: np.ndarray) -> np.ndarray:
    """[kanały, próbki] / [próbki, kanały] / [próbki] -> [próbki] float32."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=0 if wav.shape[0] <= 2 else 1)
    return wav


@dataclass
class Analysis:
    """Kosztowna część transkrypcji — zależy tylko od audio i silnika, nie od suwaków."""
    grid: BeatGrid
    track: PitchTrack
    onsets: np.ndarray
    onset_strength: np.ndarray
    rms_t: np.ndarray
    rms: np.ndarray
    duration: float


def analyze(bass: np.ndarray, sr: int, mix: Optional[np.ndarray] = None,
            engine: str = "crepe", slider: int = 40,
            grid: Optional[BeatGrid] = None, track: Optional[PitchTrack] = None) -> Analysis:
    """`grid` / `track` można podać z cache (ewaluacja) — pozostałe kroki są tanie."""
    import librosa
    y = to_mono(bass)
    duration = len(y) / sr

    if grid is None:
        grid = track_beats(to_mono(mix if mix is not None else bass), sr, duration)
    log.info("  Beaty: %.1f BPM, metrum %d/4", grid.bpm, grid.beats_per_bar)

    if track is None:
        track = (track_pitch_pyin if engine == "pyin" else track_pitch_crepe)(y, sr, slider)
    y22 = librosa.resample(y, orig_sr=sr, target_sr=22050) if sr != 22050 else y
    onsets, strength = detect_onsets(y22)
    rms_t, rms = rms_envelope(y22)
    return Analysis(grid, track, onsets, strength, rms_t, rms, duration)


def finish(a: Analysis, quantize_mode: str = "auto") -> Transcription:
    """Segmentacja i kwantyzacja — szybkie, można powtarzać przy zmianie ustawień."""
    voiced = (a.track.confidence >= a.track.threshold) & np.isfinite(a.track.midi)
    tuning = estimate_tuning(a.track, voiced)
    log.info("  Strój: %+.0f centów względem A=440", tuning * 100)
    raw = segment_notes(a.track, a.onsets, a.onset_strength, a.rms_t, a.rms, tuning)
    grid = copy.deepcopy(a.grid)  # quantize przesuwa numerację taktów
    if quantize_mode == "auto":
        quantize_mode = choose_grid(raw, grid)
    notes = quantize(raw, grid, quantize_mode)
    return Transcription(notes=notes, raw_notes=raw, grid=grid, duration=a.duration,
                         tuning_cents=round(tuning * 100, 1), quantize_mode=quantize_mode)


def transcribe(bass: np.ndarray, sr: int, mix: Optional[np.ndarray] = None,
               engine: str = "crepe", slider: int = 40, quantize_mode: str = "auto") -> Transcription:
    return finish(analyze(bass, sr, mix, engine, slider), quantize_mode)
