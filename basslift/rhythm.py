"""Rytm: beaty, pierwsze miary taktu, metrum i kwantyzacja nut do siatki beatów.

Siatka pochodzi z trackera beatów uruchamianego na pełnym miksie, więc kreski taktowe
leżą tam, gdzie w utworze, a zmiany tempa nie rozjeżdżają kwantyzacji
(stary kod liczył stałą siatkę 60/BPM od 0 s).
"""
import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

log = logging.getLogger("basslift")

SUBDIVISIONS = {"8": 2, "16": 4, "8t": 3, "16t": 6}   # podziały na jeden beat (ćwierćnutę)

_beat_model = None


@dataclass
class BeatGrid:
    times: np.ndarray       # czasy beatów [s] pokrywające cały utwór (z ekstrapolacją)
    positions: np.ndarray   # pozycja muzyczna beatu w ćwierćnutach; 0 = „raz” taktu 1
    beats_per_bar: int
    bpm: float

    def to_beats(self, t):
        return _interp(t, self.times, self.positions)

    def to_seconds(self, b):
        return _interp(b, self.positions, self.times)

    def shift_bars(self, bars: int):
        self.positions = self.positions - bars * self.beats_per_bar

    def bar_tempos(self, n_bars: int) -> List[float]:
        """Średnie tempo każdego taktu (do mapy tempa w MIDI)."""
        tempos = []
        for bar in range(n_bars):
            b0 = bar * self.beats_per_bar
            dur = self.to_seconds(b0 + self.beats_per_bar) - self.to_seconds(b0)
            tempos.append(60.0 * self.beats_per_bar / dur if dur > 0 else self.bpm)
        return tempos


def _interp(x, xp, fp):
    """np.interp z liniową ekstrapolacją poza zakresem."""
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    if np.any(lo):
        y = np.where(lo, fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), y)
    if np.any(hi):
        y = np.where(hi, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)
    return y if y.ndim else float(y)


# ────────────────────────────────────────────────
# Tracking beatów
# ────────────────────────────────────────────────
def track_beats(mix: np.ndarray, sr: int, duration: float) -> BeatGrid:
    """mix: [próbki] lub [próbki, kanały]. Zwraca siatkę beatów dla całego utworu."""
    beats, downbeats, downbeat_act = detect_beats(mix, sr)
    return build_grid(beats, downbeats, duration, downbeat_act)


def detect_beats(mix: np.ndarray, sr: int):
    """Kosztowna część: (beaty, pierwsze miary, aktywacja „raz” 50 kl./s albo None)."""
    try:
        beats, downbeats, act = _beat_this(mix, sr)
        if len(beats) >= 4:
            return beats, downbeats, act
    except ImportError:
        log.warning("  beat_this niedostępny — fallback na librosa.beat_track (bez pierwszych miar)")
    return (*_librosa_beats(mix, sr), None)


def _beat_this(mix, sr):
    global _beat_model
    import torch
    from beat_this.inference import Audio2Beats, Audio2Frames
    if _beat_model is None:
        # CPU jest tu szybszy od MPS (mały model, narzut transferu) — ~0.2 s na 20 s audio
        _beat_model = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beat_logits, db_logits = Audio2Frames.__call__(_beat_model, mix, sr)
    beats, downbeats = _beat_model.frames2beats(beat_logits, db_logits)
    act = torch.sigmoid(db_logits).cpu().numpy().ravel()
    return np.asarray(beats, float), np.asarray(downbeats, float), act


def _act_at(act, times):
    idx = np.clip(np.round(np.asarray(times) * 50).astype(int), 0, len(act) - 1)
    return np.array([act[max(0, i - 2):i + 3].max() for i in idx])


def _librosa_beats(mix, sr):
    import librosa
    y = mix.mean(axis=1) if mix.ndim == 2 else mix
    _, frames = librosa.beat.beat_track(y=y.astype(np.float32), sr=sr)
    beats = librosa.frames_to_time(frames, sr=sr)
    return beats, beats[::4]


def _beats_per_bar(beats, downbeats):
    db_idx = sorted({int(np.argmin(np.abs(beats - d))) for d in downbeats})
    counts = [b - a for a, b in zip(db_idx, db_idx[1:]) if 2 <= b - a <= 7]
    return (Counter(counts).most_common(1)[0][0] if counts else 4), db_idx


def _subdivide(times, parts: int):
    """Równy podział każdego odstępu między kolejnymi czasami na `parts` części."""
    k = np.arange((len(times) - 1) * parts + 1) / parts
    return _interp(k, np.arange(len(times), dtype=float), times)


def normalize_meter(beats, downbeats, downbeat_act=None):
    """Typowe pomyłki trackera na muzyce popularnej, ocenione na Slakh:
    pierwsza miara co pół taktu, beaty w półnutach, shuffle liczony w triolach.
    Długość taktu jest z nich najpewniejsza, więc to ją zachowujemy."""
    beats, downbeats = np.asarray(beats, float), np.asarray(downbeats, float)
    if len(beats) < 4 or len(downbeats) < 3:
        return beats, downbeats
    bpb, _ = _beats_per_bar(beats, downbeats)
    bpm = 60.0 / float(np.median(np.diff(beats)))
    if bpb == 2 and bpm < 90:        # beaty to półnuty: 2/4 @ 68 -> 4/4 @ 136
        beats = _subdivide(beats, 2)
    elif bpb == 2:                   # „raz” co pół taktu: zostaw mocniejszą połowę
        phase = 0
        if downbeat_act is not None:
            phase = int(np.argmax([_act_at(downbeat_act, downbeats[p::2]).mean() for p in (0, 1)]))
        downbeats = downbeats[phase::2]
    elif bpb == 6 and bpm > 140:     # shuffle: 6 „beatów” = triole 4/4
        beats = _subdivide(downbeats, 4)
    return beats, downbeats


def build_grid(beats, downbeats, duration: float, downbeat_act=None) -> BeatGrid:
    beats = np.asarray(beats, float)
    if len(beats) < 4:  # cisza albo brak pulsu — siatka 120 BPM od zera
        beats = np.arange(0.0, max(duration, 2.0) + 0.5, 0.5)
        downbeats = beats[::4]
    beats, downbeats = normalize_meter(beats, downbeats, downbeat_act)

    period = float(np.median(np.diff(beats)))
    beats_per_bar, db_idx = _beats_per_bar(beats, downbeats)
    first_db = db_idx[0] if db_idx else 0

    # Dociągnij siatkę do 0 s i za koniec utworu stałym okresem z brzegów
    p_head = float(np.median(np.diff(beats[:5])))
    p_tail = float(np.median(np.diff(beats[-5:])))
    n_head = int(math.ceil(beats[0] / p_head)) + 1
    n_tail = int(math.ceil(max(duration - beats[-1], 0) / p_tail)) + 2
    head = beats[0] - p_head * np.arange(n_head, 0, -1)
    tail = beats[-1] + p_tail * np.arange(1, n_tail + 1)
    times = np.concatenate([head, beats, tail])
    first_db += n_head

    # Takt 1 zaczyna się na ostatniej pierwszej mierze <= pierwszego beatu siatki
    origin = first_db - int(math.ceil(first_db / beats_per_bar)) * beats_per_bar
    positions = np.arange(len(times), dtype=float) - origin
    return BeatGrid(times=times, positions=positions, beats_per_bar=int(beats_per_bar),
                    bpm=round(60.0 / period, 1))


# ────────────────────────────────────────────────
# Kwantyzacja
# ────────────────────────────────────────────────
def choose_grid(notes: List[Dict], grid: BeatGrid) -> str:
    """Tryb „auto”: 1/16, chyba że onsety wyraźnie siedzą na triolach (shuffle, swing).
    Ostrożnie, bo surowe onsety po separacji mają ~20–30 ms rozrzutu (eval Slakh)."""
    if len(notes) < 8:
        return "16"
    frac = np.mod(grid.to_beats([n["start"] for n in notes]), 1.0)
    tol = 0.025 * grid.bpm / 60  # 25 ms w ułamkach beatu
    share = lambda *cs: float(np.mean([np.min(np.abs(f - np.array(cs))) < tol for f in frac]))
    triplets, sixteenths = share(1 / 3, 2 / 3), share(0.25, 0.75)
    return "8t" if triplets >= 0.2 and triplets >= 2 * sixteenths else "16"
def quantize(notes: List[Dict], grid: BeatGrid, mode: str = "16") -> List[Dict]:
    """Przyciąga nuty do podziałów beatu. Dodaje pola `beat` (pozycja od „raz” taktu 1,
    w ćwierćnutach) i `beats` (długość); `start`/`end` przelicza z powrotem na sekundy."""
    if not notes:
        return []
    sub = SUBDIVISIONS.get(mode, 4)
    starts = np.round(grid.to_beats([n["start"] for n in notes]) * sub) / sub
    ends = np.round(grid.to_beats([n["end"] for n in notes]) * sub) / sub

    # Na jednej pozycji zostaje najpewniejsza nuta
    best: Dict[float, Dict] = {}
    for n, b0, b1 in zip(notes, starts, ends):
        q = {**n, "beat": round(float(b0), 6), "beats": round(max(float(b1 - b0), 1 / sub), 6)}
        if q["beat"] not in best or q["confidence"] > best[q["beat"]]["confidence"]:
            best[q["beat"]] = q
    out = [best[k] for k in sorted(best)]

    # Monofonia: nuta kończy się najpóźniej tam, gdzie zaczyna się następna
    for a, b in zip(out, out[1:]):
        a["beats"] = round(min(a["beats"], b["beat"] - a["beat"]), 6)

    # Takt 1 = takt z pierwszą nutą (bez pustych taktów intro)
    lead_bars = int(math.floor(out[0]["beat"] / grid.beats_per_bar))
    if lead_bars:
        grid.shift_bars(lead_bars)
        for n in out:
            n["beat"] = round(n["beat"] - lead_bars * grid.beats_per_bar, 6)

    for n in out:
        # nuta tuż po 0 s może trafić na beat ekstrapolowany przed początkiem nagrania
        n["start"] = round(max(0.0, float(grid.to_seconds(n["beat"]))), 4)
        n["end"] = round(max(n["start"] + 1e-3, float(grid.to_seconds(n["beat"] + n["beats"]))), 4)
    return out
