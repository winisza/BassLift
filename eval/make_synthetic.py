"""Syntetyczny zestaw testowy z pełnym ground truth (nuty, beaty, pierwsze miary).

Każdy utwór celuje w inną słabość transkrypcji: przesunięcie startu względem 0 s,
metrum 3/4, przyspieszające tempo, odstrojenie, gęste szesnastki z powtórzeniami.

    python eval/make_synthetic.py            # -> eval/data/synthetic/<utwór>/
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

SR = 44100
OUT_DIR = Path(__file__).parent / "data" / "synthetic"

SONGS = [
    # nazwa,          bpm, bpm_koniec, metrum, takty, offset, odstrojenie [cent], seed
    ("offset_120",    120, 120,        4,      8,     0.43,   0,                  1),
    ("funk_96",        96,  96,        4,      8,     1.10,   0,                  2),
    ("waltz_140",     140, 140,        3,     12,     0.20,   0,                  3),
    ("accel_100",     100, 112,        4,     12,     0.75,   0,                  4),
    ("detuned_110",   110, 110,        4,      8,     0.30,  30,                  5),
]

# Rytmy taktu w szesnastkach: lista (pozycja, długość); None = pauza
PATTERNS_4 = [
    [(i * 2, 2) for i in range(8)],                                  # ósemki (powtórzenia)
    [(0, 4), (4, 4), (8, 2), (10, 2), (12, 4)],
    [(0, 6), (6, 2), (8, 4), (12, 4)],
    [(0, 1), (1, 1), (2, 1), (3, 1), (4, 2), (6, 2), (8, 4), (12, 4)],
    [(0, 3), (3, 3), (6, 2), (10, 2), (12, 2), (14, 2)],             # synkopy
    [(0, 2), (3, 1), (4, 2), (7, 1), (8, 2), (11, 1), (12, 2), (14, 1), (15, 1)],  # funk
    [(0, 8)],                                                        # półnuta + pauza
]
PATTERNS_3 = [
    [(0, 4), (4, 4), (8, 4)],
    [(0, 6), (6, 2), (8, 4)],
    [(0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2)],
    [(0, 4), (8, 2), (10, 2)],
]
PROGRESSION = [(28, "m"), (33, "m"), (31, "M"), (26, "M")]  # E-, A-, G, D (pryma w oktawie basowej)


def midi_hz(m, cents=0.0):
    return 440.0 * 2 ** ((m - 69 + cents / 100) / 12)


def pluck(freq, dur, rng, velocity=1.0):
    """Karplus-Strong (wektorowo przez lfilter) — szarpana struna z naturalnym wybrzmieniem."""
    n = max(int(dur * SR), 1)
    period = max(int(round(SR / freq - 0.5)), 2)
    x = np.zeros(n)
    burst = rng.uniform(-1, 1, min(period, n))
    x[:len(burst)] = np.convolve(burst, np.ones(3) / 3, mode="same")
    a = np.zeros(period + 2)
    a[0], a[period], a[period + 1] = 1.0, -0.4985, -0.4985
    y = lfilter([1.0], a, x)
    rel = min(int(0.015 * SR), n)
    y[-rel:] *= np.linspace(1, 0, rel)                    # wytłumienie lewą ręką
    return velocity * y / (np.max(np.abs(y)) + 1e-9)


def beat_times(bpm0, bpm1, n_beats, offset):
    bpms = np.linspace(bpm0, bpm1, n_beats + 1)
    return offset + np.concatenate([[0], np.cumsum(60.0 / bpms[:-1])])


def pos_to_time(beats, pos_beats):
    """Pozycja muzyczna (w ćwierćnutach od początku) -> sekundy, liniowo w obrębie beatu."""
    k = min(int(pos_beats), len(beats) - 2)
    return beats[k] + (pos_beats - k) * (beats[k + 1] - beats[k])


def chord_tones(root, quality):
    third = 4 if quality == "M" else 3
    return [root, root + 7, root + 12, root + third + 12, root + 5, root + 10]


def make_song(name, bpm0, bpm1, bpb, n_bars, offset, cents, seed):
    rng = np.random.default_rng(seed)
    patterns = PATTERNS_4 if bpb == 4 else PATTERNS_3
    beats = beat_times(bpm0, bpm1, n_bars * bpb + 1, offset)
    total = beats[-1] + 1.5
    N = int(total * SR)
    bass, drums, pad = np.zeros(N), np.zeros(N), np.zeros(N)
    notes = []

    for bar in range(n_bars):
        root, quality = PROGRESSION[bar % len(PROGRESSION)]
        tones = chord_tones(root, quality)
        pattern = patterns[rng.integers(len(patterns))]
        for pos16, len16 in pattern:
            # pryma na „raz”, dalej losowe dźwięki akordu; czasem ta sama nuta (powtórzenie)
            if pos16 == 0 or rng.random() < 0.35:
                pitch = root if pos16 == 0 else (notes[-1]["pitch"] if notes else root)
            else:
                pitch = int(rng.choice(tones))
            start_b = bar * bpb + pos16 / 4
            end_b = start_b + len16 / 4 * 0.9
            t0, t1 = pos_to_time(beats, start_b), pos_to_time(beats, end_b)
            s = int(t0 * SR)
            x = pluck(midi_hz(pitch, cents), t1 - t0, rng, velocity=rng.uniform(0.6, 1.0))
            bass[s:s + len(x)] += x
            notes.append({"start": round(t0, 4), "end": round(t1, 4), "pitch": pitch})

        # perkusja + akord na tle
        for b in range(bpb):
            t = beats[bar * bpb + b]
            s = int(t * SR)
            hit = _kick(rng) if (b == 0 or (bpb == 4 and b == 2)) else _snare(rng)
            drums[s:s + len(hit)] += hit
            for half in (0, 0.5):
                s2 = int(pos_to_time(beats, bar * bpb + b + half) * SR)
                h = _hat(rng)
                drums[s2:s2 + len(h)] += h
        s, e = int(beats[bar * bpb] * SR), int(beats[(bar + 1) * bpb] * SR)
        tt = np.arange(e - s) / SR
        env = np.minimum(1, tt / 0.05) * np.minimum(1, (tt[-1] - tt) / 0.1)
        for m in (root + 24, root + 24 + (4 if quality == "M" else 3), root + 31):
            f = midi_hz(m, cents)
            pad[s:e] += 0.07 * env * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt))

    mix = np.stack([0.8 * bass + 0.7 * drums + 0.9 * pad,
                    0.8 * bass + 0.63 * drums + 1.0 * pad], axis=1)
    mix += rng.normal(0, 1e-3, mix.shape)
    mix *= 0.89 / np.max(np.abs(mix))

    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "mix.wav", mix, SR)
    sf.write(out / "bass.wav", 0.9 * bass / np.max(np.abs(bass)), SR)
    beat_list = beats[:n_bars * bpb].tolist()
    json.dump({
        "notes": notes,
        "beats": [round(b, 4) for b in beat_list],
        "downbeats": [round(b, 4) for b in beat_list[::bpb]],
        "time_sig": bpb,
        "bpm": (bpm0 + bpm1) / 2,
        "tuning_cents": cents,
    }, open(out / "truth.json", "w"), indent=1)
    return len(notes), total


def _kick(rng):
    t = np.arange(int(0.18 * SR)) / SR
    f = 45 + 80 * np.exp(-t * 30)
    return 0.9 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18)


def _snare(rng):
    t = np.arange(int(0.15 * SR)) / SR
    return (0.5 * rng.uniform(-1, 1, len(t)) + 0.3 * np.sin(2 * np.pi * 185 * t)) * np.exp(-t * 25)


def _hat(rng):
    t = np.arange(int(0.03 * SR)) / SR
    return 0.15 * np.diff(rng.uniform(-1, 1, len(t)), prepend=0) * np.exp(-t * 120)


if __name__ == "__main__":
    for spec in SONGS:
        n, total = make_song(*spec)
        print(f"{spec[0]:14s} {n:3d} nut  {total:5.1f} s")
