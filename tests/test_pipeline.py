"""Testy rytmu, segmentacji i zapisu nut — bez modeli i bez audio (szybkie, lekkie)."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basslift import notation, rhythm, transcribe as tr  # noqa: E402


def make_grid(offset=0.43, period=0.5, n=40, bpb=4, first_downbeat=0, duration=None):
    beats = offset + period * np.arange(n)
    downbeats = beats[first_downbeat::bpb]
    return rhythm.build_grid(beats, downbeats, duration or beats[-1] + 1)


def note(start, end, pitch, conf=0.9):
    return {"start": start, "end": end, "pitch": pitch, "velocity": 90, "confidence": conf}


# ── rytm ─────────────────────────────────────────
def test_grid_follows_offset_and_meter():
    g = make_grid(offset=0.43, bpb=3)
    assert g.beats_per_bar == 3
    assert g.bpm == 120.0
    assert g.to_beats(0.43) % 3 == 0  # pierwszy beat to „raz”, niezależnie od ciszy przed nim
    q = rhythm.quantize([note(0.43, 0.6, 28)], g, "16")
    assert q[0]["beat"] == 0.0 and abs(g.to_seconds(0) - 0.43) < 1e-9


def test_pickup_bar_starts_before_first_downbeat():
    g = make_grid(offset=0.5, first_downbeat=1)  # pierwszy beat to przedtakt
    assert g.to_beats(0.5) % 4 == 3
    assert abs(g.to_seconds(4) - 1.0) < 1e-9


def test_quantize_keeps_repeated_notes_and_snaps_to_grid():
    g = make_grid(offset=0.43)
    eighth = 0.25
    notes = [note(0.43 + i * eighth + 0.012, 0.43 + (i + 1) * eighth - 0.03, 28) for i in range(8)]
    q = rhythm.quantize(notes, g, "16")
    assert [n["beat"] for n in q] == [i * 0.5 for i in range(8)]  # 8 ósemek, nic nie scalone
    assert all(abs(n["start"] - (0.43 + i * eighth)) < 1e-3 for i, n in enumerate(q))


def test_quantize_follows_tempo_change():
    beats = np.concatenate([np.arange(0, 9) * 0.5, 4.0 + np.arange(1, 9) * 0.4])  # 120 -> 150 BPM
    g = rhythm.build_grid(beats, beats[::4], 8.0)
    q = rhythm.quantize([note(0.0, 0.2, 28), note(4.0 + 6 * 0.4, 4.0 + 6.5 * 0.4, 33)], g, "16")
    assert [n["beat"] for n in q] == [0.0, 14.0]  # stała siatka 120 BPM dałaby 12.8


def test_quantize_never_returns_negative_times():
    g = make_grid(offset=0.1)  # beat ekstrapolowany przed 0 s leży na -0.4 s
    q = rhythm.quantize([note(0.0, 0.2, 28), note(0.6, 0.8, 31)], g, "16")
    assert all(n["start"] >= 0 and n["end"] > n["start"] for n in q)


def test_quantize_drops_leading_empty_bars():
    g = make_grid(offset=0.0)
    q = rhythm.quantize([note(8.0, 8.4, 28)], g, "16")  # takt 5 w siatce
    assert q[0]["beat"] == 0.0 and g.to_seconds(0) == 8.0


# ── segmentacja ──────────────────────────────────
def pitch_track(segments, hop=0.01, total=3.0, latency=0.05):
    t = np.arange(0, total, hop)
    midi = np.full(len(t), np.nan)
    conf = np.zeros(len(t))
    for start, end, p in segments:
        m = (t >= start) & (t < end)
        midi[m], conf[m] = p + 0.03, 0.9
    return tr.PitchTrack(times=t, midi=midi, confidence=conf, threshold=0.3, latency=latency)


def test_segmentation_splits_repeats_and_compensates_latency():
    # tracker widzi jeden długi E1 zaczęty 40 ms za późno; onsety pokazują 4 uderzenia
    track = pitch_track([(0.54, 1.5, 28)])
    onsets = np.array([0.5, 0.75, 1.0, 1.25])
    rms_t = np.arange(0, 3, 0.0116)
    notes = tr.segment_notes(track, onsets, np.ones(4), rms_t, np.ones_like(rms_t), tuning=0.0)
    assert [round(n["start"], 2) for n in notes] == [0.5, 0.75, 1.0, 1.25]
    assert all(n["pitch"] == 28 for n in notes)


def test_tuning_offset_is_removed():
    track = pitch_track([(0.1, 1.0, 40), (1.0, 2.0, 45)])
    track.midi = track.midi + 0.3  # nagranie +33 centy
    voiced = track.confidence >= track.threshold
    assert abs(tr.estimate_tuning(track, voiced) - 0.33) < 0.01


# ── notacja ──────────────────────────────────────
@pytest.mark.parametrize("names,expected", [
    ("EADG", [28, 33, 38, 43]),
    ("DADG", [26, 33, 38, 43]),
    ("BEAD", [23, 28, 33, 38]),
    ("CGCF", [24, 31, 36, 41]),
])
def test_tuning_octaves(names, expected):
    assert notation.tuning_to_midi(list(names)) == expected


def test_fingering_stays_in_position():
    # C-D-E-F-G w piątej pozycji: nie skacze po gryfie ani na puste struny
    pos = notation.fingering([36, 38, 40, 41, 43], [28, 33, 38, 43])
    frets = [f for _, f in pos]
    assert max(frets) - min(frets) <= 4
    assert notation.fingering([10], [28, 33, 38, 43]) == [None]  # poniżej gryfu


def test_tab_first_note_on_downbeat():
    g = make_grid(offset=0.43)
    q = rhythm.quantize([note(0.43, 0.6, 28), note(0.68, 0.8, 31)], g, "16")
    tab = notation.generate_tab(q, g, ["E", "A", "D", "G"], "16").splitlines()
    e_line = next(line for line in tab if line.startswith("E "))
    assert e_line.startswith("E |0-----3")  # „raz” i druga ósemka (kolumna 2 z 16)


def test_musicxml_measures_are_full_and_tied_across_barline():
    g = make_grid(offset=0.0)
    q = rhythm.quantize([note(0.0, 0.25, 28), note(1.5, 2.5, 33)], g, "16")  # druga przez kreskę
    root = ET.fromstring(notation.notes_to_musicxml(q, g).split("\n", 2)[2])
    measures = root.findall("./part/measure")
    for m in measures:
        assert sum(int(n.find("duration").text) for n in m.findall("note")) == 4 * notation.DIVISIONS
    ties = [t.get("type") for t in root.iter("tie")]
    assert ties.count("start") == ties.count("stop") >= 1


def test_midi_export_has_tempo_map():
    g = make_grid(offset=0.0)
    q = rhythm.quantize([note(0.0, 0.4, 28)], g, "16")
    assert notation.notes_to_midi_b64(q, g)
