"""Zapis wyniku: tabulatura ASCII (z palcowaniem całej frazy), MIDI i MusicXML.

Wszystkie formaty biorą pozycje nut z siatki beatów (`beat`, `beats` w ćwierćnutach
od „raz” taktu 1), więc kreski taktowe zgadzają się z utworem nawet przy zmiennym tempie.
"""
import base64
import io
import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from .rhythm import SUBDIVISIONS, BeatGrid

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
STANDARD_OPEN = [28, 33, 38, 43]  # E1 A1 D2 G2 — od najgrubszej struny
MAX_FRET = 20
QUANT_LABELS = {"8": "1/8", "8t": "1/8T", "16": "1/16", "16t": "1/16T"}


# ────────────────────────────────────────────────
# Strój i palcowanie
# ────────────────────────────────────────────────
def tuning_to_midi(tuning_list: List[str]) -> List[int]:
    """Nazwy strun (od najgrubszej) -> MIDI pustych strun. Oktawa najbliższa standardowi,
    więc drop D daje D1, a B-E-A-D daje B0 (stary kod miał oktawy wpisane na sztywno)."""
    out = []
    for i, name in enumerate(tuning_list[:4]):
        n = FLATS.get(name.strip().upper(), name.strip().upper())
        idx = NOTE_NAMES.index(n) if n in NOTE_NAMES else 4
        out.append(min((12 * o + idx for o in range(6)), key=lambda m: (abs(m - STANDARD_OPEN[i]), m)))
    return out


def fingering(pitches: List[int], open_pitches: List[int]) -> List[Optional[Tuple[int, int]]]:
    """(struna, próg) dla każdej nuty — Viterbi minimalizujący ruchy ręki w całej frazie
    zamiast zachłannego wyboru nuta po nucie. Nuty poza gryfem -> None."""
    cands = [[(s, p - o) for s, o in enumerate(open_pitches) if 0 <= p - o <= MAX_FRET]
             for p in pitches]

    def position_cost(fret):
        return 0.12 * fret + (0.8 if fret > 12 else 0.0)

    def move_cost(a, b):
        (sa, fa), (sb, fb) = a, b
        cost = 0.3 * abs(sa - sb)
        if fa and fb:  # pusta struna nie wymaga ruchu ręki
            shift = abs(fa - fb)
            cost += 0.25 * shift + 1.2 * max(0, shift - 3)  # rozpiętość dłoni ~4 progi
        return cost

    result: List[Optional[Tuple[int, int]]] = [None] * len(pitches)
    idx = [i for i, c in enumerate(cands) if c]
    if not idx:
        return result
    cost = [position_cost(f) for _, f in cands[idx[0]]]
    back = []
    for prev_i, i in zip(idx, idx[1:]):
        new_cost, ptr = [], []
        for pos in cands[i]:
            options = [cost[k] + move_cost(p, pos) for k, p in enumerate(cands[prev_i])]
            k = min(range(len(options)), key=options.__getitem__)
            new_cost.append(options[k] + position_cost(pos[1]))
            ptr.append(k)
        cost = new_cost
        back.append(ptr)
    k = min(range(len(cost)), key=cost.__getitem__)
    for step in range(len(idx) - 1, -1, -1):
        result[idx[step]] = cands[idx[step]][k]
        if step:
            k = back[step - 1][k]
    return result


# ────────────────────────────────────────────────
# Tabulatura ASCII
# ────────────────────────────────────────────────
def generate_tab(notes: List[Dict], grid: BeatGrid, tuning_list: List[str],
                 quantize_mode: str = "16", tuning_cents: float = 0.0) -> str:
    if not notes:
        return "(brak wykrytych nut — spróbuj obniżyć próg pewności)"

    names = [n.strip().upper() or "?" for n in tuning_list[:4]]
    sub = SUBDIVISIONS.get(quantize_mode, 4)
    bpb = grid.beats_per_bar
    cols, cw = sub * bpb, 3
    bars_per_line = max(1, 100 // (cols * cw + 1))

    cells: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for n, pos in zip(notes, fingering([n["pitch"] for n in notes], tuning_to_midi(tuning_list))):
        if pos is None:
            continue
        slot = int(round(n["beat"] * sub))
        cells.setdefault(divmod(slot, cols), pos)
    n_bars = max(bar for bar, _ in cells) + 1 if cells else 1

    out = []
    for first in range(0, n_bars, bars_per_line):
        bars = range(first, min(first + bars_per_line, n_bars))
        out.append("   " + "".join(str(b + 1).ljust(cols * cw + 1) for b in bars))
        for si in reversed(range(4)):  # G na górze
            row = names[si].ljust(2) + "|"
            for b in bars:
                for c in range(cols):
                    pos = cells.get((b, c))
                    row += str(pos[1]).ljust(cw, "-") if pos and pos[0] == si else "-" * cw
                row += "|"
            out.append(row)
        out.append("")

    footer = (f"Stroj: {'-'.join(names)}   BPM: {grid.bpm:.0f}   Nuty: {len(notes)}   "
              f"Siatka: {QUANT_LABELS.get(quantize_mode, '1/16')}   Metrum: {bpb}/4")
    if abs(tuning_cents) >= 10:
        footer += f"   Nagranie: {tuning_cents:+.0f} centow od A=440"
    return "\n".join(out + [footer])


# ────────────────────────────────────────────────
# MIDI
# ────────────────────────────────────────────────
def notes_to_midi_b64(notes: List[Dict], grid: BeatGrid) -> Optional[str]:
    try:
        from midiutil import MIDIFile
    except ImportError:
        return None
    if not notes:
        return None
    bpb = grid.beats_per_bar
    midi = MIDIFile(1)
    # Mapa tempa takt po takcie — MIDI zostaje zgrane z nagraniem przy zmiennym tempie
    n_bars = int((notes[-1]["beat"] + notes[-1]["beats"]) // bpb) + 1
    prev = None
    for bar, tempo in enumerate(grid.bar_tempos(n_bars)):
        tempo = round(tempo, 1)
        if prev is None or abs(tempo - prev) >= 0.5:
            midi.addTempo(0, bar * bpb, tempo)
            prev = tempo
    midi.addTimeSignature(0, 0, bpb, 2, 24)  # mianownik jako potęga dwójki: 2 -> ćwierćnuta
    midi.addProgramChange(0, 0, 0, 33)      # Electric Bass (finger)
    for n in notes:
        midi.addNote(0, 0, n["pitch"], n["beat"], max(n["beats"], 0.05),
                     max(1, min(127, n.get("velocity", 90))))
    buf = io.BytesIO()
    midi.writeFile(buf)
    return base64.b64encode(buf.getvalue()).decode()


# ────────────────────────────────────────────────
# MusicXML
# ────────────────────────────────────────────────
DIVISIONS = 12  # na ćwierćnutę: szesnastka = 3, triola ósemkowa = 4, triola szesnastkowa = 2
# (długość, typ, kropki, triola) od najdłuższej — rozkład zachłanny z łukami
PIECES = [(48, "whole", 0, False), (36, "half", 1, False), (24, "half", 0, False),
          (18, "quarter", 1, False), (12, "quarter", 0, False), (9, "eighth", 1, False),
          (6, "eighth", 0, False), (4, "eighth", 0, True), (3, "16th", 0, False),
          (2, "16th", 0, True), (1, "32nd", 0, True)]
STEPS = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
ALTER = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]


def _pieces(total: int):
    while total > 0:
        piece = next(p for p in PIECES if p[0] <= total)
        yield piece
        total -= piece[0]


def _add_note(measure, pitch: Optional[int], piece, tie_start=False, tie_stop=False):
    dur, kind, dots, triplet = piece
    note = ET.SubElement(measure, "note")
    if pitch is None:
        ET.SubElement(note, "rest")
    else:
        p = ET.SubElement(note, "pitch")
        ET.SubElement(p, "step").text = STEPS[pitch % 12]
        if ALTER[pitch % 12]:
            ET.SubElement(p, "alter").text = "1"
        ET.SubElement(p, "octave").text = str(pitch // 12 - 1)
    ET.SubElement(note, "duration").text = str(dur)
    if tie_stop:
        ET.SubElement(note, "tie", type="stop")
    if tie_start:
        ET.SubElement(note, "tie", type="start")
    ET.SubElement(note, "type").text = kind
    for _ in range(dots):
        ET.SubElement(note, "dot")
    if triplet:
        tm = ET.SubElement(note, "time-modification")
        ET.SubElement(tm, "actual-notes").text = "3"
        ET.SubElement(tm, "normal-notes").text = "2"
    if tie_start or tie_stop:
        notations = ET.SubElement(note, "notations")
        if tie_stop:
            ET.SubElement(notations, "tied", type="stop")
        if tie_start:
            ET.SubElement(notations, "tied", type="start")


def notes_to_musicxml(notes: List[Dict], grid: BeatGrid) -> Optional[str]:
    if not notes:
        return None
    bpb = grid.beats_per_bar
    capacity = bpb * DIVISIONS
    events = [(int(round(n["beat"] * DIVISIONS)), max(1, int(round(n["beats"] * DIVISIONS))), n["pitch"])
              for n in notes]
    n_measures = max(1, math.ceil(max(s + d for s, d, _ in events) / capacity))

    root = ET.Element("score-partwise", version="4.0")
    score_part = ET.SubElement(ET.SubElement(root, "part-list"), "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Bass"
    part = ET.SubElement(root, "part", id="P1")

    for m in range(n_measures):
        measure = ET.SubElement(part, "measure", number=str(m + 1))
        if m == 0:
            attrs = ET.SubElement(measure, "attributes")
            ET.SubElement(attrs, "divisions").text = str(DIVISIONS)
            ET.SubElement(ET.SubElement(attrs, "key"), "fifths").text = "0"
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(bpb)
            ET.SubElement(time_el, "beat-type").text = "4"
            clef = ET.SubElement(attrs, "clef")
            ET.SubElement(clef, "sign").text = "F"
            ET.SubElement(clef, "line").text = "4"
            direction = ET.SubElement(measure, "direction", placement="above")
            metro = ET.SubElement(ET.SubElement(direction, "direction-type"), "metronome")
            ET.SubElement(metro, "beat-unit").text = "quarter"
            ET.SubElement(metro, "per-minute").text = str(int(round(grid.bpm)))
            ET.SubElement(direction, "sound", tempo=str(int(round(grid.bpm))))

        m0, m1 = m * capacity, (m + 1) * capacity
        cursor = m0
        for s, d, p in sorted(e for e in events if e[0] < m1 and e[0] + e[1] > m0):
            start, end = max(s, m0, cursor), min(s + d, m1)
            if end <= start:
                continue
            for piece in _pieces(start - cursor):
                _add_note(measure, None, piece)
            pieces = list(_pieces(end - start))
            for k, piece in enumerate(pieces):
                continues = k < len(pieces) - 1 or (s + d > m1 and end == m1)
                continued = k > 0 or start > s
                _add_note(measure, p, piece, tie_start=continues, tie_stop=continued)
            cursor = end
        for piece in _pieces(m1 - cursor):
            _add_note(measure, None, piece)

    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
            '"http://www.musicxml.org/dtds/partwise.dtd">\n' + ET.tostring(root, encoding="unicode"))
