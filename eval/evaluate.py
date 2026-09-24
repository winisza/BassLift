"""Ewaluacja transkrypcji BassLift na zestawach z ground truth.

    python eval/make_synthetic.py                      # raz: zestaw syntetyczny
    python eval/evaluate.py                            # wszystkie zestawy, CREPE
    python eval/evaluate.py --engine pyin --dataset synthetic
    python eval/evaluate.py --baseline sciezka/do/starego/server.py

Zestawy (eval/data/):
  synthetic/    — eval/make_synthetic.py (nuty, beaty, pierwsze miary, odstrojenie)
  babyslakh_16k/ — 20 utworów Slakh2100 z MIDI: https://zenodo.org/records/4603870

Metryki (mir_eval): F1 nut (onset ±50 ms, wysokość ±50 centów) przed i po
kwantyzacji, F-measure beatów i pierwszych miar (±70 ms), metrum, błąd tempa.
Separacja Demucs i tory wysokości są cache'owane w eval/cache/ (--fresh = od nowa).
Syntetyka domyślnie używa czystej ścieżki basu (--separation oracle): Demucs myli
syntetyczny pluck z „innymi” instrumentami, a ten zestaw ma mierzyć transkrypcję i rytm.
"""
import argparse
import importlib.util
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

# CPU: przewidywalne zużycie RAM i powtarzalne wyniki (BASSLIFT_DEVICE=mps, żeby liczyć na GPU)
os.environ.setdefault("BASSLIFT_DEVICE", "cpu")

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mir_eval  # noqa: E402

from basslift import separation, transcribe as tr  # noqa: E402

DATA = ROOT / "eval" / "data"
CACHE = ROOT / "eval" / "cache"
log = logging.getLogger("basslift")


# ────────────────────────────────────────────────
# Zestawy danych
# ────────────────────────────────────────────────
def load_synthetic():
    for d in sorted((DATA / "synthetic").glob("*/")):
        if not (d / "truth.json").exists():
            continue
        truth = json.loads((d / "truth.json").read_text())
        truth["oracle_bass"] = [d / "bass.wav"]
        yield "synthetic", d.name, d / "mix.wav", truth


def load_babyslakh():
    import pretty_midi
    import yaml
    for d in sorted((DATA / "babyslakh_16k").glob("Track*/")):
        meta = yaml.safe_load((d / "metadata.yaml").read_text())
        # flagi audio_rendered/midi_saved w BabySlakh bywają false mimo plików — patrz na pliki
        bass_stems = [k for k, v in meta["stems"].items()
                      if v.get("inst_class") == "Bass" and (d / "stems" / f"{k}.wav").exists()
                      and (d / "MIDI" / f"{k}.mid").exists()]
        if not bass_stems:
            continue
        notes = []
        for k in bass_stems:
            for inst in pretty_midi.PrettyMIDI(str(d / "MIDI" / f"{k}.mid")).instruments:
                # MIDI basu w Slakh jest w wysokości zapisanej — audio brzmi oktawę niżej
                # (sprawdzone pyin na czystych ścieżkach: różnica dokładnie -12)
                notes += [{"start": n.start, "end": n.end, "pitch": n.pitch - 12} for n in inst.notes]
        pm = pretty_midi.PrettyMIDI(str(d / "all_src.mid"))
        time_sigs = [(ts.time, ts.numerator) for ts in pm.time_signature_changes if ts.denominator == 4]
        yield "babyslakh", d.name, d / "mix.wav", {
            "notes": sorted(notes, key=lambda n: n["start"]),
            "beats": pm.get_beats().tolist(),
            "downbeats": pm.get_downbeats().tolist(),
            "time_sig": time_sigs[0][1] if time_sigs else 4,
            "time_sigs": time_sigs,
            "bpm": float(np.median(60 / np.diff(pm.get_beats()))),
            "oracle_bass": [d / "stems" / f"{k}.wav" for k in bass_stems],
        }


DATASETS = {"synthetic": load_synthetic, "babyslakh": load_babyslakh}


# ────────────────────────────────────────────────
# Cache: separacja i kosztowna analiza
# ────────────────────────────────────────────────
def read_crop(path, crop):
    info = sf.info(str(path))
    t0, t1 = crop if crop else (0, None)
    start = int(t0 * info.samplerate)
    stop = int(t1 * info.samplerate) if t1 else None
    return sf.read(str(path), start=start, stop=stop, dtype="float32", always_2d=True)


def crop_truth(truth, crop):
    """Ground truth przycięty do okna [t0, t1) i przesunięty tak, by okno zaczynało się od 0 s."""
    if not crop:
        return truth
    t0, t1 = crop
    shift = lambda xs: [x - t0 for x in xs if t0 <= x < t1]
    in_force = [n for t, n in truth.get("time_sigs", []) if t <= t0 + 0.1]  # metrum w oknie
    return {**truth, "time_sig": in_force[-1] if in_force else truth["time_sig"],
            "notes": [{**n, "start": n["start"] - t0, "end": min(n["end"], t1) - t0}
                      for n in truth["notes"] if t0 <= n["start"] < t1 - 0.05],
            "beats": shift(truth["beats"]), "downbeats": shift(truth["downbeats"])}


def separated(d, mix_path, truth, mode, crop, fresh, model="htdemucs"):
    if mode == "oracle":  # czysty bas z danych, reszta = miks - bas
        bass, sr = read_crop(truth["oracle_bass"][0], crop)
        for extra in truth["oracle_bass"][1:]:
            bass = bass + read_crop(extra, crop)[0]
        mix, _ = read_crop(mix_path, crop)
        n = min(len(mix), len(bass))
        return truth["oracle_bass"][0], bass[:n], mix[:n] - bass[:n], sr
    bass_p, rest_p, src_p = d / "bass.wav", d / "rest.wav", d / "source.wav"
    if fresh or not src_p.exists():
        if not (d / "mix.wav").exists() or fresh:
            mix, sr = read_crop(mix_path, crop)
            sf.write(d / "mix.wav", mix, sr)
        sr, stems = separation.separate(d / "mix.wav", model)
        bass, rest = separation.two_stems(stems, "bass")
        source, fallback = separation.bass_for_transcription(stems, sr)  # jak w server.py
        separation.save_wav(bass_p, bass, sr)
        separation.save_wav(rest_p, rest, sr)
        separation.save_wav(src_p, source, sr)
        (d / "fallback").write_text("1") if fallback else (d / "fallback").unlink(missing_ok=True)
    bass, sr = sf.read(bass_p, dtype="float32")
    rest, _ = sf.read(rest_p, dtype="float32")
    source, _ = sf.read(src_p, dtype="float32")
    return bass_p, source, bass + rest - source, sr


def cached(path: Path, fresh, compute):
    if not fresh and path.exists():
        return pickle.loads(path.read_bytes())
    value = compute()
    path.write_bytes(pickle.dumps(value))
    return value


# ────────────────────────────────────────────────
# Metryki
# ────────────────────────────────────────────────
def note_scores(ref, est):
    if not est or not ref:
        return 0.0, 0.0, 0.0
    to_iv = lambda ns: np.array([[n["start"], max(n["end"], n["start"] + 1e-3)] for n in ns])
    to_hz = lambda ns: np.array([440 * 2 ** ((n["pitch"] - 69) / 12) for n in ns])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        to_iv(ref), to_hz(ref), to_iv(est), to_hz(est), onset_tolerance=0.05, offset_ratio=None)
    return p, r, f


def beat_scores(truth, grid):
    ref_b, ref_db = np.array(truth["beats"]), np.array(truth["downbeats"])
    period = np.median(np.diff(ref_b))
    keep = (grid.times >= ref_b[0] - period / 2) & (grid.times <= ref_b[-1] + period / 2)
    est_b = grid.times[keep]
    est_db = grid.times[keep & (np.mod(grid.positions, grid.beats_per_bar) == 0)]
    return (mir_eval.beat.f_measure(ref_b, est_b),
            mir_eval.beat.f_measure(ref_db, est_db) if len(ref_db) > 1 else float("nan"))


def evaluate_track(dataset, name, mix_path, truth, engine, separation_mode, crop, fresh, baseline,
                   model="htdemucs"):
    mode = separation_mode or ("oracle" if dataset == "synthetic" else "demucs")
    crop = crop if dataset != "synthetic" else None  # syntetyka jest krótka
    tag = (mode if mode == "oracle" or model == "htdemucs" else model) + (f"_{crop[0]:g}-{crop[1]:g}s" if crop else "")
    d = CACHE / dataset / name / tag
    d.mkdir(parents=True, exist_ok=True)
    bass_p, bass, rest, sr = separated(d, mix_path, truth, mode, crop, fresh, model)
    truth = crop_truth(truth, crop)
    from basslift import rhythm
    raw_beats = cached(d / "beats.pkl", fresh, lambda: rhythm.detect_beats(tr.to_mono(bass + rest), sr))
    beats, downbeats, downbeat_act = raw_beats
    grid = rhythm.build_grid(beats, downbeats, len(bass) / sr, downbeat_act)  # zawsze świeży
    t0 = time.time()
    fb = "_fallback" if (d / "fallback").exists() else ""  # inne źródło -> inny cache wysokości
    track = cached(d / f"pitch_{engine}{fb}.pkl", fresh,
                   lambda: (tr.track_pitch_pyin if engine == "pyin" else tr.track_pitch_crepe)(tr.to_mono(bass), sr, 40))
    result = tr.finish(tr.analyze(bass, sr, engine=engine, grid=grid, track=track))
    ref = truth["notes"]
    row = {
        "dataset": dataset, "track": name, "ref_notes": len(ref), "notes": len(result.notes),
        "F_raw": note_scores(ref, result.raw_notes)[2],
        "F_final": note_scores(ref, result.notes)[2],
        "P_final": note_scores(ref, result.notes)[0],
        "R_final": note_scores(ref, result.notes)[1],
        "bpm": result.grid.bpm, "bpm_true": truth["bpm"],
        "meter_ok": int(result.grid.beats_per_bar == truth["time_sig"]),
        "tuning": result.tuning_cents, "tuning_true": truth.get("tuning_cents"),
        "fallback": int(bool(fb)),
    }
    row["beat_F"], row["downbeat_F"] = beat_scores(truth, result.grid)
    row["bpm_err_%"] = 100 * abs(row["bpm"] - row["bpm_true"]) / row["bpm_true"]
    if baseline:
        fn = baseline.transcribe_bass_crepe if engine == "crepe" else baseline.transcribe_bass
        notes, bpm, _, ts = fn(bass_p, 40, "16")
        row["base_F_final"] = note_scores(ref, notes)[2]
        row["base_bpm_err_%"] = 100 * abs(bpm - truth["bpm"]) / truth["bpm"]
        row["base_meter_ok"] = int(ts == truth["time_sig"])
    row["sec"] = round(time.time() - t0, 1)
    return row


COLUMNS = ["notes", "F_raw", "F_final", "P_final", "R_final", "beat_F", "downbeat_F",
           "meter_ok", "bpm_err_%", "base_F_final", "base_bpm_err_%", "base_meter_ok"]


def print_table(rows):
    cols = [c for c in COLUMNS if any(c in r for r in rows)]
    head = f"{'utwór':24s}" + "".join(f"{c:>15s}" for c in cols)
    print(head)
    fmt = lambda v: f"{v:15.2f}" if isinstance(v, float) else f"{v:15d}"
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        sub = [r for r in rows if r["dataset"] == ds]
        for r in sub:
            print(f"{r['track'][:24]:24s}" + "".join(fmt(r[c]) if c in r else " " * 15 for c in cols))
        means = {c: float(np.nanmean([r[c] for r in sub if c in r])) for c in cols}
        print(f"{'ŚREDNIA ' + ds:24s}" + "".join(f"{means[c]:15.2f}" for c in cols))
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), action="append")
    ap.add_argument("--engine", choices=["crepe", "pyin"], default="crepe")
    ap.add_argument("--separation", choices=["demucs", "oracle"],
                    help="źródło basu (domyślnie: oracle dla syntetyki, demucs dla Slakh)")
    ap.add_argument("--demucs-model", default="htdemucs", help="np. htdemucs_ft")
    ap.add_argument("--baseline", type=Path, help="stary server.py do porównania")
    ap.add_argument("--limit", type=int, help="maks. liczba utworów na zestaw")
    ap.add_argument("--crop", type=float, nargs=2, default=[30, 90], metavar=("OD", "DO"),
                    help="fragment utworów Slakh w sekundach (domyślnie 30 90); --full = całość")
    ap.add_argument("--full", action="store_true", help="całe utwory zamiast fragmentu")
    ap.add_argument("--fresh", action="store_true", help="przelicz separację i analizę")
    ap.add_argument("--json", type=Path, help="zapisz wyniki do pliku")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    baseline = None
    if args.baseline:
        spec = importlib.util.spec_from_file_location("baseline_server", args.baseline)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        logging.getLogger("basslift").setLevel(logging.WARNING)
        # Stary kod liczył CREPE batchem 2048 (~76 GB RAM) — przytnij batch i uszanuj BASSLIFT_DEVICE
        import torchcrepe
        predict = torchcrepe.predict
        torchcrepe.predict = lambda *a, **k: predict(*a, **{**k, "batch_size": min(k.get("batch_size") or 32, 32)})
        baseline._torch_device = separation.torch_device

    rows = []
    for ds in args.dataset or list(DATASETS):
        tracks = list(DATASETS[ds]())
        if not tracks:
            print(f"(brak danych: eval/data/{ds} — patrz nagłówek pliku)")
        for spec in tracks[:args.limit]:
            rows.append(evaluate_track(*spec, args.engine, args.separation,
                                      None if args.full else tuple(args.crop), args.fresh, baseline,
                                      args.demucs_model))
            r = rows[-1]
            print(f"  {r['dataset']}/{r['track']}: F_final={r['F_final']:.2f} ({r['sec']} s)", file=sys.stderr)
    if rows:
        print_table(rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
