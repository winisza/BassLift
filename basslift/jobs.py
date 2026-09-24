"""Zadania w tle: separacja i transkrypcja z raportowaniem postępu.

Przeglądarka dostaje od razu `job_id` i odpytuje o stan (etap + ułamek 0..1).
Wynik analizy (beaty, tor wysokości, onsety) zostaje w pamięci, więc zmiana
suwaka, siatki czy stroju przelicza tabulaturę w ułamku sekundy, a zmiana
silnika liczy od nowa tylko tor wysokości — bez ponownej separacji.
"""
import base64
import logging
import os
import shutil
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from . import notation, separation, transcribe as tr

log = logging.getLogger("basslift")

JOB_TTL = 1800   # s — tyle trzymamy wyniki i pliki zadania od ostatniego użycia
MAX_JOBS = 8

# Udział etapów w pasku postępu, zmierzony na M5 Pro dla minuty audio:
# htdemucs 3.5 s, BS-RoFormer 23 s, beaty 1 s, CREPE 22 s
def extract_stages(model: str) -> Dict:
    sep = 0.50 if model in separation.ROFORMER_MODELS else 0.15
    return {"download": (0.0, 0.0), "separation": (0.0, sep), "beats": (sep, sep + 0.03),
            "pitch": (sep + 0.03, 0.97), "notation": (0.97, 1.0)}


PITCH_ONLY_STAGES = {"pitch": (0.00, 0.95), "notation": (0.95, 1.00)}
SEPARATE_STAGES = {"download": (0.0, 0.0), "separation": (0.00, 0.95), "saving": (0.95, 1.00)}


@dataclass
class Session:
    """Dane jednego wgranego pliku, współdzielone przez kolejne transkrypcje."""
    workdir: Path
    sr: int = 0
    analyses: Dict[str, tr.Analysis] = field(default_factory=dict)  # silnik -> analiza
    fallback: bool = False


@dataclass
class Job:
    id: str
    kind: str                      # extract | separate
    session: Session
    params: Dict
    state: str = "queued"          # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    error: Optional[str] = None
    result: Optional[Dict] = None
    touched: float = field(default_factory=time.time)

    def public(self) -> Dict:
        return {"job_id": self.id, "kind": self.kind, "state": self.state, "stage": self.stage,
                "progress": round(self.progress, 3), "error": self.error, "result": self.result}


class JobStore:
    def __init__(self, cache_dir: Path, busy: Callable = nullcontext):
        self.cache_dir = cache_dir
        self.busy = busy                    # presence.job — serwer nie zgaśnie w trakcie
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._heavy = threading.Lock()      # jedno ciężkie zadanie naraz: RAM i GPU

    # ── rejestr ──────────────────────────────────
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.touched = time.time()
            return job

    def in_use(self, workdir_name: str) -> bool:
        """Czy katalog pliku (jobs/<nazwa>) należy do żywego zadania — bez odświeżania ich czasu.
        Po nazwie, bo /var i /private/var na macOS to ta sama ścieżka zapisana inaczej."""
        with self._lock:
            return any(j.session.workdir.name == workdir_name for j in self._jobs.values())

    def _add(self, job: Job):
        with self._lock:
            self._expire()
            self._jobs[job.id] = job

    def _expire(self):
        """Usuwa zadania nieużywane dłużej niż JOB_TTL i najstarsze ponad MAX_JOBS
        (trwających nie rusza). Katalog pliku znika, gdy nie korzysta z niego już żadne zadanie."""
        now = time.time()
        idle = sorted((j for j in self._jobs.values() if j.state in ("done", "error")),
                      key=lambda j: j.touched)
        excess = max(0, len(self._jobs) + 1 - MAX_JOBS)
        drop = [j for i, j in enumerate(idle) if i < excess or now - j.touched > JOB_TTL]
        for j in drop:
            del self._jobs[j.id]
        used = {j.session.workdir for j in self._jobs.values()}
        for workdir in {j.session.workdir for j in drop} - used:
            shutil.rmtree(workdir, ignore_errors=True)

    # ── uruchamianie ─────────────────────────────
    def submit(self, kind: str, upload: Path, params: Dict, wait: bool = False) -> Job:
        """Nowe zadanie dla wgranego pliku (plik zostaje przeniesiony do katalogu zadania)."""
        job_id = uuid.uuid4().hex[:12]
        workdir = self.cache_dir / "jobs" / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        src = workdir / ("input" + upload.suffix)
        shutil.move(str(upload), src)
        job = Job(job_id, kind, Session(workdir), {**params, "src": src})
        self._add(job)
        self._start(job, self._run_extract if kind == "extract" else self._run_separate, wait)
        return job

    def retranscribe(self, parent: Job, params: Dict, wait: bool = False) -> Job:
        """Nowa transkrypcja tego samego pliku z innymi ustawieniami — bez separacji."""
        job = Job(uuid.uuid4().hex[:12], "extract", parent.session, {**parent.params, **params})
        self._add(job)
        self._start(job, self._run_retranscribe, wait)
        return job

    def _start(self, job: Job, target, wait: bool):
        def run():
            try:
                with self.busy(), self._heavy:
                    job.state = "running"
                    target(job)
                job.state, job.progress, job.stage = "done", 1.0, "done"
            except Exception as e:  # zgłaszamy przeglądarce zamiast gubić wątek
                log.exception("Zadanie %s nie powiodło się", job.id)
                job.state, job.error = "error", str(e) or e.__class__.__name__
        if wait:
            run()
        else:
            threading.Thread(target=run, daemon=True, name=f"job-{job.id}").start()

    @staticmethod
    def _reporter(job: Job, stages: Dict):
        def report(stage: str, frac: float):
            lo, hi = stages.get(stage, (job.progress, job.progress))
            job.stage = stage
            job.progress = max(job.progress, lo + (hi - lo) * min(max(frac, 0.0), 1.0))
        return report

    # ── etapy ────────────────────────────────────
    def _run_extract(self, job: Job):
        report = self._reporter(job, extract_stages(separation.effective_model(job.params["demucs_model"])))
        s, p = job.session, job.params
        s.sr, stems = _separate(p, report)
        bass, rest = separation.two_stems(stems, "bass")
        source, s.fallback = separation.bass_for_transcription(stems, s.sr)
        separation.save_wav(s.workdir / "source.wav", source, s.sr)
        separation.save_wav(self.cache_dir / f"{job.id}_bass.wav", bass, s.sr)
        p["bass_file_id"] = f"{job.id}_bass"
        del stems
        s.analyses[p["engine"]] = tr.analyze(source, s.sr, mix=bass + rest, engine=p["engine"],
                                             slider=p["slider"], on_stage=report)
        job.result = self._render(job, report)

    def _run_retranscribe(self, job: Job):
        s, p = job.session, job.params
        for f in (s.workdir, self.cache_dir / f"{p['bass_file_id']}.wav"):  # używane — nie sprzątaj
            if f.exists():
                os.utime(f)
        if p["engine"] not in s.analyses:  # inny silnik: tylko nowy tor wysokości
            report = self._reporter(job, PITCH_ONLY_STAGES)
            import soundfile as sf
            source, _ = sf.read(str(s.workdir / "source.wav"), dtype="float32")
            known = next(iter(s.analyses.values()))
            s.analyses[p["engine"]] = tr.analyze(source, s.sr, engine=p["engine"], slider=p["slider"],
                                                 grid=known.grid, on_stage=report)
        else:
            report = self._reporter(job, {"notation": (0.0, 1.0)})
        job.result = self._render(job, report)

    def _render(self, job: Job, report) -> Dict:
        """Szybka część: segmentacja, kwantyzacja i zapis — powtarzana przy zmianie ustawień."""
        report("notation", 0.0)
        p = job.params
        a = tr.with_slider(job.session.analyses[p["engine"]], p["slider"])
        result = tr.finish(a, p["quantize"])
        grid = result.grid
        tab = notation.generate_tab(result.notes, grid, p["tuning"], result.quantize_mode,
                                    result.tuning_cents)
        xml = notation.notes_to_musicxml(result.notes, grid)
        log.info("  %s: %d nut, %.1f BPM, %d/4, siatka %s, strój %+.0f ct", job.id, len(result.notes),
                 grid.bpm, grid.beats_per_bar, result.quantize_mode, result.tuning_cents)
        return {
            "job_id": job.id,
            "tab": tab,
            "bpm": grid.bpm,
            "note_count": len(result.notes),
            "duration": round(result.duration, 1),
            "tuning": ",".join(p["tuning"]),
            "time_sig": grid.beats_per_bar,
            "tuning_cents": result.tuning_cents,
            "grid": notation.QUANT_LABELS.get(result.quantize_mode, "1/16"),
            "engine": p["engine"],
            "separation_fallback": job.session.fallback,
            "separation_model": p.get("separation_model"),
            "midi_b64": notation.notes_to_midi_b64(result.notes, grid) if p["export_midi"] else None,
            "musicxml_b64": base64.b64encode(xml.encode("utf-8")).decode() if xml else None,
            "bass_download_id": p["bass_file_id"] if p["export_bass"] else None,
        }

    def _run_separate(self, job: Job):
        report = self._reporter(job, SEPARATE_STAGES)
        s, p = job.session, job.params
        s.sr, stems = _separate(p, report)
        report("saving", 0.0)
        target, accomp = separation.two_stems(stems, p["stem"])
        accomp_label = {"vocals": "instrumental", "bass": "no_bass",
                        "drums": "no_drums", "other": "no_other"}[p["stem"]]
        target_id, accomp_id = f"{job.id}_{p['stem']}", f"{job.id}_{accomp_label}"
        separation.save_wav(self.cache_dir / f"{target_id}.wav", target, s.sr)
        separation.save_wav(self.cache_dir / f"{accomp_id}.wav", accomp, s.sr)
        job.result = {"target_id": target_id, "target_label": p["stem"],
                      "accomp_id": accomp_id, "accomp_label": accomp_label}


def _separate(p: Dict, report):
    model = p["separation_model"] = separation.effective_model(p["demucs_model"])
    report("download" if separation.needs_download(model) else "separation", 0.0)
    try:
        return separation.separate(p["src"], model, progress=lambda f: report("separation", f))
    except Exception as e:  # demucs.api.LoadAudioError, brak modelu itp.
        hint = " (m4a/aac wymaga zainstalowanego ffmpeg)" if p["src"].suffix in {".m4a", ".aac"} else ""
        raise RuntimeError(f"Demucs nie przetworzył pliku{hint}: {e}") from e


ALLOWED_MODELS = {"bs_roformer_sw", "htdemucs", "htdemucs_ft", "mdx_extra"}


def extract_params(demucs_model: str, note_threshold: int, tuning: str, export_midi: str,
                   export_bass: str, quantize: str, transcription_engine: str) -> Dict:
    """Parametry formularza -> słownik zadania (wspólne dla /extract i /api/jobs)."""
    return {
        "demucs_model": demucs_model if demucs_model in ALLOWED_MODELS else separation.DEFAULT_MODEL,
        "slider": int(np.clip(note_threshold, 0, 127)),
        "tuning": [x.strip().upper() for x in tuning.split(",")],
        "export_midi": export_midi == "yes",
        "export_bass": export_bass == "yes",
        "quantize": quantize if quantize in ("auto", "8", "8t", "16", "16t") else "auto",
        "engine": "pyin" if transcription_engine == "pyin" else "crepe",
    }
