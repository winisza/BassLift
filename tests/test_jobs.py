"""Zadania w tle bez modeli: separacja i analiza podstawione, liczymy wywołania."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from basslift import jobs, rhythm, separation, transcribe as tr  # noqa: E402
from test_pipeline import make_grid, pitch_track  # noqa: E402

SR = 22050


@pytest.fixture
def fake_models(monkeypatch):
    calls = {"separate": 0, "analyze": []}

    def separate(path, model="htdemucs", progress=None):
        calls["separate"] += 1
        if Path(path).suffix == ".m4a":
            raise OSError("cannot decode")
        for f in (0.5, 1.0):
            progress and progress(f)
        n = SR * 3
        t = np.arange(n) / SR
        bass = np.stack([0.5 * np.sin(2 * np.pi * 41.2 * t)] * 2).astype(np.float32)
        quiet = np.zeros_like(bass)
        return SR, {"bass": bass, "drums": quiet, "other": quiet, "vocals": quiet}

    def analyze(bass, sr, mix=None, engine="crepe", slider=40, grid=None, track=None, on_stage=None):
        calls["analyze"].append({"engine": engine, "grid_reused": grid is not None})
        report = on_stage or (lambda *a: None)
        if grid is None:
            report("beats", 0.0)
            grid = make_grid(offset=0.0)
        for f in (0.0, 0.5, 1.0):
            report("pitch", f)
        track = pitch_track([(0.0, 0.4, 28), (0.5, 0.9, 31), (1.0, 1.4, 33)])
        track.engine = engine
        rms_t = np.arange(0, 3, 0.0116)
        return tr.Analysis(grid, track, np.array([0.0, 0.5, 1.0]), np.ones(3), rms_t,
                           np.ones_like(rms_t), 3.0)

    monkeypatch.setattr(separation, "separate", separate)
    monkeypatch.setattr(tr, "analyze", analyze)
    return calls


def params(**kw):
    return {**jobs.extract_params("htdemucs", 40, "E,A,D,G", "yes", "yes", "auto", "crepe"), **kw}


def upload(tmp_path, name="song.wav"):
    f = tmp_path / name
    f.write_bytes(b"x")
    return f


def test_extract_reports_progress_and_caches_analysis(tmp_path, fake_models):
    store = jobs.JobStore(tmp_path)
    seen = []
    orig = jobs.JobStore._reporter

    def spy(job, stages):
        report = orig(job, stages)
        return lambda stage, frac: (report(stage, frac), seen.append(job.progress))
    store._reporter = spy

    job = store.submit("extract", upload(tmp_path), params(), wait=True)
    assert job.state == "done", job.error
    assert job.progress == 1.0 and seen == sorted(seen)            # postęp tylko rośnie
    assert job.result["note_count"] == 3 and job.result["midi_b64"]
    assert (tmp_path / f"{job.result['bass_download_id']}.wav").exists()
    assert "crepe" in job.session.analyses


def test_retranscribe_same_engine_skips_separation_and_analysis(tmp_path, fake_models):
    store = jobs.JobStore(tmp_path)
    first = store.submit("extract", upload(tmp_path), params(), wait=True)
    again = store.retranscribe(first, params(slider=120, quantize="8"), wait=True)
    assert again.state == "done", again.error
    assert fake_models["separate"] == 1 and len(fake_models["analyze"]) == 1
    assert again.result["grid"] == "1/8" and again.result["job_id"] != first.id


def test_retranscribe_other_engine_reuses_beats(tmp_path, fake_models):
    store = jobs.JobStore(tmp_path)
    first = store.submit("extract", upload(tmp_path), params(), wait=True)
    other = store.retranscribe(first, params(engine="pyin"), wait=True)
    assert other.state == "done", other.error
    assert fake_models["separate"] == 1
    assert fake_models["analyze"][-1] == {"engine": "pyin", "grid_reused": True}


def test_separation_error_is_reported_with_hint(tmp_path, fake_models):
    store = jobs.JobStore(tmp_path)
    job = store.submit("extract", upload(tmp_path, "song.m4a"), params(), wait=True)
    assert job.state == "error" and "ffmpeg" in job.error


def test_old_jobs_are_evicted_with_their_files(tmp_path, fake_models, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_JOBS", 2)
    store = jobs.JobStore(tmp_path)
    first = store.submit("separate", upload(tmp_path), params(stem="vocals"), wait=True)
    assert first.state == "done", first.error
    for _ in range(2):
        store.submit("separate", upload(tmp_path), params(stem="vocals"), wait=True)
    assert store.get(first.id) is None and not first.session.workdir.exists()


def test_workdir_stays_in_use_while_a_retranscription_lives(tmp_path, fake_models, monkeypatch):
    store = jobs.JobStore(tmp_path)
    first = store.submit("extract", upload(tmp_path), params(), wait=True)
    child = store.retranscribe(first, params(slider=90), wait=True)
    with store._lock:
        del store._jobs[first.id]            # rodzic wygasł, dziecko wciąż żyje
    assert store.in_use(first.session.workdir.name)
    assert store.get(child.id).session.workdir == first.session.workdir
