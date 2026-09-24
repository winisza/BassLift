# BassLift

<p align="center">
  <img src="logo/basslift-logo-dark.png" alt="BassLift" width="500"/>
</p>

Local audio tool that does two things:

1. **Bass → Tablature** — extracts the bass line from a song, transcribes it to notes, and generates a 4-string bass tab (with optional MIDI, MusicXML export and isolated bass WAV).
2. **Vocals + Instrumental split** — separates a song into two stems (vocals/drums/bass/other + the rest) and lets you download both as WAV.
3. **Metronome** — built-in play-along metronome with tap tempo, auto-filled BPM from extraction.

Everything runs locally on your machine. No audio is ever uploaded to a third-party server.

On a Mac with Apple Silicon it runs as a native app (**BassLift.app**); on Windows and Linux in your browser via a one-click launcher — see [Quick start](#quick-start).

> **Experimental** — this is a personal project in active development. Expect rough edges, breaking changes, and quirky behavior on edge cases. Feedback and issues welcome.
>
> **Best on modern GPU or Apple Silicon.** Demucs separation is the heaviest step — a 3-minute song takes ~5-30 seconds on an NVIDIA GPU (CUDA) or Apple Silicon (M1/M2/M3/M4 via MPS), versus 2-4 minutes on CPU. See [GPU acceleration](#gpu-acceleration-strongly-recommended) below.

![Status](https://img.shields.io/badge/status-experimental-orange)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## How it works

1. **Source separation:** BS-RoFormer SW (6 stems, via [audio-separator](https://github.com/nomadkaraoke/python-audio-separator)) by default, or [Demucs](https://github.com/facebookresearch/demucs) `htdemucs` as the fast option. Models run inside the server process and stay loaded between songs. If separation "loses" the bass (Demucs put it into *other* on 3 of 19 test songs), BassLift transcribes the low band of bass + other instead.
2. **Beats, downbeats, meter:** [beat_this](https://github.com/CPJKU/beat_this) on the full mix. Bar lines follow the real song, tempo changes don't drift the grid, and the meter is read from the downbeats (common tracker mistakes such as half-bars, half-time and shuffle counted in triplets are corrected).
3. **Pitch:** [CREPE](https://github.com/maxrmorrison/torchcrepe) (default, `full` model) or librosa `pyin` (faster, less accurate). The tuning offset of the whole recording is estimated and removed before rounding to notes.
4. **Notes:** stable-pitch segments are split at bass onsets (repeated notes) and their starts are pulled to the onset (the pitch tracker reacts late).
5. **Quantization:** to subdivisions of the *detected beats*, not a fixed 60/BPM grid from 0 s. The *Auto* grid picks 1/16 or 1/8T (shuffle) per song.
6. **Tab:** string/fret chosen for the whole phrase at once (minimal hand movement), drop tunings and B-E-A-D get correct octaves. MIDI export carries a tempo map; MusicXML has ties across bar lines and dotted/triplet values.

Code layout: `server.py` (HTTP), `basslift/` (`separation`, `rhythm`, `transcribe`, `notation`), `eval/` (accuracy measurement), `tests/`.

## Accuracy

Measured with `eval/evaluate.py` (mir_eval note F1: onset within ±50 ms and correct pitch), CREPE engine:

| Data | v0.3.0¹ | now |
|---|---|---|
| Synthetic set (5 songs: offset start, 3/4, accelerando, detuned, funk 16ths), clean bass | 0.57 | **0.88** |
| [BabySlakh](https://zenodo.org/records/4603870) (19 songs, 60 s each), full pipeline | 0.41 | **0.71** |
| Meter correct on BabySlakh | 79 % | **100 %** |
| Tempo error on BabySlakh | 30 % | **2.5 %** |

¹ v0.3.0 code (htdemucs) with the CREPE `fmin` fix applied — without it the CREPE engine returned no notes at all.

Separation model comparison on BabySlakh (same transcription):

| Separation | Note F1 | Time per minute of audio (M5 Pro) |
|---|---|---|
| htdemucs | 0.59 | ~4 s |
| htdemucs_ft | 0.61 | ~15 s |
| hdemucs_mmi | 0.58 | ~5 s |
| **BS-RoFormer SW** (default) | **0.71** | ~23 s |
| clean bass stem (upper bound) | 0.76 | — |

To reproduce:

```bash
.venv/bin/pip install -r eval/requirements.txt
.venv/bin/python eval/make_synthetic.py
.venv/bin/python eval/evaluate.py --dataset synthetic   # BabySlakh: download into eval/data/babyslakh_16k
```

Unit tests: `.venv/bin/python -m pytest tests`.

Memory: a full extraction peaks at ~1.9 GB RAM (BS-RoFormer + CREPE on Apple Silicon). Set `BASSLIFT_DEVICE=cpu` to force CPU for Demucs and CREPE.

## Stack

- **Backend:** FastAPI + Uvicorn
- **Frontend:** single static HTML file (no build step)

## Requirements

- Python 3.10+
- ~4 GB RAM (CPU mode) or ~2 GB VRAM (GPU mode)
- ~800 MB disk for model weights, downloaded on first use: BS-RoFormer SW ~700 MB (in `~/.cache/basslift/models`), Demucs ~80 MB, CREPE, beat_this
- No system ffmpeg needed (a bundled binary from `imageio-ffmpeg` is used); m4a/aac input works
- A modern browser (Chrome, Firefox, Edge, Safari)

### GPU acceleration (strongly recommended)

Demucs is built on PyTorch and automatically uses your GPU if available. Without GPU acceleration, separating a 3-minute song takes 2–4 minutes on CPU; with GPU it takes 5–30 seconds.

**NVIDIA (CUDA)** — install PyTorch with CUDA support *before* installing the rest. Pick the right CUDA version for your driver at <https://pytorch.org/get-started/locally/>. Example for CUDA 12.1:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the other dependencies:

```bash
pip install -r requirements.txt
```

Verify GPU is detected:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**Apple Silicon (M1–M5)** — PyTorch's MPS (Metal Performance Shaders) backend is included by default, and both Demucs (4.1+) and CREPE use it automatically. Nothing to configure. Verify:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

**CPU only** — works everywhere, no extra setup. Just slower.

## Quick start

```bash
git clone https://github.com/winisza/BassLift.git
cd BassLift
```

### macOS (Apple Silicon) — BassLift.app, the default

A self-contained app with a native window: after building it you need no Python, Terminal or browser. macOS 14+.

```bash
scripts/build_macos_app.sh
```

About 2 minutes (Python 3.10+ is needed only for the build tools, ~2 GB disk in `build/`). Then open `dist/BassLift-0.4.0.dmg` (~350 MB) and drag **BassLift** into Applications. Launch it like any other app.

- Models are downloaded on first use (~900 MB) into the same caches the launcher uses (`~/.cache/…`), so both share them.
- Saving tab/MIDI/MusicXML/WAV opens a native "Save as" dialog; copy goes to the system clipboard.
- Logs: `~/Library/Logs/BassLift/basslift.log`.
- The app is signed ad hoc — fine on the Mac that built it. A copy moved to another Mac is blocked by Gatekeeper unless you right-click → Open (Developer ID signing and notarization are not set up).
- The native window also runs from source: `.venv/bin/python -m basslift`.
- Smoke test of a built app (runs an extraction through the UI, saves all exports, checks the clipboard):
  `open -W --env BASSLIFT_SMOKETEST=/tmp/report.json build/basslift/macos/app/BassLift.app`

### Windows, Linux, or without building — launcher in the browser

**Double-click the launcher**:

- **Windows:** `BassLift.bat`
- **macOS:** `BassLift.command` (if you downloaded a ZIP instead of cloning, right-click → Open the first time)
- **Linux / terminal:** `./BassLift.command` or `python run.py`

The launcher:

1. On first run creates a private `.venv` next to the app and installs everything from `requirements.txt` (a few minutes — PyTorch is big). On NVIDIA machines it installs the CUDA build of PyTorch automatically.
2. Starts the local server and opens BassLift in your browser.
3. Shuts itself down when you close the last BassLift tab (never while a job is still running).

Later starts take about a second. Useful flag: `python run.py --no-browser`. If you already manage your own environment with the dependencies installed, `run.py` uses it instead of creating `.venv`.

Development mode with auto-reload: `uvicorn server:app --reload --port 8000`.

## Modes

### Bass → Tablature

Upload a song, pick options, hit run. The progress bar shows the real progress of separation and pitch tracking. Output is an ASCII tab plus optional MIDI / MusicXML / isolated bass WAV. Built-in audio players let you listen to the original track and separated stems directly in the browser.

After a result is shown, changing the threshold, grid, tuning or export options re-computes the tab in well under a second (no new separation); switching the engine re-runs only the pitch tracker.

Tunable parameters:

| Setting | Default | Notes |
|---|---|---|
| Separation model | BS-RoFormer SW | `htdemucs` is ~6× faster but less accurate (note F1 0.59 vs 0.71 on BabySlakh); `htdemucs_ft` ~4× slower than htdemucs for +0.02 |
| Transcription engine | CREPE | `pyin` is faster but much less accurate |
| Detection threshold | 40 (out of 127) | Lower = more notes (incl. ghost notes), higher = only confident notes |
| Tuning | E A D G | 4 strings; drop D, drop C, B-E-A-D etc. get the right octave |
| Quantization | Auto | 1/16, or 1/8T when the song clearly swings; also manual 1/8, 1/8T, 1/16, 1/16T |

### Vocals + Instrumental split

Upload a song, choose what to extract (vocals, drums, bass, or other), hit run. Get two WAV files back.

### Metronome

Always-visible play-along metronome with a BPM input (30–300) and tap tempo. After extraction, the detected BPM auto-fills the metronome. Audio click with visual beat indicator flash.

## API endpoints

If you want to integrate the backend into your own tool:

```
POST /api/jobs                        → start a job (multipart: file, kind=extract|separate, settings) → {job_id}
GET  /api/jobs/{job_id}               → {state: queued|running|done|error, stage, progress 0..1, result}
POST /api/jobs/{job_id}/retranscribe  → new job with other settings (threshold, grid, tuning, engine,
                                        exports) — reuses separation and analysis, no Demucs re-run
POST /extract                         → same as an extract job, answered when finished (blocking)
POST /separate                        → same as a separate job, answered when finished (blocking)
GET  /download/{file_id}              → fetch a cached stem WAV
GET  /bass/{file_id}                  → fetch isolated bass stem
GET  /health                          → version info
```

Jobs and their files are kept for 30 minutes after last use. One heavy job runs at a time (others wait as `queued`), which keeps memory use bounded. See `server.py` and `basslift/jobs.py` for parameters.

## Limitations

- Pitch detection is monophonic — chords on the bass won't be transcribed correctly.
- Slap, pull-off, hammer-on, slides, and ghost notes aren't detected as techniques.
- The meter comes from detected downbeats (x/4). 6/8 and 12/8 feels are written as 4/4 with triplets; odd meters depend on the beat tracker.
- Free-time playing (no steady pulse) will produce odd bar lines.
- With htdemucs, some mixes get the bass assigned to other instruments; BassLift then falls back to the low band, which is less accurate. BS-RoFormer (default) did not show this on the test set.
- The detection threshold is the most important knob. Start at 40 and adjust to taste.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- [Demucs](https://github.com/facebookresearch/demucs) by Meta AI Research
- [librosa](https://librosa.org/) for audio analysis
- [FastAPI](https://fastapi.tiangolo.com/) for the backend
