# BassLift

<p align="center">
  <img src="logo/basslift-logo-dark.png" alt="BassLift" width="500"/>
</p>

Local audio tool that does two things:

1. **Bass → Tablature** — extracts the bass line from a song, transcribes it to notes, and generates a 4-string bass tab (with optional MIDI, MusicXML export and isolated bass WAV).
2. **Vocals + Instrumental split** — separates a song into two stems (vocals/drums/bass/other + the rest) and lets you download both as WAV.
3. **Metronome** — built-in play-along metronome with tap tempo, auto-filled BPM from extraction.

Everything runs locally on your machine. No audio is ever uploaded to a third-party server.

> **Experimental** — this is a personal project in active development. Expect rough edges, breaking changes, and quirky behavior on edge cases. Feedback and issues welcome.
>
> **Best on modern GPU or Apple Silicon.** Demucs separation is the heaviest step — a 3-minute song takes ~5-30 seconds on an NVIDIA GPU (CUDA) or Apple Silicon (M1/M2/M3/M4 via MPS), versus 2-4 minutes on CPU. See [GPU acceleration](#gpu-acceleration-strongly-recommended) below.

![Status](https://img.shields.io/badge/status-experimental-orange)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## How it works

1. **Source separation:** [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs`), run inside the server process with the model kept in memory. If Demucs "loses" the bass (puts it into *other*, which happened on 3 of 19 test songs), BassLift transcribes the low band of bass + other instead.
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
| [BabySlakh](https://zenodo.org/records/4603870) (19 songs, 60 s each), Demucs-separated bass | 0.41 | **0.59** |
| BabySlakh with the clean bass stem (upper bound without separation errors) | — | 0.76 |
| Meter correct on BabySlakh | 79 % | **100 %** |
| Tempo error on BabySlakh | 30 % | **2.5 %** |

¹ v0.3.0 code with the CREPE `fmin` fix applied — without it the CREPE engine returned no notes at all.

The biggest remaining loss is separation (0.76 → 0.59). To reproduce:

```bash
.venv/bin/pip install -r eval/requirements.txt
.venv/bin/python eval/make_synthetic.py
.venv/bin/python eval/evaluate.py --dataset synthetic   # BabySlakh: download into eval/data/babyslakh_16k
```

Unit tests: `.venv/bin/python -m pytest tests`.

Memory: a full extraction peaks at ~1.6 GB RAM (Demucs + CREPE on Apple Silicon). Set `BASSLIFT_DEVICE=cpu` to force CPU.

## Stack

- **Backend:** FastAPI + Uvicorn
- **Frontend:** single static HTML file (no build step)

## Requirements

- Python 3.10+
- ~4 GB RAM (CPU mode) or ~2 GB VRAM (GPU mode)
- ~500 MB disk for Demucs model weights (downloaded on first run)
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
```

Then **double-click the launcher**:

- **macOS:** `BassLift.command` (if you downloaded a ZIP instead of cloning, right-click → Open the first time)
- **Windows:** `BassLift.bat`
- **Linux / terminal:** `./BassLift.command` or `python run.py`

That's it. The launcher:

1. On first run creates a private `.venv` next to the app and installs everything from `requirements.txt` (a few minutes — PyTorch is big). On NVIDIA machines it installs the CUDA build of PyTorch automatically.
2. Starts the local server and opens BassLift in your browser.
3. Shuts itself down when you close the last BassLift tab (never while a job is still running).

Later starts take about a second. Useful flag: `python run.py --no-browser`. If you already manage your own environment with the dependencies installed, `run.py` uses it instead of creating `.venv`.

Development mode with auto-reload: `uvicorn server:app --reload --port 8000`.

## Modes

### Bass → Tablature

Upload a song, pick options, hit run. Output is an ASCII tab plus optional MIDI / MusicXML / isolated bass WAV. Built-in audio players let you listen to the original track and separated stems directly in the browser.

Tunable parameters:

| Setting | Default | Notes |
|---|---|---|
| Demucs model | `htdemucs` | `htdemucs_ft` is ~4× slower; on BabySlakh +0.01 note F1 on average (big wins on some songs, losses on others) |
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
GET  /health                 → version info
POST /extract                → bass tab pipeline (multipart form); returns tab, MIDI, MusicXML, time signature
POST /separate               → stem split (multipart form)
GET  /download/{file_id}     → fetch a cached stem WAV
GET  /bass/{file_id}         → fetch isolated bass stem
```

See `server.py` for full parameter docs.

## Limitations

- Pitch detection is monophonic — chords on the bass won't be transcribed correctly.
- Slap, pull-off, hammer-on, slides, and ghost notes aren't detected as techniques.
- The meter comes from detected downbeats (x/4). 6/8 and 12/8 feels are written as 4/4 with triplets; odd meters depend on the beat tracker.
- Free-time playing (no steady pulse) will produce odd bar lines.
- Separation is the weakest link: on some mixes Demucs assigns the bass to other instruments; BassLift then falls back to the low band, which is less accurate.
- The detection threshold is the most important knob. Start at 40 and adjust to taste.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- [Demucs](https://github.com/facebookresearch/demucs) by Meta AI Research
- [librosa](https://librosa.org/) for audio analysis
- [FastAPI](https://fastapi.tiangolo.com/) for the backend
