# Changelog

All notable changes to BassLift will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Ideas under consideration (no commitments):

- Triplet quantization improvements for swing/jazz feel
- Drop tunings UI presets (Drop D, Drop C, BEAD, etc.)
- 5-string bass support (B0 lowest)
- Web UI hosted version (no local backend needed)

### Added

- True one-click start: `BassLift.command` (macOS/Linux) and `BassLift.bat` (Windows) run `run.py`, which on first launch creates `.venv` and installs `requirements.txt` (CUDA PyTorch on NVIDIA machines), reuses a running instance, falls back to a free port if 8000 is taken, and opens the browser once the server is actually up
- Auto-shutdown: the page sends heartbeats (`POST /api/heartbeat`, `POST /api/bye` on tab close); `run.py` stops the server when no BassLift tab is open and no job is running
- `run.py --no-browser`
- Beat and downbeat tracking with `beat_this` on the full mix: bar lines follow the song, tempo changes don't drift the grid, meter from downbeats (half-bar, half-time and shuffle-in-triplets corrections)
- Quantization to subdivisions of the detected beats; new default grid **Auto** (1/16, or 1/8T for clearly swung songs); `grid` field in the `/extract` response
- Onset-anchored note segmentation: repeated notes split at bass onsets, note starts pulled to the onset (pitch trackers react 50–140 ms late)
- Whole-recording tuning estimation (A≠440), shown in the UI when it is ≥ 10 cents; `tuning_cents` in the response
- Tab fingering optimised over the whole phrase (Viterbi) instead of note by note; tab wrapped into systems; correct octaves for drop D/C and B-E-A-D tunings
- MIDI tempo map (bar by bar) and time signature; MusicXML ties across bar lines, dotted and triplet values
- Fallback when Demucs "loses" the bass (bass < 15 % of mix energy): transcribe the low band of bass + other
- `basslift/` package (`separation`, `rhythm`, `transcribe`, `notation`); `server.py` is HTTP only
- Background jobs with real progress: `POST /api/jobs`, `GET /api/jobs/{id}` (stage + fraction from Demucs chunks and CREPE batches); the UI polls instead of waiting on one long request
- Re-transcription without re-separation: `POST /api/jobs/{id}/retranscribe`; in the UI, changing threshold, grid, tuning or exports updates the tab in < 1 s, switching the engine re-runs only the pitch tracker
- One heavy job at a time (queue) to keep RAM bounded; jobs and files expire 30 min after last use
- Warning pill when the bass-separation fallback was used
- **BS-RoFormer SW** as the default separation model (via `audio-separator`): note F1 on BabySlakh 0.59 → 0.71, no lost-bass songs; htdemucs stays as the fast option and as automatic fallback when `audio-separator` is unavailable. Model (~700 MB) downloaded once to `~/.cache/basslift/models`, with a "downloading" stage in the UI; progress reported per chunk
- Bundled ffmpeg (`imageio-ffmpeg`) put on `PATH` for the separator — no system install; m4a/aac input now works
- `eval/`: synthetic test-set generator and evaluator (mir_eval note F1, beat/downbeat F, meter, tempo) for the synthetic set and BabySlakh, with a baseline comparison mode; `tests/` unit tests

### Changed

- GUI served by the backend connects automatically — no "Check" click, backend URL field and setup instructions hidden
- `/extract` and `/separate` no longer block the event loop (health checks and heartbeats answer during separation)
- CREPE runs on Apple Silicon (MPS); `demucs>=4.1.0` (MPS by default, reads MP3 without ffmpeg)
- Python 3.10+ required; `server.py` dev entry point binds to 127.0.0.1 instead of 0.0.0.0
- CREPE is the default engine (note F1 0.59 vs 0.41 for the previous pipeline on BabySlakh)
- Demucs runs in-process via `demucs.api` with the model cached between requests (no subprocess, no model reload)
- Onset detection from log-RMS rise instead of spectral flux (catches repeated notes, ~5 ms instead of ~23 ms bias)
- New dependencies: `beat_this`, `audio-separator`, `onnxruntime`, `imageio-ffmpeg`, `audioread`
- Progress-bar stage weights depend on the separation model (BS-RoFormer ≈ half of the time, htdemucs ≈ 15 %)
- `/extract` and `/separate` are now blocking wrappers around the same jobs

### Fixed

- Demucs was launched as `python` from `PATH` instead of the app's own interpreter — failed whenever the app ran from a virtualenv that wasn't activated
- CREPE engine returned zero notes: `fmin=30 Hz` is below CREPE's lowest bin (~31.7 Hz), which made torchcrepe mask every bin (periodicity NaN)
- Transcription step label always said "librosa pyin", even with CREPE selected
- CREPE with `batch_size=2048` needed ~37 MB RAM per frame (~76 GB) and pushed macOS into heavy swap; now 32 (~1.6 GB peak for a full extraction), GPU cache freed after each stage, `BASSLIFT_DEVICE=cpu` override
- Repeated notes were merged into one long note after quantization (8 eighths on E → 1 note)
- Quantization grid was anchored at 0 s with a single BPM, so every note after a non-zero start or a slightly wrong tempo landed in the wrong place in the bar
- BPM was limited by librosa's tempo resolution (e.g. 117/123 instead of 120) and differed between engines

## [0.3.0] - 2026-05-25

Major update: neural transcription engine (CREPE) alongside librosa pyin, one-click launcher, light/dark theme switcher, expanded metronome (timbres + meter + multi-accents), and a UX pass on the settings panel.

### Added

**Transcription — CREPE engine**

- New neural pitch tracker as a selectable alternative to librosa pyin: `torchcrepe` `full` model on 16 kHz resampled bass audio
- Engine selector dropdown in section "02 — Settings" (librosa pyin / CREPE neural), persisted to `localStorage`
- Backend: `transcription_engine` form field on `POST /extract`; new `transcribe_bass_crepe()` function reuses every existing post-processing helper (`segment_notes`, `quantize_to_grid`, `merge_close_notes`, `filter_notes`, BPM / onset / time-sig detection) — only the f0 source changes
- Adaptive periodicity threshold (mapped from the existing 0–127 confidence slider), median/mean smoothing per torchcrepe recommendations, lighter octave-error correction (CREPE rarely octave-halves)
- GPU used automatically when CUDA is available
- Lazy import with a clear HTTP 500 message if `torchcrepe` is not installed — pyin keeps working
- ~150 MB CREPE model downloaded on first use
- New dependency: `torchcrepe>=0.0.23`

**One-click launcher**

- `run.py` — Python launcher that starts uvicorn on `127.0.0.1:8000` and opens the browser after the port binds
- `BassLift.bat` — Windows double-click launcher (`cd /d "%~dp0"` + `python run.py` + `pause` so errors are visible)
- FastAPI now serves the GUI directly: `GET /` returns `web_gui.html`, `logo/` mounted as static files — no more `file://` opens, no CORS friction
- Frontend auto-points `#backendUrl` at `window.location.origin` when loaded over http(s), so the same HTML works on any host/port

**Theme switcher (light theme)**

- New light theme: cream background (`#f5efe1`), navy accent (`#1a2a52`), navy primary buttons with cream text
- Dark/light toggle next to the language switcher in the header (`●` / `○`), persisted to `localStorage('basslift_theme')`
- `:root[data-theme="light"]` block overrides theme tokens; dark stays the default (no regression for existing users)
- 6 new CSS tokens introduced to absorb previously hardcoded colors: `--accent-soft`, `--accent-glow`, `--accent-glow-strong`, `--red-soft`, `--red-border`, `--btn-text-on-accent`

**Metronome — timbres, meter, accents**

- Three selectable timbres from a dropdown:
  - "Electronic" — existing 880→440 Hz sine sweep
  - "Classic mechanical" — band-pass-filtered noise burst (dry wood-tick)
  - "Subtle" — soft 600 Hz sine with 5 ms attack
- Time signature dropdown: 2/4, 3/4, 4/4, 5/4, 6/8, 7/8 (4/4 default)
- Multi-accent selector — one toggle button per beat in the bar; accent = same timbre an octave up, ~1.6× louder
- Stronger beat-indicator flash on accented beats (`.beat-accent` with wider glow)
- New i18n keys (PL + EN): `metro_timbre`, `metro_meter`, `metro_accents`, `timbre_beep` / `timbre_mechanical` / `timbre_soft`

**Settings panel — collapsible**

- Section "02 — Settings" is now wrapped in `.settings-collapse` with a clickable header (arrow `▼`/`▲`), mirroring the existing instructions pattern
- Collapsed by default; expansion state persisted to `localStorage('basslift_settings_open')`

### Changed

- `server.py`: `app` and `VERSION` bumped to 0.3.0
- `POST /extract` accepts `transcription_engine` ("pyin" | "crepe"); logs the engine label in step 2/3
- `requirements.txt`: added `torchcrepe>=0.0.23`
- `README.md`: "How it works" now documents both pitch engines and the CREPE model download
- All `<audio>` elements (`inputAudio`, `outputAudioBass`, `outputAudioTarget`, `outputAudioAccomp`) are tracked centrally for stop/exclusivity
- Hardcoded color literals (6 occurrences across upload area, status dot, Run button text, error box, beat indicator) replaced with CSS variables — no visual change for the dark theme

### Fixed

- **Audio kept playing after tab switch / new extraction** — players were hidden via class removal but the underlying `<audio>` was never paused. Introduced `stopAllAudio()` and called it from: Run button handler, `applyMode()` (tab switch), and `fileRemove` handler
- **Two players could play simultaneously** — added `play` listeners on every `<audio>` that pause all others on start, so a new player always stops the previous one

[Unreleased]: https://github.com/winisza/BassLift/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/winisza/BassLift/compare/v0.2.0...v0.3.0

## [0.2.0] - 2026-05-18

Major update: built-in audio players, metronome, transcription quality improvements, and MusicXML export.

### Added

**Audio Players**

- Built-in input audio player — appears after file upload, allows playback of the original track
- Three output audio players (Bass, Target, Accompaniment) — appear after extraction with playable separated stems
- Dark-theme styling with inverted color filter for native `<audio>` controls

**Metronome**

- Standalone "play along" metronome section, always visible (not gated behind extraction)
- BPM input field with 30–300 clamped range, defaults to 120
- Start/Stop toggle using Web Audio API oscillator (880→440 Hz sine wave, 60 ms decay)
- Visual beat indicator circle with accent glow flash on each beat
- Tap Tempo button — averages last 8 taps with 2-second auto-reset timeout
- Auto-fills BPM from extraction results when available
- Lookahead scheduler (Chris Wilson technique) for drift-free timing

**Transcription Quality — Pitch**

- Median filter on MIDI pitches — smooths 1–2 frame outliers using rolling median
- Octave error correction — detects and fixes ±12 semitone jumps in short runs
- Adaptive confidence threshold — per-frame threshold using rolling standard deviation of f0
- Post-quantization note merge — merges consecutive same-pitch notes separated by less than one grid duration

**Transcription Quality — Rhythm**

- Multi-method BPM detection — cross-validates `librosa.feature.tempo()` with multiple start hints against `librosa.beat.beat_track()`
- Onset-assisted note boundaries — forces note splits at percussive attacks using `librosa.onset.onset_detect`
- Time signature detection (3/4 vs 4/4) — compares downbeat strength ratios across beat groupings
- Onset-aware quantization — prefers onset times over raw pYIN times when within half a grid step

**MusicXML Export**

- One-click MusicXML export for MuseScore compatibility
- Full bass clef notation with correct pitches, tempo marking, and time signature
- Uses stdlib `xml.etree.ElementTree` — no new dependencies
- Greedy rest decomposition into valid notation durations (whole, half, quarter, eighth, 16th)
- Download button appears after extraction (PL: "Pobierz MusicXML" / EN: "Download MusicXML")

**i18n**

- New translation keys (PL + EN): metronome labels, audio player labels, MusicXML button, time signature meta pill

### Changed

- `transcribe_bass()` now returns `(notes, bpm, duration, time_sig)` (was 3 values)
- `segment_notes()` accepts optional `onset_times` for onset-assisted boundaries
- `quantize_to_grid()` accepts optional `onset_times` for onset-aware snapping
- `generate_tab()` accepts `time_sig`; bar duration and column count adapt to time signature; footer includes metrum
- `/extract` response now includes `time_sig` and `musicxml_b64` fields
- Time signature displayed as meta pill in extraction results UI
- Version bumped to 0.2.0

### Fixed

- TDZ crash in `applyLang()` — `typeof` check on `let`-declared variable threw during script init; replaced with DOM class-based check
- MusicXML button visibility — `mode-bass-only` class caused `applyMode()` to override initial hidden state; button visibility now managed solely by extraction logic

### Known limitations resolved

- ~~Time signature is not detected — output assumes 4/4~~ → auto-detected (3/4 or 4/4)
- ~~MusicXML export~~ → implemented with one-click download

## [0.1.0] - 2026-05-12

First public release. Local audio tool with two operating modes.

### Added

**Bass → Tablature mode**

- Bass line extraction using Demucs (`htdemucs`, `htdemucs_ft`, `mdx_extra`)
- Monophonic pitch detection via librosa `pyin` (probabilistic YIN)
- ASCII tablature generation for standard 4-string bass tuning (EADG)
- Custom tuning support (drop tunings, alternate tunings)
- Confidence threshold slider (0–127) to balance recall vs. precision
- Quantization grid options: 1/8, 1/8T, 1/16, 1/16T (eighths, sixteenths, triplets)
- MIDI export (`.mid`) using `midiutil`
- Isolated bass stem download (`.wav`)
- Frequency detection range: 30 Hz – 262 Hz (E1 – C4, covers up to 17th fret on G string)

**Vocals + Instrumental mode**

- Source separation for vocals, drums, bass, or other stems
- Both target stem and complement (e.g. instrumental) available as WAV
- Cached download endpoint with automatic 10-minute cleanup

**Frontend**

- Single-file HTML/CSS/JS interface, no build step required
- Polish and English translations with persistent language preference
- Mode switcher (Bass → Tab / Vocals + Instrumental)
- Obsidian/cobalt color scheme with muted copper accent
- Live progress indicator with per-step status
- Backend connection check with version display
- Boldonse logo and brand assets (PNG + SVG)

**Backend API**

- `GET /health` — version and status
- `POST /extract` — bass transcription pipeline
- `POST /separate` — generic stem separation
- `GET /download/{id}` — fetch cached stem file
- CORS enabled for local frontend usage

**Project**

- MIT license
- Documentation: README, GitHub setup guide
- `.gitignore` configured for macOS, Windows, Python, and project-specific artifacts
- GPU acceleration documentation (NVIDIA CUDA / Apple Silicon MPS)

### Known limitations

- Pitch detection is monophonic — chords on bass are not transcribed correctly
- Techniques (slap, hammer-on, pull-off, slide, ghost notes) are not detected
- BPM detection assumes steady tempo; free-time playing produces erratic results
- Time signature is not detected — output assumes 4/4
- First Demucs run downloads ~300 MB of model weights

[0.2.0]: https://github.com/winisza/BassLift/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/winisza/BassLift/releases/tag/v0.1.0
