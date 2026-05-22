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

[Unreleased]: https://github.com/winisza/BassLift/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/winisza/BassLift/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/winisza/BassLift/releases/tag/v0.1.0
