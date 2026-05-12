# Changelog

All notable changes to BassLift will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Ideas under consideration (no commitments):

- Time signature detection (auto or manual selector)
- Triplet quantization improvements for swing/jazz feel
- Drop tunings UI presets (Drop D, Drop C, BEAD, etc.)
- 5-string bass support (B0 lowest)
- MusicXML export for use in notation software
- Web UI hosted version (no local backend needed)

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

[Unreleased]: https://github.com/winisza/BassLift/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/winisza/BassLift/releases/tag/v0.1.0
