"""Separacja źródeł w procesie serwera (modele ładowane raz i trzymane w pamięci).

Domyślnie BS-RoFormer SW (6 ścieżek, przez audio-separator): na BabySlakh F1 nut
0.715 wobec 0.585 dla htdemucs, bliżej górnej granicy z czystym basem (0.761), i bez
utworów, na których Demucs „gubił” bas. Kosztem ~6× dłuższej separacji i modelu
~700 MB pobieranego przy pierwszym użyciu. Demucs zostaje jako szybka opcja
i jako zapas, gdy audio-separator nie jest zainstalowany.
"""
import logging
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

log = logging.getLogger("basslift")

ROFORMER_MODELS = {"bs_roformer_sw": "BS-Roformer-SW.ckpt"}
DEFAULT_MODEL = "bs_roformer_sw"
MODEL_DIR = Path(os.environ.get("BASSLIFT_MODEL_DIR", Path.home() / ".cache" / "basslift" / "models"))

_separators: Dict[str, object] = {}
_lock = threading.Lock()  # jeden model na GPU naraz — kolejne zadania czekają


def torch_device() -> str:
    """cuda > mps > cpu; BASSLIFT_DEVICE=cpu wymusza CPU (np. przy małej ilości RAM)."""
    import torch
    forced = os.environ.get("BASSLIFT_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free_gpu_memory():
    """Oddaje systemowi pamięć z cache alokatora GPU (na Apple Silicon to wspólny RAM)."""
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ────────────────────────────────────────────────
# Wybór modelu
# ────────────────────────────────────────────────
_roformer_ok: Optional[bool] = None


def roformer_available() -> bool:
    """Pełny import (nie tylko pakietu): audio-separator ma zależności, których nie deklaruje."""
    global _roformer_ok
    if _roformer_ok is None:
        try:
            from audio_separator.separator import Separator  # noqa: F401
            _roformer_ok = True
        except ImportError as e:
            log.warning("  audio-separator nie działa: %s", e)
            _roformer_ok = False
    return _roformer_ok


def effective_model(model: str) -> str:
    """Model, którego naprawdę użyjemy (RoFormer -> htdemucs, gdy brak audio-separator)."""
    if model in ROFORMER_MODELS and not roformer_available():
        log.warning("  audio-separator niedostępny — separacja przez htdemucs "
                    "(pip install audio-separator onnxruntime imageio-ffmpeg)")
        return "htdemucs"
    return model


def needs_download(model: str) -> bool:
    return model in ROFORMER_MODELS and not (MODEL_DIR / ROFORMER_MODELS[model]).exists()


def separate(path: Path, model: str = DEFAULT_MODEL,
             progress: Optional[Callable[[float], None]] = None):
    """Rozdziela plik na ścieżki. Zwraca (samplerate, {"bass"|"drums"|"other"|"vocals": [kanały, próbki]})."""
    model = effective_model(model)
    with _lock:
        try:
            if model in ROFORMER_MODELS:
                return _separate_roformer(Path(path), ROFORMER_MODELS[model], progress)
            return _separate_demucs(Path(path), model, progress)
        finally:
            free_gpu_memory()


# ────────────────────────────────────────────────
# Demucs
# ────────────────────────────────────────────────
def _separate_demucs(path: Path, model: str, progress):
    from demucs.api import Separator
    if model not in _separators:
        log.info("  Ładuję model Demucs %s (%s)", model, torch_device())
        _separators[model] = Separator(model=model, device=torch_device())
    sep = _separators[model]

    def on_chunk(info):
        if progress and info.get("state") == "end":
            done = info["segment_offset"] / max(info["audio_length"], 1)
            bag = info.get("models", 1)
            progress(min(1.0, (info["model_idx_in_bag"] + min(done, 1.0)) / bag))

    sep.update_parameter(callback=on_chunk if progress else None)
    _, stems = sep.separate_audio_file(path)
    return sep.samplerate, {name: wav.cpu().numpy() for name, wav in stems.items()}


# ────────────────────────────────────────────────
# BS-RoFormer (audio-separator)
# ────────────────────────────────────────────────
def _ensure_ffmpeg():
    """audio-separator wymaga `ffmpeg` w PATH — bierzemy statyczną binarkę z imageio-ffmpeg,
    bez instalowania czegokolwiek w systemie. Przy okazji działa odczyt m4a/aac."""
    if shutil.which("ffmpeg"):
        return
    import imageio_ffmpeg
    exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    bin_dir = MODEL_DIR.parent / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    if not link.exists():
        if sys.platform == "win32":
            shutil.copy2(exe, link)  # dowiązania na Windows wymagają uprawnień administratora
        else:
            link.symlink_to(exe)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _roformer(filename: str):
    if filename not in _separators:
        from audio_separator.separator import Separator
        _ensure_ffmpeg()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        out_dir = Path(tempfile.mkdtemp(prefix="basslift_sep_"))
        if not (MODEL_DIR / filename).exists():
            log.info("  Pobieram model %s (~700 MB, tylko za pierwszym razem)", filename)
        sep = Separator(model_file_dir=str(MODEL_DIR), output_dir=str(out_dir),
                        output_format="WAV", log_level=logging.WARNING)
        sep.load_model(model_filename=filename)
        _separators[filename] = (sep, out_dir)
    return _separators[filename]


def _separate_roformer(path: Path, filename: str, progress):
    import soundfile as sf
    sep, out_dir = _roformer(filename)
    patched = _patch_progress(progress)
    try:
        files = sep.separate(str(path))
    finally:
        patched()
    stems, sr = {}, 0
    for f in files:
        f = Path(f) if Path(f).is_absolute() else out_dir / f
        name = f.stem.rsplit("(", 1)[1].split(")")[0].lower()  # input_(bass)_BS-Roformer-SW
        wav, sr = sf.read(str(f), dtype="float32", always_2d=True)
        stems[name] = wav.T
        f.unlink(missing_ok=True)
    # gitara i fortepian to dla nas „inne” — zachowujemy układ 4 ścieżek jak w Demucs
    other = sum(stems.pop(k) for k in ("other", "guitar", "piano") if k in stems)
    return sr, {"bass": stems["bass"], "drums": stems["drums"], "other": other, "vocals": stems["vocals"]}


def _patch_progress(progress):
    """Podpina postęp pod pętlę fragmentów audio-separatora (iteruje przez tqdm).
    To szczegół wewnętrzny biblioteki — gdy się zmieni, tracimy tylko pasek, nie separację."""
    if not progress:
        return lambda: None
    try:
        import audio_separator.separator.architectures.mdxc_separator as mdxc
    except ImportError:
        return lambda: None
    original = mdxc.tqdm

    def counting(iterable, *args, **kwargs):
        items = list(iterable)
        for i, item in enumerate(items):
            yield item
            progress((i + 1) / max(len(items), 1))

    mdxc.tqdm = counting

    def restore():
        mdxc.tqdm = original
    return restore


# ────────────────────────────────────────────────
# Wspólne
# ────────────────────────────────────────────────
def two_stems(stems: Dict[str, np.ndarray], target: str):
    """Ścieżka docelowa + suma pozostałych (odpowiednik `demucs --two-stems`)."""
    rest = sum(w for name, w in stems.items() if name != target)
    return stems[target], rest


LOST_BASS_RATIO = 0.15  # RMS basu / RMS miksu; udane separacje ~0.5, nieudane 0.003–0.08


def bass_for_transcription(stems: Dict[str, np.ndarray], sr: int):
    """Sygnał do transkrypcji basu. Gdy separacja „zgubi” bas (htdemucs wrzucił go do
    `other` w 3/19 utworów Slakh), bierzemy dolnoprzepustową sumę bas+other —
    bez perkusji i wokalu linia basu zwykle jest tam najniższym głosem.
    Zwraca (audio, czy_użyto_awaryjnego_źródła)."""
    from scipy.signal import butter, sosfiltfilt
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) + 1e-9
    bass = stems["bass"]
    if rms(bass) / rms(sum(stems.values())) >= LOST_BASS_RATIO:
        return bass, False
    log.warning("  Separacja prawie nie wydzieliła basu — transkrybuję dolne pasmo bas+other")
    sos = butter(4, 500, fs=sr, output="sos")
    return sosfiltfilt(sos, bass + stems["other"], axis=-1).astype(np.float32), True


def save_wav(path: Path, wav: np.ndarray, sr: int):
    import soundfile as sf
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0.99:  # jak demucs --clip-mode rescale
        wav = wav / (1.01 * peak)
    sf.write(str(path), wav.T, sr, subtype="PCM_16")
