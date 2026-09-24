"""Separacja źródeł Demucs w procesie serwera (model ładowany raz i trzymany w pamięci)."""
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

log = logging.getLogger("basslift")

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


def _separator(model: str):
    from demucs.api import Separator
    if model not in _separators:
        log.info("  Ładuję model Demucs %s (%s)", model, torch_device())
        _separators[model] = Separator(model=model, device=torch_device())
    return _separators[model]


def separate(path: Path, model: str = "htdemucs",
             progress: Optional[Callable[[float], None]] = None):
    """Rozdziela plik na ścieżki. Zwraca (samplerate, {"bass": [kanały, próbki], ...})."""
    with _lock:
        sep = _separator(model)

        def on_chunk(info):
            if progress and info.get("state") == "end":
                done = (info["segment_offset"] + info.get("segment_length", 0)) / max(info["audio_length"], 1)
                bag = info.get("models", 1)
                progress(min(1.0, (info["model_idx_in_bag"] + min(done, 1.0)) / bag))

        sep.update_parameter(callback=on_chunk if progress else None)
        try:
            _, stems = sep.separate_audio_file(Path(path))
            return sep.samplerate, {name: wav.cpu().numpy() for name, wav in stems.items()}
        finally:
            free_gpu_memory()


def two_stems(stems: Dict[str, np.ndarray], target: str):
    """Ścieżka docelowa + suma pozostałych (odpowiednik `demucs --two-stems`)."""
    rest = sum(w for name, w in stems.items() if name != target)
    return stems[target], rest


LOST_BASS_RATIO = 0.15  # RMS basu / RMS miksu; udane separacje ~0.5, nieudane 0.003–0.08


def bass_for_transcription(stems: Dict[str, np.ndarray], sr: int):
    """Sygnał do transkrypcji basu. Gdy Demucs „zgubi” bas (wrzuci go do `other`, co na
    Slakh zdarzyło się w 3/19 utworów), bierzemy dolnoprzepustową sumę bas+other —
    bez perkusji i wokalu linia basu zwykle jest tam najniższym głosem.
    Zwraca (audio, czy_użyto_awaryjnego_źródła)."""
    from scipy.signal import butter, sosfiltfilt
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) + 1e-9
    bass = stems["bass"]
    if rms(bass) / rms(sum(stems.values())) >= LOST_BASS_RATIO:
        return bass, False
    log.warning("  Demucs prawie nie wydzielił basu — transkrybuję dolne pasmo bas+other")
    sos = butter(4, 500, fs=sr, output="sos")
    return sosfiltfilt(sos, bass + stems["other"], axis=-1).astype(np.float32), True


def save_wav(path: Path, wav: np.ndarray, sr: int):
    import soundfile as sf
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0.99:  # jak demucs --clip-mode rescale
        wav = wav / (1.01 * peak)
    sf.write(str(path), wav.T, sr, subtype="PCM_16")
