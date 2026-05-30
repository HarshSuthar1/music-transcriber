"""
audio_input.py
--------------
Handles all audio file I/O:
  - Format validation (WAV, MP3, FLAC, M4A, OGG)
  - Loading via librosa (handles all formats via ffmpeg)
  - Resampling to target sample rate
  - Mono/stereo normalization
  - Peak normalization to prevent clipping
  - Metadata extraction (duration, BPM estimate, key)
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}

TARGET_SAMPLE_RATE = 44100
TARGET_CHANNELS = 2  # Stereo (Demucs expects stereo)


@dataclass
class AudioInfo:
    """Metadata about a loaded audio file."""
    path: Path
    duration: float           # seconds
    sample_rate: int
    channels: int
    estimated_tempo: float    # BPM
    estimated_key: str        # e.g. "C major"
    peak_amplitude: float
    rms_db: float


@dataclass
class AudioData:
    """Container for loaded audio samples + metadata."""
    samples: np.ndarray   # shape: (channels, samples) float32
    sample_rate: int
    info: AudioInfo


class AudioLoadError(Exception):
    """Raised when audio cannot be loaded or is invalid."""
    pass


def validate_audio_file(path: str | Path) -> Path:
    """
    Check that a file exists and has a supported extension.

    Args:
        path: Path to the audio file.

    Returns:
        Resolved Path object.

    Raises:
        AudioLoadError: If the file does not exist or is unsupported.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise AudioLoadError(f"File not found: {path}")

    if not path.is_file():
        raise AudioLoadError(f"Not a file: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AudioLoadError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Check file is not empty
    if path.stat().st_size == 0:
        raise AudioLoadError(f"File is empty: {path}")

    return path


def load_audio(
    path: str | Path,
    target_sr: int = TARGET_SAMPLE_RATE,
    mono: bool = False,
    normalize: bool = True,
    max_duration: Optional[float] = None,
    offset: float = 0.0,
) -> AudioData:
    """
    Load an audio file and prepare it for the transcription pipeline.

    Args:
        path:         Path to audio file.
        target_sr:    Target sample rate (default 44100 Hz).
        mono:         If True, mix down to mono. Default False (keep stereo for Demucs).
        normalize:    Peak-normalize to [-1, 1] range.
        max_duration: Optional maximum duration in seconds to load.
        offset:       Start offset in seconds.

    Returns:
        AudioData with samples, sample rate, and metadata.

    Raises:
        AudioLoadError: On any loading failure.
    """
    path = validate_audio_file(path)
    logger.info(f"Loading audio: {path}")

    try:
        # librosa loads as float32, mono by default unless mono=False
        # We pass mono=False and handle channel conversion ourselves
        y, sr = librosa.load(
            str(path),
            sr=target_sr,
            mono=mono,
            offset=offset,
            duration=max_duration,
            res_type="kaiser_best",
        )
    except Exception as e:
        raise AudioLoadError(f"Failed to load '{path}': {e}") from e

    # Ensure 2D: (channels, samples)
    if y.ndim == 1:
        # mono → duplicate to stereo so Demucs is happy
        y = np.stack([y, y], axis=0)
    elif y.ndim == 2 and y.shape[0] > y.shape[1]:
        # (samples, channels) → transpose
        y = y.T

    # Handle more than 2 channels → downmix to stereo
    if y.shape[0] > 2:
        logger.warning(f"Audio has {y.shape[0]} channels; downmixing to stereo.")
        y = y[:2]

    # Force mono if requested
    if mono:
        y = y.mean(axis=0, keepdims=True)

    # Peak normalization
    if normalize:
        y = _peak_normalize(y)

    duration = y.shape[1] / target_sr
    logger.info(f"Loaded {duration:.1f}s audio @ {target_sr} Hz, shape {y.shape}")

    info = _extract_metadata(path, y, target_sr)
    return AudioData(samples=y.astype(np.float32), sample_rate=target_sr, info=info)


def save_wav(audio: AudioData, output_path: str | Path) -> Path:
    """
    Save AudioData to a WAV file.

    Args:
        audio:       AudioData to save.
        output_path: Destination path.

    Returns:
        Path to saved WAV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # soundfile expects (samples, channels)
    data = audio.samples.T if audio.samples.ndim == 2 else audio.samples
    sf.write(str(output_path), data, audio.sample_rate, subtype="PCM_16")
    logger.debug(f"Saved WAV: {output_path}")
    return output_path


def save_stem_wav(
    samples: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
    normalize: bool = True,
) -> Path:
    """
    Save a raw numpy array (stem) to WAV.

    Args:
        samples:     (channels, samples) float32.
        sample_rate: Sample rate.
        output_path: Destination path.
        normalize:   Peak-normalize before saving.

    Returns:
        Path to saved WAV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        samples = _peak_normalize(samples)

    data = samples.T if samples.ndim == 2 else samples
    sf.write(str(output_path), data, sample_rate, subtype="PCM_16")
    logger.debug(f"Saved stem WAV: {output_path}")
    return output_path


# ── Internal helpers ──────────────────────────────────────────────────────────

def _peak_normalize(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize so the peak absolute value equals `target_peak`."""
    peak = np.abs(y).max()
    if peak < 1e-8:
        logger.warning("Audio appears silent (peak < 1e-8).")
        return y
    return y * (target_peak / peak)


def _rms_db(y: np.ndarray) -> float:
    """Compute RMS level in dBFS."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-10:
        return -120.0
    return float(20 * np.log10(rms))


def _extract_metadata(path: Path, y: np.ndarray, sr: int) -> AudioInfo:
    """
    Extract tempo, key, and other metadata from the audio.

    Args:
        path: Source file path.
        y:    Audio samples (channels, samples).
        sr:   Sample rate.

    Returns:
        AudioInfo dataclass.
    """
    mono = y.mean(axis=0) if y.ndim == 2 else y
    duration = y.shape[-1] / sr

    # Tempo estimation
    try:
        tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
        estimated_tempo = float(tempo)
    except Exception:
        estimated_tempo = 120.0

    # Key estimation using chroma
    try:
        chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
        chroma_avg = chroma.mean(axis=1)
        pitch_class = int(np.argmax(chroma_avg))
        pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        estimated_key = f"{pitch_names[pitch_class]} major"
    except Exception:
        estimated_key = "Unknown"

    return AudioInfo(
        path=path,
        duration=duration,
        sample_rate=sr,
        channels=y.shape[0] if y.ndim == 2 else 1,
        estimated_tempo=estimated_tempo,
        estimated_key=estimated_key,
        peak_amplitude=float(np.abs(y).max()),
        rms_db=_rms_db(mono),
    )


def trim_silence(
    audio: AudioData,
    top_db: float = 40.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> AudioData:
    """
    Trim leading and trailing silence.

    Args:
        audio:        AudioData to trim.
        top_db:       Threshold below reference (dB) considered silence.
        frame_length: Frame length for STFT.
        hop_length:   Hop size.

    Returns:
        Trimmed AudioData.
    """
    mono = audio.samples.mean(axis=0)
    _, intervals = librosa.effects.trim(
        mono, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    start, end = intervals[0], intervals[1]
    trimmed = audio.samples[:, start:end]
    logger.debug(f"Trimmed silence: {start/audio.sample_rate:.2f}s – {end/audio.sample_rate:.2f}s")

    new_info = AudioInfo(
        path=audio.info.path,
        duration=trimmed.shape[1] / audio.sample_rate,
        sample_rate=audio.sample_rate,
        channels=audio.info.channels,
        estimated_tempo=audio.info.estimated_tempo,
        estimated_key=audio.info.estimated_key,
        peak_amplitude=float(np.abs(trimmed).max()),
        rms_db=_rms_db(trimmed.mean(axis=0)),
    )
    return AudioData(samples=trimmed, sample_rate=audio.sample_rate, info=new_info)