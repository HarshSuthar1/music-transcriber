"""
separator.py
------------
Wraps Demucs source separation to split a mixed audio track into stems.

Supported models:
  - htdemucs        : 4 stems (drums, bass, vocals, other)
  - htdemucs_6s     : 6 stems (drums, bass, vocals, guitar, piano, other) ← default
  - htdemucs_ft     : Fine-tuned 4-stem model
  - mdx_extra       : MDX-Net 4-stem, higher quality / slower
  - mdx_extra_q     : Quantized MDX-Net (faster inference)

Demucs is called via its Python API (demucs.separate) rather than subprocess
so we can capture stems as in-memory numpy arrays without writing intermediate files.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Canonical stem names per model family
STEMS_4 = ["drums", "bass", "vocals", "other"]
STEMS_6 = ["drums", "bass", "vocals", "guitar", "piano", "other"]

MODEL_STEMS: Dict[str, List[str]] = {
    "htdemucs":    STEMS_4,
    "htdemucs_ft": STEMS_4,
    "htdemucs_6s": STEMS_6,
    "mdx_extra":   STEMS_4,
    "mdx_extra_q": STEMS_4,
}

DEFAULT_MODEL = "htdemucs_6s"


@dataclass
class SeparationResult:
    """Container for separated stems."""
    stems: Dict[str, np.ndarray]   # stem_name → (channels, samples) float32
    sample_rate: int
    model_name: str
    stem_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.stem_names = list(self.stems.keys())

    def get_stem(self, name: str) -> Optional[np.ndarray]:
        """Return stem samples or None if not present."""
        return self.stems.get(name)


class SeparationError(Exception):
    """Raised when source separation fails."""
    pass


class AudioSeparator:
    """
    Wraps Demucs to perform multi-stem source separation.

    Usage:
        separator = AudioSeparator(model="htdemucs_6s", device="cuda")
        result = separator.separate(audio_data)
        drums_wav = result.get_stem("drums")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        segment: Optional[float] = None,
        overlap: float = 0.25,
        jobs: int = 0,
    ):
        """
        Args:
            model:   Demucs model name.
            device:  "cpu", "cuda", "mps", or None (auto-detect).
            segment: Segment length in seconds for chunked processing (None = model default).
                     Reduce this if you get CUDA OOM errors.
            overlap: Overlap between segments [0, 1). Higher = smoother but slower.
            jobs:    Number of parallel jobs (0 = auto).
        """
        self.model_name = model
        self.segment = segment
        self.overlap = overlap
        self.jobs = jobs

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"AudioSeparator: model={model}, device={self.device}")
        self._model = None  # Lazy-loaded

    def _load_model(self):
        """Lazy-load the Demucs model on first use."""
        if self._model is not None:
            return

        try:
            from demucs.pretrained import get_model
            logger.info(f"Loading Demucs model '{self.model_name}'...")
            self._model = get_model(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info("Demucs model loaded.")
        except ImportError as e:
            raise SeparationError(
                "demucs is not installed. Run: pip install demucs"
            ) from e
        except Exception as e:
            raise SeparationError(f"Failed to load Demucs model '{self.model_name}': {e}") from e

    @property
    def stem_names(self) -> List[str]:
        """Return expected stem names for the current model."""
        return MODEL_STEMS.get(self.model_name, STEMS_4)

    def separate(
        self,
        audio_data,   # AudioData from audio_input.py
        instruments: Optional[List[str]] = None,
    ) -> SeparationResult:
        """
        Separate an audio track into stems.

        Args:
            audio_data:   AudioData (from audio_input.load_audio).
            instruments:  Optional list to filter which stems to return.
                          E.g. ["drums", "bass"] skips the rest.

        Returns:
            SeparationResult with per-stem numpy arrays.

        Raises:
            SeparationError: On any Demucs failure.
        """
        self._load_model()

        samples = audio_data.samples          # (channels, samples) float32
        sr = audio_data.sample_rate

        # Resample to Demucs model sample rate if needed
        model_sr = self._model.samplerate
        if sr != model_sr:
            logger.info(f"Resampling from {sr} Hz to {model_sr} Hz for Demucs.")
            import torchaudio
            samples_t = torch.from_numpy(samples)
            samples_t = torchaudio.functional.resample(samples_t, sr, model_sr)
            samples = samples_t.numpy()
            sr = model_sr

        # Demucs expects (batch, channels, samples)
        wav = torch.from_numpy(samples).unsqueeze(0).to(self.device)

        logger.info(f"Running Demucs separation (shape {wav.shape})...")
        try:
            stems_tensor = self._run_demucs(wav)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise SeparationError(
                    "CUDA out of memory. Try --segment to reduce chunk size, "
                    "or use --device cpu."
                ) from e
            raise SeparationError(f"Demucs separation failed: {e}") from e

        # stems_tensor: (batch=1, stems, channels, samples)
        stems_tensor = stems_tensor.squeeze(0).cpu()  # (stems, channels, samples)

        result_stems: Dict[str, np.ndarray] = {}
        for i, name in enumerate(self._model.sources):
            if instruments is not None and name not in instruments:
                continue
            stem_array = stems_tensor[i].numpy().astype(np.float32)
            result_stems[name] = stem_array
            logger.debug(f"  stem '{name}': {stem_array.shape}, peak={np.abs(stem_array).max():.3f}")

        logger.info(f"Separation complete. Stems: {list(result_stems.keys())}")
        return SeparationResult(
            stems=result_stems,
            sample_rate=sr,
            model_name=self.model_name,
        )

    def _run_demucs(self, wav: "torch.Tensor") -> "torch.Tensor":
        """
        Run the Demucs model with optional chunked processing.

        Args:
            wav: (1, channels, samples) tensor.

        Returns:
            (1, num_stems, channels, samples) tensor.
        """
        from demucs.apply import apply_model

        segment = self.segment
        if segment is None:
            # Use model default (usually 7.8s for htdemucs)
            segment = getattr(self._model, "segment", 7.8)

        with torch.no_grad():
            stems = apply_model(
                self._model,
                wav,
                device=self.device,
                shifts=1,
                split=True,
                overlap=self.overlap,
                progress=False,
                num_workers=self.jobs,
                segment=segment,
            )
        return stems

    def separate_to_files(
        self,
        audio_data,
        output_dir: str | Path,
        instruments: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """
        Separate and save stems as WAV files.

        Args:
            audio_data:  AudioData to separate.
            output_dir:  Directory to write stems.
            instruments: Optional instrument filter.

        Returns:
            Dict mapping stem name → WAV path.
        """
        from .audio_input import save_stem_wav

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.separate(audio_data, instruments=instruments)

        paths: Dict[str, Path] = {}
        for stem_name, stem_samples in result.stems.items():
            wav_path = output_dir / f"{stem_name}.wav"
            save_stem_wav(stem_samples, result.sample_rate, wav_path)
            paths[stem_name] = wav_path
            logger.info(f"Saved stem: {wav_path}")

        return paths

    def __repr__(self) -> str:
        return f"AudioSeparator(model={self.model_name!r}, device={self.device!r})"