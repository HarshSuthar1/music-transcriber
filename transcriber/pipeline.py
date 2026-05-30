"""
pipeline.py
===========
Orchestrates the full transcription pipeline:

    AudioInput → Separator → AMT → PostProcessor → Output Writers

Usage
-----
    from transcriber import TranscriptionPipeline

    pipeline = TranscriptionPipeline(output_dir="out/", formats=["midi", "xml"])
    result = pipeline.run("song.mp3")
    print(result.output_files)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TranscriptionResult:
    """Holds all artefacts produced by a pipeline run."""
    input_path: Path
    output_dir: Path
    stems: Dict[str, object] = field(default_factory=dict)           # stem_name → AudioData
    raw_transcriptions: Dict[str, object] = field(default_factory=dict)   # stem_name → StemTranscription
    processed_transcriptions: Dict[str, object] = field(default_factory=dict)
    output_files: List[Path] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    tempo: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)

    def summary(self) -> str:
        lines = [
            f"Input : {self.input_path.name}",
            f"Stems : {', '.join(self.stems.keys()) or 'none'}",
            f"Files : {len(self.output_files)} written",
            f"Time  : {self.elapsed_seconds:.1f}s",
        ]
        for p in self.output_files:
            lines.append(f"  → {p}")
        return "\n".join(lines)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class TranscriptionPipeline:
    """
    End-to-end music transcription pipeline.

    Parameters
    ----------
    output_dir : str | Path
        Directory where all output files are written.
    formats : list of str
        Any combination of ``"midi"``, ``"gp"`` / ``"gp5"``, ``"xml"`` / ``"mxl"``.
        Defaults to ``["midi", "xml"]``.
    stems : list of str | None
        Which Demucs stems to transcribe. ``None`` → all four
        (``vocals``, ``drums``, ``bass``, ``other``).
    demucs_model : str
        Demucs model name. ``"htdemucs"`` (default) separates into
        vocals / drums / bass / other.  ``"htdemucs_6s"`` adds guitar & piano.
    device : str
        ``"cpu"`` or ``"cuda"``.  Auto-detected when ``None``.
    tempo : float | None
        Override tempo (BPM). ``None`` → estimated from audio.
    time_signature : tuple[int, int]
        Numerator / denominator, e.g. ``(4, 4)``.
    min_note_duration : float
        Minimum note duration in seconds; shorter notes are filtered out.
    quantize_grid : float
        Quantisation grid in beats (e.g. 0.25 = 16th notes at the given tempo).
    """

    def __init__(
        self,
        output_dir: str | Path = "output",
        formats: Optional[List[str]] = None,
        stems: Optional[List[str]] = None,
        demucs_model: str = "htdemucs",
        device: Optional[str] = None,
        tempo: Optional[float] = None,
        time_signature: Tuple[int, int] = (4, 4),
        min_note_duration: float = 0.05,
        quantize_grid: float = 0.25,
    ):
        self.output_dir = Path(output_dir)
        self.formats = [f.lower() for f in (formats or ["midi", "xml"])]
        self.stems = stems  # None → Separator decides
        self.demucs_model = demucs_model
        self.device = device
        self.tempo_override = tempo
        self.time_signature = time_signature
        self.min_note_duration = min_note_duration
        self.quantize_grid = quantize_grid

    # ------------------------------------------------------------------
    def run(self, audio_path: str | Path) -> TranscriptionResult:
        """Execute the full pipeline and return a :class:`TranscriptionResult`."""
        audio_path = Path(audio_path).expanduser().resolve()
        t0 = time.perf_counter()

        result = TranscriptionResult(
            input_path=audio_path,
            output_dir=self.output_dir,
            time_signature=self.time_signature,
        )

        # 1 ── Load audio ────────────────────────────────────────────────
        logger.info("═══ Stage 1/5 · Audio loading ═══")
        audio_data = self._load_audio(audio_path)

        # 2 ── Estimate tempo ────────────────────────────────────────────
        logger.info("═══ Stage 2/5 · Tempo estimation ═══")
        tempo = self._estimate_tempo(audio_data)
        result.tempo = tempo
        logger.info("Tempo: %.1f BPM", tempo)

        # 3 ── Source separation ─────────────────────────────────────────
        logger.info("═══ Stage 3/5 · Source separation (%s) ═══", self.demucs_model)
        stems = self._separate(audio_data)
        result.stems = stems

        # 4 ── AMT per stem ──────────────────────────────────────────────
        logger.info("═══ Stage 4/5 · Automatic music transcription ═══")
        raw_transcriptions: Dict[str, object] = {}
        for stem_name, stem_audio in stems.items():
            if self.stems and stem_name not in self.stems:
                continue
            logger.info("  Transcribing stem: %s", stem_name)
            raw_transcriptions[stem_name] = self._transcribe_stem(
                stem_name, stem_audio, tempo
            )
        result.raw_transcriptions = raw_transcriptions

        # 5 ── Post-process ──────────────────────────────────────────────
        logger.info("═══ Stage 5/5 · Post-processing ═══")
        processed = self._postprocess(raw_transcriptions, tempo)
        result.processed_transcriptions = processed

        # 6 ── Write outputs ─────────────────────────────────────────────
        logger.info("═══ Writing outputs (%s) ═══", ", ".join(self.formats))
        stem_name = audio_path.stem
        output_files = self._write_outputs(processed, stem_name, tempo)
        result.output_files = output_files

        result.elapsed_seconds = time.perf_counter() - t0
        logger.info("Pipeline complete in %.1fs", result.elapsed_seconds)
        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _load_audio(self, path: Path):
        from .audio_input import AudioInput
        loader = AudioInput(target_sr=44100, mono=False, peak_normalise=True)
        return loader.load(path)

    def _estimate_tempo(self, audio_data) -> float:
        if self.tempo_override:
            return float(self.tempo_override)
        try:
            import librosa  # type: ignore
            import numpy as np
            # Use mono mix for tempo estimation
            mono = audio_data.waveform.mean(axis=0)
            tempo, _ = librosa.beat.beat_track(
                y=mono, sr=audio_data.sample_rate
            )
            # librosa may return array; take scalar
            tempo = float(np.atleast_1d(tempo)[0])
            return round(tempo, 1)
        except Exception as e:
            logger.warning("Tempo estimation failed (%s), defaulting to 120 BPM", e)
            return 120.0

    def _separate(self, audio_data) -> Dict[str, object]:
        from .separator import Separator
        sep = Separator(
            model=self.demucs_model,
            device=self.device,
            sample_rate=audio_data.sample_rate,
        )
        return sep.separate(audio_data)

    def _transcribe_stem(self, stem_name: str, stem_audio, tempo: float):
        from .amt import AMT
        amt = AMT(
            sample_rate=stem_audio.sample_rate if hasattr(stem_audio, "sample_rate")
            else 44100
        )
        return amt.transcribe(stem_name, stem_audio, tempo=tempo)

    def _postprocess(
        self,
        raw: Dict[str, object],
        tempo: float,
    ) -> Dict[str, object]:
        from .postprocess import PostProcessor
        pp = PostProcessor(
            tempo=tempo,
            time_signature=self.time_signature,
            min_note_duration=self.min_note_duration,
            quantize_grid=self.quantize_grid,
        )
        processed = {}
        for stem_name, transcription in raw.items():
            processed[stem_name] = pp.process(stem_name, transcription)
        return processed

    def _write_outputs(
        self,
        processed: Dict[str, object],
        base_name: str,
        tempo: float,
    ) -> List[Path]:
        from .output.midi_writer import MidiWriter, NoteEvent
        from .output.gp_writer import GuitarProWriter
        from .output.xml_writer import MusicXMLWriter

        self.output_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []

        # Convert processed StemTranscriptions → {stem: [NoteEvent]}
        notes_by_stem = self._to_note_events(processed)

        for fmt in self.formats:
            try:
                if fmt == "midi":
                    writer = MidiWriter(
                        tempo=tempo, time_signature=self.time_signature
                    )
                    p = writer.write(
                        notes_by_stem,
                        self.output_dir / f"{base_name}.mid",
                    )
                    written.append(p)

                elif fmt in ("gp", "gp5"):
                    writer = GuitarProWriter(
                        tempo=tempo, time_signature=self.time_signature
                    )
                    p = writer.write(
                        notes_by_stem,
                        self.output_dir / f"{base_name}.gp5",
                    )
                    written.append(p)

                elif fmt in ("xml", "musicxml"):
                    writer = MusicXMLWriter(
                        tempo=tempo,
                        time_signature=self.time_signature,
                        title=base_name,
                    )
                    p = writer.write(
                        notes_by_stem,
                        self.output_dir / f"{base_name}.xml",
                    )
                    written.append(p)

                elif fmt == "mxl":
                    writer = MusicXMLWriter(
                        tempo=tempo,
                        time_signature=self.time_signature,
                        title=base_name,
                    )
                    p = writer.write(
                        notes_by_stem,
                        self.output_dir / f"{base_name}.mxl",
                    )
                    written.append(p)

                else:
                    logger.warning("Unknown output format '%s' — skipped.", fmt)

            except Exception as e:
                logger.error("Failed to write format '%s': %s", fmt, e, exc_info=True)

        return written

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_note_events(
        processed: Dict[str, object],
    ) -> Dict[str, list]:
        """
        Convert PostProcessor output (StemTranscription objects) to the
        flat {stem: [NoteEvent]} dict expected by all three writers.

        Handles both pitched notes and drum hits.
        """
        from .output.midi_writer import NoteEvent

        result: Dict[str, list] = {}

        for stem_name, transcription in processed.items():
            events: List[NoteEvent] = []

            # Pitched notes
            if hasattr(transcription, "notes") and transcription.notes:
                for n in transcription.notes:
                    events.append(
                        NoteEvent(
                            pitch=int(n.pitch),
                            start=float(n.start_time),
                            end=float(n.end_time),
                            velocity=int(getattr(n, "velocity", 80)),
                        )
                    )

            # Drum hits (unpitched → map to GM drum pitches)
            if hasattr(transcription, "drum_hits") and transcription.drum_hits:
                for h in transcription.drum_hits:
                    events.append(
                        NoteEvent(
                            pitch=int(getattr(h, "pitch", 38)),  # default: snare
                            start=float(h.time),
                            end=float(h.time) + 0.05,
                            velocity=int(getattr(h, "velocity", 100)),
                        )
                    )

            result[stem_name] = sorted(events, key=lambda e: e.start)

        return result