"""
postprocess.py
--------------
Post-processes raw AMT output before writing to notation formats.

Operations:
  1. Tempo detection / validation
  2. Note quantization to a rhythmic grid
  3. Short-note filtering (remove noise artifacts)
  4. Velocity normalization
  5. Overlapping note removal per pitch
  6. Pitch range clamping per instrument
  7. Drum hit deduplication
  8. Beat alignment for drum hits
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .amt import DrumHit, Note, StemTranscription

logger = logging.getLogger(__name__)


@dataclass
class PostprocessConfig:
    """Configuration for the post-processor."""
    tempo: float = 120.0               # BPM
    time_signature_num: int = 4        # numerator
    time_signature_den: int = 4        # denominator (power of 2)
    quantize: bool = True              # Snap note times to grid
    quantize_divisions: int = 16       # Subdivisions per beat (16 = 16th note grid)
    min_note_duration: float = 0.05    # Seconds; shorter notes removed
    max_note_duration: float = 20.0    # Seconds; cap very long notes
    velocity_floor: int = 10           # Remove notes quieter than this
    velocity_ceil: int = 127
    remove_overlaps: bool = True       # Remove same-pitch overlaps
    pitch_floor: int = 21              # A0  (piano range)
    pitch_ceil: int = 108              # C8  (piano range)
    drum_dedup_window: float = 0.02    # Merge drum hits within this window (seconds)


@dataclass
class ProcessedScore:
    """
    Fully post-processed score ready for output writers.

    Attributes:
        stems:          Dict of processed StemTranscription objects.
        tempo:          Detected or configured BPM.
        time_sig:       (numerator, denominator) time signature.
        total_duration: Duration of the longest stem in seconds.
        ticks_per_beat: MIDI ticks per quarter note (standard 480).
    """
    stems: Dict[str, StemTranscription]
    tempo: float
    time_sig: Tuple[int, int]
    total_duration: float
    ticks_per_beat: int = 480


class PostProcessor:
    """
    Applies quantization and cleanup to AMT transcriptions.
    """

    def __init__(self, config: Optional[PostprocessConfig] = None):
        self.config = config or PostprocessConfig()

    def process(
        self,
        transcriptions: Dict[str, StemTranscription],
        audio_info=None,   # AudioInfo from audio_input.py (optional, used for tempo)
    ) -> ProcessedScore:
        """
        Run the full post-processing pipeline.

        Args:
            transcriptions: Raw AMT output per stem.
            audio_info:     Optional AudioInfo for tempo estimation.

        Returns:
            ProcessedScore ready for output writers.
        """
        cfg = self.config

        # Step 1: Determine tempo
        tempo = cfg.tempo
        if audio_info is not None and audio_info.estimated_tempo > 20:
            tempo = audio_info.estimated_tempo
            logger.info(f"Using detected tempo: {tempo:.1f} BPM")
        else:
            logger.info(f"Using configured tempo: {tempo:.1f} BPM")

        # Step 2: Process each stem
        processed_stems: Dict[str, StemTranscription] = {}
        max_duration = 0.0

        for stem_name, transcription in transcriptions.items():
            if transcription.is_drum:
                processed = self._process_drums(transcription, tempo)
            else:
                processed = self._process_pitched(transcription, tempo)

            processed_stems[stem_name] = processed

            # Track total duration
            stem_dur = self._stem_duration(processed)
            if stem_dur > max_duration:
                max_duration = stem_dur

        return ProcessedScore(
            stems=processed_stems,
            tempo=tempo,
            time_sig=(cfg.time_signature_num, cfg.time_signature_den),
            total_duration=max_duration,
            ticks_per_beat=480,
        )

    # ── Pitched processing ────────────────────────────────────────────────────

    def _process_pitched(
        self,
        transcription: StemTranscription,
        tempo: float,
    ) -> StemTranscription:
        """Apply all processing steps to a pitched stem."""
        cfg = self.config
        notes = list(transcription.notes)

        logger.debug(f"[{transcription.stem_name}] Start: {len(notes)} notes")

        # 1. Velocity filter
        notes = [n for n in notes if n.velocity >= cfg.velocity_floor]
        logger.debug(f"[{transcription.stem_name}] After velocity filter: {len(notes)}")

        # 2. Duration filter
        notes = [
            n for n in notes
            if cfg.min_note_duration <= n.duration <= cfg.max_note_duration
        ]
        logger.debug(f"[{transcription.stem_name}] After duration filter: {len(notes)}")

        # 3. Pitch range clamp
        notes = self._clamp_pitch_range(notes, transcription.stem_name)
        logger.debug(f"[{transcription.stem_name}] After pitch clamp: {len(notes)}")

        # 4. Quantize
        if cfg.quantize:
            notes = self._quantize_notes(notes, tempo, cfg.quantize_divisions)
            logger.debug(f"[{transcription.stem_name}] After quantize: {len(notes)}")

        # 5. Remove overlaps
        if cfg.remove_overlaps:
            notes = self._remove_overlapping_notes(notes)
            logger.debug(f"[{transcription.stem_name}] After overlap removal: {len(notes)}")

        # 6. Clip velocities
        for n in notes:
            n.velocity = int(np.clip(n.velocity, cfg.velocity_floor, cfg.velocity_ceil))

        # 7. Sort by start time
        notes.sort(key=lambda n: (n.start_time, n.pitch))

        logger.info(f"[{transcription.stem_name}] Final note count: {len(notes)}")

        # Return new StemTranscription with processed notes
        result = StemTranscription(
            stem_name=transcription.stem_name,
            notes=notes,
            midi_program=transcription.midi_program,
            midi_channel=transcription.midi_channel,
            is_drum=False,
        )
        return result

    def _process_drums(
        self,
        transcription: StemTranscription,
        tempo: float,
    ) -> StemTranscription:
        """Apply drum-specific processing."""
        cfg = self.config
        hits = list(transcription.drum_hits)

        logger.debug(f"[drums] Start: {len(hits)} hits")

        # 1. Sort by time
        hits.sort(key=lambda h: h.time)

        # 2. Deduplicate close hits (same pitch within window)
        hits = self._dedup_drum_hits(hits, cfg.drum_dedup_window)
        logger.debug(f"[drums] After dedup: {len(hits)}")

        # 3. Quantize to grid
        if cfg.quantize:
            hits = self._quantize_drum_hits(hits, tempo, cfg.quantize_divisions)
            logger.debug(f"[drums] After quantize: {len(hits)}")

        # 4. Velocity clip
        for h in hits:
            h.velocity = int(np.clip(h.velocity, cfg.velocity_floor, cfg.velocity_ceil))

        logger.info(f"[drums] Final hit count: {len(hits)}")

        result = StemTranscription(
            stem_name="drums",
            drum_hits=hits,
            midi_program=0,
            midi_channel=9,
            is_drum=True,
        )
        return result

    # ── Quantization ──────────────────────────────────────────────────────────

    def _quantize_notes(
        self,
        notes: List[Note],
        tempo: float,
        divisions: int,
    ) -> List[Note]:
        """Snap note start/end times to the nearest rhythmic grid point."""
        grid_size = self._grid_size(tempo, divisions)  # seconds per grid step

        quantized = []
        for note in notes:
            q_start = round(note.start_time / grid_size) * grid_size
            q_end = round(note.end_time / grid_size) * grid_size

            # Ensure minimum duration after quantization
            if q_end <= q_start:
                q_end = q_start + grid_size

            quantized.append(Note(
                start_time=max(0.0, q_start),
                end_time=q_end,
                pitch=note.pitch,
                velocity=note.velocity,
                pitch_bends=note.pitch_bends,
            ))
        return quantized

    def _quantize_drum_hits(
        self,
        hits: List[DrumHit],
        tempo: float,
        divisions: int,
    ) -> List[DrumHit]:
        """Snap drum onset times to grid."""
        grid_size = self._grid_size(tempo, divisions)

        quantized = []
        for hit in hits:
            q_time = round(hit.time / grid_size) * grid_size
            quantized.append(DrumHit(
                time=max(0.0, q_time),
                pitch=hit.pitch,
                velocity=hit.velocity,
            ))
        return quantized

    @staticmethod
    def _grid_size(tempo: float, divisions: int) -> float:
        """Seconds per grid step given tempo (BPM) and subdivisions per beat."""
        beat_duration = 60.0 / tempo
        return beat_duration / divisions

    # ── Pitch range ────────────────────────────────────────────────────────────

    # Instrument-specific pitch ranges (MIDI note numbers)
    _PITCH_RANGES: Dict[str, Tuple[int, int]] = {
        "bass":   (28, 67),   # E1 – G4
        "guitar": (40, 88),   # E2 – E6
        "piano":  (21, 108),  # A0 – C8
        "vocals": (48, 84),   # C3 – C6
        "other":  (21, 108),
    }

    def _clamp_pitch_range(self, notes: List[Note], stem_name: str) -> List[Note]:
        """Remove notes outside the expected pitch range for this instrument."""
        lo, hi = self._PITCH_RANGES.get(stem_name, (21, 108))
        return [n for n in notes if lo <= n.pitch <= hi]

    # ── Overlap removal ────────────────────────────────────────────────────────

    @staticmethod
    def _remove_overlapping_notes(notes: List[Note]) -> List[Note]:
        """
        For each MIDI pitch, trim or remove notes that overlap with a later note.
        Keeps the earlier note, truncating it just before the next note starts.
        """
        from collections import defaultdict

        by_pitch: Dict[int, List[Note]] = defaultdict(list)
        for note in notes:
            by_pitch[note.pitch].append(note)

        result: List[Note] = []
        for pitch, pitch_notes in by_pitch.items():
            pitch_notes.sort(key=lambda n: n.start_time)
            clean: List[Note] = []
            for i, note in enumerate(pitch_notes):
                if i + 1 < len(pitch_notes):
                    next_start = pitch_notes[i + 1].start_time
                    if note.end_time > next_start:
                        # Truncate current note
                        note = Note(
                            start_time=note.start_time,
                            end_time=next_start - 0.001,
                            pitch=note.pitch,
                            velocity=note.velocity,
                            pitch_bends=note.pitch_bends,
                        )
                if note.duration > 0.001:
                    clean.append(note)
            result.extend(clean)

        return result

    # ── Drum deduplication ─────────────────────────────────────────────────────

    @staticmethod
    def _dedup_drum_hits(hits: List[DrumHit], window: float) -> List[DrumHit]:
        """
        Merge drum hits of the same pitch that are within `window` seconds.
        Keeps the hit with the highest velocity.
        """
        if not hits:
            return hits

        hits.sort(key=lambda h: (h.pitch, h.time))
        deduped: List[DrumHit] = [hits[0]]

        for hit in hits[1:]:
            last = deduped[-1]
            if hit.pitch == last.pitch and (hit.time - last.time) < window:
                # Keep higher velocity
                if hit.velocity > last.velocity:
                    deduped[-1] = hit
            else:
                deduped.append(hit)

        return deduped

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _stem_duration(transcription: StemTranscription) -> float:
        """Return the time of the last event in a stem."""
        if transcription.is_drum:
            if not transcription.drum_hits:
                return 0.0
            return max(h.time for h in transcription.drum_hits)
        else:
            if not transcription.notes:
                return 0.0
            return max(n.end_time for n in transcription.notes)