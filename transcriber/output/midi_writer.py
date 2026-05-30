"""
midi_writer.py
Writes per-stem NoteEvent lists to a multi-track MIDI file using pretty_midi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# General MIDI program numbers for each stem
STEM_PROGRAMS: Dict[str, int] = {
    "vocals":  52,   # Choir Aahs
    "drums":    0,   # (channel 9 percussion — program ignored)
    "bass":    33,   # Electric Bass (finger)
    "guitar":  25,   # Acoustic Guitar (steel)
    "piano":    0,   # Acoustic Grand Piano
    "other":   48,   # String Ensemble 1
}

STEM_CHANNELS: Dict[str, int] = {
    "drums": 9,   # GM percussion channel
}


@dataclass
class NoteEvent:
    pitch: int          # MIDI note number 0-127
    start: float        # seconds
    end: float          # seconds
    velocity: int = 80  # 0-127


class MidiWriter:
    """
    Converts a dict of {stem_name: [NoteEvent, ...]} into a single
    multi-track MIDI file.
    """

    def __init__(self, tempo: float = 120.0, time_signature: tuple = (4, 4)):
        self.tempo = tempo
        self.time_signature = time_signature

    # ------------------------------------------------------------------
    def write(
        self,
        notes_by_stem: Dict[str, List[NoteEvent]],
        output_path: str | Path,
        *,
        merge_to_single_track: bool = False,
    ) -> Path:
        """
        Write a .mid file and return the resolved Path.

        Parameters
        ----------
        notes_by_stem:
            Mapping of stem name -> list of NoteEvent objects.
        output_path:
            Destination .mid file path.
        merge_to_single_track:
            If True, collapse all stems onto one instrument track.
        """
        try:
            import pretty_midi  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pretty_midi is required: pip install pretty_midi"
            ) from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)

        # Time-signature event
        num, denom = self.time_signature
        midi.time_signature_changes = [
            pretty_midi.TimeSignature(num, denom, 0.0)
        ]

        if merge_to_single_track:
            instrument = pretty_midi.Instrument(program=0, name="All Stems")
            for stem, events in notes_by_stem.items():
                self._add_notes(instrument, events)
            midi.instruments.append(instrument)
        else:
            for stem, events in notes_by_stem.items():
                if not events:
                    continue
                program = STEM_PROGRAMS.get(stem, 0)
                is_drum = stem == "drums"
                instrument = pretty_midi.Instrument(
                    program=program,
                    is_drum=is_drum,
                    name=stem.capitalize(),
                )
                self._add_notes(instrument, events)
                midi.instruments.append(instrument)
                logger.info("  MIDI track '%s': %d notes", stem, len(events))

        midi.write(str(output_path))
        logger.info("MIDI written → %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    @staticmethod
    def _add_notes(instrument, events: List[NoteEvent]) -> None:
        try:
            import pretty_midi  # type: ignore
        except ImportError:
            raise

        for ev in events:
            if ev.end <= ev.start:
                continue  # skip zero-length notes
            note = pretty_midi.Note(
                velocity=max(1, min(127, ev.velocity)),
                pitch=max(0, min(127, ev.pitch)),
                start=ev.start,
                end=ev.end,
            )
            instrument.notes.append(note)
        instrument.notes.sort(key=lambda n: n.start)

    # ------------------------------------------------------------------
    @staticmethod
    def from_basic_pitch_output(bp_output: dict) -> List[NoteEvent]:
        """
        Convert a Basic Pitch output dict (from basic_pitch.inference.predict)
        into a flat list of NoteEvent objects.

        bp_output keys used:
            "note_events" -> list of (start_s, end_s, pitch, amplitude, ...)
        """
        events: List[NoteEvent] = []
        for row in bp_output.get("note_events", []):
            start, end, pitch, amplitude = row[:4]
            velocity = int(amplitude * 127)
            events.append(NoteEvent(pitch=int(pitch), start=float(start),
                                    end=float(end), velocity=velocity))
        return events