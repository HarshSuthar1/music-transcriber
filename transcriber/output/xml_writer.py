"""
xml_writer.py
Converts per-stem NoteEvent lists into a MusicXML (.xml / .mxl) file
using music21.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .midi_writer import NoteEvent

logger = logging.getLogger(__name__)

# music21 instrument class names per stem
INSTRUMENT_NAMES: Dict[str, str] = {
    "vocals": "Vocalist",
    "drums":  "UnpitchedPercussion",
    "bass":   "ElectricBass",
    "guitar": "Guitar",
    "piano":  "Piano",
    "other":  "Instrument",
}

# Clef overrides (music21 clef class names)
CLEF_NAMES: Dict[str, str] = {
    "bass":  "BassClef",
    "drums": "PercussionClef",
}


class MusicXMLWriter:
    """
    Writes a multi-part MusicXML score from per-stem NoteEvent dicts.

    Usage
    -----
    writer = MusicXMLWriter(tempo=120, time_signature=(4, 4))
    writer.write(notes_by_stem, "output/song.xml")   # or .mxl for compressed
    """

    def __init__(
        self,
        tempo: float = 120.0,
        time_signature: Tuple[int, int] = (4, 4),
        title: str = "Transcription",
        composer: str = "music-transcriber",
    ):
        self.tempo = tempo
        self.time_signature = time_signature
        self.title = title
        self.composer = composer

    # ------------------------------------------------------------------
    def write(
        self,
        notes_by_stem: Dict[str, List[NoteEvent]],
        output_path: str | Path,
    ) -> Path:
        try:
            import music21 as m21  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "music21 is required: pip install music21"
            ) from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        score = m21.stream.Score()
        score.metadata = m21.metadata.Metadata()
        score.metadata.title = self.title
        score.metadata.composer = self.composer

        beat_duration_ql = 1.0  # quarter-length of one beat (quarter note = 1.0)
        seconds_per_beat = 60.0 / self.tempo
        num, denom = self.time_signature
        measure_ql = num * beat_duration_ql  # quarter-lengths per measure

        for stem, events in notes_by_stem.items():
            if not events:
                continue
            part = self._build_part(m21, stem, events, seconds_per_beat, measure_ql)
            score.append(part)
            logger.info("  MusicXML part '%s': %d notes", stem, len(events))

        # Write
        if output_path.suffix.lower() == ".mxl":
            score.write("musicxml.mxl", fp=str(output_path))
        else:
            score.write("musicxml", fp=str(output_path))

        logger.info("MusicXML written → %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    def _build_part(
        self,
        m21,
        stem: str,
        events: List[NoteEvent],
        seconds_per_beat: float,
        measure_ql: float,
    ):
        num, denom = self.time_signature
        part = m21.stream.Part()
        part.id = stem

        # Instrument
        instr_name = INSTRUMENT_NAMES.get(stem, "Instrument")
        try:
            instr = m21.instrument.fromString(instr_name)
        except Exception:
            instr = m21.instrument.Instrument()
            instr.instrumentName = stem.capitalize()
        part.insert(0, instr)

        # Tempo marking
        mm = m21.tempo.MetronomeMark(number=self.tempo)
        part.insert(0, mm)

        # Convert seconds → quarter-length offsets
        def s_to_ql(sec: float) -> float:
            return sec / seconds_per_beat

        total_ql = s_to_ql(max(ev.end for ev in events)) + measure_ql
        num_measures = int(total_ql / measure_ql) + 1

        measures = []
        for i in range(num_measures):
            m = m21.stream.Measure(number=i + 1)
            ts = m21.meter.TimeSignature(f"{num}/{denom}")
            if i == 0:
                m.insert(0, ts)
                clef_name = CLEF_NAMES.get(stem, "TrebleClef")
                clef_cls = getattr(m21.clef, clef_name, m21.clef.TrebleClef)
                m.insert(0, clef_cls())
            measures.append(m)

        # Insert notes
        for ev in events:
            start_ql = s_to_ql(ev.start)
            dur_ql = max(0.0625, s_to_ql(ev.end - ev.start))  # min 64th note
            m_idx = min(int(start_ql / measure_ql), num_measures - 1)
            offset_in_measure = start_ql - m_idx * measure_ql

            if stem == "drums":
                note_obj = m21.note.Unpitched()
                note_obj.duration = m21.duration.Duration(quarterLength=dur_ql)
            else:
                try:
                    note_obj = m21.note.Note(ev.pitch)
                    note_obj.duration = m21.duration.Duration(quarterLength=dur_ql)
                    note_obj.volume.velocity = max(1, min(127, ev.velocity))
                except Exception:
                    continue

            measures[m_idx].insert(offset_in_measure, note_obj)

        # Fill empty measures with a whole rest
        for m in measures:
            if len(m.notesAndRests) == 0:
                rest = m21.note.Rest()
                rest.duration = m21.duration.Duration(quarterLength=measure_ql)
                m.insert(0, rest)
            part.append(m)

        part.makeMeasures(inPlace=True)
        return part