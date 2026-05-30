"""
gp_writer.py
Converts per-stem NoteEvent lists into a Guitar Pro 5 (.gp5) file
using the `guitarpro` (PyGuitarPro) library.

Drum mapping follows the General MIDI / GP5 standard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .midi_writer import NoteEvent

logger = logging.getLogger(__name__)

# ── Guitar Pro string/tuning presets ──────────────────────────────────────────
# Each value is a list of open-string MIDI pitches, low → high.
TUNINGS: Dict[str, List[int]] = {
    "guitar": [40, 45, 50, 55, 59, 64],   # E2 A2 D3 G3 B3 E4
    "bass":   [28, 33, 38, 43],            # E1 A1 D2 G2
    "vocals": [64, 67, 69, 72, 74, 76],   # "lead sheet" — treble strings
    "piano":  [64, 67, 69, 72, 74, 76],
    "other":  [64, 67, 69, 72, 74, 76],
    "drums":  [],                          # percussion track has no string tuning
}

# GP5 channel presets (channel index, effectChannel index, instrument, volume, …)
# We keep it simple: one channel block per track.
GP_INSTRUMENT: Dict[str, int] = {
    "guitar": 25,
    "bass":   33,
    "vocals": 52,
    "piano":   0,
    "other":  48,
    "drums":   0,   # ignored for percussion
}


class GuitarProWriter:
    """
    Writes a multi-track .gp5 file from per-stem NoteEvent dicts.

    Usage
    -----
    writer = GuitarProWriter(tempo=120, time_signature=(4, 4))
    writer.write(notes_by_stem, "output/song.gp5")
    """

    def __init__(self, tempo: float = 120.0, time_signature: Tuple[int, int] = (4, 4)):
        self.tempo = int(tempo)
        self.time_signature = time_signature

    # ------------------------------------------------------------------
    def write(
        self,
        notes_by_stem: Dict[str, List[NoteEvent]],
        output_path: str | Path,
    ) -> Path:
        try:
            import guitarpro  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyGuitarPro is required: pip install PyGuitarPro"
            ) from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        song = guitarpro.models.Song()
        song.tempo = self.tempo
        song.tracks = []

        beat_duration_s = 60.0 / self.tempo  # duration of one quarter note

        for track_idx, (stem, events) in enumerate(notes_by_stem.items()):
            if not events:
                continue

            is_drum = stem == "drums"
            track = self._make_track(guitarpro, stem, track_idx, is_drum)
            measures = self._events_to_measures(
                guitarpro, events, beat_duration_s, is_drum
            )
            track.measures = measures
            song.tracks.append(track)
            logger.info("  GP5 track '%s': %d notes", stem, len(events))

        guitarpro.write(song, str(output_path))
        logger.info("Guitar Pro written → %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    def _make_track(self, gp, stem: str, idx: int, is_drum: bool):
        track = gp.models.Track()
        track.number = idx + 1
        track.name = stem.capitalize()
        track.isPercussionTrack = is_drum

        strings = TUNINGS.get(stem, TUNINGS["other"])
        track.strings = [
            gp.models.GuitarString(i + 1, pitch)
            for i, pitch in enumerate(strings)
        ] if strings else []

        channel = gp.models.MixTableItem()
        track.channel = gp.models.MidiChannel()
        track.channel.channel = idx
        track.channel.effectChannel = idx
        track.channel.instrument = GP_INSTRUMENT.get(stem, 0)
        track.channel.volume = 13
        track.channel.balance = 8
        track.channel.chorus = 0
        track.channel.reverb = 0
        return track

    # ------------------------------------------------------------------
    def _events_to_measures(
        self,
        gp,
        events: List[NoteEvent],
        beat_duration_s: float,
        is_drum: bool,
    ) -> List:
        """
        Pack NoteEvents into GP5 measures / beats.
        Each measure = 4 quarter-note beats (4/4).
        Notes are quantised to the nearest 16th note.
        """
        if not events:
            return [self._empty_measure(gp, beat_duration_s)]

        measure_duration_s = beat_duration_s * self.time_signature[0]
        sixteenth_s = beat_duration_s / 4.0
        num, denom = self.time_signature

        total_duration = max(ev.end for ev in events)
        num_measures = max(1, int(total_duration / measure_duration_s) + 1)

        # Bucket events by measure
        buckets: Dict[int, List[NoteEvent]] = {i: [] for i in range(num_measures)}
        for ev in events:
            m = int(ev.start / measure_duration_s)
            buckets[min(m, num_measures - 1)].append(ev)

        measures = []
        header = gp.models.MeasureHeader()
        header.timeSignature = gp.models.TimeSignature()
        header.timeSignature.numerator = num
        header.timeSignature.denominator = gp.models.Duration(value=denom)

        for m_idx in range(num_measures):
            measure = gp.models.Measure(header=header)
            voice = gp.models.Voice()

            m_events = sorted(buckets[m_idx], key=lambda e: e.start)
            if not m_events:
                # Rest beat filling the measure
                rest_beat = gp.models.Beat(voice)
                rest_beat.duration = gp.models.Duration(value=1)  # whole
                rest_beat.status = gp.models.BeatStatus.rest
                voice.beats = [rest_beat]
            else:
                beats = []
                for ev in m_events:
                    beat = gp.models.Beat(voice)
                    beat.duration = gp.models.Duration(value=16)  # 16th note
                    note = gp.models.Note(beat)
                    note.value = ev.pitch
                    note.velocity = gp.models.Velocities.forte
                    beat.notes = [note]
                    beats.append(beat)
                voice.beats = beats

            measure.voices = [voice]
            measures.append(measure)

        return measures

    # ------------------------------------------------------------------
    @staticmethod
    def _empty_measure(gp, beat_duration_s: float):
        header = gp.models.MeasureHeader()
        header.timeSignature = gp.models.TimeSignature()
        header.timeSignature.numerator = 4
        header.timeSignature.denominator = gp.models.Duration(value=4)
        measure = gp.models.Measure(header=header)
        voice = gp.models.Voice()
        rest = gp.models.Beat(voice)
        rest.duration = gp.models.Duration(value=1)
        rest.status = gp.models.BeatStatus.rest
        voice.beats = [rest]
        measure.voices = [voice]
        return measure