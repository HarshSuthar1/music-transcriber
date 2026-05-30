# 🎵 Music Transcriber

A full multi-instrument music transcription pipeline that converts audio files into MIDI, Guitar Pro, and MusicXML notation using AI-powered source separation and automatic music transcription.

## Pipeline Overview

```
Audio File (WAV/MP3/FLAC/M4A/OGG)
        │
        ▼
   Audio Input
   (load, validate, resample, normalize)
        │
        ▼
  Source Separation (Demucs)
  ┌─────┬─────┬────────┬───────┐
drums bass  vocals  guitar  piano/other
  │     │      │       │       │
  └─────┴──────┴───────┴───────┘
        │
        ▼
  AMT per Stem (Basic Pitch)
  (pitch detection, onset/offset, velocity)
        │
        ▼
   Post-Processing
   (quantization, alignment, voice assignment)
        │
        ▼
  ┌─────┬──────────┬──────────┐
MIDI  GuitarPro  MusicXML
```

## Features

- **Multi-format input**: WAV, MP3, FLAC, M4A, OGG
- **Source separation**: Demucs htdemucs_6s (drums, bass, vocals, guitar, piano, other)
- **AMT**: Basic Pitch per-stem transcription with pitch bend support
- **Post-processing**: Tempo detection, quantization, note filtering
- **Output formats**:
  - MIDI (multi-track, per-instrument)
  - Guitar Pro 5 (.gp5) via PyGuitarPro
  - MusicXML (.xml) via music21

## Installation

### Prerequisites

- Python 3.9+
- ffmpeg (for audio conversion)

```bash
# Install ffmpeg (Ubuntu/Debian)
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Or install as a package

```bash
pip install -e .
```

## Usage

### Command Line

```bash
# Basic usage - transcribe to all formats
python cli.py song.mp3

# Specify output formats
python cli.py song.mp3 --formats midi xml

# Specify output directory
python cli.py song.mp3 --output ./my_transcriptions

# Only transcribe specific instruments
python cli.py song.mp3 --instruments drums bass guitar

# Use a specific Demucs model
python cli.py song.mp3 --model htdemucs_6s

# Keep intermediate stems
python cli.py song.mp3 --keep-stems

# Verbose mode
python cli.py song.mp3 --verbose
```

### Python API

```python
from transcriber.pipeline import TranscriptionPipeline, PipelineConfig

config = PipelineConfig(
    output_formats=["midi", "gp", "xml"],
    instruments=["drums", "bass", "guitar", "piano", "vocals", "other"],
    demucs_model="htdemucs_6s",
    quantize=True,
    keep_stems=False,
)

pipeline = TranscriptionPipeline(config)
result = pipeline.run("song.mp3", output_dir="./output")

print(f"MIDI: {result.midi_path}")
print(f"Guitar Pro: {result.gp_path}")
print(f"MusicXML: {result.xml_path}")
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `demucs_model` | `htdemucs_6s` | Demucs model (htdemucs, htdemucs_6s, mdx_extra) |
| `output_formats` | `["midi","gp","xml"]` | Output formats to generate |
| `instruments` | all 6 | Instruments to transcribe |
| `sample_rate` | `44100` | Target sample rate |
| `quantize` | `True` | Quantize notes to grid |
| `min_note_duration` | `0.05` | Minimum note duration in seconds |
| `velocity_floor` | `10` | Minimum MIDI velocity |
| `tempo` | auto-detect | BPM (None = auto-detect) |
| `time_signature` | `(4, 4)` | Time signature |
| `keep_stems` | `False` | Keep Demucs stems on disk |

## Output Structure

```
output/
├── song.mid              # Multi-track MIDI
├── song.gp5              # Guitar Pro 5
├── song.xml              # MusicXML
└── stems/                # (if --keep-stems)
    ├── drums.wav
    ├── bass.wav
    ├── vocals.wav
    ├── guitar.wav
    ├── piano.wav
    └── other.wav
```

## Instrument MIDI Mapping

| Stem | MIDI Channel | Program | Notes |
|------|-------------|---------|-------|
| Drums | 10 (percussion) | – | GM drum map |
| Bass | 1 | 33 (Electric Bass) | |
| Guitar | 2 | 25 (Acoustic Guitar) | |
| Piano | 3 | 0 (Acoustic Grand) | |
| Vocals | 4 | 52 (Choir Aahs) | |
| Other | 5 | 48 (String Ensemble) | |

## Troubleshooting

**Demucs OOM error**: Reduce `--segment` size or use `--two-stems` mode.

**Poor transcription quality**: Try `--model mdx_extra` for better separation, or increase `--min-note-duration` to filter noise.

**Missing Guitar Pro output**: Ensure `PyGuitarPro` is installed correctly.

## License

MIT