# D&D Session Notes

Campaign session notes and transcription tools for the Ringed City campaign.

## Transcription Tool

The `transcribe.py` script transcribes audio and video files using OpenAI Whisper.

### Requirements

Install all dependencies from requirements.txt:

```bash
python -m pip install -r requirements.txt
```

For video file support, install [ffmpeg](https://ffmpeg.org/download.html) and ensure it's in your PATH.

### Usage

```bash
python transcribe.py <input_file> [--model MODEL] [--format FORMAT]
```

### Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `input_file` | | Path to audio or video file (required) | |
| `--model` | `-m` | Whisper model size | `base` |
| `--format` | `-f` | Output format | `txt` |

### Model Sizes

| Model | Accuracy | RAM Usage |
|-------|----------|-----------|
| `tiny` | Lowest | ~1GB |
| `base` | Decent | ~1GB |
| `small` | Good | ~2GB |
| `medium` | High | ~5GB |
| `large` | Highest | ~10GB |

### Output Formats

| Format | Description |
|--------|-------------|
| `txt` | Plain text transcription |
| `srt` | SubRip subtitle format with timestamps |
| `vtt` | WebVTT subtitle format with timestamps |
| `json` | Full JSON output with all metadata |

### Examples

```bash
# Basic transcription with defaults
python transcribe.py recording.mp3

# Use a more accurate model
python transcribe.py session_video.mp4 --model medium

# Output as subtitles
python transcribe.py recording.mp3 --format srt

# Video with highest accuracy and JSON output
python transcribe.py video.mkv --model large --format json
```

### Output Location

- Transcriptions are saved to `sessions/transcriptions/`
- Extracted audio from video files is saved to `sessions/recordings/`
