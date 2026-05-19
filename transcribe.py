#!/usr/bin/env python3
"""
D&D session transcription using faster-whisper.

Usage:
    python transcribe.py <audio_or_video_file> [--model MODEL_SIZE]

Supports video files (mp4, mkv, etc.) — audio is extracted via ffmpeg first.
Output is written to sessions/transcriptions/ as a timestamped text file.

Model sizes: tiny, base, small, medium, large (default: small)
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".m4a"}

MODEL_NAME = "small"

HALLUCINATION_BLOCKLIST = [
    "thanks for watching",
    "thank you for watching",
    "subtitles by",
    "subscribe to",
    "please subscribe",
    "...",
]


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found. Install from https://ffmpeg.org/download.html")
        sys.exit(1)


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """Extract mono 16kHz MP3 from a video file (optimal for Whisper)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}.mp3"

    print(f"Extracting audio from: {video_path.name}")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "24k", "-y",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Audio saved to: {output_path}\n")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def is_hallucination(text: str) -> bool:
    normalized = text.lower().strip()
    return any(
        normalized == phrase or normalized.startswith(phrase)
        for phrase in HALLUCINATION_BLOCKLIST
    )


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe(audio_path: Path, model_size: str) -> Path:
    from faster_whisper import WhisperModel

    print(f"Loading model '{model_size}' (device=cpu, compute_type=int8)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("Model ready.\n")

    print(f"Transcribing: {audio_path.name}")
    start = time.time()

    segments_gen, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments = [seg for seg in segments_gen if seg.text.strip() and not is_hallucination(seg.text)]
    elapsed = time.time() - start

    script_dir = Path(__file__).parent
    out_dir = script_dir / "sessions" / "transcriptions"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{audio_path.stem}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{format_timestamp(seg.start)}] {seg.text.strip()}\n")

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Segments: {len(segments)}")
    print(f"  Duration: ~{format_timestamp(info.duration)}")
    print(f"  Output:   {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a D&D session audio/video file using faster-whisper."
    )
    parser.add_argument("input_file", help="Path to the audio or video file")
    parser.add_argument(
        "--model", "-m",
        choices=["tiny", "base", "small", "medium", "large"],
        default=MODEL_NAME,
        help=f"Whisper model size (default: {MODEL_NAME})",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.suffix.lower() in VIDEO_EXTENSIONS:
        check_ffmpeg()
        recordings_dir = Path(__file__).parent / "sessions" / "recordings"
        audio_path = extract_audio(input_path, recordings_dir)
    elif input_path.suffix.lower() in AUDIO_EXTENSIONS:
        audio_path = input_path
    else:
        print(f"Error: Unsupported file type: {input_path.suffix}", file=sys.stderr)
        sys.exit(1)

    transcribe(audio_path, args.model)


if __name__ == "__main__":
    main()
