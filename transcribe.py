#!/usr/bin/env python3
"""
Audio/Video Transcription Script using OpenAI Whisper
Usage: python transcribe.py <audio_or_video_file> [--model MODEL_SIZE]

Supports video files (mp4, mkv, etc.) - audio will be extracted and compressed via ffmpeg.
Model sizes: tiny, base, small, medium, large (default: base)
Larger models are more accurate but slower.
"""

import argparse
import sys
import os
import subprocess
import shutil
from pathlib import Path

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v'}


def check_ffmpeg():
    """Check if ffmpeg is available on the system."""
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found.")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)


def extract_and_compress_audio(video_path, output_dir):
    """
    Extract audio from video and compress it using ffmpeg.
    Outputs mono 16kHz MP3 (optimal for Whisper).
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{video_path.stem}.mp3"

    print(f"\n{'='*60}")
    print(f"Extracting audio from: {video_path.name}")
    print(f"{'='*60}\n")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                  # No video
        "-ac", "1",             # Mono
        "-ar", "16000",         # 16kHz sample rate (Whisper optimal)
        "-b:a", "24k",          # 24kbps bitrate
        "-y",                   # Overwrite output
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Audio extracted and saved to: {output_path}\n")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr}")
        sys.exit(1)


try:
    import whisper
    from tqdm import tqdm
except ImportError as e:
    print("Error: Required packages not installed.")
    print("\nPlease install the required packages:")
    print("  pip install openai-whisper")
    print("  pip install tqdm")
    print("\nNote: For faster transcription, also install:")
    print("  pip install torch")
    sys.exit(1)


def transcribe_audio(audio_path, model_size="base", output_format="txt"):
    """
    Transcribe an audio file using Whisper
    
    Args:
        audio_path: Path to the audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        output_format: Output format (txt, srt, vtt, json)
    """
    # Verify the audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    audio_path = Path(audio_path)
    print(f"\n{'='*60}")
    print(f"Transcribing: {audio_path.name}")
    print(f"Model size: {model_size}")
    print(f"{'='*60}\n")
    
    # Load the Whisper model
    print(f"Loading Whisper '{model_size}' model...")
    print("(This may take a moment on first run as the model downloads)")
    
    try:
        model = whisper.load_model(model_size)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Transcribe the audio
    print("Transcribing audio...")
    print("(This may take several minutes for long files)\n")
    
    try:
        # Whisper's transcribe function with progress
        result = model.transcribe(
            str(audio_path),
            verbose=True,  # This shows progress during transcription
            fp16=False  # Set to True if you have a CUDA-compatible GPU
        )
        
        print("\n✓ Transcription complete!\n")
        
    except Exception as e:
        print(f"\nError during transcription: {e}")
        sys.exit(1)
    
    # Generate output filename in sessions/transcriptions folder
    script_dir = Path(__file__).parent
    transcriptions_dir = script_dir / "sessions" / "transcriptions"
    transcriptions_dir.mkdir(parents=True, exist_ok=True)

    output_path = transcriptions_dir / f"{audio_path.stem}.{output_format}"
    
    # Save the transcription
    print(f"Saving transcription to: {output_path.name}")
    
    try:
        if output_format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
        
        elif output_format == "srt":
            # Write SRT subtitle format
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(result["segments"], start=1):
                    start = format_timestamp(segment["start"])
                    end = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        
        elif output_format == "vtt":
            # Write WebVTT subtitle format
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                for segment in result["segments"]:
                    start = format_timestamp(segment["start"], vtt=True)
                    end = format_timestamp(segment["end"], vtt=True)
                    text = segment["text"].strip()
                    f.write(f"{start} --> {end}\n{text}\n\n")
        
        elif output_format == "json":
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Transcription saved successfully!\n")
        
        # Display some stats
        word_count = len(result["text"].split())
        print(f"{'='*60}")
        print(f"Statistics:")
        print(f"  Words: {word_count:,}")
        print(f"  Characters: {len(result['text']):,}")
        if "segments" in result:
            print(f"  Segments: {len(result['segments'])}")
        print(f"{'='*60}\n")
        
        # Show preview of transcription
        preview_length = 500
        preview = result["text"][:preview_length]
        if len(result["text"]) > preview_length:
            preview += "..."
        
        print("Preview:")
        print("-" * 60)
        print(preview)
        print("-" * 60)
        
    except Exception as e:
        print(f"Error saving transcription: {e}")
        sys.exit(1)
    
    return output_path


def format_timestamp(seconds, vtt=False):
    """Convert seconds to timestamp format (SRT or VTT)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    if vtt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py recording.mp3
  python transcribe.py session_video.mp4 --model medium
  python transcribe.py recording.mp3 --format srt
  python transcribe.py video.mkv --model large --format json

Video files (mp4, mkv, avi, etc.) will have audio extracted via ffmpeg.
Extracted audio is saved to sessions/recordings/.

Model Sizes (accuracy vs speed):
  tiny   - Fastest, least accurate (~1GB RAM)
  base   - Fast, decent accuracy (~1GB RAM) [DEFAULT]
  small  - Good balance (~2GB RAM)
  medium - High accuracy (~5GB RAM)
  large  - Highest accuracy (~10GB RAM)

Output Formats:
  txt  - Plain text transcription [DEFAULT]
  srt  - SubRip subtitle format with timestamps
  vtt  - WebVTT subtitle format with timestamps
  json - Full JSON output with all metadata
        """
    )

    parser.add_argument(
        "input_file",
        help="Path to the audio or video file to transcribe"
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["txt", "srt", "vtt", "json"],
        default="txt",
        help="Output format (default: txt)"
    )
    
    args = parser.parse_args()

    input_path = Path(args.input_file)

    # Check if input file exists
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # If video file, extract audio first
    if input_path.suffix.lower() in VIDEO_EXTENSIONS:
        check_ffmpeg()
        script_dir = Path(__file__).parent
        recordings_dir = script_dir / "sessions" / "recordings"
        audio_path = extract_and_compress_audio(input_path, recordings_dir)
    else:
        audio_path = input_path

    # Transcribe the audio
    output_path = transcribe_audio(
        audio_path,
        model_size=args.model,
        output_format=args.format
    )

    print(f"\n✓ All done! Transcription saved to: {output_path}")


if __name__ == "__main__":
    main()
