#!/usr/bin/env python3
"""
Interview Transcription Tool

A local transcription workflow using WhisperX for Apple Silicon Macs.
Supports two modes:
1. Separate tracks: Individual audio files per speaker (high-quality DAW exports)
2. Single mixed file: One audio file with automatic speaker diarization

All processing is done locally. Hugging Face authentication is only needed
once to download the pyannote diarization models.
"""

import argparse
import os
import sys
from pathlib import Path

import whisperx
import torch


def get_device():
    """Determine the best available device for inference."""
    if torch.backends.mps.is_available():
        # Apple Silicon - use MPS for whisper, but pyannote needs CPU
        return "cpu"  # WhisperX diarization doesn't support MPS yet
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_whisper_model(model_size: str, device: str, language: str = None):
    """Load the WhisperX model."""
    compute_type = "float32" if device == "cpu" else "float16"
    print(f"Loading Whisper model '{model_size}' on {device}...", file=sys.stderr)
    if language:
        print(f"Using language: {language}", file=sys.stderr)
    model = whisperx.load_model(
        model_size, device, compute_type=compute_type, language=language
    )
    return model


def transcribe_audio(model, audio_path: str, device: str, language: str = None) -> dict:
    """Transcribe a single audio file and return segments with timestamps."""
    print(f"Transcribing: {audio_path}", file=sys.stderr)
    audio = whisperx.load_audio(audio_path)
    transcribe_kwargs = {"batch_size": 16}
    if language:
        transcribe_kwargs["language"] = language
    result = model.transcribe(audio, **transcribe_kwargs)
    return result, audio


def align_transcription(result: dict, audio, model_size: str, device: str):
    """Align transcription to get word-level timestamps."""
    print("Aligning transcription...", file=sys.stderr)
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    return result


def diarize_audio(audio_path: str, hf_token: str, device: str, num_speakers: int = None):
    """Perform speaker diarization on audio file."""
    print("Performing speaker diarization...", file=sys.stderr)
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

    diarize_kwargs = {}
    if num_speakers:
        diarize_kwargs["num_speakers"] = num_speakers

    diarize_segments = diarize_model(audio_path, **diarize_kwargs)
    return diarize_segments


def assign_speakers(result: dict, diarize_segments) -> dict:
    """Assign speaker labels to transcription segments."""
    print("Assigning speakers to segments...", file=sys.stderr)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result


def transcribe_separate_tracks(
    model,
    track1_path: str,
    track2_path: str,
    labels: tuple[str, str],
    device: str,
    show_timestamps: bool,
    language: str = None,
) -> str:
    """
    Transcribe two separate audio tracks and interleave by timestamp.

    This mode is ideal for high-quality DAW exports where each speaker
    is on a separate track with identical timing.
    """
    # Transcribe both tracks
    result1, audio1 = transcribe_audio(model, track1_path, device, language)
    result2, audio2 = transcribe_audio(model, track2_path, device, language)

    # Align both transcriptions
    result1 = align_transcription(result1, audio1, model, device)
    result2 = align_transcription(result2, audio2, model, device)

    # Combine segments with speaker labels
    segments = []

    for seg in result1.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "speaker": labels[0],
        })

    for seg in result2.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "speaker": labels[1],
        })

    # Sort by start time to interleave conversation
    segments.sort(key=lambda x: x["start"])

    # Merge consecutive segments from the same speaker
    merged = merge_consecutive_segments(segments)

    return format_output(merged, show_timestamps)


def transcribe_mixed_audio(
    model,
    audio_path: str,
    hf_token: str,
    device: str,
    show_timestamps: bool,
    num_speakers: int = None,
    speaker_labels: dict[str, str] = None,
    language: str = None,
) -> str:
    """
    Transcribe a single mixed audio file with automatic speaker diarization.

    This mode is ideal for ambient recordings where all speakers are
    captured in a single channel.
    """
    # Transcribe
    result, audio = transcribe_audio(model, audio_path, device, language)

    # Align
    result = align_transcription(result, audio, model, device)

    # Diarize
    diarize_segments = diarize_audio(audio_path, hf_token, device, num_speakers)

    # Assign speakers
    result = assign_speakers(result, diarize_segments)

    # Extract segments with speaker info
    segments = []
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "UNKNOWN")
        if speaker_labels and speaker in speaker_labels:
            speaker = speaker_labels[speaker]

        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "speaker": speaker,
        })

    # Merge consecutive segments from the same speaker
    merged = merge_consecutive_segments(segments)

    return format_output(merged, show_timestamps)


def merge_consecutive_segments(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments from the same speaker."""
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        if seg["speaker"] == merged[-1]["speaker"]:
            # Same speaker - merge
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            # Different speaker - new segment
            merged.append(seg.copy())

    return merged


def format_output(segments: list[dict], show_timestamps: bool) -> str:
    """Format segments as plain text conversation."""
    lines = []

    for seg in segments:
        if show_timestamps:
            timestamp = format_timestamp(seg["start"])
            lines.append(f"[{timestamp}] {seg['speaker']}: {seg['text']}")
        else:
            lines.append(f"{seg['speaker']}: {seg['text']}")

    return "\n\n".join(lines)


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_hf_token() -> str:
    """Get Hugging Face token from environment or config file."""
    # Check environment variable first
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # Check .env file in current directory
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"\'')

    # Check default HuggingFace cache location
    hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if hf_token_path.exists():
        return hf_token_path.read_text().strip()

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe interview audio with speaker labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Separate tracks (high-quality DAW exports)
  python transcribe.py --speaker1 host.wav --speaker2 guest.wav --speaker-labels "Maria,John"

  # Single mixed file (ambient recording)
  python transcribe.py --input interview.wav

  # With timestamps
  python transcribe.py --input interview.wav --timestamps

  # Specify number of speakers for better diarization
  python transcribe.py --input interview.wav --num-speakers 2

  # Use a specific model size
  python transcribe.py --input interview.wav --model medium
        """,
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_argument_group("Input (choose one mode)")
    input_group.add_argument(
        "--input", "-i",
        metavar="FILE",
        help="Single audio file for mixed-audio mode with diarization",
    )
    input_group.add_argument(
        "--speaker1", "-s1",
        metavar="FILE",
        help="First speaker's audio track (separate-tracks mode)",
    )
    input_group.add_argument(
        "--speaker2", "-s2",
        metavar="FILE",
        help="Second speaker's audio track (separate-tracks mode)",
    )

    # Speaker options
    speaker_group = parser.add_argument_group("Speaker options")
    speaker_group.add_argument(
        "--speaker-labels", "-sl",
        metavar="NAMES",
        help="Comma-separated speaker names, e.g., 'Maria,John'",
    )
    speaker_group.add_argument(
        "--num-speakers", "-n",
        type=int,
        metavar="N",
        help="Number of speakers (helps diarization accuracy)",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Output file (default: stdout)",
    )
    output_group.add_argument(
        "--timestamps", "-t",
        action="store_true",
        help="Include timestamps in output",
    )

    # Model options
    model_group = parser.add_argument_group("Model options")
    model_group.add_argument(
        "--model", "-m",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v3)",
    )
    model_group.add_argument(
        "--language", "-L",
        metavar="CODE",
        help="Language code (e.g., 'en', 'es', 'fr'). Skips auto-detection if provided.",
    )

    # Authentication
    auth_group = parser.add_argument_group("Authentication")
    auth_group.add_argument(
        "--hf-token",
        metavar="TOKEN",
        help="Hugging Face token (or set HF_TOKEN env var)",
    )

    args = parser.parse_args()

    # Validate input mode
    separate_mode = args.speaker1 is not None or args.speaker2 is not None
    mixed_mode = args.input is not None

    if separate_mode and mixed_mode:
        parser.error("Cannot use both --input and --speaker1/--speaker2")

    if not separate_mode and not mixed_mode:
        parser.error("Must specify either --input or both --speaker1 and --speaker2")

    if separate_mode and (args.speaker1 is None or args.speaker2 is None):
        parser.error("Separate-tracks mode requires both --speaker1 and --speaker2")

    # Validate files exist
    if mixed_mode and not Path(args.input).exists():
        parser.error(f"File not found: {args.input}")
    if separate_mode:
        if not Path(args.speaker1).exists():
            parser.error(f"File not found: {args.speaker1}")
        if not Path(args.speaker2).exists():
            parser.error(f"File not found: {args.speaker2}")

    # Get HF token for diarization
    hf_token = args.hf_token or get_hf_token()
    if mixed_mode and not hf_token:
        parser.error(
            "Hugging Face token required for diarization. "
            "Set HF_TOKEN environment variable, create a .env file, "
            "or use --hf-token"
        )

    # Parse speaker labels
    labels = ("Speaker 1", "Speaker 2")
    if args.speaker_labels:
        parts = [l.strip() for l in args.speaker_labels.split(",")]
        if len(parts) >= 2:
            labels = (parts[0], parts[1])
        elif len(parts) == 1:
            labels = (parts[0], "Speaker 2")

    # Setup device and model
    device = get_device()
    model = load_whisper_model(args.model, device, language=args.language)

    # Run transcription
    if separate_mode:
        output = transcribe_separate_tracks(
            model=model,
            track1_path=args.speaker1,
            track2_path=args.speaker2,
            labels=labels,
            device=device,
            show_timestamps=args.timestamps,
            language=args.language,
        )
    else:
        # Create speaker label mapping if provided
        speaker_labels = None
        if args.speaker_labels:
            # Map SPEAKER_00, SPEAKER_01, etc. to provided names
            parts = [l.strip() for l in args.speaker_labels.split(",")]
            speaker_labels = {f"SPEAKER_{i:02d}": name for i, name in enumerate(parts)}

        output = transcribe_mixed_audio(
            model=model,
            audio_path=args.input,
            hf_token=hf_token,
            device=device,
            show_timestamps=args.timestamps,
            num_speakers=args.num_speakers,
            speaker_labels=speaker_labels,
            language=args.language,
        )

    # Write output
    if args.output:
        Path(args.output).write_text(output)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
