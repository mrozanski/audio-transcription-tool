# Interview Transcription Tool

A local transcription workflow for interview audio using WhisperX. Optimized for Apple Silicon Macs (M1/M2/M3/M4), with full offline processing after initial model download.

## Features

- **Two transcription modes:**
  - **Separate tracks**: For high-quality DAW exports with each speaker on their own track
  - **Mixed audio**: For ambient recordings with automatic speaker diarization
- **Speaker labeling**: Automatic speaker identification and custom naming
- **Fully local**: All processing happens on your machine
- **Conversation-style output**: Interleaved dialogue with optional timestamps

## Requirements

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- ffmpeg (for audio processing)
- Hugging Face account (one-time setup for diarization models)

## Installation

### 1. Install uv and ffmpeg

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ffmpeg
brew install ffmpeg
```

### 2. Install dependencies

```bash
cd /path/to/audio-transcription-tool
uv sync
```

This creates a virtual environment and installs all dependencies automatically.

### 3. Set up Hugging Face authentication

The speaker diarization models require accepting a license agreement on Hugging Face. This is a one-time setup.

1. Visit these model pages and accept the terms:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

2. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

3. Save your token (choose one method):

   **Option A: Environment variable**
   ```bash
   export HF_TOKEN="your_token_here"
   ```

   **Option B: Local .env file**
   ```bash
   echo 'HF_TOKEN=your_token_here' > .env
   ```

   **Option C: Hugging Face CLI**
   ```bash
   uv run huggingface-cli login
   ```

## Usage

### Mode 1: Separate Tracks (High-Quality DAW Exports)

When you have each speaker on their own audio file with identical timing:

```bash
uv run python transcribe.py --speaker1 host.wav --speaker2 guest.wav --labels "Maria,John"
```

This mode transcribes each track independently and interleaves the results by timestamp, giving you perfect speaker attribution.

### Mode 2: Single Mixed File (Ambient Recordings)

When you have a single recording with all speakers mixed:

```bash
uv run python transcribe.py --input interview.wav
```

This mode uses automatic speaker diarization to identify who spoke when.

**Tip**: Specifying the number of speakers improves accuracy:

```bash
uv run python transcribe.py --input interview.wav --num-speakers 2 --labels "Host,Guest"
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input FILE` | `-i` | Single audio file (mixed-audio mode) |
| `--speaker1 FILE` | `-s1` | First speaker's track (separate-tracks mode) |
| `--speaker2 FILE` | `-s2` | Second speaker's track (separate-tracks mode) |
| `--labels NAMES` | `-l` | Comma-separated speaker names |
| `--num-speakers N` | `-n` | Number of speakers (helps diarization) |
| `--output FILE` | `-o` | Output file (default: stdout) |
| `--timestamps` | `-t` | Include timestamps in output |
| `--model SIZE` | `-m` | Whisper model (default: large-v3) |
| `--hf-token TOKEN` | | Hugging Face token |

### Examples

```bash
# Basic mixed-audio transcription
uv run python transcribe.py -i meeting.wav

# With speaker names and timestamps
uv run python transcribe.py -i interview.mp3 -l "Alice,Bob" -t

# Save to file
uv run python transcribe.py -i podcast.wav -o transcript.txt

# Separate tracks from DAW
uv run python transcribe.py -s1 track_host.wav -s2 track_guest.wav -l "Host,Guest"

# Use a smaller model for faster processing
uv run python transcribe.py -i recording.wav -m medium
```

## Model Sizes

| Model | Parameters | Relative Speed | Best For |
|-------|------------|----------------|----------|
| tiny | 39M | Fastest | Quick tests |
| base | 74M | Fast | Drafts |
| small | 244M | Moderate | Good balance |
| medium | 769M | Slower | Better accuracy |
| large-v3 | 1.5B | Slowest | Best accuracy |

For interviews where accuracy matters, `large-v3` (the default) is recommended.

## Output Format

The tool produces conversation-style plain text:

```
Maria: So tell me about your experience with the project.

John: Well, it started about three years ago when we first began exploring the technology. We didn't know what to expect at the time.

Maria: And how did it evolve from there?
```

With `--timestamps`:

```
[00:00] Maria: So tell me about your experience with the project.

[00:05] John: Well, it started about three years ago when we first began exploring the technology. We didn't know what to expect at the time.

[00:18] Maria: And how did it evolve from there?
```

## Troubleshooting

### "No module named whisperx"

Make sure you've synced dependencies:

```bash
uv sync
```

Then run with `uv run`:

```bash
uv run python transcribe.py -i audio.wav
```

### Hugging Face authentication errors

Ensure you've:
1. Accepted the model licenses on both pyannote model pages
2. Set your HF_TOKEN correctly (check with `echo $HF_TOKEN`)

### Slow performance

- The first run downloads models (~3GB for large-v3), which takes time
- Subsequent runs use cached models
- Try `--model medium` for faster processing with slightly reduced accuracy

### Memory issues

For very long recordings (>1 hour), consider:
- Using a smaller model (`--model small`)
- Splitting the audio file into chunks

## Offline Usage

After the initial model download, the tool works completely offline. Models are cached in:
- `~/.cache/huggingface/` (Hugging Face models)
- `~/.cache/torch/` (Whisper models)

## License

MIT
