# Leb_STT

Local speech-to-text service. Whisper Large v3 Turbo for transcription, pyannote 3.1 for speaker diarization. Runs offline on GPU after a one-time model download.

## Install

```powershell
# Match all of torch/torchvision/torchaudio to the same CUDA channel
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt

# ffmpeg (for mp3/m4a decoding)
winget install Gyan.FFmpeg
```

Verify GPU is visible: `python scripts/check_gpu.py`

## Configure

```powershell
cp .env.example .env   # then edit
```

Required:
- `HF_TOKEN` — get one at https://huggingface.co/settings/tokens. Needed for the gated pyannote model.

Before first diarization run, accept the model terms (logged in to HF):
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

## Run

```powershell
python scripts/warmup_local.py   # pre-downloads Whisper (~1.6 GB)
python run.py
```

## API

`POST /transcribe` (multipart form):

| Field           | Type | Description |
|-----------------|------|-------------|
| `file`          | file | Audio: wav / mp3 / m4a / flac / ogg / webm |
| `language`      | text | Whisper language hint (e.g. `arabic`). Empty = auto |
| `prompt`        | text | Decoder prompt — biases vocabulary, names, style. ~224 tokens |
| `num_speakers`  | int  | Pin speaker count. **≥ 2 enables diarization** |
| `min_speakers`  | int  | Lower bound. ≥ 2 enables diarization |
| `max_speakers`  | int  | Upper bound. ≥ 2 enables diarization |

```powershell
# Plain
curl.exe -F "file=@a.m4a" http://localhost:8000/transcribe

# With diarization (2 speakers)
curl.exe -F "file=@a.m4a" -F "num_speakers=2" http://localhost:8000/transcribe
```

Diarized response includes `labeled_text` (`Speaker 1: ...\nSpeaker 2: ...`), `lines` with timestamps, and the raw `speaker_segments`.

`GET /health` — model + device info.
