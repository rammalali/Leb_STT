# Leb_STT

Local speech-to-text service. **Whisper Large v3 Turbo** for transcription, **SpeechBrain ECAPA-TDNN** for speaker diarization. Runs offline on GPU after a one-time model download. No gated models, no auth tokens required.

## Install (bare metal)

```powershell
# 1. CUDA wheels first (must match your NVIDIA driver)
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

# 2. Everything else
pip install -r requirements.txt

# 3. ffmpeg (audio decoding for mp3/m4a/etc)
winget install Gyan.FFmpeg
```

Verify GPU is visible: `python scripts/check_gpu.py`

## Install (Docker)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host (Linux) or Docker Desktop with WSL2 GPU support (Windows).

```bash
cp .env.example .env
docker compose build
docker compose run --rm leb_stt python scripts/warmup_local.py   # one-time, populates volumes
docker compose up -d
```

Models are persisted in named volumes (`hf_cache`, `pretrained_models`) so they survive container rebuilds. The first transcription request after a cold start uses what was warmed up. To follow logs: `docker compose logs -f`.

## Configure

```powershell
cp .env.example .env
```

### `.env` variables

All have sensible defaults — you can leave the file untouched and it'll work.

| Variable                 | Default                            | Notes |
|--------------------------|------------------------------------|-------|
| `MODEL_ID`               | `openai/whisper-large-v3-turbo`    | Any HF Whisper model id. |
| `DEVICE`                 | `auto`                             | `auto`, `cuda`, or `cpu`. |
| `HF_TOKEN`               | *(empty)*                          | Optional. Only used to speed up the one-time Whisper download (faster, no rate-limit warning). All models in the default config are public — no token required. |
| `TRANSCRIBE_LANGUAGE`    | `arabic`                           | Default Whisper language hint. Empty = auto-detect. Per-request `language` form field overrides this. |
| `TRANSCRIBE_PROMPT`      | *(empty)*                          | Default decoder prompt (~224 tokens). Per-request `prompt` form field overrides this. |
| `DEFAULT_NUM_SPEAKERS`   | `2`                                | Pinned speaker count when `num_speakers` is not passed. Setting this to `2` means every request is treated as a two-person dialogue (diarization runs). Empty disables diarization by default. |
| `HOST`                   | `0.0.0.0`                          | uvicorn bind host. |
| `PORT`                   | `8000`                             | uvicorn bind port. |

**Minimal `.env` to override nothing:** the file can be empty (or just `cp .env.example .env`).

**Common tweaks:**
- Don't always have 2 speakers? → comment out / clear `DEFAULT_NUM_SPEAKERS` so plain transcription is the default.
- Different default language? → set `TRANSCRIBE_LANGUAGE=english` (or empty for auto).
- Want a default decoder prompt? → e.g. `TRANSCRIBE_PROMPT=A clean, line-by-line transcript with proper punctuation.`

## Run

```powershell
python scripts/warmup_local.py    # one-time: download Whisper (~1.6 GB)
python run.py                     # start server
```

## API

`POST /transcribe` (multipart form):

| Field           | Type | Description |
|-----------------|------|-------------|
| `file`          | file | Audio: wav / mp3 / m4a / flac / ogg / webm |
| `language`      | text | Whisper language hint (e.g. `arabic`). Empty = auto-detect |
| `prompt`        | text | Decoder prompt — biases vocabulary, names, style. ~224 tokens |
| `num_speakers`  | int  | Pin speaker count. **≥ 2 enables diarization** |
| `min_speakers`  | int  | Lower bound. ≥ 2 enables diarization |
| `max_speakers`  | int  | Upper bound. ≥ 2 enables diarization |

```powershell
# Plain transcription
curl.exe -F "file=@a.m4a" http://localhost:8000/transcribe

# With diarization (two-person dialogue)
curl.exe -F "file=@a.m4a" -F "num_speakers=2" http://localhost:8000/transcribe
```

Diarized response includes `labeled_text` (`Speaker 1: ...\nSpeaker 2: ...`), `lines` (merged consecutive same-speaker segments with timestamps), and `speaker_segments` (per-utterance raw segments).

`GET /health` returns model + device info.

## Project layout

```
app/
  main.py                       FastAPI app: /health, /transcribe
  config.py                     .env loader
  diarize.py                    SpeechBrain ECAPA + clustering
  backends/
    base.py                     TranscribeOptions, TranscriptionError
    local_transformers.py       Whisper pipeline (lazy, thread-safe)
scripts/
  check_gpu.py                  Verify torch sees CUDA
  warmup_local.py               Pre-download Whisper weights
requirements.txt
.env.example                    Template — copy to .env
run.py                          uvicorn entrypoint
Dockerfile                      CUDA 12.6 base + torch + app
docker-compose.yml              GPU-enabled compose with persistent model volumes
.dockerignore
```
