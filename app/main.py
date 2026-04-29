import asyncio
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .backends import ALLOWED_EXTS, LocalTransformersBackend, TranscribeOptions, TranscriptionError
from .config import load_settings
from .diarize import DiarizationError, Diarizer, relabel, render_text

settings = load_settings()
backend = LocalTransformersBackend(settings)
diarizer = Diarizer(settings)

app = FastAPI(title="Leb_STT", version="0.6.0")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "transcription": backend.info(),
        "diarization": diarizer.info(),
    }


def _validate_audio(file: UploadFile) -> str:
    filename = file.filename or "audio"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '.{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
        )
    return filename


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default="", description="Whisper language hint (e.g. 'arabic'). Empty = auto-detect."),
    prompt: str = Form(default="", description="Decoder prompt. ~224 tokens max."),
    num_speakers: int | None = Form(
        default=None,
        description="Pin speaker count. >=2 enables diarization.",
    ),
    min_speakers: int | None = Form(
        default=None,
        description="Lower bound for speaker count. >=2 enables diarization.",
    ),
    max_speakers: int | None = Form(
        default=None,
        description="Upper bound for speaker count. >=2 enables diarization.",
    ),
) -> JSONResponse:
    filename = _validate_audio(file)
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty file.")

    options = TranscribeOptions(
        language=language.strip() or settings.language,
        prompt=prompt.strip() or settings.prompt,
    )

    n = num_speakers if num_speakers is not None else settings.default_num_speakers
    mn = min_speakers
    mx = max_speakers

    diarize = (
        (n is not None and n >= 2)
        or (mn is not None and mn >= 2)
        or (mx is not None and mx >= 2)
    )

    if not diarize:
        try:
            result = await backend.transcribe(audio, filename, options)
        except TranscriptionError as e:
            raise HTTPException(status_code=e.status, detail=e.detail)
        return JSONResponse(result)

    suffix = Path(filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio)
        tmp_path = tmp.name

    try:
        try:
            asr_result = await backend.transcribe_with_audio(tmp_path, options)
        except TranscriptionError as e:
            raise HTTPException(status_code=e.status, detail=e.detail)

        try:
            lines, raw_segments = await asyncio.to_thread(
                diarizer.diarize_chunks,
                asr_result["audio_array"],
                asr_result["sampling_rate"],
                asr_result["chunks"],
                n,
                mn,
                mx,
            )
        except DiarizationError as e:
            raise HTTPException(status_code=e.status, detail=e.detail)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    labeled_lines = relabel(lines)
    labeled_segments = relabel(raw_segments)

    return JSONResponse(
        {
            "text": asr_result["text"],
            "labeled_text": render_text(labeled_lines),
            "lines": labeled_lines,
            "speaker_segments": labeled_segments,
            "model": asr_result["model"],
            "device": asr_result["device"],
            "diarization_backend": "speechbrain-ecapa",
        }
    )
