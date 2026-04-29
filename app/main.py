import asyncio
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .backends import ALLOWED_EXTS, LocalTransformersBackend, TranscribeOptions, TranscriptionError
from .config import load_settings
from .diarize import DiarizationError, Diarizer, relabel, render_text, stitch

settings = load_settings()
backend = LocalTransformersBackend(settings)
diarizer = Diarizer(settings)

app = FastAPI(title="Leb_STT", version="0.5.0")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "transcription": backend.info(),
        "diarization": diarizer.info(),
    }


def _validate_audio(file: UploadFile) -> tuple[str, str]:
    filename = file.filename or "audio"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '.{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
        )
    return filename, ext


def _parse_int(value: str | None, field: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        raise HTTPException(400, f"'{field}' must be an integer.")


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    num_speakers: str | None = Form(default=None),
    min_speakers: str | None = Form(default=None),
    max_speakers: str | None = Form(default=None),
) -> JSONResponse:
    filename, _ = _validate_audio(file)
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty file.")

    options = TranscribeOptions(
        language=(language.strip() if language else None) or settings.language,
        prompt=(prompt.strip() if prompt else None) or settings.prompt,
    )

    n = _parse_int(num_speakers, "num_speakers")
    if n is None:
        n = settings.default_num_speakers
    mn = _parse_int(min_speakers, "min_speakers")
    mx = _parse_int(max_speakers, "max_speakers")

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
            asr_task = backend.transcribe_with_timestamps(tmp_path, options)
            diarize_task = asyncio.to_thread(
                diarizer.diarize, tmp_path, n, mn, mx
            )
            asr_result, segments = await asyncio.gather(asr_task, diarize_task)
        except TranscriptionError as e:
            raise HTTPException(status_code=e.status, detail=e.detail)
        except DiarizationError as e:
            raise HTTPException(status_code=e.status, detail=e.detail)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    lines = relabel(stitch(segments, asr_result["chunks"]))

    return JSONResponse(
        {
            "text": asr_result["text"],
            "labeled_text": render_text(lines),
            "lines": lines,
            "speaker_segments": segments,
            "model": asr_result["model"],
            "device": asr_result["device"],
            "diarization_model": settings.diarization_model,
        }
    )
