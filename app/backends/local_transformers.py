import asyncio
import tempfile
import threading
from pathlib import Path
from typing import Any

from ..config import Settings
from .base import TranscribeOptions, TranscriptionError


class LocalTransformersBackend:
    name = "local"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._pipeline: Any = None
        self._device: str = "cpu"
        self._dtype_repr: str = "float32"
        self._lock = threading.Lock()

    def _build_pipeline(self) -> Any:
        try:
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore
        except ImportError as e:
            raise TranscriptionError(
                500,
                "Backend requires torch + transformers. "
                "Install with: pip install -r requirements.txt",
            ) from e

        requested = self._settings.device
        if requested == "auto":
            use_cuda = torch.cuda.is_available()
        elif requested == "cuda":
            use_cuda = True
        else:
            use_cuda = False

        device = 0 if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32
        self._device = "cuda" if use_cuda else "cpu"
        self._dtype_repr = "float16" if use_cuda else "float32"

        kwargs: dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": self._settings.model_id,
            "torch_dtype": dtype,
            "device": device,
            "chunk_length_s": 30,
            "return_timestamps": False,
        }
        if self._settings.hf_token:
            kwargs["token"] = self._settings.hf_token
        return pipeline(**kwargs)

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is None:
                self._pipeline = self._build_pipeline()
        return self._pipeline

    def _run_sync(
        self,
        path: str,
        language: str | None,
        prompt: str | None,
        word_timestamps: bool = False,
    ) -> dict:
        pipe = self._ensure_pipeline()
        generate_kwargs: dict[str, Any] = {}
        if language:
            generate_kwargs["language"] = language
            generate_kwargs["task"] = "transcribe"
        if prompt:
            try:
                prompt_ids = pipe.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
                if hasattr(pipe, "device") and prompt_ids is not None:
                    prompt_ids = prompt_ids.to(pipe.device)
                generate_kwargs["prompt_ids"] = prompt_ids
            except Exception as e:
                raise TranscriptionError(400, f"Failed to encode prompt: {e}") from e

        try:
            import librosa  # type: ignore
        except ImportError as e:
            raise TranscriptionError(500, "librosa is required to decode audio.") from e

        try:
            target_sr = pipe.feature_extractor.sampling_rate
            audio_array, sr = librosa.load(path, sr=target_sr, mono=True)
        except Exception as e:
            raise TranscriptionError(400, f"Could not decode audio file: {e}") from e

        call_kwargs: dict[str, Any] = {"generate_kwargs": generate_kwargs}
        if word_timestamps:
            call_kwargs["return_timestamps"] = "word"

        out = pipe({"array": audio_array, "sampling_rate": sr}, **call_kwargs)
        if not isinstance(out, dict):
            return {"text": str(out), "chunks": []}
        return {"text": out.get("text", ""), "chunks": out.get("chunks", [])}

    async def transcribe(self, audio: bytes, filename: str, options: TranscribeOptions) -> dict:
        suffix = Path(filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio)
            tmp_path = tmp.name

        try:
            try:
                result = await asyncio.to_thread(
                    self._run_sync, tmp_path, options.language, options.prompt
                )
            except TranscriptionError:
                raise
            except Exception as e:
                raise TranscriptionError(500, f"Transcription failed: {e}") from e
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return {
            "text": result["text"],
            "model": self._settings.model_id,
            "device": self._device,
            "dtype": self._dtype_repr,
        }

    async def transcribe_with_timestamps(
        self, path: str, options: TranscribeOptions
    ) -> dict:
        try:
            result = await asyncio.to_thread(
                self._run_sync, path, options.language, options.prompt, True
            )
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(500, f"Transcription failed: {e}") from e
        return {
            "text": result["text"],
            "chunks": result["chunks"],
            "model": self._settings.model_id,
            "device": self._device,
            "dtype": self._dtype_repr,
        }

    def info(self) -> dict:
        return {
            "model": self._settings.model_id,
            "configured_device": self._settings.device,
            "loaded": self._pipeline is not None,
            "language": self._settings.language,
            "prompt": self._settings.prompt,
        }
