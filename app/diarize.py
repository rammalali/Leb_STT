import threading
from typing import Any

from .config import Settings


class DiarizationError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


class Diarizer:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._pipeline: Any = None
        self._device: str = "cpu"
        self._lock = threading.Lock()

    def _build(self) -> Any:
        try:
            import torch  # type: ignore
            from pyannote.audio import Pipeline  # type: ignore
        except ImportError as e:
            raise DiarizationError(
                500,
                "pyannote.audio is not installed. Run: pip install -r requirements.txt",
            ) from e

        if not self._settings.hf_token:
            raise DiarizationError(
                400,
                "Diarization requires HF_TOKEN. Accept the model terms on HF and set HF_TOKEN in .env.",
            )

        try:
            pipe = Pipeline.from_pretrained(
                self._settings.diarization_model,
                use_auth_token=self._settings.hf_token,
            )
        except Exception as e:
            raise DiarizationError(
                500,
                f"Could not load diarization model '{self._settings.diarization_model}'. "
                "Did you accept the gated-model terms on huggingface.co? "
                f"Underlying error: {e}",
            ) from e

        if pipe is None:
            raise DiarizationError(
                500,
                f"Pipeline.from_pretrained returned None for '{self._settings.diarization_model}'. "
                "Most common cause: HF_TOKEN doesn't have access to the gated repo.",
            )

        requested = self._settings.device
        if requested == "auto":
            use_cuda = torch.cuda.is_available()
        elif requested == "cuda":
            use_cuda = True
        else:
            use_cuda = False

        if use_cuda:
            pipe.to(torch.device("cuda"))
            self._device = "cuda"
        else:
            self._device = "cpu"

        return pipe

    def ensure_loaded(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is None:
                self._pipeline = self._build()
        return self._pipeline

    def diarize(
        self,
        audio_path: str,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[dict]:
        pipe = self.ensure_loaded()

        kwargs: dict[str, Any] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

        try:
            diarization = pipe(audio_path, **kwargs)
        except Exception as e:
            raise DiarizationError(500, f"Diarization failed: {e}") from e

        segments: list[dict] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
            )
        segments.sort(key=lambda s: s["start"])
        return segments

    def info(self) -> dict:
        return {
            "diarization_model": self._settings.diarization_model,
            "loaded": self._pipeline is not None,
            "device": self._device if self._pipeline is not None else None,
            "default_num_speakers": self._settings.default_num_speakers,
        }


def stitch(diarization: list[dict], whisper_chunks: list[dict]) -> list[dict]:
    """Assign each Whisper word/chunk to the diarization segment containing its midpoint,
    then group consecutive same-speaker chunks into one line."""
    lines: list[dict] = []

    for chunk in whisper_chunks:
        timestamp = chunk.get("timestamp")
        if not timestamp or timestamp[0] is None:
            continue
        word_start = float(timestamp[0])
        word_end = float(timestamp[1]) if timestamp[1] is not None else word_start
        midpoint = (word_start + word_end) / 2.0

        speaker = "UNKNOWN"
        best_overlap = 0.0
        for seg in diarization:
            if seg["start"] <= midpoint < seg["end"]:
                speaker = seg["speaker"]
                break
            overlap = min(seg["end"], word_end) - max(seg["start"], word_start)
            if overlap > best_overlap:
                best_overlap = overlap
                speaker = seg["speaker"]

        text = chunk.get("text", "")
        if lines and lines[-1]["speaker"] == speaker:
            lines[-1]["text"] += text
            lines[-1]["end"] = word_end
        else:
            lines.append({"speaker": speaker, "start": word_start, "end": word_end, "text": text})

    for line in lines:
        line["text"] = line["text"].strip()
    return [line for line in lines if line["text"]]


def relabel(lines: list[dict]) -> list[dict]:
    """Rewrite raw pyannote labels (SPEAKER_00, SPEAKER_01, ...) to Speaker 1/2/...
    in order of first appearance."""
    mapping: dict[str, str] = {}
    out = []
    for line in lines:
        raw = line["speaker"]
        if raw not in mapping:
            mapping[raw] = f"Speaker {len(mapping) + 1}"
        out.append({**line, "speaker": mapping[raw]})
    return out


def render_text(lines: list[dict]) -> str:
    return "\n".join(f"{line['speaker']}: {line['text']}" for line in lines)
