from dataclasses import dataclass

ALLOWED_EXTS = {"wav", "mp3", "m4a", "mp4a", "flac", "ogg", "webm"}


@dataclass(frozen=True)
class TranscribeOptions:
    language: str | None = None
    prompt: str | None = None


class TranscriptionError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail
