import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    model_id: str
    device: str
    hf_token: str | None
    language: str | None
    prompt: str | None
    default_num_speakers: int | None
    host: str
    port: int


def load_settings() -> Settings:
    device = os.getenv("DEVICE", "auto").strip().lower()
    if device not in {"auto", "cuda", "cpu"}:
        raise RuntimeError(f"DEVICE must be 'auto', 'cuda', or 'cpu' — got '{device}'.")

    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    raw_n = os.getenv("DEFAULT_NUM_SPEAKERS", "").strip()
    default_num_speakers = int(raw_n) if raw_n else None

    return Settings(
        model_id=os.getenv("MODEL_ID", "openai/whisper-large-v3-turbo").strip(),
        device=device,
        hf_token=hf_token,
        language=os.getenv("TRANSCRIBE_LANGUAGE", "arabic").strip() or None,
        prompt=os.getenv("TRANSCRIBE_PROMPT", "").strip() or None,
        default_num_speakers=default_num_speakers,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
