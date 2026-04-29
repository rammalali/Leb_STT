"""Pre-download Whisper weights and build the local pipeline once.

Reads LOCAL_MODEL / LOCAL_DEVICE / TRANSCRIBE_LANGUAGE from .env (same as the API server).
After this runs, the first /transcribe call to the running server is fast.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings
from app.backends.local_transformers import LocalTransformersBackend


def main() -> int:
    settings = load_settings()
    backend = LocalTransformersBackend(settings)

    print(f"Model:  {settings.model_id}")
    print(f"Device: {settings.device} (will resolve to cuda if available)")
    print("Loading pipeline (this downloads weights on first run)...")
    t0 = time.time()
    backend._ensure_pipeline()
    print(f"Loaded on '{backend._device}' (dtype={backend._dtype_repr}) in {time.time() - t0:.1f}s")

    if backend._device != "cuda":
        print()
        print("WARNING: pipeline is running on CPU. Run scripts/check_gpu.py to diagnose.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
