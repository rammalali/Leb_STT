"""Pre-download model weights and build pipelines once.

Downloads:
  - Whisper Large v3 Turbo (~1.6 GB) into the HF cache
  - SpeechBrain ECAPA-TDNN (~83 MB) into pretrained_models/

After this runs, the first /transcribe call to a freshly-started server is fast.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backends.local_transformers import LocalTransformersBackend  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.diarize import Diarizer  # noqa: E402


def main() -> int:
    settings = load_settings()

    print(f"Model:  {settings.model_id}")
    print(f"Device: {settings.device} (resolves to cuda when available)")
    print()

    print("[1/2] Loading Whisper pipeline (downloads weights on first run)...")
    t0 = time.time()
    backend = LocalTransformersBackend(settings)
    backend._ensure_pipeline()
    print(
        f"  Loaded on '{backend._device}' (dtype={backend._dtype_repr}) "
        f"in {time.time() - t0:.1f}s"
    )

    print("[2/2] Loading SpeechBrain ECAPA encoder for diarization...")
    t1 = time.time()
    diarizer = Diarizer(settings)
    diarizer.ensure_loaded()
    print(f"  Loaded on '{diarizer._device}' in {time.time() - t1:.1f}s")

    if backend._device != "cuda":
        print()
        print("WARNING: pipeline is running on CPU. Run scripts/check_gpu.py to diagnose.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
