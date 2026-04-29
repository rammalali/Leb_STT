"""Verify CUDA is visible to torch. Run before using the backend."""
import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch is not installed. Run: pip install -r requirements.txt")
        return 1

    print(f"torch version:      {torch.__version__}")
    print(f"CUDA available:     {torch.cuda.is_available()}")
    print(f"CUDA build version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"Device count:       {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name}  {props.total_memory / 1024**3:.1f} GiB")
        return 0

    print()
    print("CUDA is NOT available. Likely cause: torch was installed as the CPU build.")
    print("Reinstall the CUDA wheels matching your NVIDIA driver. Examples:")
    print("  CUDA 12.6:  pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio")
    print("  CUDA 12.4:  pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio")
    print("  CUDA 12.1:  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio")
    return 2


if __name__ == "__main__":
    sys.exit(main())
