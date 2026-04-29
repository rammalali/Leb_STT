"""Verify CUDA is available to torch. Run before using the local backend."""
import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch is not installed. Run: pip install -r requirements-local.txt")
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
    print("Reinstall torch matching your CUDA driver. Examples:")
    print("  CUDA 12.1:  pip install --index-url https://download.pytorch.org/whl/cu121 torch")
    print("  CUDA 12.4:  pip install --index-url https://download.pytorch.org/whl/cu124 torch")
    return 2


if __name__ == "__main__":
    sys.exit(main())
