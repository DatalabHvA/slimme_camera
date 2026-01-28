import torch

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(0))

        # Simple tensor test on GPU
        x = torch.rand((1000, 1000), device="cuda")
        y = torch.mm(x, x)
        print("GPU tensor test: SUCCESS")
    else:
        print("GPU tensor test: FAILED (CPU only)")

if __name__ == "__main__":
    main()
