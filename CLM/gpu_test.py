import torch

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")


if torch.backends.mps.is_available():
    print("GPU is available")

# Check if MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Create random tensors
a = torch.rand(5000, 5000, device=device)
b = torch.rand(5000, 5000, device=device)

c = torch.matmul(a, b)
print("Matrix multiplication result:", c)

# Verify that the operation is performed on the GPU
print("Performed on MPS:", c.is_mps)
# Performed on MPS: True (Output)




