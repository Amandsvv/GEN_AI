# LAB 1: Variational Autoencoder (VAE) on MNIST

**Name:** Aman Kumar  
**Roll Number:** 2501940051

## Overview
This project implements a Variational Autoencoder (VAE) in PyTorch to encode, reconstruct, and generate handwritten digits from the MNIST dataset, and visualizes the learned 2D latent space.

## Files in This Folder
- `lab1_vae.ipynb` — Jupyter notebook with step-by-step pointwise explanations, PyTorch VAE implementation, training loop, and saved outputs.
- `LAB 1.pdf` — Lab report containing code screenshots, generated digit grids, latent scatter plot, loss table, and pointwise explanations.

## Code Walkthrough (Pointwise)
1. **Setup & Reproducibility:** Fixes `SEED = 42`, selects `cuda` if available (else `cpu`), and creates `lab_outputs/`.
2. **Dataset Loading:** Downloads MNIST (`60,000` training / `10,000` test images, `28 x 28` grayscale), scales pixels to `[0.0, 1.0]` with `ToTensor()`, and batches training data (`batch_size=128`, `shuffle=True`).
3. **VAE Architecture:**
   - **Encoder:** `Flatten()` → `Linear(784, 256)` → `ReLU` → two heads (`mean` and `logvar`, each `Linear(256, 2)`).
   - **Reparameterization Trick:** Computes `z = mu + exp(0.5 * logvar) * eps` so sampling remains differentiable.
   - **Decoder:** `Linear(2, 256)` → `ReLU` → `Linear(256, 784)` → `Sigmoid`.
4. **Training (5 Epochs, Adam `lr=0.001`):** Minimizes per-batch `(binary_cross_entropy + KL_divergence) / batch_size`. Recorded per-image losses:
   - Epoch 1: Reconstruction `188.50`, KL `6.67`
   - Epoch 3: Reconstruction `159.62`, KL `5.38`
   - Epoch 5: Reconstruction `155.01`, KL `5.53`
5. **Visualization & Outputs:** Saves `vae_mnist.pt`, `vae_reconstructed.png`, `vae_generated.png` (16 random samples), and `vae_latent.png` (1,024 test digits plotted in 2D latent space) inside `lab_outputs/`.

## How to Run
Open `lab1_vae.ipynb` in Google Colab or Jupyter (with `torch`, `torchvision`, and `matplotlib` installed) and run all cells sequentially.
