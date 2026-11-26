# LAB 2: Generative Adversarial Network (GAN) on Fashion-MNIST

**Name:** Aman Kumar  
**Roll Number:** 2501940051

## Overview
This project builds a fully connected Generative Adversarial Network (GAN) in PyTorch to synthesize `28 x 28` grayscale clothing images from the Fashion-MNIST dataset.

## Files in This Folder
- `lab2_gan.ipynb` — Jupyter notebook with pointwise markdown explanations, Generator/Discriminator definitions, alternating training loop, and outputs.
- `LAB 2.pdf` — Lab report with code screenshots, sample grids from Epochs 1, 5, and 10, loss curve, observation table, and pointwise explanations.

## Code Walkthrough (Pointwise)
1. **Setup & Data Scaling:** Fixes `SEED = 42` and normalizes `60,000` Fashion-MNIST training images from `[0, 255]` to `[-1, 1]` using `Normalize((0.5,), (0.5,))` to match the Generator's `Tanh()` output range (`batch_size=128`).
2. **Networks & Optimizers:**
   - **Generator (`G`):** `64`-dim noise → `256` → `512` (`LeakyReLU(0.2)`) → `784` (`Tanh`).
   - **Discriminator (`D`):** `Flatten()` → `784` → `512` → `256` (`LeakyReLU(0.2)`) → `1` raw logit.
   - **Loss & Optimizers:** `BCEWithLogitsLoss()` with `Adam(lr=0.0002, betas=(0.5, 0.999))` for both `G` and `D`, plus `25` fixed noise vectors (`fixed_z`) for consistent visual tracking.
3. **Alternating Training (`train_batch`):**
   - **D step:** Trains `D` on real images (`1`) and `fake.detach()` (`0`) so gradients do not enter `G`.
   - **G step:** Temporarily freezes `D` (`D.requires_grad_(False)`) and updates `G` so `D(fake)` predicts `1`.
4. **10-Epoch Results:**
   - Epoch 1: `D=0.350`, `G=1.823`
   - Epoch 5: `D=0.406`, `G=2.082`
   - Epoch 10: `D=0.461`, `G=1.743`
5. **Saved Artifacts:** Writes `gan_epoch_01.png`, `gan_epoch_05.png`, `gan_epoch_10.png`, `gan_losses.png`, and `fashion_generator.pt` to `lab_outputs/`.

## How to Run
Open `lab2_gan.ipynb` in Google Colab or Jupyter (with `torch`, `torchvision`, and `matplotlib` installed) and run all cells in order.
