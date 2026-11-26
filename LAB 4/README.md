# LAB 4: Text-to-Image Generation with Stable Diffusion v1.5

**Name:** Aman Kumar  
**Roll Number:** 2501940051

## Overview
This project uses `diffusers` and `stable-diffusion-v1-5/stable-diffusion-v1-5` on GPU to run a controlled comparison of prompt style modifiers (`watercolor illustration` vs. `pencil sketch`) across fixed random seeds.

## Files in This Folder
- `lab4_stable_diffusion.ipynb` — Jupyter notebook with pointwise markdown explanations, pipeline setup, image generation loop, and 2×2 grid comparison.
- `LAB 4.pdf` — Lab report with code screenshots, generated library images, 2×2 comparison grid, and pointwise explanations.

## Code Walkthrough (Pointwise)
1. **Dependencies & GPU Setup:** Installs `diffusers>=0.33,<1`, `transformers>=4.51,<5`, `accelerate`, and `safetensors`; sets `SEED = 42` and asserts `device.type == "cuda"`.
2. **Memory-Efficient Pipeline:** Loads Stable Diffusion v1.5 in half precision (`torch.float16`), moves it to CUDA, and enables `pipe.enable_attention_slicing()`.
3. **Controlled Generation Loop:** Generates `512 x 512` images (`num_inference_steps=25`, `guidance_scale=7.5`) for two prompts sharing the same scene (`"A small library with a reading table, wide view"`) across fixed seeds `42` and `43`, checking `result.nsfw_content_detected` before saving.
4. **2×2 Comparison Grid:** Combines all 4 generated PNGs (`library_prompt_{i}_seed_{seed}.png`) into `lab_outputs/library_comparison.png` to isolate style vs. layout/seed differences.

## How to Run
Open `lab4_stable_diffusion.ipynb` in Google Colab with a **GPU runtime** enabled and run all cells sequentially.
