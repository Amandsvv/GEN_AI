# LAB 3: GPT-2 Text Decoding & Lightweight Fine-Tuning

**Name:** Aman Kumar  
**Roll Number:** 2501940051

## Overview
This project explores decoding strategies (greedy vs. nucleus sampling with temperature and seed control) on pretrained `openai-community/gpt2`, followed by lightweight last-block fine-tuning on study tips and held-out evaluation.

## Files in This Folder
- `lab3_gpt2.ipynb` — Jupyter notebook with pointwise explanations, decoding experiments, selective fine-tuning, and held-out evaluation.
- `LAB 3.pdf` — Lab report containing code screenshots, sample continuations, fine-tuning details, and pointwise explanations.

## Code Walkthrough (Pointwise)
1. **Dependencies & Model Loading:** Installs `transformers>=4.51,<5`, sets `SEED = 42`, loads `openai-community/gpt2` and its tokenizer, sets `tokenizer.pad_token = tokenizer.eos_token`, and switches to `model.eval()`.
2. **Decoding Comparison:** Runs `continue_text(..., max_new_tokens=45)` across 2 story prompts and 5 settings each (Greedy, plus `top_p=0.9` sampling at temperatures `0.6` and `1.0` with seeds `42` and `43`), exporting all 10 results to `lab_outputs/gpt2_decoding_outputs.csv`.
3. **Last-Block Fine-Tuning:** Freezes all GPT-2 parameters except the final Transformer block (`tuned.transformer.h[-1]`), masks prompt tokens in `labels` with `-100` so loss applies only to study-tip tokens, trains for 3 epochs (`AdamW`, `lr=2e-5`) on 6 `(topic, tip)` pairs, and saves to `lab_outputs/study_tip_gpt2/`.
4. **Held-Out Evaluation:** Compares `base_model` vs. `tuned` using greedy decoding on 3 unseen topics (`recursion`, `matrices`, `binary search`) and saves `lab_outputs/gpt2_heldout_outputs.csv`.

## How to Run
Open `lab3_gpt2.ipynb` in Google Colab (GPU recommended) and execute all cells top to bottom.
