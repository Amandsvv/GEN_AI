"""
Study Tip Generator — Streamlit Web App

Step-by-step overview:
1. Configuration & Model Path:
   - Expects the fine-tuned GPT-2 model folder at `lab_outputs/study_tip_gpt2/`
     (exported by the Capstone training notebook).
2. Cached Model Loading (`load_model`):
   - Uses `@st.cache_resource` so the tokenizer and model are loaded into memory
     once and reused across UI interactions instead of reloading on every click.
   - Sets the model to `.eval()` mode to disable dropout during inference.
3. Input Validation:
   - Verifies that `lab_outputs/study_tip_gpt2/config.json` exists; if missing,
     shows an error message and halts execution with `st.stop()`.
   - Collects a study topic from the user (up to 120 characters) and ensures it
     is non-empty when "Generate" is clicked.
4. Prompt Formatting & Deterministic Generation:
   - Formats the prompt as `"Topic: <topic>\\nStudy tip:"` to match the training format.
   - Runs `model.generate(..., do_sample=False, max_new_tokens=45)` inside
     `torch.inference_mode()` for fast, deterministic (greedy) decoding without
     gradient tracking.
5. Output Decoding & Display:
   - Slices off the input prompt tokens (`output[0, inputs.input_ids.shape[1]:]`)
     so only newly generated tip tokens are decoded and shown to the user.
"""

# Run locally: streamlit run app.py
# Folder layout:
#   app.py
#   lab_outputs/study_tip_gpt2/   (model folder exported from the capstone notebook)
from pathlib import Path
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@st.cache_resource
def load_model(folder):
    # 1. Load the saved tokenizer and fine-tuned causal language model in eval mode
    tok = AutoTokenizer.from_pretrained(folder)
    net = AutoModelForCausalLM.from_pretrained(folder).eval()
    return tok, net


# 2. Render the Streamlit page header and check that the exported model exists
st.title("Study Tip Generator")
st.caption("AI-generated study tips for instructor review.")
folder = Path("lab_outputs/study_tip_gpt2")
if not (folder / "config.json").exists():
    st.error("Run the fine-tuning step and place the exported model beside this app.")
    st.stop()

# 3. Read the user's topic and generate a study tip when the button is clicked
topic = st.text_input("Enter a study topic", max_chars=120)
if st.button("Generate"):
    if not topic.strip():
        st.warning("Enter a study topic first.")
        st.stop()

    tokenizer, model = load_model(str(folder))
    # Match the exact prompt structure used during fine-tuning
    inputs = tokenizer("Topic: " + topic.strip() + "\nStudy tip:",
                       return_tensors="pt")
    # Disable gradient computation and use greedy decoding (do_sample=False)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=45,
                                pad_token_id=tokenizer.eos_token_id)

    # Decode only the newly generated tokens (excluding the prompt prefix)
    tip = tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                           skip_special_tokens=True).strip()
    st.write(tip or "No text was generated. Try a more specific topic.")
    st.caption("AI-generated text. Review before sharing with students.")
