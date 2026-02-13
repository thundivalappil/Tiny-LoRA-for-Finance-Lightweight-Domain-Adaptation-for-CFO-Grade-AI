# Tiny LoRA (Local)

This is a local-friendly version of the Colab notebook.

## Setup (Windows / VS Code)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python tiny_lora_local.py
```

## Notes

- `bitsandbytes` is not included because it commonly fails on native Windows.
  If you need 4-bit/8-bit, run on **WSL2 Ubuntu** or **Linux/Colab** with CUDA.

- Output artifacts (LoRA adapter weights + tokenizer) are saved to `tiny_lora_out/`.
