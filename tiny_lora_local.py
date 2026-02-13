"""

- Consolidated duplicate generation blocks.
- Added a proper main() entry point and clearer device handling.
- Added small safety defaults for CPU training.

How to run:
1) Create & activate venv
   py -m venv .venv
  # .\.venv\Scripts\Activate.ps1

2) Install deps
   python -m pip install -U pip
   python -m pip install -r requirements.txt

3) Run
   python tiny_lora_local.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class Config:
    model_name: str = "distilgpt2"
    output_dir: str = "tiny_lora_out"
    max_length: int = 128
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-4
    seed: int = 42


def build_dataset() -> Dataset:
    data: List[Dict[str, str]] = [
        {
            "text": "Question: What is bank reconciliation?\n"
                    "Answer: It matches bank statement entries with ledger entries to identify differences.\n"
        },
        {
            "text": "Question: What is DSO?\n"
                    "Answer: Days Sales Outstanding measures average days to collect receivables.\n"
        },
        {
            "text": "Question: What is maker-checker control?\n"
                    "Answer: One user prepares a transaction and another approves it to reduce fraud risk.\n"
        },
    ]
    return Dataset.from_list(data)


def tokenize_fn(tokenizer: AutoTokenizer, max_length: int):
    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return _tok


def main():
    cfg = Config()

    # Determinism-ish (best effort)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = build_dataset()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # distilgpt2 has no pad token by default
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)

    # LoRA config - distilgpt2 attention projection is "c_attn"
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
    )

    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    ds_tok = ds.map(tokenize_fn(tokenizer, cfg.max_length), batched=True, remove_columns=["text"])

    # If CPU, training can be slow. Keep it simple.
    fp16 = bool(torch.cuda.is_available())

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        fp16=fp16,
        seed=cfg.seed,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds_tok)
    trainer.train()

    prompt = "Question: What is maker-checker control?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Use sampling for more natural answers
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.2,
    )

    print("\n--- Generated ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Save LoRA adapter weights (optional but useful)
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"\nSaved adapter + tokenizer to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
