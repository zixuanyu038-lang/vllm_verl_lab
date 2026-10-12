"""Full-parameter, response-only SFT. Launch with torchrun, not vLLM."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class AnswerDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.samples = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prefix = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(
                row["prompt"] + [{"role": "assistant", "content": row["answer"]}],
                tokenize=True, add_generation_prompt=False)
            if full[:len(prefix)] != prefix:
                raise ValueError("Chat template prefix mismatch; do not guess the label boundary")
            # Qwen's completed chat contains the assistant end-of-turn token.
            if len(full) > max_length:
                raise ValueError(f"Sample {row['id']} exceeds {max_length}; refusing silent truncation")
            labels = [-100] * len(prefix) + full[len(prefix):]
            if not any(x != -100 for x in labels):
                raise ValueError("No supervised answer tokens")
            self.samples.append({"input_ids": full, "labels": labels})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class AnswerCollator:
    def __init__(self, pad_token_id):
        self.pad = pad_token_id

    def __call__(self, examples):
        width = max(len(x["input_ids"]) for x in examples)
        return {
            "input_ids": torch.tensor([x["input_ids"] + [self.pad] * (width - len(x["input_ids"])) for x in examples]),
            "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (width - len(x["input_ids"])) for x in examples]),
            "labels": torch.tensor([x["labels"] + [-100] * (width - len(x["labels"])) for x in examples]),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--out", default="runs/sft")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    settings = TrainingArguments(
        output_dir=args.out, max_steps=args.steps,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=5e-5, warmup_steps=2, weight_decay=0.01,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, logging_steps=5,
        save_strategy="no", report_to="none", seed=20261012,
        data_seed=20261012, dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=settings,
                      train_dataset=AnswerDataset(args.data, tokenizer),
                      data_collator=AnswerCollator(tokenizer.pad_token_id))
    trainer.train()
    model.config.use_cache = True
    trainer.save_model(args.out)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    main()
