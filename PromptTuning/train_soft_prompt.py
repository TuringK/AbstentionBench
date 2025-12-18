#Only Training no Validation
"""
Soft Prompt Tuning for LLM Abstention
=====================================
Trains a soft prompt to teach LLMs when to abstain from answering.

Usage:
    python train_soft_prompt.py \
        --model_name google/gemma-3-1b-it \
        --train_csv data/train.csv \
        --output_dir trained_models/gemma_3_1b_soft_prompt \
        --num_virtual_tokens 50 \
        --num_epochs 3
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train soft prompt for abstention")
    
    # Data paths
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained model")
    
    # Model config
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it", help="HuggingFace model name")
    parser.add_argument("--num_virtual_tokens", type=int, default=50, help="Number of soft prompt tokens")
    
    # Training config
    parser.add_argument("--learning_rate", type=float, default=0.3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


# ============================================================
# Data Loading
# ============================================================

def load_data_from_csv(csv_path: str) -> list[dict]:
    """Load and format data from CSV file."""
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        samples.append({
            "question": row["question"],
            "should_abstain": row["should_abstain"],
        })
    return samples


def format_training_example(sample: dict) -> dict:
    """Format a sample for training with templated responses."""
    question = sample["question"]
    
    if sample["should_abstain"]:
        target = "I cannot answer this question with certainty."
    else:
        target = "I can answer this question."
    
    return {
        "input_text": question,
        "target_text": target,
        "full_text": f"{question}\n\nAnswer: {target}",
        "should_abstain": sample["should_abstain"],
    }


class AbstentionDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_length: int = 512):
        self.samples = [format_training_example(s) for s in samples]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        encodings = self.tokenizer(
            sample["full_text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================
# Training
# ============================================================

def train(args):
    """Main training function."""
    
    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Load tokenizer and model ----
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model_name}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    model.gradient_checkpointing_enable()
    
    # ---- Configure soft prompt ----
    prompt_init_text = (
        "You are a helpful assistant that knows when to abstain. "
        "If you are uncertain or the question cannot be answered, say so. "
        "Question: "
    )
    
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text=prompt_init_text,
        tokenizer_name_or_path=args.model_name,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    device = next(model.parameters()).device
    
    # ---- Load data ----
    print(f"\n{'='*60}")
    print("Loading data...")
    print(f"{'='*60}")
    
    train_samples = load_data_from_csv(args.train_csv)
    print(f"Train samples: {len(train_samples)}")
    
    train_dataset = AbstentionDataset(train_samples, tokenizer, args.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # ---- Setup optimizer and scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )
    
    # ---- Training loop ----
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    
    best_train_loss = float("inf")
    best_epoch = -1
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train loss: {avg_train_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
        model.save_pretrained(checkpoint_dir)
        
        # Track best model using training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            print(f"  New best! Saving to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
    
    # ---- Save training config ----
    config = {
        "model_name": args.model_name,
        "num_virtual_tokens": args.num_virtual_tokens,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "train_samples": len(train_samples),
    }
    
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best model (epoch {best_epoch}, train_loss={best_train_loss:.4f}) saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    train(args)