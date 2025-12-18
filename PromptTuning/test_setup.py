"""
Quick test script to verify everything works before full training.
Run this interactively on a GPU node.

Usage:
    srun --partition=gpu --gres=gpu:1 --mem=16G --time=00:30:00 --pty bash
    python PromptTuning/test_setup.py
"""

import sys
sys.path.append("/mnt/parscratch/users/acb20df/private/TuringProj/AbstentionBench")

DATA_PATH = "PromptTuning/data/sample_pairs.csv"

print("=" * 50)
print("TEST 1: Load training CSV dataset")
print("=" * 50)

try:
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Dataset loaded: {len(df)} samples")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Abstain samples: {df['should_abstain'].sum()}")
    print(f"  Answer samples: {(~df['should_abstain']).sum()}")
    print(f"  Datasets: {df['dataset'].unique().tolist()}")
    
    # Show a few examples
    print("\n  Sample examples:")
    for i in [0, 1, len(df)//2]:
        row = df.iloc[i]
        print(f"\n  [{i}] should_abstain={row['should_abstain']} | dataset={row['dataset']}")
        print(f"      Q: {str(row['question'])[:80]}...")
        
except Exception as e:
    print(f"✗ Failed to load dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("TEST 2: Load Gemma 3 1B tokenizer")
print("=" * 50)

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded successfully")
    
    # Quick tokenization test
    test_text = "What is the capital of France?"
    tokens = tokenizer(test_text)
    print(f"  Test tokenization: '{test_text}' → {len(tokens['input_ids'])} tokens")
    
except Exception as e:
    print(f"✗ Failed to load tokenizer: {e}")
    print("\n  If this is an authentication error, run:")
    print("    huggingface-cli login")
    print("  And accept the license at: https://huggingface.co/google/gemma-3-1b-it")
    sys.exit(1)

print("\n" + "=" * 50)
print("TEST 3: Load Gemma 3 1B model")
print("=" * 50)

try:
    import torch
    from transformers import AutoModelForCausalLM
    
    print("  Loading model (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f"✓ Model loaded successfully")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("TEST 4: Configure PEFT soft prompt")
print("=" * 50)

try:
    from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType
    
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=50,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text="You are a helpful assistant. Question: ",
        tokenizer_name_or_path="google/gemma-3-1b-it",
    )
    
    model = get_peft_model(model, peft_config)
    print("✓ PEFT soft prompt configured")
    model.print_trainable_parameters()
    
except Exception as e:
    print(f"✗ Failed to configure PEFT: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("TEST 5: Quick forward pass")
print("=" * 50)

try:
    test_input = "What is the meaning of life?"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("✓ Forward pass successful")
    print(f"  Input: {test_input}")
    print(f"  Output: {response[:100]}...")
    
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("ALL TESTS PASSED ✓")
print("=" * 50)
print("\nYou're ready to run the full training job:")
print("  sbatch scripts/train_soft_prompt.sh")