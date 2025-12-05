import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from recipe.system_prompt import SYSTEM_PROMPT

def extract_vectors(args):
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    df = df.dropna(subset=["response"])
    
    # group by pair_id to get (abstain, non-abstain) pairs
    pairs = []
    grouped = df.groupby("pair_id")
    
    for _, group in grouped:
        if len(group) != 2:
            continue
            
        abstain_row = group[group["did_abstain"] == True]
        non_abstain_row = group[group["did_abstain"] == False]
        
        if len(abstain_row) == 1 and len(non_abstain_row) == 1:
            pairs.append({
                "question": abstain_row.iloc[0]["question"],
                "abstain_response": abstain_row.iloc[0]["response"],
                "non_abstain_response": non_abstain_row.iloc[0]["response"]
            })

    print(f"Found {len(pairs)} valid pairs.")
    
    if args.max_pairs is not None and len(pairs) > args.max_pairs:
        print(f"Limiting to {args.max_pairs} pairs.")
        pairs = pairs[:args.max_pairs]
    
    diff_vectors = []

    for pair in tqdm(pairs, desc="Extracting vectors"):
        question = pair["question"]
        
        responses = [pair["abstain_response"], pair["non_abstain_response"]]
        
        # format inputs
        prompt_str = question_to_chat_format(
            use_system_prompt=args.use_system_prompt,
            question=question,
            tokenizer=tokenizer
        )
        
        # tokenize prompt separately to know where response starts
        prompt_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(model.device)
        prompt_len = prompt_ids.shape[1]
        
        # run forward pass for both responses
        # calculate: mean(act_abstain) - mean(act_non_abstain)
        activations = {}
        
        for i, response in enumerate(responses):    
            label = "abstain" if i == 0 else "non_abstain"
            full_text = prompt_str + response
            inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(inputs.input_ids, output_hidden_states=True)
            
            hidden_state = outputs.hidden_states[args.layer_idx + 1]
            
            # extract only the response tokens (exclude prompt)
            # shape: [1, seq_len, hidden_dim]
            response_acts = hidden_state[:, prompt_len:, :]
            
            # use only the first 10 tokens, or the length of the response, whichever is shorter
            num_tokens = response_acts.shape[1]
            slice_len = min(num_tokens, 10)
            
            # sanity check
            if i == 0:
                sliced_ids = inputs.input_ids[0, prompt_len : prompt_len+slice_len]
                print(f"DEBUG - Actual tokens being averaged: {tokenizer.decode(sliced_ids)}")
                
            mean_act = response_acts[:, :slice_len, :].mean(dim=1).squeeze()
            
            activations[label] = mean_act

        # compute difference
        diff = activations["abstain"] - activations["non_abstain"]
        diff_vectors.append(diff)

    # average over dataset
    if not diff_vectors:
        print("No vectors extracted.")
        return

    stacked_diffs = torch.stack(diff_vectors)
    mean_steering_vector = stacked_diffs.mean(dim=0)
    
    # save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(mean_steering_vector, args.output_path)
    print(f"Saved steering vector for layer {args.layer_idx} to {args.output_path}")

def question_to_chat_format(
    use_system_prompt: bool, 
    question: str, 
    tokenizer: AutoTokenizer
) -> str:
    # Tokenization and chat format inspired by the recipe
    # from https://huggingface.co/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8
    if use_system_prompt:
        prompt = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}]
    else:
        prompt = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="HF model name or path")
    parser.add_argument("--data_path", type=str, required=True, help="Path to pairs CSV")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save .pt vector")
    parser.add_argument("--layer_idx", type=int, required=True, help="Layer index to extract from")
    parser.add_argument("--use_system_prompt", action="store_true")
    parser.add_argument("--max_pairs", type=int, default=None, help="Max pairs to process")
       
    args = parser.parse_args()
    extract_vectors(args)