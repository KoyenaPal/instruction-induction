from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import re
import torch
import random
import numpy as np
import os
import csv
from tqdm import tqdm
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pandas as pd
import gc
import time
import copy
import json
# === CONFIGURATION ===
SEED = 42  # Or any integer


available_gpus = torch.cuda.device_count()
print(f"Found {available_gpus} GPUs")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (may slow down and not support all ops)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# def load_to_gpu(model):
#     model.to("cuda")
#     torch.cuda.empty_cache()

# def unload_to_cpu(model):
#     model.to("cpu")
#     torch.cuda.empty_cache()

# === LOAD MODELS ===
def load_model_and_tokenizer(name, cache_dir="/workspace/hf", device_map="auto", device_id = None):
    if device_id is not None:
        device_map = f"cuda:{device_id}"
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir, device_map=device_map, offload_folder="offload_dir", dtype=torch.bfloat16)
    model.eval()
    return tokenizer, model


# eval_tokenizers_models = [load_model_and_tokenizer(m) for m in evaluation_models]

# === GENERATE CANDIDATES ===
def truncate_to_last_sentence(text):
    """
    Truncate text to the last full sentence (ending in . ! or ?),
    avoiding decimals, detecting abbreviations dynamically, 
    and handling no-space cases.
    """
    text = text.strip()

    def is_abbreviation(word):
        """Heuristic abbreviation detection."""
        lw = word.lower()
        # Skip decimals
        if re.match(r"\d+\.\d+$", lw):
            return True
        # Single short word ending with period, e.g., "Dr."
        if re.match(r"[A-Za-z]{1,4}\.$", lw):
            return True
        # Multi-part abbreviation like "U.S." or "e.g."
        if re.match(r"(?:[A-Za-z]{1,4}\.){2,}$", lw):
            return True
        return False

    # Regex: punctuation not after digit, followed by space/end/capital letter
    matches = list(re.finditer(
        r'(?<!\d)([.!?])(?=\s|$|[A-Z])',
        text
    ))

    last_good_end = None
    for m in matches:
        end_pos = m.end()
        before = text[:end_pos].strip()
        last_word = before.split()[-1]
        if is_abbreviation(last_word):
            continue
        last_good_end = end_pos

    if last_good_end is None:
        return text
    return text[:last_good_end].strip()

def truncate_to_first_sentence(text):
    """Truncate text to the first sentence (ending in . ! or ?)."""
    match = re.search(r'(.+?[.!?])(\s|$)', text.strip())
    return match.group(1).strip() if match else text.strip()

#prev was 128
def generate_candidates(context, tokenizer, model, num_return_sequences, max_gen_tokens=15):
    set_seed(SEED)
    
    # Tokenize with attention_mask
    model_device = next(model.parameters()).device
    enc = tokenizer(context, return_tensors="pt").to(model_device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # input_ids = input_ids.to(model_device)
    # attention_mask = attention_mask.to(model_device)

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_gen_tokens,
            do_sample=True,
            top_k=50,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )

    candidates = []
    for out in output:
        full_text = tokenizer.decode(out[input_ids.shape[-1]:], skip_special_tokens=True)
        full_text = truncate_to_last_sentence(full_text)
        candidates.append(full_text)
    return candidates


# === COMPUTE PERPLEXITY ===
def compute_perplexities(context, candidates, tokenizer, model):
    """
    Compute perplexity per candidate using model's built-in loss,
    matching the single-candidate version exactly.
    """
    model_device = next(model.parameters()).device
    full_texts = [context + c for c in candidates]

    # Tokenize all at once
    encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model_device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    perplexities = []
    with torch.inference_mode():
        for i in range(len(candidates)):
            input_i = input_ids[i].unsqueeze(0)  # (1, seq_len)
            attn_i = attention_mask[i].unsqueeze(0)
            outputs = model(input_ids=input_i, attention_mask=attn_i, labels=input_i)
            loss = outputs.loss  # averaged over non-masked tokens
            perplexities.append(torch.exp(loss).item())

    return perplexities


def iterative_generate(context, gen_tokenizers_models, eval_tokenizers_models, max_total_tokens=500, num_candidates=3):
    iteration_count = 0
    contexts = {}
    total_tokens = 0
    chosen_per_iteration = []
    gen_names = [name for name, _ in gen_tokenizers_models]
    eval_names = [name for name, _ in eval_tokenizers_models]
    all_tokenizers_models = gen_tokenizers_models + eval_tokenizers_models
    while True:
        start_time = time.time()

        if iteration_count > 0:
            if total_tokens >= max_total_tokens:
                print("\n✅ Reached max length.")
                break
            # assumes DAPO or QwQ would usually be the first one.
            if "</think>" in contexts[gen_names[0]]:
                print("\n✅ Reached end of thought.")
                break

        all_candidates = []
        candidates_per_model = {}

        for model_index, (tokenizer, model) in all_tokenizers_models:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if iteration_count == 0:
                context_in_chat_temp = tokenizer.apply_chat_template(
                    context, tokenize=False, add_generation_prompt=True
                )
                if "qwen" in model_index.lower():
                    context_in_chat_temp = context_in_chat_temp + "<think>"
                if "openthinker" in model_index.lower():
                    context_in_chat_temp = context_in_chat_temp + "<|begin_of_thought|>"
                if "gpt-oss" in model_index.lower():
                    user_message = [{"role": "user", "content": context[-1]["content"]}]
                    context_in_chat_temp = tokenizer.apply_chat_template(user_message,
                                                                              model_identity=context[0]["content"],
                                                                              reasoning_effort = "low",
                                                                              tokenize=False, add_generation_prompt=True)
                max_total_tokens = len(tokenizer.encode(context_in_chat_temp, add_special_tokens=True)) + 2048 - 30
                contexts[model_index] = context_in_chat_temp
                total_tokens = len(tokenizer.encode(contexts[model_index]))

            if model_index in gen_names:
                torch.cuda.empty_cache()
                candidates = generate_candidates(
                    contexts[model_index], tokenizer, model, num_candidates
                )
                print(f"\nMODEL: {model_index}")
                print("CANDIDATES:", candidates)
    
                all_candidates.extend(candidates)
                candidates_per_model[model_index] = candidates
                torch.cuda.empty_cache()

        # === Batched Perplexity Evaluation ===
        scored_candidates_dict = {cand: [] for cand in all_candidates}
        for model_index, (tokenizer, model) in eval_tokenizers_models:
            # === move model to GPU only when needed ===
            # model = model.to("cuda")
            torch.cuda.empty_cache()
            try:
                ppls = compute_perplexities(contexts[model_index], all_candidates, tokenizer, model)
                for i, cand in enumerate(all_candidates):
                    scored_candidates_dict[cand].append(ppls[i])
            except Exception as e:
                print(f"⚠️ Error in batched scoring: {e}")
                continue
            finally:
                torch.cuda.empty_cache()

        # === Aggregate scores ===
        scored_candidates = []
        for cand, ppl_list in scored_candidates_dict.items():
            if ppl_list:
                avg_ppl = sum(ppl_list) / len(ppl_list)
                scored_candidates.append((cand, avg_ppl))

        if not scored_candidates:
            print("❌ No valid candidates scored. Exiting.")
            break

        best_candidate, best_score = min(scored_candidates, key=lambda x: x[1])
        print(f"\n🔹 Selected: {best_candidate.strip()}")

        chosen_model_index = None
        for model_index, cands in candidates_per_model.items():
            if best_candidate in cands:
                chosen_model_index = model_index
                break

        chosen_per_iteration.append({
            "iteration": iteration_count,
            "selected_candidate": best_candidate,
            "selected_model_index": chosen_model_index,
            "all_candidates": copy.deepcopy(candidates_per_model),
            "score": best_score,
        })

        # Append selected to each context
        for key in contexts:
            contexts[key] += " " + best_candidate.strip()

        iteration_count += 1
        elapsed = time.time() - start_time
        print(f"⏱️ Iteration {iteration_count} completed in {elapsed:.2f} seconds.")

    return contexts, chosen_per_iteration



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Context-aware sentence merging using perplexity.")
    # parser.add_argument("jsonl_files", nargs='+', help="Paths to input JSONL files.")
    parser.add_argument("--output", type=str, default="instruction_induction_ensemble_outputs_gen_qwq_dapo_eval_oss", help="Path to save merged output.")
    parser.add_argument("--gen_models", nargs='+', default=[
        "Qwen/QwQ-32B",
        "BytedTsinghua-SIA/DAPO-Qwen-32B"
    ], help="Hugging Face model names.")
    parser.add_argument("--eval_models", nargs='+', default=[
        "openai/gpt-oss-20b"
    ], help="Hugging Face model names.")
    set_seed(SEED)
    args = parser.parse_args()
    device = 'auto'
    generation_models = args.gen_models
    evaluation_models = args.eval_models
    #gen_tokenizers_models = eval_tokenizers_models = [load_model_and_tokenizer(m) for m in generation_models]
    # gen_tokenizers_models = [(m, load_model_and_tokenizer(m)) for m in generation_models]
    # eval_tokenizers_models = [(m, load_model_and_tokenizer(m)) for m in evaluation_models]
    gpu_ids = list(range(available_gpus))
    # ASSUMES THAT YOU CAN HAVE 1 GPU FOR EACH MODEL
    gen_gpu_ids = gpu_ids[:len(generation_models)]
    eval_gpu_ids = gpu_ids[len(generation_models):len(generation_models)+len(evaluation_models)]
    # gen_tokenizers_models = [
    #     (m, load_model_and_tokenizer(m, device_id = i % available_gpus))
    #     for i, m in enumerate(generation_models)]
    # eval_tokenizers_models = [
    #     (m, load_model_and_tokenizer(m, device_id = i % available_gpus))
        # for i, m in enumerate(evaluation_models)]
    # Load generation models
    gen_tokenizers_models = [
        (m, load_model_and_tokenizer(m, device_id=gpu_id))
        for m, gpu_id in zip(generation_models, gen_gpu_ids)
    ]
    
    # Load evaluation models
    eval_tokenizers_models = [
        (m, load_model_and_tokenizer(m, device_id=gpu_id))
        for m, gpu_id in zip(evaluation_models, eval_gpu_ids)
    ]
# Load input examples
    input_dir = "data/induction_input"
    instruction_generation_model = "."
    INDUCTION_TASKS = ['cause_and_effect', 'larger_animal', 'num_to_verbal','orthography_starts_with',
                       'rhymes', 'synonyms', 'taxonomy_animal', 'translation_en-fr',
                       'reverse_from_middle', 'smallest_item_length', 'smallest_even_no_sqrt', 'most_vowel_return_consonant',
                       'detect_rhyme_and_rewrite', 'rank_by_protein','multi_lang_to_english','square_of_zodiac_animal',
                       'alternate_synonym_antonym', 'most_consonant_return_vowel', 'least_unique_word_count', 'first_word_alphabetically_return_reverse']
    for task_name in INDUCTION_TASKS: 
        output_file_path = args.output + f"/{task_name}.csv"
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        write_header = not os.path.isfile(output_file_path)
    # task_name = "sum"
        with open(f'{input_dir}/{instruction_generation_model}/{task_name}.json', encoding='utf-8') as f_examples:
            data = json.load(f_examples)
        data = data["examples"]
        # Set seed and sample 5 items
        random.seed(42)
        # Sample 5 keys from the dictionary
        sampled_keys = random.sample(list(data.keys()), 5)
        
        # Sort the sampled keys numerically (same as your original sorting)
        sampled_keys = sorted(sampled_keys, key=lambda x: int(x))
        
        with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Instruction", "Ensembled Thought"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        
            if write_header:
                writer.writeheader()
                
            for instruction_id in tqdm(sampled_keys):
                # print("CAME HERE", flush=True)
                instruction_data = data[instruction_id]
                # print(instruction_data, flush=True)
                d = {}
                d['instruction'] = instruction_data['input']
                instruction_outputs = {}
                test_examples = instruction_data['input']
                #for id_, example in test_examples.items():
                user_prompt = instruction_data['input']
                print("user_prompt", user_prompt, flush=True)
                # Build chat conversation
                messages = [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt}
                ]
        
                # Generate output
                final_output, selected_distributions = iterative_generate(
                    messages,
                    gen_tokenizers_models,
                    eval_tokenizers_models
                )
                print(f"🔍 Iteration-wise Selection Summary:")
                for d in selected_distributions:
                    print(f"Iteration {d['iteration']}: Model {d['selected_model_index']} -> {d['selected_candidate']}")
                with open(f"{args.output}/{task_name}_selected_distributions_row_{instruction_id}.json", "w") as f:
                    json.dump(selected_distributions, f, indent=2)
                # Write a single row to the CSV
                writer.writerow({
                    "Instruction": instruction_data['input'],
                    "Ensembled Thought": final_output[generation_models[0]].strip()
                })
                csvfile.flush()
                # after writing the row:
                gc.collect()
                torch.cuda.empty_cache()
        
                print(f"✅ Done. Row-wise output saved to: {args.output}")

if __name__ == "__main__":
    set_seed(SEED)
    device = "auto"
    # === RUN ===
    main()
