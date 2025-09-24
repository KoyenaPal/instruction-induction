from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import re
import random
import numpy as np
import os
import csv
from tqdm import tqdm
import pandas as pd
import gc
import time
import copy
import json

# Configuration
SEED = 42
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:1024"

available_gpus = torch.cuda.device_count()
print(f"Found {available_gpus} GPUs")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Memory Management
def cleanup_memory():
    """Clean up GPU and system memory"""
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

def get_memory_info(device_id):
    """Get GPU memory information"""
    device = f"cuda:{device_id}"
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"GPU {device_id}: {allocated:.1f}GB/{total:.1f}GB")
    return allocated, reserved, total

# Model Management
class ModelManager:
    def __init__(self, generation_models, evaluation_models, available_gpus):
        self.generation_models = generation_models
        self.evaluation_models = evaluation_models
        self.available_gpus = available_gpus
        self.loaded_models = {}
        self.model_gpu_map = {}
        
        # Assign GPUs to models
        all_models = generation_models + evaluation_models
        for i, model_name in enumerate(all_models):
            self.model_gpu_map[model_name] = i % available_gpus
    
    def load_model(self, model_name, cache_dir="/workspace/hf"):
        """Load model with H200 optimization"""
        gpu_id = self.model_gpu_map[model_name]
        device_map = f"cuda:{gpu_id}"
        
        cleanup_memory()
        print(f"Loading {model_name} on GPU {gpu_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            max_memory={gpu_id: "130GB"},
            low_cpu_mem_usage=True,
            offload_folder="offload_dir"
        )
        model.eval()
        
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        get_memory_info(gpu_id)
        return tokenizer, model
    
    def get_model(self, model_name):
        """Get model, loading if necessary"""
        if model_name not in self.loaded_models:
            tokenizer, model = self.load_model(model_name)
            self.loaded_models[model_name] = (tokenizer, model)
        return self.loaded_models[model_name]
    
    def cleanup(self):
        """Clean up all models"""
        for model_name, (tokenizer, model) in self.loaded_models.items():
            del tokenizer, model
        self.loaded_models.clear()
        cleanup_memory()

# Text Processing
def truncate_to_last_sentence(text):
    """Truncate text to the last complete sentence"""
    text = text.strip()
    
    def is_abbreviation(word):
        lw = word.lower()
        if re.match(r"\d+\.\d+$", lw):  # Decimal
            return True
        if re.match(r"[A-Za-z]{1,4}\.$", lw):  # Short abbreviation
            return True
        if re.match(r"(?:[A-Za-z]{1,4}\.){2,}$", lw):  # Multi-part abbreviation
            return True
        return False

    matches = list(re.finditer(r'(?<!\d)([.!?])(?=\s|$|[A-Z])', text))
    
    last_good_end = None
    for m in matches:
        end_pos = m.end()
        before = text[:end_pos].strip()
        last_word = before.split()[-1] if before.split() else ""
        if not is_abbreviation(last_word):
            last_good_end = end_pos
    
    return text[:last_good_end].strip() if last_good_end else text

# Generation Functions
def generate_candidates(context, tokenizer, model, num_candidates, max_tokens=15):
    """Generate candidate continuations"""
    set_seed(SEED)
    model_device = next(model.parameters()).device
    
    with torch.inference_mode():
        enc = tokenizer(context, return_tensors="pt", truncation=True).to(model_device)
        
        output = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            num_return_sequences=num_candidates,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        
        candidates = []
        for out in output:
            full_text = tokenizer.decode(out[enc["input_ids"].shape[-1]:], skip_special_tokens=True)
            full_text = truncate_to_last_sentence(full_text)
            candidates.append(full_text)
        
        del output, enc
        torch.cuda.empty_cache()
    
    return candidates

def compute_perplexities(context, candidates, tokenizer, model):
    """Compute perplexity for each candidate"""
    model_device = next(model.parameters()).device
    perplexities = []
    
    with torch.inference_mode():
        for candidate in candidates:
            full_text = context + candidate
            
            enc = tokenizer(full_text, return_tensors="pt", truncation=True).to(model_device)
            outputs = model(input_ids=enc["input_ids"], 
                          attention_mask=enc["attention_mask"], 
                          labels=enc["input_ids"])
            
            perplexity = torch.exp(outputs.loss).item()
            perplexities.append(perplexity)
            
            del outputs, enc
            torch.cuda.empty_cache()
    
    return perplexities

# Main Generation Loop
# max tokens = 2048 - 30
def iterative_generate(context, model_manager, max_tokens=2018, num_candidates=3):
    """Main iterative generation with ensemble scoring"""
    iteration_count = 0
    contexts = {}
    chosen_per_iteration = []
    
    # Initialize contexts for each model
    for model_name in model_manager.generation_models:
        tokenizer, _ = model_manager.get_model(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        context_formatted = tokenizer.apply_chat_template(
            context, tokenize=False, add_generation_prompt=True
        )
        
        # Add model-specific tokens
        if "qwen" in model_name.lower():
            context_formatted += "<think>"
        elif "openthinker" in model_name.lower():
            context_formatted += "<|begin_of_thought|>"
        elif "gpt-oss" in model_name.lower():
            user_message = [{"role": "user", "content": context[-1]["content"]}]
            context_formatted = tokenizer.apply_chat_template(
                user_message,
                model_identity=context[0]["content"] if len(context) > 1 else "",
                reasoning_effort="low",
                tokenize=False,
                add_generation_prompt=True
            )
        
        contexts[model_name] = context_formatted
    
    # Calculate token limits
    first_model = model_manager.generation_models[0]
    tokenizer, _ = model_manager.get_model(first_model)
    total_tokens = len(tokenizer.encode(contexts[first_model]))
    max_total_tokens = total_tokens + max_tokens
    
    while True:
        print(f"\n--- Iteration {iteration_count} ---")
        
        # Check termination conditions
        if iteration_count > 0:
            if total_tokens >= max_total_tokens:
                print("✅ Reached max length")
                break
            if "</think>" in contexts[first_model]:
                print("✅ Reached end of thought")
                break
        
        # Generate candidates from each generation model
        all_candidates = []
        candidates_per_model = {}
        
        for model_name in model_manager.generation_models:
            tokenizer, model = model_manager.get_model(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            candidates = generate_candidates(
                contexts[model_name], tokenizer, model, num_candidates
            )
            
            print(f"{model_name}: {candidates}")
            all_candidates.extend(candidates)
            candidates_per_model[model_name] = candidates
        
        # Score candidates with evaluation models
        candidate_scores = {cand: [] for cand in all_candidates}
        
        for model_name in model_manager.evaluation_models:
            tokenizer, model = model_manager.get_model(model_name)
            
            try:
                perplexities = compute_perplexities(
                    contexts[model_name], all_candidates, tokenizer, model
                )
                for i, candidate in enumerate(all_candidates):
                    candidate_scores[candidate].append(perplexities[i])
            except Exception as e:
                print(f"⚠️ Error scoring with {model_name}: {e}")
                continue
        
        # Select best candidate
        scored_candidates = []
        for candidate, scores in candidate_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                scored_candidates.append((candidate, avg_score))
        
        if not scored_candidates:
            print("❌ No valid candidates")
            break
        
        best_candidate, best_score = min(scored_candidates, key=lambda x: x[1])
        print(f"🔹 Selected: {best_candidate.strip()}")
        
        # Find which model generated the best candidate
        selected_model = None
        for model_name, candidates in candidates_per_model.items():
            if best_candidate in candidates:
                selected_model = model_name
                break
        
        chosen_per_iteration.append({
            "iteration": iteration_count,
            "selected_candidate": best_candidate,
            "selected_model": selected_model,
            "all_candidates": copy.deepcopy(candidates_per_model),
            "score": best_score
        })
        
        # Update all contexts
        for model_name in contexts:
            contexts[model_name] += " " + best_candidate.strip()
        
        total_tokens += len(tokenizer.encode(best_candidate))
        iteration_count += 1
    
    return contexts, chosen_per_iteration

# Main Function
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ensemble generation with iterative selection")
    parser.add_argument("--output", type=str, 
                       default="instruction_induction_ensemble_outputs_gen_qwq_dapo_eval_oss", 
                       help="Output directory")
    parser.add_argument("--gen_models", nargs='+', 
                       default=["Qwen/QwQ-32B", "BytedTsinghua-SIA/DAPO-Qwen-32B"],
                       help="Generation models")
    parser.add_argument("--eval_models", nargs='+', 
                       default=["openai/gpt-oss-20b"],
                       help="Evaluation models")
    
    args = parser.parse_args()
    set_seed(SEED)
    
    # Initialize model manager
    model_manager = ModelManager(args.gen_models, args.eval_models, available_gpus)
    
    # Task configuration
    input_dir = "data/induction_input"
    INDUCTION_TASKS = ['cause_and_effect', 'larger_animal', 'num_to_verbal','orthography_starts_with',
                           'rhymes', 'synonyms', 'taxonomy_animal', 'translation_en-fr',
                           'reverse_from_middle', 'smallest_item_length', 'smallest_even_no_sqrt', 'most_vowel_return_consonant',
                           'detect_rhyme_and_rewrite', 'rank_by_protein','multi_lang_to_english','square_of_zodiac_animal',
                           'alternate_synonym_antonym', 'most_consonant_return_vowel', 'least_unique_word_count', 'first_word_alphabetically_return_reverse']
    
    try:
        for task_name in INDUCTION_TASKS:
            print(f"\n=== Processing {task_name} ===")
            
            output_file = f"{args.output}/{task_name}.csv"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Load task data
            with open(f'{input_dir}/./{task_name}.json', encoding='utf-8') as f:
                data = json.load(f)["examples"]
            
            # Sample 5 examples
            random.seed(42)
            sampled_keys = sorted(random.sample(list(data.keys()), 5), key=int)
            
            # Process each example
            write_header = not os.path.exists(output_file)
            with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["Instruction", "Ensembled Thought"], 
                                      quoting=csv.QUOTE_ALL)
                
                if write_header:
                    writer.writeheader()
                
                for instruction_id in tqdm(sampled_keys, desc=f"Processing {task_name}"):
                    instruction_data = data[instruction_id]
                    user_prompt = instruction_data['input']
                    
                    messages = [{"role": "user", "content": user_prompt}]
                    
                    # Generate ensemble output
                    final_contexts, selection_history = iterative_generate(
                        messages, model_manager
                    )
                    
                    # Save results
                    with open(f"{args.output}/{task_name}_history_{instruction_id}.json", "w") as f:
                        json.dump(selection_history, f, indent=2)
                    
                    writer.writerow({
                        "Instruction": user_prompt,
                        "Ensembled Thought": final_contexts[args.gen_models[0]].strip()
                    })
                    csvfile.flush()
                    
                    print(f"✅ Completed {task_name} example {instruction_id}")
    
    finally:
        model_manager.cleanup()
        print("🧹 Cleanup complete")

if __name__ == "__main__":
    main()