import argparse
import json
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import random
import re
import os
import numpy as np

random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

INDUCTION_TASKS = ['cause_and_effect', 'larger_animal', 'num_to_verbal','orthography_starts_with',
                   'rhymes', 'synonyms', 'taxonomy_animal', 'translation_en-fr',
                   'reverse_from_middle', 'smallest_item_length', 'smallest_even_no_sqrt', 'most_vowel_return_consonant',
                   'detect_rhyme_and_rewrite', 'rank_by_protein','multi_lang_to_english','square_of_zodiac_animal',
                   'alternate_synonym_antonym', 'most_consonant_return_vowel', 'least_unique_word_count', 'first_word_alphabetically_return_reverse']

end_think_patterns = [
    r'</think>',
    r'<\|channel\|>final<\|message\|>',
    r'<\|end_of_thought\|>'
]

def extract_thinking(answer, model_name="qwen"):
    # get text in between <think> and </think>
    if "openthinker" in model_name.lower():
        match = re.search(r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>', answer, re.DOTALL)
    elif "gpt-oss" in model_name.lower():
        match = re.search(r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>', answer, re.DOTALL)
    else:
        match = re.search(r'<think>(.*?)</think>', answer, re.DOTALL)
        
    if match:
        match_text = match.group(1)
        cleaned_text = match_text.replace("assistantanalysis", "").replace("<|end_of_thought|>","").replace("</think>","")
        return cleaned_text
    else:
        if "openthinker" in model_name.lower():
            match = re.search(r'<\|begin_of_thought\|>(.*?)', answer, re.DOTALL)
        elif "gpt-oss" in model_name.lower():
            match = re.search(r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)', answer, re.DOTALL)
        else:
            match = re.search(r'<think>(.*?)', answer, re.DOTALL)
        if match:
            match_text = match.group(1)
            cleaned_text = match_text.replace("assistantanalysis", "").replace("<|end_of_thought|>","").replace("</think>","")
            return cleaned_text
        else:
            return answer

def extract_user_prompt(original_ensembled_thought):
    """Extract user prompt from between first <|im_start|> and <|im_end|> tags"""
    match = re.search(r'<\|im_start\|>.*?\n(.*?)<\|im_end\|>', original_ensembled_thought, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return original_ensembled_thought

def clear_memory():
    """Clear GPU memory and run garbage collection"""
    torch.cuda.empty_cache()
    gc.collect()

def run_execution_accuracy_open_source_chat(execution_engine, task_name,
                                            source_folder, max_tokens=2048, device="cuda", thought_type="default"):
    """
    execution_engine: HuggingFace model name or path (e.g., "meta-llama/Llama-2-7b-chat-hf")
    """

    clear_memory()
    print("CAME TO THE FUNCTION TO EXECUTE ACCURACY", flush=True)
    
    # Load data from source_folder
    file_name = f"{task_name}_without_answer.json"
    with open(f'{source_folder}/{file_name}', encoding='utf-8') as f:
        source_data = json.load(f)

    # Load model & tokenizer with memory optimizations
    tokenizer = AutoTokenizer.from_pretrained(execution_engine, cache_dir="/workspace/hf")
    

    model = AutoModelForCausalLM.from_pretrained(
            execution_engine, 
            dtype=torch.bfloat16,
            device_map="auto", 
            cache_dir="/workspace/hf",
            low_cpu_mem_usage=True)
    model.eval()

    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    output_ = dict()
    
    # Set seed and sample 5 items
    random.seed(42)
    sampled_keys = random.sample(list(source_data.keys()), 5)
    sampled_keys = sorted(sampled_keys, key=lambda x: int(x))

    # Handle sample and empty conditions
    do_sample = False
    temperature = 0.0
    if "with_sampling" in thought_type or ("with_sampling_without_answer" in thought_type):
        do_sample = True
        temperature = 1.0

    for instruction_id in tqdm(sampled_keys):
        # Clear memory before each iteration
        clear_memory()
        
        instruction_data = source_data[instruction_id]
        d = {}
        
        # Extract user prompt from original_ensembled_thought
        user_prompt = extract_user_prompt(instruction_data['original_ensembled_thought'])
        d['instruction'] = user_prompt
        
        print("user_prompt", user_prompt, flush=True)
        
        # Build chat conversation
        messages = [
            {"role": "user", "content": user_prompt}
        ]

        # Get thinking message
        thinking_message = ""
        if "without_answer" in thought_type and "ensembled_thought_without_answer" in instruction_data:
            thinking_message = extract_thinking(instruction_data['ensembled_thought_without_answer'], model_name=execution_engine)
        else:
            thinking_message = extract_thinking(instruction_data['original_ensembled_thought'], model_name=execution_engine)

        # Convert to model-specific chat template
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if "oss" in execution_engine.lower():
            user_message = [{"role": "user", "content": messages[-1]["content"]}]
            chat_prompt = tokenizer.apply_chat_template(user_message,
                                                             model_identity=messages[0]["content"],
                                                             reasoning_effort = "low",
                                                             tokenize=False, add_generation_prompt=True)
            if not(thought_type == "default") and not(thought_type == "with_sampling"):
                chat_prompt = f"{chat_prompt}<|start|>assistant<|channel|>analysis<|message|>{thinking_message}<|end|><|start|>assistant<|channel|>final<|message|>"
        if "qwen" in execution_engine.lower():
            if "<think>" not in chat_prompt:
                chat_prompt = f"{chat_prompt}<think>"
            if not(thought_type == "default") and not(thought_type == "with_sampling"):
                chat_prompt = f"{chat_prompt}{thinking_message}</think>"
        elif "openthinker" in execution_engine.lower():
            chat_prompt = f"{chat_prompt}<|begin_of_thought|>"
            if not(thought_type == "default") and not(thought_type == "with_sampling"):
                chat_prompt = f"{chat_prompt}{thinking_message}<|end_of_thought|><|begin_of_solution|>"
        
        # Tokenize with truncation to prevent excessive memory usage
        inputs = tokenizer(
            chat_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096  # Limit input length
        ).to(model.device)
        
        if thought_type == "default" or thought_type == "with_sampling":
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens-30,
                    do_sample=do_sample,
                    temperature=temperature,
                    use_cache=True,  # Enable KV cache for efficiency
                    pad_token_id=tokenizer.eos_token_id  # Handle padding
                )
            
            # Clear inputs from memory immediately
            del inputs
            clear_memory()
    
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Clear outputs from memory
            del outputs
            clear_memory()
            
            found = any(re.search(pat, prediction) for pat in end_think_patterns)
            if not found:
                if "gpt-oss" in execution_engine.lower():
                    prediction = prediction + "<|channel|>final<|message|>"
                if "qwen" in execution_engine.lower():
                    prediction = prediction + "</think>"
                elif "openthinker" in execution_engine.lower():
                    prediction = prediction + f"<|end_of_thought|>"
            
            prediction = prediction + " The instruction is:"
            new_inputs = tokenizer(
                prediction, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4096
            ).to(model.device)
            
            with torch.no_grad():
                new_output = model.generate(
                    **new_inputs,
                    max_new_tokens=30,
                    do_sample=do_sample,
                    temperature=temperature,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Clear intermediate tensors
            del new_inputs
            clear_memory()
            
            prediction = tokenizer.decode(new_output[0], skip_special_tokens=False)
            
            # Clear final outputs
            del new_output
            clear_memory()
    
            d['instruction_outputs'] = prediction
            output_[instruction_id] = d
        else:
            chat_prompt = chat_prompt + " The instruction is:"
            inputs = tokenizer(
                chat_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4096
            ).to(model.device)
            
            print("CAME TO ONLY 30 TOKENS")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=do_sample,
                    temperature=temperature,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Clear tensors immediately
            del inputs
            clear_memory()
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            del outputs
            clear_memory()
            
            d['instruction_outputs'] = prediction
            output_[instruction_id] = d

    # Final cleanup
    del model
    del tokenizer
    clear_memory()
    
    return output_


if __name__ == '__main__':
    INDUCTION_TASKS_STR = ','.join(INDUCTION_TASKS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution_engine", type=str, default='text-davinci-002', help='The execution engine.')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Max number of tokens to generate.')
    parser.add_argument('--tasks', type=str, default=INDUCTION_TASKS_STR,
                        help='Tasks for execution accuracy evaluation.')
    parser.add_argument('--thought_type', type=str, help='Thought Type')
    parser.add_argument('--source_folder', type=str, required=True, help='Folder containing the task JSON files with ensembled thoughts')
    
    args = parser.parse_args()

    model_map = {"nrr": "nvidia_Nemotron-Research-Reasoning-Qwen-1.5B",
                "opent": "open-thoughts_OpenThinker-7B",
                "oss": "openai_gpt-oss-20b",
                "qwq":"Qwen_QwQ-32B",
                "dapo": "BytedTsinghua-SIA_DAPO-Qwen-32B"}
    
    task_list = args.tasks.split(',')
    execution_engine = str(args.execution_engine)
    cleaned_execution_engine = execution_engine.replace("/","_")
    print("EXECUTION ENGINE", cleaned_execution_engine)
    
    # HANDLE OUTPUT DIR VALUES FOR EACH THOUGHT TYPE HERE!
    out_dir = f"predictions_{cleaned_execution_engine}"
    if "transfer" in args.thought_type:
        source_model = str(args.source_folder).split("predictions_")[-1]
        print("SOURCE MODEL", source_model, flush=True)
        out_dir = f"predictions_{source_model}_thoughts_to_{cleaned_execution_engine}"
        print("OUT DIRECTORY", out_dir, flush=True)
    elif (args.thought_type == "ensemble") or (args.thought_type == "ensemble_without_answer"):
        ensembled_model = str(args.source_folder).split("ensemble_outputs_")[-1]
        out_dir = f"predictions_{cleaned_execution_engine}_{args.thought_type}_{ensembled_model}"
    elif not(args.thought_type == "default"):
        out_dir = f"predictions_{cleaned_execution_engine}_{args.thought_type}"
    Path(out_dir).mkdir(exist_ok=True)
    
    for induction_task in task_list:
        output_results = run_execution_accuracy_open_source_chat(execution_engine=args.execution_engine,
                                                                task_name=induction_task,
                                                                source_folder=args.source_folder,
                                                                max_tokens=args.max_tokens,
                                                                thought_type=args.thought_type)
        
        # Save results
        output_path = f'{out_dir}'
        print("OUT DIRECTORY", output_path)
        Path(output_path).mkdir(exist_ok=True)
        
        with open(f'{output_path}/{induction_task}_execution.json', 'w', encoding='utf-8') as f_predictions:
            json.dump(output_results, f_predictions, indent=2, ensure_ascii=False)
        
        # Clear memory after each task
        clear_memory()