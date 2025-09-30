#!/usr/bin/env python3
import os
import json
import re
import argparse
import time
from tqdm import tqdm
from openai import OpenAI

INDUCTION_TASKS = ['cause_and_effect', 'larger_animal', 'num_to_verbal','orthography_starts_with',
                   'rhymes', 'synonyms', 'taxonomy_animal', 'translation_en-fr',
                   'reverse_from_middle', 'smallest_item_length', 'smallest_even_no_sqrt', 'most_vowel_return_consonant',
                   'detect_rhyme_and_rewrite', 'rank_by_protein','multi_lang_to_english','square_of_zodiac_animal',
                   'alternate_synonym_antonym', 'most_consonant_return_vowel', 'least_unique_word_count', 'first_word_alphabetically_return_reverse']



def extract_thinking(answer, model_name="qwen"):
    """Extract thinking content from various model formats."""
    if "openthinker" in model_name.lower():
        pattern = r'<\|begin_of_thought\|>(.*?)(<\|end_of_thought\|>|$)'
    elif "gpt-oss" in model_name.lower():
        pattern = r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)(<\|end\|>|$)'
    else:
        pattern = r'<think>(.*?)(</think>|$)'
    
    match = re.search(pattern, answer, re.DOTALL)
    if match:
        text = match.group(1)
        return text.replace("assistantanalysis", "").replace("<|end_of_thought|>", "").replace("</think>", "").strip()
    return answer


def setup_client(api_key):
    """Initialize OpenAI client."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required")
    return OpenAI(api_key=key)

def find_tasks(input_dir):
    """Find valid execution JSON files."""
    files = [f for f in os.listdir(input_dir) if f.endswith('_execution.json')]
    tasks = [f.replace('_execution.json', '') for f in files]
    return [task for task in tasks if task in INDUCTION_TASKS]

def process_text(client, text):
    """Remove answers from text using OpenAI."""
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "user", "content": f"""
        Task: Keep only the hints from the text and remove answer sentences.
        
        Definition: 
        - A "hint/explanation sentence" provides guidance that helps someone think about the problem without giving the final solution.  
        - An "answer sentence" directly states the final answer, solution, result, or conclusion.
        
        Instructions:  
        1. Keep every hint/explanation sentence exactly as written.  
        2. Remove all answer sentences and statements.
        3. Preserve the original wording, order, and formatting of the remaining text.
        4. Do not add, rephrase, or generate any new text beyond what is already in the original.
        5. Output only the hints.
        
        Original text:
        {text}
        """}],
        reasoning_effort="medium"
    )
    return extract_thinking(response.choices[0].message.content)

def process_task(client, input_dir, task_name, sleep_time):
    """Process one task file."""
    with open(f"{input_dir}/{task_name}_execution.json", encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    for instruction_id, instruction_data in tqdm(data.items(), desc=task_name):
        if 'instruction_outputs' not in instruction_data:
            continue
            
        text = extract_thinking(str(instruction_data['instruction_outputs']))
        
        try:
            processed = process_text(client, text)
            results[instruction_id] = {
                "original_instruction_outputs": instruction_data['instruction_outputs'],
                "processed_without_answer": processed
            }
        except Exception as e:
            results[instruction_id] = {
                "original_instruction_outputs": instruction_data['instruction_outputs'],
                "processed_without_answer": "",
                "error": str(e)
            }
        
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="without_answer")
    parser.add_argument("--api_key")
    parser.add_argument("--sleep_between_calls", type=float, default=0.0)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    client = setup_client(args.api_key)
    tasks = find_tasks(args.input_dir)
    
    print(f"Processing {len(tasks)} tasks: {tasks}")
    
    for task in tasks:
        results = process_task(client, args.input_dir, task, args.sleep_between_calls)
        
        output_file = f"{args.output_dir}/{task}_execution.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {task}: {len(results)} items")
    
    print("Complete!")

if __name__ == "__main__":
    main()