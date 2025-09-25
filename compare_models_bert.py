import argparse
import json
import string
import re
from tqdm import tqdm
import os
import numpy as np
import random

from bert_score import score as bert_score

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

INDUCTION_TASKS = [
    'cause_and_effect', 'larger_animal', 'num_to_verbal', 'orthography_starts_with',
    'rhymes', 'synonyms', 'taxonomy_animal', 'translation_en-fr',
    'reverse_from_middle', 'smallest_item_length', 'smallest_even_no_sqrt', 
    'most_vowel_return_consonant', 'detect_rhyme_and_rewrite', 'rank_by_protein',
    'multi_lang_to_english', 'square_of_zodiac_animal', 'alternate_synonym_antonym', 
    'most_consonant_return_vowel', 'least_unique_word_count', 'first_word_alphabetically_return_reverse'
]


def normalize_prediction(prediction, lowercase=True):
    """Clean and normalize prediction text."""
    if not prediction:
        return ""
    
    # Clean up common model artifacts
    prediction = prediction.replace('<|return|>', '')
    prediction = prediction.replace('<|im_end|>', '')
    prediction = prediction.replace('<|endoftext|>', '')
    prediction = prediction.replace('<|begin_of_solution|>', '')
    prediction = prediction.replace('### Analysis:', '')
    prediction = prediction.replace('**Pattern Recognition**:', '')
    
    # Remove markdown formatting
    prediction = re.sub(r'\*\*([^*]+)\*\*', r'\1', prediction)  # Remove **bold**
    prediction = re.sub(r'\*([^*]+)\*', r'\1', prediction)      # Remove *italic*
    
    # Clean up extra whitespace and newlines
    prediction = re.sub(r'\s+', ' ', prediction)
    prediction = prediction.strip()
    
    # Extract meaningful content (everything after "The instruction is:")
    if "The instruction is:" in prediction:
        prediction = prediction.split("The instruction is:")[-1].strip()
    
    # Remove quotes at beginning and end
    prediction = prediction.strip('"').strip("'")
    
    if lowercase:
        prediction = prediction.lower()
    
    return prediction


def extract_answer(model_name, answer):
    """Extract final answer from model output based on model type."""
    if not answer:
        return ""
    
    # Model-specific extraction patterns
    if "gpt-oss" in model_name.lower():
        match = re.search(r'<\|channel\|>final<\|message\|>(.*)', answer, re.DOTALL)
        if match:
            answer = match.group(1).strip()
    elif "qwen" in model_name.lower():
        match = re.search(r'</think>(.*)', answer, re.DOTALL)
        if match:
            answer = match.group(1).strip()
    elif "openthinker" in model_name.lower():
        match = re.search(r'<\|end_of_thought\|>(.*)', answer, re.DOTALL)
        if match:
            answer = match.group(1).strip()
    
    # Additional cleanup for common patterns
    if not answer.strip():
        # If extraction failed, try to get the meaningful part
        if "The instruction is:" in answer:
            answer = answer.split("The instruction is:")[-1].strip()
        elif "### Analysis:" in answer:
            # Get everything before analysis section
            answer = answer.split("### Analysis:")[0].strip()
    
    return answer


def get_bert_score(prediction, reference):
    """
    Calculate BERTScore F1 between prediction and reference texts.
    
    Args:
        prediction (str): Generated text from model 1
        reference (str): Generated text from model 2
    
    Returns:
        float: BERTScore F1 (0-1)
    """
    pred_normalized = normalize_prediction(prediction, lowercase=True)
    ref_normalized = normalize_prediction(reference, lowercase=True)
    
    # Debug output
    print(f"Pred normalized: '{pred_normalized}'")
    print(f"Ref normalized: '{ref_normalized}'")
    
    if not pred_normalized or not ref_normalized:
        print("Empty normalized text detected")
        return 0.0
    
    # Ensure we have meaningful content
    if len(pred_normalized.strip()) < 3 or len(ref_normalized.strip()) < 3:
        print("Text too short for meaningful comparison")
        return 0.0
    
    try:
        P, R, F1 = bert_score(
            cands=[pred_normalized],
            refs=[ref_normalized],
            model_type="microsoft/deberta-xlarge-mnli",
            idf=False,
            rescale_with_baseline=False,
        )
        score = float(F1.item())
        print(f"BERTScore: {score:.4f}")
        return score
    except Exception as e:
        print(f"BERTScore calculation failed: {e}")
        return 0.0


def get_weighted_task_score(comparison_results):
    """Calculate average BERTScore across all instruction comparisons."""
    scores = []
    for instruction_id, data in comparison_results.items():
        if instruction_id not in ['weighted_task_score', 'task_name', 'model1_name', 'model2_name']:
            scores.append(data['bert_score'])
    
    return np.mean(scores) if scores else 0.0


def compare_models_bert(model1_name, model2_name, task_name, predictions_dir1, predictions_dir2, output_dir):
    """
    Compare two models' outputs using BERTScore.
    
    Args:
        model1_name: Name of first model
        model2_name: Name of second model  
        task_name: Task to evaluate
        predictions_dir1: Directory with model 1 predictions
        predictions_dir2: Directory with model 2 predictions
        output_dir: Directory to save comparison results
    """
    # Load predictions from both models
    pred_file1 = f'{predictions_dir1}/{task_name}_execution.json'
    pred_file2 = f'{predictions_dir2}/{task_name}_execution.json'
    
    if not os.path.exists(pred_file1):
        raise FileNotFoundError(f"Model 1 predictions not found: {pred_file1}")
    if not os.path.exists(pred_file2):
        raise FileNotFoundError(f"Model 2 predictions not found: {pred_file2}")
    
    with open(pred_file1, 'r', encoding='utf-8') as f:
        predictions1 = json.load(f)
    
    with open(pred_file2, 'r', encoding='utf-8') as f:
        predictions2 = json.load(f)
    
    # Get common instruction IDs
    ids1 = set(predictions1.keys())
    ids2 = set(predictions2.keys())
    common_ids = ids1.intersection(ids2)
    
    if not common_ids:
        raise ValueError(f"No common instruction IDs found between the two prediction files")
    
    # Sample instructions for comparison
    random.seed(42)
    sampled_ids = random.sample(list(common_ids), min(5, len(common_ids)))
    sampled_ids = sorted(sampled_ids, key=lambda x: int(x) if x.isdigit() else x)
    
    comparison_results = {}
    
    print(f"Comparing {len(sampled_ids)} instructions for task: {task_name}")
    
    for instruction_id in tqdm(sampled_ids, desc=f"Processing {task_name}"):
        # Extract outputs from both models
        output1 = extract_answer(model1_name, predictions1[instruction_id]['instruction_outputs'])
        output2 = extract_answer(model2_name, predictions2[instruction_id]['instruction_outputs'])
        
        # Calculate BERTScores (bidirectional)
        bert_1_to_2 = get_bert_score(output1, output2)
        bert_2_to_1 = get_bert_score(output2, output1)
        
        comparison_results[instruction_id] = {
            'model1_output': output1,
            'model2_output': output2,
            'bert_1_to_2': bert_1_to_2,
            'bert_2_to_1': bert_2_to_1,
            'bert_score': (bert_1_to_2 + bert_2_to_1) / 2,
            'model1_name': model1_name,
            'model2_name': model2_name
        }
        
        print(f"ID {instruction_id}: BERTScore = {comparison_results[instruction_id]['bert_score']:.4f}")
    
    # Calculate overall task score
    comparison_results['weighted_task_score'] = get_weighted_task_score(comparison_results)
    comparison_results['task_name'] = task_name
    comparison_results['model1_name'] = model1_name
    comparison_results['model2_name'] = model2_name
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    safe_model1 = model1_name.replace("/", "_")
    safe_model2 = model2_name.replace("/", "_")
    output_file = f'{output_dir}/{task_name}_bert_comparison_{safe_model1}_vs_{safe_model2}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved: {output_file}")
    print(f"Task BERTScore: {comparison_results['weighted_task_score']:.4f}")
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Compare two models using BERTScore")
    parser.add_argument("--model1_name", type=str, required=True,
                        help='Name of the first model')
    parser.add_argument("--model2_name", type=str, required=True,
                        help='Name of the second model')
    parser.add_argument('--predictions_dir1', type=str, required=True,
                        help='Directory with first model predictions')
    parser.add_argument('--predictions_dir2', type=str, required=True,
                        help='Directory with second model predictions')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save comparison results')
    parser.add_argument('--tasks', type=str, default=','.join(INDUCTION_TASKS),
                        help='Comma-separated list of tasks to compare')
    
    args = parser.parse_args()
    
    task_list = args.tasks.split(',')
    overall_scores = {}
    
    print(f"Comparing models: {args.model1_name} vs {args.model2_name}")
    print(f"Tasks to process: {len(task_list)}")
    
    for task in task_list:
        print(f"\n{'='*50}")
        print(f"Task: {task}")
        print(f"{'='*50}")
        
        try:
            results = compare_models_bert(
                model1_name=args.model1_name,
                model2_name=args.model2_name,
                task_name=task,
                predictions_dir1=args.predictions_dir1,
                predictions_dir2=args.predictions_dir2,
                output_dir=args.output_dir
            )
            overall_scores[task] = results['weighted_task_score']
        except Exception as e:
            print(f"Error processing task {task}: {e}")
            overall_scores[task] = 0.0
    
    # Save overall summary
    summary = {
        'model1_name': args.model1_name,
        'model2_name': args.model2_name,
        'task_scores': overall_scores,
        'average_bert_score': np.mean(list(overall_scores.values())) if overall_scores else 0.0
    }
    
    safe_model1 = args.model1_name.replace("/", "_")
    safe_model2 = args.model2_name.replace("/", "_")
    summary_file = f'{args.output_dir}/summary_bert_comparison_{safe_model1}_vs_{safe_model2}.json'
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model 1: {args.model1_name}")
    print(f"Model 2: {args.model2_name}")
    print(f"Average BERTScore: {summary['average_bert_score']:.4f}")
    print(f"Summary saved: {summary_file}")
    print("\nPer-task scores:")
    for task, score in overall_scores.items():
        print(f"  {task}: {score:.4f}")


if __name__ == '__main__':
    main()