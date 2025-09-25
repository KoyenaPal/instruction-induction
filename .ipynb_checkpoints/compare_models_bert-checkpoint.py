import argparse
import json
import string
import re
import os
from tqdm import tqdm
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
    """Normalize prediction text by removing unwanted tokens and punctuation."""
    replacements = [' and ', 'Sentence 1:', 'Sentence 2:', '<|return|>', '<|im_end|>', '<|endoftext|>']
    for replacement in replacements:
        prediction = prediction.replace(replacement, ' ')
    
    prediction = prediction.strip().split(".")[0]
    
    if lowercase:
        prediction = prediction.lower()

    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(str.maketrans('', '', string.punctuation))
    prediction = re.sub(r"\s+", " ", prediction).strip()
    
    return prediction


def extract_answer(model_name, answer):
    """Extract the final answer from model output based on model type."""
    model_lower = model_name.lower()
    
    if "gpt-oss" in model_lower:
        match = re.search(r'<\|channel\|>final<\|message\|>(.*)', answer, re.DOTALL)
        if match:
            return match.group(1)
    elif "qwen" in model_lower:
        match = re.search(r'</think>(.*)', answer, re.DOTALL)
        if match:
            return match.group(1)
    elif "openthinker" in model_lower:
        match = re.search(r'<\|end_of_thought\|>(.*)', answer, re.DOTALL)
        if match:
            return match.group(1)
    
    return answer


def get_bertscore_between_models(prediction, reference):
    """Calculate BERTScore between two model outputs."""
    pred_normalized = normalize_prediction(prediction, lowercase=True)
    ref_normalized = normalize_prediction(reference, lowercase=True)
    print(pred_normalized, flush=True)
    print(ref_normalized, flush=True)
    if not pred_normalized or not ref_normalized:
        return 0.0
    
    P, R, F1 = bert_score(
        cands=[pred_normalized],
        refs=[ref_normalized],
        model_type="microsoft/deberta-xlarge-mnli",
        idf=True,
        rescale_with_baseline=False,
    )
    
    return float(F1.item())


def get_weighted_task_score(scored_predictions):
    """Calculate weighted task score across all instructions."""
    scores = []
    for instruction_id, instruction_data in scored_predictions.items():
        if instruction_id != 'weighted_task_score' and 'bertscore' in instruction_data:
            scores.append(instruction_data['bertscore'])
    
    return np.mean(scores) if scores else 0.0


def compare_models_bertscore(model1_name, model2_name, task_name, execution_input_dir, 
                            predictions_dir1, predictions_dir2, output_dir):
    """Compare two models' outputs using BERTScore."""
    
    # Load input examples
    with open(f'{execution_input_dir}/{task_name}.json', encoding='utf-8') as f:
        examples = json.load(f)["examples"]
    
    # Load predictions from both models
    with open(f'{predictions_dir1}/{task_name}_execution.json', encoding='utf-8') as f:
        predictions1 = json.load(f)
    
    with open(f'{predictions_dir2}/{task_name}_execution.json', encoding='utf-8') as f:
        predictions2 = json.load(f)
    
    # Sample instructions for evaluation
    sampled_keys = random.sample(list(examples.keys()), min(5, len(examples)))
    sampled_keys = sorted(sampled_keys, key=lambda x: int(x))
    
    comparison_results = {}
    
    for instruction_id in tqdm(sampled_keys, desc=f"Comparing models on {task_name}"):
        instruction_data = examples[instruction_id]
        
        # Extract outputs from both models
        output1 = extract_answer(model1_name, predictions1[instruction_id]['instruction_outputs'])
        output2 = extract_answer(model2_name, predictions2[instruction_id]['instruction_outputs'])
        
        # Calculate BERTScores (bidirectional)
        bertscore_1_to_2 = get_bertscore_between_models(output1, output2)
        bertscore_2_to_1 = get_bertscore_between_models(output2, output1)
        avg_bertscore = (bertscore_1_to_2 + bertscore_2_to_1) / 2
        
        # Store results
        comparison_results[instruction_id] = {
            'instruction': instruction_data['input'],
            'model1_output': output1,
            'model2_output': output2,
            'bertscore_1_to_2': bertscore_1_to_2,
            'bertscore_2_to_1': bertscore_2_to_1,
            'bertscore': avg_bertscore,
            'model1_name': model1_name,
            'model2_name': model2_name
        }
    
    # Calculate overall task score
    comparison_results['weighted_task_score'] = get_weighted_task_score(comparison_results)
    comparison_results['task_name'] = task_name
    comparison_results['model1_name'] = model1_name
    comparison_results['model2_name'] = model2_name
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    safe_model1 = model1_name.replace("/", "_")
    safe_model2 = model2_name.replace("/", "_")
    output_file = f'{output_dir}/{task_name}_bertscore_comparison_{safe_model1}_vs_{safe_model2}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description='Compare two models using BERTScore')
    parser.add_argument("--model1_name", type=str, required=True, help='Name of the first model')
    parser.add_argument("--model2_name", type=str, required=True, help='Name of the second model')
    parser.add_argument('--execution_input_dir', type=str, required=True, help='Input execution data directory')
    parser.add_argument('--predictions_dir1', type=str, required=True, help='First model predictions directory')
    parser.add_argument('--predictions_dir2', type=str, required=True, help='Second model predictions directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--tasks', type=str, default=','.join(INDUCTION_TASKS), help='Comma-separated list of tasks')
    
    args = parser.parse_args()
    
    task_list = args.tasks.split(',')
    overall_scores = {}
    
    for task in task_list:
        try:
            results = compare_models_bertscore(
                model1_name=args.model1_name,
                model2_name=args.model2_name,
                task_name=task,
                execution_input_dir=args.execution_input_dir,
                predictions_dir1=args.predictions_dir1,
                predictions_dir2=args.predictions_dir2,
                output_dir=args.output_dir
            )
            overall_scores[task] = results['weighted_task_score']
            print(f"{task}: {results['weighted_task_score']:.4f}")
        except Exception as e:
            print(f"Error processing task {task}: {e}")
            overall_scores[task] = 0.0
    
    # Save overall summary
    summary = {
        'model1_name': args.model1_name,
        'model2_name': args.model2_name,
        'task_scores': overall_scores,
        'average_bertscore': np.mean(list(overall_scores.values())) if overall_scores else 0.0
    }
    
    safe_model1 = args.model1_name.replace("/", "_")
    safe_model2 = args.model2_name.replace("/", "_")
    summary_file = f'{args.output_dir}/overall_bertscore_comparison_{safe_model1}_vs_{safe_model2}.json'
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nOverall Results:")
    print(f"Model 1: {args.model1_name}")
    print(f"Model 2: {args.model2_name}")
    print(f"Average BERTScore: {summary['average_bertscore']:.4f}")
    print(f"Results saved to: {summary_file}")


if __name__ == '__main__':
    main()