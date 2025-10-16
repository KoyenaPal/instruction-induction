#!/usr/bin/env python3
"""
Aggregate and analyze BERTScore comparison results from organized directories.

Expected directory structure:
    comparison_results/
        transfer_oss_full_text/
            task1_bert_comparison_model1_vs_model2.json
            task2_bert_comparison_model1_vs_model2.json
            ...
        transfer_opent_without_answer/
            ...
"""

import argparse
import json
import os
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


INDUCTION_TASKS = [
    'cause_and_effect', 'larger_animal', 'num_to_verbal', 'orthography_starts_with',
    'rhymes', 'synonyms', 'taxonomy_animal', 'translation_en-fr',
    'reverse_from_middle', 'smallest_item_length', 'smallest_even_no_sqrt', 
    'most_vowel_return_consonant', 'detect_rhyme_and_rewrite', 'rank_by_protein',
    'multi_lang_to_english', 'square_of_zodiac_animal', 'alternate_synonym_antonym', 
    'most_consonant_return_vowel', 'least_unique_word_count', 
    'first_word_alphabetically_return_reverse'
]


def parse_comparison_filename(filename):
    """
    Parse comparison result filename to extract task and model information.
    
    Expected format: {task}_bert_comparison_{model1}_vs_{model2}.json
    
    Returns:
        tuple: (task_name, model1, model2) or (None, None, None) if parsing fails
    """
    name = filename.replace('.json', '')
    match = re.match(r'(.+?)_bert_comparison_(.+)_vs_(.+)', name)
    
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None


def extract_setting_info(setting_dir_name):
    """
    Extract configuration information from setting directory name.
    
    Examples:
        'transfer_oss_full_text' -> ('transfer_oss', 'full_text')
        'ensemble_qwq_dapo_without_answer' -> ('ensemble_qwq_dapo', 'without_answer')
    
    Returns:
        tuple: (condition, answer_condition)
    """
    parts = setting_dir_name.split('_')
    
    # Determine answer completeness
    if 'without' in parts and 'answer' in parts:
        answer_condition = 'without_answer'
        parts = [p for p in parts if p not in ['without', 'answer']]
    else:
        answer_condition = 'full_text'
        parts = [p for p in parts if p not in ['full', 'text']]
    
    # Determine condition type
    if 'transfer' in parts:
        condition = 'transfer_' + '_'.join([p for p in parts if p != 'transfer'])
    elif 'ensemble' in parts:
        condition = 'ensemble_' + '_'.join([p for p in parts if p != 'ensemble'])
    elif 'empty' in parts:
        condition = 'empty'
    elif 'default' in parts:
        condition = 'default'
    elif 'sampled' in parts:
        condition = 'sampled'
    else:
        condition = '_'.join(parts)
    
    return condition, answer_condition


def load_comparison_results(comparison_results_dir):
    """
    Load all comparison results from organized directory structure.
    
    Args:
        comparison_results_dir: Base directory containing setting subdirectories
        
    Returns:
        DataFrame with all comparison results
    """
    all_results = []
    setting_dirs = [
        d for d in os.listdir(comparison_results_dir) 
        if os.path.isdir(os.path.join(comparison_results_dir, d))
    ]
    
    print(f"Found {len(setting_dirs)} setting directories")
    
    for setting_dir in tqdm(setting_dirs, desc="Loading settings"):
        setting_path = os.path.join(comparison_results_dir, setting_dir)
        condition, answer_condition = extract_setting_info(setting_dir)
        
        json_files = glob.glob(os.path.join(setting_path, '*_bert_comparison_*.json'))
        
        for json_file in json_files:
            filename = os.path.basename(json_file)
            task_name, model1, model2 = parse_comparison_filename(filename)
            
            if task_name is None:
                print(f"Warning: Could not parse filename: {filename}")
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                weighted_score = data.get('weighted_task_score', 0.0)
                
                # Extract per-instruction scores
                instruction_scores = [
                    value['bert_score']
                    for key, value in data.items()
                    if key not in ['weighted_task_score', 'task_name', 'model1_name', 'model2_name']
                    and isinstance(value, dict) and 'bert_score' in value
                ]
                
                all_results.append({
                    'setting_dir': setting_dir,
                    'condition': condition,
                    'answer_condition': answer_condition,
                    'task_name': task_name,
                    'model1_name': model1,
                    'model2_name': model2,
                    'weighted_task_score': weighted_score,
                    'num_instructions': len(instruction_scores),
                    'mean_instruction_score': np.mean(instruction_scores) if instruction_scores else 0.0,
                    'std_instruction_score': np.std(instruction_scores) if instruction_scores else 0.0,
                    'comparison_file': json_file
                })
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
    
    df = pd.DataFrame(all_results)
    print(f"Loaded {len(df)} comparison results")
    
    return df


def aggregate_by_configuration(df):
    """
    Aggregate results by model configuration.
    
    Groups by (model1, model2, condition, answer_condition) and computes statistics.
    
    Returns:
        DataFrame with aggregated results
    """
    grouped = df.groupby([
        'model1_name', 'model2_name', 
        'condition', 'answer_condition'
    ]).agg({
        'weighted_task_score': ['mean', 'std', 'count'],
        'task_name': lambda x: list(x)
    }).reset_index()
    
    grouped.columns = [
        'model1_name', 'model2_name', 'condition', 'answer_condition',
        'average_score', 'std_score', 'num_tasks', 'tasks'
    ]
    
    return grouped


def create_summary_statistics(aggregated_df):
    """
    Create comprehensive summary statistics.
    
    Returns:
        dict: Summary statistics including per-condition and per-answer breakdowns
    """
    summary = {
        'total_configurations': len(aggregated_df.groupby(['condition', 'answer_condition'])),
        'total_model_pairs': len(aggregated_df.groupby(['model1_name', 'model2_name'])),
        'conditions': sorted(aggregated_df['condition'].unique().tolist()),
        'answer_conditions': sorted(aggregated_df['answer_condition'].unique().tolist()),
        'overall_mean_consistency': float(aggregated_df['average_score'].mean()),
        'overall_std_consistency': float(aggregated_df['average_score'].std()),
        'per_condition_stats': {},
        'per_answer_condition_stats': {}
    }
    
    # Per-condition statistics broken down by answer completeness
    for condition in aggregated_df['condition'].unique():
        summary['per_condition_stats'][condition] = {}
        
        for answer_cond in aggregated_df['answer_condition'].unique():
            cond_data = aggregated_df[
                (aggregated_df['condition'] == condition) & 
                (aggregated_df['answer_condition'] == answer_cond)
            ]
            
            if len(cond_data) > 0:
                summary['per_condition_stats'][condition][answer_cond] = {
                    'mean': float(cond_data['average_score'].mean()),
                    'std': float(cond_data['average_score'].std()),
                    'count': int(len(cond_data))
                }
    
    # Overall statistics per answer condition
    for answer_cond in aggregated_df['answer_condition'].unique():
        answer_data = aggregated_df[aggregated_df['answer_condition'] == answer_cond]
        summary['per_answer_condition_stats'][answer_cond] = {
            'mean': float(answer_data['average_score'].mean()),
            'std': float(answer_data['average_score'].std()),
            'count': int(len(answer_data))
        }
    
    return summary


def create_pairwise_matrix(df):
    """
    Create pairwise consistency matrix for all configurations.
    
    Returns:
        DataFrame: Pivot table with pairwise consistency scores
    """
    df = df.copy()
    df['config1'] = (df['model1_name'] + '_' + df['condition'] + '_' + df['answer_condition'])
    df['config2'] = (df['model2_name'] + '_' + df['condition'] + '_' + df['answer_condition'])
    
    pivot = df.pivot_table(
        index='config1',
        columns='config2', 
        values='average_score',
        aggfunc='mean'
    )
    
    return pivot


def create_summary_plots(aggregated_df, output_dir):
    """Create summary visualization plots."""
    viz_dir = os.path.join(output_dir, 'summary_plots')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Overall consistency distribution
    plt.figure(figsize=(10, 6))
    plt.hist(aggregated_df['average_score'], bins=30, edgecolor='black', alpha=0.7, color='#4A90E2')
    plt.xlabel('Average Consistency Score', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Distribution of Consistency Scores', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/consistency_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-condition comparison
    plt.figure(figsize=(14, 6))
    condition_data = aggregated_df.groupby('condition')['average_score'].agg(['mean', 'std']).reset_index()
    condition_data = condition_data.sort_values('mean', ascending=False)
    
    x = np.arange(len(condition_data))
    plt.bar(x, condition_data['mean'], yerr=condition_data['std'], 
            capsize=5, alpha=0.8, edgecolor='black', color='#5FBA7D')
    plt.xticks(x, condition_data['condition'], rotation=45, ha='right')
    plt.xlabel('Condition', fontsize=12, fontweight='bold')
    plt.ylabel('Average Consistency Score', fontsize=12, fontweight='bold')
    plt.title('Consistency by CoT Condition', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/consistency_by_condition.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Answer completeness comparison
    plt.figure(figsize=(10, 6))
    answer_data = aggregated_df.groupby('answer_condition')['average_score'].agg(['mean', 'std']).reset_index()
    
    x = np.arange(len(answer_data))
    colors = ['#4CAF50', '#FF9800']
    plt.bar(x, answer_data['mean'], yerr=answer_data['std'],
            capsize=5, alpha=0.8, edgecolor='black', color=colors[:len(answer_data)])
    plt.xticks(x, answer_data['answer_condition'])
    plt.xlabel('Answer Completeness', fontsize=12, fontweight='bold')
    plt.ylabel('Average Consistency Score', fontsize=12, fontweight='bold')
    plt.title('Consistency: Full Text vs Without Answer', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/consistency_by_answer_condition.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary visualizations saved to: {viz_dir}")


def save_results(aggregated_df, summary, output_dir):
    """Save all results and summary statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save aggregated results
    results_csv = os.path.join(output_dir, 'comprehensive_comparison_results.csv')
    aggregated_df.to_csv(results_csv, index=False)
    print(f"Saved aggregated results to: {results_csv}")
    
    # Save summary statistics
    summary_json = os.path.join(output_dir, 'comparison_summary.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_json}")
    
    # Create and save pairwise matrix
    matrix = create_pairwise_matrix(aggregated_df)
    matrix_csv = os.path.join(output_dir, 'pairwise_consistency_matrix.csv')
    matrix.to_csv(matrix_csv)
    print(f"Saved pairwise matrix to: {matrix_csv}")
    
    return results_csv


def print_summary(summary):
    """Print summary statistics to console."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total configurations: {summary['total_configurations']}")
    print(f"Total model pairs: {summary['total_model_pairs']}")
    print(f"Average consistency: {summary['overall_mean_consistency']:.4f} ± "
          f"{summary['overall_std_consistency']:.4f}")
    
    print("\nPer-answer condition statistics:")
    for answer_cond, stats in summary['per_answer_condition_stats'].items():
        print(f"  {answer_cond}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
    
    print("\nPer-condition statistics (full_text):")
    full_text_stats = {
        k: v.get('full_text', {}) 
        for k, v in summary['per_condition_stats'].items() 
        if 'full_text' in v
    }
    sorted_stats = sorted(full_text_stats.items(), key=lambda x: x[1].get('mean', 0), reverse=True)
    for condition, stats in sorted_stats[:10]:
        if stats:
            print(f"  {condition}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and analyze model comparison results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example directory structure:
    comparison_results/
        transfer_oss_full_text/
            task1_bert_comparison_model1_vs_model2.json
            ...
        transfer_opent_without_answer/
            ...
        """
    )
    parser.add_argument(
        '--comparison_results_dir', 
        type=str, 
        required=True,
        help='Base directory containing setting subdirectories with comparison results'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help='Directory to save aggregated results and analysis'
    )
    parser.add_argument(
        '--create_visualizations', 
        action='store_true',
        help='Create summary visualization plots'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.comparison_results_dir):
        print(f"Error: Directory not found: {args.comparison_results_dir}")
        return 1
    
    print("="*80)
    print("Model Comparison Results Aggregation")
    print("="*80)
    print(f"Input: {args.comparison_results_dir}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # Load results
    print("\nLoading comparison results...")
    df = load_comparison_results(args.comparison_results_dir)
    
    if len(df) == 0:
        print("Error: No comparison results found!")
        return 1
    
    print(f"✓ Loaded {len(df)} comparisons")
    print(f"  - Unique conditions: {df['condition'].nunique()}")
    print(f"  - Unique tasks: {df['task_name'].nunique()}")
    print(f"  - Unique model pairs: {len(df.groupby(['model1_name', 'model2_name']))}")
    
    # Aggregate by configuration
    print("\nAggregating by configuration...")
    aggregated_df = aggregate_by_configuration(df)
    print(f"✓ Created {len(aggregated_df)} aggregated comparisons")
    
    # Create summary
    print("\nComputing summary statistics...")
    summary = create_summary_statistics(aggregated_df)
    
    # Save results
    print("\nSaving results...")
    results_csv = save_results(aggregated_df, summary, args.output_dir)
    
    # Create visualizations
    if args.create_visualizations:
        print("\nCreating summary visualizations...")
        create_summary_plots(aggregated_df, args.output_dir)
    
    # Print summary
    print_summary(summary)
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Main results file: {results_csv}")
    print("\nNext: Use consistency_visualization.py to create detailed plots")
    
    return 0


if __name__ == '__main__':
    exit(main())