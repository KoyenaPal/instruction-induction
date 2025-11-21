#!/usr/bin/env python3
"""
Create detailed consistency visualization plots from aggregated comparison results.

Generates multiple visualization types:
- Main bar chart with all configurations (ordered by consistency)
- Detailed pairwise comparison matrix
- Transfer-specific breakdown
- Ensemble-specific breakdown

All charts use standard error bars (std / sqrt(n)).
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


COLOR_MAP = {
    'Empty': '#E5E5E5',
    'Default': '#90C97B',
    'Sampled': '#F4A582',
    'Transfer': '#92C5DE',
    'Ensemble': '#C4A5D4'
}


def parse_configuration_label(model, condition, answer_condition):
    """Create human-readable labels for each configuration."""
    condition_lower = condition.lower()
    
    if 'ensemble' in condition_lower:
        # Parse model name format: gen_model1_model2_eval_model3
        # e.g., 'gen_qwq_dapo_eval_oss' -> sources: [qwq, dapo], eval: oss
        if 'eval_' in model.lower():
            parts = model.lower().split('eval_')
            eval_model = parts[1].upper()
            
            # Extract gen models
            gen_part = parts[0].replace('gen_', '').rstrip('_')
            gen_models = [m.upper() for m in gen_part.split('_')]
            
            source_models = ' + '.join(gen_models)
            return f'Ensemble CoT {source_models} / {eval_model}'
        else:
            # Fallback to old logic
            return f'Ensemble CoT {condition.replace("ensemble_", "").replace("_", " ").title()}'
    
    elif 'transfer' in condition_lower:
        source = condition.replace('transfer_', '').upper()
        return f'Transfer CoT {source}'
    
    elif condition_lower == 'empty':
        return 'Empty'
    
    elif condition_lower == 'default':
        return 'Default'
    
    elif 'sampl' in condition_lower:  # Catches both 'sample', 'sampled', 'sampling'
        return 'Sampled'
    
    return condition.replace('_', ' ').title()

def determine_thought_type(condition):
    """Determine the thought type category for coloring."""
    condition_lower = condition.lower()
    
    if 'empty' in condition_lower:
        return 'Empty'
    elif 'default' in condition_lower:
        return 'Default'
    elif 'sampl' in condition_lower:  # Catches 'sample', 'sampled', 'sampling'
        return 'Sampled'
    elif 'transfer' in condition_lower:
        return 'Transfer'
    elif 'ensemble' in condition_lower:
        return 'Ensemble'
    
    return 'Other'


def prepare_plot_data(df):
    """Prepare and aggregate data for plotting."""
    df['configuration'] = df.apply(
        lambda row: parse_configuration_label(
            row['model1_name'], row['condition'], row['answer_condition']
        ), axis=1
    )
    df['thought_type'] = df['condition'].apply(determine_thought_type)
    
    grouped = df.groupby(['configuration', 'thought_type', 'answer_condition']).agg({
        'average_score': 'mean',
        'std_score': 'first',
        'num_tasks': 'first'
    }).reset_index()
    
    grouped['std_error'] = grouped['std_score'] / np.sqrt(grouped['num_tasks'])
    
    return grouped


def merge_answer_conditions(grouped_df):
    """Merge full_text and without_answer data for side-by-side plotting."""
    full_text = grouped_df[grouped_df['answer_condition'] == 'full_text'].copy()
    without_answer = grouped_df[grouped_df['answer_condition'] == 'without_answer'].copy()
    
    merged = pd.merge(
        full_text[['configuration', 'thought_type', 'average_score', 'std_error']],
        without_answer[['configuration', 'average_score', 'std_error']],
        on='configuration',
        how='outer',
        suffixes=('_full', '_without')
    )
    
    merged = merged.sort_values('average_score_full', ascending=False, na_position='last')
    
    return merged


def add_bar_with_error(ax, x_pos, score, std_error, color, hatch=None):
    """Add a single bar with error bar to the plot."""
    if pd.isna(score):
        return
    
    ax.bar(x_pos, score, 0.35,
           color=color, edgecolor='black', linewidth=1.2,
           alpha=0.85, hatch=hatch, label='_nolegend_')
    
    if not pd.isna(std_error) and std_error > 0:
        ax.errorbar(x_pos, score, yerr=std_error, fmt='none',
                   color='black', capsize=3, linewidth=1)
    
    ax.text(x_pos, score + 0.02, f"{score:.2f}",
           ha='center', va='bottom', fontsize=9, fontweight='bold')


def create_legends(ax):
    """Create and add both legends to the plot."""
    answer_legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black',
                 linewidth=1.2, label='Full Text', alpha=0.85),
        Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black',
                 linewidth=1.2, hatch='///', label='Without Answer', alpha=0.85)
    ]
    answer_legend = ax.legend(
        handles=answer_legend_elements,
        title='Answer Completeness',
        loc='upper left',
        framealpha=0.95,
        edgecolor='black',
        fontsize=10,
        title_fontsize=11
    )
    
    thought_legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP[t],
                 edgecolor='black', linewidth=1.2,
                 label=t, alpha=0.85)
        for t in ['Empty', 'Default', 'Sampled', 'Transfer', 'Ensemble']
    ]
    ax.legend(
        handles=thought_legend_elements,
        title='Thought Type',
        loc='upper right',
        framealpha=0.95,
        edgecolor='black',
        fontsize=10,
        title_fontsize=11
    )
    ax.add_artist(answer_legend)


def create_consistency_bar_chart(results_csv, output_path, figsize=(16, 9)):
    """Create main bar chart comparing all configurations."""
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} rows from CSV")
    
    grouped = prepare_plot_data(df)
    merged = merge_answer_conditions(grouped)
    
    print(f"Creating chart with {len(merged)} configurations")
    print("\nDebug - Configuration and Thought Type mapping:")
    for _, row in merged.iterrows():
        print(f"  {row['configuration']}: {row['thought_type']}")
    
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(merged))
    width = 0.35
    
    for i, (_, row) in enumerate(merged.iterrows()):
        color = COLOR_MAP.get(row['thought_type'], '#CCCCCC')
        print(f"Row {i}: thought_type='{row['thought_type']}', color={color}")
        add_bar_with_error(ax, x[i] - width/2, row['average_score_full'],
                          row['std_error_full'], color)
        add_bar_with_error(ax, x[i] + width/2, row['average_score_without'],
                          row['std_error_without'], color, hatch='///')
    
    ax.set_xlabel('Thought Variation', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Pairwise Target Consistency Rate', fontsize=13, fontweight='bold')
    ax.set_title('Model Consistency Across CoT Reasoning Conditions',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(merged['configuration'], rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    create_legends(ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved consistency bar chart to: {output_path}")
    plt.close()
    
    return merged


def create_detailed_comparison_matrix(results_csv, output_path):
    """Create heatmap showing all pairwise comparisons."""
    df = pd.read_csv(results_csv)
    
    df['config1'] = df.apply(
        lambda x: parse_configuration_label(x['model1_name'], x['condition'], x['answer_condition']),
        axis=1
    )
    df['config2'] = df.apply(
        lambda x: parse_configuration_label(x['model2_name'], x['condition'], x['answer_condition']),
        axis=1
    )
    
    pivot_df = df.pivot_table(index='config1', columns='config2',
                              values='average_score', aggfunc='mean')
    
    plt.figure(figsize=(18, 16))
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'BERTScore Consistency'},
                square=True, linewidths=0.5, linecolor='gray')
    
    plt.title('Comprehensive Pairwise Model Consistency Matrix\n(Detailed Transfer and Ensemble Breakdown)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Configuration 2', fontsize=12, fontweight='bold')
    plt.ylabel('Configuration 1', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved detailed comparison matrix to: {output_path}")
    plt.close()


def create_condition_breakdown_chart(results_csv, output_path, condition_type, color):
    """Create focused chart for specific condition type (transfer or ensemble)."""
    df = pd.read_csv(results_csv)
    
    mask = df['condition'].str.contains(condition_type, case=False, na=False)
    filtered_df = df[mask].copy()
    
    if len(filtered_df) == 0:
        print(f"No {condition_type} conditions found in data")
        return
    
    filtered_df['config'] = filtered_df.apply(
        lambda x: parse_configuration_label(x['model1_name'], x['condition'], x['answer_condition']),
        axis=1
    )
    
    grouped = filtered_df.groupby(['config', 'answer_condition']).agg({
        'average_score': 'mean',
        'std_score': 'first',
        'num_tasks': 'first'
    }).reset_index()
    grouped['std_error'] = grouped['std_score'] / np.sqrt(grouped['num_tasks'])
    grouped = grouped.sort_values('average_score', ascending=False)
    
    unique_configs = grouped['config'].unique()
    full_text = grouped[grouped['answer_condition'] == 'full_text'].set_index('config')
    without_answer = grouped[grouped['answer_condition'] == 'without_answer'].set_index('config')
    
    full_scores = [full_text.loc[c, 'average_score'] if c in full_text.index else 0 for c in unique_configs]
    without_scores = [without_answer.loc[c, 'average_score'] if c in without_answer.index else 0 for c in unique_configs]
    full_se = [full_text.loc[c, 'std_error'] if c in full_text.index else 0 for c in unique_configs]
    without_se = [without_answer.loc[c, 'std_error'] if c in without_answer.index else 0 for c in unique_configs]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(unique_configs))
    width = 0.35
    
    ax.bar(x - width/2, full_scores, width, label='Full Text',
           color=color, edgecolor='black',
           yerr=full_se, capsize=5, error_kw={'linewidth': 1, 'ecolor': 'black'})
    ax.bar(x + width/2, without_scores, width, label='Without Answer',
           color=color, edgecolor='black', hatch='///',
           yerr=without_se, capsize=5, error_kw={'linewidth': 1, 'ecolor': 'black'})
    
    for i, score in enumerate(full_scores):
        if score > 0:
            ax.text(x[i] - width/2, score + 0.01, f'{score:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    for i, score in enumerate(without_scores):
        if score > 0:
            ax.text(x[i] + width/2, score + 0.01, f'{score:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel(f'{condition_type.title()} Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Consistency Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{condition_type.title()} CoT Performance Breakdown by Source Model',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_configs, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {condition_type} breakdown chart to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create detailed consistency visualization plots"
    )
    parser.add_argument('--results_csv', type=str, required=True,
                       help='CSV file with comprehensive comparison results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save visualization plots')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Creating Consistency Visualizations")
    print("=" * 80)
    print(f"Input: {args.results_csv}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    print("\nCreating detailed consistency bar chart...")
    merged_df = create_consistency_bar_chart(
        args.results_csv,
        f"{args.output_dir}/consistency_bar_chart_detailed.png"
    )
    
    print("\nCreating detailed comparison matrix...")
    create_detailed_comparison_matrix(
        args.results_csv,
        f"{args.output_dir}/detailed_comparison_matrix.png"
    )
    
    print("\nCreating transfer breakdown chart...")
    create_condition_breakdown_chart(
        args.results_csv,
        f"{args.output_dir}/transfer_breakdown.png",
        'transfer',
        COLOR_MAP['Transfer']
    )
    
    print("\nCreating ensemble breakdown chart...")
    create_condition_breakdown_chart(
        args.results_csv,
        f"{args.output_dir}/ensemble_breakdown.png",
        'ensemble',
        COLOR_MAP['Ensemble']
    )
    
    if merged_df is not None:
        output_csv = f"{args.output_dir}/processed_detailed_data.csv"
        merged_df.to_csv(output_csv, index=False)
        print(f"\nSaved processed data to: {output_csv}")
    
    print("\n" + "=" * 80)
    print(f"All visualizations saved to: {args.output_dir}")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - consistency_bar_chart_detailed.png: Main detailed chart")
    print("  - detailed_comparison_matrix.png: Full pairwise matrix")
    print("  - transfer_breakdown.png: Transfer-specific analysis")
    print("  - ensemble_breakdown.png: Ensemble-specific analysis")
    print("  - processed_detailed_data.csv: Processed data for further analysis")
    print("=" * 80)


if __name__ == '__main__':
    main()