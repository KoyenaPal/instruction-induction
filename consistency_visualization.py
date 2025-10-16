#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import argparse

def parse_configuration_label(model, condition, answer_condition):
    """
    Create detailed labels for each configuration matching target chart format.
    """
    # Handle ensemble conditions
    if 'ensemble' in condition.lower():
        if 'dapo' in model.lower() and 'qwq' in condition.lower():
            label = 'Ensemble CoT QwQ + DAPO/OSS'
        elif 'opent' in model.lower() and 'qwq' in condition.lower():
            label = 'Ensemble CoT QwQ + OpenT/OSS'
        elif 'oss' in model.lower() and 'dapo' in condition.lower():
            label = 'Ensemble CoT QwQ + OSS/DAPO'
        else:
            # Extract ensemble partners from condition name if available
            label = f'Ensemble CoT {condition.replace("ensemble_", "").replace("_", " ").title()}'
    
    # Handle transfer conditions
    elif 'transfer' in condition.lower():
        if 'transfer_oss' in condition.lower():
            label = f'Transfer CoT OSS'
        elif 'transfer_opent' in condition.lower():
            label = f'Transfer CoT OpenT'
        elif 'transfer_dapo' in condition.lower():
            label = f'Transfer CoT DAPO'
        elif 'transfer_qwq' in condition.lower():
            label = f'Transfer CoT QwQ'
        elif 'transfer_nnr' in condition.lower():
            label = f'Transfer CoT NNR'
        else:
            label = f'Transfer CoT {condition.replace("transfer_", "").upper()}'
    
    # Handle basic conditions
    elif condition.lower() == 'empty':
        label = 'Empty'
    elif condition.lower() == 'default':
        label = 'Default'
    elif condition.lower() == 'sampled':
        label = 'Sampled'
    else:
        label = condition.replace('_', ' ').title()
    
    return label

def determine_thought_type(condition):
    """
    Determine the thought type category for coloring.
    """
    condition_lower = condition.lower()
    
    if 'empty' in condition_lower:
        return 'Empty'
    elif 'default' in condition_lower:
        return 'Default'
    elif 'sampled' in condition_lower:
        return 'Sampled'
    elif 'transfer' in condition_lower:
        return 'Transfer'
    elif 'ensemble' in condition_lower:
        return 'Ensemble'
    else:
        return 'Other'

def create_consistency_bar_chart(results_csv, output_path, figsize=(16, 9)):
    """
    Create a detailed bar chart matching your target consistency visualization.
    Shows specific transfer sources and ensemble combinations.
    """
    # Load results
    df = pd.read_csv(results_csv)
    
    # Define color mapping for thought types (matching your chart)
    color_map = {
        'Empty': '#E5E5E5',      # Light gray
        'Default': '#90C97B',     # Green
        'Sampled': '#F4A582',     # Salmon/pink
        'Transfer': '#92C5DE',    # Light blue
        'Ensemble': '#C4A5D4'     # Purple
    }
    
    # Prepare detailed plotting data
    plot_data = []
    
    # Process each row to create detailed configuration
    for _, row in df.iterrows():
        # Create detailed labels
        label1 = parse_configuration_label(row['model1_name'], row['condition1'], row['answer_condition1'])
        label2 = parse_configuration_label(row['model2_name'], row['condition2'], row['answer_condition2'])
        
        # Determine thought type for coloring
        thought_type1 = determine_thought_type(row['condition1'])
        thought_type2 = determine_thought_type(row['condition2'])
        
        # Add entry for configuration 1
        plot_data.append({
            'configuration': label1,
            'model': row['model1_name'],
            'condition': row['condition1'],
            'answer_completeness': row['answer_condition1'],
            'thought_type': thought_type1,
            'consistency_score': row['average_score'],
            'comparison_with': label2
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Group by configuration and answer completeness to get average scores
    grouped = plot_df.groupby(['configuration', 'thought_type', 'answer_completeness']).agg({
        'consistency_score': ['mean', 'std', 'count']
    }).reset_index()
    
    grouped.columns = ['configuration', 'thought_type', 'answer_completeness', 
                      'mean_score', 'std_score', 'count']
    
    # Define display order matching your chart
    order_priority = {
        'Ensemble CoT QwQ + DAPO/OSS': 1,
        'Ensemble CoT QwQ + OpenTv/OSS': 2,
        'Transfer CoT QwQ': 3,
        'Transfer CoT DAPO': 4,
        'Ensemble CoT QwQ + OSS/DAPO': 5,
        'Transfer CoT OpenT': 6,
        'Transfer CoT OSS': 7,
        'Transfer CoT NHR': 8,
        'Empty': 9,
        'Sampled': 10,
        'Default': 11
    }
    
    # Sort by priority (ensemble and transfer first, then by score)
    grouped['order'] = grouped['configuration'].map(lambda x: order_priority.get(x, 100))
    grouped = grouped.sort_values(['order', 'mean_score'], ascending=[True, False])
    
    # Create separate entries for full_text and without_answer
    full_text_data = grouped[grouped['answer_completeness'] == 'full_text'].copy()
    without_answer_data = grouped[grouped['answer_completeness'] == 'without_answer'].copy()
    
    # Merge to ensure matching pairs
    merged = pd.merge(
        full_text_data[['configuration', 'thought_type', 'mean_score', 'std_score']],
        without_answer_data[['configuration', 'mean_score', 'std_score']],
        on='configuration',
        how='outer',
        suffixes=('_full', '_without')
    )
    
    merged = merged.sort_values('mean_score_full', ascending=False)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up bar positions
    x = np.arange(len(merged))
    width = 0.35
    
    # Plot bars
    for i, (_, row) in enumerate(merged.iterrows()):
        thought_type = row['thought_type']
        color = color_map.get(thought_type, '#CCCCCC')
        
        # Full text bar (solid)
        if not pd.isna(row['mean_score_full']):
            bar1 = ax.bar(x[i] - width/2, row['mean_score_full'], width,
                         color=color, edgecolor='black', linewidth=1.2,
                         alpha=0.85, label='_nolegend_')
            
            # Add error bar if std available
            if not pd.isna(row['std_score_full']):
                ax.errorbar(x[i] - width/2, row['mean_score_full'], 
                           yerr=row['std_score_full'], fmt='none', 
                           color='black', capsize=3, linewidth=1)
            
            # Add value label
            ax.text(x[i] - width/2, row['mean_score_full'] + 0.02,
                   f"{row['mean_score_full']:.2f}",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Without answer bar (hatched)
        if not pd.isna(row['mean_score_without']):
            bar2 = ax.bar(x[i] + width/2, row['mean_score_without'], width,
                         color=color, edgecolor='black', linewidth=1.2,
                         alpha=0.85, hatch='///', label='_nolegend_')
            
            # Add error bar if std available
            if not pd.isna(row['std_score_without']):
                ax.errorbar(x[i] + width/2, row['mean_score_without'],
                           yerr=row['std_score_without'], fmt='none',
                           color='black', capsize=3, linewidth=1)
            
            # Add value label
            ax.text(x[i] + width/2, row['mean_score_without'] + 0.02,
                   f"{row['mean_score_without']:.2f}",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize axes
    ax.set_xlabel('Thought Variation', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Pairwise Target Consistency Rate', fontsize=13, fontweight='bold')
    ax.set_title('Model Consistency Across CoT Reasoning Conditions', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(merged['configuration'], rotation=45, ha='right', fontsize=10)
    
    # Set y-axis
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Create legends
    # Answer Completeness Legend (left side)
    answer_legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black', 
                 linewidth=1.2, label='Full Text', alpha=0.85),
        Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black',
                 linewidth=1.2, hatch='///', label='Without Answer', alpha=0.85)
    ]
    answer_legend = ax.legend(handles=answer_legend_elements,
                             title='Answer Completeness',
                             loc='upper left',
                             framealpha=0.95,
                             edgecolor='black',
                             fontsize=10,
                             title_fontsize=11)
    
    # Thought Type Legend (right side)
    thought_legend_elements = []
    for thought_type in ['Empty', 'Default', 'Sampled', 'Transfer', 'Ensemble']:
        if thought_type in color_map:
            thought_legend_elements.append(
                Rectangle((0, 0), 1, 1, facecolor=color_map[thought_type],
                         edgecolor='black', linewidth=1.2,
                         label=thought_type, alpha=0.85)
            )
    
    thought_legend = ax.legend(handles=thought_legend_elements,
                              title='Thought Type',
                              loc='upper right',
                              framealpha=0.95,
                              edgecolor='black',
                              fontsize=10,
                              title_fontsize=11)
    
    # Add both legends
    ax.add_artist(answer_legend)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved consistency bar chart to: {output_path}")
    plt.close()
    
    return merged

def create_detailed_comparison_matrix(results_csv, output_path):
    """
    Create a detailed heatmap showing all pairwise comparisons.
    """
    df = pd.read_csv(results_csv)
    
    # Create detailed configuration labels
    df['config1'] = df.apply(lambda x: parse_configuration_label(
        x['model1_name'], x['condition1'], x['answer_condition1']), axis=1)
    df['config2'] = df.apply(lambda x: parse_configuration_label(
        x['model2_name'], x['condition2'], x['answer_condition2']), axis=1)
    
    # Create pivot table
    pivot_df = df.pivot_table(index='config1', columns='config2',
                             values='average_score', aggfunc='mean')
    
    # Create figure
    plt.figure(figsize=(18, 16))
    
    # Create heatmap
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

def create_transfer_breakdown_chart(results_csv, output_path):
    """
    Create a focused chart showing only transfer condition performance.
    """
    df = pd.read_csv(results_csv)
    
    # Filter for transfer conditions only
    transfer_mask = df['condition1'].str.contains('transfer', case=False, na=False)
    transfer_df = df[transfer_mask].copy()
    
    if len(transfer_df) == 0:
        print("No transfer conditions found in data")
        return None
    
    # Create labels
    transfer_df['config'] = transfer_df.apply(
        lambda x: parse_configuration_label(x['model1_name'], x['condition1'], x['answer_condition1']),
        axis=1
    )
    
    # Group by configuration and answer completeness
    grouped = transfer_df.groupby(['config', 'answer_condition1']).agg({
        'average_score': ['mean', 'std', 'count']
    }).reset_index()
    
    grouped.columns = ['config', 'answer_completeness', 'mean_score', 'std_score', 'count']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Separate by answer completeness
    full_text = grouped[grouped['answer_completeness'] == 'full_text']
    without_answer = grouped[grouped['answer_completeness'] == 'without_answer']
    
    x = np.arange(len(full_text))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, full_text['mean_score'], width,
                   label='Full Text', color='#92C5DE', edgecolor='black')
    bars2 = ax.bar(x + width/2, without_answer['mean_score'], width,
                   label='Without Answer', color='#92C5DE', edgecolor='black',
                   hatch='///')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Transfer Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Consistency Score', fontsize=12, fontweight='bold')
    ax.set_title('Transfer CoT Performance Breakdown by Source Model',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(full_text['config'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved transfer breakdown chart to: {output_path}")
    plt.close()

def create_ensemble_breakdown_chart(results_csv, output_path):
    """
    Create a focused chart showing only ensemble condition performance.
    """
    df = pd.read_csv(results_csv)
    
    # Filter for ensemble conditions
    ensemble_mask = df['condition1'].str.contains('ensemble', case=False, na=False)
    ensemble_df = df[ensemble_mask].copy()
    
    if len(ensemble_df) == 0:
        print("No ensemble conditions found in data")
        return None
    
    # Create labels
    ensemble_df['config'] = ensemble_df.apply(
        lambda x: parse_configuration_label(x['model1_name'], x['condition1'], x['answer_condition1']),
        axis=1
    )
    
    # Group by configuration
    grouped = ensemble_df.groupby(['config', 'answer_condition1']).agg({
        'average_score': ['mean', 'std', 'count']
    }).reset_index()
    
    grouped.columns = ['config', 'answer_completeness', 'mean_score', 'std_score', 'count']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Separate by answer completeness
    full_text = grouped[grouped['answer_completeness'] == 'full_text']
    without_answer = grouped[grouped['answer_completeness'] == 'without_answer']
    
    x = np.arange(len(full_text))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, full_text['mean_score'], width,
                   label='Full Text', color='#C4A5D4', edgecolor='black')
    bars2 = ax.bar(x + width/2, without_answer['mean_score'], width,
                   label='Without Answer', color='#C4A5D4', edgecolor='black',
                   hatch='///')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Ensemble Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Consistency Score', fontsize=12, fontweight='bold')
    ax.set_title('Ensemble CoT Performance Breakdown by Model Combination',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(full_text['config'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved ensemble breakdown chart to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Create detailed consistency visualization plots")
    parser.add_argument('--results_csv', type=str, required=True,
                        help='CSV file with comprehensive comparison results')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save visualization plots')
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Creating detailed consistency bar chart...")
    merged_df = create_consistency_bar_chart(
        args.results_csv,
        f"{args.output_dir}/consistency_bar_chart_detailed.png"
    )
    
    print("Creating detailed comparison matrix...")
    create_detailed_comparison_matrix(
        args.results_csv,
        f"{args.output_dir}/detailed_comparison_matrix.png"
    )
    
    print("Creating transfer breakdown chart...")
    create_transfer_breakdown_chart(
        args.results_csv,
        f"{args.output_dir}/transfer_breakdown.png"
    )
    
    print("Creating ensemble breakdown chart...")
    create_ensemble_breakdown_chart(
        args.results_csv,
        f"{args.output_dir}/ensemble_breakdown.png"
    )
    
    # Save processed data
    if merged_df is not None:
        merged_df.to_csv(f"{args.output_dir}/processed_detailed_data.csv", index=False)
    
    print(f"\nAll visualizations saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - consistency_bar_chart_detailed.png: Main detailed chart")
    print("  - detailed_comparison_matrix.png: Full pairwise matrix")
    print("  - transfer_breakdown.png: Transfer-specific analysis")
    print("  - ensemble_breakdown.png: Ensemble-specific analysis")

if __name__ == '__main__':
    main()