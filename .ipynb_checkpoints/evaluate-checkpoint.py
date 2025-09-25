"""
CSV Model Evaluation Analysis Pipeline

Processes model evaluation data with statistical analysis and publication-ready visualizations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_TYPES = ["orange-model", "blue-model", "green-model", "purple-model"]
CRITERIA = ["Clarity of Steps", "Ease of Following", "Confidence"]

MODEL_LABELS = {
    'orange-model': 'OSS',
    'blue-model': 'DAPO', 
    'green-model': 'QwQ+DAPO/OSS',
    'purple-model': 'QwQ+OSS/DAPO',
    'best-overall': 'Best Overall'
}

MODEL_COLORS = {
    'OSS': '#FFB366',           # Orange
    'DAPO': '#6BB6FF',          # Blue  
    'QwQ+DAPO/OSS': '#66D9A3',  # Green
    'QwQ+OSS/DAPO': '#B366FF',  # Purple
    'Best Overall': '#FFD666',   # Yellow
    'default': '#FFACAC'         # Pink fallback
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def process_csv_complete_with_diagnostics_and_drop(
    file_path, 
    output_dir="results", 
    visualizations_dir="visualizations",
    drop_incomplete=False, 
    run_ttests=True, 
    bonferroni_correction=True, 
    create_plots=True
):
    """
    Process model evaluation CSV with statistical analysis and visualizations.
    
    Returns: (expanded_dataframe, clean_dataframe, ttest_results)
    """
    # Setup
    action = "Dropping" if drop_incomplete else "Keeping"
    print(f"=== Model Evaluation Pipeline + {action} Incomplete Rows ===")
    print(f"📁 Output: {output_dir}" + (f" | Plots: {visualizations_dir}" if create_plots else ""))
    
    _create_directories(output_dir, visualizations_dir if create_plots else None)
    
    # Process data
    df_clean = _process_csv(file_path, drop_incomplete)
    df_expanded = _expand_data(df_clean)
    
    # Analysis and visualization
    ttest_results = _analyze_data(df_expanded, bonferroni_correction) if run_ttests and len(df_expanded) > 0 else None
    
    if create_plots and len(df_expanded) > 0:
        _create_plots(df_expanded, visualizations_dir, drop_incomplete)
    
    _save_data(df_clean, df_expanded, ttest_results, output_dir, drop_incomplete)
    
    print("\n✅ Analysis complete!")
    return df_expanded, df_clean, ttest_results

# =============================================================================
# DATA PROCESSING
# =============================================================================

def _process_csv(file_path, drop_incomplete):
    """Load and process CSV file."""
    print(f"📖 Loading: {file_path}")
    
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}")
    
    # Clean headers
    df = df.drop(df.index[0]).reset_index(drop=True)
    df.columns = df.iloc[0].tolist()
    df = df.drop(df.index[:2]).reset_index(drop=True)
    
    # Filter relevant columns
    target_cols = MODEL_TYPES + ["best-overall"]
    relevant_cols = [col for col in df.columns if any(target in str(col) for target in target_cols)]
    df = df[relevant_cols]
    
    # Handle incomplete rows
    complete_rows = _find_complete_rows(df)
    
    if drop_incomplete and len(complete_rows) < len(df):
        print(f"🗑️ Dropping {len(df) - len(complete_rows)} incomplete rows")
        df = df.iloc[complete_rows].reset_index(drop=True)
    else:
        if len(complete_rows) < len(df):
            print(f"⚠️ Keeping {len(df) - len(complete_rows)} incomplete rows")
    
    print(f"📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def _find_complete_rows(df):
    """Find rows with complete data (5 values per model+criterion)."""
    complete_rows = []
    all_types = MODEL_TYPES + ["best-overall"]
    
    for idx in df.index:
        is_complete = True
        for model in all_types:
            for criterion in CRITERIA:
                cols = [c for c in df.columns if model in str(c) and criterion in str(c)]
                if cols:
                    valid_count = sum(1 for col in cols if _is_valid_number(df.at[idx, col]))
                    if valid_count != 5:
                        is_complete = False
                        break
            if not is_complete:
                break
        if is_complete:
            complete_rows.append(idx)
    
    return complete_rows

def _is_valid_number(value):
    """Check if value is a valid number."""
    if pd.notna(value) and str(value).strip():
        try:
            float(str(value).strip())
            return True
        except (ValueError, TypeError):
            pass
    return False

def _expand_data(df):
    """Transform data from wide to long format."""
    print("🔄 Expanding to long format...")
    
    data = []
    
    # Regular criteria
    for model in MODEL_TYPES:
        for criterion in CRITERIA:
            cols = [c for c in df.columns if model in str(c) and criterion in str(c)]
            _extract_data(df, cols, model, criterion, data)
    
    # Best-overall ratings - transform rankings to same scale
    best_cols = [c for c in df.columns if 'best-overall' in str(c).lower()]
    for col in best_cols:
        model = _infer_model(col)
        _extract_data(df, [col], model, 'best-overall', data, 'best_overall', transform_ranking=True)
    
    df_expanded = pd.DataFrame(data)
    if len(df_expanded) > 0:
        df_expanded['model_label'] = df_expanded['model'].map(MODEL_LABELS)
        print(f"📈 Created {len(df_expanded)} data points")
    
    return df_expanded

def _extract_data(df, cols, model, criterion, data, group_prefix=None, transform_ranking=False):
    """Extract numeric data from columns."""
    group_name = f"{group_prefix}_{model}" if group_prefix else f"{model}_{criterion.replace(' ', '_')}"
    
    for col in cols:
        for idx in df.index:
            value = _get_float(df.at[idx, col])
            if value is not None:
                # Transform ranking values to same scale (6 - rank)
                if transform_ranking:
                    value = 6 - value
                
                data.append({
                    'original_row': idx, 'original_col': col, 'group': group_name,
                    'model': model, 'criterion': criterion, 'value': value
                })

def _get_float(value):
    """Convert value to float safely."""
    if pd.notna(value) and str(value).strip():
        try:
            return float(str(value).strip())
        except (ValueError, TypeError):
            pass
    return None

def _infer_model(col):
    """Infer model type from column name."""
    col_lower = str(col).lower()
    for model in MODEL_TYPES:
        if model.split('-')[0] in col_lower:
            return model
    return 'best-overall'

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def _analyze_data(df_expanded, bonferroni_correction):
    """Conduct pairwise t-tests with optional Bonferroni correction."""
    print("\n=== Statistical Analysis ===")
    
    results = []
    
    # Test regular criteria
    regular_data = df_expanded[df_expanded['criterion'] != 'best-overall']
    if len(regular_data) > 0:
        criteria = regular_data['criterion'].unique()
        models = regular_data['model'].unique()
        print(f"🧪 Testing {len(criteria)} criteria across {len(models)} models")
        
        for criterion in criteria:
            print(f"\n--- {criterion} ---")
            crit_data = regular_data[regular_data['criterion'] == criterion]
            for m1, m2 in combinations(models, 2):
                result = _run_ttest(crit_data, m1, m2, criterion)
                if result:
                    results.append(result)
    
    # Test best-overall
    best_data = df_expanded[df_expanded['criterion'] == 'best-overall']
    if len(best_data) > 0:
        print("\n--- Best Overall Rankings ---")
        models = best_data['model'].unique()
        for m1, m2 in combinations(models, 2):
            result = _run_ttest(best_data, m1, m2, 'best-overall')
            if result:
                results.append(result)
    
    return _process_test_results(results, bonferroni_correction)

def _run_ttest(data, model1, model2, criterion):
    """Perform t-test between two models."""
    data1 = data[data['model'] == model1]['value'].values  
    data2 = data[data['model'] == model2]['value'].values
    
    if len(data1) == 0 or len(data2) == 0:
        return None
    
    try:
        t_stat, p_val = stats.ttest_ind(data1, data2)
        cohens_d = _cohens_d(data1, data2)
        
        # Print result
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        label1, label2 = MODEL_LABELS.get(model1, model1), MODEL_LABELS.get(model2, model2)
        print(f"  {label1} vs {label2}: t={t_stat:.3f}, p={p_val:.4f}{sig}, d={cohens_d:.3f}")
        
        return {
            'criterion': criterion, 'model1': model1, 'model2': model2,
            'model1_label': label1, 'model2_label': label2,
            'model1_mean': np.mean(data1), 'model2_mean': np.mean(data2),
            'model1_std': np.std(data1, ddof=1), 'model2_std': np.std(data2, ddof=1),
            'model1_n': len(data1), 'model2_n': len(data2),
            'mean_diff': np.mean(data1) - np.mean(data2),
            't_statistic': t_stat, 'p_value': p_val, 'cohens_d': cohens_d,
            'significant_uncorrected': p_val < 0.05
        }
    except Exception:
        return None

def _cohens_d(data1, data2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(data1), len(data2)
    pooled_std = np.sqrt(((n1-1)*np.var(data1, ddof=1) + (n2-1)*np.var(data2, ddof=1)) / (n1+n2-2))
    return (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

def _process_test_results(results, bonferroni_correction):
    """Process statistical results with optional correction."""
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    if bonferroni_correction:
        n_tests = len(df)
        df['p_value_corrected'] = (df['p_value'] * n_tests).clip(upper=1.0)
        df['significant_corrected'] = df['p_value_corrected'] < 0.05
        
        print(f"\n🔧 Bonferroni correction: {n_tests} tests, α = {0.05/n_tests:.6f}")
        print(f"   Significant: {sum(df['significant_uncorrected'])} → {sum(df['significant_corrected'])}")
    
    # Print summary
    p_col = 'p_value_corrected' if bonferroni_correction and 'p_value_corrected' in df.columns else 'p_value'
    sig_results = df[df[p_col] < 0.05]
    
    print(f"\n📊 Summary{' (Bonferroni corrected)' if bonferroni_correction else ''}:")
    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            direction = ">" if row['mean_diff'] > 0 else "<"
            print(f"   {row['model1_label']} {direction} {row['model2_label']} on {row['criterion']}: "
                  f"p={row[p_col]:.4f}, d={row['cohens_d']:.3f}")
    else:
        print("   No significant differences (p < 0.05)")
    
    return df

# =============================================================================
# VISUALIZATION
# =============================================================================

def _create_plots(df_expanded, vis_dir, drop_incomplete):
    """Create all visualizations."""
    print("\n=== Creating Visualizations ===")
    _setup_plot_style()
    
    # Individual criterion plots
    regular_data = df_expanded[df_expanded['criterion'] != 'best-overall']
    if len(regular_data) > 0:
        for criterion in regular_data['criterion'].unique():
            _plot_criterion(regular_data, criterion, vis_dir, drop_incomplete)
    
    # Best-overall plot
    best_data = df_expanded[df_expanded['criterion'] == 'best-overall'] 
    if len(best_data) > 0:
        _plot_best_overall(best_data, vis_dir, drop_incomplete)
    
    # Combined overview
    _plot_combined(df_expanded, vis_dir, drop_incomplete)
    
    print("📊 All visualizations completed!")

def _setup_plot_style():
    """Configure matplotlib for publication-quality output."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Font selection
    serif_fonts = ['DejaVu Serif', 'Times', 'Times New Roman', 'Liberation Serif', 'serif']
    font = next((f for f in serif_fonts if f in plt.rcParams['font.serif'] or f == 'serif'), 'serif')
    
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif', 'font.serif': [font],
        'figure.dpi': 300, 'axes.linewidth': 0.8, 'grid.alpha': 0.3,
        'legend.frameon': True, 'legend.shadow': True, 'legend.framealpha': 0.9
    })

def _plot_criterion(regular_data, criterion, vis_dir, drop_incomplete):
    """Create individual criterion plot."""
    crit_data = regular_data[regular_data['criterion'] == criterion]
    models = [MODEL_LABELS[m] for m in MODEL_TYPES if MODEL_LABELS[m] in crit_data['model_label'].unique()]
    
    plt.figure(figsize=(10, 6))
    _create_boxplot(crit_data, 'model_label', 'value', models)
    _style_plot(f'{criterion}', 'Model', 'Score', crit_data)
    _save_plot(vis_dir, f"boxplot_{criterion.replace(' ', '_')}", drop_incomplete)

def _plot_best_overall(best_data, vis_dir, drop_incomplete):
    """Create best-overall plot.""" 
    plt.figure(figsize=(10, 6))
    _create_boxplot(best_data, 'model_label', 'value')
    _style_plot('Best Overall Model Ranking', 'Model', 'Score', best_data)
    _save_plot(vis_dir, "boxplot_best_overall", drop_incomplete)

def _plot_combined(df_expanded, vis_dir, drop_incomplete):
    """Create combined overview plot with all criteria on single plot."""
    regular_data = df_expanded[df_expanded['criterion'] != 'best-overall']
    best_data = df_expanded[df_expanded['criterion'] == 'best-overall']
    
    # Combine regular criteria and best-overall data
    combined_data = []
    
    if len(regular_data) > 0:
        combined_data.append(regular_data)
    
    if len(best_data) > 0:
        best_data_copy = best_data.copy()
        best_data_copy['criterion'] = 'Best Overall'
        combined_data.append(best_data_copy)
    
    if not combined_data:
        return
    
    # Combine all data
    all_data = pd.concat(combined_data, ignore_index=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get models in consistent order
    models = [MODEL_LABELS[m] for m in MODEL_TYPES if MODEL_LABELS[m] in all_data['model_label'].unique()]
    
    # Create custom legend at top
    criteria = all_data['criterion'].unique()
    criteria_colors = []
    legend_labels = []
    
    for i, criterion in enumerate(criteria):
        if criterion == 'Best Overall':
            criteria_colors.append('#FFD700')  # Bright gold
            legend_labels.append('Best Overall')
        else:
            regular_criteria = [c for c in criteria if c != 'Best Overall']
            idx = list(regular_criteria).index(criterion)
            criteria_colors.append(plt.cm.Set2(idx / max(1, len(regular_criteria) - 1)))
            legend_labels.append(criterion)
    
    # Create legend elements
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, edgecolor='gray', linewidth=1.5) 
                      for color in criteria_colors]
    
    # Position legend at top of plot, in single horizontal row
    legend = ax.legend(legend_elements, legend_labels, 
                      title='Criteria', 
                      bbox_to_anchor=(0.5, 1.02),
                      loc='lower center',
                      ncol=len(criteria),
                      fontsize=11,
                      title_fontsize=12,
                      frameon=True,
                      fancybox=True,
                      shadow=True)
    
    # Highlight Best Overall in legend
    for i, label in enumerate(legend_labels):
        if 'Best Overall' in label:
            legend.get_texts()[i].set_weight('bold')
            legend.get_texts()[i].set_color('#CC8800')
    
    # Create grouped boxplot
    _create_grouped_boxplot(all_data, models, ax)
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    
    ax.set_facecolor('#FAFAFA')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    _save_plot(vis_dir, "boxplot_combined", drop_incomplete)

def _create_boxplot(data, x, y, order=None):
    """Create styled boxplot with model-specific colors."""
    models = order or data[x].unique()
    colors = [MODEL_COLORS.get(model, MODEL_COLORS['default']) for model in models]
    
    box_plot = sns.boxplot(data=data, x=x, y=y, hue=x, order=order,
                          palette=colors, linewidth=1.2, fliersize=4, legend=False)
    
    # Style enhancements
    for patch in box_plot.artists:
        patch.set_alpha(0.8)
        patch.set_edgecolor('gray')
    
    for line in box_plot.lines[4::6]:  # Medians
        line.set_color('darkred')
        line.set_linewidth(2.5)
    
    for flier in box_plot.collections:  # Outliers
        flier.set_markerfacecolor('lightcoral')
        flier.set_alpha(0.7)
    
    return box_plot

def _create_grouped_boxplot(data, models, ax=None):
    """Create grouped boxplot with all criteria on same plot."""
    if ax is None:
        ax = plt.gca()
        
    criteria = data['criterion'].unique()
    n_criteria = len(criteria)
    n_models = len(models)
    
    # Set up positions for grouped boxes
    width = 0.8 / n_criteria
    
    # Create custom color palette with bright color for Best Overall
    criteria_colors = []
    for criterion in criteria:
        if criterion == 'Best Overall':
            criteria_colors.append('#FFD700')  # Bright gold
        else:
            regular_criteria = [c for c in criteria if c != 'Best Overall']
            if regular_criteria:
                idx = list(regular_criteria).index(criterion)
                criteria_colors.append(plt.cm.Set2(idx / max(1, len(regular_criteria) - 1)))
            else:
                criteria_colors.append(plt.cm.Set2(0))
    
    # Plot each criterion
    for i, criterion in enumerate(criteria):
        crit_data = data[data['criterion'] == criterion]
        
        # Calculate positions for this criterion's boxes
        base_positions = np.arange(n_models)
        offset = (i - (n_criteria - 1) / 2) * width
        crit_positions = base_positions + offset
        
        # Prepare data for each model
        model_data = []
        for model in models:
            model_values = crit_data[crit_data['model_label'] == model]['value'].values
            model_data.append(model_values)
        
        # Create boxplot with vibrant color for Best Overall
        if criterion == 'Best Overall':
            alpha = 0.9  # More vibrant
            edge_color = '#CC8800'
        else:
            alpha = 0.7  # Regular
            edge_color = 'gray'
        
        box_plot = ax.boxplot(model_data, positions=crit_positions, widths=width*0.8,
                             patch_artist=True, 
                             boxprops=dict(facecolor=criteria_colors[i], alpha=alpha, edgecolor=edge_color),
                             medianprops=dict(color='darkred', linewidth=2.5),
                             flierprops=dict(marker='o', markerfacecolor='lightcoral', 
                                           markersize=4, alpha=0.7),
                             whiskerprops=dict(linewidth=1.0),
                             capprops=dict(linewidth=1.0))
        
        # Add means
        for j, model in enumerate(models):
            model_values = crit_data[crit_data['model_label'] == model]['value'].values
            if len(model_values) > 0:
                mean_val = np.mean(model_values)
                ax.plot(crit_positions[j], mean_val, marker='D', color='darkred', 
                       markersize=6, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    
    # Set x-axis labels
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models)

def _style_plot(title, xlabel, ylabel, data, invert_y=False):
    """Apply plot styling."""
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')  
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    _add_means(data)
    
    if invert_y:
        plt.gca().invert_yaxis()
    
    plt.legend(loc='upper right', fontsize=11)
    plt.gca().set_facecolor('#FAFAFA')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

def _add_means(data):
    """Add mean markers."""
    means = data.groupby('model_label')['value'].mean()
    for i, model in enumerate(data['model_label'].unique()):
        if model in means.index:
            plt.plot(i, means[model], marker='D', color='darkred', markersize=10,
                    markeredgecolor='white', markeredgewidth=2, 
                    label='Mean' if i == 0 else "", zorder=10)

# =============================================================================
# UTILITIES
# =============================================================================

def _create_directories(*dirs):
    """Create directories.""" 
    for d in dirs:
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)

def _save_plot(vis_dir, filename, drop_incomplete):
    """Save plot as PDF."""
    suffix = '_complete_only' if drop_incomplete else '_all_rows'
    path = Path(vis_dir) / f"{filename}{suffix}.pdf"
    
    plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300,
                facecolor='white', pad_inches=0.1)
    print(f"✅ Plot saved: {path}")
    plt.close()

def _save_data(df_clean, df_expanded, ttest_results, output_dir, drop_incomplete):
    """Save all results."""
    suffix = '_complete_only' if drop_incomplete else '_all_rows'
    out_path = Path(output_dir)
    
    df_clean.to_csv(out_path / f'clean_data{suffix}.csv', index=False)
    df_expanded.to_csv(out_path / f'expanded_data{suffix}.csv', index=False) 
    
    if ttest_results is not None and len(ttest_results) > 0:
        ttest_results.to_csv(out_path / f'statistical_tests{suffix}.csv', index=False)
    
    print(f"💾 Data saved in: {out_path}")

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    df_expanded, df_clean, ttests = process_csv_complete_with_diagnostics_and_drop(
        'user_study_raw_results.csv',
        output_dir='results_user_study', 
        visualizations_dir='plots_user_study'
    )