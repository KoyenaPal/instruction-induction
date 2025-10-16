#!/bin/bash

# Comprehensive Model Consistency Analysis Pipeline
# This script aggregates pre-computed comparison results and creates visualizations

set -e  # Exit on any error

# Configuration
COMPARISON_RESULTS_DIR="comparison_results"  # Directory containing setting subdirectories
OUTPUT_DIR="consistency_analysis_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --comparison_results_dir)
            COMPARISON_RESULTS_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --comparison_results_dir <path> [options]"
            echo ""
            echo "Aggregates pre-computed BERTScore comparison results and creates visualizations."
            echo ""
            echo "Expected directory structure:"
            echo "  comparison_results_dir/"
            echo "    transfer_oss_full_text/"
            echo "      task1_bert_comparison_model1_vs_model2.json"
            echo "      task2_bert_comparison_model1_vs_model2.json"
            echo "      ..."
            echo "    transfer_opent_without_answer/"
            echo "      ..."
            echo "    ensemble_qwq_dapo_full_text/"
            echo "      ..."
            echo ""
            echo "Options:"
            echo "  --comparison_results_dir  Directory with setting subdirectories (required)"
            echo "  --output_dir             Output directory (default: consistency_analysis_results)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$COMPARISON_RESULTS_DIR" ]]; then
    echo "Error: --comparison_results_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

if [[ ! -d "$COMPARISON_RESULTS_DIR" ]]; then
    echo "Error: Comparison results directory does not exist: $COMPARISON_RESULTS_DIR"
    exit 1
fi

echo "==================================================================="
echo "Comprehensive Model Consistency Analysis Pipeline"
echo "==================================================================="
echo "Comparison results directory: $COMPARISON_RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "==================================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Aggregate comparison results
echo ""
echo "Step 1: Aggregating comparison results from all settings..."
echo "-------------------------------------------------------------------"
python model_comparison_script.py \
    --comparison_results_dir "$COMPARISON_RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --create_visualizations

if [[ $? -ne 0 ]]; then
    echo "Error: Result aggregation failed"
    exit 1
fi

echo "✓ Results aggregated successfully"

# Step 2: Create detailed visualizations
echo ""
echo "Step 2: Creating detailed visualizations..."
echo "-------------------------------------------------------------------"

if [[ ! -f "$OUTPUT_DIR/comprehensive_comparison_results.csv" ]]; then
    echo "Error: Aggregated results CSV not found!"
    exit 1
fi

python consistency_visualization.py \
    --results_csv "$OUTPUT_DIR/comprehensive_comparison_results.csv" \
    --output_dir "$OUTPUT_DIR/visualizations"

if [[ $? -ne 0 ]]; then
    echo "Error: Visualization creation failed"
    exit 1
fi

echo "✓ Visualizations created"

# Step 3: Generate summary report
echo ""
echo "Step 3: Generating summary report..."
echo "-------------------------------------------------------------------"

SUMMARY_FILE="$OUTPUT_DIR/analysis_summary_report.txt"

cat > "$SUMMARY_FILE" << EOF
===============================================================================
Model Consistency Analysis Summary Report
Generated: $(date)
===============================================================================

CONFIGURATION:
- Comparison Results Directory: $COMPARISON_RESULTS_DIR
- Number of Setting Directories: $(ls -d $COMPARISON_RESULTS_DIR/*/ 2>/dev/null | wc -l)

RESULTS OVERVIEW:
EOF

# Add summary statistics from JSON
if [[ -f "$OUTPUT_DIR/comparison_summary.json" ]]; then
    python3 -c "
import json
with open('$OUTPUT_DIR/comparison_summary.json', 'r') as f:
    data = json.load(f)
print(f'- Total Configurations: {data[\"total_configurations\"]}')
print(f'- Total Model Pairs: {data[\"total_model_pairs\"]}')
print(f'- Overall Mean Consistency: {data[\"overall_mean_consistency\"]:.4f}')
print(f'- Overall Std: {data[\"overall_std_consistency\"]:.4f}')
print('')
print('CONDITIONS ANALYZED:')
for condition in data['conditions']:
    print(f'  - {condition}')
print('')
print('PER-CONDITION STATISTICS:')
for condition, stats in data['per_condition_stats'].items():
    print(f'  {condition}:')
    print(f'    Mean: {stats[\"mean\"]:.4f}')
    print(f'    Std:  {stats[\"std\"]:.4f}')
    print(f'    Count: {stats[\"count\"]}')
" >> "$SUMMARY_FILE"
fi

cat >> "$SUMMARY_FILE" << EOF

FILES GENERATED:
- comprehensive_comparison_results.csv: Aggregated comparison results
- comparison_summary.json: Summary statistics
- pairwise_consistency_matrix.csv: Full pairwise consistency matrix
- summary_plots/: Basic summary visualizations
  - consistency_distribution.png: Distribution of all scores
  - consistency_by_condition.png: Per-condition breakdown
  - consistency_by_answer_condition.png: Full text vs without answer
- visualizations/: Detailed visualization plots
  - consistency_bar_chart_detailed.png: Main consistency chart
  - detailed_comparison_matrix.png: Heatmap of all comparisons
  - transfer_breakdown.png: Transfer-specific analysis
  - ensemble_breakdown.png: Ensemble-specific analysis
  - processed_detailed_data.csv: Processed data for custom plots

INTERPRETATION:
- Higher BERTScore values (closer to 1.0) indicate greater consistency
- Transfer conditions show how well models utilize reasoning from other models
- Answer completeness comparison reveals impact of including/excluding answers
- Ensemble methods combine multiple reasoning approaches for improved consistency

NEXT STEPS:
1. Review the main consistency bar chart for overall patterns
2. Examine transfer_breakdown.png to compare different source models
3. Check ensemble_breakdown.png to identify best ensemble combinations
4. Use the CSV files for custom analysis or additional visualizations

===============================================================================
EOF

echo "✓ Summary report generated: $SUMMARY_FILE"

# Step 4: Display key results
echo ""
echo "==================================================================="
echo "ANALYSIS COMPLETE"
echo "==================================================================="
echo ""

# Display directory structure of results
echo "Setting directories analyzed:"
ls -d $COMPARISON_RESULTS_DIR/*/ 2>/dev/null | xargs -n 1 basename | head -10
if [[ $(ls -d $COMPARISON_RESULTS_DIR/*/ 2>/dev/null | wc -l) -gt 10 ]]; then
    echo "... and $(($(ls -d $COMPARISON_RESULTS_DIR/*/ 2>/dev/null | wc -l) - 10)) more"
fi

echo ""
echo "Key Results:"
if [[ -f "$OUTPUT_DIR/comparison_summary.json" ]]; then
    python3 -c "
import json
with open('$OUTPUT_DIR/comparison_summary.json', 'r') as f:
    data = json.load(f)
print(f'Total configurations: {data[\"total_configurations\"]}')
print(f'Average consistency: {data[\"overall_mean_consistency\"]:.4f}')
print('')
print('Top 5 conditions by consistency:')
items = sorted(data['per_condition_stats'].items(), key=lambda x: x[1]['mean'], reverse=True)
for i, (condition, stats) in enumerate(items[:5]):
    print(f'  {i+1}. {condition}: {stats[\"mean\"]:.4f}')
"
fi

echo ""
echo "Output Directory: $OUTPUT_DIR"
echo "Key Files:"
echo "  - Analysis Summary: $SUMMARY_FILE"
echo "  - Aggregated Results: $OUTPUT_DIR/comprehensive_comparison_results.csv"
echo "  - Consistency Chart: $OUTPUT_DIR/visualizations/consistency_bar_chart_detailed.png"
echo "  - Transfer Analysis: $OUTPUT_DIR/visualizations/transfer_breakdown.png"
echo "  - Ensemble Analysis: $OUTPUT_DIR/visualizations/ensemble_breakdown.png"
echo ""
echo "==================================================================="

# Optional: Open output directory (uncomment if desired)
# if command -v open >/dev/null 2>&1; then
#     open "$OUTPUT_DIR"
# elif command -v xdg-open >/dev/null 2>&1; then
#     xdg-open "$OUTPUT_DIR"
# fi

echo "Analysis pipeline completed successfully!"