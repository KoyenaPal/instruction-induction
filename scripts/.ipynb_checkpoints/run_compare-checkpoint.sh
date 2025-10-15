#!/bin/bash

# Configuration
SCRIPT_PATH="compare_models_bert.py"  # Update this to your script path

# Define the 5 models
MODELS=(
    "BytedTsinghua-SIA_DAPO-Qwen-32B" 
    "nvidia_Nemotron-Research-Reasoning-Qwen-1.5B" 
    "Qwen_QwQ-32B" 
    "openai_gpt-oss-20b" 
    "open-thoughts_OpenThinker-7B"
)

# Define setups with their prediction directory prefixes
declare -A SETUPS
# SETUPS["transfer_oss"]="predictions_openai_gpt-oss-20b_thoughts_to_"
# SETUPS["transfer_oss_without_answer"]="predictions_without_answer_oss_thoughts_to_"
# Add more setups as needed:
# SETUPS["transfer_qwq"]="predictions_Qwen_QwQ-32B_thoughts_to_"
# SETUPS["transfer_qwq_without_answer"]="predictions_without_answer_qwq_thoughts_to_"
# SETUPS["transfer_nrr"]="predictions_nvidia_Nemotron-Research-Reasoning-Qwen-1.5B_thoughts_to_"
# SETUPS["transfer_nrr_without_answer"]="predictions_without_answer_nrr_thoughts_to_"
# SETUPS["transfer_dapo"]="predictions_BytedTsinghua-SIA_DAPO-Qwen-32B_thoughts_to_"
# SETUPS["transfer_dapo_without_answer"]="predictions_without_answer_dapo_thoughts_to_"
SETUPS["transfer_opent"]="predictions_open-thoughts_OpenThinker-7B_thoughts_to_"
SETUPS["transfer_opent_without_answer"]="predictions_without_answer_opent_thoughts_to_"
# SETUPS["ensemble"]="predictions_ensemble_"
# SETUPS["default"]="predictions_default_"

# Tasks (modify if needed)
# TASKS="cause_and_effect,larger_animal,num_to_verbal,orthography_starts_with,rhymes,synonyms,taxonomy_animal,translation_en-fr,reverse_from_middle,smallest_item_length,smallest_even_no_sqrt,most_vowel_return_consonant,detect_rhyme_and_rewrite,rank_by_protein,multi_lang_to_english,square_of_zodiac_animal,alternate_synonym_antonym,most_consonant_return_vowel,least_unique_word_count,first_word_alphabetically_return_reverse"

# Base directory for predictions (modify as needed)
BASE_PRED_DIR="predictions"

# Output base directory
OUTPUT_BASE_DIR="comparison_results"

# Function to get model-specific suffix for directory names
get_model_suffix() {
    local model=$1
    case $model in
        "Qwen_QwQ-32B")
            echo "Qwen_QwQ-32B"
            ;;
        "BytedTsinghua-SIA_DAPO-Qwen-32B")
            echo "BytedTsinghua-SIA_DAPO-Qwen-32B"
            ;;
        "open-thoughts_OpenThinker-7B")
            echo "open-thoughts_OpenThinker-7B"
            ;;
        "openai_gpt-oss-20b")
            echo "openai_gpt-oss-20b"
            ;;
        "nvidia_Nemotron-Research-Reasoning-Qwen-1.5B")
            echo "nvidia_Nemotron-Research-Reasoning-Qwen-1.5B"
            ;;
        *)
            echo "$model"
            ;;
    esac
}

# Main execution
echo "=========================================="
echo "Pairwise Model Comparison Script"
echo "=========================================="
echo "Models: ${MODELS[@]}"
echo "Number of comparisons per setup: $((${#MODELS[@]} * (${#MODELS[@]} - 1) / 2))"
echo "Setups: ${!SETUPS[@]}"
echo "=========================================="
echo ""

# Loop through each setup
for setup_name in "${!SETUPS[@]}"; do
    prefix="${SETUPS[$setup_name]}"
    output_dir="${OUTPUT_BASE_DIR}/${setup_name}"
    
    echo "Processing setup: $setup_name"
    echo "Prediction prefix: $prefix"
    echo "Output directory: $output_dir"
    echo ""
    
    mkdir -p "$output_dir"
    
    # Generate all pairwise combinations
    comparison_count=0
    for ((i=0; i<${#MODELS[@]}; i++)); do
        for ((j=i+1; j<${#MODELS[@]}; j++)); do
            model1="${MODELS[$i]}"
            model2="${MODELS[$j]}"
            
            model1_suffix=$(get_model_suffix "$model1")
            model2_suffix=$(get_model_suffix "$model2")
            
            pred_dir1="${BASE_PRED_DIR}/${prefix}${model1_suffix}"
            pred_dir2="${BASE_PRED_DIR}/${prefix}${model2_suffix}"
            
            comparison_count=$((comparison_count + 1))
            
            echo "----------------------------------------"
            echo "Comparison $comparison_count: $model1 vs $model2"
            echo "Pred dir 1: $pred_dir1"
            echo "Pred dir 2: $pred_dir2"
            echo "----------------------------------------"
            
            # Check if directories exist
            if [ ! -d "$pred_dir1" ]; then
                echo "WARNING: Directory not found: $pred_dir1"
                echo "Skipping this comparison."
                echo ""
                continue
            fi
            
            if [ ! -d "$pred_dir2" ]; then
                echo "WARNING: Directory not found: $pred_dir2"
                echo "Skipping this comparison."
                echo ""
                continue
            fi
            
            # Run the comparison
            python "$SCRIPT_PATH" \
                --model1_name "$model1" \
                --model2_name "$model2" \
                --predictions_dir1 "$pred_dir1" \
                --predictions_dir2 "$pred_dir2" \
                --output_dir "$output_dir"
            
            if [ $? -eq 0 ]; then
                echo "✓ Comparison completed successfully"
            else
                echo "✗ Comparison failed"
            fi
            echo ""
        done
    done
    
    echo "Completed setup: $setup_name ($comparison_count comparisons)"
    echo "=========================================="
    echo ""
done

echo "All comparisons completed!"
echo "Results saved in: $OUTPUT_BASE_DIR"