#!/bin/bash

# Configuration
SCRIPT_PATH="compare_models_bert.py"
BASE_PRED_DIR="predictions"
OUTPUT_BASE_DIR="comparison_results"

# Define the 5 models
MODELS=(
    "BytedTsinghua-SIA_DAPO-Qwen-32B" 
    "nvidia_Nemotron-Research-Reasoning-Qwen-1.5B" 
    "Qwen_QwQ-32B" 
    "openai_gpt-oss-20b" 
    "open-thoughts_OpenThinker-7B"
)

# Define setups with their prediction directory patterns
declare -A SETUPS
#SETUPS["default"]="default"
SETUPS["default_without_answer"]="default_without_answer"

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

# Function to get shortened model name for without_answer paths
get_short_model_name() {
    local model=$1
    case $model in
        "Qwen_QwQ-32B")
            echo "qwq"
            ;;
        "BytedTsinghua-SIA_DAPO-Qwen-32B")
            echo "dapo"
            ;;
        "open-thoughts_OpenThinker-7B")
            echo "opent"
            ;;
        "openai_gpt-oss-20b")
            echo "oss"
            ;;
        "nvidia_Nemotron-Research-Reasoning-Qwen-1.5B")
            echo "nrr"
            ;;
        *)
            echo "$model"
            ;;
    esac
}

# Function to build prediction directory path based on setup name
build_pred_dir() {
    local setup_name=$1
    local pattern=$2
    local model_suffix=$3
    
    if [[ "$setup_name" == "default" ]]; then
        echo "${BASE_PRED_DIR}/predictions_${model_suffix}_thoughts_to_${model_suffix}"
    elif [[ "$setup_name" == "default_without_answer" ]]; then
        local short_name=$(get_short_model_name "$model_suffix")
        echo "${BASE_PRED_DIR}/predictions_without_answer_${short_name}_thoughts_to_${model_suffix}"
    elif [[ "$setup_name" == *"empty"* ]]; then
        echo "${BASE_PRED_DIR}/predictions_${model_suffix}_empty"
    elif [[ "$setup_name" == *"sampling"* ]]; then
        if [[ "$setup_name" == *"without_answer"* ]]; then
            echo "${BASE_PRED_DIR}/predictions_${model_suffix}_with_sampling_without_answer"
        else
            echo "${BASE_PRED_DIR}/predictions_${model_suffix}_with_sampling"
        fi
    elif [[ "$setup_name" == *"ensemble"* ]]; then
        echo "${BASE_PRED_DIR}/predictions_${model_suffix}_${pattern}"
    else
        echo "${BASE_PRED_DIR}/${pattern}${model_suffix}"
    fi
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
            
            pred_dir1=$(build_pred_dir "$setup_name" "$prefix" "$model1_suffix")
            pred_dir2=$(build_pred_dir "$setup_name" "$prefix" "$model2_suffix")
            
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