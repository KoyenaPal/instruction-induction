#!/bin/bash
# Define arrays of models and prompt styles
sourcemodels=(
    "openai/gpt-oss-20b"
)
targetmodels=(
    "BytedTsinghua-SIA/DAPO-Qwen-32B" 
    "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B" 
    "Qwen/QwQ-32B" 
    "openai/gpt-oss-20b" 
    "open-thoughts/OpenThinker-7B"
)

# Define source folder mapping
declare -A source_folders
source_folders["open-thoughts/OpenThinker-7B"]="without_answer_opent"
source_folders["openai/gpt-oss-20b"]="without_answer_oss"
source_folders["BytedTsinghua-SIA/DAPO-Qwen-32B"]="without_answer_dapo"
source_folders["Qwen/QwQ-32B"]="without_answer_qwq"
source_folders["nvidia/Nemotron-Research-Reasoning-Qwen-1.5B"]="without_answer_nrr"

# Create logs directory
mkdir -p logs

# Optional delay between runs (in seconds)
delay=5

echo "Starting batch execution with ${#sourcemodels[@]} source models and ${#targetmodels[@]} target models..."
echo "Using transfer_without_answer thought type..."

# Loop through all combinations
for sourcemodel in "${sourcemodels[@]}"; do
    for targetmodel in "${targetmodels[@]}"; do
        # Sanitize names for filenames
        safe_sourcemodel="${sourcemodel//\//_}"
        safe_targetmodel="${targetmodel//\//_}"
        
        # Get the appropriate source folder
        source_folder="${source_folders[$sourcemodel]}"
        
        outputpath="predictions_${safe_sourcemodel}"
        predictionspath="predictions_${safe_sourcemodel}_thoughts_to_${safe_targetmodel}"
        logfile="logs/${safe_sourcemodel}_to_${safe_targetmodel}_without_answer.log"
        
        echo "----------------------------------------"
        echo "Source: $sourcemodel"
        echo "Target: $targetmodel"
        echo "Source folder: $source_folder"
        echo "Output path: $outputpath"
        echo "Predictions path: $predictionspath"
        echo "Log file: $logfile"
        echo "----------------------------------------"
        
        # Run the execute_instructions command
        # python optimized_execute_instructions.py \
        #     --execution_engine "$targetmodel" \
        #     --input_dir data/induction_input \
        #     --thought_type transfer_without_answer \
        #     --source_folder "$source_folder" \
        #     > "$logfile" 2>&1
        
        # # Check exit code for execute_instructions
        # if [ $? -ne 0 ]; then
        #     echo "❌ Error running optimized_execute_instructions.py: source=$sourcemodel target=$targetmodel (check $logfile)"
        #     # Uncomment next line to exit on error:
        #     # exit 1
        #     continue
        # else
        #     echo "✅ Successfully completed execute_instructions.py: source=$sourcemodel target=$targetmodel"
        # fi
        
        # Run the evaluate command
        python evaluate.py \
            --gen_model "$targetmodel" \
            --execution_input_dir data/induction_input \
            --predictions_dir "$predictionspath" \
            >> "$logfile" 2>&1
        
        # Check exit code for evaluate
        if [ $? -ne 0 ]; then
            echo "❌ Error running evaluate.py: source=$sourcemodel target=$targetmodel (check $logfile)"
            # Uncomment next line to exit on error:
            # exit 1
        else
            echo "✅ Successfully completed evaluate.py: source=$sourcemodel target=$targetmodel"
        fi
        
        # Optional delay between runs
        if [ $delay -gt 0 ]; then
            echo "Sleeping $delay seconds..."
            sleep $delay
        fi
    done
done

echo "✅ All runs completed."