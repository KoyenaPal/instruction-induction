#!/bin/bash

# Define arrays of models and prompt styles
# sourcemodels=(
#     "BytedTsinghua-SIA/DAPO-Qwen-32B" 
#     "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B" 
#     "Qwen/QwQ-32B" 
#     "openai/gpt-oss-20b" 
#     "open-thoughts/OpenThinker-7B"
# )
sourcemodels=(
    "Qwen/QwQ-32B" 
)
targetmodels=(
    "BytedTsinghua-SIA/DAPO-Qwen-32B" 
    "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B" 
    "Qwen/QwQ-32B" 
    "openai/gpt-oss-20b" 
    "open-thoughts/OpenThinker-7B"
)

# Create logs directory
mkdir -p logs

# Optional delay between runs (in seconds)
delay=5

echo "Starting batch execution with ${#sourcemodels[@]} source models and ${#targetmodels[@]} target models..."

# Loop through all combinations
for sourcemodel in "${sourcemodels[@]}"; do
    for targetmodel in "${targetmodels[@]}"; do
        # Sanitize names for filenames
        safe_sourcemodel="${sourcemodel//\//_}"
        safe_targetmodel="${targetmodel//\//_}"
        
        outputpath="predictions_${safe_sourcemodel}"
        predictionspath="predictions_${safe_sourcemodel}_thoughts_to_${safe_targetmodel}"
        logfile="logs/${safe_sourcemodel}_to_${safe_targetmodel}.log"
        
        echo "----------------------------------------"
        echo "Source: $sourcemodel"
        echo "Target: $targetmodel"
        echo "Output path: $outputpath"
        echo "Predictions path: $predictionspath"
        echo "Log file: $logfile"
        echo "----------------------------------------"
        
        # # Run the command
        # python execute_instructions.py \
        #     --execution_engine "$targetmodel" \
        #     --input_dir data/induction_input \
        #     --thought_type transfer \
        #     --source_folder "$outputpath" \
        #     > "$logfile" 2>&1

        # Run the command
        python evaluate.py \
            --gen_model "$targetmodel" \
            --execution_input_dir data/induction_input \
            --predictions_dir "$predictionspath" \
            > "$logfile" 2>&1
        
        
        # Check exit code
        if [ $? -ne 0 ]; then
            echo "❌ Error running: source=$sourcemodel target=$targetmodel (check $logfile)"
            # Uncomment next line to exit on error:
            # exit 1
        else
            echo "✅ Successfully completed: source=$sourcemodel target=$targetmodel"
        fi
        
        # Optional delay between runs
        if [ $delay -gt 0 ]; then
            echo "Sleeping $delay seconds..."
            sleep $delay
        fi
    done
done

echo "✅ All runs completed."