#!/bin/bash

# Script Name: run_original.sh

# # Command 1
# echo "Running Nemotron Ones..."
# python optimized_execute_instructions.py --execution_engine nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --input_dir data/induction_input --thought_type with_sampling_without_answer --source_folder without_answer_nrr_with_sampling

# # Command 2
# echo "Running GPT..."
# python optimized_execute_instructions.py --execution_engine openai/gpt-oss-20b --input_dir data/induction_input --thought_type with_sampling_without_answer --source_folder without_answer_oss_with_sampling


# # Command 4
# echo "Running Qwen/QwQ-32B..."
# python optimized_execute_instructions.py --execution_engine Qwen/QwQ-32B --input_dir data/induction_input --thought_type with_sampling_without_answer --source_folder without_answer_qwq_with_sampling

# # Command 5
# echo "Running BytedTsinghua-SIA/DAPO-Qwen-32B..."
# python optimized_execute_instructions.py --execution_engine BytedTsinghua-SIA/DAPO-Qwen-32B --input_dir data/induction_input  --thought_type with_sampling_without_answer --source_folder without_answer_dapo_with_sampling

# Command 3
echo "Running openthinker..."
python optimized_execute_instructions.py --execution_engine open-thoughts/OpenThinker-7B --input_dir data/induction_input --thought_type with_sampling --source_folder without_answer_opent_with_sampling

python optimized_execute_instructions.py --execution_engine open-thoughts/OpenThinker-7B --input_dir data/induction_input --thought_type with_sampling_without_answer --source_folder without_answer_opent_with_sampling

echo "All commands executed."