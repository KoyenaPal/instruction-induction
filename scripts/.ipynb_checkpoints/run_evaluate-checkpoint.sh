#!/bin/bash

# Script Name: run_evaluate.sh

# Command 1
echo "Running Nemotron Ones..."
python evaluate.py --gen_model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --execution_input_dir data/induction_input --predictions_dir predictions_nvidia_Nemotron-Research-Reasoning-Qwen-1.5B_empty

# Command 2
echo "Running GPT..."
python execute_instructions.py --gen_model openai/gpt-oss-20b --execution_input_dir data/induction_input --predictions_dir predictions_openai_gpt-oss-20b_empty


# Command 3
echo "Running openthinker..."
python execute_instructions.py --gen_model open-thoughts/OpenThinker-7B --execution_input_dir data/induction_input --predictions_dir predictions_open-thoughts_OpenThinker-7B_empty


# Command 4
echo "Running Qwen/QwQ-32B..."
python execute_instructions.py --gen_model Qwen/QwQ-32B --execution_input_dir data/induction_input --predictions_dir predictions_Qwen_QwQ-32B_empty

# Command 5
echo "Running BytedTsinghua-SIA/DAPO-Qwen-32B..."
python execute_instructions.py --gen_model BytedTsinghua-SIA/DAPO-Qwen-32B --execution_input_dir data/induction_input --predictions_dir predictions_BytedTsinghua-SIA_DAPO-Qwen-32B_empty

echo "All commands executed."