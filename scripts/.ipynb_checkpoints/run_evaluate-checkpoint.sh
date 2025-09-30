#!/bin/bash

# Script Name: run_evaluate.sh
# Command 1
echo "Running Nemotron Ones..."
# python execute_instructions.py --execution_engine nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --input_dir data/induction_input --thought_type empty

python evaluate.py --gen_model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --execution_input_dir data/induction_input --predictions_dir predictions_nvidia_Nemotron-Research-Reasoning-Qwen-1.5B_with_sampling_without_answer

# Command 2
echo "Running GPT..."
#python execute_instructions.py --execution_engine openai/gpt-oss-20b --input_dir data/induction_input --thought_type empty

python evaluate.py --gen_model openai/gpt-oss-20b --execution_input_dir data/induction_input --predictions_dir predictions_openai_gpt-oss-20b_with_sampling_without_answer


# Command 4
echo "Running Qwen/QwQ-32B..."

#python execute_instructions.py --execution_engine Qwen/QwQ-32B --input_dir data/induction_input --thought_type empty

python evaluate.py --gen_model Qwen/QwQ-32B --execution_input_dir data/induction_input --predictions_dir predictions_Qwen_QwQ-32B_with_sampling_without_answer

# Command 5
echo "Running BytedTsinghua-SIA/DAPO-Qwen-32B..."

#python execute_instructions.py --execution_engine BytedTsinghua-SIA/DAPO-Qwen-32B --input_dir data/induction_input --thought_type empty

python evaluate.py --gen_model BytedTsinghua-SIA/DAPO-Qwen-32B --execution_input_dir data/induction_input --predictions_dir predictions_BytedTsinghua-SIA_DAPO-Qwen-32B_with_sampling_without_answer

# Command 3
echo "Running openthinker..."

#python execute_instructions.py --execution_engine open-thoughts/OpenThinker-7B --input_dir data/induction_input --thought_type empty

python evaluate.py --gen_model open-thoughts/OpenThinker-7B --execution_input_dir data/induction_input --predictions_dir predictions_open-thoughts_OpenThinker-7B_with_sampling_without_answer


python evaluate.py --gen_model open-thoughts/OpenThinker-7B --execution_input_dir data/induction_input --predictions_dir predictions_open-thoughts_OpenThinker-7B_with_sampling


echo "All commands executed."