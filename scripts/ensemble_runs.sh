#!/bin/bash
set -e  # Stop execution if any command fails


# Command 1
python execute_ensemble_instructions.py \
  --execution_engine nvidia/Nemotron-Research-Reasoning-Qwen-1.5B \
  --thought_type ensemble \
  --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_oss_eval_dapo

# Command 2
python execute_ensemble_instructions.py \
  --execution_engine nvidia/Nemotron-Research-Reasoning-Qwen-1.5B \
  --thought_type ensemble_without_answer \
  --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_oss_eval_dapo




# Command 1
python execute_ensemble_instructions.py \
  --execution_engine open-thoughts/OpenThinker-7B \
  --thought_type ensemble \
  --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_oss_eval_dapo

# Command 2
python execute_ensemble_instructions.py \
  --execution_engine open-thoughts/OpenThinker-7B \
  --thought_type ensemble_without_answer \
  --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_oss_eval_dapo


# Command 1
python execute_ensemble_instructions.py \
  --execution_engine openai/gpt-oss-20b \
  --thought_type ensemble \
  --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_oss_eval_dapo

# Command 2
python execute_ensemble_instructions.py \
  --execution_engine openai/gpt-oss-20b \
  --thought_type ensemble_without_answer \
  --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_oss_eval_dapo

# Command 3
python execute_ensemble_instructions.py \
  --execution_engine Qwen/QwQ-32B \
  --thought_type ensemble \
  --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_oss_eval_dapo

# Command 4
python execute_ensemble_instructions.py \
  --execution_engine Qwen/QwQ-32B \
  --thought_type ensemble_without_answer \
  --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_oss_eval_dapo

# Command 5
python execute_ensemble_instructions.py \
  --execution_engine BytedTsinghua-SIA/DAPO-Qwen-32B \
  --thought_type ensemble \
  --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_oss_eval_dapo

# Command 6
python execute_ensemble_instructions.py \
  --execution_engine BytedTsinghua-SIA/DAPO-Qwen-32B \
  --thought_type ensemble_without_answer \
  --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_oss_eval_dapo

echo "✅ All six commands executed successfully!"
