#!/bin/bash
set -e  # Stop execution if any command fails


# Command 1
# python execute_ensemble_instructions.py \
#   --execution_engine nvidia/Nemotron-Research-Reasoning-Qwen-1.5B \
#   --thought_type ensemble \
#   --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_opent_eval_oss

# # Command 2
# python execute_ensemble_instructions.py \
#   --execution_engine nvidia/Nemotron-Research-Reasoning-Qwen-1.5B \
#   --thought_type ensemble_without_answer \
#   --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_opent_eval_oss

# python evaluate.py --gen_model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --execution_input_dir data/induction_input --predictions_dir predictions_nvidia_Nemotron-Research-Reasoning-Qwen-1.5B_ensemble_without_answer_gen_qwq_opent_eval_oss

# python evaluate.py --gen_model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --execution_input_dir data/induction_input --predictions_dir predictions_nvidia_Nemotron-Research-Reasoning-Qwen-1.5B_ensemble_without_answer_gen_qwq_oss_eval_dapo

# python evaluate.py --gen_model nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --execution_input_dir data/induction_input --predictions_dir predictions_nvidia_Nemotron-Research-Reasoning-Qwen-1.5B_ensemble_without_answer_gen_qwq_dapo_eval_oss



# # Command 1

# python evaluate.py --gen_model open-thoughts/OpenThinker-7B --execution_input_dir data/induction_input --predictions_dir predictions_open-thoughts_OpenThinker-7B_ensemble_without_answer_gen_qwq_opent_eval_oss

# python evaluate.py --gen_model open-thoughts/OpenThinker-7B --execution_input_dir data/induction_input --predictions_dir predictions_open-thoughts_OpenThinker-7B_ensemble_without_answer_gen_qwq_oss_eval_dapo
  
# python execute_ensemble_instructions.py \
#   --execution_engine open-thoughts/OpenThinker-7B \
#   --thought_type ensemble \
#   --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_opent_eval_oss

# # Command 2
# python execute_ensemble_instructions.py \
#   --execution_engine open-thoughts/OpenThinker-7B \
#   --thought_type ensemble_without_answer \
#   --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_opent_eval_oss


# Command 1
# python execute_ensemble_instructions.py \
#   --execution_engine openai/gpt-oss-20b \
#   --thought_type ensemble_without_answer \
#   --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_dapo_eval_oss

# python evaluate.py --gen_model openai/gpt-oss-20b --execution_input_dir data/induction_input --predictions_dir predictions_openai_gpt-oss-20b_ensemble_without_answer_gen_qwq_dapo_eval_oss

# python evaluate.py --gen_model openai/gpt-oss-20b --execution_input_dir data/induction_input --predictions_dir predictions_openai_gpt-oss-20b_ensemble_without_answer_gen_qwq_oss_eval_dapo

# python evaluate.py --gen_model openai/gpt-oss-20b --execution_input_dir data/induction_input --predictions_dir predictions_openai_gpt-oss-20b_ensemble_without_answer_gen_qwq_opent_eval_oss

# # Command 2
# python execute_ensemble_instructions.py \
#   --execution_engine openai/gpt-oss-20b \
#   --thought_type ensemble_without_answer \
#   --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_opent_eval_oss

# # Command 3
# python execute_ensemble_instructions.py \
#   --execution_engine Qwen/QwQ-32B \
#   --thought_type ensemble \
#   --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_opent_eval_oss

# # Command 4
# python execute_ensemble_instructions.py \
#   --execution_engine Qwen/QwQ-32B \
#   --thought_type ensemble_without_answer \
#   --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_opent_eval_oss

python execute_ensemble_instructions.py \
  --execution_engine Qwen/QwQ-32B \
  --thought_type ensemble_without_answer \
  --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_dapo_eval_oss

# python evaluate.py --gen_model Qwen/QwQ-32B --execution_input_dir data/induction_input --predictions_dir predictions_Qwen_QwQ-32B_ensemble_without_answer_gen_qwq_dapo_eval_oss

# python evaluate.py --gen_model Qwen/QwQ-32B --execution_input_dir data/induction_input --predictions_dir predictions_Qwen_QwQ-32B_ensemble_without_answer_gen_qwq_oss_eval_dapo

# python evaluate.py --gen_model Qwen/QwQ-32B --execution_input_dir data/induction_input --predictions_dir predictions_Qwen_QwQ-32B_ensemble_without_answer_gen_qwq_opent_eval_oss

# # Command 5
# # python execute_ensemble_instructions.py \
# #   --execution_engine BytedTsinghua-SIA/DAPO-Qwen-32B \
# #   --thought_type ensemble \
# #   --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_opent_eval_oss

# # Command 6
# python execute_ensemble_instructions.py \
#   --execution_engine BytedTsinghua-SIA/DAPO-Qwen-32B \
#   --thought_type ensemble_without_answer \
#   --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_dapo_eval_oss

python execute_ensemble_instructions.py \
  --execution_engine BytedTsinghua-SIA/DAPO-Qwen-32B \
  --thought_type ensemble_without_answer \
  --source_folder without_answer_instruction_induction_ensemble_outputs_gen_qwq_dapo_eval_oss

# python evaluate.py --gen_model BytedTsinghua-SIA/DAPO-Qwen-32B --execution_input_dir data/induction_input --predictions_dir predictions_BytedTsinghua-SIA_DAPO-Qwen-32B_ensemble_without_answer_gen_qwq_dapo_eval_oss

# python evaluate.py --gen_model BytedTsinghua-SIA/DAPO-Qwen-32B --execution_input_dir data/induction_input --predictions_dir predictions_BytedTsinghua-SIA_DAPO-Qwen-32B_ensemble_without_answer_gen_qwq_oss_eval_dapo

# python evaluate.py --gen_model BytedTsinghua-SIA/DAPO-Qwen-32B --execution_input_dir data/induction_input --predictions_dir predictions_BytedTsinghua-SIA_DAPO-Qwen-32B_ensemble_without_answer_gen_qwq_opent_eval_oss

# echo "✅ All six commands executed successfully!"
