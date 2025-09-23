import argparse
import json
import openai
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import random
import re
import os
import numpy as np

random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'


# NOT COMPELTE SCRIPT






def run_execution_accuracy_openai(execution_engine, instruction_generation_model, task_name, openai_organization,
                        openai_api_key, input_dir, out_dir, max_tokens=2048):
    with open(f'{input_dir}/{instruction_generation_model}/{task_name}.json', encoding='utf-8') as f_examples:
        data = json.load(f_examples)

    openai.organization = openai_organization
    openai.api_key = openai_api_key

    output_ = dict()

    parameters = {
        'max_tokens': max_tokens,
        'top_p': 0,
        'temperature': 1,
        'logprobs': 5,
        'engine': execution_engine
    }
    for instruction_id, instruction_data in data.items():
        d = {}
        d['instruction'] = instruction_data['instruction']
        d['prediction_counter'] = instruction_data['prediction_counter']
        instruction_outputs = {}
        test_examples = instruction_data['test_inputs']
        for id_, example in test_examples.items():
            prompt = example['prompt']
            parameters['prompt'] = prompt

            response = openai.Completion.create(**parameters)

            instruction_outputs[id_] = dict()
            instruction_outputs[id_]['prompt'] = prompt
            instruction_outputs[id_]['prediction'] = response.choices[0].text

            if int(id_) % 100 == 0:
                print(f'generated {id_} predictions with OpenAI {execution_engine}')

        d['instruction_outputs'] = instruction_outputs
        output_[instruction_id] = d

    output_path = f'{out_dir}/{instruction_generation_model}'
    Path(output_path).mkdir(exist_ok=True)

    with open(f'{output_path}/{task_name}_execution.json', 'w', encoding='utf-8') as f_predictions:
        json.dump(output_, f_predictions, indent=2, ensure_ascii=False)