import os
from os.path import dirname
from functools import partial


SYSTEM_PROMPT = "You are given a math problem. Please reason step by step, and put your final answer within \\boxed{}."
REASONING_START = "<think>"
REASONING_END = "</think>" 


def get_root_dir():
    root = os.path.abspath('')
    project_name = 'rl-math-reasoning'
    print("Project name set as {}. Make sure it is correct".format(project_name))
    while root.split('/')[-1] != project_name:
        root = dirname(root)
    return root


def create_chat_template(tokenizer):

    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
            "{{ '{system_prompt}' + eos_token }}"\
            "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
        "{% endif %}"
    
    chat_template = chat_template.replace("'{system_prompt}'", f"'{SYSTEM_PROMPT}'")
    chat_template = chat_template.replace("'{reasoning_start}'", f"'{REASONING_START}'")    
    tokenizer.chat_template = chat_template

    return tokenizer


# format dataset for sft
def format_dataset(x, tokenizer, add_think=False):

    think_token = REASONING_START if add_think else ''
    messages =  [
        {"role" : "system",    "content" : SYSTEM_PROMPT},
        # add <think> token after question
        {"role" : "user",      "content" : x['question']+think_token},
        {"role" : "assistant", "content" : x['trace']},
    ]
    # no generation_prompt because this is sft, needed for rl
    x['text'] = tokenizer.apply_chat_template(messages, tokenize=False) 

    return x


def process_sft_dataset(dataset, tokenizer, config):
    dataset = dataset.map(
        partial(
            format_dataset,
            tokenizer=tokenizer,
            add_think=config.add_think,
        ),
        batched=False,
        remove_columns=dataset.column_names,
    ).shuffle(config.seed)

    return dataset