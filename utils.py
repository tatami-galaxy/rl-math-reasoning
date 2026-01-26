import os
from os.path import dirname
from functools import partial


SYSTEM_PROMPT = "You are given a math problem. Please reason step by step, and put your final answer within \\boxed{}."
SYSTEM_PROMPT_THINK = "You are given a math problem. Think carefully before producing the final answer. Put the thinking process between <think> and </think> tags. Reason step by step, and put your final answer within \\boxed{}."


def get_root_dir():
    root = os.path.abspath('')
    project_name = 'rl-math-reasoning'
    print("Project name set as {}".format(project_name))
    while root.split('/')[-1] != project_name:
        root = dirname(root)
    return root


# format dataset for sft
def format_dataset(x, tokenizer, add_think=False):

    think_token = '<think>' if add_think else ''
    messages =  [
        {"role" : "system",    "content" : SYSTEM_PROMPT},
        # add <think> token after question
        {"role" : "user",      "content" : x['question']+think_token},
        {"role" : "assistant", "content" : x['trace']},
    ]
    # standard language modeling : 
    # {"text": "The sky is blue."}
    # conversational language modeling : 
    # {"messages": [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}]}
    # when provided with a conversational dataset, the trainer will automatically apply the chat template to the datasetx
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


def process_rl_dataset(x):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT_THINK},
            {"role": "user", "content": x["problem"]},
        ],
    }


# merge all 3 trace columns from deepmath dataset
# map with batched=True, remove_columns=dataset.column_names, .shufle(seed)
def combine_deepmath(x):
    new_examples = {
        "answer": [],
        "question": [],
        "trace": [],
        "difficulty": [],
        "topic": [],
    }
    # all 3 traces
    for i, question in enumerate(x["question"]):
        traces = [x["r1_solution_1"][i], x["r1_solution_2"][i], x["r1_solution_3"][i]]
        for trace in traces:
            new_examples["answer"].append(x["final_answer"][i])
            new_examples["question"].append(question)
            new_examples["trace"].append(trace)
            new_examples["difficulty"].append(x["difficulty"][i])
            new_examples["topic"].append(x["topic"][i])
    return new_examples