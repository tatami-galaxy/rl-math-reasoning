import sys
sys.path.append('../../')
import re
from collections.abc import Callable
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def get_reward_functions(config):
    reward_functions = []
    if config.acc_reward:
        reward_functions.append(accuracy_reward)
    if config.think_reward:
        reward_functions.append(think_format_reward)
    if config.overlong_reward:
        reward_functions.append(get_soft_overlong_punishment)
    return reward_functions


# Reward function that checks if the completion matches the ground truth.
# If both gold and prediction are parseable → use math verification.
# If gold is not parseable → return `None` to skip the example.
def accuracy_reward(**kwargs):
    # each kwarg list of effective batch size
    x_types = kwargs['type']
    contents = [completion[0]["content"] for completion in kwargs['completions']]
    solutions = kwargs['solution']
    rewards = []
    for i in range(len(contents)):
        content = contents[i]
        sol = solutions[i]
        x_type = x_types[i]
        # accuracy reward only for nl
        if x_type != 'nl': reward = None
        else:
            gold_parsed = parse(sol)
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                """answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(units=True),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )"""
                answer_parsed = parse(content)
                reward = float(verify(gold_parsed, answer_parsed))
            else:
                # TODO : verify what is meant by skipping
                # if the gold solution cannot be parsed, we assign `None` to skip this example
                reward = None
        rewards.append(reward)
    return rewards


# Reward function that removes the reasoning content and checks if the final answer matches the ground truth.
# If both gold and prediction are parseable → use math verification.
# If gold is not parseable → return `None` to skip the example.
# TODO : edit
def reasoning_accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    reasoning_delimiters: list[str] | None = None,
    **kwargs,
) -> list[float | None]:
    
    if reasoning_delimiters is None:
        # Use sensible defaults for majority of reasoning models
        reasoning_delimiters = ["</think>"]

    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    for content, sol in zip(contents, solution, strict=True):
        # Split final answer from reasoning content
        is_reasoning_complete = False
        for delim in reasoning_delimiters:
            if delim in content:
                content = content.split(delim)[-1]
                is_reasoning_complete = True
                break
        if not is_reasoning_complete:
            # We assign zero reward instead of `None` to penalize incomplete reasoning
            rewards.append(0.0)
            continue

        gold_parsed = parse(sol)
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        boxed_match_priority=0,
                        normalization_config=NormalizationConfig(
                            units=True,
                        ),
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            reward = float(verify(gold_parsed, answer_parsed))
        else:
            # If the gold solution cannot be parsed, we assign `None` to skip this example
            reward = None
        rewards.append(reward)

    return rewards


# Reward function that checks if the reasoning process is enclosed within `"<think>"` and `"</think>"` tags.
# The function returns a reward of 1.0 if the format is correct, otherwise 0.0.
# TODO : edit
def think_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
   
    pattern = r"^<think>(?!.*<think>)(.*?)</think>.*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


# Reward function that penalizes overlong completions.
# It is used to penalize overlong completions, but not to reward shorter completions. 
# Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)
def get_soft_overlong_punishment(max_completion_len: int, soft_punish_cache: int) -> Callable:

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Reward function that penalizes overlong completions."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward
