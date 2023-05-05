from dataclasses import dataclass


@dataclass
class EvalPrompt:
    prompt: str
    classes: list[str]
    answer_index: int


def eval_prompt_to_dict(eval_prompt: EvalPrompt) -> dict:
    output = {}
    output["prompt"] = eval_prompt.prompt
    output["classes"] = eval_prompt.classes
    output["answer_index"] = eval_prompt.answer_index
    return output
