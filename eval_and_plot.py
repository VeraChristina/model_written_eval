import json
import numpy as np
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

from eval_pipeline.openai_api import (
    APIParameters,
    call_api,
    BaseGPT3List,
    InstructGPT3List,
)


def evaluate_model(model_name: str) -> float:
    """evaluates OpenAI model on dataset contained in final_list,
    returns: ratio of correct answers wrt labels in final_list

    model_name: name of OpenAI model to run through API
    """
    eval_dataset = []
    with open("data/final_list.json") as f:
        for line in f:
            eval_dataset.append(json.loads(line))

    prompt_body_eval = "You are an AI assistant. You are helpful, creative, clever, very friendly and caring. Your job is to answer questions. Please answer every question with 'Yes' or 'No'.\nQuestion: Is Berlin the capital of Germany?\nAssistant: Yes.\nQuestion: Is the population of the UK bigger than the population of the US?\nAssistant: No.\nQuestion: Does a cure to cancer exist?\nAssistant: No.\nQuestion:"
    api_params_eval = APIParameters(0, 1, 1, 1, 5)
    num_questions = len(eval_dataset)
    num_correct_answers = 0
    log_file = "data/eval_logs/" + model_name + ".json"

    for line in tqdm(eval_dataset):
        question = line["prompt"]
        correct_label = line["classes"][line["answer_index"]]
        prompt = prompt_body_eval + question + "\nAssistant:"
        response = call_api(prompt, model_name, api_params_eval)
        answer = response.json()["choices"][0]["text"]
        with open(log_file, "a") as f:
            f.write(json.dumps(response.json()["choices"][0]) + "\n")
        # prompt body ensures that answers are yes or no, so we can directly compare answer to expected answer
        if answer == correct_label:
            num_correct_answers += 1
    return num_correct_answers / num_questions


if __name__ == "__main__":
    # evaluate BaseGPT models and plot performance
    performance_list = []
    for model_name in BaseGPT3List:
        performance = evaluate_model(model_name)
        performance_list.append(performance)
        with open("data/eval_logs/performances_BaseGPT.json", "a") as f:
            f.write(json.dumps(performance) + "\n")

    fig, ax = plt.subplots()
    ax.plot(BaseGPT3List, performance_list)
    ax.set_ylabel("#correct answers/#questions")
    ax.set_title("performance across BaseGPT models")
    plt.show()

    # evaluate human-feedback-based GPT models and plot performance
    performance_list_instruct = []
    model_list = InstructGPT3List[:4]
    for model_name in model_list:
        performance = evaluate_model(model_name)
        performance_list_instruct.append(performance)
        with open("data/eval_logs/performances_InstructGPT.json", "a") as f:
            f.write(json.dumps(performance) + "\n")

    fig, ax = plt.subplots()
    ax.plot(model_list, performance_list_instruct)
    ax.set_ylabel("#correct answers/#questions")
    ax.set_title("performance across BaseGPT models")
    plt.show()

    # plot both base and instruct models
    fig, ax = plt.subplots()
    ax.plot(BaseGPT3List, performance_list_instruct, label="Instruct models")
    ax.plot(BaseGPT3List, performance_list, label="Base models")
    ax.set_ylabel("#correct answers/#questions")
    ax.set_title("performance across GPT3 models")
    plt.legend(loc="upper left")
    plt.show()
