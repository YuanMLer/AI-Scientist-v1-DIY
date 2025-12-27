import json
import os
import re

import backoff
import openai

MAX_NUM_TOKENS = 4096

OLLAMA_MODELS = [
    "qwen3-coder:30b",
    "qwen3-next:latest",
    "qwen3-vl:235b"
]

AVAILABLE_LLMS = OLLAMA_MODELS


def create_client(model):
    if model not in OLLAMA_MODELS:
        raise ValueError(f"Model {model} not supported. Available models: {OLLAMA_MODELS}")
    
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://1.13.248.121:17719")
    print(f"Using Ollama API with model {model} at {ollama_base_url}.")
    
    openai.api_key = "ollama"
    openai.api_base = f"{ollama_base_url}/v1"
    
    return openai, model


@backoff.on_exception(backoff.expo, Exception)
def get_batch_responses_from_llm(
    msg,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.75,
    n_responses=1,
):
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *new_msg_history,
        ],
        temperature=temperature,
        max_tokens=MAX_NUM_TOKENS,
        n=n_responses,
        stop=None,
    )
    
    content = [r.message.content for r in response.choices]
    new_msg_history = [
        new_msg_history + [{"role": "assistant", "content": c}] for c in content
    ]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@backoff.on_exception(backoff.expo, Exception)
def get_response_from_llm(
    msg,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.75,
):
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *new_msg_history,
        ],
        temperature=temperature,
        max_tokens=MAX_NUM_TOKENS,
        n=1,
        stop=None,
    )
    
    content = response.choices[0].message.content
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output):
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue

    return None
