import psutil
import urllib.request
import json
from tqdm import tqdm

from prep_data import format_input


PROMPT = """ Given the input `{input}`, and correct output `{output}`,
score the model response `{model_response}` on a scale from 0 to 100, where 100 is the best score.
Respond with the integer number only."""    


def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")


if not ollama_running:
    raise RuntimeError(
        "Ollama not running. Launch ollama before proceeding."
)
print(">>", "Ollama running:", check_if_running("ollama"), "<<")


def query_model(prompt, model="llama3", ollama_port=11434):
    """Queries running ollama model to get prompt judged for instruction following."""
    
    url = f"http://localhost:{ollama_port}/api/chat"
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,  # interesting choice
            "num_ctx": 2048
        }
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )

    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = PROMPT.format(
            input = entry["input"],
            output = entry["output"],
            model_response = entry[json_key]
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


if __name__ == '__main__':
    test_data = json.loads(open("../output/test-data-w-responses.json", "r").read())
    
    print('Scoring model responses...')
    print(f"My fine-tune: {generate_model_scores(test_data[:5], 'response-it')}")
    print(f"Base model: {generate_model_scores(test_data[:5], 'response-base')}")