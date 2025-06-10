import requests
import json
import time
import os
from datasets import load_dataset
import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
import concurrent.futures
import numpy as np
from evaluate import load



def install_llms() -> list:
    print('#########################    Installing LLMS    #########################')
    os.system('ollama pull mistral')
    os.system('ollama pull llama3')
    os.system('ollama pull gemma2')
    # os.system('ollama pull phi3')
    # os.system('ollama run deepseek-llm')
    return ['mistral', 'llama3', 'gemma2']


def dataset_init_conv_questions():
    dataset = load_dataset("conv_questions")
    train_data = dataset['train'].select(range(2))
    return train_data

def blender_full_init():
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")
    blender.loadfuser("llm-blender/gen_fuser_3b")
    return blender

def generate(llm, prompt)->list:
    url = 'http://127.0.0.1:11434/api/generate'
    f = open('input.json', 'r')
    data = json.load(f)
    data['model'] = llm
    data['prompt'] = prompt
    response = requests.post(url, json=data)
    assert response.status_code == 200
    dictionary = json.loads(response.text)
    f.close()
    return dictionary['response']

def experiment_multiprocess(blender, llmList, dataset)->list:
    fused_answers = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(len(dataset)):
            ans_set = []
            context = dataset[i]['seed_entity_text'] + "."
            for j in range(len(dataset[i]['questions'])):
                question = context + dataset[i]['questions'][j]
                candidates_texts = []
                results = [executor.submit(generate, llm, question) for llm in llmList]
                for f in concurrent.futures.as_completed(results):
                    candidates_texts.append(f.result())
                ranks = blender.rank(
                    [dataset[i]['questions'][j]], [candidates_texts], return_scores=False, batch_size=1)
                topk_candidates = get_topk_candidates_from_ranks(
                    ranks, [candidates_texts], top_k=2)
                fuse_generations = blender.fuse(
                    [dataset[i]['questions'][j]], topk_candidates, batch_size=2)
                ans_set.append(fuse_generations[0])
                context = context + fuse_generations[0]
            fused_answers.append(ans_set)

    filename = 'fused.txt'
    with open(filename, 'w') as f:
        for line in fused_answers:
            f.write(f"{line}\n")
    f.close()
    return fused_answers


if __name__ == '__main__':
    time.sleep(20)
    start = time.perf_counter()
    blender = blender_full_init()
    data = dataset_init_conv_questions()
    str = "{\"model\":\"\",\"prompt\":\"\",\"context\":[],\"stream\":false}"
    with open('input.json', 'w') as f:
        f.write(str)
    f.close()
    llmList = install_llms()

    results_1 = experiment_multiprocess(blender, llmList, data)
    bertscore = load("bertscore")

    scores = []
    for i in range(len(results_1)):
        score = bertscore.compute(predictions = results_1[i], references = data[i]['answer_texts'], lang="en")
        scores.append(np.average(score['f1']))

    print(scores)
    with open('scores.txt', 'w') as f:
        for score in scores:
            f.write(f"{score}\n")
    f.close()

    end = time.perf_counter()
    print(f'Finised in {round(end-start, 5)} seconds')
