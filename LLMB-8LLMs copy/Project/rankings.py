import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
import requests
import json
import time
import os
import random
from datasets import load_dataset

time.sleep(15)


def install_llms() -> list:
    print('#########################    Installing LLMS    #########################')
    os.system('ollama pull mistral')
    os.system('ollama pull gemma:2b')
    os.system('ollama pull qwen:4b')
    os.system('ollama pull llama2')
    # os.system('ollama pull vicuna')
    return ['mistral', 'gemma:2b', 'qwen:4b', 'llama2']
    # return ['mistral', 'qwen:4b', 'gemma:2b', 'tinyllama', 'vicuna']


def dataset_init():
    print('#########################    DatasetInit    #########################')
    dataset = load_dataset("conv_questions")

    test = []
    test = random.sample(list(dataset['test']), 100)
    inputs = []
    ref = []
    for i in range(len(test)):
        inputs.append(test[i]['questions'])
        ref.append(test[i]['answer_texts'])
    with open('ref.txt', 'w') as f:
        for line in ref:
            f.write(f"{line}\n")

    with open('ref.txt', 'r') as input_file:
        content = input_file.read()

    # Replace commas with new lines
    content = content.replace('], [', ']\n[')

    # Writing a ref file that'll be used to calculate scores
    with open('ref.txt', 'w') as output_file:
        # Write the modified content
        output_file.write(content)

    return inputs, ref


def lone_llm_output(llm, inputs) -> None:
    print('#########################    Lone LLM Outputs    #########################')
    f = open('input.json', 'r')

    data = json.load(f)
    data['model'] = llm
    f.close()
    with open('input.json', 'w') as json_file:
        json.dump(data, json_file)

    outputs = []

    url = 'http://127.0.0.1:11434/api/generate'

    for input in inputs:
        op = []
        for _ in input:
            f = open('input.json', 'r')
            data = json.load(f)
            data['prompt'] = _
            print(data)
            response = requests.post(url, json=data)
            assert response.status_code == 200
            dictionary = json.loads(response.text)
            op.append(dictionary['response'])
            f.close()
        print('row ok')
        outputs.append(op)

    # writing outputs to one file
    filename = 'op_'+llm+'.txt'
    with open(filename, 'w') as f:
        for line in outputs:
            f.write(f"{line}\n")

    return outputs


def llm_blender_nonConv(llm_list, inputs, llm_outputs):
    print('######################### Non-Conversational LLM Blender #########################')

    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")
    blender.loadfuser("llm-blender/gen_fuser_3b")

    fused_answers = []
    for i in range(len(inputs)):
        ans_set = []
        for j in range(len(inputs[i])):
            candidates_texts = []
            for k in range(len(llm_list)):
                candidates_texts.append(llm_outputs[k][i][j])
            ranks = blender.rank(
                [inputs[i][j]], [candidates_texts], return_scores=False, batch_size=1)
            topk_candidates = get_topk_candidates_from_ranks(
                ranks, [candidates_texts], top_k=2)
            fuse_generations = blender.fuse(
                [inputs[i][j]], [topk_candidates], batch_size=2)
            ans_set.append(fuse_generations)
        fused_answers.append(ans_set)

    filename = 'op_fused_nonConv.txt'
    with open(filename, 'w') as f:
        for line in fused_answers:
            f.write(f"{line}\n")
    return fused_answers


def llm_blender_Conv(llm_list, inputs):
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")
    blender.loadfuser("llm-blender/gen_fuser_3b")
    fused_answers = []
    url = 'http://127.0.0.1:11434/api/generate'
    for i in range(len(inputs)):
        ans_set = []
        context = ""
        for j in range(len(inputs[i])):
            question = inputs[i][j] + context
            candidates_texts = []
            for llm in llm_list:
                f = open('input.json', 'r')
                data = json.load(f)
                data['model'] = llm
                data['prompt'] = question
                response = requests.post(url, json=data)
                assert response.status_code == 200
                dictionary = json.loads(response.text)
                candidates_texts.append(dictionary['response'])
                f.close()
            ranks = blender.rank(
                [inputs[i][j]], [candidates_texts], return_scores=False, batch_size=1)
            topk_candidates = get_topk_candidates_from_ranks(
                ranks, [candidates_texts], top_k=2)
            fuse_generations = blender.fuse(
                [inputs[i][j]], topk_candidates, batch_size=2)
            context = context + fuse_generations[0]
            ans_set.append(fuse_generations[0])
        fused_answers.append(ans_set)

    filename = 'op_fused_Conv.txt'
    with open(filename, 'w') as f:
        for line in fused_answers:
            f.write(f"{line}\n")
    return fused_answers


if __name__ == '__main__':
    print("Hi")
    str = "{\"model\":\"\",\"prompt\":\"\",\"stream\":false}"
    with open('input.json', 'w') as f:
        f.write(str)

    llm_list = install_llms()

    inputs, ref = dataset_init()

    lone_outputs = []
    # for llm in llm_list:
    #     lone_outputs.append(lone_llm_output(llm, inputs))

    llm_blender_nonConv(llm_list, inputs, lone_outputs)

    llm_blender_Conv(llm_list, inputs)

    print("Bye")
