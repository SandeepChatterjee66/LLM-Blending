import numpy as np
import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
import concurrent.futures
import datetime
import os
import json
from datasets import load_dataset
import requests
from bleurt import score
import ast

def file_to_matrix(filename):
    liStr = []
    with open(filename, "r", encoding="cp1252", errors="ignore") as file:
        while True:
            line = file.readline()
            if not line:
                break
        # Process the line
        liStr.append(line)
    li = []
    for _ in liStr:
        li.append(ast.literal_eval(_))
    return li

def blender_full_init():
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")
    blender.loadfuser("llm-blender/gen_fuser_3b")
    return blender

def blender_rankerless_init():
    blender = llm_blender.Blender()
    blender.loadfuser("llm-blender/gen_fuser_3b")
    return blender

def dataset_init_conv_questions():
    # to return a smaller number of question sets
    # must be in the format 'dataset_name[i]['questions']'
    dataset = load_dataset("conv_questions")
    train_data = dataset['train'].select(range(500))
    # test_data = dataset['train'].select(range(2))
    # validation_data = dataset['train'].select(range(2))

    return train_data

def dataset_init_atlas_converse():
    dataset_atlas = []
    with open('atlas-converse/combined-convo.json', 'r') as f:
        li = json.load(f)
    f.close()
    dataset_atlas.extend(li[:167])
    with open('atlas-converse/combined-convo_2.json', 'r') as f:
        li = json.load(f)
    f.close()
    dataset_atlas.extend(li[:167])
    with open('atlas-converse/combined-convo_3.json', 'r') as f:
        li = json.load(f)
    f.close()
    dataset_atlas.extend(li[:167])

    return dataset_atlas

def generate(llm, prompt, context=[])->list:
    url = 'http://127.0.0.1:11434/api/generate'
    f = open('input.json', 'r')
    data = json.load(f)
    data['model'] = llm
    data['prompt'] = prompt
    data['context'] = context
    response = requests.post(url, json=data)
    print("response is",response)
    print("status code is ", response.status_code)
    #assert response.status_code == 200
    dictionary = json.loads(response.text)
    f.close()
    return [dictionary['response'], dictionary['context']]

def generate_context(llm, prompt)->list:
    prompt = "Output the text: " + prompt
    return generate(llm, prompt)[1]



def experiment_1(blender, llmList, dataset) -> list:
    print('----------------------------\nExperiment 1 Begins:')
    print(datetime.datetime.now())
    
    fused_answers = []
    fa = []
    rank_filename = 'Experiment_Results/rank.txt'
    fused_filename = 'Experiment_Results/exp_1.txt'
    
    with open(rank_filename, 'a') as rank_file, open(fused_filename, 'a') as fused_file:
        for i, data in enumerate(dataset):
            ans_set = []
            contexts = {llm: [] for llm in llmList}
            
            for j, question in enumerate(data['questions']):
                if j == 0:
                    question += data['seed_entity_text']

                # Parallel processing for candidates_texts generation
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(generate, llm, question, contexts[llm]): llm for llm in llmList}
                    candidates_texts = [future.result()[0] for future in concurrent.futures.as_completed(futures)]
                
                # Rank the candidates
                ranks = blender.rank([question], [candidates_texts], return_scores=False, batch_size=1)
                rank_file.write(f"Ranks for question {j + 1} in dataset {i + 1}: {ranks}\n")
                print('ranks are ', ranks)
                
                # Get top k candidates and fuse them
                topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=2)
                fuse_generations = blender.fuse([question], topk_candidates, batch_size=2)
                ans_set.append(fuse_generations[0])
                
                # Update contexts for each LLM
                contexts = {llm: generate_context(llm, fuse_generations[0]) for llm in llmList}
            
            fa.append(ans_set)
            
            if i % 50 == 0:
                fused_file.write('\n'.join(str(line) for line in fa) + '\n')
                fused_answers.extend(fa)
                fa = []
        
        if fa:
            fused_file.write('\n'.join(str(line) for line in fa) + '\n')
    
    print('Experiment 1 Ends:')
    print(datetime.datetime.now())

    return fused_answers


# running LLM-Blender on Atlas-Converse dataset
def experiment_2(blender, llmList, dataset)->list:
    print('----------------------------\nExperiment 2 Begins:')
    print(datetime.datetime.now())
    fused_answers = []
    fa = []
    filename = 'Experiment_Results\exp_2.txt'
    for i in range(len(dataset)):
        ans_set = []
        contexts = {}
        for llm in llmList: contexts[llm] = []
        for j in range(len(dataset[i]['conversations'])):
            if j+1>=len(dataset[i]['conversations']) or dataset[i]['conversations'][j]['from'] == 'AI':
                continue
            question = dataset[i]['conversations'][j]['value']
            candidates_texts = []
            for llm in llmList:
                candidates_texts.append(generate(llm, question, contexts[llm])[0])
            ranks = blender.rank(
                [dataset[i]['conversations'][j]['value']], [candidates_texts], return_scores=False, batch_size=1)
            topk_candidates = get_topk_candidates_from_ranks(
                ranks, [candidates_texts], top_k=2)
            fuse_generations = blender.fuse(
                [dataset[i]['conversations'][j]['value']], topk_candidates, batch_size=2)
            ans_set.append(fuse_generations[0])
            for llm in llmList:
                contexts[llm] = generate_context(llm, fuse_generations[0])
        fa.append(ans_set)
        if i%50==0:
            with open(filename, 'a') as f:
                for line in fa:
                    f.write(f"{line}\n")
            f.close()
            fused_answers.extend(fa)
            fa = []

    with open(filename, 'a') as f:
        for line in fa:
            f.write(f"{line}\n")
    f.close()
    print('Experiment 2 Ends:')
    print(datetime.datetime.now())

    return fused_answers


def experiment_3(blender, llmList, dataset)->list:
    print('----------------------------\nExperiment 3 Begins:')
    print(datetime.datetime.now())
    filename = 'Experiment_Results\exp_3.txt'
    fused_answers = []
    fa = []
    for i in range(len(dataset)):
        print(datetime.datetime.now())
        ans_set = []
        contexts = {}
        for llm in llmList: contexts[llm] = []
        for j in range(len(dataset[i]['questions'])):
            question = dataset[i]['questions'][j]
            if j==0:
                question += dataset[i]['seed_entity_text']
            candidates_texts = []
            for llm in llmList:
                candidates_texts.append(generate(llm, question, contexts[llm])[0])
            fuse_generations = blender.fuse(
                [dataset[i]['questions'][j]], [candidates_texts], batch_size=2)
            ans_set.append(fuse_generations[0])
            for llm in llmList:
                contexts[llm] = generate_context(llm, fuse_generations[0])
        fa.append(ans_set)
        if i%50==0:
            with open(filename, 'a') as f:
                for line in fa:
                    f.write(f"{line}\n")
            f.close()
            fused_answers.extend(fa)
            fa = []

    with open(filename, 'a') as f:
        for line in fa:
            f.write(f"{line}\n")
    f.close()
    print('Experiment 3 Ends:')
    print(datetime.datetime.now())

    return fused_answers

def experiment_4(blender, llmList, dataset)->list:
    print('----------------------------\nExperiment 4 Begins:')
    print(datetime.datetime.now())
    filename = 'Experiment_Results\exp_4.txt'
    fused_answers = []
    fa = []
    for i in range(len(dataset)):
        ans_set = []
        contexts = {}
        for llm in llmList: contexts[llm] = []
        for j in range(len(dataset[i]['conversations'])):
            if j+1>=len(dataset[i]['conversations']) or dataset[i]['conversations'][j]['from'] == 'AI':
                continue
            question = dataset[i]['conversations'][j]['value']
            candidates_texts = []
            for llm in llmList:
                candidates_texts.append(generate(llm, question, contexts[llm])[0])
            fuse_generations = blender.fuse(
                [dataset[i]['conversations'][j]['value']], [candidates_texts], batch_size=2)
            ans_set.append(fuse_generations[0])
            for llm in llmList:
                contexts[llm] = generate_context(llm, fuse_generations[0])
        fa.append(ans_set)
        if i%50==0:
            with open(filename, 'a') as f:
                for line in fa:
                    f.write(f"{line}\n")
            f.close()
            fused_answers.extend(fa)
            fa = []

    
    with open(filename, 'a') as f:
        for line in fa:
            f.write(f"{line}\n")
    f.close()
    print('Experiment 4 Ends:')
    print(datetime.datetime.now())

    return fused_answers

def bleurt_scorer_conv_questions(scorer, ans, dataset):
    scores = []
    for i in range(len(ans)):
        s = scorer.score(references=dataset[i]['questions'], candidates=ans[i])
        assert isinstance(s, list)
        scores.append(np.average(s))
    return scores

def bleurt_scorer_atlas_converse(scorer, ans, dataset):
    scores = []
    for i in range(len(ans)):
        reference = []
        for j in range(len(dataset[i]['conversations'])):
            if j+1>=len(dataset[i]['conversations']) or dataset[i]['conversations'][j]['from'] == 'AI':
                continue
            reference.append(dataset[i]['conversations'][j+1]['value'])
        # print(reference)
        s = scorer.score(references = reference, candidates = ans[i])
        assert isinstance(s, list)
        scores.append(np.average(s))
    return scores


if __name__ == "__main__":
    print('---------------phase 1 experiments---------------')

    str = "{\"model\":\"\",\"prompt\":\"\",\"context\":[],\"stream\":false}"
    with open('input.json', 'w') as f:
        f.write(str)
    f.close()

    dataset_conv_questions = dataset_init_conv_questions()
    dataset_atlas = dataset_init_atlas_converse()
    llmList = ['mistral', 'llama3', 'gemma', 'phi3', 'deepseek-llm']
    # llmList = ['mistral', 'llama3']

    blender = blender_full_init()
    results_1 = experiment_1(blender, llmList, dataset_conv_questions)
    results_2 = experiment_2(blender, llmList, dataset_atlas)
    results_3 = experiment_3(blender, llmList, dataset_conv_questions)
    results_4 = experiment_4(blender, llmList, dataset_atlas)


    scorer = score.BleurtScorer("C:/Users/neerz/BLEURT-20/")
    scores_1 = bleurt_scorer_conv_questions(scorer, results_1, dataset_conv_questions)
    scores_2 = bleurt_scorer_atlas_converse(scorer, results_2, dataset_atlas)
    scores_3 = bleurt_scorer_conv_questions(scorer, results_3, dataset_conv_questions)
    scores_4 = bleurt_scorer_atlas_converse(scorer, results_4, dataset_atlas)


    # writing scores
    filename = 'Experiment_Results\scores_1.txt'
    with open(filename, 'w') as f:
        for line in scores_1:
            f.write(f"{line}\n")
    f.close()
    filename = 'Experiment_Results\scores_2.txt'
    with open(filename, 'w') as f:
        for line in scores_2:
            f.write(f"{line}\n")
    f.close()
    filename = 'Experiment_Results\scores_3.txt'
    with open(filename, 'w') as f:
        for line in scores_3:
            f.write(f"{line}\n")
    f.close()
    filename = 'Experiment_Results\scores_4.txt'
    with open(filename, 'w') as f:
        for line in scores_4:
            f.write(f"{line}\n")
    f.close()
    print('end')
