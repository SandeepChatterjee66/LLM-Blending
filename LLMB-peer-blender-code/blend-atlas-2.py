import time
import random
import json
import concurrent.futures
from collections import defaultdict
from datasets import load_dataset
import evaluate
import requests
from typing import List, Tuple, Dict


API_URL = "http://127.0.0.1:11434/api/generate"
LLM_LIST = ['mistral', 'llama3.1', 'gemma:2b', 'phi3', 'qwen:4b', 
                'phi', 'tinydolphin', 'deepseek-llm', 'stablelm2', 'dog/arcee-lite']

# Paths
TRAIN_JSON_PATH = "sandeep/llm-blender/data/atlas/atlas_train.json"
OUT_FILE = "blended_outputs.jsonl"
ANSWER_COL = "answer"  # column in your JSON that contains ground-truth answers
QUESTION_COL = "question"  # column containing questions


print(f"Loading full dataset from {TRAIN_JSON_PATH}...")
dataset = load_dataset(
    path='json',
    data_files={'train': TRAIN_JSON_PATH}
)['train']

print(f"Loaded {len(dataset)} examples. Columns: {dataset.column_names}")


def generate_llm_response(model_name: str, prompt: str) -> Tuple[str, float]:
    """Generate response and return (text, time_taken)."""
    start_time = time.time()
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()
    result = resp.json().get('response', '')
    return result, time.time() - start_time


RANKING_PROMPT_TEMPLATE = """Rank responses for this question:\n\nQuestion: {question}\n\nResponses:\n{responses}\n\nRank them as list of indices like [1,2,3,...]:"""
FUSION_PROMPT_TEMPLATE = """Synthesize a single best answer:\n\nQuestion: {question}\n\nResponses:\n1: {r1}\n2: {r2}\n3: {r3}\n\nAnswer:"""


def parallel_generation(question: str) -> Dict[str, str]:
    responses = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(LLM_LIST)) as executor:
        futures = {executor.submit(generate_llm_response, llm, question): llm for llm in LLM_LIST}
        for future in concurrent.futures.as_completed(futures):
            llm = futures[future]
            try:
                text, _ = future.result()
                responses[llm] = text
            except Exception as e:
                responses[llm] = f"Error: {e}"
    return responses

def self_ranking(question: str, candidates: Dict[str, str]) -> List[str]:
    response_list = list(candidates.items())
    responses_str = "\n".join([f"{i+1}: {resp}" for i, (_, resp) in enumerate(response_list)])
    prompt = RANKING_PROMPT_TEMPLATE.format(question=question, responses=responses_str)
    ranking, _ = generate_llm_response(LLM_LIST[0], prompt)  # Use one LLM to do ranking
    # Parse indices
    indices = [int(x) - 1 for x in ranking.replace('[','').replace(']','').split(',') if x.strip().isdigit()]
    top_names = [response_list[i][0] for i in indices if i < len(response_list)]
    return top_names

def final_fusion(question: str, candidates: Dict[str, str], top_3: List[str]) -> str:
    top_resps = [candidates[n] for n in top_3]
    prompt = FUSION_PROMPT_TEMPLATE.format(question=question, r1=top_resps[0], r2=top_resps[1], r3=top_resps[2])
    final_text, _ = generate_llm_response(LLM_LIST[0], prompt)
    return final_text


bertscorer = evaluate.load('bertscore')


results = []
start_time = time.time()

for example in dataset:
    q = example[QUESTION_COL]
    gt_answer = example.get(ANSWER_COL, "")
    print(f"Processing: {q}")

    # Parallel generation
    candidates = parallel_generation(q)
    # Ranking & Fusion
    top_names = self_ranking(q, candidates)
    final_answer = final_fusion(q, candidates, top_names[:3])

    results.append({"question": q, "ground_truth": gt_answer, "final_answer": final_answer})

# Save all to JSONL
with open(OUT_FILE, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

# Compute BERTScore
print("Computing BERTScore...")
preds = [r['final_answer'] for r in results]
refs = [r['ground_truth'] for r in results]
bert_scores = bertscorer.compute(predictions=preds, references=refs, model_type='microsoft/deberta-xlarge-mnli')
avg_f1 = sum(bert_scores['f1']) / len(bert_scores['f1'])

end_time = time.time()
print(f"Average BERTScore F1: {avg_f1:.4f}")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
