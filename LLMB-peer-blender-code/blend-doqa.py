import numpy as np
from collections import defaultdict
import requests
import json
import time
import concurrent.futures
from datasets import load_dataset
from typing import List, Dict, Tuple

LLM_LIST = [
    'mistral', 'llama3.1', 'gemma:2b', 'phi3',
    'qwen:4b', 'phi', 'tinydolphin', 'deepseek-llm',
    'stablelm2', 'dog/arcee-lite'
]

API_URL = 'http://127.0.0.1:11434/api/generate'

RANKING_PROMPT_TEMPLATE = """
You are an expert evaluator of LLM responses. Given a question and several candidate responses, please rank the responses based on:
1. Relevance
2. Coherence
3. Factual Accuracy
4. Completeness

Rank the responses from 1 to {n}, where 1 is the best and {n} is the worst.
Provide only the ranked list of response identifiers (e.g. "[1, 2, 3, ...]").

Question: {question}

Candidate Responses:
{responses_str}

Ranking:
"""

FUSION_PROMPT_TEMPLATE = """
You are an expert synthesizer of information. Given the question and the top 3 candidate responses, please synthesize them into one accurate and concise final answer.

Question: {question}

Top 3 Candidate Responses:
1. {response1}
2. {response2}
3. {response3}

Final Answer:
"""

def dataset_init(split='train', filename='input_questions_doqa.txt') -> List[str]:
    st = time.time()
    dataset = load_dataset('googleresearch/doqa')[split]
    questions = [example['question'] for example in dataset]
    with open(filename, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(q + '\n')
    return questions

def generate_llm_response(llm_name: str, prompt: str, context: List = None) -> Tuple[str, List, float]:
    if context is None:
        context = []
    st = time.time()
    try:
        payload = {"model": llm_name, "prompt": prompt, "stream": False, "context": context}
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get('response', ''), result.get('context', []), time.time() - st
    except Exception as e:
        return f"Error: {e}", context, time.time() - st

def parallel_candidate_generation(question: str) -> Tuple[Dict[str, str], float]:
    st = time.time()
    responses = {}
    def generate_single(llm_name):
        resp, _, _ = generate_llm_response(llm_name, question)
        return llm_name, resp
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(LLM_LIST)) as executor:
        futures = {executor.submit(generate_single, llm): llm for llm in LLM_LIST}
        for future in concurrent.futures.as_completed(futures):
            llm = futures[future]
            try:
                name, response = future.result()
                responses[name] = response
            except:
                responses[llm] = "Error"
    return responses, time.time() - st

def self_ranking_by_all_llms(question: str, candidates: Dict[str, str]) -> Tuple[Dict[str, List[str]], float]:
    st = time.time()
    all_ranks = {}
    candidate_items = list(candidates.items())
    n = len(candidate_items)
    responses_str = "\n".join([f"{i+1}: {resp}" for i, (_, resp) in enumerate(candidate_items)])
    ranking_prompt = RANKING_PROMPT_TEMPLATE.format(n=n, question=question, responses_str=responses_str)
    def rank_single_llm(llm_name):
        resp, _, _ = generate_llm_response(llm_name, ranking_prompt)
        try:
            ranked_indices = [int(x)-1 for x in resp.split(',') if x.strip().isdigit()]
            ranked_list = [candidate_items[i][0] for i in ranked_indices if 0 <= i < n]
            all_candidates = [item[0] for item in candidate_items]
            ranked_set = set(ranked_list)
            return llm_name, ranked_list + [c for c in all_candidates if c not in ranked_set]
        except:
            return llm_name, [item[0] for item in candidate_items]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(LLM_LIST)) as executor:
        futures = {executor.submit(rank_single_llm, llm): llm for llm in LLM_LIST}
        for future in concurrent.futures.as_completed(futures):
            llm = futures[future]
            all_ranks[llm] = future.result()[1]
    return all_ranks, time.time() - st

def rank_aggregation_borda_count(all_ranks: Dict[str, List[str]]) -> Tuple[Dict[str, float], float]:
    st = time.time()
    borda_scores = defaultdict(float)
    num_candidates = len(next(iter(all_ranks.values()))) if all_ranks else 0
    for llm_name, ranked_list in all_ranks.items():
        for rank, candidate_name in enumerate(ranked_list, start=1):
            borda_scores[candidate_name] += num_candidates - rank
    return dict(borda_scores), time.time() - st

def top_3_candidate_selection(borda_scores: Dict[str, float]) -> Tuple[List[str], float]:
    st = time.time()
    top_3 = [name for name, _ in sorted(borda_scores.items(), key=lambda item: item[1], reverse=True)[:3]]
    return top_3, time.time() - st

def final_fusion(question: str, candidates: Dict[str, str], top_3_names: List[str]) -> Tuple[str, float]:
    st = time.time()
    top_3_responses = [candidates.get(name, "") for name in top_3_names]
    fusion_prompt = FUSION_PROMPT_TEMPLATE.format(
        question=question,
        response1=top_3_responses[0],
        response2=top_3_responses[1],
        response3=top_3_responses[2]
    )
    fused_answer, _, _ = generate_llm_response(top_3_names[0], fusion_prompt)
    return fused_answer, time.time() - st

def peer_reviewed_blending(question: str) -> Tuple[str, Dict[str, float]]:
    total_st = time.time()
    candidates, t_gen = parallel_candidate_generation(question)
    all_ranks, t_rank = self_ranking_by_all_llms(question, candidates)
    borda_scores, t_borda = rank_aggregation_borda_count(all_ranks)
    top_3_names, t_sel = top_3_candidate_selection(borda_scores)
    fused_answer, t_fuse = final_fusion(question, candidates, top_3_names)
    return fused_answer, {
        "generation_time": t_gen,
        "ranking_time": t_rank,
        "borda_time": t_borda,
        "selection_time": t_sel,
        "fusion_time": t_fuse,
        "total_time": time.time() - total_st
    }

if __name__ == "__main__":
    questions = dataset_init('train', filename='input_questions_doqa.txt')
    output_file = "final_answers_doqa.jsonl"
    total_st = time.time()
    with open(output_file, "w", encoding="utf-8") as f:
        for i, question in enumerate(questions, start=1):
            print(f"\n=== Processing question {i}/{len(questions)} ===")
            answer, timings = peer_reviewed_blending(question)
            record = {"question": question, "answer": answer, "timings": timings}
            f.write(json.dumps(record) + "\n")
    print(f"Total processing time: {time.time() - total_st:.2f}s")
