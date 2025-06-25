import numpy as np
from collections import defaultdict
import requests
import json
import time
import os
import concurrent.futures
import random
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
You are an expert synthesizer of information. Given the question and the top 3 candidate responses, please synthesize a single, accurate and coherent final answer.

Question: {question}

Top 3 Candidate Responses:
1. {response1}
2. {response2}
3. {response3}

Final Answer:
"""


def dataset_init(split='train') -> List[str]:
    print(f"Loading CoQA ({split})...")
    dataset = load_dataset('stanfordnlp/coqa')[split]
    all_questions = []
    for example in dataset:
        all_questions.extend(example['questions'])
    print(f"Loaded {len(all_questions)} questions")
    return all_questions


def generate_llm_response(
    llm_name: str, prompt: str, context: List = None
) -> Tuple[str, List, float]:
    if context is None:
        context = []
    start_time = time.time()
    try:
        payload = {"model": llm_name, "prompt": prompt, "stream": False, "context": context}
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get('response', ''), result.get('context', []), time.time() - start_time
    except Exception as e:
        return f"Error: {e}", context, time.time() - start_time


def parallel_candidate_generation(
    question: str, llm_list: List[str]
) -> Tuple[Dict[str, str], float]:
    responses = {}
    start_time = time.time()

    def generate_single(llm_name):
        resp, _, _ = generate_llm_response(llm_name, question)
        return llm_name, resp

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(llm_list)) as executor:
        futures = {executor.submit(generate_single, llm): llm for llm in llm_list}
        for future in concurrent.futures.as_completed(futures):
            llm = futures[future]
            try:
                name, resp = future.result()
                responses[name] = resp
            except Exception as e:
                responses[llm] = f"Error: {e}"
    return responses, time.time() - start_time


def self_ranking_by_all_llms(
    question: str, candidates: Dict[str, str], llm_list: List[str]
) -> Tuple[Dict[str, List[str]], float]:
    all_ranks = {}
    candidate_items = list(candidates.items())
    n = len(candidate_items)
    responses_str = "\n".join(
        [f"{i+1}: {resp}" for i, (_, resp) in enumerate(candidate_items)]
    )
    ranking_prompt = RANKING_PROMPT_TEMPLATE.format(
        n=n, question=question, responses_str=responses_str
    )

    start_time = time.time()

    def rank_single_llm(llm_name):
        resp, _, _ = generate_llm_response(llm_name, ranking_prompt)
        try:
            ranked_indices = [int(x)-1 for x in resp.split(',') if x.strip().isdigit()]
            ranked_list = [candidate_items[i][0] for i in ranked_indices if 0 <= i < n]
            all_candidate_names = [item[0] for item in candidate_items]
            ranked_set = set(ranked_list)
            return llm_name, ranked_list + [c for c in all_candidate_names if c not in ranked_set]
        except:
            return llm_name, [item[0] for item in candidate_items]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(llm_list)) as executor:
        futures = {executor.submit(rank_single_llm, llm): llm for llm in llm_list}
        for future in concurrent.futures.as_completed(futures):
            llm = futures[future]
            all_ranks[llm] = future.result()[1]

    return all_ranks, time.time() - start_time


def rank_aggregation_borda_count(
    all_ranks: Dict[str, List[str]]
) -> Tuple[Dict[str, float], float]:
    borda_scores = defaultdict(float)
    num_candidates = len(next(iter(all_ranks.values()))) if all_ranks else 0
    start_time = time.time()
    for llm_name, ranked_list in all_ranks.items():
        for rank, candidate_name in enumerate(ranked_list, start=1):
            borda_scores[candidate_name] += num_candidates - rank
    return dict(borda_scores), time.time() - start_time


def top_3_candidate_selection(
    borda_scores: Dict[str, float]
) -> Tuple[List[str], float]:
    start_time = time.time()
    top_3 = [name for name, _ in sorted(
        borda_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:3]]
    return top_3, time.time() - start_time


def final_fusion(
    question: str, candidates: Dict[str, str], top_3_names: List[str]
) -> Tuple[str, float]:
    top_3_responses = [candidates.get(name, "") for name in top_3_names]
    fusion_prompt = FUSION_PROMPT_TEMPLATE.format(
        question=question,
        response1=top_3_responses[0],
        response2=top_3_responses[1],
        response3=top_3_responses[2]
    )
    start_time = time.time()
    fused_answer, _, _ = generate_llm_response(top_3_names[0], fusion_prompt)
    return fused_answer, time.time() - start_time

# ===========================
# End-to-End Blending
# ===========================
def peer_reviewed_blending(question: str) -> Tuple[str, Dict[str, float]]:
    timings = {}
    total_start = time.time()

    # Parallel Generation
    candidates, timings['parallel_generation_time'] = parallel_candidate_generation(question, LLM_LIST)

    # Ranking by all LLMs
    all_ranks, timings['ranking_time'] = self_ranking_by_all_llms(question, candidates, LLM_LIST)

    # Borda Count
    borda_scores, timings['borda_time'] = rank_aggregation_borda_count(all_ranks)

    # Top-3 Selection
    top_3_names, timings['selection_time'] = top_3_candidate_selection(borda_scores)

    # Fusion
    fused_answer, timings['fusion_time'] = final_fusion(question, candidates, top_3_names)

    timings['total_time'] = time.time() - total_start
    return fused_answer, timings


if __name__ == "__main__":
    questions = dataset_init('train')
    output_file = "final_answers.jsonl"

    total_start = time.time()
    with open(output_file, "w", encoding="utf-8") as f:
        for i, question in enumerate(questions, start=1):
            print(f"\n=== Processing question {i}/{len(questions)} ===")
            answer, timings = peer_reviewed_blending(question)
            record = {
                "question": question,
                "answer": answer,
                "timings": timings
            }
            f.write(json.dumps(record) + "\n")

            # Print per-question timings summary
            print("\n--- Per-Question Timing ---")
            for stage, t in timings.items():
                print(f"{stage}: {t:.2f}s")

    total_time = time.time() - total_start
    print(f"\n Completed all questions. Total elapsed time: {total_time:.2f}s")
    print(f"Answers saved to {output_file}.")
