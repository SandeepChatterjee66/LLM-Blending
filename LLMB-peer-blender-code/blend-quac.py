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
You are an expert evaluator of LLM responses. Given a question and several candidate responses, please rank them based on:
1. Relevance
2. Coherence
3. Factual Accuracy
4. Completeness

Rank the responses from 1 to {n}, where 1 is the best and {n} is the worst.
Provide only the ranked list of response indices (e.g. "1,2,3,...").

Question: {question}

Candidate Responses:
{responses_str}

Ranking:
"""

FUSION_PROMPT_TEMPLATE = """
You are an expert synthesizer of information. Given a question and the top 3 responses, synthesize them into one accurate and thorough final answer.

Question: {question}

Top 3 Responses:
1. {response1}
2. {response2}
3. {response3}

Final Answer:
"""


# Dataset init for QuAC

def dataset_init(split='train', filename='input_questions_quac.txt') -> List[str]:
    """
    Load QuAC's questions.
    Save them to a file and return them as a list.
    """
    print(f"Loading QuAC ({split})...")
    dataset = load_dataset('squadshifts/quac')[split]

    questions = [example['question'] for example in dataset]

    with open(filename, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(q + '\n')
    print(f"Saved {len(questions)} questions to {filename}")
    return questions


def generate_llm_response(llm_name: str, prompt: str, context: List = None) -> Tuple[str, List, float]:
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

def parallel_candidate_generation(question: str) -> Dict[str, str]:
    responses = {}
    def generate_single(llm_name):
        resp, _, _ = generate_llm_response(llm_name, question)
        return llm_name, resp
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(LLM_LIST)) as executor:
        futures = {executor.submit(generate_single, llm): llm for llm in LLM_LIST}
        for future in concurrent.futures.as_completed(futures):
            llm = futures[future]
            try:
                llm_name, response = future.result()
                responses[llm_name] = response
            except:
                responses[llm] = "Error"
    return responses

def self_ranking_by_all_llms(question: str, candidates: Dict[str, str]) -> Dict[str, List[str]]:
    all_ranks = {}
    candidate_items = list(candidates.items())
    n = len(candidate_items)
    responses_str = "\n".join([f"{i+1}: {resp}" for i, (_, resp) in enumerate(candidate_items)])
    ranking_prompt = RANKING_PROMPT_TEMPLATE.format(n=n, question=question, responses_str=responses_str)

    def rank_single_llm(llm_name):
        resp, _, _ = generate_llm_response(llm_name, ranking_prompt)
        try:
            ranked_indices = [int(x)-1 for x in resp.split(',') if x.strip().isdigit()]
            ranked_names = [candidate_items[i][0] for i in ranked_indices if 0 <= i < n]
            all_names = [item[0] for item in candidate_items]
            return llm_name, ranked_names + [c for c in all_names if c not in ranked_names]
        except:
            return llm_name, [item[0] for item in candidate_items]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(LLM_LIST)) as executor:
        futures = {executor.submit(rank_single_llm, llm): llm for llm in LLM_LIST}
        for future in concurrent.futures.as_completed(futures):
            llm = futures[future]
            all_ranks[llm] = future.result()[1]

    return all_ranks

def rank_aggregation_borda_count(all_ranks: Dict[str, List[str]]) -> Dict[str, float]:
    borda_scores = defaultdict(float)
    num_candidates = len(next(iter(all_ranks.values()))) if all_ranks else 0
    for llm_name, ranked_list in all_ranks.items():
        for rank, candidate_name in enumerate(ranked_list, start=1):
            borda_scores[candidate_name] += num_candidates - rank
    return dict(borda_scores)

def top_3_candidate_selection(borda_scores: Dict[str, float]) -> List[str]:
    return [name for name, _ in sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)[:3]]

def final_fusion(question: str, candidates: Dict[str, str], top_3_names: List[str]) -> str:
    top_3_resps = [candidates.get(name, "") for name in top_3_names]
    fusion_prompt = FUSION_PROMPT_TEMPLATE.format(
        question=question,
        response1=top_3_resps[0],
        response2=top_3_resps[1],
        response3=top_3_resps[2]
    )
    fused_answer, _, _ = generate_llm_response(top_3_names[0], fusion_prompt)
    return fused_answer

def peer_reviewed_blending(question: str) -> str:
    candidates = parallel_candidate_generation(question)
    all_ranks = self_ranking_by_all_llms(question, candidates)
    borda_scores = rank_aggregation_borda_count(all_ranks)
    top_3_names = top_3_candidate_selection(borda_scores)
    return final_fusion(question, candidates, top_3_names)

import numpy as np
from collections import defaultdict
import requests
import json
import time
import concurrent.futures
from datasets import load_dataset
from typing import List, Dict, Tuple
from bert_score import score as bert_score


LLM_LIST = ['mistral', 'llama3.1', 'gemma:2b', 'phi3',
            'qwen:4b', 'phi', 'tinydolphin', 'deepseek-llm',
            'stablelm2', 'dog/arcee-lite']

API_URL = 'http://127.0.0.1:11434/api/generate'

RANKING_PROMPT_TEMPLATE = """You are an expert evaluator of LLM responses. Given a question and several candidate responses, please rank them ..."""  # shortened for clarity

FUSION_PROMPT_TEMPLATE = """You are an expert synthesizer of information..."""  # shortened for clarity


def compute_bert_scores(candidates: Dict[str, str], ref_name: str) -> Dict[str, float]:
    """
    Compute BERTScore F1 of each candidate vs. a reference response.
    """
    ref = candidates[ref_name]
    responses = list(candidates.values())
    P, R, F1 = bert_score(responses, [ref] * len(responses), model_type='roberta-large')
    return {name: f1.item() for name, f1 in zip(candidates.keys(), F1)}

# ===========================
# LLM interaction
# ===========================
def generate_llm_response(llm_name: str, prompt: str) -> Tuple[str, float]:
    start_time = time.time()
    try:
        payload = {"model": llm_name, "prompt": prompt, "stream": False}
        resp = requests.post(API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        return result.get('response', ''), time.time() - start_time
    except Exception as e:
        return f"Error: {e}", time.time() - start_time

def parallel_candidate_generation(question: str) -> Dict[str, str]:
    responses = {}
    def generate_single(llm_name):
        resp, gen_time = generate_llm_response(llm_name, question)
        return llm_name, resp, gen_time

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(LLM_LIST)) as executor:
        futures = {executor.submit(generate_single, llm): llm for llm in LLM_LIST}
        for future in concurrent.futures.as_completed(futures):
            llm = futures[future]
            try:
                llm_name, response, _ = future.result()
                responses[llm_name] = response
            except:
                responses[llm] = "Error"
    return responses


# Main

if __name__ == "__main__":
    questions = load_dataset('squadshifts/quac')['train']['question']

    for i, question in enumerate(questions[:5], 1):
        print(f"\n=== Processing QuAC question {i}/{len(questions)} ===")
        start_time = time.time()
        candidates = parallel_candidate_generation(question)

        # Compute BERTScores — use top response as ref
        top_cand_name = next(iter(candidates))  # choose first one as ref
        bert_scores = compute_bert_scores(candidates, ref_name=top_cand_name)

        print(f"\nBERT F1 Scores per candidate:")
        for llm_name, score in bert_scores.items():
            print(f"{llm_name}: {score:.4f}")

        # Run the full peer review blending
        answer = peer_reviewed_blending(question)

        total_time = time.time() - start_time
        print(f"Completed question {i} in {total_time:.2f}s")
        print(f"Final Answer:\n{answer}")

