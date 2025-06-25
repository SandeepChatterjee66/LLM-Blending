from datasets import load_dataset
import random



import numpy as np
import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
import requests
import json
import time
import os
import random
import concurrent.futures
import ast
import subprocess
from datasets import load_dataset
import evaluate
from typing import List, Dict, Tuple, Any
import math

# Configuration
LLM_list = ['mistral', 'llama3', 'gemma:2b', 'phi3', 'qwen:4b', 'deepseek-llm', 'stablelm2']
LLM_LIST = ['mistral', 'llama3.1', 'gemma:2b', 'phi3', 'qwen:4b', 
                'phi', 'tinydolphin', 'deepseek-llm', 'stablelm2', 'dog/arcee-lite']
    
API_URL = 'http://127.0.0.1:11434/api/generate'

# Define prompt templates (replace with actual file paths or content)
RANKING_PROMPT_TEMPLATE = """
You are an expert evaluator of LLM responses. Given a question and several candidate responses, please rank the responses based on the following criteria:
1.  Relevance: How well does the response address the question?
2.  Coherence: Is the response logically structured and easy to understand?
3.  Factual Accuracy: Is the information provided correct?
4.  Completeness: Does the response cover the key aspects of the question?

Rank the responses from 1 to {n}, where 1 is the best and {n} is the worst. Provide only the ranked list of response identifiers (e.g., "[1, 2, 3, ...]").

Question: {question}

Candidate Responses:
{responses_str}

Ranking:
"""

FUSION_PROMPT_TEMPLATE = """
You are an expert synthesizer of information. Given the original question and three top-ranked candidate responses, please summarize, synthesize, and refine them into a single, cohesive, and comprehensive final answer. Ensure the final answer is accurate, coherent, and addresses the question thoroughly.

Question: {question}

Top 3 Candidate Responses:
1. {response1}
2. {response2}
3. {response3}

Final Answer:
"""


# conv questions
def dataset_init(filename='input_questions.txt'):
    """
    Initialize dataset by sampling from ConvQuestions and saving to file.
    Returns the sampled questions.
    """
    try:
        # Load the ConvQuestions dataset
        print("Loading ConvQuestions dataset...")
        dataset = load_dataset("conv_questions")

        split_name = 'train'  # Available splits: train, validation, test
        data_split = dataset[split_name]

        # Sample 5 random questions
        num_samples = min(5, len(data_split))
        sample_indices = random.sample(range(len(data_split)), num_samples)

        sampled_conversations = [data_split[i] for i in sample_indices]

        # Extract the question text
        inputs = [conv['question'] for conv in sampled_conversations]

        # Save the questions to a file
        with open(filename, 'w', encoding='utf-8') as f:
            for q in inputs:
                f.write(q + '\n')

        print(f"Saved {len(inputs)} questions to {filename}")
        return inputs

    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return None




def generate_llm_response(llm_name: str, prompt: str, context: List = None) -> Tuple[str, List, float]:
    """
    Generate response from a specific LLM via Ollama API.
    Returns (response_text, new_context, generation_time).
    """
    if context is None:
        context = []
        
    start_time = time.time()
    
    try:
        payload = {
            "model": llm_name,
            "prompt": prompt,
            "stream": False,
            "context": context
        }
        
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        generation_time = time.time() - start_time
        
        return result.get('response', ''), result.get('context', []), generation_time
            
    except Exception as e:
        print(f"Error generating response from {llm_name}: {str(e)}")
        generation_time = time.time() - start_time
        return f"Error: {str(e)}", context, generation_time

def normalize_llm_name(llm_name: str) -> str:
    """Normalize LLM name for file naming (replace special characters)."""
    return llm_name.replace(':', '-').replace('/', '-')

def parallel_candidate_generation(question: str, llm_list: List[str]) -> Dict[str, str]:
    """
    Generates responses from multiple LLMs in parallel for a given question.
    Returns a dictionary mapping LLM names to their generated responses.
    """
    responses = {}
    
    def generate_single_llm(llm_name):
        response, _, _ = generate_llm_response(llm_name, question)
        return llm_name, response

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(llm_list)) as executor:
        futures = {executor.submit(generate_single_llm, llm): llm for llm in llm_list}
        
        for future in concurrent.futures.as_completed(futures):
            llm = futures[future]
            try:
                llm_name, response = future.result()
                responses[llm_name] = response
            except Exception as e:
                print(f"Error generating response from {llm}: {str(e)}")
                responses[llm] = f"Error: {str(e)}"
                
    return responses

def self_ranking_by_all_llms(question: str, candidates: Dict[str, str], llm_list: List[str]) -> Dict[str, List[str]]:
    """
    Asks each LLM to rank all candidate responses.
    Returns a dictionary mapping LLM names to their ranked lists of candidate identifiers.
    """
    all_ranks = {}
    
    candidate_items = list(candidates.items())
    n = len(candidate_items)
    
    responses_str = "\n".join([f"{i+1}: {response}" for i, (llm_name, response) in enumerate(candidate_items)])
    
    ranking_prompt = RANKING_PROMPT_TEMPLATE.format(n=n, question=question, responses_str=responses_str)

    def rank_single_llm(llm_name):
        response, _, _ = generate_llm_response(llm_name, ranking_prompt)
        
        try:
            # Parse the ranking response (e.g., "1, 2, 3, ...")
            ranked_indices = [int(x.strip()) - 1 for x in response.split(',') if x.strip().isdigit()]
            ranked_candidate_names = [candidate_items[i][0] for i in ranked_indices if 0 <= i < n]
            
            # Ensure all candidates are included, even if not ranked explicitly
            all_candidate_names = [item[0] for item in candidate_items]
            ranked_set = set(ranked_candidate_names)
            missing_candidates = [name for name in all_candidate_names if name not in ranked_set]
            
            return llm_name, ranked_candidate_names + missing_candidates
            
        except Exception as e:
            print(f"Error parsing ranking response from {llm_name}: {str(e)}")
            # Fallback: return original order if parsing fails
            return llm_name, [item[0] for item in candidate_items]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(llm_list)) as executor:
        futures = {executor.submit(rank_single_llm, llm): llm for llm in llm_list}
        
        for future in concurrent.futures.as_completed(futures):
            llm = futures[future]
            try:
                llm_name, ranked_list = future.result()
                all_ranks[llm_name] = ranked_list
            except Exception as e:
                print(f"Error ranking with {llm}: {str(e)}")
                all_ranks[llm] = [item[0] for item in candidate_items] # Fallback
                
    return all_ranks

def rank_aggregation_borda_count(all_ranks: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Aggregates ranks using Borda Count method.
    Returns a dictionary mapping candidate names to their total Borda score.
    """
    borda_scores = defaultdict(float)
    num_candidates = len(list(all_ranks.values())[0]) if all_ranks else 0
    
    if num_candidates == 0:
        return {}

    for llm_name, ranked_list in all_ranks.items():
        for rank, candidate_name in enumerate(ranked_list, start=1):
            score = num_candidates - rank
            borda_scores[candidate_name] += score
            
    return dict(borda_scores)

def top_3_candidate_selection(borda_scores: Dict[str, float]) -> List[str]:
    """
    Selects the top 3 candidates based on Borda scores.
    Returns a list of the names of the top 3 candidates.
    """
    # Sort candidates by Borda score in descending order
    sorted_candidates = sorted(borda_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Select the top 3 names
    top_3_names = [name for name, score in sorted_candidates[:3]]
    
    return top_3_names

def final_fusion(question: str, candidates: Dict[str, str], top_3_names: List[str], borda_scores: Dict[str, float]) -> str:
    """
    Performs final fusion using the LLM with the highest individual Borda score.
    Returns the final fused answer.
    """
    if not top_3_names:
        return "Error: No candidates selected for fusion."

    # Find the LLM with the highest individual Borda score
    best_llm_name = max(borda_scores, key=borda_scores.get)
    
    # Get the response text for the top 3 candidates
    top_3_responses = [candidates.get(name, f"Response not found for {name}") for name in top_3_names]
    
    # Construct the fusion prompt
    fusion_prompt = FUSION_PROMPT_TEMPLATE.format(
        question=question,
        response1=top_3_responses[0],
        response2=top_3_responses[1],
        response3=top_3_responses[2]
    )
    
    # Generate the fused answer using the best LLM
    fused_answer, _, _ = generate_llm_response(best_llm_name, fusion_prompt)
    
    return fused_answer

import time

def peer_reviewed_blending(question: str) -> Tuple[str, Dict[str, float]]:
    """
    Performs peer-reviewed LLM response blending for a given question with timing.
    Returns (final_answer, timings_dict).
    """
    timings = {}

    print(f"Starting Peer-Reviewed Blending for question: '{question}'")

    # 1. Parallel Candidate Generation
    print("\nStep 1: Parallel Candidate Generation...")
    start = time.time()
    candidates = parallel_candidate_generation(question, LLM_LIST)
    timings['parallel_generation_time'] = time.time() - start
    print(f"→ Generated candidates from {len(candidates)} LLMs in {timings['parallel_generation_time']:.2f}s.")

    # 2. Self-Ranking by All LLMs
    print("\nStep 2: Self-Ranking by All LLMs...")
    start = time.time()
    all_ranks = self_ranking_by_all_llms(question, candidates, LLM_LIST)
    timings['ranking_time'] = time.time() - start
    print(f"→ Received rankings from {len(all_ranks)} LLMs in {timings['ranking_time']:.2f}s.")

    # 3. Rank Aggregation (Borda Count)
    print("\nStep 3: Rank Aggregation (Borda Count)...")
    start = time.time()
    borda_scores = rank_aggregation_borda_count(all_ranks)
    timings['borda_time'] = time.time() - start
    print(f"→ Calculated Borda scores for {len(borda_scores)} candidates in {timings['borda_time']:.2f}s.")

    # 4. Top-3 Candidate Selection
    print("\nStep 4: Top-3 Candidate Selection...")
    start = time.time()
    top_3_names = top_3_candidate_selection(borda_scores)
    timings['selection_time'] = time.time() - start
    print(f"→ Selected top 3 candidates {top_3_names} in {timings['selection_time']:.2f}s.")

    # 5. Final Fusion by Best LLM
    print("\nStep 5: Final Fusion by Best LLM...")
    start = time.time()
    fused_answer = final_fusion(question, candidates, top_3_names, borda_scores)
    timings['fusion_time'] = time.time() - start
    print(f"→ Fusion completed in {timings['fusion_time']:.2f}s.")

    return fused_answer, timings



if __name__ == '__main__':
    total_start = time.time()
    test_question = "What are the main differences between quantum computing and classical computing?"
    final_response, timings = peer_reviewed_blending(test_question)
    total_time = time.time() - total_start

    print("\n--- Final Fused Response ---")
    print(final_response)

    print("\n--- Timings Summary ---")
    for stage, t in timings.items():
        print(f"{stage}: {t:.2f}s")
    print(f"Total end-to-end time")

    test_question = "What are the main differences between quantum computing and classical computing?"
    final_response = peer_reviewed_blending(test_question)
    
    print("\n--- Final Fused Response ---")
    print(final_response)