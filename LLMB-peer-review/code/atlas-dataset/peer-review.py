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

import requests
import json
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Any

# Configuration
LLM_LIST = ['mistral', 'llama3', 'gemma:2b', 'phi3', 'qwen:4b', 'deepseek-llm', 'stablelm2']
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

def dataset_init(self, filename='input_questions.txt'):
        """
        Initialize dataset by sampling from atlas and saving to file.
        Returns the sampled conversation data.
        """
        try:
            # Load the dataset
            print("Loading atlas dataset...")
            dataset = load_dataset("atlas")
            
            # Sample 5 random conversations from test split
            test_data = dataset['test']
            sample_indices = random.sample(range(len(test_data)), min(5, len(test_data)))
            sampled_conversations = [test_data[i] for i in sample_indices]
            
            # Extract questions from each conversation
            inputs = []
            for conv in sampled_conversations:
                if 'questions' in conv:
                    inputs.append(conv['questions'])
                else:
                    # Fallback if structure is different
                    inputs.append([str(conv)])
            
            # Save to file
            with open(filename, 'w') as f:
                f.write(str(inputs))
            
            print(f"Saved {len(inputs)} conversations to {filename}")
            return inputs
            
        except Exception as e:
            print(f"Error initializing dataset: {str(e)}")
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

import requests
import json
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Any

# Configuration
LLM_LIST = ['mistral', 'llama3', 'gemma:2b', 'phi3', 'qwen:4b', 'deepseek-llm', 'stablelm2']
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

def dataset_init(self, filename='input_coqa.txt'):
        """
        Initialize dataset by sampling from CoQA dataset and saving to file.
        Returns the sampled conversation data.
        """
        try:
            # Load the dataset
            print("Loading CoQA dataset...")
            dataset = load_dataset("coqa")

            # Sample 5 random conversations from validation split
            test_data = dataset['validation']
            sample_indices = random.sample(range(len(test_data)), min(5, len(test_data)))
            sampled_conversations = [test_data[i] for i in sample_indices]

            # Extract questions, passages, and explanations from each conversation
            inputs = []
            for conv in sampled_conversations:
                passage = conv['story']  # Use 'story' field for passage
                conversation_data = []
                
                # CoQA has questions and answers arrays
                questions = conv['questions']
                answers = conv['answers']
                
                for i in range(len(questions)):
                    question_text = questions[i]['input_text']
                    answer_text = answers[i]['input_text']
                    
                    # Check if answerable (CoQA uses 'unknown' for non-answerable)
                    is_answerable = answer_text.lower() not in ['unknown', 'unanswerable', 'not answerable']
                    
                    conversation_data.append({
                        'passage': passage,
                        'question': question_text,
                        'answer': answer_text,
                        'answerable': is_answerable,
                        'explanation': answer_text  # In CoQA, the answer itself can serve as explanation
                    })
                inputs.append(conversation_data)

            # Save to file
            with open(filename, 'w') as f:
                f.write(str(inputs))

            print(f"Saved {len(inputs)} conversations to {filename}")
            return inputs

        except Exception as e:
            print(f"Error initializing dataset: {str(e)}")
            return None

def load_inputs_from_file(self, filename='input_coqa.txt'):
        """Load previously saved conversation data from file."""
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return None

        try:
            with open(filename, 'r') as f:
                content = f.read()
                inputs = ast.literal_eval(content)
            print(f"Loaded {len(inputs)} conversations from {filename}")
            return inputs
        except Exception as e:
            print(f"Error loading from {filename}: {str(e)}")
            return None

def generate_llm_response(self, llm_name: str, prompt: str, context: List = None) -> Tuple[str, List, float]:
        """
        Generate response from a specific LLM via Ollama API.
        Returns (response_text, new_context, generation_time).
        """
        if context is None:
            context = []

        start_time = time.time()

        try:
            # Prepare request payload
            with open('input.json', 'r') as f:
                payload = json.load(f)

            payload.update({
                'model': llm_name,
                'prompt': prompt,
                'stream': False,
                'context': context
            })

            # Make API request
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            generation_time = time.time() - start_time

            return result.get('response', ''), result.get('context', []), generation_time

        except Exception as e:
            print(f"Error generating response from {llm_name}: {str(e)}")
            generation_time = time.time() - start_time
            return f"Error: {str(e)}", context, generation_time

def normalize_llm_name(self, llm_name: str) -> str:
        """Normalize LLM name for file naming (replace special characters)."""
        return llm_name.replace(':', '-').replace('/', '-')

def lone_llm_output_parallel(self, llm: str, inputs: List[List[Dict]]) -> Tuple[List[List[str]], float]:
        """
        Generate outputs for a single LLM across all conversations.
        Uses parallel processing for different conversations.
        """
        normalized_name = self.normalize_llm_name(llm)
        output_file = f"{self.testing_dir}/op_{normalized_name}.txt"

        # Check if outputs already exist
        if os.path.exists(output_file):
            print(f"Loading existing outputs for {llm}...")
            return self.load_candidates_from_file(normalized_name), 0.0

        print(f"Generating responses for {llm}...")
        start_time = time.time()

        def process_conversation(conv_idx_and_turns):
            conv_idx, turns = conv_idx_and_turns
            responses = []
            context = []  # Each conversation starts with empty context

            for turn in turns:
                passage = turn['passage']
                question = turn['question']
                prompt = f"Passage:\n{passage}\n\nQuestion: {question}\n\nAnswer:"
                response, new_context, _ = self.generate_llm_response(llm, prompt, context)
                responses.append(response)
                context = new_context # Update context for next turn

            return conv_idx, responses

        # Process conversations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            indexed_inputs = [(i, conv) for i, conv in enumerate(inputs)]
            future_to_conv = {executor.submit(process_conversation, item): item[0]
                             for item in indexed_inputs}

            # Collect results in order
            all_responses = [None] * len(inputs)
            for future in concurrent.futures.as_completed(future_to_conv):
                conv_idx = future_to_conv[future]
                try:
                    idx, responses = future.result()
                    all_responses[idx] = responses
                except Exception as e:
                    print(f"Error processing conversation {conv_idx} for {llm}: {str(e)}")
                    all_responses[conv_idx] = ["Error"] * len(inputs[conv_idx])

        # Save results
        with open(output_file, 'w') as f:
            f.write(str(all_responses))

        total_time = time.time() - start_time
        print(f"Completed {llm} in {total_time:.2f} seconds")
        return all_responses, total_time

def load_candidates_from_file(self, llm_name_normalized: str) -> List[List[str]]:
        """Load pre-generated LLM responses from file."""
        filename = f"{self.testing_dir}/op_{llm_name_normalized}.txt"
        try:
            with open(filename, 'r') as f:
                content = f.read()
                return ast.literal_eval(content)
        except Exception as e:
            print(f"Error loading candidates from {filename}: {str(e)}")
            return []



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

import numpy as np
import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
import requests
import json
import time
import os
import random
import ast
import subprocess
from datasets import load_dataset
import evaluate
from typing import List, Dict, Tuple, Any
import math

# Configuration
LLM_LIST = ['mistral', 'llama3', 'gemma:2b', 'phi3', 'qwen:4b', 'deepseek-llm', 'stablelm2']
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

def dataset_init(self, filename='input_coqa.txt'):
    """
    Initialize dataset by sampling from CoQA dataset and saving to file.
    Returns the sampled conversation data.
    """
    try:
        print("Loading CoQA dataset...")
        dataset = load_dataset("coqa")
        test_data = dataset['validation']
        sample_indices = random.sample(range(len(test_data)), min(5, len(test_data)))
        sampled_conversations = [test_data[i] for i in sample_indices]
        inputs = []
        for conv in sampled_conversations:
            passage = conv['story']
            conversation_data = []
            questions = conv['questions']
            answers = conv['answers']
            for i in range(len(questions)):
                question_text = questions[i]['input_text']
                answer_text = answers[i]['input_text']
                is_answerable = answer_text.lower() not in ['unknown', 'unanswerable', 'not answerable']
                conversation_data.append({
                    'passage': passage,
                    'question': question_text,
                    'answer': answer_text,
                    'answerable': is_answerable,
                    'explanation': answer_text
                })
            inputs.append(conversation_data)
        with open(filename, 'w') as f:
            f.write(str(inputs))
        print(f"Saved {len(inputs)} conversations to {filename}")
        return inputs
    except Exception as e:
        print(f"Error initializing dataset: {str(e)}")
        return None

def load_inputs_from_file(self, filename='input_coqa.txt'):
    """Load previously saved conversation data from file."""
    if not os.path.exists(filename):
        print(f"File {filename} not found")
        return None
    try:
        with open(filename, 'r') as f:
            content = f.read()
            inputs = ast.literal_eval(content)
        print(f"Loaded {len(inputs)} conversations from {filename}")
        return inputs
    except Exception as e:
        print(f"Error loading from {filename}: {str(e)}")
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
        with open('input.json', 'r') as f:
            payload = json.load(f)
        payload.update({
            'model': llm_name,
            'prompt': prompt,
            'stream': False,
            'context': context
        })
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

def lone_llm_output(llm: str, inputs: List[List[Dict]]) -> Tuple[List[List[str]], float]:
    """
    Generate outputs for a single LLM across all conversations (sequential, not parallel).
    """
    normalized_name = normalize_llm_name(llm)
    testing_dir = "testing"
    output_file = f"{testing_dir}/op_{normalized_name}.txt"
    if not os.path.exists(testing_dir):
        os.makedirs(testing_dir)
    if os.path.exists(output_file):
        print(f"Loading existing outputs for {llm}...")
        return load_candidates_from_file(normalized_name), 0.0
    print(f"Generating responses for {llm}...")
    start_time = time.time()
    all_responses = []
    for conv in inputs:
        responses = []
        context = []
        for turn in conv:
            passage = turn['passage']
            question = turn['question']
            prompt = f"Passage:\n{passage}\n\nQuestion: {question}\n\nAnswer:"
            response, new_context, _ = generate_llm_response(llm, prompt, context)
            responses.append(response)
            context = new_context
        all_responses.append(responses)
    with open(output_file, 'w') as f:
        f.write(str(all_responses))
    total_time = time.time() - start_time
    print(f"Completed {llm} in {total_time:.2f} seconds")
    return all_responses, total_time

def load_candidates_from_file(llm_name_normalized: str) -> List[List[str]]:
    """Load pre-generated LLM responses from file."""
    filename = f"testing/op_{llm_name_normalized}.txt"
    try:
        with open(filename, 'r') as f:
            content = f.read()
            return ast.literal_eval(content)
    except Exception as e:
        print(f"Error loading candidates from {filename}: {str(e)}")
        return []

def self_ranking_by_all_llms(question: str, candidates: Dict[str, str], llm_list: List[str]) -> Dict[str, List[str]]:
    """
    Asks each LLM to rank all candidate responses (sequential, not parallel).
    Returns a dictionary mapping LLM names to their ranked lists of candidate identifiers.
    """
    all_ranks = {}
    candidate_items = list(candidates.items())
    n = len(candidate_items)
    responses_str = "\n".join([f"{i+1}: {response}" for i, (llm_name, response) in enumerate(candidate_items)])
    ranking_prompt = RANKING_PROMPT_TEMPLATE.format(n=n, question=question, responses_str=responses_str)
    for llm_name in llm_list:
        response, _, _ = generate_llm_response(llm_name, ranking_prompt)
        try:
            ranked_indices = [int(x.strip()) - 1 for x in response.split(',') if x.strip().isdigit()]
            ranked_candidate_names = [candidate_items[i][0] for i in ranked_indices if 0 <= i < n]
            all_candidate_names = [item[0] for item in candidate_items]
            ranked_set = set(ranked_candidate_names)
            missing_candidates = [name for name in all_candidate_names if name not in ranked_set]
            all_ranks[llm_name] = ranked_candidate_names + missing_candidates
        except Exception as e:
            print(f"Error parsing ranking response from {llm_name}: {str(e)}")
            all_ranks[llm_name] = [item[0] for item in candidate_items]
    return all_ranks

def rank_aggregation_borda_count(all_ranks: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Aggregates ranks using Borda Count method.
    Returns a dictionary mapping candidate names to their total Borda score.
    """
    from collections import defaultdict
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
    sorted_candidates = sorted(borda_scores.items(), key=lambda item: item[1], reverse=True)
    top_3_names = [name for name, score in sorted_candidates[:3]]
    return top_3_names

def final_fusion(question: str, candidates: Dict[str, str], top_3_names: List[str], borda_scores: Dict[str, float]) -> str:
    """
    Performs final fusion using the LLM with the highest individual Borda score.
    Returns the final fused answer.
    """
    if not top_3_names:
        return "Error: No candidates selected for fusion."
    best_llm_name = max(borda_scores, key=borda_scores.get)
    top_3_responses = [candidates.get(name, f"Response not found for {name}") for name in top_3_names]
    fusion_prompt = FUSION_PROMPT_TEMPLATE.format(
        question=question,
        response1=top_3_responses[0],
        response2=top_3_responses[1],
        response3=top_3_responses[2]
    )
    fused_answer, _, _ = generate_llm_response(best_llm_name, fusion_prompt)
    return fused_answer

def candidate_generation(question: str, llm_list: List[str]) -> Dict[str, str]:
    """
    Generate candidate responses from all LLMs (sequential, not parallel).
    Returns a dictionary mapping LLM names to their responses.
    """
    candidates = {}
    for llm in llm_list:
        prompt = f"Question: {question}\n\nAnswer:"
        response, _, _ = generate_llm_response(llm, prompt)
        candidates[llm] = response
    return candidates

def peer_reviewed_blending(question: str) -> str:
    """
    Performs peer-reviewed LLM response blending for a given question.
    """
    print(f"Starting Peer-Reviewed Blending for question: '{question}'")
    # 1. Candidate Generation (sequential)
    print("Step 1: Candidate Generation...")
    candidates = parallel_candidate_generation(question, LLM_LIST)
    print(f"Generated candidates from {len(candidates)} LLMs.")
    # 2. Self-Ranking by All LLMs (sequential)
    print("Step 2: Self-Ranking by All LLMs...")
    all_ranks = self_ranking_by_all_llms(question, candidates, LLM_LIST)
    print(f"Received rankings from {len(all_ranks)} LLMs.")
    # 3. Rank Aggregation (Borda Count)
    print("Step 3: Rank Aggregation (Borda Count)...")
    borda_scores = rank_aggregation_borda_count(all_ranks)
    print(f"Calculated Borda scores for {len(borda_scores)} candidates.")
    # 4. Top-3 Candidate Selection
    print("Step 4: Top-3 Candidate Selection...")
    top_3_names = top_3_candidate_selection(borda_scores)
    print(f"Selected top 3 candidates: {top_3_names}")
    # 5. Final Fusion by Best LLM
    print("Step 5: Final Fusion by Best LLM...")
    fused_answer = final_fusion(question, candidates, top_3_names, borda_scores)
    print("Final fusion completed.")
    return fused_answer

# Example Usage:
if __name__ == '__main__':
    test_question = "What are the main differences between quantum computing and classical computing?"
    final_response = peer_reviewed_blending(test_question)
    print("\n--- Final Fused Response ---")
    print(final_response)