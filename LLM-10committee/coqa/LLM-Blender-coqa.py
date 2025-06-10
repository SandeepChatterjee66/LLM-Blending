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
from typing import List, Dict, Tuple, Any, Union
import math

class LLMBlenderExperiment:
    """
    A comprehensive evaluation framework for LLM Blender strategies.
    Tests different approaches to combining LLM responses with performance tracking.
    """

    def __init__(self):
        # Configuration
        self.llm_list = ['mistral', 'llama3', 'gemma:2b', 'phi3', 'qwen:4b', 'deepseek-llm', 'stablelm2']
        self.api_url = 'http://127.0.0.1:11434/api/generate'
        self.results_dir = 'Experiment_Results'
        self.testing_dir = 'testing'

        # Setup directories
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.results_dir, self.testing_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

    def install_llms_parallel(self, llm_list: List[str]):
        """
        Install LLMs in parallel using Ollama.
        This saves time when setting up multiple models.
        """
        def install_single_llm(llm_name):
            try:
                print(f"Installing {llm_name}...")
                result = subprocess.run(['ollama', 'pull', llm_name],
                                      capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    print(f"✓ Successfully installed {llm_name}")
                    return True
                else:
                    print(f"✗ Failed to install {llm_name}: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print(f"✗ Timeout installing {llm_name}")
                return False
            except Exception as e:
                print(f"✗ Error installing {llm_name}: {str(e)}")
                return False

        print("Starting parallel LLM installation...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(install_single_llm, llm): llm for llm in llm_list}
            results = {}

            for future in concurrent.futures.as_completed(futures):
                llm = futures[future]
                try:
                    results[llm] = future.result()
                except Exception as e:
                    print(f"Installation failed for {llm}: {str(e)}")
                    results[llm] = False

        successful = sum(results.values())
        print(f"Installation complete: {successful}/{len(llm_list)} models installed successfully")
        return results

    def setup_ollama_input_json(self):
        """Create the input.json file needed for Ollama API requests."""
        input_template = {
            "model": "",
            "prompt": "",
            "stream": False,
            "context": []
        }

        with open('input.json', 'w') as f:
            json.dump(input_template, f, indent=2)
        print("Created input.json template")

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

    def blender_init(self):
        """Initialize LLM Blender with ranker and fuser models."""
        print("Initializing LLM Blender...")
        blender = llm_blender.Blender()

        try:
            blender.loadranker("llm-blender/PairRM")
            print("✓ Loaded PairRM ranker")

            blender.loadfuser("llm-blender/gen_fuser_3b")
            print("✓ Loaded gen_fuser_3b")

            return blender
        except Exception as e:
            print(f"Error initializing blender: {str(e)}")
            return None

    def calculate_bertscore(self, references: List[str], candidates: List[str]) -> float:
        """Calculate average BERTScore F1 between references and candidates."""
        try:
            bertscore = evaluate.load("bertscore")
            results = bertscore.compute(predictions=candidates, references=references, lang="en")
            return np.mean(results['f1'])
        except Exception as e:
            print(f"Error calculating BERTScore: {str(e)}")
            return 0.0

    def print_ranking_analysis(self, ranks: List[int], llm_list: List[str], context: str = ""):
        """Prints the LLM ranks for a given turn."""
        print(f"    Ranks {context}:")
        for i, llm in enumerate(llm_list):
            print(f"      - {llm}: {ranks[i]}")

    def _save_experiment_results(self, strategy_name: str, fused_answers: List[str], timing_logs: List[Dict], ranking_history: List = None):
        """Save experiment results to files."""
        # Save fused answers
        answers_file = f"{self.results_dir}/fused_answers_{strategy_name}.txt"
        with open(answers_file, 'w', encoding='utf-8') as f:
            for answer in fused_answers:
                f.write(f"{answer}\n")

        # Save timing logs
        timing_file = f"{self.results_dir}/timing_{strategy_name}.json"
        with open(timing_file, 'w') as f:
            json.dump(timing_logs, f, indent=2)

        # Save ranking history if provided
        if ranking_history:
            ranking_file = f"{self.results_dir}/ranking_history_{strategy_name}.json"
            with open(ranking_file, 'w') as f:
                json.dump(ranking_history, f, indent=2)
            print(f"  Saved ranking history to {ranking_file}")

        print(f"  Saved results to {answers_file} and {timing_file}")

    def experiment_full_ranking(self, blender, llm_list: List[str], inputs: List[List[Dict]]) -> Tuple[List[str], List[Dict]]:
        """
        Baseline strategy: Full ranking for every turn.
        This is the standard LLM Blender approach, but now with K=1 (only the best LLM response is chosen for fusion).
        """
        print("\n=== Experiment: Full Ranking (Baseline, K=1) ===")
        start_time = time.time()

        fused_answers = []
        timing_logs = []
        ranking_history = []

        for conv_idx, conversation in enumerate(inputs):
            print(f"Processing conversation {conv_idx + 1}/{len(inputs)}")

            # Initialize context for each LLM
            llm_contexts = {llm: [] for llm in llm_list}

            for turn_idx, turn in enumerate(conversation):
                turn_start = time.time()

                passage = turn['passage']
                question = turn['question']
                prompt = f"Passage:\n{passage}\n\nQuestion: {question}\n\nAnswer:"

                # Generate candidates from all LLMs
                candidates_texts = []
                generation_times = {}

                for llm in llm_list:
                    response, new_context, gen_time = self.generate_llm_response(
                        llm, prompt, llm_contexts[llm]
                    )
                    candidates_texts.append(response)
                    llm_contexts[llm] = new_context
                    generation_times[llm] = gen_time

                # Ranking phase
                rank_start = time.time()
                try:
                    ranks = blender.rank([question], [candidates_texts], return_scores=False, batch_size=1)
                    ranking_time = time.time() - rank_start

                    # Store ranking history
                    ranking_history.append({
                        'conversation_idx': conv_idx,
                        'turn_idx': turn_idx,
                        'ranks': ranks[0],
                        'llm_list': llm_list.copy()
                    })

                    # Print ranking analysis
                    self.print_ranking_analysis(
                        ranks[0], llm_list,
                        f"(Conv {conv_idx+1}, Turn {turn_idx+1})"
                    )

                except Exception as e:
                    print(f"Ranking error: {str(e)}")
                    ranking_time = 0
                    ranks = [[list(range(len(candidates_texts)))]]

                # Select only the top-1 candidate (K=1)
                topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=1)

                # Fusion phase (with only the best candidate)
                fusion_start = time.time()
                try:
                    # If only one candidate, fusion is trivial (just use it)
                    fused_answer = topk_candidates[0][0]
                    fusion_time = time.time() - fusion_start
                except Exception as e:
                    print(f"Fusion error: {str(e)}")
                    fused_answer = candidates_texts[0]  # Fallback to first candidate
                    fusion_time = 0

                fused_answers.append(fused_answer)

                # Log timing data
                timing_logs.append({
                    'conversation_idx': conv_idx,
                    'turn_idx': turn_idx,
                    'llm_generation_times': generation_times,
                    'ranking_time': ranking_time,
                    'fusion_time': fusion_time,
                    'total_turn_time': time.time() - turn_start,
                    'ranks': ranks[0] if ranks else None
                })

        total_time = time.time() - start_time
        print(f"Full Ranking (K=1) completed in {total_time:.2f} seconds")

        # Save results with ranking history
        self._save_experiment_results('full_ranking', fused_answers, timing_logs, ranking_history)
        return fused_answers, timing_logs

    def experiment_dynamic_elimination(self, blender, llm_list: List[str], inputs: List[List[Dict]]) -> Tuple[List[str], List[Dict]]:
        """
        Policy 1: Dynamic conversation-specific elimination.
        Eliminates bottom half of LLMs after half the conversation.
        """
        print("\n=== Experiment: Dynamic Elimination ===")
        start_time = time.time()

        fused_answers = []
        timing_logs = []

        for conv_idx, conversation in enumerate(inputs):
            print(f"Processing conversation {conv_idx + 1}/{len(inputs)}")

            # Reset for each conversation
            llm_contexts = {llm: [] for llm in llm_list}
            llm_performance_scores = {llm: 0 for llm in llm_list}
            active_llms = list(llm_list)

            elimination_point = math.ceil(len(conversation) / 2)

            for turn_idx, turn in enumerate(conversation):
                turn_start = time.time()

                # Generate candidates from active LLMs
                candidates_texts = []
                generation_times = {}

                for llm in active_llms:
                    passage = turn['passage']
                    question = turn['question']
                    prompt = f"Passage:\n{passage}\n\nQuestion: {question}\n\nAnswer:"
                    response, new_context, gen_time = self.generate_llm_response(
                        llm, prompt, llm_contexts[llm]
                    )
                    candidates_texts.append(response)
                    llm_contexts[llm] = new_context
                    generation_times[llm] = gen_time

                # Ranking phase
                rank_start = time.time()
                try:
                    ranks = blender.rank([question], [candidates_texts], return_scores=False, batch_size=1)
                    ranking_time = time.time() - rank_start

                    # Update performance scores if we're still in the first half
                    if turn_idx < elimination_point:
                        for i, llm in enumerate(active_llms):
                            llm_performance_scores[llm] += ranks[0][i]

                except Exception as e:
                    print(f"Ranking error: {str(e)}")
                    ranking_time = 0
                    ranks = [[list(range(len(candidates_texts)))]]

                # Dynamic elimination after first half
                if turn_idx == elimination_point and len(active_llms) > 2:
                    # Sort LLMs by performance (lower rank sum is better)
                    sorted_llms = sorted(llm_performance_scores.items(), key=lambda x: x[1])
                    keep_count = max(2, len(active_llms) // 2)
                    active_llms = [llm for llm, _ in sorted_llms[:keep_count]]
                    print(f"  Eliminated to {len(active_llms)} LLMs: {active_llms}")

                # Select top candidates and fuse
                topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=min(2, len(candidates_texts)))

                fusion_start = time.time()
                try:
                    fuse_generations = blender.fuse([question], topk_candidates, batch_size=len(topk_candidates[0]))
                    fused_answer = fuse_generations[0]
                    fusion_time = time.time() - fusion_start
                except Exception as e:
                    print(f"Fusion error: {str(e)}")
                    fused_answer = candidates_texts[0]  # Fallback to first candidate
                    fusion_time = 0

                fused_answers.append(fused_answer)

                # Log timing data
                timing_logs.append({
                    'conversation_idx': conv_idx,
                    'turn_idx': turn_idx,
                    'llm_generation_times': generation_times,
                    'ranking_time': ranking_time,
                    'fusion_time': fusion_time,
                    'active_llms_count': len(active_llms),
                    'total_turn_time': time.time() - turn_start
                })

        total_time = time.time() - start_time
        print(f"Dynamic Elimination completed in {total_time:.2f} seconds")

        self._save_experiment_results('dynamic_elimination', fused_answers, timing_logs)
        return fused_answers, timing_logs

    def experiment_alternate_ranking(self, blender, llm_list: List[str], inputs: List[List[Dict]]) -> Tuple[List[str], List[Dict]]:
        """
        Policy 2: Alternate ranking - only rank on even turns, reuse rankings on odd turns.
        """
        print("\n=== Experiment: Alternate Ranking ===")
        start_time = time.time()

        fused_answers = []
        timing_logs = []

        for conv_idx, conversation in enumerate(inputs):
            print(f"Processing conversation {conv_idx + 1}/{len(inputs)}")

            llm_contexts = {llm: [] for llm in llm_list}
            last_ranks = None
            last_candidates = None

            for turn_idx, turn in enumerate(conversation):
                turn_start = time.time()

                passage = turn['passage']
                question = turn['question']
                prompt = f"Passage:\n{passage}\n\nQuestion: {question}\n\nAnswer:"

                # Generate candidates from all LLMs
                candidates_texts = []
                generation_times = {}

                for llm in llm_list:
                    response, new_context, gen_time = self.generate_llm_response(
                        llm, prompt, llm_contexts[llm]
                    )
                    candidates_texts.append(response)
                    llm_contexts[llm] = new_context
                    generation_times[llm] = gen_time

                # Alternate ranking logic
                if turn_idx % 2 == 0:  # Even turns: perform ranking
                    rank_start = time.time()
                    try:
                        ranks = blender.rank([question], [candidates_texts], return_scores=False, batch_size=1)
                        ranking_time = time.time() - rank_start
                        last_ranks = ranks
                        last_candidates = candidates_texts
                    except Exception as e:
                        print(f"Ranking error: {str(e)}")
                        ranking_time = 0
                        ranks = [[list(range(len(candidates_texts)))]]
                        last_ranks = ranks
                        last_candidates = candidates_texts
                else:  # Odd turns: reuse previous ranking
                    ranks = last_ranks
                    ranking_time = 0  # No ranking performed
                    # Note: We still use current candidates_texts for fusion

                # Select top candidates and fuse
                topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=min(2, len(candidates_texts)))

                fusion_start = time.time()
                try:
                    fuse_generations = blender.fuse([question], topk_candidates, batch_size=len(topk_candidates[0]))
                    fused_answer = fuse_generations[0]
                    fusion_time = time.time() - fusion_start
                except Exception as e:
                    print(f"Fusion error: {str(e)}")
                    fused_answer = candidates_texts[0]
                    fusion_time = 0

                fused_answers.append(fused_answer)

                # Log timing data
                timing_logs.append({
                    'conversation_idx': conv_idx,
                    'turn_idx': turn_idx,
                    'llm_generation_times': generation_times,
                    'ranking_time': ranking_time,
                    'fusion_time': fusion_time,
                    'ranking_reused': turn_idx % 2 == 1,
                    'total_turn_time': time.time() - turn_start
                })

        total_time = time.time() - start_time
        print(f"Alternate Ranking completed in {total_time:.2f} seconds")

        self._save_experiment_results('alternate_ranking', fused_answers, timing_logs)
        return fused_answers, timing_logs

    def experiment_fixed_interval_elimination(self, blender, llm_list: List[str], inputs: List[List[Dict]]) -> Tuple[List[str], List[Dict]]:
        """
        Policy 3: Fixed-interval elimination - eliminate bottom LLMs every 3 conversations.
        """
        print("\n=== Experiment: Fixed Interval Elimination ===")
        start_time = time.time()

        fused_answers = []
        timing_logs = []

        # Global state across conversations
        global_llm_performance_scores = {llm: 0 for llm in llm_list}
        current_llm_committee = list(llm_list)
        conversation_counter = 0

        for conv_idx, conversation in enumerate(inputs):
            conversation_counter += 1
            print(f"Processing conversation {conversation_counter}/{len(inputs)}")

            # Reset for each LLM
            llm_contexts = {llm: [] for llm in current_llm_committee}
            conversation_llm_rank_sum = {llm: 0 for llm in current_llm_committee}

            # Check if this is the start of a new 3-conversation cycle
            if conversation_counter == 1 or (conversation_counter - 1) % 3 == 0:
                current_llm_committee = list(llm_list)  # Reset to full committee
                llm_contexts = {llm: [] for llm in current_llm_committee}
                conversation_llm_rank_sum = {llm: 0 for llm in current_llm_committee}
                print(f"  Starting new cycle with full committee ({len(current_llm_committee)} LLMs)")

            for turn_idx, turn in enumerate(conversation):
                turn_start = time.time()

                passage = turn['passage']
                question = turn['question']
                prompt = f"Passage:\n{passage}\n\nQuestion: {question}\n\nAnswer:"

                # Generate candidates from current committee
                candidates_texts = []
                generation_times = {}

                for llm in current_llm_committee:
                    response, new_context, gen_time = self.generate_llm_response(
                        llm, prompt, llm_contexts[llm]
                    )
                    candidates_texts.append(response)
                    llm_contexts[llm] = new_context
                    generation_times[llm] = gen_time

                # Ranking phase
                rank_start = time.time()
                try:
                    ranks = blender.rank([question], [candidates_texts], return_scores=False, batch_size=1)
                    ranking_time = time.time() - rank_start

                    # Update rank sums for first conversation in cycle
                    if (conversation_counter - 1) % 3 == 0:
                        for i, llm in enumerate(current_llm_committee):
                            conversation_llm_rank_sum[llm] += ranks[0][i]

                except Exception as e:
                    print(f"Ranking error: {str(e)}")
                    ranking_time = 0
                    ranks = [[list(range(len(candidates_texts)))]]

                # Select top candidates and fuse
                topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=min(2, len(candidates_texts)))

                fusion_start = time.time()
                try:
                    fuse_generations = blender.fuse([question], topk_candidates, batch_size=len(topk_candidates[0]))
                    fused_answer = fuse_generations[0]
                    fusion_time = time.time() - fusion_start
                except Exception as e:
                    print(f"Fusion error: {str(e)}")
                    fused_answer = candidates_texts[0]  # Fallback to first candidate
                    fusion_time = 0

                fused_answers.append(fused_answer)

                # Log timing data
                timing_logs.append({
                    'conversation_idx': conv_idx,
                    'turn_idx': turn_idx,
                    'llm_generation_times': generation_times,
                    'ranking_time': ranking_time,
                    'fusion_time': fusion_time,
                    'active_llms_count': len(current_llm_committee),
                    'cycle_position': (conversation_counter - 1) % 3,
                    'total_turn_time': time.time() - turn_start
                })

            # After first conversation in cycle, eliminate bottom performers
            if (conversation_counter - 1) % 3 == 0 and len(current_llm_committee) > 2:
                # Sort LLMs by performance (lower rank sum is better)
                sorted_llms = sorted(conversation_llm_rank_sum.items(), key=lambda x: x[1])
                keep_count = max(2, len(current_llm_committee) // 2)
                current_llm_committee = [llm for llm, _ in sorted_llms[:keep_count]]
                print(f"  Eliminated to {len(current_llm_committee)} LLMs: {current_llm_committee}")

        total_time = time.time() - start_time
        print(f"Fixed Interval Elimination completed in {total_time:.2f} seconds")

        self._save_experiment_results('fixed_interval_elimination', fused_answers, timing_logs)
        return fused_answers, timing_logs

    # def evaluate_with_bertscore(self, strategy_name: str, fused_answers: List[str], references: List[str]) -> float:
    #     """Evaluate fused answers using BERTScore and save results."""
    #     try:
    #         bertscore = evaluate.load("bertscore")
    #         results = bertscore.compute(predictions=fused_answers, references=references, lang="en")
    #         avg_score = np.mean(results['f1'])

    #         score_file = f"{self.
    
    def evaluate_with_bertscore(self, strategy_name: str, fused_answers: List[str], references: List[str]) -> float:
        """Evaluate fused answers using BERTScore and save results."""
        try:
            bertscore = evaluate.load("bertscore")
            results = bertscore.compute(predictions=fused_answers, references=references, lang="en")
            avg_score = np.mean(results['f1'])

            score_file = f"{self.results_dir}/bertscore_{strategy_name}.json"
            with open(score_file, 'w') as f:
                json.dump({
                    'average_f1': avg_score,
                    'individual_scores': results['f1'],
                    'strategy': strategy_name
                }, f, indent=2)

            print(f"  BERTScore F1 for {strategy_name}: {avg_score:.4f}")
            print(f"  Saved scores to {score_file}")
            return avg_score

        except Exception as e:
            print(f"Error calculating BERTScore for {strategy_name}: {str(e)}")
            return 0.0

    def extract_ground_truth_answers(self, inputs: List[List[Dict]]) -> List[str]:
        """Extract ground truth answers from input data for evaluation."""
        references = []
        for conversation in inputs:
            for turn in conversation:
                references.append(turn.get('answer', ''))
        return references

    def analyze_timing_performance(self, timing_logs: List[Dict], strategy_name: str) -> Dict:
        """Analyze timing performance and save summary statistics."""
        if not timing_logs:
            return {}

        # Extract relevant timing data
        generation_times = []
        ranking_times = []
        fusion_times = []
        total_times = []

        for log in timing_logs:
            # LLM generation times (sum per turn)
            gen_time = sum(log.get('llm_generation_times', {}).values())
            generation_times.append(gen_time)
            
            ranking_times.append(log.get('ranking_time', 0))
            fusion_times.append(log.get('fusion_time', 0))
            total_times.append(log.get('total_turn_time', 0))

        # Calculate statistics
        timing_stats = {
            'strategy': strategy_name,
            'total_turns': len(timing_logs),
            'generation_stats': {
                'mean': np.mean(generation_times),
                'std': np.std(generation_times),
                'total': sum(generation_times)
            },
            'ranking_stats': {
                'mean': np.mean(ranking_times),
                'std': np.std(ranking_times),
                'total': sum(ranking_times)
            },
            'fusion_stats': {
                'mean': np.mean(fusion_times),
                'std': np.std(fusion_times),
                'total': sum(fusion_times)
            },
            'total_stats': {
                'mean': np.mean(total_times),
                'std': np.std(total_times),
                'total': sum(total_times)
            }
        }

        # Save timing analysis
        timing_analysis_file = f"{self.results_dir}/timing_analysis_{strategy_name}.json"
        with open(timing_analysis_file, 'w') as f:
            json.dump(timing_stats, f, indent=2)

        print(f"  Timing analysis saved to {timing_analysis_file}")
        return timing_stats

    def run_complete_experiment(self):
        """
        Run the complete experimental pipeline comparing all strategies.
        """
        print("="*60)
        print("STARTING COMPLETE LLM BLENDER EXPERIMENT")
        print("="*60)

        # Step 1: Setup
        print("\n1. Setting up environment...")
        self.setup_ollama_input_json()

        # Step 2: Install LLMs (optional - comment out if already installed)
        print("\n2. Installing LLMs...")
        install_results = self.install_llms_parallel(self.llm_list)
        successful_llms = [llm for llm, success in install_results.items() if success]
        
        if len(successful_llms) < 2:
            print("Not enough LLMs installed successfully. Exiting.")
            return

        # Use only successfully installed LLMs
        self.llm_list = successful_llms
        print(f"Using LLMs: {self.llm_list}")

        # Step 3: Initialize dataset
        print("\n3. Initializing dataset...")
        inputs = self.dataset_init()
        if inputs is None:
            print("Failed to initialize dataset. Exiting.")
            return

        # Step 4: Initialize LLM Blender
        print("\n4. Initializing LLM Blender...")
        blender = self.blender_init()
        if blender is None:
            print("Failed to initialize LLM Blender. Exiting.")
            return

        # Step 5: Extract ground truth for evaluation
        references = self.extract_ground_truth_answers(inputs)

        # Step 6: Run all experiments
        print("\n5. Running experiments...")
        
        strategies = [
            ("Full Ranking (Baseline)", self.experiment_full_ranking),
            ("Dynamic Elimination", self.experiment_dynamic_elimination),
            ("Alternate Ranking", self.experiment_alternate_ranking),
            ("Fixed Interval Elimination", self.experiment_fixed_interval_elimination)
        ]

        results_summary = {}

        for strategy_name, strategy_func in strategies:
            print(f"\n{'='*40}")
            print(f"Running: {strategy_name}")
            print(f"{'='*40}")
            
            try:
                fused_answers, timing_logs = strategy_func(blender, self.llm_list, inputs)
                
                # Evaluate with BERTScore
                bertscore = self.evaluate_with_bertscore(
                    strategy_name.lower().replace(" ", "_").replace("(", "").replace(")", ""),
                    fused_answers, references
                )
                
                # Analyze timing performance
                timing_stats = self.analyze_timing_performance(
                    timing_logs, 
                    strategy_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                )
                
                results_summary[strategy_name] = {
                    'bertscore_f1': bertscore,
                    'total_time': timing_stats.get('total_stats', {}).get('total', 0),
                    'avg_turn_time': timing_stats.get('total_stats', {}).get('mean', 0),
                    'total_turns': len(timing_logs)
                }
                
            except Exception as e:
                print(f"Error running {strategy_name}: {str(e)}")
                results_summary[strategy_name] = {'error': str(e)}

        # Step 7: Generate final comparison report
        print("\n6. Generating final comparison report...")
        self.generate_comparison_report(results_summary)

        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED")
        print("="*60)

    def generate_comparison_report(self, results_summary: Dict):
        """Generate a comprehensive comparison report of all strategies."""
        report_file = f"{self.results_dir}/comparison_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("LLM BLENDER EXPERIMENT COMPARISON REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Experiment Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"LLMs Used: {', '.join(self.llm_list)}\n\n")
            
            # Performance summary table
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Strategy':<25} {'BERTScore F1':<12} {'Total Time (s)':<15} {'Avg Turn Time (s)':<18}\n")
            f.write("-" * 70 + "\n")
            
            for strategy, results in results_summary.items():
                if 'error' not in results:
                    f.write(f"{strategy:<25} {results['bertscore_f1']:<12.4f} {results['total_time']:<15.2f} {results['avg_turn_time']:<18.4f}\n")
                else:
                    f.write(f"{strategy:<25} {'ERROR':<12} {'N/A':<15} {'N/A':<18}\n")
            
            f.write("\n")
            
            # Best performing strategy
            valid_results = {k: v for k, v in results_summary.items() if 'error' not in v}
            if valid_results:
                best_quality = max(valid_results.items(), key=lambda x: x[1]['bertscore_f1'])
                fastest = min(valid_results.items(), key=lambda x: x[1]['total_time'])
                
                f.write("HIGHLIGHTS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best Quality (BERTScore): {best_quality[0]} ({best_quality[1]['bertscore_f1']:.4f})\n")
                f.write(f"Fastest Strategy: {fastest[0]} ({fastest[1]['total_time']:.2f}s)\n\n")
            
            # Detailed analysis
            f.write("DETAILED ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for strategy, results in results_summary.items():
                f.write(f"\n{strategy}:\n")
                if 'error' not in results:
                    f.write(f"  - Quality Score: {results['bertscore_f1']:.4f}\n")
                    f.write(f"  - Total Runtime: {results['total_time']:.2f} seconds\n")
                    f.write(f"  - Average Turn Time: {results['avg_turn_time']:.4f} seconds\n")
                    f.write(f"  - Total Turns Processed: {results['total_turns']}\n")
                else:
                    f.write(f"  - Error: {results['error']}\n")

        print(f"Comparison report saved to {report_file}")
        
        # Also print summary to console
        print("\nEXPERIMENT SUMMARY:")
        print("-" * 40)
        for strategy, results in results_summary.items():
            if 'error' not in results:
                print(f"{strategy}:")
                print(f"  BERTScore F1: {results['bertscore_f1']:.4f}")
                print(f"  Total Time: {results['total_time']:.2f}s")
                print()


def main():
    """Main execution function."""
    experiment = LLMBlenderExperiment()
    experiment.run_complete_experiment()


if __name__ == "__main__":
    main()