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
                    print(f"Successfully installed {llm_name}")
                    return True
                else:
                    print(f"Failed to install {llm_name}: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print(f" Timeout installing {llm_name}")
                return False
            except Exception as e:
                print(f" Error installing {llm_name}: {str(e)}")
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

    def load_inputs_from_file(self, filename='input_questions.txt'):
        """Load previously saved conversation questions from file."""
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

    def lone_llm_output_parallel(self, llm: str, inputs: List[List[str]]) -> Tuple[List[List[str]], float]:
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
        
        def process_conversation(conv_idx_and_questions):
            conv_idx, questions = conv_idx_and_questions
            responses = []
            context = []  # Each conversation starts with empty context
            
            for question in questions:
                response, context, _ = self.generate_llm_response(llm, question, context)
                responses.append(response)
            
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

    def experiment_full_ranking(self, blender, llm_list: List[str], inputs: List[List[str]]) -> Tuple[List[str], List[Dict]]:
        """
        Baseline strategy: Full ranking for every turn.
        This is the standard LLM Blender approach.
        """
        print("\n=== Experiment: Full Ranking (Baseline) ===")
        start_time = time.time()
        
        fused_answers = []
        timing_logs = []
        
        for conv_idx, conversation in enumerate(inputs):
            print(f"Processing conversation {conv_idx + 1}/{len(inputs)}")
            
            # Initialize context for each LLM
            llm_contexts = {llm: [] for llm in llm_list}
            
            for turn_idx, question in enumerate(conversation):
                turn_start = time.time()
                
                # Generate candidates from all LLMs
                candidates_texts = []
                generation_times = {}
                
                for llm in llm_list:
                    response, new_context, gen_time = self.generate_llm_response(
                        llm, question, llm_contexts[llm]
                    )
                    candidates_texts.append(response)
                    llm_contexts[llm] = new_context
                    generation_times[llm] = gen_time
                
                # Ranking phase
                rank_start = time.time()
                try:
                    ranks = blender.rank([question], [candidates_texts], return_scores=False, batch_size=1)
                    ranking_time = time.time() - rank_start
                except Exception as e:
                    print(f"Ranking error: {str(e)}")
                    ranking_time = 0
                    ranks = [[list(range(len(candidates_texts)))]]  # Default ranking
                
                # Select top candidates
                topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=2)
                
                # Fusion phase
                fusion_start = time.time()
                try:
                    fuse_generations = blender.fuse([question], topk_candidates, batch_size=2)
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
                    'total_turn_time': time.time() - turn_start
                })
        
        total_time = time.time() - start_time
        print(f"Full Ranking completed in {total_time:.2f} seconds")
        
        # Save results
        self._save_experiment_results('full_ranking', fused_answers, timing_logs)
        return fused_answers, timing_logs

    # def experiment_dynamic_elimination(self, blender, llm_list: List[str], inputs: List[List[str]]) -> Tuple[List[str], List[Dict]]:
    #     """
    #     Policy 1: Dynamic conversation-specific elimination.
    #     Eliminates bottom half of LLMs after half the conversation.
    #     """
    #     print("\n=== Experiment: Dynamic Elimination ===")
    #     start_time = time.time()
        
    #     fused_answers = []
    #     timing_logs = []
        
    #     for conv_idx, conversation in enumerate(inputs):
    #         print(f"Processing conversation {conv_idx + 1}/{len(inputs)}")
            
    #         # Reset for each conversation
    #         llm_contexts = {llm: [] for llm in llm_list}
    #         llm_performance_scores = {llm: 0 for llm in llm_list}
    #         active_llms = list(llm_list)
            
    #         elimination_point = math.ceil(len(conversation) / 2)
            
    #         for turn_idx, question in enumerate(conversation):
    #             turn_start = time.time()
                
    #             # Generate candidates from active LLMs
    #             candidates_texts = []
    #             generation_times = {}
                
    #             for llm in active_llms:
    #                 response, new_context, gen_time = self.generate_llm_response(
    #                     llm, question, llm_contexts[llm]
    #                 )
    #                 candidates_texts.append(response)
    #                 llm_contexts[llm] = new_context
    #                 generation_times[llm] = gen_time
                
    #             # Ranking phase
    #             rank_start = time.time()
    #             try:
    #                 ranks = blender.rank([question], [candidates_texts], return_scores=False, batch_size=1)
    #                 ranking_time = time.time() - rank_start
                    
    #                 # Update performance scores if we're still in the first half
    #                 if turn_idx < elimination_point:
    #                     for i, llm in enumerate(active_llms):
    #                         llm_performance_scores[llm] += ranks[0][i]
                    
    #             except Exception as e:
    #                 print(f"Ranking error: {str(e)}")
    #                 ranking_time = 0
    #                 ranks = [[list(range(len(candidates_texts)))]]
                
    #             # Dynamic elimination after first half
    #             if turn_idx == elimination_point and len(active_llms) > 2:
    #                 # Sort LLMs by performance (lower rank sum is better)
    #                 sorted_llms = sorted(llm_performance_scores.items(), key=lambda x: x[1])
    #                 keep_count = max(2, len(active_llms) // 2)
    #                 active_llms = [llm for llm, _ in sorted_llms[:keep_count]]
    #                 print(f"  Eliminated to {len(active_llms)} LLMs: {active_llms}")
                
    #             # Select top candidates and fuse
    #             topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=min(2, len(candidates_texts)))
                
    #             fusion_start = time.time()
    #             try:
    #                 fuse_generations = blender.fuse([question], topk_candidates, batch_size=len(topk_candidates[0]))
    #                 fused_answer = fuse_generations[0]
    #                 fusion_time = time.time() - fusion_start
    #             except Exception as e:
    #                 print(f"Fusion error: {str(e)}")
    #                 fused_answer = candidates_texts[0]
    #                 fusion_time = 0
                
    #             fused_answers.append(fused_answer)
                
    #             timing_logs.append({
    #                 'conversation_idx': conv_idx,
    #                 'turn_idx': turn_idx,
    #                 'llm_generation_times': generation_times,
    #                 'ranking_time': ranking_time,
    #                 'fusion_time': fusion_time,
    #                 'active_llms_count': len(active_llms),
    #                 'total_turn_time': time.time() - turn_start
    #             })
        
    #     total_time = time.time() - start_time
    #     print(f"Dynamic Elimination completed in {total_time:.2f} seconds")
        
    #     self._save_experiment_results('dynamic_elimination', fused_answers, timing_logs)
    #     return fused_answers, timing_logs

    def experiment_alternate_ranking(self, blender, llm_list: List[str], inputs: List[List[str]]) -> Tuple[List[str], List[Dict]]:
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
            
            for turn_idx, question in enumerate(conversation):
                turn_start = time.time()
                
                # Generate candidates from all LLMs
                candidates_texts = []
                generation_times = {}
                
                for llm in llm_list:
                    response, new_context, gen_time = self.generate_llm_response(
                        llm, question, llm_contexts[llm]
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
                topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=2)
                
                fusion_start = time.time()
                try:
                    fuse_generations = blender.fuse([question], topk_candidates, batch_size=2)
                    fused_answer = fuse_generations[0]
                    fusion_time = time.time() - fusion_start
                except Exception as e:
                    print(f"Fusion error: {str(e)}")
                    fused_answer = candidates_texts[0]
                    fusion_time = 0
                
                fused_answers.append(fused_answer)
                
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

    def experiment_fixed_interval_elimination(self, blender, llm_list: List[str], inputs: List[List[str]]) -> Tuple[List[str], List[Dict]]:
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
            
            llm_contexts = {llm: [] for llm in current_llm_committee}
            conversation_llm_rank_sum = {llm: 0 for llm in current_llm_committee}
            
            # Check if this is the start of a new 3-conversation cycle
            if conversation_counter == 1 or (conversation_counter - 1) % 3 == 0:
                current_llm_committee = list(llm_list)  # Reset to full committee
                llm_contexts = {llm: [] for llm in current_llm_committee}
                conversation_llm_rank_sum = {llm: 0 for llm in current_llm_committee}
                print(f"  Starting new cycle with full committee ({len(current_llm_committee)} LLMs)")
            
            for turn_idx, question in enumerate(conversation):
                turn_start = time.time()
                
                # Generate candidates from current committee
                candidates_texts = []
                generation_times = {}
                
                for llm in current_llm_committee:
                    response, new_context, gen_time = self.generate_llm_response(
                        llm, question, llm_contexts[llm]
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
                    fused_answer = candidates_texts[0]
                    fusion_time = 0
                
                fused_answers.append(fused_answer)
                
                timing_logs.append({
                    'conversation_idx': conv_idx,
                    'turn_idx': turn_idx,
                    'llm_generation_times': generation_times,
                    'ranking_time': ranking_time,
                    'fusion_time': fusion_time,
                    'committee_size': len(current_llm_committee),
                    'cycle_position': (conversation_counter - 1) % 3,
                    'total_turn_time': time.time() - turn_start
                })
            
            # After first conversation in cycle, eliminate bottom performers
            if (conversation_counter - 1) % 3 == 0 and len(current_llm_committee) > 2:
                sorted_llms = sorted(conversation_llm_rank_sum.items(), key=lambda x: x[1])
                keep_count = max(2, len(current_llm_committee) // 2)
                current_llm_committee = [llm for llm, _ in sorted_llms[:keep_count]]
                print(f"  Eliminated to {len(current_llm_committee)} LLMs for next 2 conversations")
        
        total_time = time.time() - start_time
        print(f"Fixed Interval Elimination completed in {total_time:.2f} seconds")
        
        self._save_experiment_results('fixed_interval_elimination', fused_answers, timing_logs)
        return fused_answers, timing_logs

    def _save_experiment_results(self, strategy_name: str, fused_answers: List[str], timing_logs: List[Dict]):
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
        
        print(f"  Saved results to {answers_file} and {timing_file}")

    def evaluate_with_bertscore(self, strategy_name: str, fused_answers: List[str], references: List[str]):
        """Evaluate fused answers using BERTScore and save results."""
        try:
            avg_score = self.calculate_bertscore(references, fused_answers)
            
            score_file = f"{self.results_dir}/bertscore_{strategy_name}.txt"
            with open(score_file, 'w') as f:
                f.write(f"Average BERTScore F1: {avg_score:.4f}\n")
                f.write(f"Total answers evaluated: {len(fused_answers)}\n")
            
            print(f"  BERTScore F1 for {strategy_name}: {avg_score:.4f}")
            return avg_score
            
        except Exception as e:
            print(f"Error evaluating {strategy_name}: {str(e)}")
            return 0.0

def main():
    """Main execution function."""
    print("🚀 Starting LLM Blender Comprehensive Evaluation")
    print("=" * 60)
    
    # Initialize experiment framework
    experiment = LLMBlenderExperiment()
    
    # Setup
    experiment.setup_ollama_input_json()
    
    # Load or create dataset
    inputs = experiment.load_inputs_from_file()
    if inputs is None:
        print("No existing dataset found, creating new one...")
        inputs = experiment.dataset_init()
        if inputs is None:
            print("❌ Failed to initialize dataset")
            return
    
    # Check if LLM outputs exist, otherwise generate them
    print("\n📋 Checking LLM outputs...")
    need_generation = False
    for llm in experiment.llm_list:
        normalized_name = experiment.normalize_llm_name(llm)
        if not os.path.exists(f"{experiment.testing_dir}/op_{normalized_name}.txt"):
            need_generation = True
            break
    
    if need_generation:
        print("Missing LLM outputs, installing and generating...")
        experiment.install_llms_parallel(experiment.llm_list)
        
        # Generate outputs for each LLM
        for llm in experiment.llm_list:
            experiment.lone_llm_output_parallel(llm, inputs)
    else:
        print("✓ All LLM outputs found")
    
    # Initialize blender
    print("\n🧪 Initializing LLM Blender...")
    blender = experiment.blender_init()
    if blender is None:
        print("❌ Failed to initialize blender")
        return
    
    # Prepare references for BERTScore evaluation
    # Using questions as references (proxy for how well answers address the questions)
    references = []
    for conversation in inputs:
        for question in conversation:
            references.append(question)
    
    print(f"\n📊 Running experiments on {len(inputs)} conversations...")
    print(f"Total turns to process: {len(references)}")
    
    # Run all experiments
    experiments_to_run = [
        ("Full Ranking", experiment.experiment_full_ranking),
        ("Dynamic Elimination", experiment.experiment_dynamic_elimination),
        ("Alternate Ranking", experiment.experiment_alternate_ranking),
        ("Fixed Interval Elimination", experiment.experiment_fixed_interval_elimination),
    ]