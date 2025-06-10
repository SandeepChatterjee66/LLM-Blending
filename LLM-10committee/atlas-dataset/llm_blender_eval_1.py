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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, Counter
import statistics

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
        self.analysis_dir = 'Analysis_Results'
        
        # Setup directories
        self._setup_directories()
        
        # Initialize metrics
        self.bertscore_metric = None
        self.bleu_metric = None
        self.bleurt_metric = None
        self._initialize_metrics()
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.results_dir, self.testing_dir, self.analysis_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

    def _initialize_metrics(self):
        """Initialize evaluation metrics."""
        try:
            print("Initializing evaluation metrics...")
            self.bertscore_metric = evaluate.load("bertscore")
            print(" BERTScore loaded")
            
            self.bleu_metric = evaluate.load("bleu")
            print(" BLEU loaded")
            
            # Try to load BLEURT
            try:
                self.bleurt_metric = evaluate.load("bleurt", "bleurt-20")
                print(" BLEURT-20 loaded")
            except Exception as e:
                print(f" BLEURT not available: {str(e)}")
                print("  You can install it with: pip install bleurt")
                self.bleurt_metric = None
                
        except Exception as e:
            print(f"Error initializing metrics: {str(e)}")

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

    def dataset_init(self, filename='input_questions.txt'):
        """
        Initialize dataset by sampling from conv_questions and saving to file.
        Returns the sampled conversation data.
        """
        try:
            # Load the dataset
            print("Loading conv_questions dataset...")
            dataset = load_dataset("conv_questions")
            
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

    def calculate_comprehensive_metrics(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics including BLUERT."""
        metrics = {}
        
        try:
            # BERTScore
            if self.bertscore_metric:
                bertscore_results = self.bertscore_metric.compute(
                    predictions=candidates, references=references, lang="en"
                )
                metrics['bertscore_f1'] = np.mean(bertscore_results['f1'])
                metrics['bertscore_precision'] = np.mean(bertscore_results['precision'])
                metrics['bertscore_recall'] = np.mean(bertscore_results['recall'])
            
            # BLEU Score
            if self.bleu_metric:
                # BLEU expects references as list of lists
                bleu_references = [[ref.split()] for ref in references]
                bleu_predictions = [pred.split() for pred in candidates]
                bleu_result = self.bleu_metric.compute(
                    predictions=bleu_predictions, references=bleu_references
                )
                metrics['bleu'] = bleu_result['bleu']
            
            # BLEURT Score
            if self.bleurt_metric:
                bleurt_result = self.bleurt_metric.compute(
                    predictions=candidates, references=references
                )
                metrics['bleurt'] = np.mean(bleurt_result['scores'])
            
            # Response length statistics
            pred_lengths = [len(pred.split()) for pred in candidates]
            ref_lengths = [len(ref.split()) for ref in references]
            
            metrics['avg_pred_length'] = np.mean(pred_lengths)
            metrics['avg_ref_length'] = np.mean(ref_lengths)
            metrics['length_ratio'] = metrics['avg_pred_length'] / metrics['avg_ref_length'] if metrics['avg_ref_length'] > 0 else 0
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
        
        return metrics

    def print_ranking_analysis(self, ranks: List[List[int]], llm_list: List[str], turn_info: str = ""):
        """Print detailed ranking analysis."""
        if not ranks or not ranks[0]:
            return
        
        print(f"\n📊 Ranking Analysis {turn_info}")
        print("-" * 50)
        
        # Current turn ranking
        current_ranks = ranks[0]
        print("Current Turn Rankings (0=best):")
        for i, rank in enumerate(current_ranks):
            llm_name = llm_list[i] if i < len(llm_list) else f"LLM_{i}"
            print(f"  {llm_name:<15}: Rank {rank}")
        
        # Best and worst performers
        best_idx = current_ranks.index(min(current_ranks))
        worst_idx = current_ranks.index(max(current_ranks))
        best_llm = llm_list[best_idx] if best_idx < len(llm_list) else f"LLM_{best_idx}"
        worst_llm = llm_list[worst_idx] if worst_idx < len(llm_list) else f"LLM_{worst_idx}"
        
        print(f"\n Best Performer: {best_llm} (Rank {current_ranks[best_idx]})")
        print(f" Worst Performer: {worst_llm} (Rank {current_ranks[worst_idx]})")
        
        # Ranking array for easy copying
        print(f"\nRanking Array: {current_ranks}")

    def experiment_full_ranking(self, blender, llm_list: List[str], inputs: List[List[str]]) -> Tuple[List[str], List[Dict]]:
        """
        Baseline strategy: Full ranking for every turn.
        This is the standard LLM Blender approach.
        """
        print("\n=== Experiment: Full Ranking (Baseline) ===")
        start_time = time.time()
        
        fused_answers = []
        timing_logs = []
        ranking_history = []
        
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
                    
                    # Store ranking history
                    ranking_history.append({
                        'conversation_idx': conv_idx,
                        'turn_idx': turn_idx,
                        'ranks': ranks[0],
                        'llm_list': llm_list.copy()
                    })
                    
                    # Print ranking analysis
                    self.print_ranking_analysis(
                        ranks, llm_list, 
                        f"(Conv {conv_idx+1}, Turn {turn_idx+1})"
                    )
                    
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
                    'total_turn_time': time.time() - turn_start,
                    'ranks': ranks[0] if ranks else None
                })
        
        total_time = time.time() - start_time
        print(f"Full Ranking completed in {total_time:.2f} seconds")
        
        # Save results with ranking history
        self._save_experiment_results('full_ranking', fused_answers, timing_logs, ranking_history)
        return fused_answers, timing_logs

    def experiment_dynamic_elimination(self, blender, llm_list: List[str], inputs: List[List[str]]) -> Tuple[List[str], List[Dict]]:
        """
        Policy 1: Dynamic conversation-specific elimination.
        Eliminates bottom half of LLMs after half the conversation.
        """
        print("\n=== Experiment: Dynamic Elimination ===")
        start_time = time.time()
        
        fused_answers = []
        timing_logs = []
        ranking_history = []
        elimination_history = []
        
        for conv_idx, conversation in enumerate(inputs):
            print(f"Processing conversation {conv_idx + 1}/{len(inputs)}")
            
            # Reset for each conversation
            llm_contexts = {llm: [] for llm in llm_list}
            llm_performance_scores = {llm: 0 for llm in llm_list}
            active_llms = list(llm_list)
            
            elimination_point = math.ceil(len(conversation) / 2)
            
            for turn_idx, question in enumerate(conversation):
                turn_start = time.time()
                
                # Generate candidates from active LLMs
                candidates_texts = []
                generation_times = {}
                
                for llm in active_llms:
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
                    
                    # Store ranking history
                    ranking_history.append({
                        'conversation_idx': conv_idx,
                        'turn_idx': turn_idx,
                        'ranks': ranks[0],
                        'llm_list': active_llms.copy()
                    })
                    
                    # Print ranking analysis
                    self.print_ranking_analysis(
                        ranks, active_llms, 
                        f"(Conv {conv_idx+1}, Turn {turn_idx+1}, Active: {len(active_llms)})"
                    )
                    
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
                    eliminated_llms = [llm for llm, _ in sorted_llms[keep_count:]]
                    active_llms = [llm for llm, _ in sorted_llms[:keep_count]]
                    
                    elimination_event = {
                        'conversation_idx': conv_idx,
                        'turn_idx': turn_idx,
                        'eliminated_llms': eliminated_llms,
                        'remaining_llms': active_llms.copy(),
                        'performance_scores': dict(sorted_llms)
                    }
                    elimination_history.append(elimination_event)
                    
                    print(f"    ELIMINATION EVENT:")
                    print(f"    Eliminated: {eliminated_llms}")
                    print(f"    Remaining: {active_llms}")
                    print(f"    Performance scores: {dict(sorted_llms)}")
                
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
                    'active_llms_count': len(active_llms),
                    'total_turn_time': time.time() - turn_start,
                    'ranks': ranks[0] if ranks else None
                })
        
        total_time = time.time() - start_time
        print(f"Dynamic Elimination completed in {total_time:.2f} seconds")
        
        self._save_experiment_results('dynamic_elimination', fused_answers, timing_logs, ranking_history, elimination_history)
        return fused_answers, timing_logs

    def experiment_alternate_ranking(self, blender, llm_list: List[str], inputs: List[List[str]]) -> Tuple[List[str], List[Dict]]:
        """
        Policy 2: Alternate ranking - only rank on even turns, reuse rankings on odd turns.
        """
        print("\n=== Experiment: Alternate Ranking ===")
        start_time = time.time()
        
        fused_answers = []
        timing_logs = []
        ranking_history = []
        
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
                        
                        # Store ranking history
                        ranking_history.append({
                            'conversation_idx': conv_idx,
                            'turn_idx': turn_idx,
                            'ranks': ranks[0],
                            'llm_list': llm_list.copy(),
                            'ranking_performed': True
                        })
                        
                        # Print ranking analysis
                        self.print_ranking_analysis(
                            ranks, llm_list, 
                            f"(Conv {conv_idx+1}, Turn {turn_idx+1}, NEW RANKING)"
                        )
                        
                    except Exception as e:
                        print(f"Ranking error: {str(e)}")
                        ranking_time = 0
                        ranks = [[list(range(len(candidates_texts)))]]
                        last_ranks = ranks
                        last_candidates = candidates_texts
                else:  # Odd turns: reuse previous ranking
                    ranks = last_ranks
                    ranking_time = 0  # No ranking performed
                    
                    # Store ranking history
                    ranking_history.append({
                        'conversation_idx': conv_idx,
                        'turn_idx': turn_idx,
                        'ranks': ranks[0] if ranks else None,
                        'llm_list': llm_list.copy(),
                        'ranking_performed': False
                    })
                    
                    print(f"\n📊 Ranking Analysis (Conv {conv_idx+1}, Turn {turn_idx+1}, REUSED RANKING)")
                    print("-" * 50)
                    print("Reusing previous ranking:")
                    if ranks and ranks[0]:
                        for i, rank in enumerate(ranks[0]):
                            llm_name = llm_list[i] if i < len(llm_list) else f"LLM_{i}"
                            print(f"  {llm_name:<15}: Rank {rank}")
                        print(f"Ranking Array: {ranks[0]}")
                
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
                    'total_turn_time': time.time() - turn_start,
                    'ranks': ranks[0] if ranks else None
                })
        
        total_time = time.time() - start_time
        print(f"Alternate Ranking completed in {total_time:.2f} seconds")
        
        self._save_experiment_results('alternate_ranking', fused_answers, timing_logs, ranking_history)
        return fused_answers, timing_logs

    # def experiment_fixed_interval_elimination(self, blender, llm_list: List[str], inputs: List[List[str]]) -> Tuple[List[str], List[Dict]]:
    #     """
    #     Policy 3: Fixed-interval elimination - eliminate bottom LLMs every 3 conversations.
    #     """
    #     print("\n=== Experiment: Fixed Interval Elimination ===")
    #     start_time = time.time()
        
    #     fused_answers = []
    #     timing_logs = []
    #     ranking_history = []
    #     elimination_history = []
        
    #     # Global state across conversations
    #     global_llm_performance_scores = {llm: 0 for llm in llm_list}
    #     current_llm_committee = list(llm_list)
    #     conversation_counter = 0
        
    #     for conv_idx, conversation in enumerate(inputs):
    #         conversation_counter += 1
    #         print(f"Processing conversation {conversation_counter}/{len(inputs)}")
            
    #         llm_contexts = {llm: [] for llm in current_llm_committee}
    #         conversation_llm_rank_sum = {llm: 0 for llm in current_llm_committee}
            
    #         # Check if this is the start of a new 3-conversation cycle
    #         if conversation_counter == 1 or (conversation_counter - 1) % 3 == 0
    
    def experiment_fixed_interval_elimination(self, blender, llm_list: List[str], inputs: List[List[str]]) -> Tuple[List[str], List[Dict]]:
        """
        Policy 3: Fixed-interval elimination - eliminate bottom LLMs every 3 conversations.
        """
        print("\n=== Experiment: Fixed Interval Elimination ===")
        start_time = time.time()
        
        fused_answers = []
        timing_logs = []
        ranking_history = []
        elimination_history = []
        
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
                print(f"   Starting new 3-conversation cycle")
            
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
                    
                    # Store ranking history
                    ranking_history.append({
                        'conversation_idx': conv_idx,
                        'turn_idx': turn_idx,
                        'ranks': ranks[0],
                        'llm_list': current_llm_committee.copy(),
                        'cycle_position': (conversation_counter - 1) % 3 + 1
                    })
                    
                    # Print ranking analysis
                    self.print_ranking_analysis(
                        ranks, current_llm_committee, 
                        f"(Conv {conv_idx+1}, Turn {turn_idx+1}, Committee: {len(current_llm_committee)})"
                    )
                    
                    # Update scores
                    for i, llm in enumerate(current_llm_committee):
                        rank_score = ranks[0][i]
                        global_llm_performance_scores[llm] += rank_score
                        conversation_llm_rank_sum[llm] += rank_score
                    
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
                    'cycle_position': (conversation_counter - 1) % 3 + 1,
                    'total_turn_time': time.time() - turn_start,
                    'ranks': ranks[0] if ranks else None
                })
            
            # Fixed interval elimination every 3 conversations
            if conversation_counter % 3 == 0 and len(current_llm_committee) > 2:
                print(f"\n  FIXED INTERVAL ELIMINATION (after {conversation_counter} conversations)")
                
                # Sort by global performance
                sorted_llms = sorted(global_llm_performance_scores.items(), key=lambda x: x[1])
                
                # Keep top performers (at least 2)
                elimination_count = len(current_llm_committee) // 3
                if elimination_count == 0:
                    elimination_count = 1
                
                keep_count = max(2, len(current_llm_committee) - elimination_count)
                eliminated_llms = [llm for llm, _ in sorted_llms[keep_count:] if llm in current_llm_committee]
                current_llm_committee = [llm for llm, _ in sorted_llms[:keep_count] if llm in current_llm_committee]
                
                elimination_event = {
                    'conversation_idx': conv_idx,
                    'elimination_cycle': conversation_counter // 3,
                    'eliminated_llms': eliminated_llms,
                    'remaining_committee': current_llm_committee.copy(),
                    'global_performance_scores': dict(sorted_llms)
                }
                elimination_history.append(elimination_event)
                
                print(f"    Eliminated: {eliminated_llms}")
                print(f"    Remaining committee: {current_llm_committee}")
                print(f"    Global scores: {dict(sorted_llms)}")
        
        total_time = time.time() - start_time
        print(f"Fixed Interval Elimination completed in {total_time:.2f} seconds")
        
        self._save_experiment_results('fixed_interval_elimination', fused_answers, timing_logs, ranking_history, elimination_history)
        return fused_answers, timing_logs

'''
    def experiment_weighted_fusion(self, blender, llm_list: List[str], inputs: List[List[str]]) -> Tuple[List[str], List[Dict]]:
        """
        Policy 4: Weighted fusion based on historical performance.
        Uses ranking scores to weight fusion instead of hard selection.
        """
        print("\n=== Experiment: Weighted Fusion ===")
        start_time = time.time()
        
        fused_answers = []
        timing_logs = []
        ranking_history = []
        
        # Historical performance tracking
        llm_historical_performance = {llm: [] for llm in llm_list}
        
        for conv_idx, conversation in enumerate(inputs):
            print(f"Processing conversation {conv_idx + 1}/{len(inputs)}")
            
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
                    ranks = blender.rank([question], [candidates_texts], return_scores=True, batch_size=1)
                    ranking_time = time.time() - rank_start
                    
                    # Extract rank scores for weighting
                    rank_scores = ranks[0] if isinstance(ranks[0][0], (int, float)) else [0] * len(candidates_texts)
                    
                    # Store ranking history
                    ranking_history.append({
                        'conversation_idx': conv_idx,
                        'turn_idx': turn_idx,
                        'rank_scores': rank_scores,
                        'llm_list': llm_list.copy()
                    })
                    
                    # Update historical performance
                    for i, llm in enumerate(llm_list):
                        llm_historical_performance[llm].append(rank_scores[i])
                    
                    print(f"\n📊 Weighted Fusion Analysis (Conv {conv_idx+1}, Turn {turn_idx+1})")
                    print("-" * 50)
                    print("Current Rank Scores:")
                    for i, score in enumerate(rank_scores):
                        llm_name = llm_list[i] if i < len(llm_list) else f"LLM_{i}"
                        avg_score = np.mean(llm_historical_performance[llm_name]) if llm_historical_performance[llm_name] else 0
                        print(f"  {llm_name:<15}: Score {score:.3f}, Avg: {avg_score:.3f}")
                    
                except Exception as e:
                    print(f"Ranking error: {str(e)}")
                    ranking_time = 0
                    rank_scores = [1.0] * len(candidates_texts)  # Equal weights as fallback
                
                # Weighted fusion (custom implementation)
                fusion_start = time.time()
                try:
                    # Convert scores to weights (higher score = higher weight)
                    weights = np.array(rank_scores)
                    weights = np.exp(weights - np.max(weights))  # Numerical stability
                    weights = weights / np.sum(weights)  # Normalize
                    
                    # Select top 3 candidates by weight for fusion
                    top_indices = np.argsort(weights)[-3:]
                    top_candidates = [candidates_texts[i] for i in top_indices]
                    
                    if len(top_candidates) > 1:
                        fuse_generations = blender.fuse([question], [top_candidates], batch_size=len(top_candidates))
                        fused_answer = fuse_generations[0]
                    else:
                        fused_answer = top_candidates[0]
                    
                    fusion_time = time.time() - fusion_start
                    
                    print(f"  Top weights: {[f'{w:.3f}' for w in sorted(weights, reverse=True)[:3]]}")
                    
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
                    'rank_scores': rank_scores,
                    'weights': weights.tolist() if 'weights' in locals() else None,
                    'total_turn_time': time.time() - turn_start
                })
        
        total_time = time.time() - start_time
        print(f"Weighted Fusion completed in {total_time:.2f} seconds")
        
        self._save_experiment_results('weighted_fusion', fused_answers, timing_logs, ranking_history)
        return fused_answers, timing_logs
'''

    def _save_experiment_results(self, experiment_name: str, fused_answers: List[str], timing_logs: List[Dict], 
                                ranking_history: List[Dict], elimination_history: List[Dict] = None):
        """Save experiment results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save fused answers
        answers_file = f"{self.results_dir}/{experiment_name}_answers_{timestamp}.txt"
        with open(answers_file, 'w') as f:
            f.write(str(fused_answers))
        
        # Save timing logs
        timing_file = f"{self.results_dir}/{experiment_name}_timing_{timestamp}.json"
        with open(timing_file, 'w') as f:
            json.dump(timing_logs, f, indent=2, default=str)
        
        # Save ranking history
        ranking_file = f"{self.results_dir}/{experiment_name}_rankings_{timestamp}.json"
        with open(ranking_file, 'w') as f:
            json.dump(ranking_history, f, indent=2, default=str)
        
        # Save elimination history if provided
        if elimination_history:
            elimination_file = f"{self.results_dir}/{experiment_name}_eliminations_{timestamp}.json"
            with open(elimination_file, 'w') as f:
                json.dump(elimination_history, f, indent=2, default=str)
        
        print(f"Results saved with timestamp: {timestamp}")

    def run_comprehensive_comparison(self, inputs: List[List[str]], reference_answers: List[str] = None):
        """
        Run all experimental strategies and compare their performance.
        """
        print("\n" + "="*80)
        print(" COMPREHENSIVE LLM BLENDER STRATEGY COMPARISON")
        print("="*80)
        
        # Initialize blender
        blender = self.blender_init()
        if not blender:
            print("Failed to initialize blender. Aborting.")
            return
        
        # Store results from all experiments
        all_results = {}
        
        # Run all experiments
        experiments = [
            ('Full Ranking (Baseline)', self.experiment_full_ranking),
            ('Dynamic Elimination', self.experiment_dynamic_elimination),
            ('Alternate Ranking', self.experiment_alternate_ranking),
            ('Fixed Interval Elimination', self.experiment_fixed_interval_elimination),
            ('Weighted Fusion', self.experiment_weighted_fusion)
        ]
        
        for exp_name, exp_function in experiments:
            try:
                print(f"\n{'='*20} Starting {exp_name} {'='*20}")
                answers, timing_logs = exp_function(blender, self.llm_list, inputs)
                all_results[exp_name] = {
                    'answers': answers,
                    'timing_logs': timing_logs
                }
                print(f"✓ {exp_name} completed successfully")
            except Exception as e:
                print(f"✗ {exp_name} failed: {str(e)}")
                all_results[exp_name] = {'error': str(e)}
        
        # Performance analysis
        self._analyze_comparative_performance(all_results, reference_answers)
        
        return all_results

    def _analyze_comparative_performance(self, all_results: Dict, reference_answers: List[str] = None):
        """Analyze and compare performance across all strategies."""
        print("\n" + "="*80)
        print(" COMPARATIVE PERFORMANCE ANALYSIS")
        print("="*80)
        
        performance_summary = {}
        
        for strategy_name, results in all_results.items():
            if 'error' in results:
                print(f"\n❌ {strategy_name}: Failed ({results['error']})")
                continue
            
            timing_logs = results['timing_logs']
            answers = results['answers']
            
            # Calculate timing statistics
            total_time = sum(log['total_turn_time'] for log in timing_logs)
            avg_turn_time = total_time / len(timing_logs) if timing_logs else 0
            
            generation_times = []
            ranking_times = []
            fusion_times = []
            
            for log in timing_logs:
                if 'llm_generation_times' in log:
                    generation_times.extend(log['llm_generation_times'].values())
                if 'ranking_time' in log:
                    ranking_times.append(log['ranking_time'])
                if 'fusion_time' in log:
                    fusion_times.append(log['fusion_time'])
            
            strategy_stats = {
                'total_time': total_time,
                'avg_turn_time': avg_turn_time,
                'avg_generation_time': np.mean(generation_times) if generation_times else 0,
                'avg_ranking_time': np.mean(ranking_times) if ranking_times else 0,
                'avg_fusion_time': np.mean(fusion_times) if fusion_times else 0,
                'total_turns': len(timing_logs),
                'answer_count': len(answers)
            }
            
            # Calculate quality metrics if reference answers provided
            if reference_answers and len(answers) == len(reference_answers):
                quality_metrics = self.calculate_comprehensive_metrics(reference_answers, answers)
                strategy_stats.update(quality_metrics)
            
            performance_summary[strategy_name] = strategy_stats
        
        # Display results
        self._display_performance_comparison(performance_summary)
        
        # Save analysis results
        self._save_comparative_analysis(performance_summary)

    def _display_performance_comparison(self, performance_summary: Dict):
        """Display formatted performance comparison."""
        print("\n TIMING PERFORMANCE COMPARISON")
        print("-" * 60)
        print(f"{'Strategy':<25} {'Total Time':<12} {'Avg/Turn':<10} {'Ranking':<10} {'Fusion':<10}")
        print("-" * 60)
        
        for strategy, stats in performance_summary.items():
            print(f"{strategy:<25} {stats['total_time']:<12.2f} {stats['avg_turn_time']:<10.3f} "
                  f"{stats['avg_ranking_time']:<10.3f} {stats['avg_fusion_time']:<10.3f}")
        
        # Quality metrics if available
        quality_metrics = ['bertscore_f1', 'bleu', 'bleurt']
        has_quality_data = any(metric in list(performance_summary.values())[0] for metric in quality_metrics)
        
        if has_quality_data:
            print("\n📊 QUALITY METRICS COMPARISON")
            print("-" * 60)
            header = f"{'Strategy':<25}"
            for metric in quality_metrics:
                if any(metric in stats for stats in performance_summary.values()):
                    header += f" {metric.upper():<10}"
            print(header)
            print("-" * 60)
            
            for strategy, stats in performance_summary.items():
                row = f"{strategy:<25}"
                for metric in quality_metrics:
                    if metric in stats:
                        row += f" {stats[metric]:<10.3f}"
                    else:
                        row += f" {'N/A':<10}"
                print(row)
        
        # Speed improvement analysis
        print("\n⚡ SPEED IMPROVEMENTS (vs Baseline)")
        print("-" * 40)
        baseline_time = performance_summary.get('Full Ranking (Baseline)', {}).get('total_time', 0)
        
        if baseline_time > 0:
            for strategy, stats in performance_summary.items():
                if strategy != 'Full Ranking (Baseline)':
                    speedup = (baseline_time - stats['total_time']) / baseline_time * 100
                    print(f"{strategy:<25}: {speedup:>+6.1f}%")

    def _save_comparative_analysis(self, performance_summary: Dict):
        """Save comparative analysis results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        analysis_file = f"{self.analysis_dir}/comparative_analysis_{timestamp}.json"
        
        with open(analysis_file, 'w') as f:
            json.dump(performance_summary, f, indent=2, default=str)
        
        print(f"\n💾 Comparative analysis saved to: {analysis_file}")

    # def create_visualization_dashboard(self, all_results: Dict):
    #     """Create comprehensive visualization dashboard."""
    #     print("\n Creating Visualization Dashboard...")
   
    #     # Extract data for visualization
    #     strategies = list(all_results.keys())
    #     total_times = []
    #     avg_turn_times = []
        
    #     for strategy in strategies:
    #         if 'timing_logs' in all_results[strategy]:
    #             timing_logs = all_results[strategy]['timing_logs']
    #             total_time = sum(log['total_turn_time'] for log in timing_logs)
    #             avg_turn_time = total_time / len(timing_logs) if timing_logs else 0
    #             total_times.append(total_time)
    #             avg_turn_times.append(avg_turn_time)
        
    #     # Create multi-panel visualization
    #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    #     fig.suptitle('LLM Blender Strategy Comparison Dashboard', fontsize=16, fontweight='bold')
        
    #     # Panel 1: Total execution time
    #     colors = sns.color_palette("husl", len(strategies))
    #     bars1 = ax1.bar(strategies, total_times, color=colors)
    #     ax1.set_title('Total Execution Time by Strategy')
    #     ax1.set_ylabel('Time (seconds)')
    #     ax1.tick_params(axis='x', rotation=45)
        
    #     # Add value labels on bars
    #     for bar, time in zip(bars1, total_times):
    #         ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
    #                 f'{time:.1f}s', ha='center', va='bottom')
        
    #     # Panel 2: Average turn time
    #     bars2 = ax2.bar(strategies, avg_turn_times, color=colors)
    #     ax2.set_title('Average Time per Turn')
    #     ax2.set_ylabel('Time (seconds)')
    #     ax2.tick_params(axis='x', rotation=45)
        
    #     for bar, time in zip(bars2, avg_turn_times):
    #         ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
    #                 f'{time:.2f}s', ha='center', va='bottom')
        
    #     # Panel 3: Speed improvement vs baseline
    #     if total_times and 'Full Ranking (Baseline)' in strategies:
    #         baseline_idx = strategies.index('Full Ranking (Baseline)')
    #         baseline_time = total_times[baseline_idx]
    #         speedups = []
            
    #         for i, time in enumerate(total_times):
    #             if i != baseline_idx:
    #                 speedup = (baseline_time - time) / baseline_time * 100
    #                 speedups.append(speedup)
    #             else:
    #                 speedups.append(0)
            
    #         bars3 = ax3.bar(strategies, speedups, color=colors)
    #         ax3.set_title('Speed Improvement vs Baseline (%)')
    #         ax3.set_ylabel('Improvement (%)')
    #         ax3.tick_params(axis='x', rotation=45)
    #         ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
    #         for bar, speedup in zip(bars3, speedups):
    #             ax3.text(bar.get_x() + bar.get_width()/2, 
    #                     bar.get_height() + (1 if speedup >= 0 else -3),
    #                     f'{speedup:+.1f}%', ha='center', va='bottom' if speedup >= 0 else 'top')
        
        # Panel 4: Strategy complexity analysis
        complexity_scores = []
        complexity_labels = []
        
        for strategy in strategies:
            if 'Dynamic' in strategy:
                complexity_scores.append(3)
                complexity_labels.append('High')
            elif 'Alternate' in strategy or 'Weighted' in strategy:
                complexity_scores.append(2)
                complexity_labels.append('Medium')
            elif 'Fixed' in strategy:
                complexity_scores.append(2.5)
                complexity_labels.append('Med-High')
            else:
                complexity_scores.append(1)
                complexity_labels.append('Low')
        
        bars4 = ax4.bar(strategies, complexity_scores, color=colors)
        ax4.set_title('Implementation Complexity')
        ax4.set_ylabel('Complexity Score')
        ax4.set_ylim(0, 4)
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, label in zip(bars4, complexity_labels):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    label, ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # viz_file = f"{self.analysis_dir}/dashboard_{timestamp}.png"
        # plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        # print(f" Dashboard saved to: {viz_file}")
        
        plt.show()

def main():
    """Main execution function demonstrating the complete workflow."""
    print(" LLM Blender Strategy Evaluation Framework")
    print("=" * 60)
    
    # Initialize experiment
    experiment = LLMBlenderExperiment()
    
    # Setup phase
    print("\n1️ Setup Phase")
    print("-" * 30)
    
    # Install LLMs (uncomment if needed)
    # experiment.install_llms_parallel(experiment.llm_list)
    
    # Setup Ollama input template
    experiment.setup_ollama_input_json()
    
    # Initialize dataset
    print("\n2️ Dataset Initialization")
    print("-" * 30)
    inputs = experiment.dataset_init()
    
    if inputs is None:
        print("Trying to load from existing file...")
        inputs = experiment.load_inputs_from_file()
    
    if inputs is None:
        print(" Failed to initialize dataset. Please check your setup.")
        return
    
    print(f" Dataset ready with {len(inputs)} conversations")
    
    # Run comprehensive comparison
    print("\n3️ Running Comprehensive Strategy Comparison")
    print("-" * 50)
    
    all_results = experiment.run_comprehensive_comparison(inputs)
    
    # # Create visualization dashboard
    # print("\n4️Generating Visualization Dashboard")
    # print("-" * 40)
    # experiment.create_visualization_dashboard(all_results)
    
    print("\n Experiment completed successfully!")
    print(f" Results saved in: {experiment.results_dir}")
    print(f" Analysis saved in: {experiment.analysis_dir}")

if __name__ == "__main__":
    main()