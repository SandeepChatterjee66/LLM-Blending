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
from typing import List, Dict, Tuple, Any, Optional
import math
import re

class DoQALLMBlenderExperiment:
    """
    A comprehensive evaluation framework for LLM Blender strategies adapted for Document-based QA.
    Tests different approaches to combining LLM responses with DoQA-specific performance tracking.
    """
    
    def __init__(self):
        # Configuration
        self.llm_list = ['mistral', 'llama3', 'gemma:2b', 'phi3', 'qwen:4b', 'deepseek-llm', 'stablelm2']
        self.api_url = 'http://127.0.0.1:11434/api/generate'
        self.results_dir = 'DoQA_Experiment_Results'
        self.testing_dir = 'doqa_testing'
        
        # # DoQA-specific settings
        # self.doqa_prompt_template = 
#infer accurate answers from the provided document.
# Document: {document}
# Question: {question}
# Instructions:
# - Extract the most accurate and concise answer directly from the document
# - If the answer cannot be found in the document, respond with "[N/A]"
# - Focus on factual accuracy and faithfulness to the source
# - Avoid adding external knowledge not present in the document
# Answer:

        self.unanswerable_indicator = "[N/A]"
        
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

    def load_doqa_dataset(self, dataset_name='squad', num_samples=50):
        """
        Load DoQA-compatible dataset (SQuAD, Natural Questions, etc.)
        Returns list of (document, question, answer) tuples.
        """
        try:
            print(f"Loading {dataset_name} dataset for DoQA evaluation...")
            
            if dataset_name == 'squad':
                dataset = load_dataset("squad")
                test_data = dataset['validation']  # Use validation as test
            elif dataset_name == 'natural_questions':
                dataset = load_dataset("natural_questions")
                test_data = dataset['validation']
            else:
                # Default to SQuAD
                dataset = load_dataset("squad")
                test_data = dataset['validation']
            
            # Sample data points
            sample_indices = random.sample(range(len(test_data)), min(num_samples, len(test_data)))
            doqa_samples = []
            
            for idx in sample_indices:
                sample = test_data[idx]
                
                if dataset_name == 'squad':
                    document = sample['context']
                    question = sample['question']
                    # Use first answer if multiple exist
                    answer = sample['answers']['text'][0] if sample['answers']['text'] else self.unanswerable_indicator
                else:
                    # Adapt for other datasets as needed
                    document = str(sample.get('context', sample.get('passage', '')))
                    question = str(sample.get('question', ''))
                    answer = str(sample.get('answer', self.unanswerable_indicator))
                
                doqa_samples.append({
                    'document': document,
                    'question': question,
                    'reference_answer': answer,
                    'sample_id': idx
                })
            
            print(f"Loaded {len(doqa_samples)} DoQA samples")
            return doqa_samples
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {str(e)}")
            return self._create_sample_doqa_data()

    def save_doqa_dataset(self, doqa_samples: List[Dict], filename='doqa_samples.json'):
        """Save DoQA samples to file."""
        with open(filename, 'w') as f:
            json.dump(doqa_samples, f, indent=2)
        print(f"Saved {len(doqa_samples)} DoQA samples to {filename}")

    def load_doqa_dataset_from_file(self, filename='doqa_samples.json'):
        """Load previously saved DoQA samples from file."""
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return None
        
        try:
            with open(filename, 'r') as f:
                doqa_samples = json.load(f)
            print(f"Loaded {len(doqa_samples)} DoQA samples from {filename}")
            return doqa_samples
        except Exception as e:
            print(f"Error loading from {filename}: {str(e)}")
            return None

    def generate_doqa_response(self, llm_name: str, document: str, question: str, context: List = None) -> Tuple[str, List, float]:
        """
        Generate DoQA response from a specific LLM via Ollama API.
        Uses DoQA-specific prompt template.
        Returns (response_text, new_context, generation_time).
        """
        if context is None:
            context = []
            
        start_time = time.time()
        
        # Format DoQA prompt
        doqa_prompt = self.doqa_prompt_template.format(
            document=document,
            question=question
        )
        
        try:
            # Prepare request payload
            with open('input.json', 'r') as f:
                payload = json.load(f)
            
            payload.update({
                'model': llm_name,
                'prompt': doqa_prompt,
                'context': context
            })
            
            # Make API request
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            generation_time = time.time() - start_time
            
            # Extract answer from response
            answer = result.get('response', '').strip()
            
            # Clean up the answer (remove common response prefixes)
            answer = self._clean_doqa_answer(answer)
            
            return answer, result.get('context', []), generation_time
            
        except Exception as e:
            print(f"Error generating DoQA response from {llm_name}: {str(e)}")
            generation_time = time.time() - start_time
            return f"Error: {str(e)}", context, generation_time

    def _clean_doqa_answer(self, answer: str) -> str:
        """Clean and normalize DoQA answers."""
        # Remove common response prefixes
        prefixes_to_remove = [
            "Answer:", "The answer is:", "Based on the document:",
            "According to the text:", "From the document:",
            "The document states:", "As mentioned in the document:"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Handle quotes
        answer = answer.strip('"\'')
        
        # Limit answer length for conciseness
        if len(answer) > 200:
            answer = answer[:200].strip()
        
        return answer

    def normalize_llm_name(self, llm_name: str) -> str:
        """Normalize LLM name for file naming (replace special characters)."""
        return llm_name.replace(':', '-').replace('/', '-')

    def generate_doqa_candidates_parallel(self, llm: str, doqa_samples: List[Dict]) -> Tuple[List[str], float]:
        """
        Generate DoQA responses for a single LLM across all samples.
        Uses parallel processing for efficiency.
        """
        normalized_name = self.normalize_llm_name(llm)
        output_file = f"{self.testing_dir}/doqa_op_{normalized_name}.txt"
        
        # Check if outputs already exist
        if os.path.exists(output_file):
            print(f"Loading existing DoQA outputs for {llm}...")
            return self.load_candidates_from_file(f"doqa_op_{normalized_name}"), 0.0
        
        print(f"Generating DoQA responses for {llm}...")
        start_time = time.time()
        
        def process_sample(sample_data):
            sample_idx, sample = sample_data
            try:
                response, _, _ = self.generate_doqa_response(
                    llm, sample['document'], sample['question']
                )
                return sample_idx, response
            except Exception as e:
                print(f"Error processing sample {sample_idx} for {llm}: {str(e)}")
                return sample_idx, self.unanswerable_indicator
        
        # Process samples in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            indexed_samples = [(i, sample) for i, sample in enumerate(doqa_samples)]
            future_to_sample = {executor.submit(process_sample, item): item[0] 
                               for item in indexed_samples}
            
            # Collect results in order
            all_responses = [None] * len(doqa_samples)
            for future in concurrent.futures.as_completed(future_to_sample):
                sample_idx = future_to_sample[future]
                try:
                    idx, response = future.result()
                    all_responses[idx] = response
                except Exception as e:
                    print(f"Error collecting result for sample {sample_idx}: {str(e)}")
                    all_responses[sample_idx] = self.unanswerable_indicator
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(str(all_responses))
        
        total_time = time.time() - start_time
        print(f"Completed DoQA responses for {llm} in {total_time:.2f} seconds")
        return all_responses, total_time

    def load_candidates_from_file(self, llm_name_normalized: str) -> List[str]:
        """Load pre-generated LLM responses from file."""
        filename = f"{self.testing_dir}/{llm_name_normalized}.txt"
        try:
            with open(filename, 'r', encoding='utf-8') as f:
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

    def calculate_doqa_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate DoQA-specific metrics including Exact Match, F1, and BERTScore."""
        metrics = {}
        
        try:
            # Exact Match
            exact_matches = [1 if pred.strip().lower() == ref.strip().lower() else 0 
                           for pred, ref in zip(predictions, references)]
            metrics['exact_match'] = np.mean(exact_matches)
            
            # F1 Score (token-level)
            f1_scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = set(pred.lower().split())
                ref_tokens = set(ref.lower().split())
                
                if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                    f1_scores.append(1.0)
                elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
                    f1_scores.append(0.0)
                else:
                    precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
                    recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    f1_scores.append(f1)
            
            metrics['f1_score'] = np.mean(f1_scores)
            
            # BERTScore
            bertscore = evaluate.load("bertscore")
            bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")
            metrics['bertscore_f1'] = np.mean(bert_results['f1'])
            
            # Answerability accuracy (checking for [N/A] handling)
            unanswerable_predictions = [1 if self.unanswerable_indicator in pred else 0 for pred in predictions]
            unanswerable_references = [1 if self.unanswerable_indicator in ref else 0 for ref in references]
            
            if sum(unanswerable_references) > 0:
                metrics['answerability_accuracy'] = np.mean([
                    1 if up == ur else 0 for up, ur in zip(unanswerable_predictions, unanswerable_references)
                ])
            else:
                metrics['answerability_accuracy'] = 1.0  # No unanswerable questions
                
        except Exception as e:
            print(f"Error calculating DoQA metrics: {str(e)}")
            metrics = {'exact_match': 0.0, 'f1_score': 0.0, 'bertscore_f1': 0.0, 'answerability_accuracy': 0.0}
        
        return metrics

    def experiment_doqa_full_ranking(self, blender, llm_list: List[str], doqa_samples: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        DoQA Baseline: Full ranking for every question.
        Standard LLM Blender approach adapted for document-based QA.
        """
        print("\n=== DoQA Experiment: Full Ranking (Baseline) ===")
        start_time = time.time()
        
        fused_answers = []
        timing_logs = []
        
        for sample_idx, sample in enumerate(doqa_samples):
            print(f"Processing DoQA sample {sample_idx + 1}/{len(doqa_samples)}")
            
            turn_start = time.time()
            document = sample['document']
            question = sample['question']
            
            # Generate candidates from all LLMs
            candidates_texts = []
            generation_times = {}
            
            for llm in llm_list:
                response, _, gen_time = self.generate_doqa_response(llm, document, question)
                candidates_texts.append(response)
                generation_times[llm] = gen_time
            
            # Ranking phase
            rank_start = time.time()
            try:
                # Use document + question as context for ranking
                ranking_input = f"Document: {document}\nQuestion: {question}"
                ranks = blender.rank([ranking_input], [candidates_texts], return_scores=False, batch_size=1)
                ranking_time = time.time() - rank_start
            except Exception as e:
                print(f"Ranking error: {str(e)}")
                ranking_time = 0
                ranks = [[list(range(len(candidates_texts)))]]  # Default ranking
            
            # Select top candidates
            topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=min(3, len(candidates_texts)))
            
            # Fusion phase
            fusion_start = time.time()
            try:
                # Include document context in fusion
                fusion_input = f"Document: {document}\nQuestion: {question}"
                fuse_generations = blender.fuse([fusion_input], topk_candidates, batch_size=len(topk_candidates[0]))
                fused_answer = fuse_generations[0]
                fusion_time = time.time() - fusion_start
            except Exception as e:
                print(f"Fusion error: {str(e)}")
                fused_answer = candidates_texts[0]  # Fallback to first candidate
                fusion_time = 0
            
            # Clean the fused answer
            fused_answer = self._clean_doqa_answer(fused_answer)
            fused_answers.append(fused_answer)
            
            # Log timing data
            timing_logs.append({
                'sample_idx': sample_idx,
                'llm_generation_times': generation_times,
                'ranking_time': ranking_time,
                'fusion_time': fusion_time,
                'total_sample_time': time.time() - turn_start,
                'document_length': len(document),
                'question_length': len(question)
            })
        
        total_time = time.time() - start_time
        print(f"DoQA Full Ranking completed in {total_time:.2f} seconds")
        
        # Save results
        self._save_doqa_experiment_results('doqa_full_ranking', fused_answers, timing_logs, doqa_samples)
        return fused_answers, timing_logs

    def experiment_doqa_document_aware_elimination(self, blender, llm_list: List[str], doqa_samples: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        DoQA Policy 1: Document-aware dynamic elimination.
        Eliminates LLMs based on their document comprehension performance.
        """
        print("\n=== DoQA Experiment: Document-Aware Elimination ===")
        start_time = time.time()
        
        fused_answers = []
        timing_logs = []
        
        # Track LLM performance on document comprehension
        llm_comprehension_scores = {llm: [] for llm in llm_list}
        active_llms = list(llm_list)
        elimination_threshold = len(doqa_samples) // 3  # Eliminate after 1/3 of samples
        
        for sample_idx, sample in enumerate(doqa_samples):
            print(f"Processing DoQA sample {sample_idx + 1}/{len(doqa_samples)}")
            
            turn_start = time.time()
            document = sample['document']
            question = sample['question']
            
            # Generate candidates from active LLMs
            candidates_texts = []
            generation_times = {}
            
            for llm in active_llms:
                response, _, gen_time = self.generate_doqa_response(llm, document, question)
                candidates_texts.append(response)
                generation_times[llm] = gen_time
            
            # Ranking phase with document context
            rank_start = time.time()
            try:
                ranking_input = f"Document: {document}\nQuestion: {question}"
                ranks = blender.rank([ranking_input], [candidates_texts], return_scores=True, batch_size=1)
                ranking_time = time.time() - rank_start
                
                # Update comprehension scores based on ranking
                if sample_idx < elimination_threshold:
                    for i, llm in enumerate(active_llms):
                        # Higher score is better for comprehension
                        score = ranks[1][0][i] if len(ranks) > 1 else 1.0 / (ranks[0][0][i] + 1)
                        llm_comprehension_scores[llm].append(score)
                
            except Exception as e:
                print(f"Ranking error: {str(e)}")
                ranking_time = 0
                ranks = [[list(range(len(candidates_texts)))]]
            
            # Dynamic elimination based on document comprehension
            if sample_idx == elimination_threshold and len(active_llms) > 2:
                # Calculate average comprehension scores
                avg_scores = {}
                for llm in active_llms:
                    if llm_comprehension_scores[llm]:
                        avg_scores[llm] = np.mean(llm_comprehension_scores[llm])
                    else:
                        avg_scores[llm] = 0.0
                
                # Keep top performers
                sorted_llms = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
                keep_count = max(2, len(active_llms) // 2)
                active_llms = [llm for llm, _ in sorted_llms[:keep_count]]
                print(f"  Eliminated to {len(active_llms)} LLMs based on document comprehension: {active_llms}")
            
            # Select top candidates and fuse
            topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=min(3, len(candidates_texts)))
            
            fusion_start = time.time()
            try:
                fusion_input = f"Document: {document}\nQuestion: {question}"
                fuse_generations = blender.fuse([fusion_input], topk_candidates, batch_size=len(topk_candidates[0]))
                fused_answer = fuse_generations[0]
                fusion_time = time.time() - fusion_start
            except Exception as e:
                print(f"Fusion error: {str(e)}")
                fused_answer = candidates_texts[0]
                fusion_time = 0
            
            fused_answer = self._clean_doqa_answer(fused_answer)
            fused_answers.append(fused_answer)
            
            timing_logs.append({
                'sample_idx': sample_idx,
                'llm_generation_times': generation_times,
                'ranking_time': ranking_time,
                'fusion_time': fusion_time,
                'active_llms_count': len(active_llms),
                'total_sample_time': time.time() - turn_start,
                'document_length': len(document),
                'question_length': len(question)
            })
        
        total_time = time.time() - start_time
        print(f"DoQA Document-Aware Elimination completed in {total_time:.2f} seconds")
        
        self._save_doqa_experiment_results('doqa_document_aware_elimination', fused_answers, timing_logs, doqa_samples)
        return fused_answers, timing_logs

    def experiment_doqa_context_caching(self, blender, llm_list: List[str], doqa_samples: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        DoQA Policy 2: Context-aware caching strategy.
        Cache rankings for similar documents to reduce computation.
        """
        print("\n=== DoQA Experiment: Context-Aware Caching ===")
        start_time = time.time()
        
        fused_answers = []
        timing_logs = []
        
        # Cache for similar document contexts
        document_cache = {}
        similarity_threshold = 0.7  # Jaccard similarity threshold
        
        for sample_idx, sample in enumerate(doqa_samples):
            print(f"Processing DoQA sample {sample_idx + 1}/{len(doqa_samples)}")
            
            turn_start = time.time()
            document = sample['document']
            question = sample['question']
            
            # Check cache for similar documents
            cached_ranks = None
            doc_tokens = set(document.lower().split())
            
            for cached_doc, cached_data in document_cache.items():
                cached_tokens = set(cached_doc.lower().split())
                similarity = len(doc_tokens & cached_tokens) / len(doc_tokens | cached_tokens)
                
                if similarity > similarity_threshold:
                    cached_ranks = cached_data['ranks']
                    print(f"  Using cached ranking (similarity: {similarity:.3f})")
                    break
            
            # Generate candidates from all LLMs
            candidates_texts = []
            generation_times = {}
            
            for llm in llm_list:
                response, _, gen_time = self.generate_doqa_response(llm, document, question)
                candidates_texts.append(response)
                generation_times[llm] = gen_time
            
            # Ranking phase (use cache if available)
            if cached_ranks is not None:
                ranks = cached_ranks
                ranking_time = 0.0  # No ranking computation needed
            else:
                rank_start = time.time()
                try:
                    ranking_input = f"Document: {document}\nQuestion: {question}"
                    ranks = blender.rank([ranking_input], [candidates_texts], return_scores=False, batch_size=1)
                    ranking_time = time.time() - rank_start
                    
                    # Cache the ranking for this document
                    document_cache[document] = {
                        'ranks': ranks,
                        'timestamp': time.time()
                    }
                    
                    # Limit cache size
                    if len(document_cache) > 20:
                        oldest_doc = min(document_cache.keys(), 
                                       key=lambda k: document_cache[k]['timestamp'])
                        del document_cache[oldest_doc]
                        
                except Exception as e:
                    print(f"Ranking error: {str(e)}")
                    ranking_time = 0
                    ranks = [[list(range(len(candidates_texts)))]]
            
            # Select top candidates and fuse
            topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=min(3, len(candidates_texts)))
            
            fusion_start = time.time()
            try:
                fusion_input = f"Document: {document}\nQuestion: {question}"
                fuse_generations = blender.fuse([fusion_input], topk_candidates, batch_size=len(topk_candidates[0]))
                fused_answer = fuse_generations[0]
                fusion_time = time.time() - fusion_start
            except Exception as e:
                print(f"Fusion error: {str(e)}")
                fused_answer = candidates_texts[0]
                fusion_time = 0
            
            fused_answer = self._clean_doqa_answer(fused_answer)
            fused_answers.append(fused_answer)
            
            timing_logs.append({
                'sample_idx': sample_idx,
                'llm_generation_times': generation_times,
                'ranking_time': ranking_time,
                'fusion_time': fusion_time,
                'used_cache': cached_ranks is not None,
                'cache_size': len(document_cache),
                'total_sample_time': time.time() - turn_start,
                'document_length': len(document),
                'question_length': len(question)
            })
        
        total_time = time.time() - start_time
        print(f"DoQA Context-Aware Caching completed in {total_time:.2f} seconds")
        
        self._save_doqa_experiment_results('doqa_context_caching', fused_answers, timing_logs, doqa_samples)
        return fused_answers, timing_logs
    
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

    def experiment_dynamic_elimination(self, blender, llm_list: List[str], inputs: List[List[str]]) -> Tuple[List[str], List[Dict]]:
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
            
            # Reset for each LLM
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
    print(" Starting LLM Blender Comprehensive Evaluation")
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
            print(" Failed to initialize dataset")
            return
    
    # Check if LLM outputs exist, otherwise generate them
    print("\n Checking LLM outputs...")
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
        print(" Failed to initialize blender")
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
    
    all_results = {}
    
    for name, experiment_func in experiments_to_run:
        fused_answers, timing_logs = experiment_func(blender, experiment.llm_list, inputs)
        bertscore_score = experiment.evaluate_with_bertscore(name, fused_answers, references)
        all_results[name] = {
            'fused_answers': fused_answers,
            'timing_logs': timing_logs,
            'bertscore_score': bertscore_score
        }
        
    print("\n--- Evaluation Summary ---")
    for name, results in all_results.items():
        print(f"{name}: Average BERTScore F1 = {results['bertscore_score']:.4f}")

if __name__ == "__main__":
    main()