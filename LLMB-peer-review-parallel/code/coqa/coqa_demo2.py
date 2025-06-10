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
                passage = conv['passage']
                conversation_data = []
                for turn in conv['questions']:
                    conversation_data.append({
                        'passage': passage,
                        'question': turn['question'],
                        'answerable': turn['answerable'],
                        'explanation': turn.get('explanation', '') # Include explanation
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

    def evaluate_with_bertscore(self, strategy_name: str, fused_answers: List[str], references: List[str]) -> float:
        """Evaluate fused answers using BERTScore and save results."""
        try:
            bertscore = evaluate.load("bertscore")
            results = bertscore.compute(predictions=fused_answers, references=references, lang="en")
            avg_score = np.mean(results['f1'])

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

    # Prepare references and answerable flags for BERTScore evaluation
    references = []
    answerable_flags = []
    for conversation in inputs:
        for turn in conversation:
            if turn['answerable']:
                references.append(turn['answer']) # Use the actual answer as reference
            else:
                references.append(turn['explanation']) # Use the explanation as reference
            answerable_flags.append(turn['answerable'])

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
        
        # Calculate BERTScore, filtering for answerable questions
        bertscore_score = experiment.calculate_bertscore_filtered(references, fused_answers, answerable_flags)
        
        all_results[name] = {
            'fused_answers': fused_answers,
            'timing_logs': timing_logs,
            'bertscore_score_filtered': bertscore_score # Store the filtered score
        }

    print("\n--- Evaluation Summary ---")
    for name, results in all_results.items():
        print(f"{name}: Average BERTScore F1 (Answerable Only) = {results['bertscore_score_filtered']:.4f}")

if __name__ == "__main__":
    main()