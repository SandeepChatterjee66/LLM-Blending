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

    def dataset_init(self, filename='convai2_data.json'): # Changed filename
        """
        Initialize dataset by sampling from ConvAI2 (Persona-Chat)
        and saving to file. Returns the sampled conversation data
        in a format suitable for conversational turns.
        """
        try:
            # Load the ConvAI2 dataset (Persona-Chat)
            print("Loading ConvAI2 (Persona-Chat) dataset...")
            # We use 'original_no_cands' configuration for simplicity, focusing on turns
            dataset = load_dataset("persona_chat", "original_no_cands")

            # Sample from the 'train' split for demonstration, or 'validation'/'test' if preferred
            # Note: For real evaluation, use 'test' or 'validation' splits, not train.
            data_split = dataset['train'] # Using 'train' for quick testing
            sample_size = 5 # Number of individual conversation turns to sample

            # Randomly sample individual turns (utterance-response pairs)
            # Each example in Persona-Chat represents one turn.
            sample_indices = random.sample(range(len(data_split)), min(sample_size, len(data_split)))
            sampled_turns = [data_split[i] for i in sample_indices]

            # Store data in a more structured format for multi-turn processing
            # We'll store lists of dictionaries, where each dict represents a turn
            # { 'id', 'context', 'question', 'answer' }
            # For ConvAI2, 'context' is persona + history.
            # 'question' is the 'utterance', 'answer' is the 'labels'.
            formatted_conversations = []

            for i, turn_data in enumerate(sampled_turns):
                # Construct context: persona + history
                persona_context = " ".join(turn_data['persona'])
                # History is already interleaved turns
                history_context = " ".join(turn_data['history'])
                
                # Combine them, consider adding delimiters or roles if needed
                # For basic setup, a simple concatenation works.
                # More advanced: "Persona: [persona_facts] History: [dialog_turns]"
                full_context = f"Persona: {persona_context}. Dialogue History: {history_context}."
                
                question_or_statement = turn_data['utterance']
                
                # 'labels' is typically a list, take the first one as the ground truth response
                ground_truth_response = turn_data['labels'][0] if turn_data['labels'] else ""

                formatted_conversations.append({
                    'id': f"turn_{i}", # Assign a unique ID for each turn
                    'context': full_context.strip(),
                    'question': question_or_statement.strip(),
                    'answer': ground_truth_response.strip()
                })
            
            # Save to a JSON file for easier loading later
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(formatted_conversations, f, indent=2)
            
            print(f"Saved {len(formatted_conversations)} conversational turns to {filename}")
            return formatted_conversations # Return the structured data
            
        except Exception as e:
            print(f"Error initializing ConvAI2 dataset: {str(e)}")
            return None

    def load_inputs_from_file(self, filename='convai2_data.json'): # Changed filename
        """Load previously saved conversation data from file (structured for ConvAI2)."""
        if not os.path.exists(filename):
            print(f"File {filename} not found.")
            return None
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                inputs = json.load(f)
            print(f"Loaded {len(inputs)} conversational turns from {filename}")
            return inputs
        except Exception as e:
            print(f"Error loading from {filename}: {str(e)}")
            return None

    def generate_llm_response(self, llm_name: str, prompt: str, conversation_history: List[Dict[str, str]] = None) -> Tuple[str, List[Dict[str, str]], float]:
        """
        Generate response from a specific LLM via Ollama API.
        `conversation_history` should be a list of dicts like [{"role": "user", "content": "..."}]
        Returns (response_text, updated_conversation_history, generation_time).
        """
        if conversation_history is None:
            conversation_history = []
            
        start_time = time.time()
        
        try:
            # Append current prompt to history for this turn
            current_messages = conversation_history + [{"role": "user", "content": prompt}]

            payload = {
                'model': llm_name,
                'messages': current_messages, # Ollama uses 'messages' for chat models
                'stream': False
            }
            
            response = requests.post(self.api_url, json=payload, timeout=120) # Increased timeout
            response.raise_for_status()
            
            result = response.json()
            generation_time = time.time() - start_time
            
            # Ollama returns the new message in the 'message' field of the response
            # and potentially an updated 'context' token list (though 'messages' is more common for chat)
            
            # Update history with the model's response for the next turn
            # Assuming the response is under 'message' field and has 'content'
            model_response_content = result.get('message', {}).get('content', '')
            updated_history = current_messages + [{"role": "assistant", "content": model_response_content}]
            
            return model_response_content, updated_history, generation_time
            
        except Exception as e:
            print(f"Error generating response from {llm_name}: {str(e)}")
            generation_time = time.time() - start_time
            # Return original history + an error message as the response
            return f"Error: {str(e)}", conversation_history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"Error: {str(e)}"}], generation_time

'''
def dataset_init(self, dataset_name: str, num_conversations: int = 50) -> List[List[Dict[str, str]]]:
    """
    Initialize dataset for ConvAI2.
    Returns a list of conversations, where each conversation is a list of turns.
    Each turn contains: {'id', 'persona', 'context', 'question', 'answer'}
    
    Args:
        dataset_name: Name of the dataset (should be 'convai2')
        num_conversations: Number of conversations to sample
        
    Returns:
        List[List[Dict[str, str]]]: List of conversations, each containing turns
    """
    if dataset_name.lower() != 'convai2':
        raise ValueError(f"This dataset_init is specifically for ConvAI2, got: {dataset_name}")
    
    # Load ConvAI2 dataset
    try:
        from datasets import load_dataset
        dataset = load_dataset("convai2", split="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load ConvAI2 dataset: {e}")
    
    # Sample conversations
    sampled_indices = random.sample(range(len(dataset)), min(num_conversations, len(dataset)))
    formatted_conversations = []
    
    for idx in sampled_indices:
        conversation_data = dataset[idx]
        
        # Extract persona information
        persona_info = conversation_data.get('personality', [])
        persona_text = " ".join(persona_info) if persona_info else "No persona provided."
        
        # Extract the conversation history
        conv_history = conversation_data.get('history', [])
        
        if not conv_history:
            continue  # Skip conversations with no history
        
        # Build the conversation turns
        conversation_turns = []
        accumulated_context = f"Persona: {persona_text}\n\nConversation:"
        
        # Process each turn in the conversation
        for turn_idx, turn in enumerate(conv_history):
            # Each turn should be a string representing what was said
            if turn_idx == 0:
                # First turn - this is typically the opening message
                current_context = accumulated_context
                question = turn
                
                # For the first turn, we might not have a previous response to use as 'answer'
                # We'll use the next turn as the expected response, if available
                if turn_idx + 1 < len(conv_history):
                    answer = conv_history[turn_idx + 1]
                else:
                    answer = "Thank you for sharing that with me."
                
                turn_data = {
                    'id': f"conv_{idx}_turn_{turn_idx}",
                    'persona': persona_text,
                    'context': current_context,
                    'question': question,
                    'answer': answer
                }
                conversation_turns.append(turn_data)
                
                # Update accumulated context
                accumulated_context += f"\nPerson 1: {question}"
                accumulated_context += f"\nPerson 2: {answer}"
                
            elif turn_idx % 2 == 0:  # Even indices (0, 2, 4, ...) are Person 1's turns
                current_context = accumulated_context
                question = turn
                
                # Get the response (next turn) if available
                if turn_idx + 1 < len(conv_history):
                    answer = conv_history[turn_idx + 1]
                else:
                    answer = "I understand. Thank you for the conversation."
                
                turn_data = {
                    'id': f"conv_{idx}_turn_{turn_idx}",
                    'persona': persona_text,
                    'context': current_context,
                    'question': question,
                    'answer': answer
                }
                conversation_turns.append(turn_data)
                
                # Update accumulated context
                accumulated_context += f"\nPerson 1: {question}"
                accumulated_context += f"\nPerson 2: {answer}"
        
        # Only add conversations that have at least one turn
        if conversation_turns:
            formatted_conversations.append(conversation_turns)
    
    print(f"Loaded {len(formatted_conversations)} conversations from ConvAI2 dataset")
    print(f"Average turns per conversation: {sum(len(conv) for conv in formatted_conversations) / len(formatted_conversations):.1f}")
    
    return formatted_conversations
'''


def dataset_init(self, dataset_name: str, num_conversations: int = 50) -> List[List[Dict[str, str]]]:
    """
    Alternative ConvAI2 dataset initialization that handles both personas in a conversation.
    This version treats the conversation as a true back-and-forth between two people with different personas.
    """
    if dataset_name.lower() != 'convai2':
        raise ValueError(f"This dataset_init is specifically for ConvAI2, got: {dataset_name}")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("convai2", split="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load ConvAI2 dataset: {e}")
    
    sampled_indices = random.sample(range(len(dataset)), min(num_conversations, len(dataset)))
    formatted_conversations = []
    
    for idx in sampled_indices:
        conversation_data = dataset[idx]
        
        # Extract both personas
        persona1 = conversation_data.get('personality', [])
        persona2 = conversation_data.get('candidates', [])[:4] if conversation_data.get('candidates') else []
        
        persona1_text = " ".join(persona1) if persona1 else "No persona provided."
        persona2_text = " ".join(persona2) if persona2 else "No persona provided."
        
        # Get conversation history
        conv_history = conversation_data.get('history', [])
        
        if len(conv_history) < 2:
            continue  # Skip conversations that are too short
        
        conversation_turns = []
        
        # Build conversation context progressively
        for turn_idx in range(0, len(conv_history) - 1, 2):
            # Person 1's turn
            if turn_idx < len(conv_history):
                person1_msg = conv_history[turn_idx]
                person2_msg = conv_history[turn_idx + 1] if turn_idx + 1 < len(conv_history) else ""
                
                # Build context up to this point
                context_history = []
                for i in range(0, turn_idx, 2):
                    if i < len(conv_history):
                        context_history.append(f"Person 1: {conv_history[i]}")
                    if i + 1 < len(conv_history):
                        context_history.append(f"Person 2: {conv_history[i + 1]}")
                
                context = f"Person 1 Persona: {persona1_text}\nPerson 2 Persona: {persona2_text}\n\n"
                if context_history:
                    context += "Previous conversation:\n" + "\n".join(context_history) + "\n\n"
                context += f"Current message from Person 1: {person1_msg}"
                
                turn_data = {
                    'id': f"conv_{idx}_turn_{turn_idx}",
                    'persona': persona2_text,  # AI takes role of Person 2
                    'context': context,
                    'question': person1_msg,
                    'answer': person2_msg
                }
                conversation_turns.append(turn_data)
        
        if conversation_turns:
            formatted_conversations.append(conversation_turns)
    
    print(f"Loaded {len(formatted_conversations)} conversations from ConvAI2 dataset")
    print(f"Total turns: {sum(len(conv) for conv in formatted_conversations)}")
    
    return formatted_conversations


def dataset_init_simple(self, dataset_name: str, num_conversations: int = 50) -> List[List[Dict[str, str]]]:
    """
    Simplified ConvAI2 dataset initialization focusing on question-answer pairs.
    Each conversation becomes a series of context-aware Q&A turns.
    """
    if dataset_name.lower() != 'convai2':
        raise ValueError(f"This dataset_init is specifically for ConvAI2, got: {dataset_name}")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("convai2", split="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load ConvAI2 dataset: {e}")
    
    sampled_indices = random.sample(range(len(dataset)), min(num_conversations, len(dataset)))
    formatted_conversations = []
    
    for idx in sampled_indices:
        conversation_data = dataset[idx]
        
        # Extract persona
        persona_info = conversation_data.get('personality', [])
        persona_text = " ".join(persona_info) if persona_info else ""
        
        # Extract conversation
        conv_history = conversation_data.get('history', [])
        
        if len(conv_history) < 2:
            continue
        
        conversation_turns = []
        
        # Create turns with progressive context
        for i in range(len(conv_history) - 1):
            # Current message is the question
            question = conv_history[i]
            # Next message is the answer
            answer = conv_history[i + 1]
            
            # Build context from previous messages
            context_messages = conv_history[:i]
            context = ""
            
            if persona_text:
                context += f"Your persona: {persona_text}\n\n"
            
            if context_messages:
                context += "Previous conversation:\n"
                for j, msg in enumerate(context_messages):
                    speaker = "You" if j % 2 == 1 else "Them"
                    context += f"{speaker}: {msg}\n"
                context += "\n"
            
            context += f"They said: {question}\nYou should respond:"
            
            turn_data = {
                'id': f"conv_{idx}_turn_{i}",
                'persona': persona_text,
                'context': context,
                'question': question,
                'answer': answer
            }
            conversation_turns.append(turn_data)
        
        if conversation_turns:
            formatted_conversations.append(conversation_turns)
    
    print(f"Processed {len(formatted_conversations)} conversations")
    print(f"Total turns: {sum(len(conv) for conv in formatted_conversations)}")
    
    return formatted_conversations

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

# def main():
#     """Main execution function."""
#     print(" Starting LLM Blender Comprehensive Evaluation")
#     print("=" * 60)
    
#     # Initialize experiment framework
#     experiment = LLMBlenderExperiment()
    
#     # Setup
#     experiment.setup_ollama_input_json()
    
#     # Load or create dataset
#     inputs = experiment.load_inputs_from_file()
#     if inputs is None:
#         print("No existing dataset found, creating new one...")
#         inputs = experiment.dataset_init()
#         if inputs is None:
#             print(" Failed to initialize dataset")
#             return
    
#     # Check if LLM outputs exist, otherwise generate them
#     print("\n Checking LLM outputs...")
#     need_generation = False
#     for llm in experiment.llm_list:
#         normalized_name = experiment.normalize_llm_name(llm)
#         if not os.path.exists(f"{experiment.testing_dir}/op_{normalized_name}.txt"):
#             need_generation = True
#             break
    
#     if need_generation:
#         print("Missing LLM outputs, installing and generating...")
#         experiment.install_llms_parallel(experiment.llm_list)
        
#         # Generate outputs for each LLM
#         for llm in experiment.llm_list:
#             experiment.lone_llm_output_parallel(llm, inputs)
#     else:
#         print("✓ All LLM outputs found")
    
#     # Initialize blender
#     print("\n🧪 Initializing LLM Blender...")
#     blender = experiment.blender_init()
#     if blender is None:
#         print(" Failed to initialize blender")
#         return
    
#     # Prepare references for BERTScore evaluation
#     # Using questions as references (proxy for how well answers address the questions)
#     references = []
#     for conversation in inputs:
#         for question in conversation:
#             references.append(question)
    
#     print(f"\n📊 Running experiments on {len(inputs)} conversations...")
#     print(f"Total turns to process: {len(references)}")
    
#     # Run all experiments
#     experiments_to_run = [
#         ("Full Ranking", experiment.experiment_full_ranking),
#         ("Dynamic Elimination", experiment.experiment_dynamic_elimination),
#         ("Alternate Ranking", experiment.experiment_alternate_ranking),
#         ("Fixed Interval Elimination", experiment.experiment_fixed_interval_elimination),
#     ]
    
#     all_results = {}
    
#     for name, experiment_func in experiments_to_run:
#         fused_answers, timing_logs = experiment_func(blender, experiment.llm_list, inputs)
#         bertscore_score = experiment.evaluate_with_bertscore(name, fused_answers, references)
#         all_results[name] = {
#             'fused_answers': fused_answers,
#             'timing_logs': timing_logs,
#             'bertscore_score': bertscore_score
#         }
        
#     print("\n--- Evaluation Summary ---")
#     for name, results in all_results.items():
#         print(f"{name}: Average BERTScore F1 = {results['bertscore_score']:.4f}")

# if __name__ == "__main__":
#     main()

def _save_experiment_results(self, strategy_name: str, fused_answers: List[str], timing_logs: List[Dict]):
        """Save experiment results to files."""
        # Ensure fused_answers is a flat list of all responses
        # The experiments now return a list of responses, one per turn across all conversations.
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
        """
        Evaluate fused answers using BERTScore.
        References are now the ground-truth responses from ConvAI2.
        """
        try:
            # Ensure BERTScore is loaded
            if not hasattr(self, '_bertscore_metric'):
                self._bertscore_metric = evaluate.load("bertscore")

            # Check lengths
            if len(fused_answers) != len(references):
                print(f"Warning: Mismatch in lengths for {strategy_name}. Fused answers: {len(fused_answers)}, References: {len(references)}")
                # Adjust to the minimum length to avoid errors
                min_len = min(len(fused_answers), len(references))
                fused_answers = fused_answers[:min_len]
                references = references[:min_len]
                if not fused_answers: # If after trimming, it's empty
                    print(f"Skipping BERTScore for {strategy_name} due to empty lists after mismatch correction.")
                    return 0.0

            results = self._bertscore_metric.compute(predictions=fused_answers, references=references, lang="en")
            avg_score = np.mean(results['f1'])
            
            score_file = f"{self.results_dir}/bertscore_{strategy_name}.txt"
            with open(score_file, 'w') as f:
                f.write(f"Average BERTScore F1: {avg_score:.4f}\n")
                f.write(f"Total answers evaluated: {len(fused_answers)}\n")
            
            print(f"  BERTScore F1 for {strategy_name}: {avg_score:.4f}")
            return avg_score
            
        except Exception as e:
            print(f"Error calculating BERTScore for {strategy_name}: {str(e)}")
            return 0.0


def main():
    """Main execution function."""
    print(" Starting LLM Blender Comprehensive Evaluation for ConvAI2")
    print("=" * 60)
    
    # Initialize experiment framework
    experiment = LLMBlenderExperiment()
    
    # Setup
    experiment.setup_ollama_input_json()
    
    # Load or create dataset
    # `inputs` will now be a list of structured turns, not just questions.
    structured_inputs = experiment.load_inputs_from_file()
    if structured_inputs is None:
        print("No existing dataset found, creating new one...")
        structured_inputs = experiment.dataset_init()
        if structured_inputs is None:
            print(" Failed to initialize dataset")
            return
    
    # No `lone_llm_output_parallel` needed as experiments manage per-LLM context now.
    # We still need to ensure LLMs are installed.
    print("\n Checking LLM installations...")
    experiment.install_llms_parallel(experiment.llm_list) # Always try to install
    
    # Initialize blender
    print("\n🧪 Initializing LLM Blender...")
    blender = experiment.blender_init()
    if blender is None:
        print(" Failed to initialize blender")
        return
    
    # Prepare references for BERTScore evaluation
    # References are now the ground-truth responses from ConvAI2.
    references = [turn['answer'] for turn in structured_inputs]
    
    print(f"\n📊 Running experiments on {len(structured_inputs)} conversational turns...")
    
    # Re-package inputs for experiment functions.
    # The existing experiment functions expect `inputs` to be `List[List[str]]` (list of conversations, list of questions).
    # We need to pass the full structured turns to them.
    
    # For `experiment_full_ranking`, `experiment_dynamic_elimination`, `experiment_alternate_ranking`,
    # and `experiment_fixed_interval_elimination`:
    # They currently expect `inputs` as `List[List[str]]` where inner list is `questions`.
    # We need to change them to accept `List[Dict]` (individual turns) or `List[List[Dict]]` (conversations of turns).
    # The latter is better for managing context.

    # Let's adjust `dataset_init` to return `List[List[Dict]]` directly,
    # where each inner list is a conversation, and each dict is a turn.
    # This maintains the structure your experiment functions seem to expect for looping.

    # Re-re-thinking `dataset_init` return:
    # Original: `List[List[str]]` (conversations, each a list of questions)
    # New: `List[List[Dict]]` (conversations, each a list of turns where each turn is a dict)

    # Let's modify `dataset_init` to return `List[List[Dict]]` (a list of conversations, where each conversation is a list of turns)
    # This allows `llm_contexts` to accumulate within a conversation across turns.

    # --- REVISED `dataset_init` ---
    # This should be the first modification in your code, before the class.
    # The class init remains the same.

# ======================================================================
# START OF REVISED dataset_init and related functions
# ======================================================================
class LLMBlenderExperiment:
    # ... (rest of your __init__, _setup_directories, install_llms_parallel, setup_ollama_input_json) ...

    def dataset_init(self, filename='convai2_conversations.json', sample_conversations=5): # Changed filename and added sample_conversations
        """
        Initialize dataset by sampling from ConvAI2 (Persona-Chat)
        and saving to file. Returns sampled conversations with full turns.
        
        Returns: List[List[Dict]] where each inner list is a conversation,
                 and each Dict represents a turn: {'context_str', 'question', 'answer_ref', 'full_dialog_history'}
        """
        try:
            print("Loading ConvAI2 (Persona-Chat) dataset...")
            dataset = load_dataset("persona_chat", "original_no_cands")
            
            # The 'train' split of Persona-Chat is a flat list of turns.
            # We need to group them back into conversations if we want to sample full conversations.
            # However, for simplicity and matching the original `inputs` structure (list of lists),
            # let's assume we want to sample *N independent turns* and treat them as short "conversations"
            # for the purpose of your experiment structure.
            # If you want full conversations, Persona-Chat structure makes it harder without
            # manually reconstructing based on 'dialog_id' if available (not standard in original).

            # Let's continue with the assumption that `inputs` is a list of *independent conversational turns*
            # where each "conversation" in your `inputs` list becomes a single turn.
            # This simplifies the loop structure in your experiment functions.

            # Re-read: "inputs is List[List[str]]" means inputs represents a list of *conversations*,
            # where each conversation is a list of *questions*.
            # For ConvAI2, we define a "conversation" as a sequence of (utterance, response) pairs.
            # Each `turn_data` from `dataset['train']` represents one such pair.
            # So, we need to artificially group these into "conversations" or adapt the loops.

            # Simplest adaptation: Treat each sampled `turn_data` as a single-turn "conversation"
            # for the purpose of your experiment functions' outer loop.
            # And then the 'question' inside is the `utterance`. The 'answer' is `labels`.

            data_split = dataset['train']
            
            # Sample a fixed number of *individual turns* from the dataset.
            # Each sampled turn will be treated as a one-turn "conversation" in your framework.
            sample_indices = random.sample(range(len(data_split)), min(sample_conversations, len(data_split)))
            
            sampled_data_for_experiments = [] # This will be List[List[Dict]]
            all_ground_truth_references = [] # To collect all answers for BERTScore

            for i, turn_data in enumerate(sample_indices):
                example = data_split[turn_data] # Get the actual turn data

                persona_context = " ".join(example['persona'])
                history_context = " ".join(example['history'])
                
                # Combine them to form a full context string for the prompt
                # The format of this string can be adjusted.
                context_string_for_prompt = f"Persona: {persona_context.strip()}. Dialogue History: {history_context.strip()}."
                
                question_prompt = example['utterance'].strip()
                ground_truth_response = example['labels'][0].strip() if example['labels'] else ""

                # For ConvAI2, the 'question' to the LLM will be a combination of the context and the utterance.
                # The 'answer' is the ground truth label.

                # We'll put each turn into a list of one turn, to fit the `List[List[str]]` expectation
                # of your `inputs` structure, but now containing structured dicts.
                
                # This dict will represent one 'turn' within a conceptual 'conversation'.
                # The 'question' field here will be the prompt for the LLM.
                # The 'answer_ref' will be for BERTScore.
                turn_dict = {
                    'context_str': context_string_for_prompt, # String context for prompting
                    'question_prompt': question_prompt,       # The actual utterance
                    'answer_ref': ground_truth_response,      # Ground truth answer for evaluation
                    'dialog_id': i # A simple unique ID for this 'conversation'
                }
                
                # Append as a list of one dict (one "conversation" with one "turn")
                sampled_data_for_experiments.append([turn_dict])
                all_ground_truth_references.append(ground_truth_response)
            
            # Save the raw structured data for loading later
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(sampled_data_for_experiments, f, indent=2)
            
            print(f"Saved {len(sampled_data_for_experiments)} 'conversations' (turns) to {filename}")
            return sampled_data_for_experiments, all_ground_truth_references
            
        except Exception as e:
            print(f"Error initializing ConvAI2 dataset: {str(e)}")
            return None, []

    def load_inputs_from_file(self, filename='convai2_conversations.json'): # Changed filename
        """Load previously saved conversation data from file (structured for ConvAI2)."""
        if not os.path.exists(filename):
            print(f"File {filename} not found.")
            return None, []
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                inputs_raw = json.load(f)
            
            # Reconstruct references from the loaded data
            references = []
            for conv in inputs_raw:
                for turn in conv:
                    references.append(turn['answer_ref'])

            print(f"Loaded {len(inputs_raw)} 'conversations' (turns) from {filename}")
            return inputs_raw, references
        except Exception as e:
            print(f"Error loading from {filename}: {str(e)}")
            return None, []

    # ... (rest of the class methods: normalize_llm_name, blender_init, calculate_bertscore) ...

    # Adapt experiment functions to use the new structured input
    def experiment_full_ranking(self, blender, llm_list: List[str], inputs: List[List[Dict]]) -> Tuple[List[str], List[Dict]]:
        """
        Baseline strategy: Full ranking for every turn.
        Inputs are now List[List[Dict]] (list of conversations, each a list of turns dicts)
        """
        print("\n=== Experiment: Full Ranking (Baseline) ===")
        start_time = time.time()
        
        fused_answers = [] # Flat list of all fused answers
        timing_logs = []
        
        for conv_idx, conversation_turns in enumerate(inputs):
            print(f"Processing conversation {conv_idx + 1}/{len(inputs)}")
            
            # For each conversation, initialize Ollama contexts per LLM
            # This holds the *actual Ollama conversation history* for each LLM
            llm_ollama_contexts = {llm: [] for llm in llm_list} 
            
            for turn_idx, turn_data in enumerate(conversation_turns):
                turn_start = time.time()
                
                # Construct the prompt by combining context_str and question_prompt
                # The 'context_str' from `dataset_init` contains persona + history up to this point.
                # So the `question_prompt` is effectively the *new user utterance*.
                current_prompt = f"{turn_data['context_str']}\nUser: {turn_data['question_prompt']}"

                candidates_texts = []
                generation_times = {}
                
                for llm in llm_list:
                    # Pass the accumulated Ollama context for each LLM
                    response, updated_ollama_context, gen_time = self.generate_llm_response(
                        llm, current_prompt, llm_ollama_contexts[llm]
                    )
                    candidates_texts.append(response)
                    llm_ollama_contexts[llm] = updated_ollama_context # Update context for next turn
                    generation_times[llm] = gen_time
                
                # Ranking phase
                rank_start = time.time()
                try:
                    # Note: The prompt for ranker/fuser is still just the `question_prompt`
                    # because the context is handled by the LLMs generating candidates.
                    # If LLMBlender's ranker/fuser also need explicit context, you'd add it here.
                    ranks = blender.rank([turn_data['question_prompt']], [candidates_texts], return_scores=False, batch_size=1)
                    ranking_time = time.time() - rank_start
                except Exception as e:
                    print(f"Ranking error (conv {conv_idx}, turn {turn_idx}): {str(e)}")
                    ranking_time = 0
                    ranks = [[list(range(len(candidates_texts)))]]
                
                topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=min(2, len(candidates_texts)))
                
                # Fusion phase
                fusion_start = time.time()
                try:
                    fuse_generations = blender.fuse([turn_data['question_prompt']], topk_candidates, batch_size=len(topk_candidates[0]))
                    fused_answer = fuse_generations[0]
                    fusion_time = time.time() - fusion_start
                except Exception as e:
                    print(f"Fusion error (conv {conv_idx}, turn {turn_idx}): {str(e)}")
                    fused_answer = candidates_texts[0]
                    fusion_time = 0
                
                fused_answers.append(fused_answer)
                
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
        
        self._save_experiment_results('full_ranking', fused_answers, timing_logs)
        return fused_answers, timing_logs

    # `experiment_dynamic_elimination`, `experiment_alternate_ranking`,
    # `experiment_fixed_interval_elimination` will need similar adjustments:
    # 1. Change `inputs` type hint to `List[List[Dict]]`.
    # 2. Iterate through `conversation_turns` instead of `conversation` (or rename variable).
    # 3. Construct `current_prompt = f"{turn_data['context_str']}\nUser: {turn_data['question_prompt']}"`
    # 4. Pass `llm_ollama_contexts[llm]` to `self.generate_llm_response` and update it.
    # 5. Use `turn_data['question_prompt']` for `blender.rank` and `blender.fuse` as the primary query.
    # 6. Ensure `llm_performance_scores` are tracked per LLM *within* the `active_llms` or `current_llm_committee` set.

    def experiment_dynamic_elimination(self, blender, llm_list: List[str], inputs: List[List[Dict]]) -> Tuple[List[str], List[Dict]]:
        """
        Policy 1: Dynamic conversation-specific elimination.
        Eliminates bottom half of LLMs after half the conversation.
        """
        print("\n=== Experiment: Dynamic Elimination ===")
        start_time = time.time()
        
        fused_answers = []
        timing_logs = []
        
        for conv_idx, conversation_turns in enumerate(inputs):
            print(f"Processing conversation {conv_idx + 1}/{len(inputs)}")
            
            llm_ollama_contexts = {llm: [] for llm in llm_list}
            llm_performance_scores = {llm: 0 for llm in llm_list} # Track score for all possible LLMs
            active_llms = list(llm_list) # Initialize with all LLMs

            elimination_point = math.ceil(len(conversation_turns) / 2) # Changed `conversation` to `conversation_turns`
            
            for turn_idx, turn_data in enumerate(conversation_turns): # Changed `conversation` to `conversation_turns`
                turn_start = time.time()
                
                current_prompt = f"{turn_data['context_str']}\nUser: {turn_data['question_prompt']}"
                
                candidates_texts = []
                generation_times = {}
                
                for llm in active_llms:
                    response, updated_ollama_context, gen_time = self.generate_llm_response(
                        llm, current_prompt, llm_ollama_contexts[llm]
                    )
                    candidates_texts.append(response)
                    llm_ollama_contexts[llm] = updated_ollama_context
                    generation_times[llm] = gen_time
                
                rank_start = time.time()
                try:
                    ranks = blender.rank([turn_data['question_prompt']], [candidates_texts], return_scores=False, batch_size=1)
                    ranking_time = time.time() - rank_start
                    
                    # Update performance scores using the current active LLMs' ranks
                    if turn_idx < elimination_point:
                        # Ranks are given based on the order of candidates_texts, which corresponds to active_llms
                        for i, llm_in_active in enumerate(active_llms):
                            llm_performance_scores[llm_in_active] += ranks[0][i]
                    
                except Exception as e:
                    print(f"Ranking error (conv {conv_idx}, turn {turn_idx}): {str(e)}")
                    ranking_time = 0
                    ranks = [[list(range(len(candidates_texts)))]]
                
                # Dynamic elimination after first half
                if turn_idx == elimination_point - 1 and len(active_llms) > 2: # Trigger at the *end* of the first half turn
                    # Filter scores to only include active LLMs for sorting
                    current_active_scores = {llm: llm_performance_scores[llm] for llm in active_llms}
                    sorted_llms = sorted(current_active_scores.items(), key=lambda x: x[1])
                    keep_count = max(2, len(active_llms) // 2)
                    active_llms = [llm for llm, _ in sorted_llms[:keep_count]]
                    print(f"  Eliminated to {len(active_llms)} LLMs: {active_llms}")
                
                topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=min(2, len(candidates_texts)))
                
                fusion_start = time.time()
                try:
                    fuse_generations = blender.fuse([turn_data['question_prompt']], topk_candidates, batch_size=len(topk_candidates[0]))
                    fused_answer = fuse_generations[0]
                    fusion_time = time.time() - fusion_start
                except Exception as e:
                    print(f"Fusion error (conv {conv_idx}, turn {turn_idx}): {str(e)}")
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
        
        for conv_idx, conversation_turns in enumerate(inputs):
            print(f"Processing conversation {conv_idx + 1}/{len(inputs)}")
            
            llm_ollama_contexts = {llm: [] for llm in llm_list}
            last_ranks = None
            
            for turn_idx, turn_data in enumerate(conversation_turns):
                turn_start = time.time()
                
                current_prompt = f"{turn_data['context_str']}\nUser: {turn_data['question_prompt']}"
                
                candidates_texts = []
                generation_times = {}
                
                for llm in llm_list:
                    response, updated_ollama_context, gen_time = self.generate_llm_response(
                        llm, current_prompt, llm_ollama_contexts[llm]
                    )
                    candidates_texts.append(response)
                    llm_ollama_contexts[llm] = updated_ollama_context
                    generation_times[llm] = gen_time
                
                # Alternate ranking logic
                if turn_idx % 2 == 0:  # Even turns (0, 2, 4...): perform ranking
                    rank_start = time.time()
                    try:
                        ranks = blender.rank([turn_data['question_prompt']], [candidates_texts], return_scores=False, batch_size=1)
                        ranking_time = time.time() - rank_start
                        last_ranks = ranks # Store for reuse
                    except Exception as e:
                        print(f"Ranking error (conv {conv_idx}, turn {turn_idx}): {str(e)}")
                        ranking_time = 0
                        ranks = [[list(range(len(candidates_texts)))]]
                        last_ranks = ranks
                else:  # Odd turns: reuse previous ranking
                    ranks = last_ranks if last_ranks is not None else [[list(range(len(candidates_texts)))]]
                    ranking_time = 0  # No ranking performed on this turn
                
                topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=min(2, len(candidates_texts)))
                
                fusion_start = time.time()
                try:
                    fuse_generations = blender.fuse([turn_data['question_prompt']], topk_candidates, batch_size=len(topk_candidates[0]))
                    fused_answer = fuse_generations[0]
                    fusion_time = time.time() - fusion_start
                except Exception as e:
                    print(f"Fusion error (conv {conv_idx}, turn {turn_idx}): {str(e)}")
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

    def experiment_fixed_interval_elimination(self, blender, llm_list: List[str], inputs: List[List[Dict]]) -> Tuple[List[str], List[Dict]]:
        """
        Policy 3: Fixed-interval elimination - eliminate bottom LLMs every N conversations.
        """
        print("\n=== Experiment: Fixed Interval Elimination ===")
        start_time = time.time()
        
        fused_answers = []
        timing_logs = []
        
        elimination_interval = 3 # Eliminate every 3 conversations
        
        # Global state across conversations
        # This experiment uses a GLOBAL committee that changes after N conversations
        global_llm_performance_scores = {llm: 0 for llm in llm_list} # Accumulate scores over the cycle
        current_llm_committee = list(llm_list) # Start with full committee
        
        for conv_idx, conversation_turns in enumerate(inputs):
            print(f"Processing conversation {conv_idx + 1}/{len(inputs)}")
            
            # Re-initialize contexts for the current committee
            llm_ollama_contexts = {llm: [] for llm in current_llm_committee}
            
            # Check if this is the start of a new elimination cycle
            if (conv_idx % elimination_interval == 0): # For the first conv in a cycle
                current_llm_committee = list(llm_list) # Reset committee for a fresh cycle
                llm_ollama_contexts = {llm: [] for llm in current_llm_committee} # Reset contexts too
                global_llm_performance_scores = {llm: 0 for llm in llm_list} # Reset scores for the new cycle
                print(f"  Starting new elimination cycle (conv {conv_idx+1}) with full committee ({len(current_llm_committee)} LLMs)")

            for turn_idx, turn_data in enumerate(conversation_turns):
                turn_start = time.time()
                
                current_prompt = f"{turn_data['context_str']}\nUser: {turn_data['question_prompt']}"
                
                candidates_texts = []
                generation_times = {}
                
                for llm in current_llm_committee:
                    response, updated_ollama_context, gen_time = self.generate_llm_response(
                        llm, current_prompt, llm_ollama_contexts[llm]
                    )
                    candidates_texts.append(response)
                    llm_ollama_contexts[llm] = updated_ollama_context
                    generation_times[llm] = gen_time
                
                rank_start = time.time()
                try:
                    ranks = blender.rank([turn_data['question_prompt']], [candidates_texts], return_scores=False, batch_size=1)
                    ranking_time = time.time() - rank_start
                    
                    # Accumulate scores for LLMs in the current committee
                    for i, llm_in_committee in enumerate(current_llm_committee):
                        global_llm_performance_scores[llm_in_committee] += ranks[0][i]
                    
                except Exception as e:
                    print(f"Ranking error (conv {conv_idx}, turn {turn_idx}): {str(e)}")
                    ranking_time = 0
                    ranks = [[list(range(len(candidates_texts)))]]
                
                topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=min(2, len(candidates_texts)))
                
                fusion_start = time.time()
                try:
                    fuse_generations = blender.fuse([turn_data['question_prompt']], topk_candidates, batch_size=len(topk_candidates[0]))
                    fused_answer = fuse_generations[0]
                    fusion_time = time.time() - fusion_start
                except Exception as e:
                    print(f"Fusion error (conv {conv_idx}, turn {turn_idx}): {str(e)}")
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
                    'total_turn_time': time.time() - turn_start
                })
            
            # After each conversation, check if it's an elimination point
            if (conv_idx + 1) % elimination_interval == 0 and len(current_llm_committee) > 2:
                # Eliminate based on accumulated scores *within this cycle*
                sorted_llms = sorted(global_llm_performance_scores.items(), key=lambda x: x[1])
                keep_count = max(2, len(current_llm_committee) // 2)
                current_llm_committee = [llm for llm, _ in sorted_llms[:keep_count]]
                print(f"  Eliminated to {len(current_llm_committee)} LLMs for next cycle")
        
        total_time = time.time() - start_time
        print(f"Fixed Interval Elimination completed in {total_time:.2f} seconds")
        
        self._save_experiment_results('fixed_interval_elimination', fused_answers, timing_logs)
        return fused_answers, timing_logs

# ======================================================================
# END OF REVISED dataset_init and related functions
# ======================================================================

def main():
    """Main execution function."""
    print(" Starting LLM Blender Comprehensive Evaluation for ConvAI2")
    print("=" * 60)
    
    experiment = LLMBlenderExperiment()
    experiment.setup_ollama_input_json() # This file is actually unused now for 'messages' API

    # Load or create dataset. Now returns `(conversations_list, references_list)`
    conversations_data, ground_truth_references = experiment.load_inputs_from_file()
    if conversations_data is None:
        print("No existing dataset found, creating new one...")
        conversations_data, ground_truth_references = experiment.dataset_init()
        if conversations_data is None:
            print(" Failed to initialize dataset")
            return
    
    # Check LLM installations
    print("\n Checking LLM installations...")
    experiment.install_llms_parallel(experiment.llm_list)
    
    # Initialize blender
    print("\n Initializing LLM Blender...")
    blender = experiment.blender_init()
    if blender is None:
        print(" Failed to initialize blender")
        return
    
    # `ground_truth_references` is already prepared from `dataset_init` or `load_inputs_from_file`
    print(f"\n Running experiments on {len(conversations_data)} conversations, total {len(ground_truth_references)} turns...")
    
    experiments_to_run = [
        ("Full Ranking", experiment.experiment_full_ranking),
        ("Dynamic Elimination", experiment.experiment_dynamic_elimination),
        ("Alternate Ranking", experiment.experiment_alternate_ranking),
        ("Fixed Interval Elimination", experiment.experiment_fixed_interval_elimination),
    ]
    
    all_results = {}
    
    for name, experiment_func in experiments_to_run:
        fused_answers, timing_logs = experiment_func(blender, experiment.llm_list, conversations_data)
        # Pass the pre-computed ground_truth_references to evaluation
        bertscore_score = experiment.evaluate_with_bertscore(name, fused_answers, ground_truth_references)
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