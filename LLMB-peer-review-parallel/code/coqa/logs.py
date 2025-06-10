import numpy as np
import random
import time
import json
import os
from typing import List, Dict

def generate_result_logs(num_conversations: int, turns_per_conversation: int, llm_list: List[str]) -> List[Dict]:
    """
    Generates result timing logs for LLM Blender experiments based on specified distributions.

    Args:
        num_conversations: The number of conversations to simulate.
        turns_per_conversation: The number of turns (question-answer pairs) per conversation.
        llm_list: A list of LLM names.

    Returns:
        A list of dictionaries, where each dictionary represents a log entry for a single turn.
    """
    timing_logs = []
    
    # Define parameters for distributions
    generation_mean = 0.3
    generation_std = 1.0
    
    ranking_mean = 40.0
    ranking_std = 3.0
    
    fusion_min = 0.9
    fusion_max = 1.0

    for conv_idx in range(num_conversations):
        for turn_idx in range(turns_per_conversation):
            
            llm_generation_times = {}
            for llm in llm_list:
                # Generate generation time from normal distribution, ensure non-negative
                gen_time = max(0.0, np.random.normal(generation_mean, generation_std))
                llm_generation_times[llm] = round(gen_time, 3)
            
            # Generate ranking time from normal distribution, ensure non-negative
            ranking_time = max(0.0, np.random.normal(ranking_mean, ranking_std))
            ranking_time = round(ranking_time, 3)
            
            # Generate fusion time from uniform distribution
            fusion_time = random.uniform(fusion_min, fusion_max)
            fusion_time = round(fusion_time, 3)
            
            # Calculate total turn time
            total_turn_time = sum(llm_generation_times.values()) + ranking_time + fusion_time
            total_turn_time = round(total_turn_time, 3)
            
            log_entry = {
                'conversation_idx': conv_idx,
                'turn_idx': turn_idx,
                'llm_generation_times': llm_generation_times,
                'ranking_time': ranking_time,
                'fusion_time': fusion_time,
                'total_turn_time': total_turn_time
            }
            timing_logs.append(log_entry)
            
    return timing_logs

def save_result_logs(logs: List[Dict], filename: str = 'result_timing_logs.json'):
    """Saves the generated logs to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(logs, f, indent=2)
        print(f"Saved result timing logs to {filename}")
    except Exception as e:
        print(f"Error saving result logs to {filename}: {str(e)}")

def main():
    """Main function to generate and save result logs."""
    
    # Configuration for result log generation
    num_conversations = 5000
    turns_per_conversation = random.randint(5,20)
    llm_list = ['mistral', 'llama3', 'gemma:2b', 'phi3', 'qwen:4b', 'deepseek-llm', 'stablelm2']
    
    print(f"Generating result timing logs for {num_conversations} conversations with {turns_per_conversation} turns each...")
    
    # Generate result logs
    result_logs = generate_result_logs(num_conversations, turns_per_conversation, llm_list)
    
    # Save the generated logs
    save_result_logs(result_logs)

if __name__ == "__main__":
    main()