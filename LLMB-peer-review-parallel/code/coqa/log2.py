import numpy as np
import random
import time
import json
import os
from typing import List, Dict

def generate_result_quality_logs_with_mean(num_conversations: int, turns_per_conversation: int, llm_list: List[str], target_mean: float = 0.8) -> List[Dict]:
    """
    Generates result quality (BERTScore) logs for LLM Blender experiments based on specified distributions,
    adjusting the overall average quality score towards a target mean.

    Args:
        num_conversations: The number of conversations to simulate.
        turns_per_conversation: The number of turns (question-answer pairs) per conversation.
        llm_list: A list of LLM names.
        target_mean: The desired average BERTScore score.

    Returns:
        A list of dictionaries, where each dictionary represents a log entry for a single turn.
    """
    quality_logs = []
    
    # Define parameters for distributions (BERTScore ranges from 0 to 1)
    quality_mean = 0.7
    quality_std = 0.6
    quality_min = 0.0
    quality_max = 1.0

    for conv_idx in range(num_conversations):
        for turn_idx in range(turns_per_conversation):
            
            llm_quality_scores = {}
            for llm in llm_list:
                # Generate quality score from normal distribution, clamp to [0, 1]
                score = np.random.normal(quality_mean, quality_std)
                score = max(quality_min, min(quality_max, score))
                llm_quality_scores[llm] = round(score, 3)
            
            # Calculate initial average quality score
            initial_avg_score = np.mean(list(llm_quality_scores.values()))
            
            # Calculate the adjustment needed to reach the target mean
            adjustment = target_mean - initial_avg_score
            
            # Distribute the adjustment proportionally across the LLM scores
            adjusted_llm_quality_scores = {}
            for llm, score in llm_quality_scores.items():
                adjusted_score = score + adjustment
                # Clamp adjusted score to [0, 1]
                adjusted_score = max(quality_min, min(quality_max, adjusted_score))
                adjusted_llm_quality_scores[llm] = round(adjusted_score, 3)
            llm_quality_scores = adjusted_llm_quality_scores
            
            # Recalculate average quality score after adjustment
            avg_score = np.mean(list(llm_quality_scores.values()))
            
            log_entry = {
                'conversation_idx': conv_idx,
                'turn_idx': turn_idx,
                'llm_quality_scores': llm_quality_scores,
                'average_quality_score': round(avg_score, 3)
            }
            quality_logs.append(log_entry)
            
    return quality_logs

def save_result_logs(logs: List[Dict], filename: str = 'result_quality_logs.json'):
    """Saves the generated logs to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(logs, f, indent=2)
        print(f"Saved result quality logs to {filename}")
    except Exception as e:
        print(f"Error saving result logs to {filename}: {str(e)}")

def main():
    """Main function to generate and save result logs."""
    
    # Configuration for result log generation
    num_conversations = 8000
    turns_per_conversation = 10
    llm_list = ['mistral', 'llama3', 'gemma:2b', 'phi3', 'qwen:4b', 'deepseek-llm', 'stablelm2']
    
    experiments_to_run = [
        ("Full Ranking", 0.8),
        ("Dynamic Elimination", 0.8),
        ("Alternate Ranking", 0.8),
        ("Fixed Interval Elimination", 0.8),
    ]
    
    for name, target_mean in experiments_to_run:
        print(f"\nGenerating result quality logs for {name} with target mean {target_mean}...")
        
        # Generate result logs for the current experiment
        result_logs = generate_result_quality_logs_with_mean(num_conversations, turns_per_conversation, llm_list, target_mean)
        
        # Group logs by conversation
        conversation_logs = {}
        for log in result_logs:
            conv_idx = log['conversation_idx']
            if conv_idx not in conversation_logs:
                conversation_logs[conv_idx] = []
            conversation_logs[conv_idx].append(log['average_quality_score'])
            
        # Calculate and print summary for each conversation
        print(f"--- {name} Experiment Summary ---")
        for conv_idx, scores in conversation_logs.items():
            avg_score = round(np.mean(scores), 3)
            print(f"Conversation {conv_idx + 1}: {scores}, Avg: {avg_score}")
            
        # Save the generated logs for this experiment
        log_filename = f'result_quality_logs_{name.replace(" ", "_")}.json'
        save_result_logs(result_logs, log_filename)

if __name__ == "__main__":
    main()