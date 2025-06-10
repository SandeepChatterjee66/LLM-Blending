import json

def analyze_bert_and_time(bert_scores_file, timing_logs_file):
    """
    Analyze a list of BERT scores and timing logs.
    Prints the average BERT score, total time, and average time per turn.
    """
    # Load BERT scores (should be a list of floats)
    with open(bert_scores_file, 'r', encoding='utf-8') as f:
        bert_scores = json.load(f)
    if not bert_scores:
        print("No BERT scores found.")
        return

    avg_bert = sum(bert_scores) / len(bert_scores)
    print(f"Average BERTScore: {avg_bert:.4f} (from {len(bert_scores)} scores)")

    # Load timing logs (should be a list of dicts with 'total_turn_time' key)
    with open(timing_logs_file, 'r', encoding='utf-8') as f:
        timing_logs = json.load(f)
    total_time = sum(log.get('total_turn_time', 0) for log in timing_logs)
    avg_time = total_time / len(timing_logs) if timing_logs else 0

    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per turn: {avg_time:.2f} seconds (from {len(timing_logs)} turns)")

if __name__ == "__main__":
    # Example usage:
    # Place your bert_scores.json and timing_logs.json in the same directory.
    bert_scores_file = "bert_scores.json"         # List of floats, e.g. [0.85, 0.87, ...]
    timing_logs_file = "timing_logs.json"         # List of dicts with 'total_turn_time' key

    analyze_bert_and_time(bert_scores_file, timing_logs_file)