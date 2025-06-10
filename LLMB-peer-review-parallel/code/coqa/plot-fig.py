import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Union

def load_timing_logs(timing_file: str) -> List[Dict]:
    """Loads timing logs from a JSON file."""
    try:
        with open(timing_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load timing logs from {timing_file}. {e}")
        return []

def analyze_timing(timing_logs: List[Dict]) -> Dict:
    """Analyzes timing logs to calculate summary statistics."""
    if not timing_logs:
        return {
            'total_time': 0.0,
            'avg_time': 0.0,
            'avg_ranking_time': 0.0,
            'avg_fusion_time': 0.0,
            'turns': 0
        }

    total_time = sum(log.get('total_turn_time', 0.0) for log in timing_logs)
    avg_time = total_time / len(timing_logs)
    ranking_times = [log.get('ranking_time', 0.0) for log in timing_logs]
    fusion_times = [log.get('fusion_time', 0.0) for log in timing_logs]
    
    avg_ranking_time = sum(ranking_times) / len(ranking_times) if ranking_times else 0.0
    avg_fusion_time = sum(fusion_times) / len(fusion_times) if fusion_times else 0.0
    
    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'avg_ranking_time': avg_ranking_time,
        'avg_fusion_time': avg_fusion_time,
        'turns': len(timing_logs)
    }

def plot_times(timing_logs: List[Dict], strategy_name: str):
    """Plots the total turn time for each turn."""
    times = [log.get('total_turn_time', 0.0) for log in timing_logs]
    plt.figure(figsize=(10, 5))
    plt.plot(times, marker='o')
    plt.title(f'Total Turn Time per Turn - {strategy_name}')
    plt.xlabel('Turn')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def export_csv(timing_logs: List[Dict], csv_file: str):
    """Exports timing logs to a CSV file."""
    try:
        df = pd.DataFrame(timing_logs)
        df.to_csv(csv_file, index=False)
        print(f"Exported timing logs to {csv_file}")
    except Exception as e:
        print(f"Error exporting timing logs to {csv_file}: {e}")

def load_bertscore(bertscore_file: str) -> Union[List[float], float, None]:
    """Loads BERTScore data from a file."""
    try:
        with open(bertscore_file, 'r', encoding='utf-8') as f:
            data = f.read()
            try:
                # Try to parse as JSON list
                scores = json.loads(data)
                if isinstance(scores, list):
                    return scores
            except json.JSONDecodeError:
                # Try to parse as single value in text
                for line in data.splitlines():
                    if "Average BERTScore F1" in line:
                        try:
                            return float(line.split(":")[-1].strip())
                        except (IndexError, ValueError):
                            continue
    except Exception as e:
        print(f"Could not load BERTScore from {bertscore_file}: {e}")
    return None

def plot_bertscore(scores: Union[List[float], float, None], strategy_name: str):
    """Plots BERTScore scores per turn or prints the average score."""
    if isinstance(scores, list):
        plt.figure(figsize=(10, 5))
        plt.plot(scores, marker='o')
        plt.title(f'BERTScore F1 per Turn - {strategy_name}')
        plt.xlabel('Turn')
        plt.ylabel('BERTScore F1')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif isinstance(scores, float):
        print(f"Average BERTScore F1 for {strategy_name}: {scores:.4f}")
    elif scores is None:
        print(f"No BERTScore data found for {strategy_name}")

def main():
    """Main function to analyze experiment results."""
    
    # Set your results directory here
    results_dir = "Experiment_Results"
    strategies = [
        "full_ranking",
        "dynamic_elimination",
        "alternate_ranking",
        "fixed_interval_elimination"
    ]

    for strategy in strategies:
        print(f"\n=== Analysis for {strategy} ===")
        timing_file = os.path.join(results_dir, f"timing_{strategy}.json")
        bertscore_file = os.path.join(results_dir, f"bertscore_{strategy}.txt")
        csv_file = os.path.join(results_dir, f"timing_{strategy}.csv")

        if not os.path.exists(timing_file):
            print(f"Timing log not found: {timing_file}")
            continue

        timing_logs = load_timing_logs(timing_file)
        timing_stats = analyze_timing(timing_logs)
        print(f"Total time: {timing_stats['total_time']:.2f} s")
        print(f"Average time per turn: {timing_stats['avg_time']:.2f} s")
        print(f"Average ranking time: {timing_stats['avg_ranking_time']:.2f} s")
        print(f"Average fusion time: {timing_stats['avg_fusion_time']:.2f} s")
        print(f"Total turns: {timing_stats['turns']}")

        plot_times(timing_logs, strategy)
        export_csv(timing_logs, csv_file)

        scores = load_bertscore(bertscore_file)
        plot_bertscore(scores, strategy)

if __name__ == "__main__":
    main()