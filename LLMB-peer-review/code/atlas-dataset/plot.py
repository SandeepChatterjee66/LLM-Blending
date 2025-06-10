import json
import os
import matplotlib.pyplot as plt
import pandas as pd

def load_timing_logs(timing_file):
    with open(timing_file, 'r', encoding='utf-8') as f:
        timing_logs = json.load(f)
    return timing_logs

def analyze_timing(timing_logs):
    total_time = sum(log.get('total_turn_time', 0) for log in timing_logs)
    avg_time = total_time / len(timing_logs) if timing_logs else 0
    ranking_times = [log.get('ranking_time', 0) for log in timing_logs]
    fusion_times = [log.get('fusion_time', 0) for log in timing_logs]
    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'avg_ranking_time': sum(ranking_times) / len(ranking_times) if ranking_times else 0,
        'avg_fusion_time': sum(fusion_times) / len(fusion_times) if fusion_times else 0,
        'turns': len(timing_logs)
    }

def plot_times(timing_logs, strategy_name):
    times = [log.get('total_turn_time', 0) for log in timing_logs]
    plt.figure(figsize=(10, 5))
    plt.plot(times, marker='o')
    plt.title(f'Total Turn Time per Turn - {strategy_name}')
    plt.xlabel('Turn')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def export_csv(timing_logs, csv_file):
    df = pd.DataFrame(timing_logs)
    df.to_csv(csv_file, index=False)
    print(f"Exported timing logs to {csv_file}")

def load_bertscore(bertscore_file):
    # If the file contains a list of scores
    try:
        with open(bertscore_file, 'r', encoding='utf-8') as f:
            data = f.read()
            try:
                # Try to parse as JSON list
                scores = json.loads(data)
                if isinstance(scores, list):
                    return scores
            except Exception:
                # Try to parse as single value in text
                for line in data.splitlines():
                    if "Average BERTScore F1" in line:
                        return float(line.split(":")[-1].strip())
    except Exception as e:
        print(f"Could not load BERTScore: {e}")
    return None

def plot_bertscore(scores, strategy_name):
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

def main():
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

        if os.path.exists(bertscore_file):
            scores = load_bertscore(bertscore_file)
            plot_bertscore(scores, strategy)
        else:
            print(f"BERTScore file not found: {bertscore_file}")

if __name__ == "__main__":
    main()