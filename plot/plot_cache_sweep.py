import json
import matplotlib.pyplot as plt
import numpy as np
import os


def get_metrics(data):
    """Extract concurrency sweep metrics from result JSON"""
    metrics = data["results"][0]["metrics"]
    return {
        "conc_num_list": metrics["conc_num_list"],
        "conc_qps_list": metrics["conc_qps_list"],
        "conc_latency_p99_list": metrics["conc_latency_p99_list"],
        "conc_latency_p95_list": metrics["conc_latency_p95_list"],
        "conc_latency_avg_list": metrics["conc_latency_avg_list"],
    }


def load_directory(directory):
    """Load all JSON result files from a directory"""
    results = []
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return results

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith('.json'):
            continue
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r') as f:
                    curr_data = json.load(f)
                    results.append(get_metrics(curr_data))
                    print(f"Loaded: {file_path}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return results


def average_sweep(runs, key):
    """Average a per-concurrency list across multiple runs."""
    # All runs should share the same conc_num_list
    n = len(runs[0][key])
    averaged = []
    for i in range(n):
        averaged.append(np.mean([run[key][i] for run in runs]))
    return averaged


def plot_results():
    runs = load_directory("./")

    if not runs:
        print("No data loaded. Exiting.")
        return

    print(f"\nLoaded {len(runs)} run(s).")

    # Validate all runs share the same concurrency levels
    conc_nums = runs[0]["conc_num_list"]
    for i, run in enumerate(runs):
        if run["conc_num_list"] != conc_nums:
            print(f"Warning: run {i} has different conc_num_list: {run['conc_num_list']}")

    avg_qps    = average_sweep(runs, "conc_qps_list")
    avg_p99    = average_sweep(runs, "conc_latency_p99_list")
    avg_p95    = average_sweep(runs, "conc_latency_p95_list")
    avg_lat    = average_sweep(runs, "conc_latency_avg_list")

    # Per-run arrays for shading
    all_qps = np.array([r["conc_qps_list"] for r in runs])
    all_p99 = np.array([r["conc_latency_p99_list"] for r in runs])

    x = np.array(conc_nums)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── QPS ──────────────────────────────────────────────────────────────
    ax = axes[0]
    # ax.fill_between(x, all_qps.min(axis=0), all_qps.max(axis=0), alpha=0.15, color='steelblue', label='min/max range')
    ax.plot(x, avg_qps, marker='o', color='steelblue', linewidth=2, label='avg QPS')
    for xi, yi in zip(x, avg_qps):
        ax.annotate(f'{yi:.0f}', (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=8)
    ax.set_xscale('log', base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x])
    ax.set_xlabel('Number of Concurrent Threads', fontsize=11)
    ax.set_ylabel('Queries per Second (QPS)', fontsize=11)
    ax.set_title('Throughput vs Concurrency', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Latency ───────────────────────────────────────────────────────────
    ax = axes[1]
    # ax.fill_between(x, all_p99.min(axis=0) * 1000, all_p99.max(axis=0) * 1000, alpha=0.15, color='tomato', label='p99 min/max range')
    ax.plot(x, [v * 1000 for v in avg_p99], marker='s', color='tomato',
            linewidth=2, label='p99')
    ax.plot(x, [v * 1000 for v in avg_p95], marker='^', color='darkorange',
            linewidth=2, linestyle='--', label='p95')
    ax.plot(x, [v * 1000 for v in avg_lat], marker='o', color='goldenrod',
            linewidth=2, linestyle=':', label='avg')
    ax.set_xscale('log', base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x])
    ax.set_xlabel('Number of Concurrent Threads', fontsize=11)
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Latency vs Concurrency', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "concurrency_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_results()