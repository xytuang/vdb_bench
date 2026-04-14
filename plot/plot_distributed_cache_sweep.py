import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re


RESULTS_DIR = "../distributed_setup/results/new_disk_cache_sweep"


def parse_cache_gb_from_filename(filename):
    """Extract cache size in GB from file names"""
    match = re.match(r'^disk_(\d+).json', filename)
    if match:
        return float(match.group(1))
    return None


def get_metrics(data):
    """
    Extract metrics from a cache sweep result file.
    These are serial (non-concurrent) runs, so we use qps and serial latency fields.
    """
    metrics = data["results"][0]["metrics"]
    return {
        "qps":         metrics["qps"],
        "latency_p99": metrics["conc_latency_p99_list"][0],
        "latency_p95": metrics["conc_latency_p95_list"][0],
    }


def load_all_cache_levels(base_dir):
    """
    Load all disk_X files under base_dir.
    Returns a dict: {cache_gb: [metrics_dict, ...]} where the list contains one
    entry per run (r1.json, r2.json, ...).
    """
    cache_data = {}

    if not os.path.exists(base_dir):
        print(f"Error: directory '{base_dir}' does not exist.")
        return cache_data

    for entry in sorted(os.listdir(base_dir)):
        cache_gb = parse_cache_gb_from_filename(entry)
        if cache_gb is None:
            continue

        print(f"Loading cache level: {entry}")

        file_path = os.path.join(base_dir, entry)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            cache_data[cache_gb] = [get_metrics(data)]

            print(f"  Loaded: {file_path}")
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")


    return cache_data


def average_runs(runs, key):
    return np.mean([r[key] for r in runs])


def plot_results():
    cache_data = load_all_cache_levels(RESULTS_DIR)

    if not cache_data:
        print("No data loaded. Exiting.")
        return

    cache_sizes = sorted(cache_data.keys())
    print(f"\nCache sizes found: {cache_sizes} %")

    x = np.array(cache_sizes)

    avg_qps = np.array([average_runs(cache_data[c], "qps")         for c in cache_sizes])
    avg_p99 = np.array([average_runs(cache_data[c], "latency_p99") for c in cache_sizes])
    avg_p95 = np.array([average_runs(cache_data[c], "latency_p95") for c in cache_sizes])

    # Per-run arrays for min/max shading
    all_qps = np.array([[r["qps"]         for r in cache_data[c]] for c in cache_sizes])
    all_p99 = np.array([[r["latency_p99"] for r in cache_data[c]] for c in cache_sizes])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("spacev1b50m — Disk High Watermark Sweep", fontsize=13, fontweight='bold')

    # ── QPS ──────────────────────────────────────────────────────────────
    ax = axes[0]
    max_runs = max(len(cache_data[c]) for c in cache_sizes)
    if max_runs > 1:
        ax.fill_between(x, all_qps.min(axis=1), all_qps.max(axis=1),
                        alpha=0.15, color='steelblue', label='min/max range')
    ax.plot(x, avg_qps, marker='o', color='steelblue', linewidth=2, label='avg QPS')
    for xi, yi in zip(x, avg_qps):
        ax.annotate(f'{yi:.2f}', (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}' for c in cache_sizes])
    ax.set_xlabel('Disk High Watermark', fontsize=11)
    ax.set_ylabel('Queries per Second (QPS)', fontsize=11)
    ax.set_title('Throughput vs Disk High Watermark', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Latency ───────────────────────────────────────────────────────────
    ax = axes[1]
    if max_runs > 1:
        ax.fill_between(x, all_p99.min(axis=1) * 1000, all_p99.max(axis=1) * 1000,
                        alpha=0.15, color='tomato', label='p99 min/max range')
    ax.plot(x, avg_p99 * 1000, marker='s', color='tomato',
            linewidth=2, label='p99')
    ax.plot(x, avg_p95 * 1000, marker='^', color='darkorange',
            linewidth=2, linestyle='--', label='p95')

    for xi, yi in zip(x, avg_p99 * 1000):
        ax.annotate(f'{yi:.2f}', (xi, yi),
                    textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=8)

    for xi, yi in zip(x, avg_p95 * 1000):
        ax.annotate(f'{yi:.2f}', (xi, yi),
                    textcoords="offset points", xytext=(0, -14),
                    ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}' for c in cache_sizes])
    ax.set_xlabel('Disk High Watermark', fontsize=11)
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Latency vs Disk High Watermark', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "new_disk_cache_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_results()