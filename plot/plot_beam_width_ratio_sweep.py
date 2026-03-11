import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re


RESULTS_DIR = "../results/spacev1b50m/beam_width_ratio_sweep"


def parse_ratio_from_dirname(dirname):
    """Extract ratio size in GB from directory names like ratio_2, ratio_4."""
    match = re.match(r'^ratio_(\d+)$', dirname)
    if match:
        return int(match.group(1))
    return None


def get_metrics(data):
    """
    Extract metrics from a beam width ratio sweep result file.
    These are concurrent runs, so we use qps and concurrent latency fields.
    """
    metrics = data["results"][0]["metrics"]
    return {
        "qps":         metrics["qps"],
        "latency_p99": metrics["conc_latency_p99_list"][0],
        "latency_p95": metrics["conc_latency_p95_list"][0],
    }


def load_all_ratio_levels(base_dir):
    """
    Load all ratio_x subdirectories under base_dir.
    Returns a dict: {ratio: [metrics_dict, ...]} where the list contains one
    entry per run (r1.json, r2.json, ...).
    """
    ratio_data = {}

    if not os.path.exists(base_dir):
        print(f"Error: directory '{base_dir}' does not exist.")
        return ratio_data

    for entry in sorted(os.listdir(base_dir)):
        ratio = parse_ratio_from_dirname(entry)
        if ratio is None:
            continue

        ratio_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(ratio_dir):
            continue

        print(f"Loading ratio level: {entry}")
        runs = []
        for filename in sorted(os.listdir(ratio_dir)):
            if not re.match(r'^r\d+\.json$', filename):
                continue
            file_path = os.path.join(ratio_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                runs.append(get_metrics(data))
                print(f"  Loaded: {file_path}")
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")

        if runs:
            ratio_data[ratio] = runs

    return ratio_data


def average_runs(runs, key):
    return np.mean([r[key] for r in runs])


def plot_results():
    ratio_data = load_all_ratio_levels(RESULTS_DIR)

    if not ratio_data:
        print("No data loaded. Exiting.")
        return

    ratio_sizes = sorted(ratio_data.keys())
    print(f"\nratio sizes found: {ratio_sizes} GB")

    x = np.array(ratio_sizes)

    avg_qps = np.array([average_runs(ratio_data[c], "qps")         for c in ratio_sizes])
    avg_p99 = np.array([average_runs(ratio_data[c], "latency_p99") for c in ratio_sizes])
    avg_p95 = np.array([average_runs(ratio_data[c], "latency_p95") for c in ratio_sizes])

    # Per-run arrays for min/max shading
    all_qps = np.array([[r["qps"]         for r in ratio_data[c]] for c in ratio_sizes])
    all_p99 = np.array([[r["latency_p99"] for r in ratio_data[c]] for c in ratio_sizes])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("spacev1b50m — Beam Width Ratio Sweep", fontsize=13, fontweight='bold')

    # ── QPS ──────────────────────────────────────────────────────────────
    ax = axes[0]
    max_runs = max(len(ratio_data[c]) for c in ratio_sizes)
    if max_runs > 1:
        ax.fill_between(x, all_qps.min(axis=1), all_qps.max(axis=1),
                        alpha=0.15, color='steelblue', label='min/max range')
    ax.plot(x, avg_qps, marker='o', color='steelblue', linewidth=2, label='avg QPS')
    for xi, yi in zip(x, avg_qps):
        ax.annotate(f'{yi:.0f}', (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}' for c in ratio_sizes])
    ax.set_xlabel('Beam Width Ratio', fontsize=11)
    ax.set_ylabel('Queries per Second (QPS)', fontsize=11)
    ax.set_title('Throughput vs Beam Width Ratio', fontsize=12)
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
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}' for c in ratio_sizes])
    ax.set_xlabel('Beam Width Ratio', fontsize=11)
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Latency vs Beam Width Ratio', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "ratio_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_results()