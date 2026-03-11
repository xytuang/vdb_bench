import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re


RESULTS_DIR = "../results/spacev1b50m/concurrency_sweep"


def parse_concurrency_from_filename(filename):
    """Extract concurrency number from filenames like c1.json, c128.json."""
    match = re.match(r'^c(\d+)\.json$', filename)
    if match:
        return int(match.group(1))
    return None


def get_scalar_metrics(data):
    """
    Extract metrics from a single-concurrency result file.
    Each conc_*_list contains exactly one element matching the single
    concurrency level run in this file.
    """
    metrics = data["results"][0]["metrics"]
    return {
        "qps":         metrics["conc_qps_list"][0],
        "latency_p99": metrics["conc_latency_p99_list"][0],
        "latency_p95": metrics["conc_latency_p95_list"][0],
        "latency_avg": metrics["conc_latency_avg_list"][0],
    }


def load_run_directory(run_dir):
    """
    Load all c*.json files from a single run directory.
    Returns a dict: {concurrency_int: metrics_dict}
    """
    run_data = {}
    for filename in sorted(os.listdir(run_dir)):
        conc = parse_concurrency_from_filename(filename)
        if conc is None:
            continue  # skip serial_search.json and anything else
        file_path = os.path.join(run_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            conc_in_file = data["results"][0]["metrics"]["conc_num_list"][0]
            if conc_in_file != conc:
                print(f"  Warning: {filename} has conc_num_list=[{conc_in_file}] but filename implies {conc}")
            run_data[conc] = get_scalar_metrics(data)
            print(f"  Loaded: {file_path}  (concurrency={conc})")
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
    return run_data


def load_all_runs(base_dir):
    """
    Load all r* subdirectories under base_dir.
    Returns a list of dicts, each {concurrency: metrics}.
    """
    runs = []
    if not os.path.exists(base_dir):
        print(f"Error: directory '{base_dir}' does not exist.")
        return runs

    for entry in sorted(os.listdir(base_dir)):
        run_path = os.path.join(base_dir, entry)
        if os.path.isdir(run_path) and re.match(r'^r\d+$', entry):
            print(f"Loading run: {run_path}")
            run_data = load_run_directory(run_path)
            if run_data:
                runs.append(run_data)
    return runs


def average_across_runs(runs, conc_nums, key):
    """Average a metric across all runs for each concurrency level."""
    averaged = []
    for c in conc_nums:
        values = [run[c][key] for run in runs if c in run]
        averaged.append(np.mean(values) if values else np.nan)
    return np.array(averaged)


def plot_results():
    runs = load_all_runs(RESULTS_DIR)

    if not runs:
        print("No data loaded. Exiting.")
        return

    print(f"\nLoaded {len(runs)} run(s).")

    # Union of all concurrency levels seen, sorted
    all_conc = sorted(set(c for run in runs for c in run.keys()))
    print(f"Concurrency levels: {all_conc}")

    x = np.array(all_conc)

    avg_qps = average_across_runs(runs, all_conc, "qps")
    avg_p99 = average_across_runs(runs, all_conc, "latency_p99")
    avg_p95 = average_across_runs(runs, all_conc, "latency_p95")
    avg_lat = average_across_runs(runs, all_conc, "latency_avg")

    # Per-run arrays for optional shading
    all_qps = np.array([[run.get(c, {}).get("qps", np.nan) for c in all_conc] for run in runs])
    all_p99 = np.array([[run.get(c, {}).get("latency_p99", np.nan) for c in all_conc] for run in runs])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("spacev1b50m — Concurrency Sweep", fontsize=13, fontweight='bold')

    # ── QPS ──────────────────────────────────────────────────────────────
    ax = axes[0]
    if len(runs) > 1:
        ax.fill_between(x, np.nanmin(all_qps, axis=0), np.nanmax(all_qps, axis=0),
                        alpha=0.15, color='steelblue', label='min/max range')
    ax.plot(x, avg_qps, marker='o', color='steelblue', linewidth=2, label='avg QPS')
    for xi, yi in zip(x, avg_qps):
        if not np.isnan(yi):
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
    if len(runs) > 1:
        ax.fill_between(x, np.nanmin(all_p99, axis=0) * 1000, np.nanmax(all_p99, axis=0) * 1000,
                        alpha=0.15, color='tomato', label='p99 min/max range')
    ax.plot(x, avg_p99 * 1000, marker='s', color='tomato',
            linewidth=2, label='p99')
    ax.plot(x, avg_p95 * 1000, marker='^', color='darkorange',
            linewidth=2, linestyle='--', label='p95')
    ax.plot(x, avg_lat * 1000, marker='o', color='goldenrod',
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