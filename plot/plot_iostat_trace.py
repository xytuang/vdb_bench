import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

RESULTS_DIR = "../distributed_setup/results/iostat_output"
DEVICE = "nvme0n1"
METRICS = ["kB_read/s", "kB_wrtn/s", "tps"]
METRIC_LABELS = {
    "kB_read/s": "Read (kB/s)",
    "kB_wrtn/s": "Write (kB/s)",
    "tps":       "TPS",
}
INTERVAL_S = 1


def parse_size_from_filename(filename):
    match = re.match(r'^disk_iostat_(\d+)\.json$', filename)
    if match:
        return int(match.group(1))
    return None


def load_iostat(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    statistics = data["sysstat"]["hosts"][0]["statistics"]
    result = {m: [] for m in METRICS}
    for snapshot in statistics:
        device = next((d for d in snapshot["disk"] if d["disk_device"] == DEVICE), None)
        if device is None:
            continue
        for m in METRICS:
            result[m].append(device[m])
    return result


def plot_results():
    entries = {}
    for filename in sorted(os.listdir(RESULTS_DIR)):
        size = parse_size_from_filename(filename)
        if size is None:
            continue
        filepath = os.path.join(RESULTS_DIR, filename)
        try:
            entries[size] = load_iostat(filepath)
            print(f"  Loaded: {filepath}")
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")

    if not entries:
        print("No data loaded. Exiting.")
        return

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(14, 4 * len(METRICS)), sharex=False)
    fig.suptitle(f"iostat — {DEVICE} — Disk Cache Sweep", fontsize=13, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(entries)))

    for ax, metric in zip(axes, METRICS):
        for (size, data), color in zip(sorted(entries.items()), colors):
            y = np.array(data[metric])
            x = np.arange(len(y)) * INTERVAL_S
            ax.plot(x, y, linewidth=1.2, label=f'disk_{size}', color=color)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_xlabel('Elapsed Time (s)', fontsize=10)
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "iostat_disk_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_results()