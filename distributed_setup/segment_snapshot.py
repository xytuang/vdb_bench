#!/usr/bin/env python3
"""
segment_snapshot.py  —  run from node0 before/after each round

Uses two APIs:
  - list_persistent_segments: authoritative row count from datacoord/object storage (no lag)
  - get_query_segment_info:   index breakdown from querynode (may lag after inserts)

Milvus segment states (integer):
    2 = Growing
    3 = Sealed
    4 = Flushed
    5 = Flushing

Usage:
    python3 segment_snapshot.py --round 1 --phase pre
    python3 segment_snapshot.py --round 1 --phase post
"""

import argparse
import os
from datetime import datetime

HOST            = "10.10.1.8"
PORT            = 31053
COLLECTION_NAME = "VDBBench"
LOG_DIR         = "/mydata/vdb_bench/distributed_setup/results/segment_logs"

GROWING  = 2
SEALED   = 3
FLUSHED  = 4
FLUSHING = 5


def snapshot(round_num: int, phase: str):
    from pymilvus import MilvusClient, connections, utility

    os.makedirs(LOG_DIR, exist_ok=True)
    logfile = os.path.join(LOG_DIR, f"segments_round{round_num}_{phase}.log")

    client = MilvusClient(uri=f"http://{HOST}:{PORT}")
    connections.connect("default", host=HOST, port=PORT)
    try:
        persistent = client.list_persistent_segments(COLLECTION_NAME)
        query_segs = utility.get_query_segment_info(COLLECTION_NAME)
    finally:
        connections.disconnect("default")

    # ── Persistent segments (authoritative) ───────────────────────────────────
    p_total_rows = sum(s.num_rows for s in persistent)
    p_by_state   = {}
    for s in persistent:
        p_by_state[s.state_name] = p_by_state.get(s.state_name, 0) + 1

    # ── Query segments (index breakdown, may lag) ─────────────────────────────
    growing   = [s for s in query_segs if s.state == GROWING]
    sealed    = [s for s in query_segs if s.state in (SEALED, FLUSHED, FLUSHING)]
    diskann   = [s for s in sealed if s.index_name == "DISKANN"]
    stl_sort  = [s for s in sealed if s.index_name == "STL_SORT"]
    unindexed = [s for s in sealed if not s.index_name]
    other     = [s for s in sealed if s.index_name and
                 s.index_name not in ("DISKANN", "STL_SORT")]

    q_state_counts = {}
    for s in query_segs:
        q_state_counts[s.state] = q_state_counts.get(s.state, 0) + 1

    q_total_rows    = sum(s.num_rows for s in query_segs)
    diskann_rows    = sum(s.num_rows for s in diskann)
    stl_sort_rows   = sum(s.num_rows for s in stl_sort)
    unindexed_rows  = sum(s.num_rows for s in unindexed)
    other_rows      = sum(s.num_rows for s in other)
    growing_rows    = sum(s.num_rows for s in growing)

    unindexed_pct = (unindexed_rows / q_total_rows * 100) if q_total_rows > 0 else 0.0
    lag_rows      = p_total_rows - q_total_rows

    lines = [
        "",
        f"{'='*65}",
        f"SEGMENT SNAPSHOT — round={round_num}  phase={phase}",
        f"timestamp: {datetime.now().isoformat()}",
        f"{'='*65}",
        "",
        "── PERSISTENT (datacoord / object storage) ─────────────────────",
        f"  Total segments   : {len(persistent):>6}",
        f"  Total rows       : {p_total_rows:>12,}",
        f"  By state         : {p_by_state}",
        "",
        "── QUERYNODE (loaded in memory, may lag) ────────────────────────",
        f"  Total segments   : {len(query_segs):>6}   (states: {q_state_counts})",
        f"  Total rows       : {q_total_rows:>12,}",
        f"  Growing (state=2): {len(growing):>6}   rows: {growing_rows:>12,}",
        f"  Sealed  (state=3): {len(sealed):>6}   rows: {q_total_rows - growing_rows:>12,}",
        f"    └─ DISKANN     : {len(diskann):>6}   rows: {diskann_rows:>12,}",
        f"    └─ STL_SORT    : {len(stl_sort):>6}   rows: {stl_sort_rows:>12,}",
        f"    └─ unindexed   : {len(unindexed):>6}   rows: {unindexed_rows:>12,}",
        f"    └─ other       : {len(other):>6}   rows: {other_rows:>12,}",
        f"  Unindexed %      : {unindexed_pct:>11.1f}%",
        "",
        "── LAG ──────────────────────────────────────────────────────────",
        f"  Persistent - querynode: {lag_rows:>+13,} rows",
        f"  (positive = querynode hasn't loaded all inserts yet)",
        f"{'─'*65}",
        "",
    ]

    if growing:
        lines.append("  Growing segment details:")
        for s in growing:
            lines.append(f"    seg_id={s.segmentID}  rows={s.num_rows:,}  node={s.nodeIds}")

    if sealed:
        lines.append("  Querynode sealed segment details (first 20):")
        for s in sealed[:20]:
            lines.append(f"    seg_id={s.segmentID}  rows={s.num_rows:,}  "
                         f"state={s.state}  index={s.index_name or 'none'}  "
                         f"node={s.nodeIds}")
        if len(sealed) > 20:
            lines.append(f"    ... and {len(sealed) - 20} more")

    output = "\n".join(lines)
    print(output)

    with open(logfile, "w") as f:
        f.write(output)

    print(f"\nSaved to: {logfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--phase", choices=["pre", "post"], required=True)
    args = parser.parse_args()
    snapshot(args.round, args.phase)
