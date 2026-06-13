"""
E27 — CUDA Graphs comparison: parse results and print table.
Also runs a pure conceptual simulation showing where CUDA Graph overhead matters.

Usage (standalone on MacBook — simulation only):
  python e27_cuda_graphs_compare.py

Usage (after running e27_cuda_graphs.sh on RunPod):
  python e27_cuda_graphs_compare.py --results-dir results/e27_cuda_graphs
"""

import argparse
import json
import os
from pathlib import Path


# ── Conceptual simulation ──────────────────────────────────────────────────────

def simulate_cuda_graph_benefit():
    """
    Model the CUDA Graph speedup analytically.

    Each decode step has:
      - Kernel launch overhead: O(num_ops × launch_latency_us)
        With CUDA Graphs: this becomes ONE replay call, O(1)
      - Actual compute time: proportional to batch_size × model_flops

    At small batch sizes, launch overhead is significant.
    At large batch sizes, compute dominates and graphs help less.
    """
    print("E27 — CUDA Graphs: Conceptual Model\n")

    LAUNCH_LATENCY_US = 5.0       # μs per CUDA kernel launch
    NUM_OPS_PER_STEP = 200        # approximate kernel launches per decode step
    COMPUTE_PER_TOKEN_US = 150.0  # compute time per token in batch (μs)
    GRAPH_REPLAY_US = 10.0        # single graph replay call cost

    overhead_eager = NUM_OPS_PER_STEP * LAUNCH_LATENCY_US   # 1000μs = 1ms

    print(f"  Kernel launches per decode step: {NUM_OPS_PER_STEP}")
    print(f"  Launch overhead (eager):  {overhead_eager:.0f}μs per step")
    print(f"  Graph replay overhead:    {GRAPH_REPLAY_US:.0f}μs per step")
    print(f"  Overhead reduction:       {overhead_eager/GRAPH_REPLAY_US:.0f}x\n")

    print(f"  {'Batch size':>12} {'Compute (μs)':>14} {'Eager TPOT (ms)':>16} "
          f"{'Graph TPOT (ms)':>16} {'Speedup':>10}")
    print("  " + "─" * 72)

    results = []
    for batch in [1, 2, 4, 8, 16, 32, 64, 128]:
        compute_us = batch * COMPUTE_PER_TOKEN_US
        eager_tpot_ms = (compute_us + overhead_eager) / 1000
        graph_tpot_ms = (compute_us + GRAPH_REPLAY_US) / 1000
        speedup = eager_tpot_ms / graph_tpot_ms

        print(f"  {batch:>12} {compute_us:>14.0f} {eager_tpot_ms:>16.2f} "
              f"{graph_tpot_ms:>16.2f} {speedup:>9.2f}x")
        results.append({
            "batch_size": batch,
            "eager_tpot_ms": eager_tpot_ms,
            "graph_tpot_ms": graph_tpot_ms,
            "speedup": speedup,
        })

    print()
    print("  Key insight: at batch_size=1, launch overhead is ~87% of step time.")
    print("  CUDA Graphs cut this to near-zero → large TPOT reduction.")
    print("  At batch_size=128, compute dominates → graphs give marginal gain.")
    return results


# ── Parse real vLLM results ────────────────────────────────────────────────────

def parse_vllm_results(results_dir: str):
    p = Path(results_dir)
    if not p.exists():
        print(f"  Results dir not found: {results_dir}")
        return

    # Collect all result files
    on_files  = sorted(p.glob("cuda_graphs_on_rps*.json"))
    off_files = sorted(p.glob("cuda_graphs_off_rps*.json"))

    if not on_files or not off_files:
        print("  No benchmark result files found.")
        return

    def load(f):
        with open(f) as fh:
            return json.load(fh)

    print("\n── Real vLLM Results: CUDA Graphs On vs Off ─────────────────────────")
    print(f"  {'RPS':>5} {'On TPOT p50':>13} {'Off TPOT p50':>13} "
          f"{'On tok/s':>10} {'Off tok/s':>10} {'Speedup':>9}")
    print("  " + "─" * 65)

    for on_f, off_f in zip(on_files, off_files):
        on  = load(on_f)
        off = load(off_f)

        rps = on_f.stem.split("rps")[-1]
        on_tpot  = on.get("mean_tpot_ms", on.get("p50_tpot_ms", 0))
        off_tpot = off.get("mean_tpot_ms", off.get("p50_tpot_ms", 0))
        on_toks  = on.get("output_throughput", 0)
        off_toks = off.get("output_throughput", 0)
        speedup  = off_tpot / on_tpot if on_tpot > 0 else 0

        print(f"  {rps:>5} {on_tpot:>13.1f} {off_tpot:>13.1f} "
              f"{on_toks:>10.0f} {off_toks:>10.0f} {speedup:>8.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None,
                        help="Path to e27 results (RunPod). Omit for simulation only.")
    args = parser.parse_args()

    sim_results = simulate_cuda_graph_benefit()

    # Save simulation results as JSON
    import os
    os.makedirs("results/e27_cuda_graphs", exist_ok=True)
    sim_output = {
        "experiment_id": "E27_cuda_graphs_simulation",
        "description": "Analytical CUDA Graphs speedup model by batch size",
        "model_params": {
            "launch_latency_us": 5.0,
            "num_ops_per_step": 200,
            "compute_per_token_us": 150.0,
            "graph_replay_us": 10.0,
        },
        "batch_size_sweep": sim_results,
    }
    import json
    sim_path = "results/e27_cuda_graphs/E27_simulation.json"
    with open(sim_path, "w") as f:
        json.dump(sim_output, f, indent=2)
    print(f"\nSimulation results saved → {sim_path}")

    if args.results_dir:
        parse_vllm_results(args.results_dir)

    print("\n[Article insight]")
    print("  CUDA Graphs work by recording a CUDA kernel sequence at warmup,")
    print("  then replaying the entire graph with one API call per decode step.")
    print("  This eliminates Python→CUDA dispatch overhead for every kernel launch.")
    print("  Effect is largest at low batch sizes (latency-sensitive workloads).")
    print("  vLLM captures separate graphs for each possible batch size during warmup.")
    print("  --enforce-eager is useful for debugging but hurts production throughput.")
    print("  Startup time increase = graph capture time (scales with num_batch_sizes).")


if __name__ == "__main__":
    main()
