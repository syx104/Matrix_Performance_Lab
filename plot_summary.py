
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    results_dir = Path("results")
    input_file = results_dir / "benchmark_summary.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            "Run analyze_runs.py first"
        )

    data = defaultdict(list)

    with input_file.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            method = row["method"]
            threads = int(row["threads"])

            if method == "PyTorch":
                label = f"PyTorch ({threads} threads)"
            else:
                label = method

            data[label].append({
                "size": int(row["size"]),
                "mean": float(row["mean_latency_ms"]),
                "std": float(row["std_latency_ms"]),
                "gflops": float(
                    row["gflops_from_mean_latency"]
                ),
            })

    # Figure 1: Mean latency with standard deviation
    plt.figure(figsize=(9, 6))

    for label, values in data.items():
        values.sort(key=lambda x: x["size"])

        sizes = [v["size"] for v in values]
        means = [v["mean"] for v in values]
        stds = [v["std"] for v in values]

        plt.errorbar(
            sizes,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=label
        )

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Mean Latency (ms)")
    plt.title("CPU Matrix Multiplication: Mean Latency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        results_dir / "latency_summary.png",
        dpi=200
    )
    plt.close()

    # Figure 2: Throughput from mean latency
    plt.figure(figsize=(9, 6))

    for label, values in data.items():
        sizes = [v["size"] for v in values]
        gflops = [v["gflops"] for v in values]

        plt.plot(
            sizes,
            gflops,
            marker="o",
            label=label
        )

    plt.xscale("log", base=2)
    plt.xlabel("Matrix Size N")
    plt.ylabel("Estimated GFLOP/s")
    plt.title("CPU Matrix Multiplication: Effective Throughput")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        results_dir / "throughput_summary.png",
        dpi=200
    )
    plt.close()

    print("Saved results/latency_summary.png")
    print("Saved results/throughput_summary.png")


if __name__ == "__main__":
    main()