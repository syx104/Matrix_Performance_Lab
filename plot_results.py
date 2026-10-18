
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    input_file = Path("results/benchmark.csv")

    if not input_file.exists():
        raise FileNotFoundError(
            "Run benchmark.py first"
        )

    data = defaultdict(list)

    with input_file.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            method = row["method"]
            threads = int(row["threads"])

            if method == "PyTorch":
                label = f"PyTorch ({threads} threads)"
            else:
                label = method

            data[label].append({
                "size": int(row["size"]),
                "latency": float(row["latency_ms"]),
                "gflops": float(row["gflops"])
            })

    if not data:
        raise ValueError("Benchmark CSV is empty")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Figure 1: Latency
    plt.figure(figsize=(9, 6))

    for label, values in data.items():
        values.sort(key=lambda x: x["size"])

        sizes = [v["size"] for v in values]
        latency = [v["latency"] for v in values]

        plt.plot(
            sizes, latency,
            marker="o",
            label=label
        )

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Median Latency (ms)")
    plt.title("CPU Matrix Multiplication Latency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "latency.png",
        dpi=200
    )
    plt.close()

    # Figure 2: Throughput
    plt.figure(figsize=(9, 6))

    for label, values in data.items():
        sizes = [v["size"] for v in values]
        gflops = [v["gflops"] for v in values]

        plt.plot(
            sizes, gflops,
            marker="o",
            label=label
        )

    plt.xscale("log", base=2)
    plt.xlabel("Matrix Size N")
    plt.ylabel("Estimated GFLOP/s")
    plt.title("CPU Matrix Multiplication Throughput")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "throughput.png",
        dpi=200
    )
    plt.close()

    print("Saved results/latency.png")
    print("Saved results/throughput.png")


if __name__ == "__main__":
    main()