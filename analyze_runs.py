
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    results_dir = Path("results")

    input_files = [
        results_dir / "benchmark_run1.csv",
        results_dir / "benchmark_run2.csv",
        results_dir / "benchmark_run3.csv",
    ]

    for file in input_files:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    data = defaultdict(list)

    for file in input_files:
        with file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            seen = set()

            for row in reader:
                method = row["method"]
                size = int(row["size"])
                threads = int(row["threads"])
                latency = float(row["latency_ms"])

                key = (method, size, threads)

                if key in seen:
                    raise ValueError(
                        f"Duplicate configuration in {file}: {key}"
                    )

                seen.add(key)
                data[key].append(latency)

    summary = []

    for (method, size, threads), latencies in sorted(data.items()):

        if len(latencies) != 3:
            raise ValueError(
                f"Expected 3 runs for {(method, size, threads)}, "
                f"got {len(latencies)}"
            )

        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies)

        cv_percent = std_latency / mean_latency * 100

        mean_seconds = mean_latency / 1000

        gflops = 2 * size**3 / mean_seconds / 1e9

        summary.append({
            "method": method,
            "size": size,
            "threads": threads,
            "mean_latency_ms": mean_latency,
            "std_latency_ms": std_latency,
            "cv_percent": cv_percent,
            "gflops_from_mean_latency": gflops,
        })

    output_file = results_dir / "benchmark_summary.csv"

    fieldnames = [
        "method",
        "size",
        "threads",
        "mean_latency_ms",
        "std_latency_ms",
        "cv_percent",
        "gflops_from_mean_latency",
    ]

    with output_file.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved: {output_file}")
    print(f"Total configurations: {len(summary)}")

    print(
        f"{'Method':15s} {'N':>5s} {'Threads':>8s} "
        f"{'Mean(ms)':>12s} {'CV(%)':>10s}"
    )

    for row in summary:
        print(
            f"{row['method']:15s} "
            f"{row['size']:5d} "
            f"{row['threads']:8d} "
            f"{row['mean_latency_ms']:12.6f} "
            f"{row['cv_percent']:10.2f}"
        )


if __name__ == "__main__":
    main()