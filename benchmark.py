
import csv
import platform
from pathlib import Path

import numpy as np
import torch
from torch.utils.benchmark import Timer

from matmul import naive_matmul


SIZES = [16, 32, 128, 256, 512]
THREADS = [1, 2, 4]
MIN_RUN_TIME = 0.2


def measure(statement, variables, num_threads=1):
    timer = Timer(
        stmt=statement,
        globals=variables,
        num_threads=num_threads
    )

    result = timer.blocked_autorange(
        min_run_time=MIN_RUN_TIME
    )

    return result.median


def main():
    print("Python:", platform.python_version())
    print("NumPy:", np.__version__)
    print("PyTorch:", torch.__version__)
    print("CPU:", platform.processor())

    results = []

    for n in SIZES:
        print(f"\nTesting matrix size: {n} x {n}")

        rng = np.random.default_rng(2026 + n)

        a = rng.standard_normal(
            (n, n)
        ).astype(np.float32)

        b = rng.standard_normal(
            (n, n)
        ).astype(np.float32)

        ta = torch.from_numpy(a.copy())
        tb = torch.from_numpy(b.copy())

        expected = a @ b

        np.testing.assert_allclose(
            (ta @ tb).numpy(),
            expected,
            rtol=2e-3,
            atol=2e-3
        )

        cases = [
            (
                "NumPy",
                "a @ b",
                {"a": a, "b": b},
                1
            )
        ]

        if n <= 32:
            np.testing.assert_allclose(
                naive_matmul(a, b),
                expected,
                rtol=2e-3,
                atol=2e-3
            )

            cases.append((
                "Python loops",
                "naive_matmul(a, b)",
                {
                    "a": a,
                    "b": b,
                    "naive_matmul": naive_matmul
                },
                1
            ))

        for threads in THREADS:
            cases.append((
                "PyTorch",
                "ta @ tb",
                {"ta": ta, "tb": tb},
                threads
            ))

        for method, statement, variables, threads in cases:
            seconds = measure(
                statement,
                variables,
                num_threads=threads
            )

            milliseconds = seconds * 1000
            gflops = 2 * n**3 / seconds / 1e9

            row = {
                "method": method,
                "size": n,
                "threads": threads,
                "latency_ms": milliseconds,
                "gflops": gflops
            }

            results.append(row)

            print(
                f"{method:15s} "
                f"threads={threads} "
                f"time={milliseconds:.4f} ms "
                f"GFLOP/s={gflops:.3f}"
            )

    output = Path("results/benchmark.csv")
    output.parent.mkdir(exist_ok=True)

    with output.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "size",
                "threads",
                "latency_ms",
                "gflops"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()