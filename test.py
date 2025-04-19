#!/usr/bin/env python3

import argparse
import subprocess
import statistics
import matplotlib.pyplot as plt
import numpy as np

default_algorithms = ["openblas", "unroll", "kernel"]
WARMUP_RUNS = 2
MEASURE_RUNS = 10
DEFAULT_TESTS_PATH = "tests.txt"
DEFAULT_EXEC_PATH = "./build/tests"

def read_tests(file_path):
    tests = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            M, N, K, reps = map(int, parts)
            tests.append((M, N, K, reps))

    return tests

def run_invocation(exe_path, algo, M, N, K, reps):
    result = subprocess.run(
        [exe_path, algo, str(M), str(N), str(K), str(reps)],
        capture_output=True,
        check=True,
        text=True
    )

    return int(result.stdout.strip())


def measure_algorithm(exe_path, algo, tests):
    results = []
    print(f"\nBenchmarking algorithm: {algo}")

    for M, N, K, reps in tests:
        print(f"  Test case M={M}, N={N}, K={K}, reps={reps}")

        for _ in range(WARMUP_RUNS):
            subprocess.run([exe_path, algo, str(M), str(N), str(K), str(reps)], capture_output=True, check=True)

        cycles = [run_invocation(exe_path, algo, M, N, K, reps) for _ in range(MEASURE_RUNS)]
        results.append((f"{M}x{N}x{K}", statistics.mean(cycles), statistics.stdev(cycles)))
    return results


def plot_grouped_bar_chart(grouped_stats, output_file="performance.png"):
    test_labels = list({label for algo_stats in grouped_stats.values() for (label, _, _) in algo_stats})
    test_labels.sort()
    x = np.arange(len(test_labels))
    width = 0.15

    _, ax = plt.subplots()

    for i, algo in enumerate(default_algorithms):
        means = []
        stds = []
        stats_dict = {label: (mean, std) for label, mean, std in grouped_stats[algo]}
        for label in test_labels:
            mean, std = stats_dict.get(label, (0, 0))
            means.append(mean)
            stds.append(std)
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4, label=algo)

    ax.set_xlabel("(MxNxK)")
    ax.set_ylabel("Clock Cycles")
    ax.set_title("Matrix Multiplication Algorithms Comparison")
    ax.set_xticks(x + width * (len(default_algorithms) - 1) / 2)
    ax.set_xticklabels(test_labels, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Results saved to {output_file}")


def main():
    tests = read_tests(DEFAULT_TESTS_PATH)
    if not tests:
        print("No valid tests found. Exiting.")
        return

    grouped_stats = {}
    for algo in default_algorithms:
        grouped_stats[algo] = measure_algorithm(DEFAULT_EXEC_PATH, algo, tests)

    plot_grouped_bar_chart(grouped_stats)


if __name__ == "__main__":
    main()

