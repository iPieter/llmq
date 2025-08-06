#!/usr/bin/env python3
"""
Benchmark script to test different VLLM_QUEUE_PREFETCH batch sizes and measure processing speed.

Usage:
    python benchmark_batch_sizes.py

Requirements:
    - vLLM worker should be running: llmq worker run Unbabel/Tower-Plus-9B translation-queue
    - Set VLLM_MAX_NUM_SEQS=1000 before running the worker
"""

import os
import subprocess
import time
import re
import json
import sys
from typing import Dict, Tuple

try:
    import matplotlib.pyplot as plt
    import pandas as pd

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    pd = None


class BatchSizeBenchmark:
    def __init__(self, model_name="Unbabel/Tower-Plus-9B"):
        self.batch_sizes = list(range(50, 2100, 50))  # 50 to 2000 in steps of 50
        self.results = []
        self.queue_name = "translation-queue"
        self.dataset = "Aleph-Alpha/Aleph-Alpha-GermanWeb"
        self.max_samples = 1000  # Smaller sample size for faster benchmarking
        self.map_template = 'messages=[{"role": "user", "content": "Translate the following German source text to Dutch:\\nGerman: {text}\\nDutch: "}]'
        self.model_name = model_name
        self.worker_process = None

    def start_worker(self, batch_size: int):
        """Start vLLM worker with specific batch size."""
        print(
            f"Starting worker with batch_size={batch_size}, prefetch={batch_size * 2}..."
        )

        # Set environment variables
        env = os.environ.copy()
        env["VLLM_MAX_NUM_SEQS"] = str(batch_size)
        env["VLLM_QUEUE_PREFETCH"] = str(batch_size * 2)

        # Start worker process
        cmd = ["llmq", "worker", "run", self.model_name, self.queue_name]
        self.worker_process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait a moment for worker to start
        time.sleep(10)  # Give worker time to initialize

        # Check if worker started successfully
        if self.worker_process.poll() is not None:
            stdout, stderr = self.worker_process.communicate()
            raise Exception(f"Worker failed to start: {stderr}")

        print("✓ Worker started successfully")

    def stop_worker(self):
        """Stop the current worker."""
        if self.worker_process and self.worker_process.poll() is None:
            print("Stopping worker...")
            self.worker_process.terminate()
            try:
                self.worker_process.wait(timeout=10)
                print("✓ Worker stopped")
            except subprocess.TimeoutExpired:
                print("Worker didn't stop gracefully, killing...")
                self.worker_process.kill()
                self.worker_process.wait()
                print("✓ Worker killed")
            self.worker_process = None

    def run_single_test(self, batch_size: int) -> Tuple[float, Dict]:
        """Run a single benchmark test with given batch size."""
        print(f"\n{'='*60}")
        print(f"Testing batch size: {batch_size} (prefetch: {batch_size * 2})")
        print(f"{'='*60}")

        # Restart worker with new batch size
        self.stop_worker()
        time.sleep(2)  # Brief pause between stop/start

        try:
            self.start_worker(batch_size)
        except Exception as e:
            return 0.0, {"error": f"Worker start failed: {e}"}

        # Prepare command
        cmd = [
            "llmq",
            "submit",
            self.queue_name,
            self.dataset,
            "--subset",
            "synthetic",
            "--map",
            self.map_template,
            "--max-samples",
            str(self.max_samples),
        ]

        # Run the command and capture output
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            end_time = time.time()

            if result.returncode != 0:
                print(f"ERROR: Command failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return 0.0, {"error": result.stderr}

            # Parse the completion rate from stderr
            stderr_output = result.stderr
            stdout_output = result.stdout

            # Look for the completion rate pattern
            # "Completed X jobs in Y.Zs (A.B jobs/sec)"
            rate_pattern = r"Completed \d+ jobs in [\d.]+s \(([\d.]+) jobs/sec\)"
            match = re.search(rate_pattern, stderr_output)

            if match:
                jobs_per_sec = float(match.group(1))
                print(f"✓ Completion rate: {jobs_per_sec:.1f} jobs/sec")

                # Count actual results
                result_lines = len(
                    [line for line in stdout_output.strip().split("\n") if line.strip()]
                )
                total_time = end_time - start_time

                return jobs_per_sec, {
                    "batch_size": batch_size,
                    "jobs_per_sec": jobs_per_sec,
                    "total_time": total_time,
                    "result_count": result_lines,
                    "stderr": stderr_output,
                    "success": True,
                }
            else:
                print("ERROR: Could not parse completion rate from output")
                print(f"STDERR: {stderr_output}")
                return 0.0, {
                    "error": "Could not parse completion rate",
                    "stderr": stderr_output,
                }

        except subprocess.TimeoutExpired:
            print("ERROR: Command timed out after 10 minutes")
            return 0.0, {"error": "timeout"}
        except Exception as e:
            print(f"ERROR: {e}")
            return 0.0, {"error": str(e)}

    def run_benchmark(self):
        """Run the full benchmark across all batch sizes."""
        print("Starting batch size benchmark...")
        print(f"Testing batch sizes: {self.batch_sizes}")
        print(f"Using {self.max_samples} samples per test")
        print("Note: Worker will be restarted for each batch size test")

        try:
            for batch_size in self.batch_sizes:
                rate, details = self.run_single_test(batch_size)
                self.results.append(details)

                if details.get("success"):
                    print(f"✓ Batch size {batch_size}: {rate:.1f} jobs/sec")
                else:
                    print(
                        f"✗ Batch size {batch_size}: FAILED - {details.get('error', 'unknown error')}"
                    )

                # Small delay between tests
                time.sleep(2)

        finally:
            # Always cleanup worker at the end
            print("\nCleaning up...")
            self.stop_worker()

        # Save results
        self.save_results()
        self.create_chart()

    def save_results(self):
        """Save results to JSON and CSV files."""
        timestamp = int(time.time())

        # Save detailed JSON
        json_file = f"benchmark_results_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: {json_file}")

        # Save CSV summary
        csv_file = f"benchmark_summary_{timestamp}.csv"
        successful_results = [r for r in self.results if r.get("success")]

        if successful_results and HAS_PLOTTING:
            df = pd.DataFrame(successful_results)
            # Only keep essential columns for CSV
            summary_df = df[
                ["batch_size", "jobs_per_sec", "total_time", "result_count"]
            ]
            summary_df.to_csv(csv_file, index=False)
            print(f"Summary CSV saved to: {csv_file}")
        elif successful_results and not HAS_PLOTTING:
            # Manual CSV creation without pandas
            with open(csv_file, "w") as f:
                f.write("batch_size,jobs_per_sec,total_time,result_count\n")
                for r in successful_results:
                    f.write(
                        f"{r['batch_size']},{r['jobs_per_sec']},{r['total_time']},{r['result_count']}\n"
                    )
            print(f"Summary CSV saved to: {csv_file}")

        return json_file, csv_file

    def create_chart(self):
        """Create a performance chart."""
        successful_results = [r for r in self.results if r.get("success")]

        if not successful_results:
            print("No successful results to chart")
            return

        # Extract data for plotting
        batch_sizes = [r["batch_size"] for r in successful_results]
        rates = [r["jobs_per_sec"] for r in successful_results]

        if HAS_PLOTTING:
            # Create the plot
            plt.figure(figsize=(12, 8))
            plt.plot(batch_sizes, rates, "b-o", linewidth=2, markersize=6)
            plt.xlabel("Batch Size (VLLM_QUEUE_PREFETCH)", fontsize=12)
            plt.ylabel("Processing Rate (jobs/sec)", fontsize=12)
            plt.title("LLMQ Processing Rate vs Batch Size", fontsize=14)
            plt.grid(True, alpha=0.3)

            # Add some annotations for key points
            max_rate_idx = rates.index(max(rates))
            max_batch_size = batch_sizes[max_rate_idx]
            max_rate = rates[max_rate_idx]

            plt.annotate(
                f"Peak: {max_rate:.1f} jobs/sec\n@ batch size {max_batch_size}",
                xy=(max_batch_size, max_rate),
                xytext=(max_batch_size + 200, max_rate + 2),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

            plt.tight_layout()

            # Save the chart
            timestamp = int(time.time())
            chart_file = f"batch_size_performance_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches="tight")
            print(f"Performance chart saved to: {chart_file}")

            # Show the chart
            plt.show()

        # Print summary
        max_rate_idx = rates.index(max(rates))
        max_batch_size = batch_sizes[max_rate_idx]
        max_rate = rates[max_rate_idx]

        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Tests completed: {len(successful_results)}/{len(self.batch_sizes)}")
        print(
            f"Peak performance: {max_rate:.1f} jobs/sec at batch size {max_batch_size}"
        )
        print(f"Performance range: {min(rates):.1f} - {max(rates):.1f} jobs/sec")
        if HAS_PLOTTING:
            print(f"Chart saved to: {chart_file}")
        else:
            print("Note: Install matplotlib and pandas for chart generation")


def main():
    """Main entry point."""
    # Check if llmq command is available
    try:
        result = subprocess.run(["llmq", "--help"], capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: llmq command not found. Please ensure LLMQ is installed.")
            sys.exit(1)
    except FileNotFoundError:
        print("ERROR: llmq command not found. Please ensure LLMQ is installed.")
        sys.exit(1)

    print("LLMQ Batch Size Benchmark")
    print("=" * 40)
    print("This will test different batch sizes and measure processing speed.")
    print("The script will automatically start/restart vLLM workers as needed.")
    print()
    print("Requirements:")
    print("- llmq must be installed and available")
    print("- Model 'Unbabel/Tower-Plus-9B' should be available")
    print("- No existing workers should be running on 'translation-queue'")
    print()

    if not HAS_PLOTTING:
        print("Note: matplotlib and pandas not found. Charts will not be generated.")
        print("To enable charts, install with: pip install matplotlib pandas")
        print()

    response = input("Ready to start automated benchmark? (y/N): ")
    if response.lower() != "y":
        print("Benchmark cancelled.")
        sys.exit(1)

    benchmark = BatchSizeBenchmark()
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
