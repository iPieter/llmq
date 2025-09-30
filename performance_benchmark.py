#!/usr/bin/env python3
"""
Performance benchmarking script for llmq workers with batch size analysis.

This script measures the impact of batch size (VLLM_MAX_NUM_SEQS) on throughput
measured in tokens per second for translation tasks.
"""

import json
import time
import asyncio
import subprocess
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import statistics
import argparse
import re
from dateutil import parser as date_parser  # type: ignore[import-untyped]

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Warning: tiktoken not available, using character-based token estimation")


@dataclass
class RequestTiming:
    """Detailed timing information for a single request."""

    job_id: str
    submit_time: Optional[datetime] = None
    process_start_time: Optional[datetime] = None
    process_end_time: Optional[datetime] = None
    receive_time: Optional[datetime] = None
    processing_duration_ms: Optional[float] = None  # From worker logs
    queue_wait_time_ms: Optional[float] = None  # Time waiting in queue
    total_latency_ms: Optional[float] = None  # End-to-end latency
    llmq_overhead_ms: Optional[float] = None  # Overhead vs pure processing


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    batch_size: int
    total_requests: int
    total_time_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    input_tokens_per_second: float
    output_tokens_per_second: float
    total_tokens_per_second: float
    avg_request_time_ms: float
    successful_requests: int
    failed_requests: int
    # New detailed timing metrics
    avg_processing_time_ms: float = 0.0
    avg_queue_wait_time_ms: float = 0.0
    avg_total_latency_ms: float = 0.0
    avg_llmq_overhead_ms: float = 0.0
    p95_total_latency_ms: float = 0.0
    p99_total_latency_ms: float = 0.0
    detailed_timings: Optional[List[RequestTiming]] = None


class TokenCounter:
    """Token counting utility that works with or without tiktoken."""

    def __init__(self, model_name: str = "Unbabel/Tower-Plus-2B"):
        self.model_name = model_name
        self.encoder = None

        if HAS_TIKTOKEN:
            try:
                # Try to get appropriate encoder for the model
                if "gpt" in model_name.lower():
                    self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
                else:
                    # Default to cl100k_base which works for most modern models
                    self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                print(f"Warning: Could not load tiktoken encoder for {model_name}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Rough estimation: 1 token ≈ 4 characters for most models
            return len(text) // 4


class PerformanceBenchmark:
    """Main benchmarking class."""

    def __init__(
        self,
        model_name: str = "Unbabel/Tower-Plus-2B",
        queue_name: str = "benchmark-queue",
    ):
        self.model_name = model_name
        self.queue_name = queue_name
        self.token_counter = TokenCounter(model_name)
        self.results: List[BenchmarkResult] = []
        self.gpu_info = self._get_gpu_info()

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append(
                            {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "memory_mb": int(parts[2]),
                            }
                        )

            return {
                "gpu_count": len(gpus),
                "gpus": gpus,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            }
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return {
                "gpu_count": 0,
                "gpus": [],
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "unknown"
                ),
                "error": "Could not detect GPU information",
            }

    def _parse_worker_logs(self, log_content: str) -> Dict[str, RequestTiming]:
        """Parse worker logs to extract timing information for each job."""
        timings = {}
        debug_count = 0

        # Split logs by lines and parse JSON entries
        for line in log_content.split("\n"):
            line = line.strip()
            if not line:
                continue

            try:
                # Try to parse as JSON log entry
                if line.startswith('{"timestamp"'):
                    log_entry = json.loads(line)
                    timestamp_str = log_entry.get("timestamp", "")
                    message = log_entry.get("message", "")

                    # Parse timestamp
                    try:
                        timestamp = date_parser.parse(timestamp_str)

                        # Debug: print raw timestamp to understand the issue
                        if debug_count < 2:
                            print(
                                f"  Raw worker log timestamp: {timestamp_str} -> {timestamp}"
                            )
                            debug_count += 1

                        # Ensure timezone-aware
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                    except Exception as e:
                        print(
                            f"Warning: Could not parse worker log timestamp {timestamp_str}: {e}"
                        )
                        continue

                    # Check for job processing events
                    if "Processing job" in message:
                        # Extract job ID from "Processing job {job_id}"
                        match = re.search(r"Processing job (\S+)", message)
                        if match:
                            job_id = match.group(1)
                            if job_id not in timings:
                                timings[job_id] = RequestTiming(job_id=job_id)
                            timings[job_id].process_start_time = timestamp

                    elif "Completed job" in message:
                        # Extract job ID and duration from "Completed job {job_id}"
                        match = re.search(r"Completed job (\S+)", message)
                        if match:
                            job_id = match.group(1)
                            if job_id not in timings:
                                timings[job_id] = RequestTiming(job_id=job_id)
                            timings[job_id].process_end_time = timestamp

                            # Try to get duration from extra fields
                            if "duration_ms" in line:
                                duration_match = re.search(
                                    r'"duration_ms":\s*([\d.]+)', line
                                )
                                if duration_match:
                                    timings[job_id].processing_duration_ms = float(
                                        duration_match.group(1)
                                    )

            except json.JSONDecodeError:
                # Skip non-JSON lines
                continue

        return timings

    async def run_benchmark(
        self,
        batch_sizes: List[int],
        dataset_name: str = "pdelobelle/fineweb-dutch-synthetic-mt",
        dataset_size: int = 5000,
        timeout_minutes: int = 20,
    ) -> List[BenchmarkResult]:
        """Run benchmark for multiple batch sizes using a HuggingFace dataset."""
        print(f"Starting benchmark with batch sizes: {batch_sizes}")
        print(f"Dataset: {dataset_name}")
        print(f"Dataset size: {dataset_size} samples")
        print(f"Model: {self.model_name}")
        print(f"Queue: {self.queue_name}")

        # Print GPU info
        if "error" not in self.gpu_info:
            print(f"GPUs detected: {self.gpu_info['gpu_count']}")
            for gpu in self.gpu_info["gpus"]:
                print(f"  - GPU {gpu['index']}: {gpu['name']} ({gpu['memory_mb']} MB)")
            print(f"CUDA_VISIBLE_DEVICES: {self.gpu_info['cuda_visible_devices']}")
        else:
            print(f"GPU detection: {self.gpu_info.get('error', 'Unknown error')}")

        print("-" * 60)

        results = []

        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            result = await self._run_single_benchmark_hf(
                dataset_name, dataset_size, batch_size, timeout_minutes
            )
            results.append(result)
            self._print_result(result)

        return results

    async def _run_single_benchmark_hf(
        self,
        dataset_name: str,
        dataset_size: int,
        batch_size: int,
        timeout_minutes: int,
    ) -> BenchmarkResult:
        """Run benchmark for a single batch size using HuggingFace dataset."""

        # Set environment variable for batch size
        env = os.environ.copy()
        env["VLLM_MAX_NUM_SEQS"] = str(batch_size)

        # Initialize variables for cleanup
        results_file = None
        worker_log_file = None

        try:
            # Start worker in background and capture logs
            print(f"  Starting worker with batch size {batch_size}...")

            # Create temporary file for worker logs
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".log", delete=False
            ) as f:
                worker_log_file = f.name

            worker_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "llmq",
                    "worker",
                    "run",
                    self.model_name,
                    self.queue_name,
                ],
                env=env,
                stdout=open(worker_log_file, "w"),
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
            )

            # Wait for worker to be fully ready
            print("  Waiting for worker to initialize...")
            await self._wait_for_worker_ready(worker_log_file)

            # Submit jobs using HF dataset and measure time
            print(f"  Submitting {dataset_size} jobs from {dataset_name}...")
            submit_start_time = time.time()

            # Build the prompt template for translation
            # Note: Use actual newline in the string, not \n literal
            prompt_template = "Translate the following Dutch source text to English:\\nDutch:{text}\\nEnglish: "

            # Build the --map argument - this is passed directly to subprocess, no shell escaping needed
            map_arg = f'messages=[{{"role": "user", "content": "{prompt_template}"}}]'

            submit_process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "llmq",
                "submit",
                self.queue_name,
                dataset_name,
                "--map",
                map_arg,
                "--max-samples",
                str(dataset_size),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await submit_process.wait()

            # Record submit time (use UTC)
            submit_time = datetime.fromtimestamp(submit_start_time, tz=timezone.utc)
            timings: Dict[str, RequestTiming] = {}
            # We won't know individual job IDs ahead of time, will populate from results

            # Receive results
            print("  Receiving results...")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                results_file = f.name

            receive_process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "llmq",
                "receive",
                self.queue_name,
                stdout=open(results_file, "w"),
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait until we have all results (with safety timeout)
            expected_results = dataset_size
            received_results = 0
            max_wait_time = timeout_minutes * 60
            start_wait = time.time()

            print(f"  Waiting for {expected_results} results...")
            while (
                received_results < expected_results
                and (time.time() - start_wait) < max_wait_time
            ):
                await asyncio.sleep(2)
                received_results = self._count_results_in_file(results_file)
                if received_results > 0 and received_results % 500 == 0:
                    print(
                        f"  Received {received_results}/{expected_results} results..."
                    )

            if received_results < expected_results:
                print(
                    f"  Warning: Only received {received_results}/{expected_results} results after {timeout_minutes} minutes"
                )

            receive_process.terminate()
            await receive_process.wait()

            receive_end_time = time.time()
            total_time = receive_end_time - submit_start_time

            # Stop worker
            worker_process.terminate()
            worker_process.wait()

            # Analyze results and create timings from results
            (
                successful_requests,
                total_output_tokens,
                total_input_tokens,
                avg_request_time,
            ) = self._analyze_results_hf(results_file, submit_time, timings)

            # Calculate tokens/sec based on total batch processing time
            if total_time > 0:
                input_tps = total_input_tokens / total_time
                output_tps = total_output_tokens / total_time
                total_tps = (total_input_tokens + total_output_tokens) / total_time
            else:
                input_tps = output_tps = total_tps = 0

            # Calculate simple, reliable statistics
            valid_timings = [
                t for t in timings.values() if t.total_latency_ms is not None
            ]
            total_latencies = [
                t.total_latency_ms
                for t in valid_timings
                if t.total_latency_ms is not None
            ]

            avg_total_latency = (
                statistics.mean(total_latencies) if total_latencies else 0
            )

            # Calculate percentiles
            if len(total_latencies) >= 20:
                p95_latency = statistics.quantiles(total_latencies, n=20)[18]
            else:
                p95_latency = max(total_latencies) if total_latencies else 0

            if len(total_latencies) >= 100:
                p99_latency = statistics.quantiles(total_latencies, n=100)[98]
            else:
                p99_latency = max(total_latencies) if total_latencies else 0

            # Calculate LLMQ efficiency
            avg_batch_time_per_request = (
                (total_time * 1000) / dataset_size if dataset_size > 0 else 0
            )
            llmq_efficiency = avg_batch_time_per_request - avg_total_latency

            return BenchmarkResult(
                batch_size=batch_size,
                total_requests=dataset_size,
                total_time_seconds=total_time,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                input_tokens_per_second=input_tps,
                output_tokens_per_second=output_tps,
                total_tokens_per_second=total_tps,
                avg_request_time_ms=avg_request_time,
                successful_requests=successful_requests,
                failed_requests=dataset_size - successful_requests,
                avg_processing_time_ms=0,
                avg_queue_wait_time_ms=0,
                avg_total_latency_ms=avg_total_latency,
                avg_llmq_overhead_ms=llmq_efficiency,
                p95_total_latency_ms=p95_latency,
                p99_total_latency_ms=p99_latency,
                detailed_timings=list(timings.values()),
            )

        finally:
            # Cleanup
            if results_file:
                Path(results_file).unlink(missing_ok=True)
            if worker_log_file:
                Path(worker_log_file).unlink(missing_ok=True)

    async def _run_single_benchmark(
        self,
        dataset: List[Dict[str, Any]],
        batch_size: int,
        total_input_tokens: int,
        timeout_minutes: int,
    ) -> BenchmarkResult:
        """Run benchmark for a single batch size."""

        # Set environment variable for batch size
        env = os.environ.copy()
        env["VLLM_MAX_NUM_SEQS"] = str(batch_size)

        # Initialize variables for cleanup
        dataset_file = None
        results_file = None
        worker_log_file = None

        # Create temporary files for dataset and results
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for job in dataset:
                f.write(json.dumps(job) + "\n")
            dataset_file = f.name

        try:

            # Start worker in background and capture logs
            print(f"  Starting worker with batch size {batch_size}...")

            # Create temporary file for worker logs
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".log", delete=False
            ) as f:
                worker_log_file = f.name

            worker_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "llmq",
                    "worker",
                    "run",
                    self.model_name,
                    self.queue_name,
                ],
                env=env,
                stdout=open(worker_log_file, "w"),
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
            )

            # Wait for worker to be fully ready
            print("  Waiting for worker to initialize...")
            await self._wait_for_worker_ready(worker_log_file)

            # Submit jobs and measure time
            print(f"  Submitting {len(dataset)} jobs...")
            submit_start_time = time.time()

            submit_process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "llmq",
                "submit",
                self.queue_name,
                dataset_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await submit_process.wait()

            # Record submit time for each job (use UTC to match worker logs)
            submit_time = datetime.fromtimestamp(submit_start_time, tz=timezone.utc)
            timings = {}
            for job in dataset:
                job_id = job["id"]
                timings[job_id] = RequestTiming(job_id=job_id, submit_time=submit_time)

            # Receive results
            print("  Receiving results...")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                results_file = f.name

            receive_process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "llmq",
                "receive",
                self.queue_name,
                stdout=open(results_file, "w"),
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait until we have all results (with safety timeout)
            expected_results = len(dataset)
            received_results = 0
            max_wait_time = timeout_minutes * 60
            start_wait = time.time()

            print(f"  Waiting for {expected_results} results...")
            while (
                received_results < expected_results
                and (time.time() - start_wait) < max_wait_time
            ):
                await asyncio.sleep(1)
                received_results = self._count_results_in_file(results_file)
                if received_results > 0:
                    print(
                        f"  Received {received_results}/{expected_results} results..."
                    )

            if received_results < expected_results:
                print(
                    f"  Warning: Only received {received_results}/{expected_results} results after {timeout_minutes} minutes"
                )

            receive_process.terminate()
            await receive_process.wait()

            receive_end_time = time.time()
            total_time = receive_end_time - submit_start_time

            # Stop worker and read logs
            worker_process.terminate()
            worker_process.wait()

            # Read and parse worker logs
            try:
                with open(worker_log_file, "r") as f:
                    worker_logs = f.read()
                worker_timings = self._parse_worker_logs(worker_logs)

                # Merge timing information
                for job_id, worker_timing in worker_timings.items():
                    if job_id in timings:
                        timings[job_id].process_start_time = (
                            worker_timing.process_start_time
                        )
                        timings[job_id].process_end_time = (
                            worker_timing.process_end_time
                        )
                        timings[job_id].processing_duration_ms = (
                            worker_timing.processing_duration_ms
                        )
            except Exception as e:
                print(f"  Warning: Could not parse worker logs: {e}")

            # Analyze results and add receive times
            successful_requests, total_output_tokens, avg_request_time = (
                self._analyze_results(results_file, timings)
            )

            # Calculate tokens/sec based on end-to-end latency (first principles approach)
            # Use actual measured time from submit to receive for all completed requests
            valid_timings = [
                t for t in timings.values() if t.total_latency_ms is not None
            ]

            # Tokens per second based on total batch processing time
            # This gives aggregate throughput for the entire batch
            if total_time > 0:
                input_tps = total_input_tokens / total_time
                output_tps = total_output_tokens / total_time
                total_tps = (total_input_tokens + total_output_tokens) / total_time
            else:
                input_tps = output_tps = total_tps = 0

            # Calculate simple, reliable statistics
            total_latencies = [
                t.total_latency_ms
                for t in valid_timings
                if t.total_latency_ms is not None
            ]

            avg_total_latency = (
                statistics.mean(total_latencies) if total_latencies else 0
            )

            # Calculate percentiles
            if len(total_latencies) >= 20:
                p95_latency = statistics.quantiles(total_latencies, n=20)[18]
            else:
                p95_latency = max(total_latencies) if total_latencies else 0

            if len(total_latencies) >= 100:
                p99_latency = statistics.quantiles(total_latencies, n=100)[98]
            else:
                p99_latency = max(total_latencies) if total_latencies else 0

            # Calculate LLMQ overhead as difference between batch time and individual latencies
            # If batch processing is more efficient, this will show the benefit
            avg_batch_time_per_request = (
                (total_time * 1000) / len(dataset) if len(dataset) > 0 else 0
            )
            llmq_efficiency = avg_batch_time_per_request - avg_total_latency

            return BenchmarkResult(
                batch_size=batch_size,
                total_requests=len(dataset),
                total_time_seconds=total_time,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                input_tokens_per_second=input_tps,
                output_tokens_per_second=output_tps,
                total_tokens_per_second=total_tps,
                avg_request_time_ms=avg_request_time,
                successful_requests=successful_requests,
                failed_requests=len(dataset) - successful_requests,
                avg_processing_time_ms=0,  # Not reliably measurable with current setup
                avg_queue_wait_time_ms=0,  # Not reliably measurable with current setup
                avg_total_latency_ms=avg_total_latency,
                avg_llmq_overhead_ms=llmq_efficiency,  # Positive means batching is efficient
                p95_total_latency_ms=p95_latency,
                p99_total_latency_ms=p99_latency,
                detailed_timings=list(timings.values()),
            )

        finally:
            # Cleanup
            if dataset_file:
                Path(dataset_file).unlink(missing_ok=True)
            if results_file:
                Path(results_file).unlink(missing_ok=True)
            if worker_log_file:
                Path(worker_log_file).unlink(missing_ok=True)

    def _count_results_in_file(self, results_file: str) -> int:
        """Count the number of results currently in the file."""
        try:
            count = 0
            with open(results_file, "r") as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        except FileNotFoundError:
            return 0

    def _analyze_results_hf(
        self,
        results_file: str,
        submit_time: datetime,
        timings: Dict[str, RequestTiming],
    ) -> tuple[int, int, int, float]:
        """Analyze HF dataset results and return (successful_count, total_output_tokens, total_input_tokens, avg_time_ms)."""
        successful_count = 0
        total_output_tokens = 0
        total_input_tokens = 0
        durations = []

        try:
            with open(results_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            result = json.loads(line)
                            successful_count += 1
                            job_id = result.get("id")

                            # Create timing entry for this job
                            if job_id and job_id not in timings:
                                timings[job_id] = RequestTiming(
                                    job_id=job_id, submit_time=submit_time
                                )

                            # Update timing with receive time
                            if job_id and job_id in timings:
                                if "timestamp" in result:
                                    try:
                                        parsed_time = date_parser.parse(
                                            result["timestamp"]
                                        )
                                        if parsed_time.tzinfo is None:
                                            parsed_time = parsed_time.replace(
                                                tzinfo=timezone.utc
                                            )
                                        timings[job_id].receive_time = parsed_time
                                    except Exception:
                                        pass

                            # Count output tokens
                            if "result" in result:
                                total_output_tokens += self.token_counter.count_tokens(
                                    result["result"]
                                )

                            # Count input tokens from the prompt
                            if "prompt" in result:
                                total_input_tokens += self.token_counter.count_tokens(
                                    result["prompt"]
                                )
                            elif "messages" in result:
                                for msg in result["messages"]:
                                    if "content" in msg:
                                        total_input_tokens += (
                                            self.token_counter.count_tokens(
                                                msg["content"]
                                            )
                                        )

                            # Collect duration
                            if "duration_ms" in result:
                                durations.append(result["duration_ms"])

                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass

        # Calculate timing metrics
        self._calculate_timing_metrics(timings)

        avg_duration = statistics.mean(durations) if durations else 0
        return successful_count, total_output_tokens, total_input_tokens, avg_duration

    def _analyze_results(
        self, results_file: str, timings: Dict[str, RequestTiming]
    ) -> tuple[int, int, float]:
        """Analyze results file and return (successful_count, total_output_tokens, avg_time_ms)."""
        successful_count = 0
        total_output_tokens = 0
        durations = []

        try:
            with open(results_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            result = json.loads(line)
                            successful_count += 1
                            job_id = result.get("id")

                            # Update timing with receive time
                            if job_id and job_id in timings:
                                # Use result timestamp as receive time
                                if "timestamp" in result:
                                    try:
                                        # Parse result timestamp and ensure it's timezone-aware
                                        parsed_time = date_parser.parse(
                                            result["timestamp"]
                                        )
                                        if parsed_time.tzinfo is None:
                                            # If somehow timezone-naive, assume UTC
                                            parsed_time = parsed_time.replace(
                                                tzinfo=timezone.utc
                                            )
                                        timings[job_id].receive_time = parsed_time
                                    except Exception as e:
                                        print(
                                            f"Warning: Could not parse timestamp {result['timestamp']}: {e}"
                                        )
                                        pass

                            # Count output tokens
                            if "result" in result:
                                total_output_tokens += self.token_counter.count_tokens(
                                    result["result"]
                                )

                            # Collect duration
                            if "duration_ms" in result:
                                durations.append(result["duration_ms"])

                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass

        # Calculate detailed timing metrics
        self._calculate_timing_metrics(timings)

        avg_duration = statistics.mean(durations) if durations else 0
        return successful_count, total_output_tokens, avg_duration

    def _calculate_timing_metrics(self, timings: Dict[str, RequestTiming]) -> None:
        """Calculate timing metrics using first principles approach."""
        debug_count = 0
        for timing in timings.values():
            # Only calculate what we can reliably measure: submit to receive latency
            if timing.submit_time and timing.receive_time:
                timing.total_latency_ms = (
                    timing.receive_time - timing.submit_time
                ).total_seconds() * 1000

                # Debug first few timings
                if debug_count < 3:
                    print(f"  Debug timing {timing.job_id}:")
                    print(f"    Submit: {timing.submit_time}")
                    print(f"    Receive: {timing.receive_time}")
                    print(f"    End-to-end latency: {timing.total_latency_ms:.1f}ms")
                    debug_count += 1

            # For LLMQ overhead, we'll calculate it differently - based on batch vs individual performance
            # This will be done at the batch level, not per-request

    async def _wait_for_worker_ready(
        self, worker_log_file: str, timeout_minutes: int = 5
    ) -> None:
        """Wait for worker to be fully ready by monitoring its log file."""
        ready_message = "starting to consume from queue"
        timeout_seconds = timeout_minutes * 60
        start_time = time.time()

        while (time.time() - start_time) < timeout_seconds:
            try:
                with open(worker_log_file, "r") as f:
                    content = f.read()

                # Look for the ready message
                if ready_message in content:
                    print(f"  Worker ready! (took {time.time() - start_time:.1f}s)")
                    # Give it a moment extra to be fully settled
                    await asyncio.sleep(2)
                    return

            except FileNotFoundError:
                # Log file might not exist yet
                pass

            await asyncio.sleep(1)

        raise TimeoutError(
            f"Worker did not become ready within {timeout_minutes} minutes"
        )

    def _print_result(self, result: BenchmarkResult):
        """Print formatted benchmark result."""
        print("  Results:")
        print(f"    Batch processing time: {result.total_time_seconds:.1f}s")
        print(
            f"    Successful requests: {result.successful_requests}/{result.total_requests}"
        )
        print("    Tokens/sec (based on end-to-end latency):")
        print(f"      Input tokens/sec: {result.input_tokens_per_second:.1f}")
        print(f"      Output tokens/sec: {result.output_tokens_per_second:.1f}")
        print(f"      Total tokens/sec: {result.total_tokens_per_second:.1f}")
        print("    Latency metrics:")
        print(f"      Average end-to-end latency: {result.avg_total_latency_ms:.1f}ms")
        print(f"      P95 latency: {result.p95_total_latency_ms:.1f}ms")
        print(f"      P99 latency: {result.p99_total_latency_ms:.1f}ms")
        if result.avg_llmq_overhead_ms > 0:
            print(
                f"      Batching efficiency: +{result.avg_llmq_overhead_ms:.1f}ms saved per request"
            )
        else:
            print(
                f"      Batching overhead: {abs(result.avg_llmq_overhead_ms):.1f}ms extra per request"
            )

    def print_summary(self, results: List[BenchmarkResult]):
        """Print summary comparison of all results."""
        print("\n" + "=" * 100)
        print("BENCHMARK SUMMARY")
        print("=" * 100)

        print(
            f"{'Batch':<6} {'Total':<8} {'Process':<8} {'Queue':<8} {'Overhead':<9} {'P95':<8} {'P99':<8} {'Success':<8}"
        )
        print(
            f"{'Size':<6} {'t/s':<8} {'ms':<8} {'ms':<8} {'ms':<9} {'ms':<8} {'ms':<8} {'Rate':<8}"
        )
        print("-" * 100)

        for result in results:
            success_rate = result.successful_requests / result.total_requests * 100
            print(
                f"{result.batch_size:<6} {result.total_tokens_per_second:<8.0f} "
                f"{result.avg_processing_time_ms:<8.0f} {result.avg_queue_wait_time_ms:<8.0f} "
                f"{result.avg_llmq_overhead_ms:<9.0f} {result.p95_total_latency_ms:<8.0f} "
                f"{result.p99_total_latency_ms:<8.0f} {success_rate:<8.1f}%"
            )

        # Find best performing batch size
        best_result = max(results, key=lambda r: r.total_tokens_per_second)
        print(
            f"\nBest throughput: Batch size {best_result.batch_size} "
            f"({best_result.total_tokens_per_second:.1f} tokens/sec)"
        )

        # Find lowest latency
        best_latency = min(results, key=lambda r: r.avg_total_latency_ms)
        print(
            f"Lowest latency: Batch size {best_latency.batch_size} "
            f"({best_latency.avg_total_latency_ms:.1f}ms avg)"
        )

        # Find lowest overhead
        best_overhead = min(results, key=lambda r: r.avg_llmq_overhead_ms)
        print(
            f"Lowest overhead: Batch size {best_overhead.batch_size} "
            f"({best_overhead.avg_llmq_overhead_ms:.1f}ms avg)"
        )

    def save_results(
        self, results: List[BenchmarkResult], filename: Optional[str] = None
    ):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        data = {
            "model": self.model_name,
            "queue": self.queue_name,
            "timestamp": datetime.now().isoformat(),
            "gpu_info": self.gpu_info,
            "results": [
                {
                    "batch_size": r.batch_size,
                    "total_requests": r.total_requests,
                    "total_time_seconds": r.total_time_seconds,
                    "total_input_tokens": r.total_input_tokens,
                    "total_output_tokens": r.total_output_tokens,
                    "input_tokens_per_second": r.input_tokens_per_second,
                    "output_tokens_per_second": r.output_tokens_per_second,
                    "total_tokens_per_second": r.total_tokens_per_second,
                    "avg_request_time_ms": r.avg_request_time_ms,
                    "successful_requests": r.successful_requests,
                    "failed_requests": r.failed_requests,
                }
                for r in results
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {filename}")


async def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark llmq worker performance across batch sizes"
    )
    parser.add_argument(
        "--model", default="Unbabel/Tower-Plus-2B", help="Model name to benchmark"
    )
    parser.add_argument("--queue", default="benchmark-queue", help="Queue name to use")
    parser.add_argument(
        "--dataset",
        default="pdelobelle/fineweb-dutch-synthetic-mt",
        help="HuggingFace dataset to use",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--dataset-size", type=int, default=5000, help="Number of samples to test"
    )
    parser.add_argument(
        "--timeout", type=int, default=20, help="Timeout in minutes per batch size"
    )
    parser.add_argument("--output", help="Output filename for results")

    args = parser.parse_args()

    benchmark = PerformanceBenchmark(args.model, args.queue)

    try:
        results = await benchmark.run_benchmark(
            batch_sizes=args.batch_sizes,
            dataset_name=args.dataset,
            dataset_size=args.dataset_size,
            timeout_minutes=args.timeout,
        )

        benchmark.print_summary(results)
        benchmark.save_results(results, args.output)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
