#!/usr/bin/env python3
"""
Continuous Batching (Iteration-Level Scheduling) Simulator.
Compares Static Batching vs. Continuous Batching on throughput,
latency (TTFT, turnaround time), and GPU utilization.
"""

from dataclasses import dataclass
from typing import List, Tuple
import random


@dataclass
class Job:
    job_id: int
    arrival_step: int
    prompt_len: int
    gen_len: int
    tokens_generated: int = 0
    start_step: int = -1
    finish_step: int = -1


def simulate_static_batching(jobs: List[Job], batch_capacity: int = 8) -> Tuple[int, float, List[Job]]:
    """
    Static Batching: Groups up to batch_capacity jobs.
    Runs until the LONGEST job in the batch finishes.
    Completed jobs wait idle; no new jobs are added until the entire batch finishes.
    """
    # Clone jobs
    work_jobs = [Job(j.job_id, j.arrival_step, j.prompt_len, j.gen_len) for j in jobs]
    step = 0
    total_slots_executed = 0
    total_useful_tokens = sum(j.prompt_len + j.gen_len for j in work_jobs)

    queue = list(work_jobs)
    idx = 0

    while idx < len(queue):
        # Form a static batch
        current_batch = queue[idx:idx + batch_capacity]
        idx += len(current_batch)

        batch_start_step = max(step, max(j.arrival_step for j in current_batch))
        step = batch_start_step

        for j in current_batch:
            j.start_step = step

        max_gen = max(j.gen_len for j in current_batch)
        # Advance step by prefill + max_gen
        # Prefill: roughly 1 step per job or amortized
        prefill_steps = len(current_batch)
        step += prefill_steps + max_gen

        for j in current_batch:
            j.finish_step = step

        # Slots used: batch_capacity * (prefill_steps + max_gen)
        total_slots_executed += len(current_batch) * (prefill_steps + max_gen)

    utilization = (total_useful_tokens / total_slots_executed) * 100.0 if total_slots_executed > 0 else 0.0
    return step, utilization, work_jobs


def simulate_continuous_batching(jobs: List[Job], batch_capacity: int = 8) -> Tuple[int, float, List[Job]]:
    """
    Continuous Batching (Iteration-level scheduling):
    At EVERY single decode step, if a job finishes, it is evicted immediately,
    and a waiting job from the queue is dynamically admitted.
    """
    work_jobs = [Job(j.job_id, j.arrival_step, j.prompt_len, j.gen_len) for j in jobs]
    waiting_queue = list(work_jobs)
    running_batch: List[Job] = []

    step = 0
    total_slots = 0
    useful_tokens = 0

    while waiting_queue or running_batch:
        # 1. Admit new requests if capacity exists
        while len(running_batch) < batch_capacity and waiting_queue:
            if waiting_queue[0].arrival_step <= step:
                admitted = waiting_queue.pop(0)
                admitted.start_step = step
                running_batch.append(admitted)
                # Prefill token accounted
                useful_tokens += admitted.prompt_len
            else:
                break

        if not running_batch:
            step = waiting_queue[0].arrival_step
            continue

        # 2. Execute 1 decode step across all running jobs
        step += 1
        total_slots += batch_capacity  # Batch slots on GPU

        finished_this_step = []
        for job in running_batch:
            job.tokens_generated += 1
            useful_tokens += 1
            if job.tokens_generated >= job.gen_len:
                job.finish_step = step
                finished_this_step.append(job)

        # 3. Immediately evict finished requests
        for job in finished_this_step:
            running_batch.remove(job)

    utilization = (useful_tokens / total_slots) * 100.0 if total_slots > 0 else 0.0
    return step, utilization, work_jobs


def print_batching_comparison():
    print("=" * 84)
    print("  CONTINUOUS BATCHING vs. STATIC BATCHING SCHEDULER BENCHMARK")
    print("=" * 84)

    random.seed(42)
    num_jobs = 48
    batch_capacity = 8

    # Generate requests with variable arrival times and generation lengths
    test_jobs = []
    current_time = 0
    for i in range(num_jobs):
        current_time += random.randint(0, 4)
        gen_len = random.randint(15, 120)  # High variance in completion time
        prompt_len = random.randint(64, 256)
        test_jobs.append(Job(job_id=i, arrival_step=current_time, prompt_len=prompt_len, gen_len=gen_len))

    static_steps, static_util, static_results = simulate_static_batching(test_jobs, batch_capacity)
    cont_steps, cont_util, cont_results = simulate_continuous_batching(test_jobs, batch_capacity)

    static_latencies = [j.finish_step - j.arrival_step for j in static_results]
    cont_latencies = [j.finish_step - j.arrival_step for j in cont_results]

    avg_static_lat = sum(static_latencies) / len(static_latencies)
    avg_cont_lat = sum(cont_latencies) / len(cont_latencies)

    print(f"Workload: {num_jobs} jobs, GPU Batch Slot Capacity = {batch_capacity}\n")
    print(f"{'Performance Metric':<36} | {'Static Batching':<20} | {'Continuous Batching':<20}")
    print("-" * 82)
    print(f"{'Total Time to Complete (steps)':<36} | {static_steps:<20} | {cont_steps:<20}")
    print(f"{'Overall Throughput Speedup':<36} | {'1.00x (Baseline)':<20} | {static_steps / cont_steps:<20.2f}x")
    print(f"{'Average Turnaround Latency (steps)':<36} | {avg_static_lat:<20.1f} | {avg_cont_lat:<20.1f}")
    print(f"{'Latency Improvement':<36} | {'1.00x (Baseline)':<20} | {avg_static_lat / avg_cont_lat:<20.2f}x faster")
    print(f"{'Effective GPU Utilization':<36} | {static_util:<20.1f}% | {cont_util:<20.1f}%")
    print("=" * 84)


if __name__ == "__main__":
    print_batching_comparison()
