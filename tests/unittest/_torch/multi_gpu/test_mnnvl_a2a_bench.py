#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal correctness test and benchmark for MNNVL One-Sided and Two-Sided
All2All communication kernels (EP4).

Run via mpirun:
    mpirun --allow-run-as-root -n 4 python this_script.py
"""

import sys
import time
import traceback

import torch
from mpi4py import MPI

import tensorrt_llm as tllm
from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.distributed import MoeAlltoAll
from tensorrt_llm.mapping import Mapping

tllm.logger.set_level('error')

# ---------------------------------------------------------------------------
EP_SIZE = 4
NUM_EXPERTS = 32
HIDDEN_SIZE = 2880
WORKSPACE_MB = 512
WARMUP_ITERS = 3
BENCH_ITERS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_token_selected_experts(local_num_tokens, num_experts, top_k):
    return torch.randint(0, num_experts, (local_num_tokens, top_k),
                         dtype=torch.int32, device='cuda')


def create_experts_per_rank(num_experts_per_rank, hidden_size, ep_rank, device,
                            dtype=torch.bfloat16):
    experts = torch.empty((num_experts_per_rank, hidden_size, hidden_size),
                          dtype=dtype, device=device)
    for i in range(num_experts_per_rank):
        torch.manual_seed(ep_rank * 1000 + i)
        torch.nn.init.xavier_uniform_(experts[i])
    return experts


def fake_moe(hidden_states, token_selected_experts, token_final_scales,
             experts, is_ep=False, ep_rank=None, num_experts_per_rank=None):
    num_tokens, _ = hidden_states.shape
    _, top_k = token_selected_experts.shape
    processed = torch.zeros_like(hidden_states)
    for t in range(num_tokens):
        for k in range(top_k):
            eid = token_selected_experts[t, k].item()
            if is_ep:
                if not (ep_rank * num_experts_per_rank <= eid <
                        (ep_rank + 1) * num_experts_per_rank):
                    continue
                local_eid = eid - ep_rank * num_experts_per_rank
                expert = experts[local_eid]
            else:
                expert = experts[eid]
            scale = token_final_scales[t, k]
            processed[t] += hidden_states[t] @ expert * scale
    return processed


def make_bfloat16_payloads(local_num_tokens, hidden_size, top_k, rank,
                           token_selected_experts):
    payloads = []
    hs = torch.randn(local_num_tokens, hidden_size, dtype=torch.bfloat16,
                     device='cuda') + rank
    payloads.append(hs)
    payloads.append(token_selected_experts)
    tfs = torch.rand(local_num_tokens, top_k, dtype=torch.bfloat16,
                     device='cuda')
    payloads.append(tfs)
    return payloads, 1  # expert_id_payload_index = 1


# ---------------------------------------------------------------------------
# One-Sided test
# ---------------------------------------------------------------------------
MAX_NUM_TOKENS = 512  # Fixed for workspace singleton

def run_one_sided(rank, ep_size, num_tokens, top_k, hidden_size, num_experts):
    mapping = Mapping(rank=rank, tp_size=ep_size, moe_ep_size=ep_size,
                      world_size=ep_size)
    max_num_tokens = MAX_NUM_TOKENS
    workspace_size = WORKSPACE_MB * 1024 * 1024
    num_experts_per_rank = num_experts // ep_size

    # Workspace is a singleton keyed by max_num_tokens; top_k is per-instance
    moe_a2a = MoeAlltoAll(mapping=mapping, max_num_tokens=max_num_tokens,
                          top_k=top_k, num_slots=num_experts,
                          workspace_size_per_rank=workspace_size)

    torch.manual_seed(0x1234 + rank)
    tse = generate_token_selected_experts(num_tokens, num_experts, top_k)
    payloads, eid_idx = make_bfloat16_payloads(num_tokens, hidden_size,
                                               top_k, rank, tse)
    experts = create_experts_per_rank(num_experts_per_rank, hidden_size,
                                      rank, 'cuda')

    # -- Correctness round --
    recv = moe_a2a.dispatch(tse, payloads, max_num_tokens,
                            invalid_token_expert_id=-1,
                            expert_id_payload_index=eid_idx)
    hs_r, tse_r, tfs_r = recv[0], recv[1], recv[2]
    flat_hs = hs_r.view(ep_size * max_num_tokens, hs_r.shape[-1])
    flat_tse = tse_r.view(ep_size * max_num_tokens, tse_r.shape[-1])
    flat_tfs = tfs_r.view(ep_size * max_num_tokens, tfs_r.shape[-1])
    moe_out = fake_moe(flat_hs, flat_tse, flat_tfs, experts, is_ep=True,
                       ep_rank=rank,
                       num_experts_per_rank=num_experts_per_rank)
    moe_out_3d = moe_out.view(ep_size, max_num_tokens, hs_r.shape[-1])
    combined = moe_a2a.combine(moe_out_3d, max_num_tokens)

    # -- Benchmark --
    dummy_moe_out = moe_out_3d.clone()
    for _ in range(WARMUP_ITERS):
        r = moe_a2a.dispatch(tse, payloads, max_num_tokens,
                             invalid_token_expert_id=-1,
                             expert_id_payload_index=eid_idx)
        moe_a2a.combine(dummy_moe_out, max_num_tokens)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(BENCH_ITERS):
        r = moe_a2a.dispatch(tse, payloads, max_num_tokens,
                             invalid_token_expert_id=-1,
                             expert_id_payload_index=eid_idx)
        moe_a2a.combine(dummy_moe_out, max_num_tokens)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / BENCH_ITERS

    # Gather for correctness check
    return combined, tse, payloads, experts, ms


# ---------------------------------------------------------------------------
# Two-Sided test
# ---------------------------------------------------------------------------
def run_two_sided(rank, ep_size, num_tokens, top_k, hidden_size, num_experts):
    from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_two_sided import NVLinkTwoSided

    mapping = Mapping(rank=rank, tp_size=ep_size, moe_ep_size=ep_size,
                      world_size=ep_size)
    num_experts_per_rank = num_experts // ep_size
    all_rank_num_tokens = [num_tokens] * ep_size

    comm = NVLinkTwoSided(mapping=mapping, num_experts=num_experts,
                          num_slots=num_experts, top_k=top_k,
                          alltoall_result_do_sum=True)

    torch.manual_seed(0x1234 + rank)
    tse = generate_token_selected_experts(num_tokens, num_experts, top_k)
    hs = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16,
                     device='cuda') + rank
    tfs = torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device='cuda')
    experts = create_experts_per_rank(num_experts_per_rank, hidden_size,
                                      rank, 'cuda')

    # -- Correctness --
    comm.prepare_dispatch(tse, all_rank_num_tokens)
    hs_recv, _, tse_recv, tfs_recv = comm.dispatch(
        hs, None, tse, tfs, all_rank_num_tokens)
    moe_out = fake_moe(hs_recv, tse_recv, tfs_recv, experts,
                       is_ep=True, ep_rank=rank,
                       num_experts_per_rank=num_experts_per_rank)
    combined = comm.combine(moe_out)

    # -- Benchmark --
    dummy_out = moe_out.clone()
    for _ in range(WARMUP_ITERS):
        comm.prepare_dispatch(tse, all_rank_num_tokens)
        comm.dispatch(hs, None, tse, tfs, all_rank_num_tokens)
        comm.combine(dummy_out)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(BENCH_ITERS):
        comm.prepare_dispatch(tse, all_rank_num_tokens)
        comm.dispatch(hs, None, tse, tfs, all_rank_num_tokens)
        comm.combine(dummy_out)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / BENCH_ITERS

    return combined, tse, hs, tfs, experts, ms


# ---------------------------------------------------------------------------
# Verification (runs on all ranks, uses MPI allgather)
# ---------------------------------------------------------------------------
def verify_one_sided(comm_mpi, rank, ep_size, combined, tse, payloads, experts):
    all_tse = comm_mpi.allgather(tse.cpu())
    all_hs = comm_mpi.allgather(payloads[0].cpu())
    all_tfs = comm_mpi.allgather(payloads[2].cpu())
    all_experts = comm_mpi.allgather(experts.cpu())
    all_experts_cat = torch.cat(all_experts, dim=0)

    expected = fake_moe(all_hs[rank], all_tse[rank], all_tfs[rank],
                        all_experts_cat, is_ep=False)
    try:
        torch.testing.assert_close(combined.cpu(), expected, rtol=0.1, atol=0.5)
        return True
    except AssertionError as e:
        if rank == 0:
            print(f"  FAIL rank {rank}: {e}", flush=True)
        return False


def verify_two_sided(comm_mpi, rank, ep_size, combined, tse, hs, tfs, experts):
    all_tse = comm_mpi.allgather(tse.cpu())
    all_hs = comm_mpi.allgather(hs.cpu())
    all_tfs = comm_mpi.allgather(tfs.cpu())
    all_experts = comm_mpi.allgather(experts.cpu())
    all_experts_cat = torch.cat(all_experts, dim=0)

    expected = fake_moe(all_hs[rank], all_tse[rank], all_tfs[rank],
                        all_experts_cat, is_ep=False)
    try:
        torch.testing.assert_close(combined.cpu(), expected, rtol=0.1, atol=0.5)
        return True
    except AssertionError as e:
        if rank == 0:
            print(f"  FAIL rank {rank}: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    size = comm_mpi.Get_size()
    assert size == EP_SIZE, f"Must run with {EP_SIZE} ranks, got {size}"

    torch.cuda.set_device(rank)
    torch.manual_seed(0x1234)

    MnnvlMemory.initialize()
    if not MnnvlMemory.supports_mnnvl():
        if rank == 0:
            print("SKIP: MNNVL not supported", flush=True)
        return

    configs = [
        (32, 2),
        (128, 2),
        (32, 4),
        (128, 4),
    ]

    if rank == 0:
        print(f"\n{'='*70}", flush=True)
        print(f"MNNVL All2All Benchmark  EP={EP_SIZE}  hidden={HIDDEN_SIZE}  "
              f"experts={NUM_EXPERTS}", flush=True)
        print(f"warmup={WARMUP_ITERS}  bench_iters={BENCH_ITERS}", flush=True)
        print(f"{'='*70}", flush=True)

    # ---- One-Sided ----
    if rank == 0:
        print(f"\n--- One-Sided All2All ---", flush=True)
    for num_tokens, top_k in configs:
        comm_mpi.Barrier()
        try:
            combined, tse, payloads, experts, ms = run_one_sided(
                rank, EP_SIZE, num_tokens, top_k, HIDDEN_SIZE, NUM_EXPERTS)
            ok = verify_one_sided(comm_mpi, rank, EP_SIZE, combined, tse,
                                  payloads, experts)
            all_ms = comm_mpi.gather(ms, root=0)
            all_ok = comm_mpi.gather(ok, root=0)
            if rank == 0:
                avg_ms = sum(all_ms) / len(all_ms)
                passed = all(all_ok)
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] tokens/rank={num_tokens:4d}  top_k={top_k}  "
                      f"avg={avg_ms:8.3f} ms  per_rank={[f'{m:.3f}' for m in all_ms]}",
                      flush=True)
        except Exception as e:
            if rank == 0:
                print(f"  [ERROR] tokens/rank={num_tokens} top_k={top_k}: {e}",
                      flush=True)
                traceback.print_exc()

    # ---- Two-Sided ----
    if rank == 0:
        print(f"\n--- Two-Sided All2All ---", flush=True)
    for num_tokens, top_k in configs:
        comm_mpi.Barrier()
        try:
            combined, tse, hs, tfs, experts, ms = run_two_sided(
                rank, EP_SIZE, num_tokens, top_k, HIDDEN_SIZE, NUM_EXPERTS)
            ok = verify_two_sided(comm_mpi, rank, EP_SIZE, combined, tse,
                                  hs, tfs, experts)
            all_ms = comm_mpi.gather(ms, root=0)
            all_ok = comm_mpi.gather(ok, root=0)
            if rank == 0:
                avg_ms = sum(all_ms) / len(all_ms)
                passed = all(all_ok)
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] tokens/rank={num_tokens:4d}  top_k={top_k}  "
                      f"avg={avg_ms:8.3f} ms  per_rank={[f'{m:.3f}' for m in all_ms]}",
                      flush=True)
        except Exception as e:
            if rank == 0:
                print(f"  [ERROR] tokens/rank={num_tokens} top_k={top_k}: {e}",
                      flush=True)
                traceback.print_exc()

    if rank == 0:
        print(f"\n{'='*70}", flush=True)
        print("Done.", flush=True)


if __name__ == '__main__':
    main()
