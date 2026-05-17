# MegaMoE TEP support via DG kernel modification — design doc

**Status**: prototype-branch design only. Not yet implemented in code.
**Target kernel**: `tensorrt_llm/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` (and `scheduler/mega_moe.cuh`, `layout/mega_moe.cuh`).
**Companion**: PR https://github.com/NVIDIA/TensorRT-LLM/pull/14222 (the safety guard) is the prerequisite — it correctly rejects TEP today. This design replaces the rejection with a working code path.

## 1. Problem statement

`MegaMoEDeepGemm` is currently only usable in DEP topology. In TEP (`use_dp=False`, `parallel_size>1`, `ep_size==parallel_size`):

- MoE input is **TP-replicated** across all ranks (attention's `o_proj` AllReduce gives every rank the same full-hidden tensor).
- DG's `fp8_fp4_mega_moe` kernel does internal SymmBuffer all-to-all dispatch assuming each rank's input is a unique cluster-wide slice.
- The result: every rank pushes the same replicated tokens into its SymmBuffer, the dispatch warps pull `parallel_size` duplicate copies into each expert-owner's pool, and the expert compute runs `parallel_size×` on the duplicated rows. Empirically `−25.8%` vs TRTLLM at DSv4-Pro TEP8 c=128.

We want to **repurpose the existing dispatch + combine warps** so that:

- **Dispatch path** becomes a no-op AllGather (input is already replicated; just consume the local rank's slot).
- **Combine path** becomes an AllReduce-style cross-rank accumulation: each rank's local experts compute their partial contributions, then the combine warps broadcast-and-sum across ranks so every rank ends with the full reduced output.

No new CUDA kernels written; only the existing kernel's dispatch + combine warp roles change behavior under a new template flag.

## 2. Topology gating (recap)

| Topology | `use_dp` | `ep_size` vs `parallel_size` | Status today | After this design |
|---|---|---|---|---|
| Single rank | × | `==` (both 1) | Works | Works (no change) |
| DEP (attn-DP, EP=PP=World) | True | `==` | Works | Works (no change) |
| DEP > EP (ADP wider than EP) | True | `<` | Rejected at line 250 (assert) | Rejected (unchanged) — needs allgather-pre/RS-post wrapper, distinct effort |
| **TEP (no DP, parallel_size>1)** | False | `==` | Rejected by PR #14222 (NotImplementedError) | **NEW** — runs via `kTepMode` |

## 3. Architectural design

### 3.1 New template parameter

In `sm100_fp8_fp4_mega_moe.cuh` (kernel) and `scheduler/mega_moe.cuh`:

```cpp
template <
    uint32_t kNumMaxTokensPerRank,
    ...
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kTepMode = false,        // NEW
    ...>
```

When `kTepMode=false` the kernel behaves exactly as today (no perf regression, no behavioral change). When `kTepMode=true` the dispatch + combine paths switch to the AG/AR semantics below.

### 3.2 Dispatch path (lines ~356-649 of the kernel)

Current DEP behavior:
- Every rank pushes its unique tokens into SymmBuffer slots indexed by `(sm_buffer.rank_idx, dst_rank_idx, token_idx)`.
- Each rank's dispatch warps pull tokens from `kNumRanks` source ranks via `sym_buffer.map(remote_ptr, source_rank_idx)` (the cross-rank NVLink pull, lines 548-559).
- Each rank's local experts thus see all tokens cluster-wide that route to them.

TEP behavior (under `if constexpr (kTepMode)`):
- Input is already replicated. Each rank's local SymmBuffer slot already holds the full token batch.
- **Skip the cross-rank pull.** Each rank's dispatch warps read only from the LOCAL rank's slot — i.e. iterate `current_rank_in_expert_idx = sym_buffer.rank_idx` always, never any peer rank.
- Scheduler iteration shrinks from `kNumRanks × num_max_tokens_per_rank` slots down to `1 × num_max_tokens_per_rank` slots. This is the "AG no-op" semantics — the AG was already done implicitly by attention's `o_proj` AllReduce.

Concrete code-site edits in dispatch warps:
- Line 548-549 TMA load: when `kTepMode`, replace `current_rank_in_expert_idx` with `sym_buffer.rank_idx` so the TMA reads local memory. (Local NVLink TMA is still cheaper than `cudaMemcpy`-style reads from cross-rank slot — same arena.)
- Line 557-559 SF read: same substitution.
- Line 575-577 weights read: same substitution.
- Round-robin rank selection logic (lines 488-536) collapses: only one source rank to iterate, so the "min peeling" loop becomes a single-iter pass. Easiest to gate the entire loop with `if constexpr (kTepMode)` and have a TEP-specific tight loop that just maps `token_idx → token_idx_in_rank` directly without searching across ranks.

Estimated lines changed in dispatch: ~80 LoC of `if constexpr` branches (some new tight loops, some bypass of the round-robin).

### 3.3 Compute path

**No changes.** Each rank's local experts (`kNumExpertsPerRank = kNumExperts / kNumRanks`) run their GEMM1 + SwiGLU + GEMM2 chain on whatever the pool contains. The scheduler's iteration count changes (fewer per-expert tokens in TEP because each rank only sees tokens that route to its experts via local routing), but the per-expert math is identical.

### 3.4 Combine path — the hardest part (lines ~1100-1351)

Current DEP behavior:
- Phase 1 (line 1112-1197): for each token-topk pair, the L2 epilogue pushes the GEMM output to the **origin rank's slot** via `sym_buffer.map(combine_token_buffer.get_rank_buffer(dst_topk_idx).get_data_buffer(dst_token), origin_rank)`.
- Phase 2 (line 1255-1351): each rank's combine-reduction epilogue reads its own `combine_token_buffer` slots from peer ranks (the topk contributions from different expert-owners) and sums them locally to produce the final output.

The `combine_token_buffer` has shape `(kNumTopk, kNumMaxTokensPerRank)` per rank. Each rank in DEP has its own unique `kNumMaxTokensPerRank` tokens; the topk dimension stores contributions from up-to-`kNumTopk` different expert-owner ranks per token.

TEP behavior under `if constexpr (kTepMode)`:

Two viable designs (pick one — leaning toward Design A as simpler):

**Design A — broadcast-write + local sum-reduce ("AG style")**

- Phase 1 (push): each rank's L2 epilogue writes its partial-expert contribution to ITS OWN local `combine_token_buffer[topk_slot][token_idx]` slot. No cross-rank write. The "topk slot" assignment becomes the global expert-rank index `i = expert_idx / kNumExpertsPerRank` (0..kNumRanks-1).
- NVLink barrier (reuse `kBeforeCombineReduceBarrierTag` — already exists in the kernel).
- Phase 2 (read + sum): each rank's combine-reduction epilogue iterates `for src_rank in 0..kNumRanks: out += sym_buffer.map(combine_token_buffer[token_idx], src_rank)`. The existing reduction loop already iterates over `kNumTopk` slots; in TEP mode it iterates over `kNumRanks` slots instead. Same TMA-load + sum-reduce structure.
- After phase 2, every rank's local `out` holds the full sum across all rank-owners.
- The existing reduction code already does this *pattern* (just over `kNumTopk` instead of `kNumRanks`) — easiest substitution.

Memory: `kNumTopk × kNumMaxTokensPerRank` slots reused for `kNumRanks × kNumMaxTokensPerRank`. If `kNumRanks <= kNumTopk` (e.g. 8 <= 8 for DSv4-Pro top-k=6), this is tight; if `kNumRanks > kNumTopk` we'd need a slightly bigger buffer. For DSv4-Pro top-k=6, kNumRanks=8, kNumTopk=6, we'd need `max(kNumTopk, kNumRanks) = 8` slots — a small buffer enlargement under `kTepMode`. Update `combine_token_buffer` shape to `max(kNumTopk, kNumRanks)`.

Concrete code-site edits:
- Line 1197 — the per-topk-slot push: in `kTepMode`, replace `dst_topk_idx` with `expert_idx / kNumExpertsPerRank` (the global expert-owner-rank index), and the `origin_rank` write target with `sym_buffer.rank_idx` (write to LOCAL).
- Line 1255-1351 — the read-and-sum loop: in `kTepMode`, change the outer iteration from `i in 0..kNumTopk` to `i in 0..kNumRanks`, and source from `sym_buffer.map(combine_token_buffer[i][token_idx], i)` — i.e. read peer rank `i`'s slot via NVLink.
- Empty-slot handling: if a token didn't route to any expert on a particular rank, the partial is zero. The push site needs a clear-slot init (already done via `combine_token_buffer` zero-init at the workspace setup), so zero contributions are safe.

**Design B — atomic-add broadcast (true in-kernel AllReduce, no two-phase)**

- Each rank's combine warps atomic-add their partial contribution to ALL ranks' output slot via `ptx::atomic_add_sys`. After all ranks finish + NVLink barrier, every rank's output buffer holds the sum.
- Issue: NVLink HW atomic-add doesn't natively support bf16. Workaround: accumulate in fp32 (separate per-rank fp32 accumulator), then a final fp32→bf16 conversion pass at the end.
- More invasive than Design A; introduces an fp32 accumulator buffer + extra type-conversion pass. Skip for now.

→ **Design A is the chosen path** (less new memory, reuses existing reduction structure).

Estimated lines changed in combine: ~100 LoC of `if constexpr` branches in phases 1 and 2.

### 3.5 Layout/Workspace edits (`layout/mega_moe.cuh`)

- `combine_token_buffer` shape: `max(kNumTopk, kNumRanks) × kNumMaxTokensPerRank` instead of `kNumTopk × kNumMaxTokensPerRank`. (Or template the shape per `kTepMode` if we want to keep DEP's footprint unchanged.)
- `num_max_recv_tokens_per_expert`: in `kTepMode`, this becomes `num_max_tokens_per_rank` (×1, not ×`num_ranks`). Update `get_num_max_pool_tokens` accordingly:

```cpp
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(
    T num_ranks, T num_max_tokens_per_rank, T num_topk, T num_experts_per_rank, T block_m,
    bool tep_mode = false) {
    const auto num_max_recv_tokens = tep_mode ? num_max_tokens_per_rank
                                              : num_ranks * num_max_tokens_per_rank;
    ...
}
```

### 3.6 Python wiring (`mega_moe_deepgemm.py`)

Currently, after PR #14222, `__init__` raises `NotImplementedError` for TEP. The change:

```python
elif (not self.use_dp) and self.parallel_size > 1:
    # NEW: enable TEP path via kTepMode kernel template
    self._tep_mode = True
    # Buffer must hold max(kNumTopk, parallel_size) combine slots per token.
    # See MEGAMOE_TEP_KERNEL_DESIGN.md §3.5.
```

In `run_moe`, pass `tep_mode=self._tep_mode` through to `dg.fp8_fp4_mega_moe(...)`:

```python
dg.fp8_fp4_mega_moe(
    y, self._t_l1, self._t_l2, buf,
    activation=self.activation,
    activation_clamp=self.swiglu_limit_scalar,
    fast_math=self.fast_math,
    tep_mode=self._tep_mode,   # NEW
)
```

The DG Python binding (`csrc/jit_kernels/heuristics/mega_moe.hpp`) needs the new param plumbed through to the template instantiation. A new JIT-template entry per `kTepMode` value.

### 3.7 What stays the same

- All TMA descriptors (input, L1/L2 weights, L1/L2 activations, output).
- All MMA pipeline stages + the SwiGLU activation function.
- All barrier counts (the NVLink barriers and grid syncs are reusable as-is since `kTepMode` flow has the same phase structure).
- Scheduler's `Linear1 → Linear2` block-phase machinery.

## 4. Estimated effort

| Sub-task | Effort | Risk |
|---|---|---|
| Add `kTepMode` template + JIT plumbing | 0.5 day | low |
| Scheduler iteration changes | 0.5 day | low |
| Dispatch warp short-circuit | 1 day | medium (must match existing barrier semantics) |
| Combine path Design A | 2-3 days | high (reading the existing combine code carefully + new local-buffer indexing) |
| Layout/workspace shape changes | 0.5 day | low |
| Python wire-up + `tep_mode` kwarg | 0.5 day | low |
| Build + JIT cubin instantiation | 0.5 day | medium (CUDA template explosions, slow NVRTC) |
| Correctness validation (bit-exact vs TRTLLM backend on a small fixture) | 1 day | high — this is where bugs surface |
| Perf bench at TEP8 c=128 | 0.5 day | low |
| Total | ~7-10 days focused | — |

## 5. Verification plan

1. **Numerical correctness**: small fixture (TP=2, 8 tokens, 4 experts, top-k=2, hidden=512) where you can dump full output from both TRTLLM backend and MEGAMOE+kTepMode, bit-compare. Use a fixed RNG seed and `CUDA_LAUNCH_BLOCKING=1`.
2. **DEP regression**: rerun DSv4-Pro DEP8 c=512 mtp0 with `MEGAMOE_DEEPGEMM` — must still produce ~33752 sys tok/s. If broken, the `kTepMode=false` default path got compromised.
3. **TEP perf**: DSv4-Pro TEP8 c=128 mtp0 with `MEGAMOE_DEEPGEMM` should now run (no error) and produce ≥ TRTLLM throughput (16k tok/s). Stretch target: match DEP's `+2.66%` margin vs TRTLLM.
4. **Multi-config sweep**: rerun the 13-row MTP0 pareto sweep with `MEGAMOE_DEEPGEMM` (including TEP rows) to confirm no config regresses vs TRTLLM.

## 6. Upstream PR strategy

This is a real DG-upstream change. Plan:

1. Prototype on `deepseek-ai/DeepGEMM` fork.
2. Sync `tensorrt_llm/deep_gemm/include/deep_gemm/*.cuh` from the DG fork.
3. Bench in TRT-LLM tree.
4. Once validated, open PR against `deepseek-ai/DeepGEMM` upstream.
5. Once that lands, open follow-up PR on `NVIDIA/TensorRT-LLM` to bump the DG fetch revision in `3rdparty/fetch_content.json` and flip the TEP guard in `mega_moe_deepgemm.py` from `raise NotImplementedError` to `tep_mode=True`.

## 7. Why not implement now

This session has limited capacity (~couple of hours of focused work). A partial implementation that compiles but has wrong combine semantics would be worse than no implementation — it'd produce silently-wrong outputs that pass smoke tests but fail at scale. Best to land:

1. PR #14222 (the safety guard) — DONE.
2. This design doc — committed on prototype branch `prototype/megamoe-tep-kernel-mod`.
3. Future session (or a dedicated DG-team engineer) picks up the design and implements the body.

The dispatch path is the easier 30% of the work; combine path Design A is the harder 70% and requires uninterrupted focus + correctness fixture before claiming it works.
