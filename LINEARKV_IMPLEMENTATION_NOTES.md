# LinearKV implementation notes

This branch adds the paper-based `linearkv` PIC mode. The algorithmic source
is [LinearKV: One Cached State Suffices for Position-Independent Caching in
Hybrid LLMs](https://arxiv.org/html/2608.11231); no official LinearKV runtime
implementation is used.

## Code-level split

- `python/sglang/srt/pic/policy.py` adds `PICStateInit.LAST_BLOCK` and the
  `linearkv` policy. Existing `addition`, `transition`, `transition_rope`, and
  `transition_rope_recompute` policy values are unchanged.
- `python/sglang/srt/pic/selector.py` is a selector adapter. The first
  selector is a static EPIC/LegoLink-style selector: it selects a prefix of
  every matched chunk, configured either by a fixed token count or by a
  per-chunk ratio (the KVBench LinearKV adapter defaults to 20%). It returns
  absolute prompt positions, so the selector can later be replaced by
  CacheBlend or ProphetKV without changing the recurrent backend.
- `schedule_batch.py`, `forward_batch_info.py`, and `pic_alloc.py` carry those
  positions through the existing PIC input packing, slot allocation, and RoPE
  metadata paths. `ForwardBatch` keeps scheduler/cache accounting based on the
  logical miss length, but gives LinearKV's GDN/conv kernels the physical
  packed online length (miss rows plus selected hit rows); this distinction is
  required when selected hit rows remain in the FA prefix mapping.
- `hybrid_linear_attn_backend.py` dispatches `linearkv` before the HYPIC
  addition/transition families. Full Attention reuses HYPIC's public/private
  KV layout and position-aware RoPE correction. Its repair rows are the
  selector rows, not HYPIC's fixed seam rows.
- `gdn_backend.py` uses a direct rank-local copy of the last matched
  segment's cached temporal state, then invokes the existing fused GDN
  prefill kernel on only the selected hit tokens and ordinary miss tokens in
  absolute prompt order. The packed cu-seqlens now match that replay stream.

## What is deliberately not used

LinearKV does not call HYPIC `init_pic_metadata`, transition-state
construction, affine transition composition, or HYPIC's staged recurrent
recompute path. The `pic_alloc_transition_rope` function name is retained
because it is the existing position-aware KV slot allocator; in `linearkv` it
allocates/repositions KV and mamba slots only and does not allocate or compute
`T_chunk`.

The transition buffer remains conditional on
`POLICIES[pic_mode].compose is PICCompose.TRANSITION`; `linearkv` uses the
non-transition policy value and therefore does not allocate it.

## Cache construction and recurrent initialization

An independent one-segment prefill publishes its Full Attention public KV and
its final GDN state into the reusable segment entry. A multi-segment online
request can use miss slots transiently, but its joint recurrent state is not
published as a reusable local state. This prevents an online composed prefix
from being mistaken for an independently warmed chunk.

For matched chunks `C1, ..., CK`, initialization chooses the matched chunk
with the greatest prompt start position and copies that entry's rank-local
temporal state directly into the active request state. It does not sum states
and does not compose transitions. The replay stream is the sorted union of
all miss tokens and selector-selected hit tokens.

## Causal convolution decision

Qwen GDN has runtime causal-convolution history in addition to the temporal
recurrent state. The paper defines the initializer at the recurrent-state
level and does not specify the Qwen-specific conv-tail boundary protocol.
LinearKV therefore restores the cached state as the same-block pair
`(temporal_state, conv_tail)`, then feeds the selected replay stream through
the ordinary fused Qwen GDN kernel. It does not chain conv tails across all
matched segments and does not add HYPIC's causal-convolution warm-up. Copying
the paired tail preserves a valid Qwen runtime state; the exact behavior at the
first replay token is an explicit implementation decision and remains a known
ambiguity requiring model-level validation.

## TP behavior

The initializer copies each rank's local temporal and convolution tensors from
that rank's own mamba cache slot. Full Attention KV continues to use the
existing tensor-parallel KV pool and RoPE path. No gather-to-rank-0, CPU
composition, or new collective is introduced; selected replay uses the normal
per-rank fused GDN path. TP=1 was exercised on the available GPU; TP=2 still
needs to be run on a two-GPU host.

## Validation completed

- Policy/selector diagnostics pass, including the no-transition policy check
  and prompt-order selector ordering check.
- Targeted Python compilation and `git diff --check` pass.
- Qwen3.8-27B loaded successfully with `--pic-mode linearkv` on one GPU.
- Independent warmups for two reusable segments completed.
- Two-hit request: `cached_tokens=25`.
- Reordered two-hit request: `cached_tokens=25`.
- Partial-hit request: `cached_tokens=13`.
- All-miss request: `cached_tokens=0`.
- These requests exercised actual generation and returned successfully.
- KVBench is wired through `LinearKVMethod`; `config.yaml` is the model-path
  source of truth (no `KVBENCH_MODEL_PATH` override is used).
- Before the metadata fix, HotpotQA sample 64 reached only F1 `0.0109`.
  Diagnosis showed 374 packed replay rows paired with a 54-token GDN
  `query_start_loc`.
- After the fix, HotpotQA sample 64 on one GPU, with a 32-token/chunk static
  selector budget, reached `reuse_ratio=0.9618`, mean TTFT `0.171s`, F1
  `0.4985`, and exact accuracy `0.3906`.
- The one-sample full-prefill control scored F1/EM `1.0`; the corrected hit
  path also scored `1.0` on the corresponding one-sample diagnostic.
- `python -m pytest -q test/registered/unit/pic` passed (2 tests), and
  KVBench's LinearKV adapter tests passed (3 tests). A TP=2 smoke test is not
  possible with only one visible GPU.

## Current limitations and next step

The selector is intentionally the simple static EPIC/LegoLink-style
approximation, not a full learned or production EPIC scorer. Qwen causal-conv
history remains the main model-specific ambiguity: the implementation restores
the same-block tail and treats the packed replay stream as the online causal
stream, which matches the fused runtime contract but is not explicitly defined
by the paper.
Independent cache population is currently enforced as one reusable segment per
warmup request; KVBench's natural segments can still be inserted through the
same cache entry interface. The next integration step is a KVBench adapter
that performs explicit offline independent warmups, records cache-build cost
separately, and submits online requests with the `linearkv` mode and selector
configuration.
