# `rbrotli-enc` Optimization Status & Handoff TODO

## Workspace & Branch Reference
- **Git repository**: `git@github.com:google/rbrotli-enc.git`, branch `gemini-tests`
  - Pre-low-Q-overhaul baseline (`v9`/`v10` base): commit `8d2d35a`
  - Phase 1 implementation (`v11`): commit `e1ba46c`
- **Google3 / CitC integration**: client `rbrotli-pg2` (`//third_party/rbrotli/...` and `//util/compression/playground2:{benchmark,parallel_benchmark}`)
  - Directory mapping when syncing `rbrotli-enc` -> `//third_party/rbrotli`:
    - `bounded-utils/` -> `bounded_utils/`
    - `hugepage-buffer/` -> `hugepage_buffer/`
    - `lsb-bitwriter/` -> `lsb_bitwriter/`
    - `lib/` -> `lib/`, `clib/` -> `clib/`, `bin/` -> `bin/`

---

## Completed in Phase 1 (`v1..v11`, through `e1ba46c`)

1. **Coalesced 57-bit Bitwriter & Register-Spill Elimination**
   - `lsb-bitwriter/src/lib.rs`: `BitWriterCursor` supports writing up to 56/57 bits per `put` and coalesces 4 writes per buffer-bounds check via `put_quad` / `put_quad_foreach`.
   - Replaced non-diverging `#[inline(never)]` slow-path helpers inside hot loops with `#[inline(always)]` checks + diverging `assert!(...)`, eliminating caller-saved GPR and `%ymm` stack spills in `emit_bits_avx2` and `emit_sym_range_avx2`.
   - `lib/src/map_to_sym.rs`, `lib/src/emit.rs`, `lib/src/fast_onepass.rs`: unrolled 3-4 literals per `put` (`emit_literals_avx2`, `encode_symbols_avx2`) and combined IAC Huffman codes + extra bits + distance codes + extra bits into single `put` / `put_pair` / `put_triple` calls.

2. **`Q0` Single-Pass Overhaul (`lib/src/fast_onepass.rs`)**
   - Regime-classified literal Huffman tree presets (`LIT_LEN_WEB_CODE`, `LIT_LEN_PROSE_UTF8`, `LIT_LEN_ASCII_GENERAL`, `LIT_LEN_STRUCT_BIN`) chosen from the first ~2 KiB of chunk 0 and rolling histograms thereafter, with pre-baked metablock headers and `1 MiB`-periodic ICD table adaptation.
   - Dual-hash matchfinding (`8`-byte primary `ht` + `4`-byte `ht4` near table) plus single-probe RFC 7932 static dictionary (`static_dict::lookup_fast`) on the first `32 KiB` of the stream (Closing the small-payload MSS compression gap vs `zstd:1` and `cbrotli:0..1`).
   - Marked `chunk_avx2`, `chunk_scalar`, and `encode_one_chunk` `#[inline(never)]` to avoid code-layout bloat in `OnePassEncoder::run_chunks`.

3. **`Q1..Q3` Dual-Hash + Fast Static Dictionary (`lib/src/hashtable.rs`, `lib/src/static_dict.rs`, `lib/src/compress.rs`)**
   - `Q1..Q3` now use an `8`-byte primary hash table (`PRIMARY_HASH_LEN = 8`: 1-way `Q1`, 4-way `Q2`, 4/8-way `Q3`) with pure non-overlapping secondary hash tags, paired with a compact `32 KiB` (`8,192 x u32`) 1-way `ht4_near` `4`-byte table for near `4..7B` and unaligned `8B` matches.
   - Added `FastDictTable` (`64 KiB`, `static_dict::lookup_fast` with deferred bounds closure) gated by loop-invariant `pos < C::STATIC_DICT_LIMIT.saturating_sub(stream_base)` (`32 KiB` for `Q1`, `64 KiB` for `Q2..Q3`) so zero static-dictionary overhead is paid on large streams past the prefix limit.
   - Local hardware-counter A/B (`instructions:u` vs `v10`):
     - `Q1`: `-0.74%` size (`mss50`), `-0.81%` size & `-3.35%` insns (`corpus12`)
     - `Q2`: `-0.60%` size & `-6.83%` insns (`mss50`), `-0.63%` size & `-7.60%` insns (`corpus12`)
     - `Q3`: `-0.35%` size & `-2.54%` insns (`mss50`), `-0.32%` size & `-4.76%` insns (`corpus12`)
     - `Q4..Q9`: `0.00%` size change.

---

## Key Invariants & Guardrails (Do Not Break)
- **Zero `unsafe` in core library**: `#![deny(unsafe_code)]` in `lib/src/lib.rs` must remain intact.
- **`lsb-bitwriter` two-pass output buffer sizing**: `max_two_pass_output_size(input_len)` in `lib/src/compress.rs` reserves `input_len + (input_len >> 10) + 8192` bytes (`270,592` bytes for a `256 KiB` metablock). Do **not** replace the `safe_end_quad` split in `BitWriterCursor::put_quad_foreach` with an upfront `assert!(buf.len() >= bytes_written + iter.len() * 32 + 8)`, which would exceed `270,592` bytes on compressible `256 KiB` literal runs (`16,384 * 32 = 524,288`).
- **Streaming & multithreaded determinism**: `Q10..Q12` multithreaded compression must remain bit-for-bit identical to single-threaded (`stream::tests::test_streaming_q10_q11_q12_mt_matches_1t_bit_for_bit`).

---

## Remaining TODOs (Next Session on Cloudtop)

### 1. Perflab Benchmark Verification of Phase 1 (`v11`)
- [ ] In CitC (`//third_party/rbrotli`), run:
  - `blaze test //third_party/rbrotli/...`
  - `blaze build -c opt //util/compression/playground2:benchmark //util/compression/playground2:parallel_benchmark`
- [ ] Run Perflab head-to-head A/B (`v10` vs `v11`, alongside `cbrotli:0..11` and `zstd:1..19`) across Genoa, Milan, and Sapphire Rapids on:
  - `mss_50` and `mss_1000` (via `parallel_benchmark`)
  - `silesia`, `moz28`, and `corpus12` (via `benchmark`)
- [ ] Check `Q0..Q3` wall-clock speed/ratio Pareto curve on Perflab; if `Q0` trawl step / `ht4` skip gate (`lib/src/fast_onepass.rs:822`, `lib/src/compress.rs:2108`) wants minor tuning on wall-clock throughput, adjust and re-verify.

### 2. Phase 2A — Migrate `Q4..Q9` to Pure-Tag Dual-Hash Matchfinding (`lib/src/hashtable.rs`)
Currently `Q4..Q9` still use a `4`-byte primary key (`PRIMARY_HASH_LEN = 4`) with a byte-shifted secondary hash (`SEC_HASH_OFFSET = 1` or `2`), which:
- Shifts the secondary tag to bytes `1..5` / `2..6`, implicitly requiring `RECENT_4B_SLOTS` / `Buckets::recent_dist` workarounds to avoid losing `4`-byte matches at `SEC_HASH_OFFSET > 0`, and
- Requires a separate `2 MiB` `ht6` side table (`[[u32; 4]; 131_072]`) at `Q8..Q9` (`USE_6B_TABLE = true`) to find longer matches in deep buckets.

Planned cleanup & upgrade:
- [ ] Switch `Q4..Q9` primary N-way `HashTable` (`ht`) to `8`-byte primary hashing (`PRIMARY_HASH_LEN = 8`) with pure non-overlapping upper-bit secondary tags (`7`-bit tag in `Buckets<25, WAYS>` for standard windows `<= 16 MiB`, `16`-bit tag in `TagWays<WAYS>` for large windows), filtering same-bucket collisions before touching input memory.
- [ ] Pair `Q4..Q9` with compact short-match secondary tables so `3..7B` matches are preserved or improved:
  - Dedicated `4`-byte near/short table (`ht4_near`, e.g. `1`-way at `Q4..Q5` and `2..4`-way at `Q6..Q9`) for `4..7B` matches.
  - Keep the tiny `3`-byte table (`ht3`, `32K` entries) at `Q5..Q9` for very near `3`-byte matches, and **add match-tail `ht3` updates** (insert trailing positions of emitted matches into `ht3`) so back-to-back near `3B` matches after copies are caught.
- [ ] Delete `MatchConfig::SEC_HASH_OFFSET`, `secondary_hash32`, `MatchConfig::RECENT_4B_SLOTS`, `Buckets::recent_dist`, and the `2 MiB` `ht6` table (`USE_6B_TABLE`, `prefetch_ht6`, `step_ht6`), cutting `Q8..Q9` L3 cache footprint by `2 MiB`.
- [ ] Benchmark `Q4..Q9` on Perflab (`mss_50`, `mss_1000`, `silesia`, `moz28`, `corpus12`) to confirm faster matchfinding and equal-or-better compression ratios across all qualities.

### 3. Phase 2B — `lib/src/q11/` Code-Size & Complexity Audit
- [ ] Audit `lib/src/q11/{mod.rs, match_gen.rs, optimal_parse.rs}` for unnecessary complexity/LOC — in particular whether `dyadic_lit_min` (segment tree over literal costs) or auxiliary Zopfli structures in `optimal_parse.rs` can be replaced with simpler prefix-sum / sliding-window logic without hurting `Q10..Q12` speed or compression ratio.

### 4. Final Verification & Benchmark Plots
- [ ] Run full verification suite:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace --release`
  - `cargo +nightly miri test -p lsb-bitwriter -p bounded-utils -p rbrotli-enc-clib`
  - `blaze test //third_party/rbrotli/...`
- [ ] Regenerate all speed-vs-size Pareto plots with final Perflab numbers.
