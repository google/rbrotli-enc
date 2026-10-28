// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Subquadratic $O(N \log N)$ optimal Brotli encoder (`Q10`..=`Q12`).

mod match_gen;
mod optimal_parse;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use lsb_bitwriter::BitWriter;

use crate::block_split::{build_metablock_plan_with_effort, MetablockPlan};
use crate::constants::{
    window_size, HASH_MUL32, INITIAL_LAST_DISTANCES, Q11_MAX_METABLOCK_SIZE, STD_MAX_WBITS,
};
use crate::emit::{should_emit_uncompressed_plan, write_compressed_metablock};
use crate::hashtable::PreparedDictionary;
use crate::map_to_sym::copy_code_only;
use crate::metablock::{
    write_empty_last_metablock, write_uncompressed_metablock, write_window_bits, OptimalCommand,
    DIST_MODE_IMPLICIT_D0, DIST_MODE_RAW, DIST_MODE_STATIC_DICT,
};

use match_gen::{
    build_chunk_matches, ChunkLdmMatches, ChunkMatches, DictChain, MatchGenCfg, MatchGenScratch,
};
use optimal_parse::{
    advance_dist_rb, parse_chunk_greedy_seed, parse_chunk_optimal, resolve_short_dist_code,
    ChunkCostModel, ChunkParseResult, OptimalParseScratch,
};

const RFC_MAX_METABLOCK_SIZE: usize = 1 << 24; // 16 MiB RFC 7932 ceiling

#[derive(Clone, Copy)]
pub(super) struct ChunkEnv<'a> {
    pub(super) data: &'a [u8],
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) wsize: usize,
    pub(super) dict: &'a [u8],
}

impl<'a> ChunkEnv<'a> {
    #[inline(always)]
    pub(super) fn new(d: &'a [u8], s: usize, e: usize, w: usize, dict: &'a [u8]) -> Self {
        Self {
            data: d,
            start: s,
            end: e,
            wsize: w,
            dict,
        }
    }

    #[inline(always)]
    pub(super) fn len(&self) -> usize {
        self.end - self.start
    }

    #[inline(always)]
    pub(super) fn slice(&self) -> &'a [u8] {
        &self.data[self.start..self.end]
    }

    #[inline(always)]
    pub(super) fn sub(&self, start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            ..*self
        }
    }

    #[inline(always)]
    pub(super) fn max_lz_dist(&self, sp: usize) -> u32 {
        ((sp + self.dict.len()).min(self.wsize)) as u32
    }

    #[inline(always)]
    pub(super) fn has_2b(&self, sp: usize, dist: u32, b0: u8, b1: u8) -> bool {
        let du = dist as usize;
        if du <= sp {
            let c = sp - du;
            self.data[c] == b0 && self.data[c + 1] == b1
        } else {
            let rem = du - sp;
            (2..=self.dict.len()).contains(&rem) && {
                let o = self.dict.len() - rem;
                self.dict[o] == b0 && self.dict[o + 1] == b1
            }
        }
    }
}

const DP_CHUNK_SIZE: usize = if cfg!(all(test, debug_assertions)) {
    8 << 10
} else {
    1 << 20
};
const DP_TAIL_MERGE: usize = if cfg!(all(test, debug_assertions)) {
    2 << 10
} else {
    1 << 18
};

#[derive(Default)]
struct ParallelPools {
    mg_workers: Vec<MatchGenScratch>,
    dp_workers: Vec<(OptimalParseScratch, ChunkCostModel)>,
    wave_cm: Vec<ChunkMatches>,
    wave_ldm: Vec<ChunkLdmMatches>,
}

pub(crate) struct Q11Encoder {
    effort: u8,
    wbits: usize,
    large_window: bool,
    num_threads: usize,
    mg_scratch: MatchGenScratch,
    dp_scratch: (OptimalParseScratch, ChunkCostModel),
    pools: ParallelPools,
}

impl Q11Encoder {
    pub(crate) fn new(quality: u32, wbits: usize, num_threads: usize) -> Self {
        let effort = quality.clamp(10, 12) as u8;
        let large_window = wbits > STD_MAX_WBITS;
        if effort >= 11 && !large_window {
            crate::static_dict::warm_dict121();
        }
        Self {
            effort,
            wbits,
            large_window,
            num_threads,
            mg_scratch: MatchGenScratch::default(),
            dp_scratch: Default::default(),
            pools: ParallelPools::default(),
        }
    }

    #[inline]
    fn effective_threads(&self) -> usize {
        if cfg!(target_family = "wasm") {
            return 1;
        }
        let n = if self.num_threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            self.num_threads
        };
        n.clamp(1, 256)
    }

    #[inline]
    fn sa_window_cap(&self) -> usize {
        if self.effort >= 12 {
            3 << 20
        } else if self.effort == 11 && self.large_window {
            2 << 20
        } else {
            1 << 20
        }
    }
}

fn push_chopped_range(ranges: &mut Vec<(usize, usize)>, mut pos: usize, end: usize) {
    while pos < end {
        let rem = end - pos;
        let step = if rem <= DP_CHUNK_SIZE + DP_TAIL_MERGE {
            rem
        } else {
            DP_CHUNK_SIZE
        };
        ranges.push((pos, pos + step));
        pos += step;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(test, derive(Debug))]
enum PreChunkKind {
    Normal,
    PureLiterals,
    UncompressedRaw,
}

/// Fast pre-H10 classifier for zero-LZ-repeat chunks (`>= 16 KiB`): distinguishes `Normal`,
/// `PureLiterals` (`H < 7.97 bpb`, e.g. Base58/Base64 IID tokens), and `UncompressedRaw`.
fn classify_chunk(env: ChunkEnv<'_>, large_window: bool) -> PreChunkKind {
    let chunk_len = env.len();
    if chunk_len < 16_384 {
        return PreChunkKind::Normal;
    }
    let (slice, span) = (env.slice(), chunk_len - 2048);
    let wins: [&[u8]; 8] = std::array::from_fn(|w| &slice[(w * span) / 7..][..2048]);
    let (mut table4, mut hits_4b) = ([u64::MAX; 512], 0u32);
    for win in wins {
        for s in win.windows(4) {
            let w4 = u32::from_le_bytes(s.try_into().unwrap());
            let slot = &mut table4[(w4.wrapping_mul(HASH_MUL32) >> 23) as usize];
            if *slot == w4 as u64 {
                hits_4b += 1;
                if hits_4b > 4 {
                    return PreChunkKind::Normal;
                }
            } else {
                *slot = w4 as u64;
            }
        }
    }

    let (mut hist, mut table2, mut hits_2b) = ([0u32; 256], [u64::MAX; 1024], 0u32);
    for (w, win) in wins.into_iter().enumerate() {
        for &b in win {
            hist[b as usize] += 1;
        }
        for (p, s) in win.windows(2).enumerate() {
            let w2 = u16::from_le_bytes([s[0], s[1]]) as u64;
            let cur_pos = (w * 8192 + p) as u64;
            let slot = &mut table2[((w2.wrapping_mul(0x9E37)) >> 6) as usize & 1023];
            if (*slot >> 32) == w2 && cur_pos - (*slot & 0xFFFF_FFFF) <= 64 {
                hits_2b += 1;
            }
            *slot = (w2 << 32) | cur_pos;
        }
    }

    let (mut sum_sq, mut sum_xlog2x, mut ascii_lower) = (0u64, 0u64, 0u32);
    for (b, &c) in hist.iter().enumerate().filter(|&(_, &c)| c > 0) {
        sum_sq += (c as u64) * (c as u64);
        sum_xlog2x += crate::context::xlog2x_fp16(c);
        if (b as u8).is_ascii_lowercase() || b == b' ' as usize {
            ascii_lower += c;
        }
    }
    let p1 = (sum_sq as f64) * (1.0 / 268_435_456.0);
    if hits_2b > (16_376.0 * (64.0 * p1 * p1).min(1.0)) as u32 + 200
        || (!large_window && ascii_lower > 12_500)
    {
        return PreChunkKind::Normal;
    }

    let prev = &env.data[env.start.saturating_sub(env.wsize)..env.start];
    let active_dict = if env.start < env.wsize { env.dict } else { &[] };
    if !prev.is_empty() || active_dict.len() >= 264 {
        let hash8 = |s: &[u8]| {
            let w8 = u64::from_le_bytes(s.try_into().unwrap());
            (
                w8 | 1,
                (w8.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 52) as usize,
            )
        };
        let mut head_tbl = [0u64; 4096];
        for s in slice[..2055].windows(8) {
            let (tag, h) = hash8(s);
            head_tbl[h] = tag;
        }
        let mut cross_hits = 0u32;
        let mut has_cross = |r: &[u8]| {
            r.windows(8).step_by(256).any(|s| {
                let (tag, h) = hash8(s);
                head_tbl[h] == tag && {
                    cross_hits += 1;
                    cross_hits >= 2
                }
            })
        };
        if has_cross(prev) || has_cross(active_dict) {
            return PreChunkKind::Normal;
        }
    }
    if 15_032_385_536u64.saturating_sub(sum_xlog2x) >= 8_557_723_648 {
        PreChunkKind::UncompressedRaw
    } else {
        PreChunkKind::PureLiterals
    }
}

fn compute_stream_chunks_in_range(
    range_env: ChunkEnv<'_>,
    large_window: bool,
    isolate_raw: bool,
) -> Vec<(usize, usize)> {
    const RAW_TILE: usize = 16_384;
    const MIN_RAW_RUN: usize = if cfg!(all(test, debug_assertions)) {
        16_384
    } else {
        65_536
    };

    let (mut cur, range_end) = (range_env.start, range_env.end);
    let mut ranges = Vec::new();
    let is_raw =
        |s, e| classify_chunk(range_env.sub(s, e), large_window) == PreChunkKind::UncompressedRaw;
    if isolate_raw && range_end - cur > MIN_RAW_RUN {
        let mut pos = cur;
        while pos + RAW_TILE <= range_end {
            if !is_raw(pos, pos + RAW_TILE) {
                pos += RAW_TILE;
                continue;
            }
            let lo = pos;
            pos += RAW_TILE;
            while pos + RAW_TILE <= range_end && is_raw(pos, pos + RAW_TILE) {
                pos += RAW_TILE;
            }
            let hi = if pos + RAW_TILE > range_end
                && pos < range_end
                && is_raw(range_end - RAW_TILE, range_end)
            {
                range_end
            } else {
                pos
            };
            if hi - lo >= MIN_RAW_RUN {
                push_chopped_range(&mut ranges, cur, lo);
                let mut r = lo;
                while r < hi {
                    let r_end = (r + RFC_MAX_METABLOCK_SIZE).min(hi);
                    if is_raw(r, r_end) {
                        ranges.push((r, r_end));
                    } else {
                        push_chopped_range(&mut ranges, r, r_end);
                    }
                    r = r_end;
                }
                cur = hi;
            }
            pos += RAW_TILE;
        }
    }
    push_chopped_range(&mut ranges, cur, range_end);
    ranges
}

fn append_stitched_commands(
    out: &mut Vec<OptimalCommand>,
    carry_trail: &mut u32,
    cmds: &[OptimalCommand],
    trailing: u32,
) {
    if cmds.is_empty() {
        *carry_trail += trailing;
    } else {
        let mut first = cmds[0];
        if *carry_trail > 0 {
            first.insert_len += *carry_trail;
            if first.dist_mode == DIST_MODE_IMPLICIT_D0 && first.insert_len >= 10 {
                first.dist_mode = 0;
            }
        }
        out.push(first);
        out.extend_from_slice(&cmds[1..]);
        *carry_trail = trailing;
    }
}

fn reconcile_chunk_dist_rb(
    res: &mut ChunkParseResult,
    assumed: ([u32; 4], u8),
    actual: ([u32; 4], u8),
) -> (([u32; 4], u8), bool) {
    if assumed == actual {
        return ((res.final_rb, res.final_valid_mask), false);
    }
    let (mut sim_a, mut sim_r) = (assumed, actual);
    let (mut changed, mut converged) = (false, false);

    for cmd in &mut res.commands {
        if sim_a == sim_r {
            converged = true;
            break;
        }
        let (old_mode, d) = (cmd.dist_mode, cmd.dist);
        sim_a = advance_dist_rb(sim_a.0, sim_a.1, d, old_mode);
        if old_mode == DIST_MODE_STATIC_DICT {
            continue;
        }
        let new_mode = if let Some(k) =
            (0..4u8).find(|&k| (sim_r.1 & (1 << k)) != 0 && d == sim_r.0[k as usize])
        {
            if k == 0 && cmd.insert_len <= 9 && copy_code_only(cmd.copy_len) < 16 {
                DIST_MODE_IMPLICIT_D0
            } else {
                k
            }
        } else if old_mode == DIST_MODE_RAW {
            DIST_MODE_RAW
        } else {
            (4..16u8)
                .find(|&sc| resolve_short_dist_code(sc, sim_r.0, sim_r.1, u32::MAX) == Some(d))
                .unwrap_or(DIST_MODE_RAW)
        };
        if new_mode != old_mode {
            cmd.dist_mode = new_mode;
            changed = true;
        }
        sim_r = advance_dist_rb(sim_r.0, sim_r.1, d, new_mode);
    }

    if !converged {
        (res.final_rb, res.final_valid_mask) = sim_r;
    }
    ((res.final_rb, res.final_valid_mask), changed)
}

struct ChunkDpCtx<'a> {
    env: ChunkEnv<'a>,
    large_window: bool,
    effort: u8,
    init: ([u32; 4], u8),
    intra_threads: usize,
    cm: &'a ChunkMatches,
    op_scratch: &'a mut OptimalParseScratch,
    cost_model: &'a mut ChunkCostModel,
}

impl ChunkDpCtx<'_> {
    #[inline]
    fn build_plan(&self, res: &ChunkParseResult, effort: u8) -> MetablockPlan {
        MergeCfg(self.large_window, effort, self.intra_threads).build_plan(
            self.env,
            &res.commands,
            res.trailing_literals,
        )
    }

    fn em_step<const BEAM: usize>(
        &mut self,
        best: &mut (ChunkParseResult, MetablockPlan),
        plan_effort: u8,
    ) -> (u64, bool) {
        self.cost_model
            .refine_from_metablock_plan(self.env, &best.0.commands, &best.1);
        let res = parse_chunk_optimal::<BEAM>(
            self.env,
            self.cm,
            self.cost_model,
            self.init,
            self.op_scratch,
        );
        let plan = self.build_plan(&res, plan_effort);
        let chunk_len = self.env.len();
        let both_uncomp = should_emit_uncompressed_plan(chunk_len, &plan)
            && should_emit_uncompressed_plan(chunk_len, &best.1);
        let prev_bits = best.1.exact_compressed_bits;
        let gain = prev_bits.saturating_sub(plan.exact_compressed_bits);
        if plan.exact_compressed_bits <= prev_bits {
            *best = (res, plan);
        }
        (gain, both_uncomp)
    }

    fn run(mut self, pre_res0: Option<ChunkParseResult>) -> (ChunkParseResult, MetablockPlan) {
        let chunk_len = self.env.len();
        let seed_plan_effort = if self.effort >= 12 { 11 } else { 10 };
        let seed_res0 = pre_res0.unwrap_or_else(|| {
            parse_chunk_greedy_seed(self.env, self.cm, self.init, self.effort >= 11)
        });
        let seed_plan0 = self.build_plan(&seed_res0, seed_plan_effort);
        if chunk_len >= 4096 && seed_plan0.exact_compressed_bits >= (chunk_len as u64) * 8 + 512 {
            return (seed_res0, MetablockPlan::uncompressed_sentinel(chunk_len));
        }

        let mut best = (seed_res0, seed_plan0);
        if self.effort >= 11 && chunk_len <= 524_288 {
            self.em_step::<1>(&mut best, seed_plan_effort);
        }
        if self.effort < 11 {
            self.em_step::<1>(&mut best, 10);
            return best;
        }
        if self.effort == 11 {
            let plan0_bits = best.1.exact_compressed_bits;
            let (pass1_gain, both_uncomp) = if chunk_len > 262_144 {
                self.em_step::<2>(&mut best, 11)
            } else {
                self.em_step::<4>(&mut best, 11)
            };
            if !both_uncomp && chunk_len <= 524_288 && pass1_gain * 650 >= plan0_bits {
                self.em_step::<4>(&mut best, 11);
            }
            return best;
        }

        let (pass1_gain, both_uncomp) = self.em_step::<8>(&mut best, 12);
        if both_uncomp || (chunk_len > 524_288 && pass1_gain < 64) {
            return best;
        }
        let prev_bits = best.1.exact_compressed_bits;
        let (pass2_gain, _) = self.em_step::<8>(&mut best, 12);
        if pass2_gain >= 128 && pass2_gain * 1250 >= prev_bits {
            let prev3 = best.1.exact_compressed_bits;
            let (pass3_gain, _) = self.em_step::<8>(&mut best, 12);
            if chunk_len <= 524_288 && pass3_gain * 900 >= prev3 {
                self.em_step::<8>(&mut best, 12);
            }
        }
        best
    }
}

enum EncodedGroupPiece<'a> {
    Uncompressed(&'a [u8]),
    Compressed(Vec<u8>, usize, usize, u64),
}

struct PendingMb {
    start: usize,
    end: usize,
    cmds: Vec<OptimalCommand>,
    trailing: u32,
    plan: MetablockPlan,
    dirty: bool,
}

#[allow(clippy::large_enum_variant)]
enum MbJob<'a> {
    Uncompressed(&'a [u8]),
    Compressed(PendingMb),
}

#[derive(Clone, Copy)]
struct MergeCfg(bool, u8, usize); // (large_window, effort, sub_threads)

impl MergeCfg {
    #[inline]
    fn build_plan(
        self,
        env: ChunkEnv<'_>,
        cmds: &[OptimalCommand],
        trailing: u32,
    ) -> MetablockPlan {
        build_metablock_plan_with_effort(
            env.data, env.start, env.end, cmds, trailing, self.0, self.1, self.2,
        )
    }
}

#[inline]
fn write_pending_mb(data: &[u8], p: &PendingMb, is_last: bool, bw: &mut BitWriter<'_>) {
    write_compressed_metablock(
        data, p.start, p.end, &p.cmds, p.trailing, &p.plan, is_last, bw,
    );
}

fn build_merged_pending<'a>(
    data: &[u8],
    span: impl IntoIterator<Item = &'a PendingMb>,
    mcfg: MergeCfg,
) -> PendingMb {
    let (mut cmds, mut trailing, mut start, mut end) = (Vec::new(), 0u32, usize::MAX, 0usize);
    for p in span {
        (start, end) = (start.min(p.start), p.end);
        cmds.reserve(p.cmds.len());
        append_stitched_commands(&mut cmds, &mut trailing, &p.cmds, p.trailing);
    }
    let plan = mcfg.build_plan(ChunkEnv::new(data, start, end, 0, &[]), &cmds, trailing);
    PendingMb {
        start,
        end,
        cmds,
        trailing,
        plan,
        dirty: false,
    }
}

/// Merges a contiguous run of compressed chunks: whole-run + greedy pairwise at `Q10`/`Q11`
/// (`<= 4 MiB`), or shortest-path metablock-boundary DAG DP over `run[0..m]` at `Q12` (`<= 16 MiB`).
fn flush_compressed_run(
    data: &[u8],
    run: Vec<PendingMb>,
    mcfg: MergeCfg,
    out: &mut Vec<MbJob<'_>>,
) {
    let m = run.len();
    if m <= 1 {
        out.extend(run.into_iter().map(MbJob::Compressed));
        return;
    }
    if mcfg.1 < 12 {
        if m >= 3 && run[m - 1].end - run[0].start <= Q11_MAX_METABLOCK_SIZE {
            let sum_bits: u64 = run.iter().map(|p| p.plan.exact_compressed_bits).sum();
            let merged = build_merged_pending(data, &run, mcfg);
            if merged.plan.exact_compressed_bits < sum_bits {
                out.push(MbJob::Compressed(merged));
                return;
            }
        }
        let mut it = run.into_iter();
        let mut cur = it.next().unwrap();
        for item in it {
            if item.end - cur.start <= Q11_MAX_METABLOCK_SIZE {
                let merged = build_merged_pending(data, [&cur, &item], mcfg);
                if merged.plan.exact_compressed_bits
                    < cur.plan.exact_compressed_bits + item.plan.exact_compressed_bits
                {
                    cur = merged;
                    continue;
                }
            }
            out.push(MbJob::Compressed(cur));
            cur = item;
        }
        out.push(MbJob::Compressed(cur));
        return;
    }

    debug_assert!(m <= 8);
    let (mut sub_dp, mut split_at, mut coherent) =
        ([[u64::MAX; 9]; 9], [[0usize; 9]; 9], [[false; 9]; 9]);
    let mut edges: Vec<Option<PendingMb>> = (0..81).map(|_| None).collect();
    for (i, p) in run.iter().enumerate() {
        sub_dp[i][i + 1] = p.plan.exact_compressed_bits;
        coherent[i][i + 1] = true;
    }
    for len in 2..=m {
        for i in 0..=(m - len) {
            let j = i + len;
            let (best_split, best_k) = ((i + 1)..j)
                .map(|k| (sub_dp[i][k].saturating_add(sub_dp[k][j]), k))
                .min_by_key(|&(c, _)| c)
                .unwrap();
            (sub_dp[i][j], split_at[i][j]) = (best_split, best_k);
            let mid = (i + j) / 2;
            let should_eval = run[j - 1].end - run[i].start <= RFC_MAX_METABLOCK_SIZE
                && (len == 2
                    || ((len <= 4 || i == 0 || j == m || ((i | j) & 1) == 0)
                        && coherent[i][mid]
                        && coherent[mid][j]
                        && (len != 4 || coherent[i][i + 3] || coherent[i + 1][i + 4])
                        && (i..(j - 1)).all(|s| coherent[s][s + 2])));
            if should_eval {
                let merged = build_merged_pending(data, &run[i..j], mcfg);
                let mbits = merged.plan.exact_compressed_bits;
                coherent[i][j] =
                    mbits <= best_split.saturating_add(if len == 2 { 1536 } else { 512 });
                if mbits < best_split {
                    edges[i * 9 + j] = Some(merged);
                    (sub_dp[i][j], split_at[i][j]) = (mbits, 0);
                }
            }
        }
    }
    for (i, mb) in run.into_iter().enumerate() {
        edges[i * 9 + i + 1] = Some(mb);
    }
    let mut stack = vec![(0, m)];
    while let Some((i, j)) = stack.pop() {
        let k = split_at[i][j];
        if k == 0 {
            out.push(MbJob::Compressed(edges[i * 9 + j].take().unwrap()));
        } else {
            stack.push((k, j));
            stack.push((i, k));
        }
    }
}

fn process_merge_group<'a>(data: &'a [u8], jobs: Vec<MbJob<'a>>, mcfg: MergeCfg) -> Vec<MbJob<'a>> {
    let (mut out, mut comp_run) = (
        Vec::with_capacity(jobs.len()),
        Vec::with_capacity(jobs.len()),
    );
    for job in jobs {
        match job {
            MbJob::Uncompressed(raw_slice) => {
                if !comp_run.is_empty() {
                    flush_compressed_run(data, std::mem::take(&mut comp_run), mcfg, &mut out);
                }
                out.push(MbJob::Uncompressed(raw_slice));
            }
            MbJob::Compressed(mut p) => {
                if p.dirty {
                    let cenv = ChunkEnv::new(data, p.start, p.end, 0, &[]);
                    p.plan = mcfg.build_plan(cenv, &p.cmds, p.trailing);
                    p.dirty = false;
                }
                comp_run.push(p);
            }
        }
    }
    if !comp_run.is_empty() {
        flush_compressed_run(data, comp_run, mcfg, &mut out);
    }
    out
}

fn map_vec_with_scratch<In, Out, S, Init, F>(
    items: Vec<In>,
    workers: usize,
    single_scratch: &mut S,
    pool: &mut Vec<S>,
    init_scratch: Init,
    f: F,
) -> Vec<Out>
where
    In: Send,
    Out: Send,
    S: Send,
    Init: FnMut() -> S,
    F: Fn(usize, In, &mut S) -> Out + Sync,
{
    let count = items.len();
    if count == 0 {
        return Vec::new();
    }
    let workers = workers.min(count);
    if workers <= 1 {
        return items
            .into_iter()
            .enumerate()
            .map(|(i, x)| f(i, x, single_scratch))
            .collect();
    }
    if pool.len() < workers {
        pool.resize_with(workers, init_scratch);
    }
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<(Option<In>, Option<Out>)>> = items
        .into_iter()
        .map(|x| Mutex::new((Some(x), None)))
        .collect();
    std::thread::scope(|s| {
        for scratch in &mut pool[..workers] {
            let (next_ref, slots_ref, f_ref) = (&next, &slots, &f);
            s.spawn(move || loop {
                let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                if idx >= count {
                    break;
                }
                let item = slots_ref[idx].lock().unwrap().0.take().unwrap();
                let out = f_ref(idx, item, scratch);
                slots_ref[idx].lock().unwrap().1 = Some(out);
            });
        }
    });
    slots
        .into_iter()
        .map(|m| m.into_inner().unwrap().1.unwrap())
        .collect()
}

fn ensure_len<T: Default>(v: &mut Vec<T>, n: usize) {
    if v.len() < n {
        v.resize_with(n, T::default);
    }
}

type RbPair = ([u32; 4], u8);

struct Q11WaveState {
    pred: RbPair,
    actual: RbPair,
}

impl Default for Q11WaveState {
    #[inline]
    fn default() -> Self {
        Self {
            pred: (INITIAL_LAST_DISTANCES, 0x0F),
            actual: (INITIAL_LAST_DISTANCES, 0x0F),
        }
    }
}

impl Q11Encoder {
    #[allow(clippy::too_many_arguments)]
    fn encode_chunks_pipeline<'a>(
        &mut self,
        base_env: ChunkEnv<'a>,
        chunks: &[(usize, usize)],
        full_dict_len: usize,
        dc_ref: Option<&DictChain>,
        eff_win: usize,
        is_eof: bool,
        state: &mut Q11WaveState,
        bw: &mut BitWriter<'_>,
    ) -> bool {
        let (effort, is_q11, is_q12) = (self.effort, self.effort >= 11, self.effort >= 12);
        let (sa_window_cap, large_window) = (self.sa_window_cap(), self.large_window);
        let eff_threads = self.effective_threads().max(1);
        if is_q11 && !large_window && eff_threads >= 2 {
            crate::static_dict::warm_dict121();
        }

        let merge_group_chunks: usize = if is_q12 { 8 } else { 4 };
        let wave_cap = if eff_threads <= 1 {
            merge_group_chunks
        } else {
            (eff_threads
                .saturating_mul(2)
                .saturating_add(merge_group_chunks - 1)
                & !(merge_group_chunks - 1))
                .clamp(8, 32)
        };
        ensure_len(&mut self.pools.wave_cm, wave_cap);
        ensure_len(&mut self.pools.wave_ldm, wave_cap);

        let mut ended_with_last_mb = false;
        let num_waves = chunks.len().div_ceil(wave_cap);

        for (wave_idx, wave_chunks) in chunks.chunks(wave_cap).enumerate() {
            let is_last_wave = wave_idx + 1 == num_waves;
            let w_len = wave_chunks.len();
            let mut kinds = Vec::with_capacity(w_len);

            for (i, &(c_start, c_end)) in wave_chunks.iter().enumerate() {
                let cenv = base_env.sub(c_start, c_end);
                let kind = classify_chunk(cenv, large_window);
                kinds.push(kind);
                if kind == PreChunkKind::Normal {
                    self.mg_scratch.ldm.extract_chunk_matches(
                        cenv,
                        eff_win,
                        sa_window_cap,
                        &mut self.pools.wave_ldm[i],
                    );
                } else {
                    self.pools.wave_ldm[i].clear();
                }
            }

            {
                let ldm_slice = &self.pools.wave_ldm[..w_len];
                let mg_tasks: Vec<(usize, &mut ChunkMatches)> = self.pools.wave_cm[..w_len]
                    .iter_mut()
                    .enumerate()
                    .filter(|&(i, _)| kinds[i] == PreChunkKind::Normal)
                    .collect();
                let mg_workers = eff_threads.min(16).min(mg_tasks.len());
                map_vec_with_scratch(
                    mg_tasks,
                    mg_workers,
                    &mut self.mg_scratch,
                    &mut self.pools.mg_workers,
                    MatchGenScratch::default,
                    |_, (i, cm_out), mg_s| {
                        let (c_start, c_end) = wave_chunks[i];
                        mg_s.reset_h10();
                        build_chunk_matches(
                            MatchGenCfg {
                                env: base_env.sub(c_start, c_end),
                                local_window_cap: sa_window_cap,
                                large_window,
                                effort,
                                full_dict_len,
                                dict_chain: dc_ref,
                                ldm: &ldm_slice[i],
                            },
                            mg_s,
                            cm_out,
                        );
                    },
                );
            }

            let cm_slice = &self.pools.wave_cm[..w_len];
            let mut assumed = Vec::with_capacity(w_len);
            let mut dp_tasks: Vec<(usize, Option<ChunkParseResult>)> = Vec::with_capacity(w_len);
            for i in 0..w_len {
                assumed.push(state.pred);
                match kinds[i] {
                    PreChunkKind::UncompressedRaw => {}
                    PreChunkKind::PureLiterals => dp_tasks.push((i, None)),
                    PreChunkKind::Normal => {
                        let seed = (w_len > 1).then(|| {
                            let (s, e) = wave_chunks[i];
                            let res0 = parse_chunk_greedy_seed(
                                base_env.sub(s, e),
                                &cm_slice[i],
                                state.pred,
                                is_q11,
                            );
                            state.pred = (res0.final_rb, res0.final_valid_mask);
                            res0
                        });
                        dp_tasks.push((i, seed));
                    }
                }
            }

            let dp_workers = eff_threads.min(dp_tasks.len());
            let intra_dp_plan = (eff_threads / dp_workers.max(1)).max(1);
            let mut dp_results = map_vec_with_scratch(
                dp_tasks,
                dp_workers,
                &mut self.dp_scratch,
                &mut self.pools.dp_workers,
                Default::default,
                |_, (i, pre_res0), (op_s, model_s)| {
                    let (s, e) = wave_chunks[i];
                    let cenv = base_env.sub(s, e);
                    if kinds[i] == PreChunkKind::PureLiterals {
                        let len32 = cenv.len() as u32;
                        let plan = MergeCfg(large_window, effort, intra_dp_plan).build_plan(
                            cenv,
                            &[],
                            len32,
                        );
                        (ChunkParseResult::empty(assumed[i], len32), plan)
                    } else {
                        ChunkDpCtx {
                            env: cenv,
                            large_window,
                            effort,
                            init: assumed[i],
                            intra_threads: intra_dp_plan,
                            cm: &cm_slice[i],
                            op_scratch: op_s,
                            cost_model: model_s,
                        }
                        .run(pre_res0)
                    }
                },
            )
            .into_iter();

            let mut emit_jobs: Vec<MbJob<'_>> = Vec::with_capacity(w_len);
            for i in 0..w_len {
                let (c_start, c_end) = wave_chunks[i];
                let chunk_len = c_end - c_start;
                if kinds[i] == PreChunkKind::UncompressedRaw {
                    emit_jobs.push(MbJob::Uncompressed(&base_env.data[c_start..c_end]));
                    continue;
                }
                let (mut res, mut plan) = dp_results.next().unwrap();
                if should_emit_uncompressed_plan(chunk_len, &plan) {
                    emit_jobs.push(MbJob::Uncompressed(&base_env.data[c_start..c_end]));
                    continue;
                }
                let (next_actual, mut dirty) =
                    reconcile_chunk_dist_rb(&mut res, assumed[i], state.actual);
                if dirty
                    && plan.exact_compressed_bits.saturating_add(65_536)
                        > (chunk_len as u64) * 8 + 40
                {
                    plan = MergeCfg(large_window, effort, eff_threads).build_plan(
                        base_env.sub(c_start, c_end),
                        &res.commands,
                        res.trailing_literals,
                    );
                    if should_emit_uncompressed_plan(chunk_len, &plan) {
                        emit_jobs.push(MbJob::Uncompressed(&base_env.data[c_start..c_end]));
                        continue;
                    }
                    dirty = false;
                }
                state.actual = next_actual;
                emit_jobs.push(MbJob::Compressed(PendingMb {
                    start: c_start,
                    end: c_end,
                    cmds: res.commands,
                    trailing: res.trailing_literals,
                    plan,
                    dirty,
                }));
            }

            let num_groups = w_len.div_ceil(merge_group_chunks);
            let group_workers = eff_threads.min(num_groups);
            let mcfg = MergeCfg(
                large_window,
                effort,
                (eff_threads / num_groups.max(1)).max(1),
            );
            let mut job_iter = emit_jobs.into_iter();
            let groups: Vec<Vec<MbJob<'_>>> = (0..num_groups)
                .map(|_| job_iter.by_ref().take(merge_group_chunks).collect())
                .collect();
            let run_group = |g: usize, g_jobs: Vec<MbJob<'a>>| {
                let is_final = is_eof && is_last_wave && (g + 1 == num_groups);
                let mbs = process_merge_group(base_env.data, g_jobs, mcfg);
                let closed = is_final && matches!(mbs.last(), Some(MbJob::Compressed(_)));
                (is_final, mbs, closed)
            };

            if group_workers <= 1 {
                for (g, g_jobs) in groups.into_iter().enumerate() {
                    let (is_final, mbs, closed) = run_group(g, g_jobs);
                    let n_mbs = mbs.len();
                    for (idx, mb) in mbs.into_iter().enumerate() {
                        match mb {
                            MbJob::Uncompressed(s) => write_uncompressed_metablock(s, bw),
                            MbJob::Compressed(p) => write_pending_mb(
                                base_env.data,
                                &p,
                                is_final && idx + 1 == n_mbs,
                                bw,
                            ),
                        }
                    }
                    ended_with_last_mb |= closed;
                }
            } else {
                let (mut dummy, mut dummy_pool) = ((), Vec::new());
                let group_results = map_vec_with_scratch(
                    groups,
                    group_workers,
                    &mut dummy,
                    &mut dummy_pool,
                    || (),
                    |g, g_jobs, ()| {
                        let (is_final, mbs, closed) = run_group(g, g_jobs);
                        let n_mbs = mbs.len();
                        let pieces: Vec<EncodedGroupPiece<'_>> = mbs
                            .into_iter()
                            .enumerate()
                            .map(|(idx, mb)| match mb {
                                MbJob::Uncompressed(s) => EncodedGroupPiece::Uncompressed(s),
                                MbJob::Compressed(p) => {
                                    let cap = ((p.plan.exact_compressed_bits as usize) >> 3) + 64;
                                    let mut buf = vec![0u8; cap];
                                    let (blen, tn, tb) = {
                                        let mut c_bw = BitWriter::new(&mut buf[..]);
                                        let last = is_final && idx + 1 == n_mbs;
                                        write_pending_mb(base_env.data, &p, last, &mut c_bw);
                                        let (written, tn, tb) = c_bw.finalize_unpadded();
                                        (written.len(), tn, tb)
                                    };
                                    EncodedGroupPiece::Compressed(buf, blen, tn, tb)
                                }
                            })
                            .collect();
                        (pieces, closed)
                    },
                );

                for (pieces, closed) in group_results {
                    for piece in pieces {
                        match piece {
                            EncodedGroupPiece::Compressed(buf, blen, tn, tb) => {
                                bw.append_unpadded(&buf[..blen], tn, tb)
                            }
                            EncodedGroupPiece::Uncompressed(s) => {
                                write_uncompressed_metablock(s, bw)
                            }
                        }
                    }
                    ended_with_last_mb |= closed;
                }
            }
        }

        state.pred = state.actual;
        ended_with_last_mb
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_range_with_dict(
        &mut self,
        data: &[u8],
        (r_start, r_end): (usize, usize),
        dict: Option<&PreparedDictionary>,
        cached_dc: Option<&DictChain>,
        streaming: bool,
        is_eof: bool,
        state: &mut Q11WaveState,
        bw: &mut BitWriter<'_>,
    ) -> bool {
        let wsize = window_size(self.wbits);
        let eff_win = if streaming {
            wsize
        } else {
            wsize.min(data.len())
        };
        let (dict_active, full_dict_len): (&[u8], usize) = match dict {
            Some(d) if !d.is_empty() => (d.active_slice(wsize), d.raw_len()),
            _ => (&[], 0),
        };
        let owned_dc = cached_dc
            .is_none()
            .then(|| DictChain::new(dict_active))
            .flatten();
        let range_env = ChunkEnv::new(data, r_start, r_end, wsize, dict_active);
        let chunks =
            compute_stream_chunks_in_range(range_env, self.large_window, self.effort >= 11);
        self.encode_chunks_pipeline(
            range_env,
            &chunks,
            full_dict_len,
            cached_dc.or(owned_dc.as_ref()),
            eff_win,
            is_eof,
            state,
            bw,
        )
    }

    pub(crate) fn compress_into(
        &mut self,
        data: &[u8],
        dict: Option<&PreparedDictionary>,
        bw: &mut BitWriter<'_>,
    ) {
        self.mg_scratch.reset_stream();
        write_window_bits(self.wbits, self.large_window, bw);
        let mut st = Q11WaveState::default();
        let rng = (0, data.len());
        if data.is_empty()
            || !self.encode_range_with_dict(data, rng, dict, None, false, true, &mut st, bw)
        {
            write_empty_last_metablock(bw);
        }
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    #[inline]
    pub(crate) fn dbg_or<T>(dbg: T, rel: T) -> T {
        if cfg!(debug_assertions) {
            dbg
        } else {
            rel
        }
    }

    pub(crate) fn append_lcg_bytes(out: &mut Vec<u8>, count: usize, mut st: u64, add: u64) {
        out.reserve(count);
        for _ in 0..count {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(add);
            out.push((st >> 56) as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::test_helpers::{append_lcg_bytes, dbg_or};
    use super::ChunkEnv;
    use crate::metablock::test_helpers::{decompress_brotli, decompress_brotli_with_dict};
    use crate::{Encoder, PreparedDictionary};

    fn comp_verified(
        q: u32,
        lgwin: u32,
        data: &[u8],
        dict: Option<(&PreparedDictionary, &[u8])>,
    ) -> Vec<u8> {
        let mut enc = Encoder::with_lgwin(q, lgwin);
        match dict {
            Some((prep, raw)) => {
                let c = enc.compress_with_dict(data, prep, None).unwrap().to_vec();
                assert_eq!(decompress_brotli_with_dict(&c, raw), data);
                c
            }
            None => {
                let c = enc.compress(data, None).unwrap().to_vec();
                assert_eq!(decompress_brotli(&c), data);
                c
            }
        }
    }

    fn assert_1t_mt_bit_exact(data: &[u8], q: u32, lgwin: u32, mt_threads: &[usize]) -> Vec<u8> {
        let comp_mt = |t| {
            Encoder::with_threads(q, lgwin, t)
                .compress(data, None)
                .unwrap()
                .to_vec()
        };
        let comp_1t = comp_mt(1);
        assert_eq!(decompress_brotli(&comp_1t), data);
        for &t in mt_threads {
            assert_eq!(comp_mt(t), comp_1t);
        }
        comp_1t
    }

    fn build_mixed_regime_buf(target_len: usize) -> Vec<u8> {
        let (mut buf, mut i) = (Vec::with_capacity(target_len + 256), 0u32);
        while buf.len() < target_len {
            match (i / 32) % 3 {
                0 => buf.extend_from_slice(
                    format!("Information about international government policy and university research: section-{i}\n").as_bytes(),
                ),
                1 => (0..8u32).for_each(|k| {
                    let v = i * 8 + k;
                    buf.extend_from_slice(&v.wrapping_mul(2654435761).to_le_bytes());
                    buf.extend_from_slice(&(0x0102_0300u32 | (v & 0xFF)).to_le_bytes());
                }),
                _ => buf.extend_from_slice(
                    format!("{{\"id\":{i},\"active\":true,\"kind\":\"storage#object\"}}\n").as_bytes(),
                ),
            }
            i += 1;
        }
        buf.truncate(target_len);
        buf
    }

    #[test]
    fn test_q10_q11_roundtrip_edge_cases_windows_and_multi_regime() {
        let (mut prng, rle_q) = (Vec::new(), vec![b'Q'; dbg_or(4096, 16_384)]);
        append_lcg_bytes(&mut prng, dbg_or(2048, 8192), 0xDEAD_BEEF_CAFE_BABE, 1);
        let multi = build_mixed_regime_buf(dbg_or(6 * 1024, 96 * 1024));
        #[rustfmt::skip]
        let cases = [&b""[..], b"z", b"ab", b"abc", b"abcd", b"timing the government of the United States", &rle_q, &prng, &multi];
        for q in [10, 11] {
            for &lgwin in dbg_or(&[10u32, 22, 26][..], &[10, 16, 22, 24, 26, 30][..]) {
                let mut enc = Encoder::with_lgwin(q, lgwin);
                for &inp in &cases {
                    let mut buf = vec![MaybeUninit::uninit(); enc.max_required_size(inp.len())];
                    assert_eq!(
                        decompress_brotli(enc.compress(inp, Some(&mut buf)).unwrap()),
                        inp
                    );
                }
                assert!(enc
                    .compress(&multi, Some(&mut [MaybeUninit::uninit(); 4]))
                    .is_none());
            }
        }
    }

    #[test]
    fn test_q10_q11_custom_dict_and_multithreaded_multi_chunk() {
        let dict_raw = b"HTTP/1.1 200 OK\nContent-Type: application/json; charset=utf-8\n{\"service\":\"rbrotli-optimal\",\"region\":\"europe-west4\",\"permissions\":[\"read\",\"write\",\"list\"]}";
        let payload = b"HTTP/1.1 200 OK\nContent-Type: application/json; charset=utf-8\n{\"service\":\"rbrotli-optimal\",\"region\":\"europe-west4\",\"permissions\":[\"read\",\"write\",\"list\"],\"id\":99}";
        let mut big_dict = vec![b'X'; 3000];
        big_dict.extend_from_slice(dict_raw);

        for q in [10, 11] {
            let prep = PreparedDictionary::with_lgwin(dict_raw, q, 22);
            let c_no = comp_verified(q, 22, payload, None);
            let c_dict = comp_verified(q, 22, payload, Some((&prep, dict_raw)));
            assert!(c_dict.len() * 2 < c_no.len());

            let prep10 = PreparedDictionary::with_lgwin(&big_dict, q, 10);
            comp_verified(q, 10, payload, Some((&prep10, &big_dict)));
        }

        let (c0_end, c1_end, tail_rep) =
            dbg_or((17_000, 25_000, 9_000), (1_100_000, 2_100_000, 300_000));
        let mut big = Vec::with_capacity(c1_end + tail_rep);
        while big.len() < c0_end {
            big.extend_from_slice(
                b"Cross-chunk long-distance repeat anchor for optimal parser verification! ",
            );
            big.extend_from_slice(&(big.len() as u32 & 63).to_le_bytes());
        }
        let rlen = c1_end - big.len();
        append_lcg_bytes(&mut big, rlen, 0x1234_5678_9ABC_DEF0, 1);
        big.extend_from_within(..tail_rep);
        for q in [10, 11] {
            assert_1t_mt_bit_exact(&big, q, 24, &[2, 5, 0]);
        }
    }

    #[test]
    fn test_p3_single_chunk_intra_parallel_bit_exact() {
        let buf = build_mixed_regime_buf(dbg_or(9_500, 220_000));
        let env = ChunkEnv::new(&buf, 0, buf.len(), 1 << 22, &[]);
        assert_eq!(
            super::compute_stream_chunks_in_range(env, false, false).len(),
            1
        );
        for q in [10, 11] {
            assert_1t_mt_bit_exact(&buf, q, 22, &[4]);
        }
    }

    #[test]
    fn test_p2_reconcile_chunk_dist_rb() {
        use super::{
            advance_dist_rb, reconcile_chunk_dist_rb, ChunkParseResult, OptimalCommand,
            DIST_MODE_IMPLICIT_D0, DIST_MODE_RAW,
        };
        let assumed_rb = [40u32, 11, 15, 16];
        let cmds: Vec<_> = [
            (3, 6, 40, DIST_MODE_IMPLICIT_D0),
            (2, 5, 40, DIST_MODE_IMPLICIT_D0),
            (1, 4, 41, 5),
            (2, 4, 77, DIST_MODE_RAW),
            (2, 4, 88, DIST_MODE_RAW),
            (2, 4, 120, DIST_MODE_RAW),
        ]
        .into_iter()
        .map(|(insert_len, copy_len, dist, dist_mode)| OptimalCommand {
            insert_len,
            copy_len,
            out_len: copy_len,
            dist,
            dist_mode,
        })
        .collect();
        let (final_rb, final_valid_mask) = cmds.iter().fold((assumed_rb, 0x0Fu8), |s, c| {
            advance_dist_rb(s.0, s.1, c.dist, c.dist_mode)
        });
        for (actual_rb, exp_mode0) in [([99, 42, 15, 16], 12), ([99, 90, 15, 16], DIST_MODE_RAW)] {
            let mut res = ChunkParseResult {
                commands: cmds.clone(),
                trailing_literals: 0,
                final_rb,
                final_valid_mask,
            };
            let (fin, changed) =
                reconcile_chunk_dist_rb(&mut res, (assumed_rb, 0x0F), (actual_rb, 0x0F));
            assert!(changed && res.commands[0].dist_mode == exp_mode0);
            assert_eq!(
                (res.commands[1].dist_mode, res.commands[2].dist_mode),
                (DIST_MODE_IMPLICIT_D0, 5)
            );
            assert_eq!(fin, ([120, 88, 77, 41], 0x0F));
        }
    }

    #[test]
    fn test_p1_lw2_multithreaded_large_window_pipeline() {
        let (anchor_len, gap_len) = dbg_or((12_000usize, 36_000usize), (262_144, 4_500_000));
        let (mut data, mut s) = (
            Vec::with_capacity(anchor_len * 2 + gap_len),
            0x9E37_79B9_7F4A_7C15u64,
        );
        for i in 0..anchor_len {
            if (i & 15) == 0 {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            }
            data.push(b'A' + (((s >> ((i & 7) * 8)) as u8) & 31));
        }
        append_lcg_bytes(&mut data, gap_len, s, 3037000493);
        data.extend_from_within(..anchor_len);
        for q in [10, 11] {
            let comp_1t = assert_1t_mt_bit_exact(&data, q, 28, &[3, 6]);
            assert!(comp_1t.len() < gap_len + (anchor_len * 6) / 5);
        }
    }

    #[test]
    fn test_q12_archival_subchunk_raw_dag_dp_and_mt_bit_exact() {
        let dict_raw = br#"{"protocol":"rbrotli-q12","capabilities":["sais-3mib","beam8","mtf-viterbi","dag-dp"]}"#;
        let dict_msg = br#"{"protocol":"rbrotli-q12","capabilities":["sais-3mib","beam8","mtf-viterbi","dag-dp"],"ok":true}"#;
        let prep = PreparedDictionary::with_lgwin(dict_raw, 12, 22);
        comp_verified(12, 22, dict_msg, Some((&prep, dict_raw)));

        for lgwin in [10, 22, 28] {
            for inp in [&b""[..], b"x", b"xy", b"xyz", dict_msg] {
                comp_verified(12, lgwin, inp, None);
            }
        }

        let (prose_a, raw_len, tail_prose) =
            dbg_or((9_500usize, 24_576, 12_000), (160_000, 98_304, 1_200_000));
        let mut mixed = build_mixed_regime_buf(prose_a);
        let raw_s = mixed.len();
        append_lcg_bytes(&mut mixed, raw_len, 0xDEAD_BEEF_1357_2468, 1);
        let raw_e = mixed.len();
        mixed.extend_from_slice(&build_mixed_regime_buf(tail_prose));
        mixed.extend_from_within(..(prose_a / 2));

        let env = ChunkEnv::new(&mixed, 0, mixed.len(), super::window_size(24), &[]);
        assert!(super::compute_stream_chunks_in_range(env, false, true)
            .iter()
            .any(|&(s, e)| {
                s >= raw_s.saturating_sub(16_384)
                    && e <= raw_e + 16_384
                    && super::classify_chunk(env.sub(s, e), false)
                        == super::PreChunkKind::UncompressedRaw
            }));

        let c10 = comp_verified(10, 24, &mixed, None);
        let c12_1t =
            assert_1t_mt_bit_exact(&mixed, 12, 24, dbg_or(&[3usize][..], &[2, 4, 8, 0][..]));
        assert!(c12_1t.len() < c10.len());
    }

    #[test]
    fn test_usize_max_threads_expired_dict_classify_and_h10_4gib() {
        // `num_threads = usize::MAX` must clamp cleanly without overflowing `eff_threads * 2`.
        let sample = build_mixed_regime_buf(32_768);
        let c1 = Encoder::with_threads(11, 20, 1)
            .compress(&sample, None)
            .unwrap()
            .to_vec();
        let c_max = Encoder::with_threads(11, 20, usize::MAX)
            .compress(&sample, None)
            .unwrap()
            .to_vec();
        assert_eq!(c_max, c1);

        // Once `env.start >= env.wsize`, `env.dict` has expired out of the sliding window and
        // must not force an incompressible LCG chunk to `PreChunkKind::Normal`.
        let wsize = super::window_size(14); // 16,368
        let mut raw_stream = Vec::new();
        append_lcg_bytes(&mut raw_stream, wsize + 16_384, 0xA5A5_5A5A_F00D_CAFE, 3);
        let expired_dict = raw_stream[wsize..wsize + 4096].to_vec();
        let in_win_env = ChunkEnv::new(&raw_stream, 0, 16_384, wsize, &raw_stream[..4096]);
        assert_eq!(
            super::classify_chunk(in_win_env, false),
            super::PreChunkKind::Normal
        );
        let expired_env = ChunkEnv::new(&raw_stream, wsize, wsize + 16_384, wsize, &expired_dict);
        assert_eq!(
            super::classify_chunk(expired_env, false),
            super::PreChunkKind::UncompressedRaw
        );

        // On 64-bit Linux, verify H10 BST/3B/2B match generation straddling `2^32 - 1` produces
        // identical intervals to low stream offsets instead of wrapping `(pos + 1) as u32` to `0`.
        #[cfg(all(target_os = "linux", target_pointer_width = "64", not(miri)))]
        {
            use super::match_gen::{
                build_chunk_matches, ChunkLdmMatches, ChunkMatches, MatchGenCfg, MatchGenScratch,
            };
            const TWO_32: usize = 1usize << 32;
            let mut big = vec![0u8; TWO_32 + 4096];
            for i in (TWO_32 - 4096)..(TWO_32 + 4096) {
                big[i] = sample[i % sample.len()];
            }
            let empty_ldm = ChunkLdmMatches::default();
            let (mut mg_hi, mut cm_hi) = (MatchGenScratch::default(), ChunkMatches::default());
            build_chunk_matches(
                MatchGenCfg {
                    env: ChunkEnv::new(&big, TWO_32 - 2048, TWO_32 + 2048, 1 << 20, &[]),
                    local_window_cap: 1 << 20,
                    large_window: false,
                    effort: 11,
                    full_dict_len: 0,
                    dict_chain: None,
                    ldm: &empty_ldm,
                },
                &mut mg_hi,
                &mut cm_hi,
            );
            assert!(!cm_hi.intervals.is_empty());
            let low = &big[TWO_32 - 4096..TWO_32 + 4096];
            let (mut mg_lo, mut cm_lo) = (MatchGenScratch::default(), ChunkMatches::default());
            build_chunk_matches(
                MatchGenCfg {
                    env: ChunkEnv::new(low, 2048, 6144, 1 << 20, &[]),
                    local_window_cap: 1 << 20,
                    large_window: false,
                    effort: 11,
                    full_dict_len: 0,
                    dict_chain: None,
                    ldm: &empty_ldm,
                },
                &mut mg_lo,
                &mut cm_lo,
            );
            assert_eq!(cm_hi.intervals.len(), cm_lo.intervals.len());
            for (a, b) in cm_hi.intervals.iter().zip(cm_lo.intervals.iter()) {
                assert_eq!((a.l_lo, a.l_hi, a.dist), (b.l_lo, b.l_hi, b.dist));
            }
        }
    }
}
