#![allow(unused_imports)]
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

use crate::block_split::{try_plan_metablock, MetablockPlan, PlanOpts};
use crate::constants::{
    window_size, HASH_MUL32, INITIAL_LAST_DISTANCES, Q11_MAX_METABLOCK_SIZE, STD_MAX_WBITS,
};
use crate::hashtable::PreparedDictionary;
use crate::map_to_sym::{advance_dist_rb, can_use_implicit_d0, match_short_dist_code_masked};
use crate::metablock::{
    should_emit_uncompressed_plan, write_compressed_metablock, write_empty_last_metablock,
    write_uncompressed_metablock, write_window_bits, OptimalCommand, DIST_MODE_IMPLICIT_D0,
    DIST_MODE_RAW, DIST_MODE_STATIC_DICT,
};

pub(crate) use match_gen::DictChain;
use match_gen::{
    build_chunk_matches, ChunkLdmMatches, ChunkMatches, MatchGenCfg, MatchGenScratch, SparseLdm,
};
use optimal_parse::{
    parse_chunk_greedy_seed, parse_chunk_optimal, ChunkCostModel, ChunkParseResult,
    OptimalParseScratch,
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
    pub(super) fn new(
        data: &'a [u8],
        start: usize,
        end: usize,
        wsize: usize,
        dict: &'a [u8],
    ) -> Self {
        Self {
            data,
            start,
            end,
            wsize,
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
    pub(super) fn cand_slice(&self, sp: usize, dist: u32) -> &'a [u8] {
        let d = dist as usize;
        if d == 0 || d > self.max_lz_dist(sp) as usize || sp >= self.end {
            return &[];
        }
        let rem = self.end - sp;
        if d <= sp {
            &self.data[sp - d..sp - d + rem]
        } else {
            let dict_rem = d - sp;
            let (off, max_l) = (self.dict.len() - dict_rem, dict_rem.min(rem));
            &self.dict[off..off + max_l]
        }
    }

    #[inline(always)]
    pub(super) fn has_2b(&self, sp: usize, dist: u32, b0: u8, b1: u8) -> bool {
        matches!(self.cand_slice(sp, dist), [x0, x1, ..] if *x0 == b0 && *x1 == b1)
    }

    #[inline(always)]
    pub(super) fn match_len(&self, local_pos: usize, dist: u32) -> u32 {
        let sp = self.start + local_pos;
        let c = self.cand_slice(sp, dist);
        crate::hashtable::common_prefix_len(c, &self.data[sp..sp + c.len()], usize::MAX) as u32
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
            if std::mem::replace(slot, w4 as u64) == w4 as u64 {
                hits_4b += 1;
                if hits_4b > 4 {
                    return PreChunkKind::Normal;
                }
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
                for r in (lo..hi).step_by(RFC_MAX_METABLOCK_SIZE) {
                    let r_end = (r + RFC_MAX_METABLOCK_SIZE).min(hi);
                    if is_raw(r, r_end) {
                        ranges.push((r, r_end));
                    } else {
                        push_chopped_range(&mut ranges, r, r_end);
                    }
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
        let new_mode = match match_short_dist_code_masked(d, sim_r.0, sim_r.1) {
            Some(0) if can_use_implicit_d0(cmd.insert_len, cmd.copy_len) => DIST_MODE_IMPLICIT_D0,
            Some(sc) if sc < 4 || old_mode != DIST_MODE_RAW => sc,
            _ => DIST_MODE_RAW,
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

    fn em_step(
        &mut self,
        bw: usize,
        plan_effort: u8,
        best: &mut (ChunkParseResult, MetablockPlan),
    ) -> (u64, u64, bool) {
        self.cost_model
            .refine_from_metablock_plan(self.env, &best.0.commands, &best.1);
        let res = parse_chunk_optimal(
            bw,
            self.env,
            self.cm,
            self.cost_model,
            self.init,
            self.op_scratch,
        );
        let plan = self.build_plan(&res, plan_effort);
        let (chunk_len, prev_bits) = (self.env.len(), best.1.exact_compressed_bits);
        let both_uncomp = should_emit_uncompressed_plan(chunk_len, &plan)
            && should_emit_uncompressed_plan(chunk_len, &best.1);
        let gain = prev_bits.saturating_sub(plan.exact_compressed_bits);
        if plan.exact_compressed_bits <= prev_bits {
            *best = (res, plan);
        }
        (prev_bits, gain, both_uncomp)
    }

    fn run(mut self, pre_res0: Option<ChunkParseResult>) -> (ChunkParseResult, MetablockPlan) {
        let (chunk_len, seed_eff) = (self.env.len(), (self.effort - 1).max(10));
        let seed_res0 = pre_res0.unwrap_or_else(|| {
            parse_chunk_greedy_seed(self.env, self.cm, self.init, self.effort >= 11)
        });
        let seed_plan0 = self.build_plan(&seed_res0, seed_eff);
        if chunk_len >= 4096 && seed_plan0.exact_compressed_bits >= (chunk_len as u64) * 8 + 512 {
            return (seed_res0, MetablockPlan::uncompressed_sentinel(chunk_len));
        }
        let mut best = (seed_res0, seed_plan0);
        if self.effort < 11 {
            self.em_step(1, 10, &mut best);
            return best;
        }
        let small = chunk_len <= 524_288;
        if small {
            self.em_step(1, seed_eff, &mut best);
        }
        let bw11 = if chunk_len > 262_144 { 2 } else { 4 };
        let schedule: &[(usize, u64, u64, bool, bool)] = if self.effort == 11 {
            &[(bw11, 0, 1, false, false), (4, 0, 650, true, true)]
        } else {
            &[
                (8, 0, 1, false, false),
                (8, if small { 0 } else { 64 }, 0, false, true),
                (8, 128, 1250, false, false),
                (8, 0, 900, true, false),
            ]
        };
        let (mut prev_bits, mut gain, mut both_uncomp) = (0u64, 0u64, false);
        for &(bw, min_gain, mul, req_small, chk_uncomp) in schedule {
            if (chk_uncomp && both_uncomp)
                || (req_small && !small)
                || gain < min_gain
                || (mul > 0 && gain * mul < prev_bits)
            {
                break;
            }
            (prev_bits, gain, both_uncomp) = self.em_step(bw, self.effort, &mut best);
        }
        best
    }
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
        try_plan_metablock(
            env.data,
            env.start,
            env.end,
            cmds,
            trailing,
            PlanOpts::q11(self.0, self.1, self.2),
        )
        .unwrap()
    }
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

/// Merges a contiguous run of compressed chunks via whole-run + greedy pairwise coalescing.
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
    let max_merge_span = if mcfg.1 >= 12 {
        RFC_MAX_METABLOCK_SIZE
    } else {
        Q11_MAX_METABLOCK_SIZE
    };
    if m >= 3 && run[m - 1].end - run[0].start <= max_merge_span {
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
        if item.end - cur.start <= max_merge_span {
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
}

fn process_merge_group<'a>(data: &'a [u8], jobs: Vec<MbJob<'a>>, mcfg: MergeCfg) -> Vec<MbJob<'a>> {
    let (mut out, mut comp_run) = (
        Vec::with_capacity(jobs.len()),
        Vec::with_capacity(jobs.len()),
    );
    for job in jobs {
        match job {
            MbJob::Uncompressed(raw_slice) => {
                flush_compressed_run(data, std::mem::take(&mut comp_run), mcfg, &mut out);
                out.push(MbJob::Uncompressed(raw_slice));
            }
            MbJob::Compressed(mut p) => {
                if p.dirty {
                    p.plan = mcfg.build_plan(
                        ChunkEnv::new(data, p.start, p.end, 0, &[]),
                        &p.cmds,
                        p.trailing,
                    );
                    p.dirty = false;
                }
                comp_run.push(p);
            }
        }
    }
    flush_compressed_run(data, comp_run, mcfg, &mut out);
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

    #[inline]
    pub(crate) fn miri_or<T>(miri: T, normal: T) -> T {
        if cfg!(miri) {
            miri
        } else {
            normal
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
        .map(|(insert_len, copy_len, dist, dist_mode)| {
            OptimalCommand::new(insert_len, copy_len, copy_len, dist, dist_mode)
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
}
