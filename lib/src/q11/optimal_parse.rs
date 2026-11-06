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

//! Subquadratic forward-DP optimal Brotli parser for `Q10` (`BEAM = 1`), `Q11` (`BEAM = 2, 4`),
//! and `Q12` (`BEAM = 8`), with exact short-literal buckets + a monotonic Pareto stack over long
//! literal runs, a 512-slot `u64` Range-Chmin horizon ring over copy-code intervals, all 16 RFC
//! 7932 `dist_rb` short codes + implicit `d0`, and EM cost-model refinement.

use crate::block_split::MetablockPlan;
use crate::constants::{INITIAL_LAST_DISTANCES, MAX_IAC, MAX_LIT};
use crate::context;
use crate::huffman::{fill_smoothed_symbol_costs, HuffmanCodeEntry};
use crate::map_to_sym::{
    advance_dist_rb, can_use_implicit_d0, copy_code_only, encode_raw_distance_params,
    iac_sym_from_codes, ins_code_only, resolve_short_dist_code, COPY_BASE, COPY_EXTRA, COPY_MAX,
    INS_BASE, INS_EXTRA, SHORT_CODE_SPEC,
};
use crate::metablock::{
    BlockSwitchCursor, ContextMode, OptimalCommand, DIST_MODE_IMPLICIT_D0, DIST_MODE_RAW,
    DIST_MODE_STATIC_DICT,
};

use super::match_gen::ChunkMatches;
use super::ChunkEnv;

const COST_INF: u64 = u64::MAX >> 4;
const PACKED_COST_SHIFT: u32 = 28;
const PACKED_ID_MASK: u64 = (1u64 << PACKED_COST_SHIFT) - 1;
const MAX_PACKED_COST: u64 = (u64::MAX >> PACKED_COST_SHIFT) - 1;
const COST_SLICE_SHIFT: usize = 8; // 256 B cost-regime slices (`<= 4096` per 1 MiB chunk)

/// Maps 7-bit SWAR hit index `(swar.wrapping_mul(0x0004_0810_2040_8100) >> 57)` to `rb[0]` short-code mask.
const SWAR7_TO_SC0: [u16; 128] = {
    const MAP: [u8; 7] = [9, 7, 5, 0, 4, 6, 8];
    let (mut t, mut m) = ([0u16; 128], 0usize);
    while m < 128 {
        let (mut k, mut out) = (0usize, 0u16);
        while k < 7 {
            if (m & (1 << k)) != 0 {
                out |= 1u16 << MAP[k];
            }
            k += 1;
        }
        t[m] = out;
        m += 1;
    }
    t
};

#[derive(Default)]
pub(super) struct ChunkCostModel {
    pub(super) npostfix: u8,
    pub(super) ndirect: u8,
    lit_pref: Vec<u64>,
    iac_exp_by_cc: Vec<[[u16; 24]; 24]>,
    iac_imp_by_ic: Vec<[[u16; 16]; 8]>,
    pos_iac_tree: Vec<u8>,
    dist_alpha: usize,
    dist_trees: Vec<u16>,
    pos_dist_tree: Vec<u8>,
}

#[inline(always)]
fn nlog2_prob_q8(count: u32, total: u32, floor_q8: u16) -> u16 {
    if count == 0 || total == 0 {
        return floor_q8;
    }
    let bits = context::fast_log2f((total as f32) / (count as f32));
    ((bits * 256.0).round() as u32).clamp(256, floor_q8 as u32) as u16
}

fn fill_huffman_cost_q8(
    counts: &[u32],
    floor_q8: u16,
    out: &mut [u16],
    buf: &mut [HuffmanCodeEntry],
) {
    fill_smoothed_symbol_costs::<8>(counts, (floor_q8, 64, 128), out, buf, |c, total| {
        nlog2_prob_q8(c, total, floor_q8) as u32
    });
}

fn build_smoothed_q8_trees<Row: AsRef<[u32]>>(
    tree_counts: &[Row],
    alpha: usize,
    small_parse: bool,
    seed_small: impl Fn(&mut [u32]),
    hbuf: &mut [HuffmanCodeEntry],
    mut write_tree: impl FnMut(usize, &[u16]),
) {
    let mut global = vec![0u32; alpha];
    for tc in tree_counts {
        context::add_hist_slice(&mut global, tc.as_ref());
    }
    let (mut blended, mut costs) = (vec![0u32; alpha], vec![14 * 256u16; alpha]);
    for (ti, tc) in tree_counts.iter().enumerate() {
        blended.copy_from_slice(tc.as_ref());
        for (b, &g) in blended.iter_mut().zip(&global) {
            if *b == 0 && g > 0 {
                *b = 1;
            }
        }
        if small_parse {
            seed_small(&mut blended);
        }
        fill_huffman_cost_q8(&blended, 14 * 256 + 64, &mut costs, hbuf);
        write_tree(ti, &costs);
    }
}

impl ChunkCostModel {
    #[inline(always)]
    fn dist_sym_cost_q8(&self, dist_tree: usize, dist_ctx: usize, sym: usize) -> u64 {
        self.dist_trees[(dist_tree * 4 + dist_ctx) * self.dist_alpha + sym] as u64
    }

    pub(super) fn refine_from_metablock_plan(
        &mut self,
        env: ChunkEnv<'_>,
        prev_cmds: &[OptimalCommand],
        plan: &MetablockPlan,
    ) {
        let chunk_len = env.len();
        if chunk_len == 0 {
            return;
        }
        (self.npostfix, self.ndirect, self.dist_alpha) =
            (plan.npostfix, plan.ndirect, plan.dist_alpha);

        let mut hbuf = vec![HuffmanCodeEntry::default(); 1128];
        let mut lit_tree_costs = vec![[14 * 256u16; MAX_LIT]; plan.lit_tree_counts.len()];
        for (ti, tc) in plan.lit_tree_counts.iter().enumerate() {
            fill_huffman_cost_q8(tc, 14 * 256, &mut lit_tree_costs[ti], &mut hbuf);
        }
        let mut mode_luts = [context::context_luts(ContextMode::UTF8); 256];
        for (bt, &m) in plan.context_modes.iter().enumerate() {
            mode_luts[bt] = context::context_luts(m);
        }

        const SLICE_MASK: usize = (1 << COST_SLICE_SHIFT) - 1;
        const SLICE_HALF: usize = 1 << (COST_SLICE_SHIFT - 1);
        let num_slices = chunk_len.div_ceil(1 << COST_SLICE_SHIFT).max(1);
        self.pos_iac_tree.clear();
        self.pos_iac_tree.resize(num_slices, 0);
        self.pos_dist_tree.clear();
        self.pos_dist_tree.resize(num_slices, 0);
        self.lit_pref.resize(chunk_len + 1, 0);
        self.lit_pref[0] = 0;

        let mut acc = 0u64;
        {
            let mut lit_cur = BlockSwitchCursor::new(&plan.lit_split);
            let mut iac_cur = BlockSwitchCursor::new(&plan.iac_split);
            let mut dist_cur = BlockSwitchCursor::new(&plan.dist_split);
            let mut record_span =
                |lo: usize, hi: usize, step_lit: bool, iac_bt: u8, dist_bt: u8| {
                    for q in lo..hi {
                        let bt = if step_lit {
                            lit_cur.next_type()
                        } else {
                            lit_cur.cur_type()
                        } as usize;
                        let off_in_sl = q & SLICE_MASK;
                        if off_in_sl == 0 || off_in_sl == SLICE_HALF {
                            let sl = q >> COST_SLICE_SHIFT;
                            self.pos_iac_tree[sl] = iac_bt;
                            self.pos_dist_tree[sl] = dist_bt;
                        }
                        let (lut0, lut1) = mode_luts[bt];
                        let sp = env.start + q;
                        let ctx = context::stream_byte_ctx(lut0, lut1, env.data, sp);
                        let tree_id = plan.lit_cmap[bt * 64 + ctx] as usize;
                        acc += lit_tree_costs[tree_id][env.data[sp] as usize] as u64;
                        self.lit_pref[q + 1] = acc;
                    }
                };

            let mut p = 0usize;
            for &cmd in prev_cmds {
                let iac_bt = iac_cur.next_type();
                let ins_end = p + (cmd.insert_len as usize);
                record_span(p, ins_end, true, iac_bt, dist_cur.cur_type());
                if cmd.emits_dist_sym() {
                    dist_cur.next_type();
                }
                let copy_end = ins_end + (cmd.out_len() as usize);
                record_span(ins_end, copy_end, false, iac_bt, dist_cur.cur_type());
                p = copy_end;
            }
            record_span(p, chunk_len, true, iac_cur.cur_type(), dist_cur.cur_type());
        }

        let small_parse = prev_cmds.len() < 128;
        let nt = plan.iac_tree_counts.len();
        self.iac_exp_by_cc.resize(nt, [[0u16; 24]; 24]);
        self.iac_imp_by_ic.resize(nt, [[0u16; 16]; 8]);
        build_smoothed_q8_trees(
            &plan.iac_tree_counts,
            MAX_IAC,
            small_parse,
            |b| {
                for ic in 0..8 {
                    for cc in 0..8 {
                        let imp = iac_sym_from_codes(ic, cc, true);
                        b[imp] = b[imp].max(1);
                    }
                }
            },
            &mut hbuf,
            |ti, costs| {
                for cc in 0..24 {
                    for ic in 0..24 {
                        self.iac_exp_by_cc[ti][cc][ic] = costs[iac_sym_from_codes(ic, cc, false)];
                    }
                }
                for ic in 0..8 {
                    for cc in 0..16 {
                        self.iac_imp_by_ic[ti][ic][cc] = costs[iac_sym_from_codes(ic, cc, true)];
                    }
                }
            },
        );

        let num_dslots = plan.dist_split.num_types * 4;
        self.dist_trees.clear();
        self.dist_trees
            .resize(num_dslots * self.dist_alpha, 14 * 256);
        build_smoothed_q8_trees(
            &plan.dist_tree_counts,
            self.dist_alpha,
            small_parse,
            |b| b[..6].iter_mut().for_each(|s| *s = (*s).max(1)),
            &mut hbuf,
            |ti, costs| {
                for (slot, &cid) in plan.dist_cmap[..num_dslots].iter().enumerate() {
                    if cid as usize == ti {
                        let base = slot * self.dist_alpha;
                        self.dist_trees[base..base + self.dist_alpha].copy_from_slice(costs);
                    }
                }
            },
        );
    }
}

#[derive(Clone, Copy)]
struct BeamMeta {
    prev_end: u32,
    cmd_pos: u32,
    dist: u32,
    rb: [u32; 4],
    prev_slot: u8,
    valid_mask: u8,
    dist_mode: u8,
    static_base_len: u8,
}

impl BeamMeta {
    #[inline(always)]
    fn step(
        (rb0, vm0): ([u32; 4], u8),
        (prev_end, prev_slot): (u32, u8),
        (cmd_pos, dist, dist_mode, static_base_len): (u32, u32, u8, u8),
    ) -> Self {
        let (rb, valid_mask) = advance_dist_rb(rb0, vm0, dist, dist_mode);
        Self {
            prev_end,
            cmd_pos,
            dist,
            rb,
            prev_slot,
            valid_mask,
            dist_mode,
            static_base_len,
        }
    }

    #[inline(always)]
    fn from_slot0(bm: &[Self], sc_anc: u32, bw_shift: u32, cmd: (u32, u32, u8, u8)) -> Self {
        let anc = &bm[(sc_anc as usize) << bw_shift];
        Self::step((anc.rb, anc.valid_mask), (sc_anc, 0), cmd)
    }
}

impl Default for BeamMeta {
    #[inline(always)]
    fn default() -> Self {
        Self {
            prev_end: u32::MAX,
            cmd_pos: 0,
            dist: 0,
            rb: INITIAL_LAST_DISTANCES,
            prev_slot: 0,
            valid_mask: 0,
            dist_mode: 0,
            static_base_len: 0,
        }
    }
}

const LIT_MONO_BASE: usize = INS_BASE[6] as usize; // ins >= 6 (`ic = 6..=22`)
const LIT_IC23_BASE: usize = INS_BASE[23] as usize; // ins >= 22594 (`ic = 23`)

struct LitPareto {
    mono: Vec<(u32, i64)>,
    head: usize,
    p23: (u32, i64),
}

impl Default for LitPareto {
    fn default() -> Self {
        Self {
            mono: Vec::new(),
            head: 0,
            p23: (0, i64::MAX),
        }
    }
}

impl LitPareto {
    fn reset(&mut self) {
        self.mono.clear();
        self.head = 0;
        self.p23 = (0, i64::MAX);
    }

    #[inline(always)]
    fn push_at(&mut self, pos: usize, g_mono: Option<i64>, g_23: Option<i64>) {
        while self.head < self.mono.len() && pos - self.mono[self.head].0 as usize >= LIT_IC23_BASE
        {
            self.head += 1;
        }
        if let Some(g) = g_mono {
            while self.mono[self.head..]
                .last()
                .is_some_and(|&(_, gt)| g <= gt)
            {
                self.mono.pop();
            }
            if self.head >= 2048 && self.head * 2 >= self.mono.len() {
                self.mono.drain(..self.head);
                self.head = 0;
            }
            self.mono.push(((pos - LIT_MONO_BASE) as u32, g));
        }
        if let Some(g) = g_23 {
            if g <= self.p23.1 {
                self.p23 = ((pos - LIT_IC23_BASE) as u32, g);
            }
        }
    }

    #[inline(always)]
    fn fill_ge6(
        &self,
        pos: usize,
        lit_pref_pos: u64,
        mut best_ge: u64,
        tol: u64,
        sc_cache: &mut StartCostCache,
    ) {
        let s = &self.mono[self.head..];
        let mut prev_ic = usize::MAX;
        let mut add = |&(anc, g): &(u32, i64)| {
            let ic = ins_code_only((pos - anc as usize) as u32);
            let ge = (g.wrapping_add(lit_pref_pos as i64) as u64) + ((INS_EXTRA[ic] as u64) << 8);
            if ic == prev_ic || ge <= best_ge.saturating_add(tol) {
                sc_cache.set(ic, anc, ge);
                prev_ic = ic;
            }
            best_ge = best_ge.min(ge);
        };
        let cap = if tol <= 1536 { 16 } else { 32 };
        if s.len() <= cap {
            s.iter().rev().for_each(&mut add);
        } else {
            s[s.len() - (cap - 1)..].iter().rev().for_each(&mut add);
            add(&s[0]);
        }
        if self.p23.1 < i64::MAX {
            let ge = (self.p23.1.wrapping_add(lit_pref_pos as i64) as u64)
                + ((INS_EXTRA[23] as u64) << 8);
            if ge <= best_ge.saturating_add(tol) {
                sc_cache.set(23, self.p23.0, ge);
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
struct CarryTrack {
    rb: [u32; 4],
    valid_mask: u8,
    anchor: u32,
    slot: u8,
    g: i64,
}

impl CarryTrack {
    #[inline(always)]
    fn new(pos: usize, slot_idx: usize, g: i64, node: &BeamMeta) -> Self {
        Self {
            rb: node.rb,
            valid_mask: node.valid_mask,
            anchor: pos as u32,
            slot: slot_idx as u8,
            g,
        }
    }

    #[inline(always)]
    fn pre_cost(self, pos: usize, lit_pref_pos: u64) -> (u8, u64) {
        let ic = ins_code_only((pos - self.anchor as usize) as u32);
        let base = ((self.g + (lit_pref_pos as i64)) as u64) + ((INS_EXTRA[ic] as u64) << 8);
        (ic as u8, base)
    }

    #[inline(always)]
    fn to_meta(self, cmd_pos: u32, dist: u32, dist_mode: u8) -> BeamMeta {
        BeamMeta::step(
            (self.rb, self.valid_mask),
            (self.anchor, self.slot),
            (cmd_pos, dist, dist_mode, 0),
        )
    }
}

#[derive(Clone, Copy, Default)]
struct LitCarry {
    used: bool,
    has_short_ins: bool,
    any: CarryTrack,
    short: CarryTrack,
}

impl LitCarry {
    #[inline(always)]
    fn from_node(pos: usize, slot_idx: usize, g: i64, node: &BeamMeta) -> Self {
        let track = CarryTrack::new(pos, slot_idx, g, node);
        Self {
            used: true,
            has_short_ins: true,
            any: track,
            short: track,
        }
    }

    #[inline(always)]
    fn refresh(&mut self, pos: usize, slot_idx: usize, g: i64, node: &BeamMeta, any_tol: i64) {
        let track = CarryTrack::new(pos, slot_idx, g, node);
        let old_extra =
            (INS_EXTRA[ins_code_only((pos - self.any.anchor as usize) as u32)] as i64) << 8;
        if g <= self.any.g + old_extra + any_tol {
            self.any = track;
        }
        if !self.has_short_ins || g <= self.short.g + 128 {
            self.has_short_ins = true;
            self.short = track;
        }
    }
}

const HORIZON_LEN: usize = 512;
const HORIZON_MASK: usize = HORIZON_LEN - 1;

struct HorizonScratch {
    horizon: [u64; HORIZON_LEN],
    origins: Vec<BeamMeta>,
}

impl Default for HorizonScratch {
    fn default() -> Self {
        Self {
            horizon: [u64::MAX; HORIZON_LEN],
            origins: Vec::new(),
        }
    }
}

#[derive(Default)]
pub(super) struct OptimalParseScratch {
    beam_costs: Vec<u64>,
    beam_meta: Vec<BeamMeta>,
    hz: HorizonScratch,
    lit_pareto: LitPareto,
}

#[inline(always)]
fn swar_2byte_match_mask(win8: u64, b0: u8, b1: u8) -> u64 {
    const LO: u64 = 0x0101_0101_0101_0101;
    const MASK7F: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const HI: u64 = 0x8080_8080_8080_8080;
    let x0 = win8 ^ LO.wrapping_mul(b0 as u64);
    let nz0 = (x0 & MASK7F).wrapping_add(MASK7F) | x0;
    let x1 = win8 ^ LO.wrapping_mul(b1 as u64);
    let nz1 = (x1 & MASK7F).wrapping_add(MASK7F) | x1;
    (!nz0 & HI) & ((!nz1 & HI) >> 8)
}

#[inline(always)]
fn rb_key(bw: usize, rb: [u32; 4], valid_mask: u8) -> (u32, u32) {
    let v = |b, x| if (valid_mask & b) != 0 { x } else { 0 };
    (v(1, rb[0]), if bw > 4 { v(2, rb[1]) } else { 0 })
}

struct BeamBuf<'a, const SINGLE_BEAM: bool> {
    costs: &'a mut [u64],
    meta: &'a mut [BeamMeta],
    bw_shift: u32,
}

impl<const SINGLE_BEAM: bool> BeamBuf<'_, SINGLE_BEAM> {
    #[inline(always)]
    fn shift(&self) -> u32 {
        if SINGLE_BEAM {
            0
        } else {
            self.bw_shift & 3
        }
    }

    #[inline(always)]
    fn width(&self) -> usize {
        if SINGLE_BEAM {
            1
        } else {
            (1usize << (self.bw_shift & 3)).max(2)
        }
    }

    #[inline(always)]
    fn can_improve(&self, pos: usize, cost: u64) -> bool {
        cost < self.costs[(pos << self.shift()) + self.width() - 1]
    }

    #[inline(always)]
    fn g_at(&self, pos: usize, lit_pref: &[u64]) -> Option<i64> {
        let c = self.costs[pos << self.shift()];
        (c < COST_INF).then(|| (c as i64) - (lit_pref[pos] as i64))
    }

    #[inline(always)]
    fn insert(&mut self, target_pos: usize, cand_cost: u64, cand_meta: BeamMeta) {
        let (base, bw) = (target_pos << self.shift(), self.width());
        let costs = &mut self.costs[base..base + bw];
        if cand_cost >= costs[bw - 1] {
            return;
        }
        let metas = &mut self.meta[base..base + bw];
        let mut pos_i = bw - 1;
        if !SINGLE_BEAM {
            let (cand_d0, cand_d1) = rb_key(bw, cand_meta.rb, cand_meta.valid_mask);
            let (max_per_d0, mut same_d0_cnt) = (1 + usize::from(bw > 4), 0usize);
            for idx in 0..bw {
                if costs[idx] >= COST_INF {
                    break;
                }
                let (m_d0, m_d1) = rb_key(bw, metas[idx].rb, metas[idx].valid_mask);
                if m_d0 == cand_d0 {
                    let exact_key = bw <= 4 || m_d1 == cand_d1;
                    same_d0_cnt += usize::from(!exact_key);
                    if exact_key || same_d0_cnt >= max_per_d0 {
                        if cand_cost >= costs[idx] {
                            return;
                        }
                        pos_i = idx;
                        break;
                    }
                }
            }
        }
        costs[pos_i] = cand_cost;
        metas[pos_i] = cand_meta;
        while !SINGLE_BEAM && pos_i > 0 && costs[pos_i] < costs[pos_i - 1] {
            costs.swap(pos_i, pos_i - 1);
            metas.swap(pos_i, pos_i - 1);
            pos_i -= 1;
        }
    }

    #[inline(always)]
    fn insert_range(
        &mut self,
        pos: usize,
        lo_l: u32,
        hi_l: u32,
        cand_cost: u64,
        make_meta: impl FnOnce(&[BeamMeta]) -> BeamMeta,
    ) {
        let (t_lo, t_hi) = (pos + lo_l as usize, pos + hi_l as usize + 1);
        if (t_lo..t_hi).any(|t| self.can_improve(t, cand_cost)) {
            let cand_meta = make_meta(self.meta);
            for t in t_lo..t_hi {
                self.insert(t, cand_cost, cand_meta);
            }
        }
    }
}

#[inline(always)]
fn update_lit_carries<const SINGLE_BEAM: bool>(
    carries: &mut [LitCarry],
    pos: usize,
    costs_at_pos: &[u64],
    metas_at_pos: &[BeamMeta],
    lit_pref_pos: u64,
) {
    let bw = if SINGLE_BEAM { 1 } else { carries.len() };
    for c in &mut carries[..bw] {
        if c.used && c.has_short_ins && pos - (c.short.anchor as usize) > 9 {
            c.has_short_ins = false;
        }
    }
    for slot_idx in 0..bw {
        let ncost = costs_at_pos[slot_idx];
        if ncost >= COST_INF {
            continue;
        }
        let node = &metas_at_pos[slot_idx];
        let g = (ncost as i64) - (lit_pref_pos as i64);
        let (node_d0, node_d1) = rb_key(bw, node.rb, node.valid_mask);
        let (mut match_i, mut empty_i) = (None, None);
        let (mut worst_i, mut worst_eff) = (0usize, i64::MIN);
        let (mut same_d0_cnt, mut worst_same_d0_i, mut worst_same_d0_eff) =
            (0usize, 0usize, i64::MIN);

        for (ci, c) in carries[..bw].iter().enumerate() {
            if !c.used {
                empty_i.get_or_insert(ci);
                continue;
            }
            let (c_d0, c_d1) = rb_key(bw, c.any.rb, c.any.valid_mask);
            if SINGLE_BEAM || (c_d0 == node_d0 && (bw <= 4 || c_d1 == node_d1)) {
                match_i = Some(ci);
                break;
            }
            let ins_c = pos - (c.any.anchor as usize);
            let eff_c = c.any.g + ((INS_EXTRA[ins_code_only(ins_c as u32)] as i64) << 8);
            if c_d0 == node_d0 {
                same_d0_cnt += 1;
                if eff_c > worst_same_d0_eff {
                    (worst_same_d0_eff, worst_same_d0_i) = (eff_c, ci);
                }
            }
            if eff_c > worst_eff {
                (worst_eff, worst_i) = (eff_c, ci);
            }
        }

        let tol = if SINGLE_BEAM { 256 } else { 128 };
        if let Some(mi) = match_i {
            carries[mi].refresh(pos, slot_idx, g, node, tol);
        } else if bw > 4 && same_d0_cnt >= 2 {
            carries[worst_same_d0_i].refresh(pos, slot_idx, g, node, tol);
        } else if let Some(ei) = empty_i {
            carries[ei] = LitCarry::from_node(pos, slot_idx, g, node);
        } else if g < worst_eff {
            carries[worst_i] = LitCarry::from_node(pos, slot_idx, g, node);
        }
    }
}

pub(super) struct ChunkParseResult {
    pub(super) commands: Vec<OptimalCommand>,
    pub(super) trailing_literals: u32,
    pub(super) final_rb: [u32; 4],
    pub(super) final_valid_mask: u8,
}

impl ChunkParseResult {
    #[inline(always)]
    pub(super) fn empty(
        (final_rb, final_valid_mask): ([u32; 4], u8),
        trailing_literals: u32,
    ) -> Self {
        Self {
            commands: Vec::new(),
            trailing_literals,
            final_rb,
            final_valid_mask,
        }
    }

    #[inline(always)]
    pub(super) fn covered_len(&self) -> usize {
        self.commands
            .iter()
            .map(|c| (c.insert_len + c.out_len()) as usize)
            .sum::<usize>()
            + (self.trailing_literals as usize)
    }
}

#[derive(Clone, Copy)]
struct PendingSpan {
    cost: u64,
    lo: u32,
    hi: u32,
    meta: BeamMeta,
}

impl Default for PendingSpan {
    #[inline(always)]
    fn default() -> Self {
        Self {
            cost: COST_INF,
            lo: 0,
            hi: 0,
            meta: BeamMeta::default(),
        }
    }
}

struct HorizonDispatcher<'a> {
    s: &'a mut HorizonScratch,
    pending: [PendingSpan; 14],
    active_mask: u16,
    chunk_len: usize,
}

impl<'a> HorizonDispatcher<'a> {
    fn new(s: &'a mut HorizonScratch, chunk_len: usize) -> Self {
        s.horizon.fill(u64::MAX);
        s.origins.clear();
        s.origins.reserve(chunk_len);
        Self {
            s,
            pending: [Default::default(); 14],
            active_mask: 0,
            chunk_len,
        }
    }

    #[inline(always)]
    fn pop_at(&mut self, pos: usize) -> Option<(u64, usize)> {
        let pk = std::mem::replace(&mut self.s.horizon[pos & HORIZON_MASK], u64::MAX);
        (pk != u64::MAX).then_some((pk >> PACKED_COST_SHIFT, (pk & PACKED_ID_MASK) as usize))
    }

    #[inline(always)]
    fn emit_copy_interval<const SINGLE_BEAM: bool>(
        &mut self,
        beam: &mut BeamBuf<'_, SINGLE_BEAM>,
        pos: usize,
        cc: usize,
        (lo, hi): (u32, u32),
        cost: u64,
        make_meta: impl FnOnce(&[BeamMeta]) -> BeamMeta,
    ) {
        if cc < 10 || hi == lo {
            beam.insert_range(pos, lo, hi, cost, make_meta);
            return;
        }
        if cost >= MAX_PACKED_COST {
            return;
        }
        let idx = cc - 10;
        let cur = &mut self.pending[idx];
        if cur.cost <= cost && cur.lo <= lo && cur.hi >= hi {
            return;
        }
        let new_sp = PendingSpan {
            cost,
            lo,
            hi,
            meta: make_meta(beam.meta),
        };
        if cur.cost < COST_INF && !(cost <= cur.cost && lo <= cur.lo && hi >= cur.hi) {
            let old = std::mem::replace(cur, new_sp);
            self.emit_flushed_span(beam, pos, old);
        } else {
            *cur = new_sp;
            self.active_mask |= 1u16 << idx;
        }
    }

    #[inline(always)]
    fn emit_flushed_span<const SINGLE_BEAM: bool>(
        &mut self,
        beam: &mut BeamBuf<'_, SINGLE_BEAM>,
        pos: usize,
        sp: PendingSpan,
    ) {
        let (lo_pos, hi_pos) = (
            pos + sp.lo as usize,
            (pos + sp.hi as usize).min(self.chunk_len),
        );
        if lo_pos > hi_pos {
            return;
        }
        let horizon_cap = pos + HORIZON_LEN;
        if lo_pos <= horizon_cap {
            let oid = self.s.origins.len() as u64;
            self.s.origins.push(sp.meta);
            let cand_pk = (sp.cost << PACKED_COST_SHIFT) | oid;
            let (len, s) = (hi_pos.min(horizon_cap) - lo_pos + 1, lo_pos & HORIZON_MASK);
            let (a, b) = if s + len <= HORIZON_LEN {
                (&mut self.s.horizon[s..s + len], &mut [][..])
            } else {
                let (head, tail) = self.s.horizon.split_at_mut(s);
                (tail, &mut head[..len - (HORIZON_LEN - s)])
            };
            for x in a {
                *x = (*x).min(cand_pk);
            }
            for x in b {
                *x = (*x).min(cand_pk);
            }
        }
        let z_lo = lo_pos.max(horizon_cap + 1);
        if z_lo <= hi_pos {
            let tail_lo = z_lo.max(hi_pos.saturating_sub(7));
            if z_lo < tail_lo {
                beam.insert(z_lo, sp.cost, sp.meta);
                beam.insert((z_lo + hi_pos) >> 1, sp.cost, sp.meta);
            }
            for t in tail_lo..=hi_pos {
                beam.insert(t, sp.cost, sp.meta);
            }
        }
    }

    #[inline(always)]
    fn flush_at_pos<const SINGLE_BEAM: bool>(
        &mut self,
        beam: &mut BeamBuf<'_, SINGLE_BEAM>,
        pos: usize,
    ) {
        let mut mask = std::mem::take(&mut self.active_mask);
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let sp = std::mem::take(&mut self.pending[idx]);
            self.emit_flushed_span(beam, pos, sp);
        }
    }
}

#[derive(Default)]
struct StartCostCache {
    ic_mask: u32,
    computed_mask: u32,
    anchor: [u32; 24],
    g_extra: [u64; 24],
    cost_by_cc: [u64; 24],
    anchor_by_cc: [u32; 24],
}

impl StartCostCache {
    #[inline(always)]
    fn reset(&mut self) {
        self.ic_mask = 0;
        self.computed_mask = 0;
    }

    #[inline(always)]
    fn set(&mut self, ic: usize, anc: u32, ge: u64) {
        let bit = 1u32 << ic;
        if (self.ic_mask & bit) == 0 || ge < self.g_extra[ic] {
            self.ic_mask |= bit;
            (self.anchor[ic], self.g_extra[ic]) = (anc, ge);
        }
    }

    #[inline(always)]
    fn copy_end_cost(
        &mut self,
        cc: usize,
        (d_sym, d_nbits, _): (u16, u8, u32),
        dist_tree_idx: usize,
        model: &ChunkCostModel,
        iac_exp_tree: &[[u16; 24]; 24],
    ) -> (u64, u32) {
        if (self.computed_mask & (1u32 << cc)) == 0 {
            let row = &iac_exp_tree[cc];
            let (mut best_sc, mut best_anc, mut m) = (COST_INF, 0u32, self.ic_mask);
            while m != 0 {
                let ic = m.trailing_zeros() as usize;
                m &= m - 1;
                let c = self.g_extra[ic] + (row[ic] as u64);
                if c < best_sc {
                    (best_sc, best_anc) = (c, self.anchor[ic]);
                }
            }
            (self.cost_by_cc[cc], self.anchor_by_cc[cc]) = (best_sc, best_anc);
            self.computed_mask |= 1u32 << cc;
        }
        let d_huff_q8 = model.dist_sym_cost_q8(dist_tree_idx, cc.min(3), d_sym as usize);
        let extra_q8 = ((COPY_EXTRA[cc] + d_nbits as u32) as u64) << 8;
        (
            self.cost_by_cc[cc] + d_huff_q8 + extra_q8,
            self.anchor_by_cc[cc],
        )
    }
}

#[inline(always)]
fn probe_sc_mask<const SINGLE_BEAM: bool>(
    env: ChunkEnv<'_>,
    sp: usize,
    max_lz_dist: u32,
    (b0, b1): (u8, u8),
    (rb, vmask): ([u32; 4], u8),
    (allowed_sc, sc_lim): (u16, u8),
) -> u16 {
    let swar = |bit: u8, d: u32| -> Option<u16> {
        ((vmask & bit) != 0
            && (4..=max_lz_dist.saturating_sub(3)).contains(&d)
            && (d as usize) + 3 <= sp)
            .then(|| {
                let base = sp - (d as usize) - 3;
                let win = u64::from_le_bytes(env.data[base..base + 8].try_into().unwrap());
                let m = swar_2byte_match_mask(win, b0, b1);
                SWAR7_TO_SC0[(m.wrapping_mul(0x0004_0810_2040_8100) >> 57) as usize]
            })
    };
    let (s0, s1) = (
        swar(1, rb[0]),
        if SINGLE_BEAM { None } else { swar(2, rb[1]) },
    );
    if ((vmask & 1) != 0 && s0.is_none()) || (!SINGLE_BEAM && (vmask & 2) != 0 && s1.is_none()) {
        let mut bits = 0u16;
        for sc in 0..sc_lim {
            if resolve_short_dist_code(sc, rb, vmask, max_lz_dist)
                .is_some_and(|d| env.has_2b(sp, d, b0, b1))
            {
                bits |= 1u16 << sc;
            }
        }
        return bits;
    }
    let v2b = |bit: u8, d: u32| u16::from((vmask & bit) != 0 && env.has_2b(sp, d, b0, b1));
    let sc1 = if SINGLE_BEAM {
        v2b(2, rb[1]) << 1
    } else {
        s1.map_or(0, |x| ((x & 1) << 1) | ((x & 0x03F0) << 6))
    };
    (s0.unwrap_or(0) | sc1 | (v2b(4, rb[2]) << 2) | (v2b(8, rb[3]) << 3)) & allowed_sc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{MAX_ALLOWED_DISTANCE, STD_MAX_RAW_STATIC_DIST};
    use crate::map_to_sym::dist_alphabet_size;
    use crate::q11::test_helpers::miri_or;

    #[test]
    fn test_lit_pareto_mono_invariants_and_p23_oracle() {
        let (n, check_mod, win_lo, win_hi) = miri_or(
            (384usize, 31usize, 320usize, 350usize),
            (26_000, 197, 22_590, 22_615),
        );
        let mut raw_g = vec![None::<i64>; n];
        let mut lp = LitPareto::default();
        let mut sc = StartCostCache::default();
        let mut rng = 0xCAFE_F00D_1234_5678u64;

        for pos in 0..n {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            raw_g[pos] = if pos == 0 {
                Some(0)
            } else if (rng >> 61) == 0 {
                None
            } else {
                // Mix upward fluctuations with realistic steady downward drift so `head` advances
                // and compaction (`head >= 2048`) triggers too.
                Some((((rng >> 48) as i64) & 0x3FF) - 512 + ((pos & 1023) as i64) / 2)
            };
            if pos >= LIT_MONO_BASE {
                let g18 = raw_g[pos - LIT_MONO_BASE];
                let g23 = (pos >= LIT_IC23_BASE)
                    .then(|| raw_g[pos - LIT_IC23_BASE])
                    .flatten();
                lp.push_at(pos, g18, g23);
            }

            if pos < 96 || pos.is_multiple_of(check_mod) || (win_lo..=win_hi).contains(&pos) {
                let s = &lp.mono[lp.head..];
                for w in s.windows(2) {
                    assert!(w[0].0 < w[1].0 && w[0].1 < w[1].1);
                }
                for &(anc, g) in s {
                    let ins = pos - anc as usize;
                    assert!((LIT_MONO_BASE..LIT_IC23_BASE).contains(&ins));
                    assert_eq!(raw_g[anc as usize], Some(g));
                }
                if pos >= LIT_MONO_BASE {
                    let lo = pos.saturating_sub(LIT_IC23_BASE - 1);
                    let hi = pos - LIT_MONO_BASE;
                    let mut exp_min: Option<(u32, i64)> = None;
                    for (a, &og) in raw_g.iter().enumerate().take(hi + 1).skip(lo) {
                        if let Some(g) = og {
                            if exp_min.is_none_or(|(_, bg)| g <= bg) {
                                exp_min = Some((a as u32, g));
                            }
                        }
                    }
                    assert_eq!(s.first().copied(), exp_min, "mono min at pos={pos}");
                }

                sc.reset();
                let lit_pref_pos = (pos as u64) * 1024;
                lp.fill_ge6(pos, lit_pref_pos, COST_INF, COST_INF, &mut sc);
                if pos >= LIT_IC23_BASE {
                    let mut exp23: Option<(u32, i64)> = None;
                    for (a, &og) in raw_g.iter().enumerate().take(pos - LIT_IC23_BASE + 1) {
                        if let Some(g) = og {
                            if exp23.is_none_or(|(_, bg)| g <= bg) {
                                exp23 = Some((a as u32, g));
                            }
                        }
                    }
                    assert_eq!(
                        ((sc.ic_mask & (1 << 23)) != 0).then_some(sc.anchor[23]),
                        exp23.map(|x| x.0),
                        "p23 at pos={pos}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_swar_2byte_match_mask_exhaustive_borrow_safety() {
        for m7 in 0..128usize {
            let swar: u64 = (0..7)
                .filter(|&k| (m7 & (1 << k)) != 0)
                .map(|k| 1u64 << (7 + 8 * k))
                .sum();
            assert_eq!(
                (swar.wrapping_mul(0x0004_0810_2040_8100) >> 57) as usize,
                m7
            );
        }
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for trial in 0..miri_or(256usize, 4000) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let win8 = match trial % 4 {
                0 => 0x01_00_01_00_80_7F_01_00u64 ^ (rng & 0x00FF_0000_0000_FF00),
                1 => 0,
                2 => 0xFF_80_7F_01_02_00_01_00u64,
                _ => rng,
            };
            let bytes = win8.to_le_bytes();
            let (b0, b1) = (bytes[(trial * 3) & 7], bytes[(trial * 5 + 1) & 7]);
            let mask = swar_2byte_match_mask(win8, b0, b1);
            assert_eq!(mask & !0x0080_8080_8080_8080, 0);
            for k in 0..7 {
                assert_eq!(
                    (mask & (1u64 << (7 + 8 * k))) != 0,
                    bytes[k] == b0 && bytes[k + 1] == b1
                );
            }
        }
    }

    #[test]
    fn test_encode_raw_distance_all_npostfix_ndirect() {
        #[rustfmt::skip]
        let extra = [1008, 4096, 65535, 65536, 1_000_000, 16_777_200, MAX_ALLOWED_DISTANCE as u32];
        let test_dists: Vec<u32> = (1..=miri_or(36u32, 300)).chain(extra).collect();
        for np in 0..=3u8 {
            for k in 0..=15u8 {
                let nd = k << np;
                let (std_a, lw_a) = (
                    dist_alphabet_size(np, nd, false),
                    dist_alphabet_size(np, nd, true),
                );
                for &d in &test_dists {
                    let (sym, nbits, bits) = encode_raw_distance_params(d, np, nd);
                    assert!(
                        (sym as usize) < lw_a
                            && (d > STD_MAX_RAW_STATIC_DIST || (sym as usize) < std_a)
                    );
                    let su = sym as u32;
                    let dec = if su < 16 + nd as u32 {
                        su - 15
                    } else {
                        let c = su - 16 - nd as u32;
                        let ndistbits = 1 + (c >> (np + 1));
                        assert_eq!(ndistbits, nbits as u32);
                        let offset = ((2 + ((c >> np) & 1) as u64) << ndistbits) - 4;
                        (((offset + bits as u64) << np) as u32)
                            + (c & ((1u32 << np) - 1))
                            + (nd as u32)
                            + 1
                    };
                    assert_eq!(dec, d);
                }
            }
        }
    }
}
