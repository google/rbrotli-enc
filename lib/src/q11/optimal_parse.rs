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

pub(super) fn parse_chunk_optimal(
    beam_width: usize,
    env: ChunkEnv<'_>,
    matches: &ChunkMatches,
    model: &ChunkCostModel,
    init: ([u32; 4], u8),
    scratch: &mut OptimalParseScratch,
) -> ChunkParseResult {
    if beam_width == 1 {
        parse_chunk_optimal_impl::<true>(1, env, matches, model, init, scratch)
    } else {
        parse_chunk_optimal_impl::<false>(beam_width, env, matches, model, init, scratch)
    }
}

fn parse_chunk_optimal_impl<const SINGLE_BEAM: bool>(
    beam_width: usize,
    env: ChunkEnv<'_>,
    matches: &ChunkMatches,
    model: &ChunkCostModel,
    init: ([u32; 4], u8),
    scratch: &mut OptimalParseScratch,
) -> ChunkParseResult {
    assert!(matches!(beam_width, 1 | 2 | 4 | 8));
    let bw = if SINGLE_BEAM {
        1
    } else {
        beam_width.clamp(2, 8)
    };
    let bw_shift = if SINGLE_BEAM {
        0
    } else {
        beam_width.trailing_zeros().min(3)
    };
    let chunk_len = env.len();
    if chunk_len == 0 {
        return ChunkParseResult::empty(init, 0);
    }

    let total_nodes = (chunk_len + 1) << bw_shift;
    scratch.beam_costs.clear();
    scratch.beam_costs.resize(total_nodes, COST_INF);
    if scratch.beam_meta.len() < total_nodes {
        scratch.beam_meta.resize(total_nodes, BeamMeta::default());
    }
    let mut beam = BeamBuf::<SINGLE_BEAM> {
        costs: &mut scratch.beam_costs,
        meta: &mut scratch.beam_meta,
        bw_shift,
    };
    let mut hz = HorizonDispatcher::new(&mut scratch.hz, chunk_len);
    let lit_pareto = &mut scratch.lit_pareto;
    lit_pareto.reset();

    beam.costs[0] = 0;
    beam.meta[0] = BeamMeta {
        rb: init.0,
        valid_mask: init.1,
        ..BeamMeta::default()
    };

    let mut carries_buf = [LitCarry::default(); 8];
    let carries = &mut carries_buf[..bw];
    let mut sc_cache = StartCostCache::default();
    let mut cont_cache = [(u32::MAX, 0u32, 0u32); 16];
    let (mut long_skip_start, mut long_skip_end) = (0usize, 0usize);

    #[derive(Clone, Copy, Default)]
    struct CarryPre {
        ic_any: u8,
        base_any: u64,
        ic_short: u8,
        base_short: u64,
    }
    #[derive(Clone, Copy)]
    struct UniqueDist {
        d: u32,
        mlen: u32,
        exp_sc: [u8; 8],
        imp_mask: u8,
    }
    let mut cpre_buf = [CarryPre::default(); 8];
    let cpre = &mut cpre_buf[..bw];
    let mut unique_dists = [UniqueDist {
        d: 0,
        mlen: 0,
        exp_sc: [u8::MAX; 8],
        imp_mask: 0,
    }; 64];
    let (allowed_sc, sc_lim) = match bw {
        1 => (0x003Fu16, 6u8),
        2 => (0x0FFF, 12),
        _ => (0xFFFF, 16),
    };
    let (lo_margin, hi_margin) = match bw {
        1 => (2, 4),
        8 => (8, 8),
        _ => (4, 4),
    };
    let min_skip_l = if SINGLE_BEAM || bw == 8 { 128 } else { 64 };

    for pos in 0..=chunk_len {
        if pos >= 1 {
            if let Some((cost, orig_id)) = hz.pop_at(pos) {
                if beam.can_improve(pos, cost) {
                    beam.insert(pos, cost, hz.s.origins[orig_id]);
                }
            }
        }
        if pos == chunk_len {
            break;
        }

        let lit_pref_pos = model.lit_pref[pos];
        if pos >= LIT_MONO_BASE {
            let g18 = beam.g_at(pos - LIT_MONO_BASE, &model.lit_pref);
            let g23 = (pos >= LIT_IC23_BASE)
                .then(|| beam.g_at(pos - LIT_IC23_BASE, &model.lit_pref))
                .flatten();
            lit_pareto.push_at(pos, g18, g23);
        }

        let base_idx = pos << bw_shift;
        update_lit_carries::<SINGLE_BEAM>(
            carries,
            pos,
            &beam.costs[base_idx..base_idx + bw],
            &beam.meta[base_idx..base_idx + bw],
            lit_pref_pos,
        );

        if chunk_len - pos < 2 {
            continue;
        }

        let ivs = matches.at(pos);
        let stream_pos = env.start + pos;
        if pos >= long_skip_start + lo_margin
            && pos + hi_margin < long_skip_end
            && (SINGLE_BEAM
                || ivs
                    .last()
                    .is_none_or(|top| pos + (top.l_hi as usize) <= long_skip_end))
        {
            continue;
        }
        if let Some(top_iv) = ivs.last() {
            let cand_end = pos + (top_iv.l_hi as usize);
            let extends = if SINGLE_BEAM {
                pos + 4 >= long_skip_end || cand_end > long_skip_end + 8
            } else {
                cand_end > long_skip_end
            };
            if (SINGLE_BEAM || bw != 8 || top_iv.dist == 1)
                && top_iv.l_hi >= min_skip_l
                && (top_iv.dist as usize) <= stream_pos
                && extends
            {
                (long_skip_start, long_skip_end) = (pos, cand_end);
            }
        }

        let cost_slice = pos >> COST_SLICE_SHIFT;
        let iac_tree_idx = model.pos_iac_tree[cost_slice] as usize;
        let iac_exp_tree = &model.iac_exp_by_cc[iac_tree_idx];
        let iac_imp_tree = &model.iac_imp_by_ic[iac_tree_idx];
        let dist_tree_idx = model.pos_dist_tree[cost_slice] as usize;

        let max_lz_dist = env.max_lz_dist(stream_pos);
        if max_lz_dist >= 1 {
            let bp = (env.data[stream_pos], env.data[stream_pos + 1]);
            let mut n_ud = 0usize;
            let mut get_ud = |unique_dists: &mut [UniqueDist; 64], d: u32| -> usize {
                if let Some(k) = unique_dists[..n_ud].iter().position(|ud| ud.d == d) {
                    return k;
                }
                if n_ud >= unique_dists.len() {
                    return unique_dists.len();
                }
                let (pos_u32, slot) = (
                    pos as u32,
                    &mut cont_cache[(d as usize).wrapping_mul(0x9E37) & 15],
                );
                let mlen = if slot.1 == d && pos_u32 > slot.0 && pos_u32 + 1 < slot.0 + slot.2 {
                    slot.0 + slot.2 - pos_u32
                } else {
                    let l = env.match_len(pos, d);
                    if l >= 3 {
                        *slot = (pos_u32, d, l);
                    }
                    l
                };
                let idx = n_ud;
                unique_dists[idx] = UniqueDist {
                    d,
                    mlen,
                    exp_sc: [u8::MAX; 8],
                    imp_mask: 0,
                };
                n_ud += 1;
                idx
            };

            for ci in 0..bw {
                let carry = carries[ci];
                if !carry.used {
                    continue;
                }
                let mut any_d0_hit = false;
                if carry.any.valid_mask != 0 {
                    let rb = carry.any.rb;
                    let mut sc_hits = probe_sc_mask::<SINGLE_BEAM>(
                        env,
                        stream_pos,
                        max_lz_dist,
                        bp,
                        (rb, carry.any.valid_mask),
                        (allowed_sc, sc_lim),
                    );
                    any_d0_hit = (sc_hits & 1) != 0;
                    if sc_hits != 0 {
                        (cpre[ci].ic_any, cpre[ci].base_any) =
                            carry.any.pre_cost(pos, lit_pref_pos);
                        while sc_hits != 0 {
                            let sc = sc_hits.trailing_zeros() as u8;
                            sc_hits &= sc_hits - 1;
                            let (slot, delta) = SHORT_CODE_SPEC[sc as usize];
                            let d = rb[slot as usize].wrapping_add_signed(delta as i32);
                            let u_idx = get_ud(&mut unique_dists, d);
                            if u_idx < unique_dists.len()
                                && unique_dists[u_idx].exp_sc[ci] == u8::MAX
                            {
                                unique_dists[u_idx].exp_sc[ci] = sc;
                            }
                        }
                    }
                }

                if carry.has_short_ins && (carry.short.valid_mask & 1) != 0 {
                    let d_imp = carry.short.rb[0];
                    let imp_hit = if (carry.any.valid_mask & 1) != 0 && d_imp == carry.any.rb[0] {
                        any_d0_hit
                    } else {
                        env.has_2b(stream_pos, d_imp, bp.0, bp.1)
                    };
                    if imp_hit {
                        (cpre[ci].ic_short, cpre[ci].base_short) =
                            carry.short.pre_cost(pos, lit_pref_pos);
                        let u_idx = get_ud(&mut unique_dists, d_imp);
                        if u_idx < unique_dists.len() {
                            unique_dists[u_idx].imp_mask |= 1u8 << ci;
                        }
                    }
                }
            }

            for ud in &unique_dists[..n_ud] {
                let d = ud.d;
                let min_l = if (d as usize) > stream_pos {
                    4u32
                } else {
                    2u32
                };
                if ud.mlen < min_l {
                    continue;
                }
                for cc in copy_code_only(min_l)..=copy_code_only(ud.mlen) {
                    let (lo_l, hi_l) = (COPY_BASE[cc].max(min_l), COPY_MAX[cc].min(ud.mlen));
                    if lo_l > hi_l {
                        continue;
                    }
                    let copy_extra_q8 = (COPY_EXTRA[cc] as u64) << 8;
                    let d_ctx = cc.min(3);
                    let exp_iac_row = &iac_exp_tree[cc];
                    let (mut best_c, mut best_ci, mut best_mode) = (COST_INF, 0usize, 0u8);

                    for ci in 0..bw {
                        let sc = ud.exp_sc[ci];
                        if sc != u8::MAX {
                            let c = cpre[ci].base_any
                                + (exp_iac_row[cpre[ci].ic_any as usize] as u64)
                                + copy_extra_q8
                                + model.dist_sym_cost_q8(dist_tree_idx, d_ctx, sc as usize);
                            if c < best_c {
                                (best_c, best_ci, best_mode) = (c, ci, sc);
                            }
                        }
                        if cc < 16 && (ud.imp_mask & (1u8 << ci)) != 0 {
                            let c = cpre[ci].base_short
                                + (iac_imp_tree[cpre[ci].ic_short as usize][cc] as u64)
                                + copy_extra_q8;
                            if c < best_c {
                                (best_c, best_ci, best_mode) = (c, ci, DIST_MODE_IMPLICIT_D0);
                            }
                        }
                    }
                    if best_c < COST_INF {
                        hz.emit_copy_interval::<SINGLE_BEAM>(
                            &mut beam,
                            pos,
                            cc,
                            (lo_l, hi_l),
                            best_c,
                            |_| {
                                let tr = if best_mode == DIST_MODE_IMPLICIT_D0 {
                                    carries[best_ci].short
                                } else {
                                    carries[best_ci].any
                                };
                                tr.to_meta(pos as u32, d, best_mode)
                            },
                        );
                    }
                }
            }
        }

        let static_cands = matches.static_at(pos);
        if !ivs.is_empty() || !static_cands.is_empty() {
            sc_cache.reset();
            let mut best_ge = COST_INF;
            for ic in 0..LIT_MONO_BASE.min(pos + 1) {
                let anc = pos - ic;
                let bc = beam.costs[anc << bw_shift];
                if bc < COST_INF {
                    let ge = bc + (lit_pref_pos - model.lit_pref[anc]);
                    sc_cache.ic_mask |= 1u32 << ic;
                    (sc_cache.anchor[ic], sc_cache.g_extra[ic]) = (anc as u32, ge);
                    best_ge = best_ge.min(ge);
                }
            }
            if pos >= LIT_MONO_BASE {
                let tol = if SINGLE_BEAM { 1536 } else { 2816 };
                lit_pareto.fill_ge6(pos, lit_pref_pos, best_ge, tol, &mut sc_cache);
            }

            if sc_cache.ic_mask != 0 {
                let pos_u = pos as u32;
                for iv in ivs {
                    let d = iv.dist;
                    let dp = encode_raw_distance_params(d, model.npostfix, model.ndirect);
                    for cc in copy_code_only(iv.l_lo)..=copy_code_only(iv.l_hi) {
                        let (lo_l, hi_l) = (COPY_BASE[cc].max(iv.l_lo), COPY_MAX[cc].min(iv.l_hi));
                        if lo_l > hi_l {
                            continue;
                        }
                        let (end_cost, sc_anc) =
                            sc_cache.copy_end_cost(cc, dp, dist_tree_idx, model, iac_exp_tree);
                        hz.emit_copy_interval::<SINGLE_BEAM>(
                            &mut beam,
                            pos,
                            cc,
                            (lo_l, hi_l),
                            end_cost,
                            |bm| {
                                BeamMeta::from_slot0(
                                    bm,
                                    sc_anc,
                                    bw_shift,
                                    (pos_u, d, DIST_MODE_RAW, 0),
                                )
                            },
                        );
                    }
                }

                for sm in static_cands {
                    let cc = copy_code_only(sm.base_len as u32);
                    let dp = encode_raw_distance_params(sm.raw_dist, model.npostfix, model.ndirect);
                    let (end_cost, sc_anc) =
                        sc_cache.copy_end_cost(cc, dp, dist_tree_idx, model, iac_exp_tree);
                    let (out_l, rd, bl) = (sm.out_len as u32, sm.raw_dist, sm.base_len);
                    beam.insert_range(pos, out_l, out_l, end_cost, |bm| {
                        BeamMeta::from_slot0(
                            bm,
                            sc_anc,
                            bw_shift,
                            (pos_u, rd, DIST_MODE_STATIC_DICT, bl),
                        )
                    });
                }
            }
        }

        hz.flush_at_pos(&mut beam, pos);
    }

    let lit_end_total = model.lit_pref[chunk_len];
    let last_iac_cc2 = &model.iac_exp_by_cc[*model.pos_iac_tree.last().unwrap_or(&0) as usize][2];
    let (mut best_final_cost, mut best_end_pos, mut best_end_slot) = (COST_INF, 0usize, 0usize);

    for end_pos in 0..=chunk_len {
        let base_idx = end_pos << bw_shift;
        for (s_idx, &ncost) in beam.costs[base_idx..base_idx + bw].iter().enumerate() {
            if ncost >= COST_INF {
                continue;
            }
            let total_c = if end_pos == chunk_len {
                ncost
            } else {
                let ic = ins_code_only((chunk_len - end_pos) as u32);
                let trail_cmd_q8 = (last_iac_cc2[ic] as u64) + ((INS_EXTRA[ic] as u64) << 8);
                ncost + (lit_end_total - model.lit_pref[end_pos]) + trail_cmd_q8
            };
            if total_c < best_final_cost {
                (best_final_cost, best_end_pos, best_end_slot) = (total_c, end_pos, s_idx);
            }
        }
    }

    let final_meta = beam.meta[(best_end_pos << bw_shift) + best_end_slot];
    let trailing_literals = (chunk_len - best_end_pos) as u32;
    let mut commands = Vec::new();
    let (mut cur_pos, mut cur_slot) = (best_end_pos, best_end_slot);
    while cur_pos > 0 {
        let node = beam.meta[(cur_pos << bw_shift) + cur_slot];
        if node.prev_end == u32::MAX {
            break;
        }
        let (cmd_pos, prev_end) = (node.cmd_pos as usize, node.prev_end as usize);
        let out_len = (cur_pos - cmd_pos) as u32;
        let copy_len = if node.dist_mode == DIST_MODE_STATIC_DICT {
            node.static_base_len as u32
        } else {
            out_len
        };
        commands.push(OptimalCommand::new(
            (cmd_pos - prev_end) as u32,
            copy_len,
            out_len,
            node.dist,
            node.dist_mode,
        ));
        (cur_pos, cur_slot) = (prev_end, node.prev_slot as usize);
    }
    commands.reverse();

    let res = ChunkParseResult {
        commands,
        trailing_literals,
        final_rb: final_meta.rb,
        final_valid_mask: final_meta.valid_mask,
    };
    debug_assert_eq!(res.covered_len(), chunk_len);
    res
}

/// Fast single-pass greedy seed parser over prebuilt `ChunkMatches` for Pass-0 histograms and
/// cross-chunk `dist_rb` prediction.
pub(super) fn parse_chunk_greedy_seed(
    env: ChunkEnv<'_>,
    matches: &ChunkMatches,
    (mut rb, mut vmask): ([u32; 4], u8),
    is_q11: bool,
) -> ChunkParseResult {
    let (chunk_len, slice) = (env.len(), env.slice());
    let boost_static = is_q11
        && context::choose_context_mode_modes(
            &slice[..slice.len().min(65_536)],
            &context::ALL_CONTEXT_MODES[..2],
        ) == ContextMode::UTF8;
    let static_ov = if boost_static { 72i32 } else { 96i32 };
    let d_bits8 = |d: u32| (encode_raw_distance_params(d, 0, 0).1 as i32) * 8;
    let eval_at = |p: usize, anchor: usize, rb: [u32; 4], vmask: u8| {
        let (sp, ins_len) = (env.start + p, (p - anchor) as u32);
        let mut best = (0i32, 0u32, 0u32, 0u32, 0u8);
        let consider =
            |best: &mut (i32, u32, u32, u32, u8), (ol, cl, d, dm): (u32, u32, u32, u8), ov: i32| {
                let gain = (ol as i32) * 44 - (ov + (COPY_EXTRA[copy_code_only(cl)] as i32) * 8);
                if gain > best.0 {
                    *best = (gain, ol, cl, d, dm);
                }
            };
        let mut hits = probe_sc_mask::<true>(
            env,
            sp,
            env.max_lz_dist(sp),
            (env.data[sp], env.data[sp + 1]),
            (rb, vmask),
            (0x000F, 4),
        );
        while hits != 0 {
            let sc = hits.trailing_zeros() as usize;
            hits &= hits - 1;
            let d = rb[sc];
            let mlen = env.match_len(p, d);
            if mlen
                >= if (d as usize) > sp {
                    4
                } else {
                    3 - u32::from(sc == 0)
                }
            {
                let (dm, ov) = if sc == 0 && can_use_implicit_d0(ins_len, mlen) {
                    (DIST_MODE_IMPLICIT_D0, 44)
                } else {
                    (sc as u8, 72 + (sc as i32) * 6)
                };
                consider(&mut best, (mlen, mlen, d, dm), ov);
            }
        }
        for iv in matches.at(p) {
            consider(
                &mut best,
                (iv.l_hi, iv.l_hi, iv.dist, DIST_MODE_RAW),
                if vmask == 0 && iv.dist <= 64 { 56 } else { 88 } + d_bits8(iv.dist),
            );
        }
        if boost_static || best.1 < 7 {
            for sm in matches.static_at(p) {
                consider(
                    &mut best,
                    (
                        sm.out_len as u32,
                        sm.base_len as u32,
                        sm.raw_dist,
                        DIST_MODE_STATIC_DICT,
                    ),
                    static_ov + d_bits8(sm.raw_dist),
                );
            }
        }
        best
    };

    let mut commands = Vec::with_capacity(chunk_len / 8);
    let (mut anchor, mut pos, mut cached) = (0usize, 0usize, None);
    while pos + 2 <= chunk_len {
        let mut best = cached
            .take()
            .unwrap_or_else(|| eval_at(pos, anchor, rb, vmask));
        if best.0 <= 0 {
            pos += 1;
            continue;
        }
        if best.1 < 32 && pos + 3 <= chunk_len {
            let nxt = eval_at(pos + 1, anchor, rb, vmask);
            if nxt.0 > best.0 + 10 {
                cached = Some(nxt);
                pos += 1;
                continue;
            }
        }
        let ins_len = (pos - anchor) as u32;
        if best.4 == DIST_MODE_IMPLICIT_D0 && !can_use_implicit_d0(ins_len, best.2) {
            best.4 = 0;
        }
        commands.push(OptimalCommand::new(ins_len, best.2, best.1, best.3, best.4));
        (rb, vmask) = advance_dist_rb(rb, vmask, best.3, best.4);
        pos += best.1 as usize;
        anchor = pos;
    }

    ChunkParseResult {
        commands,
        trailing_literals: (chunk_len - anchor) as u32,
        final_rb: rb,
        final_valid_mask: vmask,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_split::{try_plan_metablock, PlanOpts};
    use crate::constants::{MAX_ALLOWED_DISTANCE, STD_MAX_RAW_STATIC_DIST};
    use crate::map_to_sym::dist_alphabet_size;
    use crate::q11::match_gen::{tests::build_test_matches, ChunkMatches};
    use crate::q11::test_helpers::miri_or;

    const INIT_RB: ([u32; 4], u8) = (INITIAL_LAST_DISTANCES, 0x0F);

    fn test_plan(env: ChunkEnv<'_>, res: &ChunkParseResult, effort: u8) -> MetablockPlan {
        try_plan_metablock(
            env.data,
            env.start,
            env.end,
            &res.commands,
            res.trailing_literals,
            PlanOpts::q11(false, effort, 1),
        )
        .unwrap()
    }

    fn setup_dp(buf: &[u8]) -> (ChunkEnv<'_>, ChunkMatches, ChunkCostModel) {
        let env = ChunkEnv::new(buf, 0, buf.len(), 1 << 20, &[]);
        let cm = build_test_matches(env, miri_or(10, 11));
        let res0 = parse_chunk_greedy_seed(env, &cm, INIT_RB, true);
        let mut model = ChunkCostModel::default();
        model.refine_from_metablock_plan(env, &res0.commands, &test_plan(env, &res0, 11));
        (env, cm, model)
    }

    fn check_opt(
        bw: usize,
        env: ChunkEnv<'_>,
        cm: &ChunkMatches,
        model: &ChunkCostModel,
        sc: &mut OptimalParseScratch,
    ) -> ChunkParseResult {
        let res = parse_chunk_optimal(bw, env, cm, model, INIT_RB, sc);
        assert_eq!(res.covered_len(), env.len());
        res
    }

    #[test]
    fn test_lit_pareto_mono_invariants_and_p23_oracle() {
        let n = miri_or(23_000usize, 26_000);
        let (head_chk, stride, tail_chk) = miri_or(
            (32usize, 4099usize, 22593..=22597),
            (160, 197, 22590..=22615),
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

            if pos < head_chk || pos % stride == 0 || tail_chk.contains(&pos) {
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
        for trial in 0..miri_or(128usize, 4000) {
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
        let test_dists: Vec<u32> = (1..=miri_or(16, 300)).chain(extra).collect();
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

    #[test]
    fn test_optimal_parse_rle_and_prose_roundtrip_accounting() {
        let mut sc = OptimalParseScratch::default();
        let rle = vec![b'A'; miri_or(96, 65_536)];
        let (env_r, cm_r, model_r) = setup_dp(&rle);
        assert!(
            check_opt(miri_or(2, 4), env_r, &cm_r, &model_r, &mut sc)
                .commands
                .len()
                <= 2
        );

        let prose: Vec<u8> = (0..miri_or(1u32, 80))
            .flat_map(|i| {
                format!(
                    "Information about the government of the United States and University research: {{\"idx\":{i},\"ok\":true}}\n"
                )
                .into_bytes()
            })
            .collect();
        let (env_p, cm_p, mut model_p) = setup_dp(&prose);
        let bw = miri_or(2, 4);
        let pass1 = check_opt(bw, env_p, &cm_p, &model_p, &mut sc);
        model_p.refine_from_metablock_plan(env_p, &pass1.commands, &test_plan(env_p, &pass1, 11));
        check_opt(miri_or(8, 4), env_p, &cm_p, &model_p, &mut sc);
    }

    #[test]
    fn test_horizon_ring512_oracle_parity_and_long_span_dp() {
        let n = miri_or(640usize, 4096);
        let mut oracle = vec![u64::MAX; n + 1];
        let mut s_hz = HorizonScratch::default();
        let mut disp = HorizonDispatcher::new(&mut s_hz, n);
        let (mut dc, mut dm) = (vec![COST_INF; n + 1], vec![BeamMeta::default(); n + 1]);
        let mut dummy_beam = BeamBuf::<true> {
            costs: &mut dc,
            meta: &mut dm,
            bw_shift: 0,
        };
        let (mut rng, mut next_oid) = (0xD1CE_BEEF_4242_9999u64, 0u32);
        for pos in 0..=n {
            if pos >= 1 {
                let exp_pk = oracle[pos];
                let exp = (exp_pk != u64::MAX).then_some((
                    exp_pk >> PACKED_COST_SHIFT,
                    (exp_pk & PACKED_ID_MASK) as usize,
                ));
                assert_eq!(disp.pop_at(pos), exp, "pos={pos}");
            }
            if pos == n {
                break;
            }
            for _ in 0..=(((rng >> 60) as usize) & 3) {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let a = 1 + ((rng as usize) % 350);
                let b = (a + (((rng >> 16) as usize) % 250)).min(HORIZON_LEN);
                let (lo_pos, hi_pos) = (pos + a, (pos + b).min(n));
                if lo_pos <= hi_pos {
                    let cost = pos as u64 * 1024 + ((rng >> 32) & 7);
                    let pk = (cost << PACKED_COST_SHIFT) | (next_oid as u64);
                    next_oid += 1;
                    oracle[lo_pos..=hi_pos]
                        .iter_mut()
                        .for_each(|x| *x = (*x).min(pk));
                    disp.emit_flushed_span(
                        &mut dummy_beam,
                        pos,
                        PendingSpan {
                            cost,
                            lo: a as u32,
                            hi: b as u32,
                            meta: BeamMeta::default(),
                        },
                    );
                }
            }
        }

        let block600 = b"LongRepeatedStructuredBlock_0123456789_ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
            .repeat(miri_or(8, 10));
        let mut buf = block600.clone();
        for i in 0..miri_or(1u32, 120) {
            buf.extend_from_slice(
                format!("entry={i} the quick brown fox jumps over the lazy dog; ").as_bytes(),
            );
        }
        buf.extend_from_slice(&block600);

        let (env, cm, mut model) = setup_dp(&buf);
        let mut sc = OptimalParseScratch::default();
        check_opt(1, env, &cm, &model, &mut sc);
        if !cfg!(miri) {
            check_opt(2, env, &cm, &model, &mut sc);
            let r4 = check_opt(4, env, &cm, &model, &mut sc);
            model.refine_from_metablock_plan(env, &r4.commands, &test_plan(env, &r4, 12));
            check_opt(8, env, &cm, &model, &mut sc);
        }
    }
}
