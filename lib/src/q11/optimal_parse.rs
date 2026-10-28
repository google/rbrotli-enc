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

//! Subquadratic $O(N \log N)$ forward-DP optimal Brotli parser for `Q10` (`BEAM = 1`), `Q11`
//! (`BEAM = 2, 4`), and `Q12` (`BEAM = 8`), with a 15-level dyadic Sparse Table over literal runs,
//! a 512-slot `u64` Range-Chmin horizon ring + lazy ZKW segment tree over copy-code intervals,
//! all 16 RFC 7932 `dist_rb` short codes + implicit `d0`, and EM cost-model refinement.

use crate::block_split::MetablockPlan;
use crate::constants::{INITIAL_LAST_DISTANCES, MAX_IAC, MAX_LIT};
use crate::context;
use crate::huffman::{HuffmanCode, HuffmanCodeEntry};
use crate::map_to_sym::{
    copy_code_only, encode_raw_distance_params, iac_sym_from_codes, ins_code_only, COPY_BASE,
    COPY_EXTRA, COPY_MAX, INS_BASE, INS_EXTRA,
};
use crate::metablock::{
    ContextMode, OptimalCommand, DIST_MODE_IMPLICIT_D0, DIST_MODE_RAW, DIST_MODE_STATIC_DICT,
};

use super::match_gen::ChunkMatches;
use super::ChunkEnv;

const COST_INF: u64 = u64::MAX >> 4;
const PACKED_COST_SHIFT: u32 = 28;
const PACKED_ID_MASK: u64 = (1u64 << PACKED_COST_SHIFT) - 1;
const MAX_PACKED_COST: u64 = (u64::MAX >> PACKED_COST_SHIFT) - 1;
const COST_SLICE_SHIFT: usize = 8; // 256 B cost-regime slices (`<= 4096` per 1 MiB chunk)

#[rustfmt::skip]
const SHORT_CODE_SPEC: [(u8, i8); 16] = [
    (0, 0), (1, 0), (2, 0), (3, 0),
    (0, -1), (0, 1), (0, -2), (0, 2), (0, -3), (0, 3),
    (1, -1), (1, 1), (1, -2), (1, 2), (1, -3), (1, 3),
];

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

#[inline(always)]
pub(super) fn resolve_short_dist_code(
    sc: u8,
    rb: [u32; 4],
    valid_mask: u8,
    max_dist: u32,
) -> Option<u32> {
    let &(slot, delta) = SHORT_CODE_SPEC.get(sc as usize)?;
    if (valid_mask & (1u8 << slot)) == 0 {
        return None;
    }
    let d = (rb[slot as usize] as i64 + delta as i64).min(u32::MAX as i64);
    (1..=max_dist as i64).contains(&d).then_some(d as u32)
}

#[inline(always)]
pub(super) fn advance_dist_rb(
    rb: [u32; 4],
    valid_mask: u8,
    dist: u32,
    dist_mode: u8,
) -> ([u32; 4], u8) {
    match dist_mode {
        0 | DIST_MODE_IMPLICIT_D0 | DIST_MODE_STATIC_DICT => (rb, valid_mask),
        _ => ([dist, rb[0], rb[1], rb[2]], ((valid_mask << 1) | 1) & 0x0F),
    }
}

#[derive(Default)]
pub(super) struct ChunkCostModel {
    npostfix: u8,
    ndirect: u8,
    lit_pref: Vec<u64>,
    iac_trees: Vec<[u16; MAX_IAC]>,
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
    let total: u32 = counts.iter().sum();
    if total == 0 {
        out.fill(floor_q8);
        return;
    }
    let (_, code) = HuffmanCode::from_counts(counts, 15, buf);
    if let Some(single) = code.singleton {
        out.fill(floor_q8);
        out[single as usize] = 64;
        return;
    }
    for (i, (&c, slot)) in counts.iter().zip(out.iter_mut()).enumerate() {
        *slot = if c > 0 {
            let hlen_q8 = (code.code[i].len as u32) << 8;
            (((hlen_q8 * 7 + nlog2_prob_q8(c, total, floor_q8) as u32) >> 3) as u16).max(128)
        } else {
            floor_q8
        };
    }
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
        for (g, &c) in global.iter_mut().zip(tc.as_ref()) {
            *g += c;
        }
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
            let mut lit_cur = crate::emit::BlockSwitchCursor::new(&plan.lit_split);
            let mut iac_cur = crate::emit::BlockSwitchCursor::new(&plan.iac_split);
            let mut dist_cur = crate::emit::BlockSwitchCursor::new(&plan.dist_split);
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
                let copy_end = ins_end + (cmd.out_len as usize);
                record_span(ins_end, copy_end, false, iac_bt, dist_cur.cur_type());
                p = copy_end;
            }
            record_span(p, chunk_len, true, iac_cur.cur_type(), dist_cur.cur_type());
        }

        let small_parse = prev_cmds.len() < 128;
        let nt = plan.iac_tree_counts.len();
        self.iac_trees.resize(nt, [14 * 256; MAX_IAC]);
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
                self.iac_trees[ti].copy_from_slice(costs);
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

        let mut raw_dtrees = vec![vec![14 * 256u16; self.dist_alpha]; plan.dist_tree_counts.len()];
        build_smoothed_q8_trees(
            &plan.dist_tree_counts,
            self.dist_alpha,
            small_parse,
            |b| b[..6].iter_mut().for_each(|s| *s = (*s).max(1)),
            &mut hbuf,
            |ti, costs| raw_dtrees[ti].copy_from_slice(costs),
        );
        let num_dbt = plan.dist_split.num_types;
        self.dist_trees.clear();
        self.dist_trees
            .resize(num_dbt * 4 * self.dist_alpha, 14 * 256);
        for dbt in 0..num_dbt {
            for d_ctx in 0..4 {
                let tid = plan.dist_cmap[dbt * 4 + d_ctx] as usize;
                let base = (dbt * 4 + d_ctx) * self.dist_alpha;
                self.dist_trees[base..base + self.dist_alpha].copy_from_slice(&raw_dtrees[tid]);
            }
        }
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
    fn from_slot0<const BEAM: usize>(bm: &[Self], sc_anc: u32, cmd: (u32, u32, u8, u8)) -> Self {
        let anc = &bm[(sc_anc as usize) * BEAM];
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

const G_BIAS: i64 = 1i64 << 39;
const G_ANCHOR_BITS: u32 = 22;
const G_ANCHOR_MASK: u64 = (1u64 << G_ANCHOR_BITS) - 1;

const DYADIC_CAPS: [usize; 15] = [
    8, 16, 16, 32, 64, 128, 256, 256, 512, 1024, 2048, 2048, 4096, 8192, 8192,
];
const DYADIC_OFF: [usize; 15] = {
    let (mut off, mut acc, mut i) = ([0usize; 15], 0usize, 0);
    while i < 15 {
        off[i] = acc;
        acc += DYADIC_CAPS[i];
        i += 1;
    }
    off
};
const DYADIC_BUF_LEN: usize = DYADIC_OFF[14] + DYADIC_CAPS[14];

#[inline(always)]
fn pack_g_anchor(g: i64, anchor: u32) -> u64 {
    (((g + G_BIAS) as u64) << G_ANCHOR_BITS) | (G_ANCHOR_MASK - (anchor as u64))
}

#[inline(always)]
fn unpack_g_anchor(packed: u64) -> (u32, i64) {
    (
        (G_ANCHOR_MASK - (packed & G_ANCHOR_MASK)) as u32,
        ((packed >> G_ANCHOR_BITS) as i64) - G_BIAS,
    )
}

#[derive(Default)]
struct DyadicLitMin {
    buf: Vec<u64>,
    min_23: u64,
    active_lvls: usize,
}

impl DyadicLitMin {
    fn reset(&mut self, chunk_len: usize) {
        let (active_lvls, active_len) = if chunk_len <= 2114 {
            (11, DYADIC_OFF[11])
        } else if chunk_len <= 6210 {
            (13, DYADIC_OFF[13])
        } else {
            (15, DYADIC_BUF_LEN)
        };
        if self.buf.len() < DYADIC_BUF_LEN {
            self.buf.resize(DYADIC_BUF_LEN, u64::MAX);
        }
        self.buf[..active_len].fill(u64::MAX);
        self.min_23 = u64::MAX;
        self.active_lvls = active_lvls;
    }

    #[inline(always)]
    fn push(&mut self, pos: usize, mut v: u64, p23_packed: u64) {
        self.min_23 = self.min_23.min(p23_packed);
        let b: &mut [u64; DYADIC_BUF_LEN] = (&mut self.buf[..DYADIC_BUF_LEN]).try_into().unwrap();
        for lvl in 0..14 {
            let (mask, off) = (DYADIC_CAPS[lvl] - 1, DYADIC_OFF[lvl]);
            if lvl + 1 == self.active_lvls {
                b[off + (pos & mask)] = v;
                return;
            }
            let prev = b[off + (pos.wrapping_sub(1 << lvl) & mask)];
            b[off + (pos & mask)] = v;
            v = v.min(prev);
        }
        b[DYADIC_OFF[14] + (pos & (DYADIC_CAPS[14] - 1))] = v;
    }

    #[inline(always)]
    fn query_ic(&self, pos: usize, ic: usize) -> Option<(u32, i64)> {
        let base = INS_BASE[ic] as usize;
        if pos < base {
            return None;
        }
        let packed = if ic == 23 {
            self.min_23
        } else {
            let b: &[u64; DYADIC_BUF_LEN] = (&self.buf[..DYADIC_BUF_LEN]).try_into().unwrap();
            let lvl = INS_EXTRA[ic] as usize;
            b[DYADIC_OFF[lvl] + ((pos - base) & (DYADIC_CAPS[lvl] - 1))]
        };
        (packed != u64::MAX).then(|| unpack_g_anchor(packed))
    }
}

#[derive(Clone, Copy)]
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
    fn to_meta(self, cmd_pos: u32, dist: u32, dist_mode: u8) -> BeamMeta {
        BeamMeta::step(
            (self.rb, self.valid_mask),
            (self.anchor, self.slot),
            (cmd_pos, dist, dist_mode, 0),
        )
    }
}

impl Default for CarryTrack {
    fn default() -> Self {
        Self {
            rb: INITIAL_LAST_DISTANCES,
            valid_mask: 0,
            anchor: 0,
            slot: 0,
            g: i64::MAX / 4,
        }
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

struct HorizonRing512([u64; HORIZON_LEN]);

impl Default for HorizonRing512 {
    fn default() -> Self {
        Self([u64::MAX; HORIZON_LEN])
    }
}

impl HorizonRing512 {
    #[inline(always)]
    fn pop_at(&mut self, pos: usize) -> u64 {
        std::mem::replace(&mut self.0[pos & HORIZON_MASK], u64::MAX)
    }

    #[inline(always)]
    fn range_chmin(&mut self, lo_pos: usize, hi_pos: usize, cand_pk: u64) {
        let (len, s) = (hi_pos - lo_pos + 1, lo_pos & HORIZON_MASK);
        if s + len <= HORIZON_LEN {
            chmin_u64_slice(&mut self.0[s..s + len], cand_pk);
        } else {
            let mid = HORIZON_LEN - s;
            chmin_u64_slice(&mut self.0[s..], cand_pk);
            chmin_u64_slice(&mut self.0[..len - mid], cand_pk);
        }
    }
}

#[inline(always)]
fn chmin_u64_slice(slots: &mut [u64], cand: u64) {
    for s in slots {
        *s = (*s).min(cand);
    }
}

#[derive(Default)]
struct ZkwScratch {
    horizon: HorizonRing512,
    zkw_tree: Vec<u64>,
    origins: Vec<BeamMeta>,
}

#[derive(Default)]
pub(super) struct OptimalParseScratch {
    beam_costs: Vec<u64>,
    beam_meta: Vec<BeamMeta>,
    zkw: ZkwScratch,
    dyadic: DyadicLitMin,
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
fn rb_key<const BEAM: usize>(rb: [u32; 4], valid_mask: u8) -> (u32, u32) {
    let d0 = if (valid_mask & 1) != 0 { rb[0] } else { 0 };
    let d1 = if BEAM > 4 && (valid_mask & 2) != 0 {
        rb[1]
    } else {
        0
    };
    (d0, d1)
}

struct BeamBuf<'a, const BEAM: usize> {
    costs: &'a mut [u64],
    meta: &'a mut [BeamMeta],
}

impl<const BEAM: usize> BeamBuf<'_, BEAM> {
    #[inline(always)]
    fn can_improve(&self, pos: usize, cost: u64) -> bool {
        cost < self.costs[pos * BEAM + BEAM - 1]
    }

    #[inline(always)]
    fn packed_g_at(&self, pos: usize, lit_pref: &[u64]) -> u64 {
        let c = self.costs[pos * BEAM];
        if c < COST_INF {
            pack_g_anchor((c as i64) - (lit_pref[pos] as i64), pos as u32)
        } else {
            u64::MAX
        }
    }

    #[inline(always)]
    fn insert(&mut self, target_pos: usize, cand_cost: u64, cand_meta: BeamMeta) {
        let base = target_pos * BEAM;
        if BEAM == 1 {
            if cand_cost < self.costs[base] {
                self.costs[base] = cand_cost;
                self.meta[base] = cand_meta;
            }
            return;
        }
        let costs = &mut self.costs[base..base + BEAM];
        if cand_cost >= costs[BEAM - 1] {
            return;
        }
        let metas = &mut self.meta[base..base + BEAM];
        let (cand_d0, cand_d1) = rb_key::<BEAM>(cand_meta.rb, cand_meta.valid_mask);
        let max_per_d0 = if BEAM > 4 { 2usize } else { 1usize };
        let (mut same_d0_cnt, mut pos_i) = (0usize, BEAM - 1);
        for idx in 0..BEAM {
            if costs[idx] >= COST_INF {
                break;
            }
            let (m_d0, m_d1) = rb_key::<BEAM>(metas[idx].rb, metas[idx].valid_mask);
            if m_d0 == cand_d0 {
                let exact_key = BEAM <= 4 || m_d1 == cand_d1;
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
        costs[pos_i] = cand_cost;
        metas[pos_i] = cand_meta;
        while pos_i > 0 && costs[pos_i] < costs[pos_i - 1] {
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
        if (lo_l..=hi_l).any(|l| self.can_improve(pos + l as usize, cand_cost)) {
            let cand_meta = make_meta(self.meta);
            for l in lo_l..=hi_l {
                self.insert(pos + (l as usize), cand_cost, cand_meta);
            }
        }
    }
}

#[inline(always)]
fn range_chmin_zkw(tree: &mut [u64], cap: usize, lo: usize, hi: usize, packed: u64) {
    let (mut l, mut r) = (cap + lo, cap + hi);
    while l <= r {
        if (l & 1) == 1 {
            tree[l] = tree[l].min(packed);
            l += 1;
        }
        if (r & 1) == 0 {
            tree[r] = tree[r].min(packed);
            r -= 1;
        }
        l >>= 1;
        r >>= 1;
    }
}

#[inline(always)]
fn query_zkw_point(tree: &[u64], cap: usize, idx_pos: usize) -> u64 {
    let (mut k, mut best) = (cap + idx_pos, u64::MAX);
    while k >= 1 {
        best = best.min(tree[k]);
        k >>= 1;
    }
    best
}

#[inline(always)]
fn update_lit_carries<const BEAM: usize>(
    carries: &mut [LitCarry; BEAM],
    pos: usize,
    costs_at_pos: &[u64],
    metas_at_pos: &[BeamMeta],
    lit_pref_pos: u64,
) {
    for c in carries.iter_mut() {
        if c.used && c.has_short_ins && pos - (c.short.anchor as usize) > 9 {
            c.has_short_ins = false;
        }
    }
    for slot_idx in 0..BEAM {
        let ncost = costs_at_pos[slot_idx];
        if ncost >= COST_INF {
            continue;
        }
        let node = &metas_at_pos[slot_idx];
        let g = (ncost as i64) - (lit_pref_pos as i64);
        if BEAM == 1 {
            if !carries[0].used {
                carries[0] = LitCarry::from_node(pos, slot_idx, g, node);
            } else {
                carries[0].refresh(pos, slot_idx, g, node, 256);
            }
            continue;
        }

        let (node_d0, node_d1) = rb_key::<BEAM>(node.rb, node.valid_mask);
        let (mut match_i, mut empty_i) = (None, None);
        let (mut worst_i, mut worst_eff) = (0usize, i64::MIN);
        let (mut same_d0_cnt, mut worst_same_d0_i, mut worst_same_d0_eff) =
            (0usize, 0usize, i64::MIN);

        for (ci, c) in carries.iter().enumerate() {
            if !c.used {
                empty_i.get_or_insert(ci);
                continue;
            }
            let (c_d0, c_d1) = rb_key::<BEAM>(c.any.rb, c.any.valid_mask);
            if c_d0 == node_d0 && (BEAM <= 4 || c_d1 == node_d1) {
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

        if let Some(mi) = match_i {
            carries[mi].refresh(pos, slot_idx, g, node, 128);
        } else if BEAM > 4 && same_d0_cnt >= 2 {
            carries[worst_same_d0_i].refresh(pos, slot_idx, g, node, 128);
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
            .map(|c| (c.insert_len + c.out_len) as usize)
            .sum::<usize>()
            + (self.trailing_literals as usize)
    }
}

#[derive(Clone, Copy)]
struct PendingZkw {
    full_cost: u64,
    full_meta: BeamMeta,
    part_cost: u64,
    part_lo: u32,
    part_hi: u32,
    part_meta: BeamMeta,
}

impl Default for PendingZkw {
    #[inline(always)]
    fn default() -> Self {
        Self {
            full_cost: COST_INF,
            full_meta: BeamMeta::default(),
            part_cost: COST_INF,
            part_lo: 0,
            part_hi: 0,
            part_meta: BeamMeta::default(),
        }
    }
}

struct ZkwDispatcher<'a> {
    s: &'a mut ZkwScratch,
    pending: [PendingZkw; 14],
    active_mask: u16,
    chunk_len: usize,
    tree_cap: usize,
    zkw_initialized: bool,
    zkw_min_active: usize,
    max_zkw_reach: usize,
}

impl<'a> ZkwDispatcher<'a> {
    fn new(s: &'a mut ZkwScratch, chunk_len: usize) -> Self {
        s.horizon.0.fill(u64::MAX);
        s.origins.clear();
        s.origins.reserve(chunk_len);
        Self {
            s,
            pending: [PendingZkw::default(); 14],
            active_mask: 0,
            chunk_len,
            tree_cap: chunk_len.next_power_of_two(),
            zkw_initialized: false,
            zkw_min_active: usize::MAX,
            max_zkw_reach: 0,
        }
    }

    #[inline(always)]
    fn pop_at(&mut self, pos: usize) -> Option<(u64, usize)> {
        let mut pk = self.s.horizon.pop_at(pos);
        if self.zkw_initialized && pos >= self.zkw_min_active && pos <= self.max_zkw_reach {
            pk = pk.min(query_zkw_point(&self.s.zkw_tree, self.tree_cap, pos - 1));
        }
        (pk != u64::MAX).then_some((pk >> PACKED_COST_SHIFT, (pk & PACKED_ID_MASK) as usize))
    }

    #[inline(always)]
    fn dispatch_range(&mut self, pos: usize, lo_pos: usize, hi_pos: usize, cost: u64, oid: u32) {
        let pk = (cost << PACKED_COST_SHIFT) | (oid as u64);
        let horizon_cap = pos + HORIZON_LEN;
        if lo_pos <= horizon_cap {
            self.s
                .horizon
                .range_chmin(lo_pos, hi_pos.min(horizon_cap), pk);
        }
        let z_lo = lo_pos.max(horizon_cap + 1);
        if z_lo <= hi_pos {
            if !self.zkw_initialized {
                self.s.zkw_tree.clear();
                self.s.zkw_tree.resize(2 * self.tree_cap, u64::MAX);
                self.zkw_initialized = true;
            }
            self.zkw_min_active = self.zkw_min_active.min(z_lo);
            self.max_zkw_reach = self.max_zkw_reach.max(hi_pos);
            range_chmin_zkw(
                &mut self.s.zkw_tree,
                self.tree_cap,
                z_lo - 1,
                hi_pos - 1,
                pk,
            );
        }
    }

    #[inline(always)]
    fn push_origin_range(&mut self, pos: usize, lo_l: u32, hi_l: u32, cost: u64, meta: BeamMeta) {
        let lo_pos = pos + (lo_l as usize);
        let hi_pos = (pos + (hi_l as usize)).min(self.chunk_len);
        if lo_pos <= hi_pos {
            let oid = self.s.origins.len() as u32;
            self.s.origins.push(meta);
            self.dispatch_range(pos, lo_pos, hi_pos, cost, oid);
        }
    }

    #[inline(always)]
    fn emit_copy_interval<const BEAM: usize>(
        &mut self,
        beam: &mut BeamBuf<'_, BEAM>,
        pos: usize,
        cc: usize,
        (lo_l, hi_l): (u32, u32),
        cost: u64,
        make_meta: impl FnOnce(&[BeamMeta]) -> BeamMeta,
    ) {
        if cc < 10 || hi_l == lo_l {
            beam.insert_range(pos, lo_l, hi_l, cost, make_meta);
            return;
        }
        if cost >= MAX_PACKED_COST {
            return;
        }
        let idx = cc - 10;
        if lo_l == COPY_BASE[cc] && hi_l == COPY_MAX[cc] {
            if cost < self.pending[idx].full_cost {
                self.pending[idx].full_cost = cost;
                self.pending[idx].full_meta = make_meta(beam.meta);
                self.active_mask |= 1u16 << idx;
            }
        } else if cost < self.pending[idx].full_cost {
            if self.pending[idx].part_cost == COST_INF {
                let p = &mut self.pending[idx];
                (p.part_cost, p.part_lo, p.part_hi, p.part_meta) =
                    (cost, lo_l, hi_l, make_meta(beam.meta));
                self.active_mask |= 1u16 << idx;
            } else if self.pending[idx].part_lo == lo_l && self.pending[idx].part_hi == hi_l {
                if cost < self.pending[idx].part_cost {
                    self.pending[idx].part_cost = cost;
                    self.pending[idx].part_meta = make_meta(beam.meta);
                }
            } else {
                let old = self.pending[idx];
                self.push_origin_range(pos, old.part_lo, old.part_hi, old.part_cost, old.part_meta);
                let p = &mut self.pending[idx];
                (p.part_cost, p.part_lo, p.part_hi, p.part_meta) =
                    (cost, lo_l, hi_l, make_meta(beam.meta));
            }
        }
    }

    #[inline(always)]
    fn flush_at_pos(&mut self, pos: usize) {
        let mut mask = std::mem::take(&mut self.active_mask);
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let cc = idx + 10;
            let p = self.pending[idx];
            (self.pending[idx].full_cost, self.pending[idx].part_cost) = (COST_INF, COST_INF);
            if p.full_cost < COST_INF {
                self.push_origin_range(pos, COPY_BASE[cc], COPY_MAX[cc], p.full_cost, p.full_meta);
            }
            if p.part_cost < p.full_cost {
                self.push_origin_range(pos, p.part_lo, p.part_hi, p.part_cost, p.part_meta);
            }
        }
    }
}

struct StartCostCache {
    computed_mask: u32,
    bucket_anchor: [u32; 24],
    bucket_g_extra: [u64; 24],
    cost_by_cc: [u64; 24],
    anchor_by_cc: [u32; 24],
}

impl StartCostCache {
    #[inline(always)]
    fn copy_end_cost(
        &mut self,
        cc: usize,
        (d_sym, d_nbits, _): (u16, u8, u32),
        dist_tree_idx: usize,
        model: &ChunkCostModel,
        iac_exp_tree: &[[u16; 24]; 24],
    ) -> (u64, u32) {
        let (sc_cost, sc_anc) = if (self.computed_mask & (1u32 << cc)) != 0 {
            (self.cost_by_cc[cc], self.anchor_by_cc[cc])
        } else {
            let (mut best_sc, mut best_anc) = (COST_INF, 0u32);
            for ((&bg, &anc), &iac) in self
                .bucket_g_extra
                .iter()
                .zip(&self.bucket_anchor)
                .zip(&iac_exp_tree[cc])
            {
                let c = bg + (iac as u64);
                if c < best_sc {
                    (best_sc, best_anc) = (c, anc);
                }
            }
            (self.cost_by_cc[cc], self.anchor_by_cc[cc]) = (best_sc, best_anc);
            self.computed_mask |= 1u32 << cc;
            (best_sc, best_anc)
        };
        let d_huff_q8 = model.dist_sym_cost_q8(dist_tree_idx, cc.min(3), d_sym as usize);
        let extra_q8 = ((COPY_EXTRA[cc] + d_nbits as u32) as u64) << 8;
        (sc_cost + d_huff_q8 + extra_q8, sc_anc)
    }
}

pub(super) fn parse_chunk_optimal<const BEAM: usize>(
    env: ChunkEnv<'_>,
    matches: &ChunkMatches,
    model: &ChunkCostModel,
    init: ([u32; 4], u8),
    scratch: &mut OptimalParseScratch,
) -> ChunkParseResult {
    assert!(BEAM == 1 || BEAM == 2 || BEAM == 4 || BEAM == 8);
    let chunk_len = env.len();
    if chunk_len == 0 {
        return ChunkParseResult::empty(init, 0);
    }

    let total_nodes = (chunk_len + 1) * BEAM;
    scratch.beam_costs.clear();
    scratch.beam_costs.resize(total_nodes, COST_INF);
    if scratch.beam_meta.len() < total_nodes {
        scratch.beam_meta.resize(total_nodes, BeamMeta::default());
    }
    let mut beam = BeamBuf::<BEAM> {
        costs: &mut scratch.beam_costs,
        meta: &mut scratch.beam_meta,
    };
    let mut zkw = ZkwDispatcher::new(&mut scratch.zkw, chunk_len);
    let dyadic = &mut scratch.dyadic;
    dyadic.reset(chunk_len);

    beam.costs[0] = 0;
    beam.meta[0] = BeamMeta {
        rb: init.0,
        valid_mask: init.1,
        ..BeamMeta::default()
    };

    let mut carries = [LitCarry::default(); BEAM];
    let mut sc_cache = StartCostCache {
        computed_mask: 0,
        bucket_anchor: [0; 24],
        bucket_g_extra: [COST_INF; 24],
        cost_by_cc: [COST_INF; 24],
        anchor_by_cc: [0; 24],
    };
    let mut cont_cache = [(u32::MAX, 0u32, 0u32); 64];
    let cont_cap = if BEAM > 4 { 64usize } else { 32usize };
    let mut cont_ring_idx = 0usize;
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
    let mut cpre = [CarryPre::default(); BEAM];
    let mut unique_dists = [UniqueDist {
        d: 0,
        mlen: 0,
        exp_sc: [u8::MAX; 8],
        imp_mask: 0,
    }; 64];
    let (allowed_sc, sc_lim): (u16, u8) = match BEAM {
        1 => (0x003F, 6),
        2 => (0x0FFF, 12),
        _ => (0xFFFF, 16),
    };

    for pos in 0..=chunk_len {
        if pos >= 1 {
            if let Some((cost, orig_id)) = zkw.pop_at(pos) {
                if beam.can_improve(pos, cost) {
                    beam.insert(pos, cost, zkw.s.origins[orig_id]);
                }
            }
        }
        if pos == chunk_len {
            break;
        }

        let lit_pref_pos = model.lit_pref[pos];
        let p23 = if pos >= 22594 {
            beam.packed_g_at(pos - 22594, &model.lit_pref)
        } else {
            u64::MAX
        };
        dyadic.push(pos, beam.packed_g_at(pos, &model.lit_pref), p23);

        let base_idx = pos * BEAM;
        update_lit_carries::<BEAM>(
            &mut carries,
            pos,
            &beam.costs[base_idx..base_idx + BEAM],
            &beam.meta[base_idx..base_idx + BEAM],
            lit_pref_pos,
        );

        if chunk_len - pos < 2 {
            continue;
        }

        let ivs = matches.at(pos);
        let stream_pos = env.start + pos;
        let (lo_margin, hi_margin) = match BEAM {
            1 => (2, 4),
            8 => (8, 8),
            _ => (4, 4),
        };
        if pos >= long_skip_start + lo_margin
            && pos + hi_margin < long_skip_end
            && (BEAM == 1
                || ivs
                    .last()
                    .is_none_or(|top| pos + (top.l_hi as usize) <= long_skip_end))
        {
            continue;
        }
        if let Some(top_iv) = ivs.last() {
            let min_skip_l = if BEAM == 1 || BEAM == 8 { 128 } else { 64 };
            let cand_end = pos + (top_iv.l_hi as usize);
            let extends = if BEAM == 1 {
                pos + 4 >= long_skip_end || cand_end > long_skip_end + 8
            } else {
                cand_end > long_skip_end
            };
            if (BEAM != 8 || top_iv.dist == 1)
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
            let (b0, b1) = (env.data[stream_pos], env.data[stream_pos + 1]);
            let mut n_ud = 0usize;

            let swar_sc0 = |bit: u8, d: u32, vmask: u8| -> Option<u16> {
                ((vmask & bit) != 0
                    && d >= 4
                    && d <= max_lz_dist.saturating_sub(3)
                    && (d as usize) + 3 <= stream_pos)
                    .then(|| {
                        let base = stream_pos - (d as usize) - 3;
                        let win = u64::from_le_bytes(env.data[base..base + 8].try_into().unwrap());
                        let m = swar_2byte_match_mask(win, b0, b1);
                        SWAR7_TO_SC0[(m.wrapping_mul(0x0004_0810_2040_8100) >> 57) as usize]
                    })
            };
            let mut get_ud = |unique_dists: &mut [UniqueDist; 64], d: u32| -> usize {
                if let Some(k) = unique_dists[..n_ud].iter().position(|ud| ud.d == d) {
                    return k;
                }
                if n_ud >= unique_dists.len() {
                    return unique_dists.len();
                }
                let pos_u32 = pos as u32;
                let mut mlen = cont_cache[..cont_cap]
                    .iter()
                    .find(|&&(lp, ld, ll)| ld == d && pos_u32 > lp && pos_u32 + 1 < lp + ll)
                    .map_or(0, |&(lp, _, ll)| lp + ll - pos_u32);
                if mlen == 0 {
                    mlen = env.match_len(pos, d);
                    if mlen >= 3 {
                        if let Some(slot) = cont_cache[..cont_cap].iter_mut().find(|e| e.1 == d) {
                            *slot = (pos_u32, d, mlen);
                        } else {
                            cont_cache[cont_ring_idx] = (pos_u32, d, mlen);
                            cont_ring_idx = (cont_ring_idx + 1) & (cont_cap - 1);
                        }
                    }
                }
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

            for ci in 0..BEAM {
                let carry = carries[ci];
                if !carry.used {
                    continue;
                }
                let mut any_d0_matched: Option<bool> = None;

                if carry.any.valid_mask != 0 {
                    let (vmask, rb) = (carry.any.valid_mask, carry.any.rb);
                    let swar0 = swar_sc0(1, rb[0], vmask);
                    let swar1 = if BEAM > 1 {
                        swar_sc0(2, rb[1], vmask)
                    } else {
                        None
                    };
                    let need_slow = ((vmask & 1) != 0 && swar0.is_none())
                        || (BEAM > 1 && (vmask & 2) != 0 && swar1.is_none());

                    let mut sc_hits = if !need_slow {
                        if let Some(s0) = swar0 {
                            any_d0_matched = Some((s0 & 1) != 0);
                        }
                        let valid_2b = |bit: u8, d: u32| -> u16 {
                            u16::from(
                                (vmask & bit) != 0
                                    && (1..=max_lz_dist).contains(&d)
                                    && env.has_2b(stream_pos, d, b0, b1),
                            )
                        };
                        let sc1_bits = if BEAM > 1 {
                            swar1.map_or(0, |s0| ((s0 & 1) << 1) | ((s0 & 0x03F0) << 6))
                        } else {
                            valid_2b(2, rb[1]) << 1
                        };
                        (swar0.unwrap_or(0)
                            | sc1_bits
                            | (valid_2b(4, rb[2]) << 2)
                            | (valid_2b(8, rb[3]) << 3))
                            & allowed_sc
                    } else {
                        let mut bits = 0u16;
                        for sc in 0..sc_lim {
                            if let Some(d) = resolve_short_dist_code(sc, rb, vmask, max_lz_dist) {
                                let hit = env.has_2b(stream_pos, d, b0, b1);
                                if sc == 0 {
                                    any_d0_matched = Some(hit);
                                }
                                bits |= u16::from(hit) << sc;
                            }
                        }
                        bits
                    };

                    if sc_hits != 0 {
                        let ic_any = ins_code_only((pos - carry.any.anchor as usize) as u32);
                        cpre[ci].ic_any = ic_any as u8;
                        cpre[ci].base_any = ((carry.any.g + (lit_pref_pos as i64)) as u64)
                            + ((INS_EXTRA[ic_any] as u64) << 8);
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
                        any_d0_matched.unwrap_or_else(|| {
                            (1..=max_lz_dist).contains(&d_imp)
                                && env.has_2b(stream_pos, d_imp, b0, b1)
                        })
                    } else {
                        (1..=max_lz_dist).contains(&d_imp) && env.has_2b(stream_pos, d_imp, b0, b1)
                    };
                    if imp_hit {
                        let ic_s = ins_code_only((pos - carry.short.anchor as usize) as u32);
                        cpre[ci].ic_short = ic_s as u8;
                        cpre[ci].base_short = ((carry.short.g + (lit_pref_pos as i64)) as u64)
                            + ((INS_EXTRA[ic_s] as u64) << 8);
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

                    for ci in 0..BEAM {
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
                        zkw.emit_copy_interval(&mut beam, pos, cc, (lo_l, hi_l), best_c, |_| {
                            let tr = if best_mode == DIST_MODE_IMPLICIT_D0 {
                                carries[best_ci].short
                            } else {
                                carries[best_ci].any
                            };
                            tr.to_meta(pos as u32, d, best_mode)
                        });
                    }
                }
            }
        }

        let static_cands = matches.static_at(pos);
        if !ivs.is_empty() || !static_cands.is_empty() {
            let mut has_any_ic = false;
            let direct_lim = 6.min(pos + 1);
            for ic in 0..direct_lim {
                let anc = pos - ic;
                let bc = beam.costs[anc * BEAM];
                if bc < COST_INF {
                    sc_cache.bucket_anchor[ic] = anc as u32;
                    sc_cache.bucket_g_extra[ic] = bc + (lit_pref_pos - model.lit_pref[anc]);
                    has_any_ic = true;
                } else {
                    sc_cache.bucket_g_extra[ic] = COST_INF;
                }
            }
            if direct_lim < 6 {
                sc_cache.bucket_g_extra[direct_lim..].fill(COST_INF);
            } else {
                for ic in 6..24usize {
                    if let Some((a, g)) = dyadic.query_ic(pos, ic) {
                        sc_cache.bucket_anchor[ic] = a;
                        sc_cache.bucket_g_extra[ic] =
                            ((g + (lit_pref_pos as i64)) as u64) + ((INS_EXTRA[ic] as u64) << 8);
                        has_any_ic = true;
                    } else {
                        sc_cache.bucket_g_extra[ic] = COST_INF;
                        if pos < INS_BASE[ic] as usize {
                            sc_cache.bucket_g_extra[ic + 1..].fill(COST_INF);
                            break;
                        }
                    }
                }
            }

            if has_any_ic {
                sc_cache.computed_mask = 0;
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
                        zkw.emit_copy_interval(&mut beam, pos, cc, (lo_l, hi_l), end_cost, |bm| {
                            BeamMeta::from_slot0::<BEAM>(bm, sc_anc, (pos_u, d, DIST_MODE_RAW, 0))
                        });
                    }
                }

                for sm in static_cands {
                    let cc = copy_code_only(sm.base_len as u32);
                    let dp = encode_raw_distance_params(sm.raw_dist, model.npostfix, model.ndirect);
                    let (end_cost, sc_anc) =
                        sc_cache.copy_end_cost(cc, dp, dist_tree_idx, model, iac_exp_tree);
                    let (out_l, rd, bl) = (sm.out_len as u32, sm.raw_dist, sm.base_len);
                    beam.insert_range(pos, out_l, out_l, end_cost, |bm| {
                        BeamMeta::from_slot0::<BEAM>(
                            bm,
                            sc_anc,
                            (pos_u, rd, DIST_MODE_STATIC_DICT, bl),
                        )
                    });
                }
            }
        }

        zkw.flush_at_pos(pos);
    }

    let lit_end_total = model.lit_pref[chunk_len];
    let last_iac_tree = &model.iac_trees[*model.pos_iac_tree.last().unwrap_or(&0) as usize];
    let (mut best_final_cost, mut best_end_pos, mut best_end_slot) = (COST_INF, 0usize, 0usize);

    for end_pos in 0..=chunk_len {
        let base_idx = end_pos * BEAM;
        for (s_idx, &ncost) in beam.costs[base_idx..base_idx + BEAM].iter().enumerate() {
            if ncost >= COST_INF {
                continue;
            }
            let total_c = if end_pos == chunk_len {
                ncost
            } else {
                let ic = ins_code_only((chunk_len - end_pos) as u32);
                let trail_cmd_q8 = (last_iac_tree[iac_sym_from_codes(ic, 2, false)] as u64)
                    + ((INS_EXTRA[ic] as u64) << 8);
                ncost + (lit_end_total - model.lit_pref[end_pos]) + trail_cmd_q8
            };
            if total_c < best_final_cost {
                (best_final_cost, best_end_pos, best_end_slot) = (total_c, end_pos, s_idx);
            }
        }
    }

    let final_meta = beam.meta[best_end_pos * BEAM + best_end_slot];
    let trailing_literals = (chunk_len - best_end_pos) as u32;
    let mut commands = Vec::new();
    let (mut cur_pos, mut cur_slot) = (best_end_pos, best_end_slot);
    while cur_pos > 0 {
        let node = beam.meta[cur_pos * BEAM + cur_slot];
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
        commands.push(OptimalCommand {
            insert_len: (cmd_pos - prev_end) as u32,
            copy_len,
            out_len,
            dist: node.dist,
            dist_mode: node.dist_mode,
        });
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
    let chunk_len = env.len();
    if chunk_len == 0 {
        return ChunkParseResult::empty((rb, vmask), 0);
    }

    let prefix = &env.data[env.start..env.start + chunk_len.min(4096)];
    let boost_static = is_q11
        && context::choose_context_mode_modes(prefix, &context::ALL_CONTEXT_MODES[..2])
            == ContextMode::UTF8;
    let static_ov = if boost_static { 72i32 } else { 96i32 };

    let d_bits8 = |d: u32| (encode_raw_distance_params(d, 0, 0).1 as i32) * 8;
    let eval_at = |p: usize, anchor: usize, rb: [u32; 4], vmask: u8| {
        let sp = env.start + p;
        let max_lz_dist = env.max_lz_dist(sp);
        let (b0, b1, ins_len) = (env.data[sp], env.data[sp + 1], p - anchor);
        let mut best = (0i32, 0u32, 0u32, 0u32, 0u8);
        let consider = |best: &mut (i32, u32, u32, u32, u8),
                        (out_len, copy_len, dist, dist_mode): (u32, u32, u32, u8),
                        ov: i32| {
            let gain =
                (out_len as i32) * 44 - (ov + (COPY_EXTRA[copy_code_only(copy_len)] as i32) * 8);
            if gain > best.0 {
                *best = (gain, out_len, copy_len, dist, dist_mode);
            }
        };

        for (sc, &d) in rb.iter().enumerate() {
            if (vmask & (1u8 << sc)) != 0
                && (1..=max_lz_dist).contains(&d)
                && env.has_2b(sp, d, b0, b1)
            {
                let mlen = env.match_len(p, d);
                let min_req = if (d as usize) > sp {
                    4
                } else if sc == 0 {
                    2
                } else {
                    3
                };
                if mlen >= min_req {
                    let is_imp = sc == 0 && ins_len <= 9 && copy_code_only(mlen) < 16;
                    let (mode, ov) = if is_imp {
                        (DIST_MODE_IMPLICIT_D0, 44)
                    } else {
                        (sc as u8, 72 + (sc as i32) * 6)
                    };
                    consider(&mut best, (mlen, mlen, d, mode), ov);
                }
            }
        }
        for iv in matches.at(p) {
            let base_ov = if vmask == 0 && iv.dist <= 64 { 56 } else { 88 };
            consider(
                &mut best,
                (iv.l_hi, iv.l_hi, iv.dist, DIST_MODE_RAW),
                base_ov + d_bits8(iv.dist),
            );
        }
        if boost_static || best.1 < 7 {
            for sm in matches.static_at(p) {
                let cand = (
                    sm.out_len as u32,
                    sm.base_len as u32,
                    sm.raw_dist,
                    DIST_MODE_STATIC_DICT,
                );
                consider(&mut best, cand, static_ov + d_bits8(sm.raw_dist));
            }
        }
        best
    };

    let mut commands = Vec::with_capacity(chunk_len / 8);
    let (mut anchor, mut pos, mut cached) = (0usize, 0usize, None);
    while pos + 2 <= chunk_len {
        let mut cur = cached
            .take()
            .unwrap_or_else(|| eval_at(pos, anchor, rb, vmask));
        if cur.0 <= 0 {
            pos += 1;
            continue;
        }
        if cur.1 < 32 && pos + 3 <= chunk_len {
            let nxt = eval_at(pos + 1, anchor, rb, vmask);
            if nxt.0 > cur.0 + 10 {
                cached = Some(nxt);
                pos += 1;
                continue;
            }
        }
        let ins_len = (pos - anchor) as u32;
        if cur.4 == DIST_MODE_IMPLICIT_D0 && (ins_len > 9 || copy_code_only(cur.2) >= 16) {
            cur.4 = 0;
        }
        commands.push(OptimalCommand {
            insert_len: ins_len,
            copy_len: cur.2,
            out_len: cur.1,
            dist: cur.3,
            dist_mode: cur.4,
        });
        (rb, vmask) = advance_dist_rb(rb, vmask, cur.3, cur.4);
        pos += cur.1 as usize;
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
    use crate::block_split::build_metablock_plan_with_effort;
    use crate::constants::{MAX_ALLOWED_DISTANCE, STD_MAX_RAW_STATIC_DIST};
    use crate::map_to_sym::dist_alphabet_size;
    use crate::q11::match_gen::{tests::build_test_matches, ChunkMatches};

    const INIT_RB: ([u32; 4], u8) = (INITIAL_LAST_DISTANCES, 0x0F);

    fn test_plan(env: ChunkEnv<'_>, res: &ChunkParseResult, effort: u8) -> MetablockPlan {
        build_metablock_plan_with_effort(
            env.data,
            env.start,
            env.end,
            &res.commands,
            res.trailing_literals,
            false,
            effort,
            1,
        )
    }

    fn setup_dp(buf: &[u8]) -> (ChunkEnv<'_>, ChunkMatches, ChunkCostModel) {
        let env = ChunkEnv::new(buf, 0, buf.len(), 1 << 20, &[]);
        let cm = build_test_matches(env, 11);
        let res0 = parse_chunk_greedy_seed(env, &cm, INIT_RB, true);
        let mut model = ChunkCostModel::default();
        model.refine_from_metablock_plan(env, &res0.commands, &test_plan(env, &res0, 11));
        (env, cm, model)
    }

    fn check_opt<const BEAM: usize>(
        env: ChunkEnv<'_>,
        cm: &ChunkMatches,
        model: &ChunkCostModel,
        sc: &mut OptimalParseScratch,
    ) -> ChunkParseResult {
        let res = parse_chunk_optimal::<BEAM>(env, cm, model, INIT_RB, sc);
        assert_eq!(res.covered_len(), env.len());
        res
    }

    #[test]
    fn test_dyadic_lit_min_matches_brute_force_oracle() {
        let n = 25_000usize;
        let (mut raw_g, mut packed_hist) = (vec![i64::MAX; n], vec![u64::MAX; n]);
        let mut dyadic = DyadicLitMin::default();
        dyadic.reset(n);

        let mut rng = 0xCAFE_F00D_1234_5678u64;
        for pos in 0..n {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let p = if pos == 0 {
                raw_g[pos] = 0;
                pack_g_anchor(0, 0)
            } else if (rng >> 61) == 0 {
                u64::MAX
            } else {
                let g = (((rng >> 48) as i64) & 0x3FF) - 512;
                raw_g[pos] = g;
                pack_g_anchor(g, pos as u32)
            };
            packed_hist[pos] = p;
            let p23 = if pos >= 22594 {
                packed_hist[pos - 22594]
            } else {
                u64::MAX
            };
            dyadic.push(pos, p, p23);

            if pos < 160 || pos % 197 == 0 || (22590..=22610).contains(&pos) {
                for (ic, &base_u32) in INS_BASE.iter().enumerate() {
                    let (got, base) = (dyadic.query_ic(pos, ic), base_u32 as usize);
                    if pos < base {
                        assert_eq!(got, None);
                        continue;
                    }
                    let max_ins = INS_BASE.get(ic + 1).map_or(usize::MAX, |&b| b as usize - 1);
                    let (lo, hi) = (pos.saturating_sub(max_ins), pos - base);
                    let mut exp: Option<(u32, i64)> = None;
                    for (a, &g) in raw_g.iter().enumerate().take(hi + 1).skip(lo) {
                        if g != i64::MAX && exp.is_none_or(|(_, bg)| g <= bg) {
                            exp = Some((a as u32, g));
                        }
                    }
                    assert_eq!(got, exp, "pos={pos} ic={ic}");
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
        for trial in 0..4000usize {
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
        let test_dists: Vec<u32> = (1..=300).chain(extra).collect();
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
        let rle = vec![b'A'; 65_536];
        let (env_r, cm_r, model_r) = setup_dp(&rle);
        assert!(
            check_opt::<4>(env_r, &cm_r, &model_r, &mut sc)
                .commands
                .len()
                <= 2
        );

        let prose: Vec<u8> = (0..80u32)
            .flat_map(|i| {
                format!(
                    "Information about the government of the United States and University research: {{\"idx\":{i},\"ok\":true}}\n"
                )
                .into_bytes()
            })
            .collect();
        let (env_p, cm_p, mut model_p) = setup_dp(&prose);
        let pass1 = check_opt::<4>(env_p, &cm_p, &model_p, &mut sc);
        model_p.refine_from_metablock_plan(env_p, &pass1.commands, &test_plan(env_p, &pass1, 11));
        check_opt::<4>(env_p, &cm_p, &model_p, &mut sc);
    }

    #[test]
    fn test_horizon_ring512_and_zkw_oracle_parity() {
        let (n, tree_cap) = (4096usize, 4096usize);
        let mut oracle_tree = vec![u64::MAX; 2 * tree_cap];
        let mut s_zkw = ZkwScratch::default();
        let mut disp = ZkwDispatcher::new(&mut s_zkw, n);

        let (mut rng, mut next_oid) = (0xD1CE_BEEF_4242_9999u64, 0u32);
        for pos in 0..=n {
            if pos >= 1 {
                let exp_pk = query_zkw_point(&oracle_tree, tree_cap, pos - 1);
                let exp = (exp_pk != u64::MAX).then_some((
                    exp_pk >> PACKED_COST_SHIFT,
                    (exp_pk & PACKED_ID_MASK) as usize,
                ));
                assert_eq!(disp.pop_at(pos), exp, "pos={pos}");
            }
            if pos == n {
                break;
            }
            let base_cost = if (1800..2100).contains(&pos) {
                200_000
            } else {
                (pos as u64) * 3_000_000
            };
            for _ in 0..=(((rng >> 60) as usize) & 3) {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let (lo_l, hi_l) = match (rng >> 56) & 7 {
                    0 => (1, 17),
                    1..=4 => {
                        let a = 18 + ((rng as usize) % 490);
                        (a, (a + 1 + (((rng >> 16) as usize) % 120)).min(512))
                    }
                    5 => (400, 620),
                    _ => (520, 950),
                };
                let (lo_pos, hi_pos) = (pos + lo_l, (pos + hi_l).min(n));
                if lo_pos <= hi_pos {
                    let (cost, oid) = (base_cost + (((rng >> 32) & 7) * 64), next_oid);
                    next_oid += 1;
                    let pk = (cost << PACKED_COST_SHIFT) | (oid as u64);
                    range_chmin_zkw(&mut oracle_tree, tree_cap, lo_pos - 1, hi_pos - 1, pk);
                    disp.dispatch_range(pos, lo_pos, hi_pos, cost, oid);
                }
            }
        }

        let block600 =
            b"LongRepeatedStructuredBlock_0123456789_ABCDEFGHIJKLMNOPQRSTUVWXYZ_".repeat(10);
        let mut buf = block600.clone();
        for i in 0..120u32 {
            buf.extend_from_slice(
                format!("entry={i} the quick brown fox jumps over the lazy dog; ").as_bytes(),
            );
        }
        buf.extend_from_slice(&block600);

        let (env, cm, mut model) = setup_dp(&buf);
        let mut sc = OptimalParseScratch::default();
        check_opt::<1>(env, &cm, &model, &mut sc);
        check_opt::<2>(env, &cm, &model, &mut sc);
        let r4 = check_opt::<4>(env, &cm, &model, &mut sc);
        model.refine_from_metablock_plan(env, &r4.commands, &test_plan(env, &r4, 12));
        check_opt::<8>(env, &cm, &model, &mut sc);
    }
}
