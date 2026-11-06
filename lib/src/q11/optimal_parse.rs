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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{MAX_ALLOWED_DISTANCE, STD_MAX_RAW_STATIC_DIST};
    use crate::map_to_sym::dist_alphabet_size;
    use crate::q11::test_helpers::miri_or;

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
