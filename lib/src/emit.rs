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
#![allow(dead_code)]

//! Generalized RFC 7932 multi-block, multi-context, `(NPOSTFIX, NDIRECT)` metablock emitter.

use lsb_bitwriter::BitWriter;

use crate::block_split::{BlockSplit, MetablockPlan};
use crate::context;
use crate::huffman::{ContextMap, HuffmanCode, HuffmanCodeEntry};
use crate::map_to_sym::{copy_code_only, encode_iac, encode_raw_distance_params};
use crate::metablock::{step_block_type_mtf, BlockTypeInfo, Header, OptimalCommand};

pub(crate) struct BlockSwitchCursor<'a> {
    split: &'a BlockSplit,
    block_idx: usize,
    rem: usize,
    cur_type: usize,
    prev: usize,
    second_prev: usize,
}

impl<'a> BlockSwitchCursor<'a> {
    #[inline(always)]
    pub(crate) fn new(split: &'a BlockSplit) -> Self {
        let rem = if split.num_types <= 1 {
            usize::MAX
        } else {
            split.lengths[0]
        };
        Self {
            split,
            block_idx: 0,
            rem,
            cur_type: split.types.first().copied().unwrap_or(0) as usize,
            prev: 0,
            second_prev: 1,
        }
    }

    #[inline(always)]
    pub(crate) fn cur_type(&self) -> u8 {
        self.cur_type as u8
    }

    #[inline(always)]
    fn try_switch(&mut self) -> bool {
        if self.rem > 0 || self.block_idx + 1 >= self.split.lengths.len() {
            return false;
        }
        self.block_idx += 1;
        self.cur_type = self.split.types[self.block_idx] as usize;
        self.rem = self.split.lengths[self.block_idx];
        true
    }

    #[inline(always)]
    pub(crate) fn next_type(&mut self) -> u8 {
        self.try_switch();
        self.rem = self.rem.saturating_sub(1);
        self.cur_type as u8
    }

    #[inline(always)]
    fn step(&mut self, bti: &BlockTypeInfo<'_>, bw: &mut BitWriter<'_>) -> usize {
        self.advance_if_needed(bti, bw);
        self.rem -= 1;
        self.cur_type
    }

    #[inline(always)]
    fn advance_if_needed(&mut self, bti: &BlockTypeInfo<'_>, bw: &mut BitWriter<'_>) -> bool {
        if !self.try_switch() {
            return false;
        }
        debug_assert!(self.split.num_types >= 2);
        let t_sym = step_block_type_mtf(
            self.cur_type,
            self.split.num_types,
            &mut self.prev,
            &mut self.second_prev,
        );
        let (tn, tb) = bti.block_type_code.symbol_info(t_sym);
        bw.write(tn, tb);
        let (cn, cb) = bti.encode_block_count(self.rem);
        bw.write(cn, cb);
        true
    }
}

/// Returns `true` if emitting `data[mb_start..mb_end]` as an `ISUNCOMPRESSED = 1` metablock saves
/// bits over the compressed `MetablockPlan`.
#[inline]
pub(crate) fn should_emit_uncompressed_plan(mb_len: usize, plan: &MetablockPlan) -> bool {
    mb_len >= 64 && plan.exact_compressed_bits > (mb_len as u64) * 8 + 40
}

/// Writes a compressed metablock (`ISUNCOMPRESSED = 0`) described by `plan` and `(commands,
/// trailing_literals)`. Never byte-pads, so it can be written either directly to the outer stream
/// `BitWriter` or into a relocatable per-worker `BitWriter` (`finalize_unpadded`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_compressed_metablock(
    data: &[u8],
    mb_start: usize,
    mb_end: usize,
    commands: &[OptimalCommand],
    trailing_literals: u32,
    plan: &MetablockPlan,
    is_last_metablock: bool,
    bw: &mut BitWriter<'_>,
) {
    let mb_len = mb_end - mb_start;
    assert!(mb_len >= 1);
    let start_bits = bw.total_bits();

    let split_cap = |s: &BlockSplit| {
        if s.num_types >= 2 {
            s.num_types + 28
        } else {
            0
        }
    };
    let arena_cap = split_cap(&plan.lit_split)
        + split_cap(&plan.iac_split)
        + split_cap(&plan.dist_split)
        + plan.lit_tree_counts.len() * 256
        + plan.iac_tree_counts.len() * 704
        + plan.dist_tree_counts.len() * plan.dist_alpha;

    let mut arena = vec![HuffmanCodeEntry::default(); arena_cap];
    let mut rem_arena: &mut [HuffmanCodeEntry] = &mut arena[..];
    let mut alloc_code = |counts: &[u32]| {
        let (rest, code) = HuffmanCode::from_counts(counts, 15, std::mem::take(&mut rem_arena));
        rem_arena = rest;
        code
    };
    let mut build_bti = |split: &BlockSplit| {
        if split.num_types <= 1 {
            BlockTypeInfo::default()
        } else {
            BlockTypeInfo {
                num: split.num_types as u64,
                block_type_code: alloc_code(&split.type_counts),
                block_count_code: alloc_code(&split.len_counts),
                first_num: split.lengths[0],
            }
        }
    };

    let (lit_bti, iac_bti, dist_bti) = (
        build_bti(&plan.lit_split),
        build_bti(&plan.iac_split),
        build_bti(&plan.dist_split),
    );
    let lit_codes: Vec<_> = plan.lit_tree_counts.iter().map(|c| alloc_code(c)).collect();
    let iac_codes: Vec<_> = plan.iac_tree_counts.iter().map(|c| alloc_code(c)).collect();
    let dist_codes: Vec<_> = plan
        .dist_tree_counts
        .iter()
        .map(|c| alloc_code(c))
        .collect();
    debug_assert_eq!(rem_arena.len(), 0);

    let header = Header {
        islast: is_last_metablock,
        len: mb_len,
        isuncompressed: false,
        literals: lit_bti,
        insert_and_copy: iac_bti,
        distance: dist_bti,
        npostfix: plan.npostfix,
        ndirect: plan.ndirect,
        context_mode: &plan.context_modes,
        literals_codes: &lit_codes,
        literals_cmap: ContextMap::new(&plan.lit_cmap),
        insert_and_copy_codes: &iac_codes,
        distance_codes: &dist_codes,
        distance_cmap: ContextMap::new(&plan.dist_cmap),
    };
    header.write(bw);

    let mut lit_cur = BlockSwitchCursor::new(&plan.lit_split);
    let mut iac_cur = BlockSwitchCursor::new(&plan.iac_split);
    let mut dist_cur = BlockSwitchCursor::new(&plan.dist_split);

    let (mut lut0, mut lut1) = context::context_luts(plan.context_modes[0]);
    let mut lit_cmap_base = 0usize;
    let mut stream_pos = mb_start;

    let mut emit_iac_and_literals = |insert_len: u32,
                                     copy_len: u32,
                                     implicit_d0: bool,
                                     sp_out: &mut usize,
                                     bw: &mut BitWriter<'_>| {
        let iac_bt = iac_cur.step(&header.insert_and_copy, bw);
        let (iac_sym, iac_eb_n, iac_eb_b) = encode_iac(insert_len, copy_len, implicit_d0);
        let (hn, hb) = header.insert_and_copy_codes[iac_bt].symbol_info(iac_sym as usize);
        bw.write(hn, hb);
        if iac_eb_n > 0 {
            bw.write(iac_eb_n as usize, iac_eb_b);
        }

        let mut sp = *sp_out;
        let ins_end = sp + (insert_len as usize);
        while sp < ins_end {
            if lit_cur.advance_if_needed(&header.literals, bw) {
                let bt = lit_cur.cur_type() as usize;
                (lut0, lut1) = context::context_luts(plan.context_modes[bt]);
                lit_cmap_base = bt * 64;
            }
            if sp >= 2 && ins_end - sp >= 3 && lit_cur.rem >= 3 {
                let fast_n = ((ins_end - sp).min(lit_cur.rem)) / 3;
                let (mut p2, mut p1) = (data[sp - 2], data[sp - 1]);
                for _ in 0..fast_n {
                    let (b0, b1, b2) = (data[sp], data[sp + 1], data[sp + 2]);
                    let t0 = plan.lit_cmap
                        [lit_cmap_base + (lut0[p1 as usize] | lut1[p2 as usize]) as usize];
                    let t1 = plan.lit_cmap
                        [lit_cmap_base + (lut0[b0 as usize] | lut1[p1 as usize]) as usize];
                    let t2 = plan.lit_cmap
                        [lit_cmap_base + (lut0[b1 as usize] | lut1[b0 as usize]) as usize];
                    let (n0, c0) = header.literals_codes[t0 as usize].symbol_info(b0 as usize);
                    let (n1, c1) = header.literals_codes[t1 as usize].symbol_info(b1 as usize);
                    let (n2, c2) = header.literals_codes[t2 as usize].symbol_info(b2 as usize);
                    bw.write(n0 + n1 + n2, c0 | (c1 << n0) | (c2 << (n0 + n1)));
                    (p2, p1) = (b1, b2);
                    sp += 3;
                }
                lit_cur.rem -= fast_n * 3;
                continue;
            }
            lit_cur.rem -= 1;
            let tree_id =
                plan.lit_cmap[lit_cmap_base + context::stream_byte_ctx(lut0, lut1, data, sp)];
            let (ln, lb) = header.literals_codes[tree_id as usize].symbol_info(data[sp] as usize);
            bw.write(ln, lb);
            sp += 1;
        }
        *sp_out = ins_end;
    };

    for &cmd in commands {
        emit_iac_and_literals(
            cmd.insert_len,
            cmd.copy_len,
            cmd.is_implicit_d0(),
            &mut stream_pos,
            bw,
        );

        if cmd.emits_dist_sym() {
            let dist_bt = dist_cur.step(&header.distance, bw);
            let d_ctx = copy_code_only(cmd.copy_len).min(3);
            let (d_sym, d_eb_n, d_eb_b) = if cmd.dist_mode < 16 {
                (cmd.dist_mode as u16, 0u8, 0u64)
            } else {
                let (s, n, b) = encode_raw_distance_params(cmd.dist, plan.npostfix, plan.ndirect);
                (s, n, b as u64)
            };
            let tree_id = plan.dist_cmap[dist_bt * 4 + d_ctx] as usize;
            let (dn, db) = header.distance_codes[tree_id].symbol_info(d_sym as usize);
            bw.write(dn, db);
            if d_eb_n > 0 {
                bw.write(d_eb_n as usize, d_eb_b);
            }
        }
        stream_pos += cmd.out_len as usize;
    }

    if trailing_literals > 0 {
        emit_iac_and_literals(trailing_literals, 2, false, &mut stream_pos, bw);
    }

    debug_assert_eq!(stream_pos, mb_end);
    for cur in [&lit_cur, &iac_cur, &dist_cur] {
        if cur.split.num_types >= 2 {
            debug_assert_eq!(cur.rem, 0);
            debug_assert_eq!(cur.block_idx + 1, cur.split.lengths.len());
        }
    }
    debug_assert_eq!(
        bw.total_bits() - start_bits,
        plan.exact_compressed_bits,
        "bit-exact cost model mismatch in write_compressed_metablock"
    );
}
