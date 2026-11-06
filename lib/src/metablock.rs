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

use bounded_utils::BoundedUsize;
use lsb_bitwriter::{BitStreamCursor, BitWriter};

use crate::block_split::{BlockSplit, MetablockPlan};
use crate::constants::{try_avx2, Scalar, MAX_IAC, MAX_LIT};
use crate::context;
use crate::huffman::{ContextMap, HuffmanCode, HuffmanCodeEntry};
use crate::map_to_sym::{
    bound64_from_u63, copy_code_only, emit_batch1_slice, emit_literals, encode_iac_bounded,
    encode_raw_distance_params, pack_entries_quad_bounded, pack_huff_extra_bounded,
};

/// `dist_mode` values `0..=15` represent explicit RFC 7932 short distance codes `0..=15`.
pub(crate) const DIST_MODE_IMPLICIT_D0: u8 = 16;
pub(crate) const DIST_MODE_RAW: u8 = 17;
pub(crate) const DIST_MODE_STATIC_DICT: u8 = 18;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct OptimalCommand {
    pub(crate) insert_len: u32,
    pub(crate) copy_len: u32,
    pub(crate) out_len: u32,
    pub(crate) dist: u32,
    pub(crate) dist_mode: u8,
}

impl OptimalCommand {
    #[inline(always)]
    pub(crate) const fn new(
        insert_len: u32,
        copy_len: u32,
        out_len: u32,
        dist: u32,
        dist_mode: u8,
    ) -> Self {
        Self {
            insert_len,
            copy_len,
            out_len,
            dist,
            dist_mode,
        }
    }

    #[inline(always)]
    pub(crate) const fn out_len(self) -> u32 {
        self.out_len
    }

    #[inline(always)]
    pub(crate) fn is_implicit_d0(self) -> bool {
        self.dist_mode == DIST_MODE_IMPLICIT_D0
    }

    #[inline(always)]
    pub(crate) fn emits_dist_sym(self) -> bool {
        self.dist_mode != DIST_MODE_IMPLICIT_D0
    }
}

pub(crate) fn write_window_bits(wbits: usize, large_window: bool, bw: &mut BitWriter) {
    let (nbits, bits) = if large_window {
        assert!((10..=30).contains(&wbits), "invalid wbits: {wbits}");
        (14, (((wbits as u64) & 0x3F) << 8) | 0x11)
    } else {
        match wbits {
            16 => (1, 0),
            18..=24 => (4, (((wbits - 17) as u64) << 1) | 1),
            17 => (7, 1),
            10..=15 => (7, (((wbits - 8) as u64) << 4) | 1),
            _ => panic!("invalid wbits: {wbits}"),
        }
    };
    bw.write(nbits, bits);
}

/// Writes an empty final metablock (`ISLAST = 1, ISLASTEMPTY = 1`) and zero-pads to a byte boundary.
#[inline]
pub(crate) fn write_empty_last_metablock(bw: &mut BitWriter) {
    bw.write(2, 0b11);
    bw.zero_pad_to_byte();
}

/// Writes an RFC 7932 §9.2 empty metadata metablock (`ISLAST = 0, MNIBBLES = 11, reserved = 0,
/// MSKIPBYTES = 00`) and zero-pads to a byte boundary for `SyncFlush`.
#[inline]
pub(crate) fn write_empty_metadata_metablock(bw: &mut BitWriter) {
    bw.write(6, 0b000110);
    bw.zero_pad_to_byte();
}

/// Writes an `ISUNCOMPRESSED = 1` metablock (`islast = false`) directly to `bw` at the current
/// stream bit alignment.
pub(crate) fn write_uncompressed_metablock(raw_slice: &[u8], bw: &mut BitWriter) {
    write_mlen_header(raw_slice.len(), false, true, bw);
    bw.zero_pad_to_byte();
    bw.write_bytes_aligned(raw_slice);
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockTypeInfo<'a> {
    pub(crate) num: u64,
    pub(crate) block_type_code: HuffmanCode<'a>,
    pub(crate) block_count_code: HuffmanCode<'a>,
    pub(crate) first_num: usize,
}

impl<'a> Default for BlockTypeInfo<'a> {
    fn default() -> Self {
        BlockTypeInfo {
            num: 1,
            block_type_code: HuffmanCode::default(),
            block_count_code: HuffmanCode::default(),
            first_num: 0,
        }
    }
}

#[inline]
fn encode_header_count(count: u64) -> (usize, u64) {
    assert!((1..=256).contains(&count), "invalid count: {count}");
    if count == 1 {
        return (1, 0);
    }
    let nbits = (count - 1).ilog2() as usize;
    let extra = (count - 1) - (1u64 << nbits);
    (4 + nbits, (((nbits as u64) << 1) | 1) | (extra << 4))
}

#[inline]
pub(crate) fn header_count_bit_cost(count: u64) -> usize {
    encode_header_count(count).0
}

#[inline]
pub(crate) fn write_header_count(count: u64, bw: &mut BitWriter) {
    let (nbits, bits) = encode_header_count(count);
    bw.write(nbits, bits);
}

/// Computes the RFC 7932 block-switch type symbol (`0` = `second_prev`, `1` = `(prev + 1) % num_types`,
/// else `new_type + 2`) and advances `(prev, second_prev)`.
#[inline(always)]
pub(crate) fn step_block_type_mtf(
    new_type: usize,
    num_types: usize,
    prev: &mut usize,
    second_prev: &mut usize,
) -> usize {
    let sym = if new_type == *second_prev {
        0
    } else if new_type == (*prev + 1) % num_types {
        1
    } else {
        new_type + 2
    };
    *second_prev = *prev;
    *prev = new_type;
    sym
}

pub(crate) const NUM_BLOCK_LEN_SYMBOLS: usize = 26;

const SYM_EB_COUNT: [u8; NUM_BLOCK_LEN_SYMBOLS] = [
    2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 24,
];
const SYM_OFFSET: [u64; NUM_BLOCK_LEN_SYMBOLS] = [
    1, 5, 9, 13, 17, 25, 33, 41, 49, 65, 81, 97, 113, 145, 177, 209, 241, 305, 369, 497, 753, 1265,
    2289, 4337, 8433, 16625,
];

/// Maps an RFC 7932 block count (`1..=16_793_840`) to `(symbol, extra_bits_len, extra_bits)`.
#[inline(always)]
pub(crate) fn block_count_to_sym_and_bits(count: usize) -> (usize, usize, u64) {
    debug_assert!((1..=16_793_840).contains(&count));
    let v = count - 1;
    let sym = if v < 240 {
        let g = ((v >> 4) + 1).ilog2() as usize;
        4 * g + ((v - (((1usize << g) - 1) << 4)) >> (g + 2))
    } else if v < 16_624 {
        11 + ((v - 240) | 32).ilog2() as usize
    } else {
        25
    };
    (
        sym,
        SYM_EB_COUNT[sym] as usize,
        count as u64 - SYM_OFFSET[sym],
    )
}

impl<'a> BlockTypeInfo<'a> {
    pub(crate) fn encode_block_count(&self, count: usize) -> (usize, u64) {
        let (sym, eb_count, eb) = block_count_to_sym_and_bits(count);
        let (hufn, hufb) = self.block_count_code.symbol_info(sym);
        (hufn + eb_count, (eb << hufn) | hufb)
    }

    pub(crate) fn write(&self, bw: &mut BitWriter) {
        write_header_count(self.num, bw);
        if self.num >= 2 {
            self.block_type_code.write(bw);
            self.block_count_code.write(bw);
            let (bcn, bcb) = self.encode_block_count(self.first_num);
            bw.write(bcn, bcb);
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ContextMode {
    LSB6 = 0,
    MSB6 = 1,
    UTF8 = 2,
    #[default]
    Signed = 3,
}

#[derive(Default, Debug)]
pub(crate) struct Header<'a> {
    pub(crate) islast: bool,
    pub(crate) len: usize,
    pub(crate) literals: BlockTypeInfo<'a>,
    pub(crate) insert_and_copy: BlockTypeInfo<'a>,
    pub(crate) distance: BlockTypeInfo<'a>,
    pub(crate) npostfix: u8,
    pub(crate) ndirect: u8,
    pub(crate) context_mode: &'a [ContextMode],
    pub(crate) literals_codes: &'a [HuffmanCode<'a>],
    pub(crate) literals_cmap: ContextMap<'a>,
    pub(crate) insert_and_copy_codes: &'a [HuffmanCode<'a>],
    pub(crate) distance_codes: &'a [HuffmanCode<'a>],
    pub(crate) distance_cmap: ContextMap<'a>,
}

#[inline(always)]
fn write_mlen_header(len: usize, is_last: bool, is_uncompressed: bool, bw: &mut BitWriter) {
    assert!((1..=(1 << 24)).contains(&len), "invalid len {len}");
    debug_assert!(!(is_last && is_uncompressed));
    let mnibbles = usize::from(len > (1 << 16)) + usize::from(len > (1 << 20));
    let nbits = 20 + 4 * mnibbles;
    let mlen_bits = (mnibbles as u64) | (((len - 1) as u64) << 2);
    let bits = if is_last {
        1 | (mlen_bits << 2)
    } else {
        (mlen_bits << 1) | (u64::from(is_uncompressed) << (nbits - 1))
    };
    bw.write(nbits, bits);
}

/// Writes the RFC 7932 compressed metablock prefix (`ISLAST`, `MNIBBLES`, `MLEN - 1`, and for
/// `!is_last`, `ISUNCOMPRESSED = 0`).
#[inline(always)]
pub(crate) fn write_compressed_metablock_prefix(len: usize, is_last: bool, bw: &mut BitWriter) {
    write_mlen_header(len, is_last, false, bw);
}

impl<'a> Header<'a> {
    pub(crate) fn write(&self, bw: &mut BitWriter) {
        write_mlen_header(self.len, self.islast, false, bw);
        assert!(self.npostfix <= 3);
        assert!(self.ndirect & ((1 << self.npostfix) - 1) == 0);
        let ndir_code = (self.ndirect >> self.npostfix) as u64;
        assert!(ndir_code < 16);
        assert_eq!(self.context_mode.len(), self.literals.num as usize);
        if self.literals.num == 1
            && self.insert_and_copy.num == 1
            && self.distance.num == 1
            && self.literals_codes.len() == 1
            && self.distance_codes.len() == 1
        {
            let mid = ((self.npostfix as u64) << 3)
                | (ndir_code << 5)
                | ((self.context_mode[0] as u64) << 9);
            bw.write(13, mid);
        } else {
            self.literals.write(bw);
            self.insert_and_copy.write(bw);
            self.distance.write(bw);
            bw.write(6, (self.npostfix as u64) | (ndir_code << 2));
            for &cm in self.context_mode {
                bw.write(2, cm as u64);
            }
            for (codes, cmap) in [
                (self.literals_codes, &self.literals_cmap),
                (self.distance_codes, &self.distance_cmap),
            ] {
                write_header_count(codes.len() as u64, bw);
                if codes.len() >= 2 {
                    cmap.check(codes.len());
                    cmap.write(bw, codes.len());
                }
            }
        }
        debug_assert_eq!(
            self.insert_and_copy_codes.len() as u64,
            self.insert_and_copy.num
        );
        for codes in [
            self.literals_codes,
            self.insert_and_copy_codes,
            self.distance_codes,
        ] {
            for code in codes {
                code.write(bw);
            }
        }
    }
}

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
    fn step(&mut self, bti: &BlockTypeInfo<'_>, cur: &mut BitStreamCursor<'_, '_>) -> usize {
        self.advance_if_needed(bti, cur);
        self.rem -= 1;
        self.cur_type
    }

    #[inline(always)]
    fn advance_if_needed(
        &mut self,
        bti: &BlockTypeInfo<'_>,
        cur: &mut BitStreamCursor<'_, '_>,
    ) -> bool {
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
        let (cn, cb) = bti.encode_block_count(self.rem);
        cur.put_bounded(BoundedUsize::<64>::new_clamped(tn + cn), tb | (cb << tn));
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

    let split_cap = |s: &BlockSplit| usize::from(s.num_types >= 2) * (s.num_types + 28);
    let arena_cap = split_cap(&plan.lit_split)
        + split_cap(&plan.iac_split)
        + split_cap(&plan.dist_split)
        + plan.lit_tree_counts.len() * MAX_LIT
        + plan.iac_tree_counts.len() * MAX_IAC
        + plan.dist_tree_counts.iter().map(|c| c.len()).sum::<usize>();

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

    let [lit_bti, iac_bti, dist_bti] =
        [&plan.lit_split, &plan.iac_split, &plan.dist_split].map(&mut build_bti);
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

    let lit_luts: Vec<&[HuffmanCodeEntry; MAX_LIT]> = lit_codes
        .iter()
        .map(|hc| hc.code.try_into().unwrap())
        .collect();
    let iac_luts: Vec<&[HuffmanCodeEntry; MAX_IAC]> = iac_codes
        .iter()
        .map(|hc| hc.code.try_into().unwrap())
        .collect();

    let resolve_lit_bt = |bt: usize| {
        let cmap_row: &[u8; 64] = plan.lit_cmap[bt * 64..bt * 64 + 64].try_into().unwrap();
        let t0 = cmap_row[0];
        let single = cmap_row
            .iter()
            .all(|&t| t == t0)
            .then(|| lit_luts[t0 as usize]);
        let ctx_trees: [&[HuffmanCodeEntry; MAX_LIT]; 64] =
            std::array::from_fn(|c| lit_luts[cmap_row[c] as usize]);
        let (l0, l1) = context::context_luts(plan.context_modes[bt]);
        (l0, l1, single, ctx_trees)
    };

    let mut lit_cur = BlockSwitchCursor::new(&plan.lit_split);
    let mut iac_cur = BlockSwitchCursor::new(&plan.iac_split);
    let mut dist_cur = BlockSwitchCursor::new(&plan.dist_split);

    let (mut lut0, mut lut1, mut single_lit_tree, mut ctx_trees) =
        resolve_lit_bt(lit_cur.cur_type() as usize);
    let avx2 = try_avx2(true);
    let mut stream_pos = mb_start;
    let mut cur = bw.cursor();

    let mut emit_iac_and_literals =
        |insert_len: u32,
         copy_len: u32,
         implicit_d0: bool,
         sp_out: &mut usize,
         cur: &mut BitStreamCursor<'_, '_>| {
            let iac_bt = iac_cur.step(&header.insert_and_copy, cur);
            let (iac_sym, iac_eb_len, iac_eb_b) =
                encode_iac_bounded(insert_len, copy_len, implicit_d0);
            let h_iac = iac_luts[iac_bt][iac_sym.get() as usize];
            let (iac_nb, iac_b) = pack_huff_extra_bounded(h_iac, iac_eb_len.get(), iac_eb_b);
            cur.put_bounded(iac_nb, iac_b);

            let mut sp = *sp_out;
            let ins_end = sp + (insert_len as usize);
            while sp < ins_end {
                if lit_cur.advance_if_needed(&header.literals, cur) {
                    (lut0, lut1, single_lit_tree, ctx_trees) =
                        resolve_lit_bt(lit_cur.cur_type() as usize);
                }
                let avail = (ins_end - sp).min(lit_cur.rem);
                if let Some(lut) = single_lit_tree {
                    match avx2 {
                        #[cfg(target_arch = "x86_64")]
                        Some(tok) => emit_literals(tok, &data[sp..sp + avail], lut, cur),
                        _ => emit_literals(Scalar, &data[sp..sp + avail], lut, cur),
                    }
                    (sp, lit_cur.rem) = (sp + avail, lit_cur.rem - avail);
                    continue;
                }
                if sp >= 2 && avail >= 4 {
                    let fast_n = avail / 4;
                    let (mut p2, mut p1) = (data[sp - 2], data[sp - 1]);
                    let (quads, _) = data[sp..sp + fast_n * 4].as_chunks::<4>();
                    emit_batch1_slice(cur, quads, |&[b0, b1, b2, b3]| {
                        let e0 = ctx_trees[((lut0[p1 as usize] | lut1[p2 as usize]) & 63) as usize]
                            [b0 as usize];
                        let e1 = ctx_trees[((lut0[b0 as usize] | lut1[p1 as usize]) & 63) as usize]
                            [b1 as usize];
                        let e2 = ctx_trees[((lut0[b1 as usize] | lut1[b0 as usize]) & 63) as usize]
                            [b2 as usize];
                        let e3 = ctx_trees[((lut0[b2 as usize] | lut1[b1 as usize]) & 63) as usize]
                            [b3 as usize];
                        (p2, p1) = (b2, b3);
                        pack_entries_quad_bounded([e0, e1, e2, e3])
                    });
                    sp += fast_n * 4;
                    lit_cur.rem -= fast_n * 4;
                    continue;
                }
                lit_cur.rem -= 1;
                let ctx = context::stream_byte_ctx(lut0, lut1, data, sp) & 63;
                let e = ctx_trees[ctx][data[sp] as usize];
                cur.put_bounded(bound64_from_u63(e.len as usize), e.bits as u64);
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
            &mut cur,
        );

        if cmd.emits_dist_sym() {
            let dist_bt = dist_cur.step(&header.distance, &mut cur);
            let d_ctx = copy_code_only(cmd.copy_len).min(3);
            let (d_sym, d_eb_n, d_eb_b) = if cmd.dist_mode < 16 {
                (cmd.dist_mode as u16, 0u8, 0u64)
            } else {
                let (s, n, b) = encode_raw_distance_params(cmd.dist, plan.npostfix, plan.ndirect);
                (s, n, b as u64)
            };
            let tree_id = plan.dist_cmap[dist_bt * 4 + d_ctx] as usize;
            let (dn, db) = header.distance_codes[tree_id].symbol_info(d_sym as usize);
            cur.put_bounded(bound64_from_u63(dn + d_eb_n as usize), db | (d_eb_b << dn));
        }
        stream_pos += cmd.out_len as usize;
    }

    if trailing_literals > 0 {
        emit_iac_and_literals(trailing_literals, 4, false, &mut stream_pos, &mut cur);
    }
    drop(cur);

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

#[cfg(test)]
pub(crate) mod test_helpers {
    use std::io::Read;
    use std::sync::OnceLock;

    pub(crate) fn decompress_brotli(compressed: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut dec = brotli_decompressor::Decompressor::new(compressed, 4096);
        dec.read_to_end(&mut out).unwrap();
        out
    }

    pub(crate) fn decompress_brotli_with_dict(compressed: &[u8], raw_dict: &[u8]) -> Vec<u8> {
        let mut r = compressed;
        let mut out = Vec::new();
        let mut in_buf = [0u8; 4096];
        let mut out_buf = [0u8; 4096];
        brotli_decompressor::BrotliDecompressCustomDict(
            &mut r,
            &mut out,
            &mut in_buf,
            &mut out_buf,
            raw_dict.to_vec(),
        )
        .unwrap();
        out
    }

    /// Returns whether the linked `brotli-decompressor` crate implements RFC 9841 Compound
    /// Dictionary static-dictionary offset subtraction once `stream_pos >= wsize - dict.len()`
    /// (`v6+` does; legacy `v5` omits `- custom_dict_size` once `pos >= wsize - dict.len()`).
    pub(crate) fn custom_dict_supports_rfc9841_past_window() -> bool {
        static SUPPORTS: OnceLock<bool> = OnceLock::new();
        *SUPPORTS.get_or_init(|| {
            let dict_raw = b"0123456789abcdef0123456789abcdef";
            let mut probe = Vec::with_capacity(1400);
            probe.extend_from_slice(dict_raw);
            probe.resize(1040, b'x');
            probe.extend_from_slice(
                b"Information about international government policy and university research! \
                  The universal application of standard dictionary transformations in Brotli.",
            );
            let dict = crate::PreparedDictionary::with_lgwin(dict_raw, 11, 10);
            let mut enc = crate::Encoder::with_lgwin(11, 10);
            let comp = enc.compress_with_dict(&probe, &dict, None).unwrap();
            assert!(comp.len() < 110, "probe must emit static dictionary words");
            let mut r = comp;
            let mut out = Vec::new();
            let (mut in_buf, mut out_buf) = ([0u8; 512], [0u8; 512]);
            brotli_decompressor::BrotliDecompressCustomDict(
                &mut r,
                &mut out,
                &mut in_buf,
                &mut out_buf,
                dict_raw.to_vec(),
            )
            .is_ok_and(|()| out == probe)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman::HuffmanCodeEntry;

    #[test]
    fn test_block_count_encoding_boundaries() {
        let mut counts = [1u32; NUM_BLOCK_LEN_SYMBOLS];
        counts[0] = 10;
        let mut buf = [HuffmanCodeEntry::default(); NUM_BLOCK_LEN_SYMBOLS];
        let (_, code) = HuffmanCode::from_counts(&counts, 15, &mut buf);
        let bti = BlockTypeInfo {
            num: 2,
            block_type_code: HuffmanCode::default(),
            block_count_code: code,
            first_num: 1,
        };

        let dense_hi = if cfg!(miri) { 64 } else { 20_000 };
        for count in
            (1..=dense_hi).chain([256, 1040, 16_624, 16_625, 100_000, 1_000_000, 16_793_840])
        {
            let (sym, eb_len, eb) = block_count_to_sym_and_bits(count);
            let ref_sym = SYM_OFFSET.partition_point(|&off| off <= count as u64) - 1;
            assert_eq!(sym, ref_sym, "mismatch at count={count}");
            assert_eq!(eb_len, SYM_EB_COUNT[sym] as usize);
            assert!(eb < (1u64 << eb_len));
            assert_eq!(SYM_OFFSET[sym] + eb, count as u64);

            let (tot_bits, packed) = bti.encode_block_count(count);
            let (hufn, hufb) = bti.block_count_code.symbol_info(sym);
            assert_eq!(tot_bits, hufn + eb_len);
            assert_eq!(packed & ((1u64 << hufn) - 1), hufb);
            assert_eq!(packed >> hufn, eb);
        }
    }

    #[test]
    fn test_header_count_bit_cost_matches_writer() {
        for c in 1..=if cfg!(miri) { 32u64 } else { 256u64 } {
            let mut mem = [0u8; 32];
            let mut bw = BitWriter::new(&mut mem);
            write_header_count(c, &mut bw);
            assert_eq!(bw.total_bits(), header_count_bit_cost(c) as u64);
        }
    }

    #[test]
    fn test_window_bits_10_to_30() {
        let wbs: Vec<usize> = if cfg!(miri) {
            vec![10, 16, 17, 22, 24, 25, 30]
        } else {
            (10..=30).collect()
        };
        for wbits in wbs {
            let mut mem = [0u8; 16];
            let mut bw = BitWriter::new(&mut mem);
            write_window_bits(wbits, wbits > 24, &mut bw);
            let hdr_bits = bw.total_bits();
            write_empty_last_metablock(&mut bw);
            let bytes = bw.finalize();
            if wbits <= 24 {
                assert!(matches!(hdr_bits, 1 | 4 | 7));
                assert!(test_helpers::decompress_brotli(bytes).is_empty());
            } else {
                assert_eq!(hdr_bits, 14);
                let raw14 = u16::from_le_bytes([bytes[0], bytes[1]]) & 0x3FFF;
                assert_eq!(raw14, ((wbits as u16) << 8) | 0x11);
            }
        }
    }
}
