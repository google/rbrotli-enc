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

//! 3-stage MDL block splitter and joint context clustering for `Q8..=Q12`:
//! 1. Micro-block JSD min-heap neighbor merge (`<= 384` macro-segments).
//! 2. Farthest-first JSD prototype seeding + Lloyd EM Huffman refinement (`<= K` types).
//! 3. $O(N_w \cdot K)$ Viterbi DP (`min1`/`min2` transition trick) + symbol-exact boundary polish,
//!    guarded against a bit-exact 1-block baseline (`HuffmanCode::bit_cost` + `ContextMap::bit_cost`).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::constants::{MAX_IAC, MAX_LIT};
use crate::context;
use crate::huffman::{ContextMap, HuffmanCode, HuffmanCodeEntry, MAX_HUFFMAN_ALPHABET};
use crate::map_to_sym::{
    copy_code_only, dist_alphabet_size, encode_raw_distance_params, max_dist_symbol,
};
use crate::metablock::{
    block_count_to_sym_and_bits, header_count_bit_cost, step_block_type_mtf, ContextMode,
    OptimalCommand, NUM_BLOCK_LEN_SYMBOLS,
};

#[derive(Clone, Debug)]
pub(crate) struct BlockSplit {
    pub(crate) num_types: usize,
    pub(crate) types: Vec<u8>,
    pub(crate) lengths: Vec<usize>,
    pub(crate) type_counts: Vec<u32>,
    pub(crate) len_counts: [u32; NUM_BLOCK_LEN_SYMBOLS],
}

impl BlockSplit {
    pub(crate) fn single_block(total_len: usize) -> Self {
        let (types, lengths) = if total_len > 0 {
            (vec![0], vec![total_len])
        } else {
            (Vec::new(), Vec::new())
        };
        Self {
            num_types: 1,
            types,
            lengths,
            type_counts: Vec::new(),
            len_counts: [0; NUM_BLOCK_LEN_SYMBOLS],
        }
    }
}

/// Complete planned header & block-switch layout for one Brotli metablock.
pub(crate) struct MetablockPlan {
    pub(crate) npostfix: u8,
    pub(crate) ndirect: u8,
    pub(crate) lit_split: BlockSplit,
    pub(crate) iac_split: BlockSplit,
    pub(crate) dist_split: BlockSplit,
    pub(crate) context_modes: Vec<ContextMode>,
    pub(crate) lit_cmap: Vec<u8>,
    pub(crate) lit_tree_counts: Vec<[u32; MAX_LIT]>,
    pub(crate) iac_tree_counts: Vec<[u32; MAX_IAC]>,
    pub(crate) dist_cmap: Vec<u8>,
    pub(crate) dist_alpha: usize,
    pub(crate) dist_tree_counts: Vec<Vec<u32>>,
    pub(crate) exact_compressed_bits: u64,
}

impl MetablockPlan {
    pub(crate) fn uncompressed_sentinel(chunk_len: usize) -> Self {
        Self {
            npostfix: 0,
            ndirect: 0,
            lit_split: BlockSplit::single_block(chunk_len),
            iac_split: BlockSplit::single_block(0),
            dist_split: BlockSplit::single_block(0),
            context_modes: Vec::new(),
            lit_cmap: Vec::new(),
            lit_tree_counts: Vec::new(),
            iac_tree_counts: Vec::new(),
            dist_cmap: Vec::new(),
            dist_alpha: 64,
            dist_tree_counts: Vec::new(),
            exact_compressed_bits: (chunk_len as u64) * 8 + 1024,
        }
    }
}

/// Computes `type_counts`, `len_counts`, and the **exact** total bit cost of `BlockTypeInfo`.
fn populate_and_price_block_switches(split: &mut BlockSplit, hbuf: &mut [HuffmanCodeEntry]) -> u64 {
    if split.num_types <= 1 {
        split.type_counts.clear();
        split.len_counts = [0; NUM_BLOCK_LEN_SYMBOLS];
        return 1;
    }
    debug_assert!(split.types.len() >= split.num_types && split.types[0] == 0);
    split.type_counts.clear();
    split.type_counts.resize(split.num_types + 2, 0);
    split.len_counts = [0; NUM_BLOCK_LEN_SYMBOLS];

    let (first_sym, first_eb, _) = block_count_to_sym_and_bits(split.lengths[0]);
    split.len_counts[first_sym] += 1;
    let mut extra_bits = first_eb as u64;
    let (mut prev, mut second_prev) = (0usize, 1usize);
    for idx in 1..split.types.len() {
        let t = split.types[idx] as usize;
        debug_assert_ne!(t, prev);
        split.type_counts[step_block_type_mtf(t, split.num_types, &mut prev, &mut second_prev)] +=
            1;
        let (lsym, leb, _) = block_count_to_sym_and_bits(split.lengths[idx]);
        split.len_counts[lsym] += 1;
        extra_bits += leb as u64;
    }

    let type_bits = HuffmanCode::from_counts(&split.type_counts, 15, hbuf)
        .1
        .bit_cost(&split.type_counts);
    let count_bits = HuffmanCode::from_counts(&split.len_counts, 15, hbuf)
        .1
        .bit_cost(&split.len_counts);
    header_count_bit_cost(split.num_types as u64) as u64 + type_bits + count_bits + extra_bits
}

/// Runs $O(N_w \cdot K)$ 1D Viterbi DP (`min1`/`min2` transition trick) followed by
/// symbol-exact boundary polish (`±viterbi_win`).
fn viterbi_and_polish(
    total_len: usize,
    k_num: usize,
    viterbi_win: usize,
    hbuf: &mut [HuffmanCodeEntry],
    sym_cost_q4: impl Fn(usize, usize) -> u16,
) -> (BlockSplit, u64) {
    const SWITCH_PEN_Q4: u64 = 11 * 16;
    let num_win = total_len.div_ceil(viterbi_win);
    if k_num <= 1 || num_win <= 1 {
        return (BlockSplit::single_block(total_len), 1);
    }

    let mut emit = vec![0u64; k_num];
    let mut backptr = vec![0u8; num_win * k_num];
    let (mut dp_prev, mut dp_cur) = (vec![0u64; k_num], vec![0u64; k_num]);

    for w in 0..num_win {
        let (ws, we) = (w * viterbi_win, ((w + 1) * viterbi_win).min(total_len));
        for (k, slot) in emit.iter_mut().enumerate() {
            *slot = (ws..we).map(|i| sym_cost_q4(k, i) as u64).sum();
        }

        if w == 0 {
            dp_cur.copy_from_slice(&emit);
        } else {
            let (mut min1, mut min2, mut k_min1, mut k_min2) = (u64::MAX, u64::MAX, 0usize, 0usize);
            for (k, &v) in dp_prev.iter().enumerate() {
                if v < min1 {
                    (min2, k_min2, min1, k_min1) = (min1, k_min1, v, k);
                } else if v < min2 {
                    (min2, k_min2) = (v, k);
                }
            }
            for k in 0..k_num {
                let stay = dp_prev[k];
                let (best_other, src_k) = if k != k_min1 {
                    (min1, k_min1)
                } else {
                    (min2, k_min2)
                };
                let sw = best_other.saturating_add(SWITCH_PEN_Q4);
                let (base, prev_k) = if stay <= sw { (stay, k) } else { (sw, src_k) };
                dp_cur[k] = base + emit[k];
                backptr[w * k_num + k] = prev_k as u8;
            }
        }
        std::mem::swap(&mut dp_prev, &mut dp_cur);
    }

    let mut win_labels = vec![0u8; num_win];
    let mut cur_k = (0..k_num).min_by_key(|&k| dp_prev[k]).unwrap_or(0);
    for w in (0..num_win).rev() {
        win_labels[w] = cur_k as u8;
        cur_k = backptr[w * k_num + cur_k] as usize;
    }

    let mut runs: Vec<(u8, usize, usize)> = Vec::new();
    for (w, &lbl) in win_labels.iter().enumerate() {
        let (ws, we) = (w * viterbi_win, ((w + 1) * viterbi_win).min(total_len));
        if let Some(last) = runs.last_mut().filter(|last| last.0 == lbl) {
            last.2 = we;
        } else {
            runs.push((lbl, ws, we));
        }
    }
    if runs.len() <= 1 {
        return (BlockSplit::single_block(total_len), 1);
    }

    for r_idx in 0..(runs.len() - 1) {
        let (k_l, k_r) = (runs[r_idx].0 as usize, runs[r_idx + 1].0 as usize);
        let b0 = runs[r_idx].2;
        let min_b = (runs[r_idx].1 + 1).max(b0.saturating_sub(viterbi_win));
        let max_b = (runs[r_idx + 1].2 - 1).min(b0 + viterbi_win);
        if min_b > max_b {
            continue;
        }
        let (mut cur_delta, mut best_delta, mut best_b) = (0i64, 0i64, min_b);
        for p in min_b..max_b {
            cur_delta += (sym_cost_q4(k_l, p) as i64) - (sym_cost_q4(k_r, p) as i64);
            if cur_delta < best_delta {
                (best_delta, best_b) = (cur_delta, p + 1);
            }
        }
        runs[r_idx].2 = best_b;
        runs[r_idx + 1].1 = best_b;
    }

    let (mut types, lengths): (Vec<u8>, Vec<usize>) =
        runs.into_iter().map(|(l, s, e)| (l, e - s)).unzip();
    let actual_types = context::canonicalize_cmap_ids(&mut types, None);
    if actual_types <= 1 {
        return (BlockSplit::single_block(total_len), 1);
    }
    let mut out_split = BlockSplit {
        num_types: actual_types,
        types,
        lengths,
        type_counts: Vec::new(),
        len_counts: [0; NUM_BLOCK_LEN_SYMBOLS],
    };
    let sw = populate_and_price_block_switches(&mut out_split, hbuf);
    (out_split, sw)
}

fn fill_symbol_cost_q4(
    counts: &[u32],
    floor_q4: u16,
    out_q4: &mut [u16],
    hbuf: &mut [HuffmanCodeEntry],
) {
    crate::huffman::fill_smoothed_symbol_costs::<4>(
        counts,
        (floor_q4, 4, 8),
        out_q4,
        hbuf,
        |c, total| {
            let p = (c as f32) * (1.0f32 / (total as f32));
            (((127i32 << 4) - ((p.to_bits() >> 19) as i32)).max(4)) as u32
        },
    );
}

fn build_cost_tables_q4(
    flat: &[u32],
    alpha: usize,
    hbuf: &mut [HuffmanCodeEntry],
) -> Vec<Vec<u16>> {
    flat.chunks_exact(alpha)
        .map(|c| {
            let mut tbl = vec![14 * 16u16; alpha];
            fill_symbol_cost_q4(c, 14 * 16, &mut tbl, hbuf);
            tbl
        })
        .collect()
}

#[inline]
fn nearest_cost_table(nz: &[(u16, u32)], tbls: &[Vec<u16>]) -> usize {
    tbls.iter()
        .enumerate()
        .min_by_key(|(_, t)| {
            nz.iter()
                .map(|&(s, cnt)| (cnt as u64) * (t[s as usize] as u64))
                .sum::<u64>()
        })
        .map_or(0, |(k, _)| k)
}

fn price_flat_trees(flat: &[u32], alpha: usize, hbuf: &mut [HuffmanCodeEntry]) -> u64 {
    flat.chunks_exact(alpha)
        .map(|s| HuffmanCode::from_counts(s, 15, hbuf).1.bit_cost(s))
        .sum()
}

fn price_cmap_and_flat(
    cmap: &[u8],
    flat: &[u32],
    alpha: usize,
    hbuf: &mut [HuffmanCodeEntry],
) -> u64 {
    let nt = flat.len() / alpha;
    let tree_bits = price_flat_trees(flat, alpha, hbuf);
    if nt == 1 {
        1 + tree_bits
    } else {
        header_count_bit_cost(nt as u64) as u64
            + ContextMap::new(cmap).bit_cost(nt) as u64
            + tree_bits
    }
}

fn count_split_syms_flat(split: &BlockSplit, syms: &[u16], alpha: usize, out_flat: &mut [u32]) {
    let mut idx = 0usize;
    for (&bt, &blen) in split.types.iter().zip(split.lengths.iter()) {
        let dst = &mut out_flat[(bt as usize) * alpha..(bt as usize + 1) * alpha];
        for &s in &syms[idx..idx + blen] {
            dst[s as usize] += 1;
        }
        idx += blen;
    }
    debug_assert_eq!(idx, syms.len());
}

use context::add_hist_slice as add_hist_row;

/// Propagates empty rows in `cmap`, canonicalizes IDs in order of first appearance, and rebuilds
/// flat-stride cluster histograms `(cl_h, cl_t)`.
fn rebuild_clusters_from_cmap(
    hists: &[u32],
    totals: &[u32],
    alpha: usize,
    cmap: &mut [u8],
) -> (Vec<u32>, Vec<u32>) {
    for i in 0..totals.len() {
        if totals[i] == 0 {
            cmap[i] = if i > 0 { cmap[i - 1] } else { 0 };
        }
    }
    let num_c = context::canonicalize_cmap_ids(cmap, totals.iter().position(|&t| t > 0));
    let (mut cl_h, mut cl_t) = (vec![0u32; num_c * alpha], vec![0u32; num_c]);
    for (i, (&t, &cid)) in totals.iter().zip(cmap.iter()).enumerate() {
        if t > 0 {
            let c = cid as usize;
            cl_t[c] += t;
            add_hist_row(
                &mut cl_h[c * alpha..(c + 1) * alpha],
                &hists[i * alpha..(i + 1) * alpha],
            );
        }
    }
    (cl_h, cl_t)
}

/// Shared Lloyd / EM refinement over flat-stride histograms using Q4 Huffman code-length tables.
#[allow(clippy::too_many_arguments)]
fn lloyd_refine_flat(
    hists: &[u32],
    totals: &[u32],
    alpha: usize,
    rounds: usize,
    mut cl_h: Vec<u32>,
    mut cl_t: Vec<u32>,
    mut cmap: Vec<u8>,
    hbuf: &mut [HuffmanCodeEntry],
    mut on_step: impl FnMut(&[u32], &[u8], &mut [HuffmanCodeEntry]),
) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let mut nz_buf = [(0u16, 0u32); MAX_HUFFMAN_ALPHABET];
    for _ in 0..rounds {
        if cl_t.len() < 2 {
            break;
        }
        let tbls = build_cost_tables_q4(&cl_h, alpha, hbuf);
        let mut raw_map = vec![0u8; totals.len()];
        for (hi, &tot) in totals.iter().enumerate() {
            if tot == 0 {
                continue;
            }
            let mut nz_len = 0usize;
            for (b, &cnt) in hists[hi * alpha..(hi + 1) * alpha].iter().enumerate() {
                if cnt != 0 {
                    nz_buf[nz_len] = (b as u16, cnt);
                    nz_len += 1;
                }
            }
            raw_map[hi] = nearest_cost_table(&nz_buf[..nz_len], &tbls) as u8;
        }
        let (next_h, next_t) = rebuild_clusters_from_cmap(hists, totals, alpha, &mut raw_map);
        let unchanged = raw_map == cmap;
        (cl_h, cl_t, cmap) = (next_h, next_t, raw_map);
        if cl_t.len() < 2 {
            break;
        }
        on_step(&cl_h, &cmap, hbuf);
        if unchanged {
            break;
        }
    }
    (cl_h, cl_t, cmap)
}

#[derive(Clone, Copy)]
struct SplitSpec(usize, usize, usize, i32);

fn split_symbol_stream(syms: &[u16], spec: SplitSpec, hbuf: &mut [HuffmanCodeEntry]) -> BlockSplit {
    let SplitSpec(alpha, max_types, micro_size, mdl_gain) = spec;
    let (n, viterbi_win) = (syms.len(), micro_size / 4);
    if n < micro_size * 2 || max_types <= 1 || alpha == 0 {
        return BlockSplit::single_block(n);
    }

    // --- Stage 1: Micro-block partition & JSD min-heap neighbor merge ---
    let num_micro = n.div_ceil(micro_size);
    let mut seg_hists = vec![0u32; num_micro * alpha];
    let mut seg_totals = vec![0u32; num_micro];
    let mut prev_seg: Vec<usize> = (0..num_micro).map(|m| m.wrapping_sub(1)).collect();
    let mut next_seg: Vec<usize> = (0..num_micro)
        .map(|m| if m + 1 < num_micro { m + 1 } else { usize::MAX })
        .collect();
    let mut ver = vec![0u32; num_micro];

    for m in 0..num_micro {
        let (s, e) = (m * micro_size, ((m + 1) * micro_size).min(n));
        let h = &mut seg_hists[m * alpha..(m + 1) * alpha];
        for &sym in &syms[s..e] {
            h[sym as usize] += 1;
        }
        seg_totals[m] = (e - s) as u32;
    }

    let make_edge = |l: usize, r: usize, sh: &[u32], st: &[u32], v: &[u32]| {
        let cost = context::excess_jsd_slice(
            &sh[l * alpha..(l + 1) * alpha],
            st[l],
            &sh[r * alpha..(r + 1) * alpha],
            st[r],
        );
        Reverse((cost, l, r, v[l], v[r]))
    };

    let mut heap = BinaryHeap::with_capacity(num_micro * 2);
    for m in 0..(num_micro - 1) {
        heap.push(make_edge(m, m + 1, &seg_hists, &seg_totals, &ver));
    }

    let mut active_segs = num_micro;
    while let Some(Reverse((cost, l, r, vl, vr))) = heap.pop() {
        if ver[l] != vl || ver[r] != vr {
            continue;
        }
        if cost > 24 && active_segs <= 384 {
            break;
        }
        let (left_part, right_part) = seg_hists.split_at_mut(r * alpha);
        add_hist_row(
            &mut left_part[l * alpha..(l + 1) * alpha],
            &right_part[..alpha],
        );
        seg_totals[l] += seg_totals[r];
        ver[r] = u32::MAX;
        ver[l] += 1;
        active_segs -= 1;

        let nr = next_seg[r];
        next_seg[l] = nr;
        if nr != usize::MAX {
            prev_seg[nr] = l;
        }
        if prev_seg[l] != usize::MAX {
            heap.push(make_edge(prev_seg[l], l, &seg_hists, &seg_totals, &ver));
        }
        if nr != usize::MAX {
            heap.push(make_edge(l, nr, &seg_hists, &seg_totals, &ver));
        }
    }
    if active_segs <= 1 {
        return BlockSplit::single_block(n);
    }

    // --- Stage 2: Farthest-first JSD prototype seeding + Lloyd EM refinement ---
    let mut macro_flat = Vec::with_capacity(active_segs * alpha);
    let mut macro_tots = Vec::with_capacity(active_segs);
    let mut cur = 0usize;
    while cur != usize::MAX {
        macro_flat.extend_from_slice(&seg_hists[cur * alpha..(cur + 1) * alpha]);
        macro_tots.push(seg_totals[cur]);
        cur = next_seg[cur];
    }
    let m_cnt = macro_tots.len();

    let (proto_flat, proto_tots, proto_cm) = context::cluster_flat::<false>(
        &macro_flat,
        &macro_tots,
        alpha,
        max_types.min(m_cnt),
        mdl_gain,
        context::excess_jsd_slice,
        context::excess_jsd_slice,
    );
    if proto_tots.len() <= 1 {
        return BlockSplit::single_block(n);
    }
    let (proto_flat, proto_tots, _) = lloyd_refine_flat(
        &macro_flat,
        &macro_tots,
        alpha,
        2,
        proto_flat,
        proto_tots,
        proto_cm,
        hbuf,
        |_, _, _| {},
    );
    let k_num = proto_tots.len();
    if k_num <= 1 || n.div_ceil(viterbi_win) <= 1 {
        return BlockSplit::single_block(n);
    }
    let cost_tables_q4 = build_cost_tables_q4(&proto_flat, alpha, hbuf);

    // --- Stage 3: Global O(N_w * K) Viterbi DP + symbol-exact boundary polish ---
    viterbi_and_polish(n, k_num, viterbi_win, hbuf, |k, i| {
        cost_tables_q4[k][syms[i] as usize]
    })
    .0
}

#[inline]
fn rank_pos_ctx_modes(data: &[u8], pos: &[u32]) -> [(ContextMode, u64); 4] {
    if pos.len() < 32 {
        context::ALL_CONTEXT_MODES.map(|m| (m, u64::MAX))
    } else {
        context::rank_sampled_ctx_modes::<8>(&context::ALL_CONTEXT_MODES, data, pos.len(), |i| {
            pos[i] as usize
        })
    }
}

#[inline]
fn fill_block_ctx_histograms_flat(
    data: &[u8],
    positions: &[u32],
    mode: ContextMode,
    out_flat: &mut [u32],
    out_totals: &mut [u32],
) {
    out_flat.fill(0);
    out_totals.fill(0);
    let (lut0, lut1) = context::context_luts(mode);
    for &sp_u32 in positions {
        let sp = sp_u32 as usize;
        let ctx = context::stream_byte_ctx(lut0, lut1, data, sp);
        out_flat[ctx * MAX_LIT + (data[sp] as usize)] += 1;
        out_totals[ctx] += 1;
    }
}

fn mdl_slice_gain(a: &[u32], ta: u32, b: &[u32], tb: u32) -> i32 {
    let raw = context::histogram_distance_slice(a, ta, b, tb);
    if raw <= 0 {
        return 0;
    }
    let (mut ka, mut kb) = (0u32, 0u32);
    for (&ca, &cb) in a.iter().zip(b) {
        ka += u32::from(ca != 0);
        kb += u32::from(cb != 0);
    }
    (raw - context::jsd_null_bias(a, b) - (ka.min(kb) as i32) * 2).max(0)
}

fn cluster_flat_mdl(
    hists: &[u32],
    totals: &[u32],
    max_clusters: usize,
    min_gain: i32,
) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    context::cluster_flat::<true>(
        hists,
        totals,
        MAX_LIT,
        max_clusters,
        min_gain,
        mdl_slice_gain,
        context::excess_jsd_slice,
    )
}

/// Unified flat-stride histogram clusterer for both `64 * num_lit_types` literal contexts
/// (`is_lit_64ctx = true`, `effort in 8..=12`) and `4 * num_dist_types` distance contexts
/// (`is_lit_64ctx = false`, arbitrary `alpha`), pricing every candidate with exact
/// `ContextMap::bit_cost + HuffmanCode::bit_cost`.
pub(crate) fn optimize_flat_clustering(
    hists: &[u32],
    totals: &[u32],
    alpha: usize,
    max_trees: usize,
    effort: u8,
    is_lit_64ctx: bool,
    hbuf: &mut [HuffmanCodeEntry],
) -> (Vec<u32>, Vec<u8>, u64) {
    let n = totals.len();
    let mut single = vec![0u32; alpha];
    let mut grand_total = 0u64;
    for (i, &t) in totals.iter().enumerate() {
        if t > 0 {
            grand_total += t as u64;
            add_hist_row(&mut single, &hists[i * alpha..(i + 1) * alpha]);
        }
    }
    if grand_total == 0 {
        single[0] = 1;
    }
    let mut best_bits = price_cmap_and_flat(&[0], &single, alpha, hbuf);
    let (mut best_flat, mut best_cmap) = (single, vec![0u8; n]);
    if grand_total == 0 || max_trees <= 1 || n <= 1 {
        return (best_flat, best_cmap, best_bits);
    }

    let (thresholds, lloyd_rounds): (&[i32], usize) = if !is_lit_64ctx {
        (&[20, 40, 72, 120], 0)
    } else {
        match effort {
            12.. => (&[14, 28, 52, 88, 140, 210], 3),
            11 => (&[24, 52, 104, 180], 2),
            10 => (&[40, 80, 140], 0),
            9 => (&[48], 1),
            ..=8 => (&[32], 0),
        }
    };

    let stage1 = (is_lit_64ctx && n.is_multiple_of(64) && ((effort <= 8 && n > 64) || n > 1024))
        .then(|| {
            let (cap, g) = if effort <= 8 {
                ((max_trees.div_ceil(n / 64) + 3).clamp(6, 12), 24)
            } else {
                (12, 20)
            };
            let (mut mh, mut mt, mut s1) = (Vec::new(), Vec::new(), vec![0usize; n]);
            for bt in 0..(n / 64) {
                let (lh, lt, lcm) = cluster_flat_mdl(
                    &hists[bt * 64 * MAX_LIT..(bt + 1) * 64 * MAX_LIT],
                    &totals[bt * 64..(bt + 1) * 64],
                    cap,
                    g,
                );
                let base = mt.len();
                for (c, &lc) in lcm.iter().enumerate() {
                    s1[bt * 64 + c] = base + (lc as usize);
                }
                mh.extend_from_slice(&lh);
                mt.extend_from_slice(&lt);
            }
            (mh, mt, s1)
        });
    let run_stage2 = |h: &[u32], t: &[u32], g: i32| {
        if is_lit_64ctx && effort <= 8 {
            cluster_flat_mdl(h, t, max_trees, g)
        } else {
            context::cluster_flat_jsd(h, t, alpha, max_trees, g)
        }
    };

    let mut best_seed: Option<(Vec<u32>, Vec<u32>, Vec<u8>)> = None;
    for &min_gain in thresholds {
        let (cl_h, cl_t, cm) = if let Some((mh, mt, s1)) = &stage1 {
            let (_, _, s2_cm) = run_stage2(mh, mt, min_gain);
            let mut cm: Vec<u8> = s1.iter().map(|&idx| s2_cm[idx]).collect();
            let (cl_h, cl_t) = rebuild_clusters_from_cmap(hists, totals, alpha, &mut cm);
            (cl_h, cl_t, cm)
        } else {
            run_stage2(hists, totals, min_gain)
        };
        if cl_t.len() > 1 && !cl_t.contains(&0) {
            let bits = price_cmap_and_flat(&cm, &cl_h, alpha, hbuf);
            if bits < best_bits {
                best_bits = bits;
                best_flat.clone_from(&cl_h);
                best_cmap.clone_from(&cm);
                if lloyd_rounds > 0 {
                    best_seed = Some((cl_h, cl_t, cm));
                }
            }
        }
    }

    if let Some((cl_h, cl_t, cmap)) = best_seed {
        lloyd_refine_flat(
            hists,
            totals,
            alpha,
            lloyd_rounds,
            cl_h,
            cl_t,
            cmap,
            hbuf,
            |ch, cm, hb| {
                let bits = price_cmap_and_flat(cm, ch, alpha, hb);
                if bits < best_bits {
                    best_bits = bits;
                    best_flat.clear();
                    best_flat.extend_from_slice(ch);
                    best_cmap.copy_from_slice(cm);
                }
            },
        );
    }

    (best_flat, best_cmap, best_bits)
}

fn prescreen_dist_params(
    raw_dists: &[(u8, u32)],
    raw_seen: usize,
    large_window: bool,
    two_ctx: bool,
    np: u8,
    nd: u8,
    counts: &mut [[u32; MAX_HUFFMAN_ALPHABET]; 2],
) -> u64 {
    let alpha = dist_alphabet_size(np, nd, large_window);
    let sym_limit = max_dist_symbol(np, nd, large_window);
    counts[0][16..alpha].fill(0);
    if two_ctx {
        counts[1][16..alpha].fill(0);
    }
    let (mut n0, mut n1, mut extra) = (0u32, 0u32, 0u64);
    for &(ctx, d) in raw_dists {
        let (s, nb, _) = encode_raw_distance_params(d, np, nd);
        if (s as usize) >= sym_limit {
            return u64::MAX;
        }
        let ci = usize::from(two_ctx && ctx != 0);
        counts[ci][s as usize] += 1;
        (n0, n1, extra) = (n0 + (1 - ci as u32), n1 + ci as u32, extra + nb as u64);
    }
    let mut entropy_fp16 = extra << 16;
    let (mut used_tot, mut span_hdr) = (0u64, 0u64);
    for (ci, n) in [(0, n0), (1, n1)] {
        if n == 0 {
            continue;
        }
        let (mut sym_fp16, mut max_sym) = (0u64, 16usize);
        for (s, &c) in counts[ci][16..alpha].iter().enumerate() {
            if c > 0 {
                used_tot += 1;
                max_sym = 16 + s;
                sym_fp16 += context::xlog2x_fp16(c);
            }
        }
        entropy_fp16 += context::xlog2x_fp16(n).saturating_sub(sym_fp16);
        span_hdr += ((max_sym - 16) as u64) / 4;
    }
    if raw_seen > raw_dists.len() {
        entropy_fp16 =
            (entropy_fp16 + used_tot * 47_274) * (raw_seen as u64) / (raw_dists.len() as u64);
    }
    let hdr_bits = if two_ctx {
        used_tot * 4 + span_hdr
    } else {
        used_tot * 6
    };
    entropy_fp16 + (hdr_bits << 16)
}

/// Fast 15-candidate sampled entropy selector for `Q6..=Q9` unsplit metablocks.
pub(crate) fn select_distance_params_from_samples(
    samples: &[(u8, u32)],
    raw_seen: usize,
    max_raw_d: u32,
    large_window: bool,
) -> (u8, u8) {
    if raw_seen < 48 || samples.len() < 24 {
        return (0, 0);
    }
    let mut counts = [[0u32; MAX_HUFFMAN_ALPHABET]; 2];
    #[rustfmt::skip]
    const CANDIDATES: [(u8, u8); 15] = [
        (0, 0), (0, 4), (0, 8), (0, 12), (0, 15),
        (1, 0), (1, 4), (1, 8), (1, 16), (1, 30),
        (2, 0), (2, 8), (2, 16), (2, 32), (2, 60),
    ];
    // Evaluating `encode_raw_distance_params` only at `max_raw_d` relies on `np <= 2`
    // (at `np == 3`, symbol indices are non-monotonic near `2^30`, e.g. `(3, 32)`).
    const {
        let mut i = 0;
        while i < CANDIDATES.len() {
            assert!(CANDIDATES[i].0 <= 2);
            i += 1;
        }
    };
    let base_cost = prescreen_dist_params(samples, raw_seen, large_window, true, 0, 0, &mut counts);
    let (mut best_cost, mut best_pair) = (base_cost, (0u8, 0u8));
    for &(np, nd) in &CANDIDATES[1..] {
        if large_window
            && max_raw_d > 0
            && (encode_raw_distance_params(max_raw_d, np, nd).0 as usize)
                >= max_dist_symbol(np, nd, true)
        {
            continue;
        }
        let c = prescreen_dist_params(samples, raw_seen, large_window, true, np, nd, &mut counts);
        if c < best_cost {
            (best_cost, best_pair) = (c, (np, nd));
        }
    }
    if best_pair != (0, 0) && best_cost + (64 << 16) < base_cost {
        best_pair
    } else {
        (0, 0)
    }
}

/// Full-grid prescreen -> top-K exact 4-context pricing for `Q10..=Q12` metablocks.
fn select_optimal_distance_params(
    cmds: &[OptimalCommand],
    large_window: bool,
    effort: u8,
    hbuf: &mut [HuffmanCodeEntry],
) -> (u8, u8) {
    let mut raw_dists: Vec<(u8, u32)> = Vec::new();
    let mut short_counts = [[0u32; 16]; 4];
    for c in cmds {
        if c.emits_dist_sym() {
            let d_ctx = copy_code_only(c.copy_len).min(3) as u8;
            if c.dist_mode < 16 {
                short_counts[d_ctx as usize][c.dist_mode as usize] += 1;
            } else {
                raw_dists.push((d_ctx, c.dist));
            }
        }
    }
    if raw_dists.len() < 16 {
        return (0, 0);
    }
    let mut counts = [[0u32; MAX_HUFFMAN_ALPHABET]; 2];
    let (mut cands, mut n_cands) = ([(0u64, 0u8, 0u8); 64], 0usize);
    let (max_np, k_step) = if effort >= 11 {
        (3u8, 1usize)
    } else {
        (2u8, 2usize)
    };
    for np in 0..=max_np {
        for k in (0..15u8).step_by(k_step).chain([15]) {
            let nd = k << np;
            let score = prescreen_dist_params(
                &raw_dists,
                raw_dists.len(),
                large_window,
                false,
                np,
                nd,
                &mut counts,
            );
            let mut j = n_cands;
            while j > 0 && cands[j - 1].0 > score {
                cands[j] = cands[j - 1];
                j -= 1;
            }
            cands[j] = (score, np, nd);
            n_cands += 1;
        }
    }
    let top_n = n_cands.min(match effort {
        12.. => 24,
        11 => 10,
        _ => 4,
    });
    if !cands[..top_n].iter().any(|&(_, np, nd)| np == 0 && nd == 0) {
        cands[top_n - 1] = (0, 0, 0);
    }

    let (mut best_pair, mut best_exact_bits) = ((0u8, 0u8), u64::MAX);
    let mut ctx_flat = Vec::with_capacity(4 * MAX_HUFFMAN_ALPHABET);
    'cand: for &(_, np, nd) in &cands[..top_n] {
        let alpha = dist_alphabet_size(np, nd, large_window);
        let sym_limit = max_dist_symbol(np, nd, large_window);
        ctx_flat.clear();
        ctx_flat.resize(4 * alpha, 0);
        let mut ctx_tots = [0u32; 4];
        for ctx in 0..4 {
            ctx_flat[ctx * alpha..ctx * alpha + 16].copy_from_slice(&short_counts[ctx]);
            ctx_tots[ctx] = short_counts[ctx].iter().sum();
        }
        let mut extra = 0u64;
        for &(d_ctx, d) in &raw_dists {
            let (s, eb, _) = encode_raw_distance_params(d, np, nd);
            if (s as usize) >= sym_limit {
                continue 'cand;
            }
            ctx_flat[(d_ctx as usize) * alpha + (s as usize)] += 1;
            ctx_tots[d_ctx as usize] += 1;
            extra += eb as u64;
        }
        let (_, _, tree_and_cmap_bits) =
            optimize_flat_clustering(&ctx_flat, &ctx_tots, alpha, 4, effort, false, hbuf);
        if tree_and_cmap_bits + extra < best_exact_bits {
            (best_exact_bits, best_pair) = (tree_and_cmap_bits + extra, (np, nd));
        }
    }
    best_pair
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_symbol_stream_homogeneous_and_multi() {
        let mut hbuf = vec![HuffmanCodeEntry::default(); MAX_HUFFMAN_ALPHABET];
        let (homo_n, seg_n, spec) = if cfg!(miri) {
            (512, 256, SplitSpec(256, 16, 64, 32))
        } else {
            (4096, 2048, SplitSpec(256, 16, 256, 64))
        };
        let homo: Vec<u16> = (0..homo_n).map(|i| ((i * 13 + 7) & 15) as u16).collect();
        assert_eq!(split_symbol_stream(&homo, spec, &mut hbuf).num_types, 1);

        let mut multi = Vec::with_capacity(seg_n * 4);
        let mut rng = 0xCAFE_BABE_1234_5678u64;
        for base in [0u16, 128, 0, 200] {
            for _ in 0..seg_n {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                multi.push(base + (((rng >> 58) as u16) & 7));
            }
        }
        let s_multi = split_symbol_stream(&multi, spec, &mut hbuf);
        assert_eq!((s_multi.num_types, s_multi.types[0]), (3, 0));
        assert_eq!(s_multi.lengths.iter().sum::<usize>(), multi.len());
        assert!(s_multi.types.windows(2).all(|w| w[0] != w[1]));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_large_window_distance_params_respect_max_dist_symbol() {
        let mut hbuf = vec![HuffmanCodeEntry::default(); MAX_HUFFMAN_ALPHABET];
        for base_stride in [4u32, 8] {
            for extreme_d in [
                0x4000_0026u32,
                0x6000_0001,
                0x6000_0021,
                0x7FFF_FFF1,
                0x7FFF_FFFC,
            ] {
                let mut cmds: Vec<OptimalCommand> = (0..600u32)
                    .map(|i| {
                        let d = if base_stride == 8 {
                            if i % 2 == 0 {
                                25 + ((i / 2) % 8)
                            } else {
                                609 + ((i / 2) % 8) * 8
                            }
                        } else {
                            4 + (i % 16) * 4
                        };
                        OptimalCommand::new(
                            1,
                            if (i & 1) == 0 { 4 } else { 6 },
                            5,
                            d,
                            crate::metablock::DIST_MODE_RAW,
                        )
                    })
                    .collect();
                let to_samples = |cs: &[OptimalCommand]| -> (Vec<(u8, u32)>, u32) {
                    let s: Vec<_> = cs
                        .iter()
                        .map(|c| (u8::from(c.copy_len > 4), c.dist))
                        .collect();
                    let m = cs.iter().map(|c| c.dist).max().unwrap_or(0);
                    (s, m)
                };
                let (s0, m0) = to_samples(&cmds);
                assert_ne!(
                    select_distance_params_from_samples(&s0, s0.len(), m0, true),
                    (0, 0)
                );
                if base_stride == 8 {
                    assert_eq!(
                        select_optimal_distance_params(&cmds, true, 12, &mut hbuf),
                        (3, 32)
                    );
                }

                cmds.push(OptimalCommand::new(
                    1,
                    5,
                    5,
                    extreme_d,
                    crate::metablock::DIST_MODE_STATIC_DICT,
                ));
                let (s1, m1) = to_samples(&cmds);
                for effort in [6u8, 10, 11, 12] {
                    let (np, nd) = if effort < 10 {
                        select_distance_params_from_samples(&s1, s1.len(), m1, true)
                    } else {
                        select_optimal_distance_params(&cmds, true, effort, &mut hbuf)
                    };
                    if extreme_d > 0x6000_0000 {
                        assert_ne!((np, nd), (3, 32));
                    }
                    for c in &cmds {
                        let s = encode_raw_distance_params(c.dist, np, nd).0 as usize;
                        assert!(
                            s < max_dist_symbol(np, nd, true),
                            "stride={base_stride} effort={effort} d={:#x} selected ({np}, {nd}) -> sym={s} >= {}",
                            c.dist,
                            max_dist_symbol(np, nd, true)
                        );
                    }
                }
            }
        }
    }
}
