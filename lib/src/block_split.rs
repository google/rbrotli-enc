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
    copy_code_only, dist_alphabet_size, encode_iac, encode_raw_distance_params, max_dist_symbol,
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

/// Runs $O(N_w \cdot K)$ 1D Viterbi DP (`min1`/`min2` transition trick) with an optional empirical
/// per-switch penalty from `seed_split`, followed by symbol-exact boundary polish (`±polish_rad`).
#[allow(clippy::too_many_arguments)]
fn viterbi_and_polish(
    total_len: usize,
    k_num: usize,
    viterbi_win: usize,
    polish_rad: usize,
    seed_split: Option<&BlockSplit>,
    hbuf: &mut [HuffmanCodeEntry],
    mut emit_window_costs: impl FnMut(usize, usize, &mut [u64]),
    mut sym_cost_q4: impl FnMut(usize, usize) -> u16,
) -> (BlockSplit, u64) {
    let num_win = total_len.div_ceil(viterbi_win);
    if k_num <= 1 || num_win <= 1 {
        return (BlockSplit::single_block(total_len), 1);
    }

    let switch_pen_q4 = match seed_split.filter(|s| s.lengths.len() >= 2 && s.num_types >= 2) {
        Some(s) => {
            let tc = HuffmanCode::from_counts(&s.type_counts, 15, hbuf)
                .1
                .payload_bit_cost(&s.type_counts);
            let lc = HuffmanCode::from_counts(&s.len_counts, 15, hbuf)
                .1
                .payload_bit_cost(&s.len_counts);
            let eb: u64 = s
                .lengths
                .iter()
                .map(|&l| block_count_to_sym_and_bits(l).1 as u64)
                .sum();
            let n_blk = s.lengths.len() as u64;
            ((tc * 16) / (n_blk - 1) + ((lc + eb) * 16) / n_blk).clamp(6 * 16, 15 * 16)
        }
        None => 11 * 16,
    };

    let mut emit = vec![0u64; k_num];
    let mut backptr = vec![0u8; num_win * k_num];
    let (mut dp_prev, mut dp_cur) = (vec![0u64; k_num], vec![0u64; k_num]);

    for w in 0..num_win {
        let ws = w * viterbi_win;
        emit_window_costs(ws, (ws + viterbi_win).min(total_len), &mut emit);

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
                let sw = best_other.saturating_add(switch_pen_q4);
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
        let min_b = (runs[r_idx].1 + 1).max(b0.saturating_sub(polish_rad));
        let max_b = (runs[r_idx + 1].2 - 1).min(b0 + polish_rad);
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
    let total: u32 = counts.iter().sum();
    if total == 0 {
        out_q4.fill(floor_q4);
        return;
    }
    let (_, code) = HuffmanCode::from_counts(counts, 15, hbuf);
    if let Some(s) = code.singleton {
        out_q4.fill(floor_q4);
        out_q4[s as usize] = 4;
        return;
    }
    let inv_tot = 1.0f32 / (total as f32);
    for (i, (&c, slot)) in counts.iter().zip(out_q4.iter_mut()).enumerate() {
        *slot = if c > 0 {
            let hlen_q4 = (code.code[i].len as u32) << 4;
            let p = (c as f32) * inv_tot;
            let sh_q4 = (((127i32 << 4) - ((p.to_bits() >> 19) as i32)).max(4)) as u32;
            (((hlen_q4 * 7 + sh_q4) >> 3) as u16).max(8)
        } else {
            floor_q4
        };
    }
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

#[inline(always)]
fn add_hist_row(dst: &mut [u32], src: &[u32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

fn viterbi_with_tables(
    syms: &[u16],
    tables: &[Vec<u16>],
    viterbi_win: usize,
    polish_rad: usize,
    seed_split: Option<&BlockSplit>,
    hbuf: &mut [HuffmanCodeEntry],
) -> (BlockSplit, u64) {
    viterbi_and_polish(
        syms.len(),
        tables.len(),
        viterbi_win,
        polish_rad,
        seed_split,
        hbuf,
        |ws, we, emit_out| {
            for (tbl, slot) in tables.iter().zip(emit_out.iter_mut()) {
                *slot = syms[ws..we].iter().map(|&s| tbl[s as usize] as u64).sum();
            }
        },
        |k, p| tables[k][syms[p] as usize],
    )
}

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
struct SplitSpec {
    alpha: usize,
    max_types: usize,
    micro_size: usize,
    mdl_gain: i32,
    mtf_polish: bool,
}

impl SplitSpec {
    #[inline]
    const fn new(
        alpha: usize,
        max_types: usize,
        micro_size: usize,
        mdl_gain: i32,
        mtf_polish: bool,
    ) -> Self {
        Self {
            alpha,
            max_types,
            micro_size,
            mdl_gain,
            mtf_polish,
        }
    }
}

fn split_symbol_stream(syms: &[u16], spec: SplitSpec, hbuf: &mut [HuffmanCodeEntry]) -> BlockSplit {
    let (n, alpha, viterbi_win) = (syms.len(), spec.alpha, spec.micro_size / 4);
    if n < spec.micro_size * 2 || spec.max_types <= 1 || alpha == 0 {
        return BlockSplit::single_block(n);
    }

    // --- Stage 1: Micro-block partition & JSD min-heap neighbor merge ---
    let num_micro = n.div_ceil(spec.micro_size);
    let mut seg_hists = vec![0u32; num_micro * alpha];
    let mut seg_totals = vec![0u32; num_micro];
    let mut prev_seg: Vec<usize> = (0..num_micro).map(|m| m.wrapping_sub(1)).collect();
    let mut next_seg: Vec<usize> = (0..num_micro)
        .map(|m| if m + 1 < num_micro { m + 1 } else { usize::MAX })
        .collect();
    let mut ver = vec![0u32; num_micro];

    for m in 0..num_micro {
        let (s, e) = (m * spec.micro_size, ((m + 1) * spec.micro_size).min(n));
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

    let (proto_flat, proto_tots, proto_cm) = context::cluster_flat::<false, false>(
        &macro_flat,
        &macro_tots,
        alpha,
        spec.max_types.min(m_cnt),
        spec.mdl_gain,
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
    let (cand_split, switch_bits) =
        viterbi_with_tables(syms, &cost_tables_q4, viterbi_win, viterbi_win, None, hbuf);
    if cand_split.num_types <= 1 || !spec.mtf_polish {
        return cand_split;
    }

    let mut per_type_flat = vec![0u32; cand_split.num_types * alpha];
    count_split_syms_flat(&cand_split, syms, alpha, &mut per_type_flat);
    let multi_trees = price_flat_trees(&per_type_flat, alpha, hbuf);
    let tbls2 = build_cost_tables_q4(&per_type_flat, alpha, hbuf);
    let fine_win = (viterbi_win / 2).max(16);
    let (cand2, sw2) = viterbi_with_tables(
        syms,
        &tbls2,
        fine_win,
        fine_win * 2,
        Some(&cand_split),
        hbuf,
    );
    if cand2.num_types >= 2 {
        let mut ptc2 = vec![0u32; cand2.num_types * alpha];
        count_split_syms_flat(&cand2, syms, alpha, &mut ptc2);
        if sw2 + price_flat_trees(&ptc2, alpha, hbuf) < switch_bits + multi_trees {
            return cand2;
        }
    }
    cand_split
}

#[inline]
fn choose_context_mode_for_positions(data: &[u8], pos: &[u32]) -> ContextMode {
    if pos.len() < 32 {
        ContextMode::UTF8
    } else {
        context::best_sampled_ctx_mode::<8>(&context::ALL_CONTEXT_MODES, data, pos.len(), |i| {
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
    min_net_gain: i32,
) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    context::cluster_flat::<false, true>(
        hists,
        totals,
        MAX_LIT,
        max_clusters,
        min_net_gain,
        mdl_slice_gain,
        |s, ts, h, th| {
            context::histogram_distance_slice(s, ts, h, th) - context::jsd_null_bias(s, h)
        },
    )
}

fn precompact_local_64ctx_flat(
    hists: &[u32],
    totals: &[u32],
    local_cap: usize,
    min_gain: i32,
) -> (Vec<u32>, Vec<u32>, Vec<usize>) {
    let num_bt = totals.len() / 64;
    let mut merged_h = Vec::with_capacity(num_bt * local_cap * MAX_LIT);
    let mut merged_t = Vec::with_capacity(num_bt * local_cap);
    let mut stage1_map = vec![0usize; totals.len()];
    for bt in 0..num_bt {
        let (lh, lt, lcm) = cluster_flat_mdl(
            &hists[bt * 64 * MAX_LIT..(bt + 1) * 64 * MAX_LIT],
            &totals[bt * 64..(bt + 1) * 64],
            local_cap,
            min_gain,
        );
        let base = merged_t.len();
        for (c, &lc) in lcm.iter().enumerate() {
            stage1_map[bt * 64 + c] = base + (lc as usize);
        }
        merged_h.extend_from_slice(&lh);
        merged_t.extend_from_slice(&lt);
    }
    (merged_h, merged_t, stage1_map)
}

/// Unified flat-stride histogram clusterer for both `64 * num_lit_types` literal contexts
/// (`is_lit_64ctx = true`, `effort in 8..=12`) and `4 * num_dist_types` distance contexts
/// (`is_lit_64ctx = false`, arbitrary `alpha`), pricing every candidate with exact
/// `ContextMap::bit_cost + HuffmanCode::bit_cost`.
fn optimize_flat_clustering(
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
            9 => (&[48], 1),
            ..=8 => (&[32], 0),
            10 => (&[40, 80, 140], 0),
        }
    };

    let stage1 = (is_lit_64ctx && n.is_multiple_of(64) && ((effort <= 8 && n > 64) || n > 1024))
        .then(|| {
            let cap = if effort <= 8 {
                (max_trees.div_ceil(n / 64) + 3).clamp(6, 12)
            } else {
                12
            };
            precompact_local_64ctx_flat(hists, totals, cap, if effort <= 8 { 24 } else { 20 })
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
    let (mut n_ctx, mut extra) = ([0u32; 2], 0u64);
    for &(ctx, d) in raw_dists {
        let (s, nb, _) = encode_raw_distance_params(d, np, nd);
        if (s as usize) >= sym_limit {
            return u64::MAX;
        }
        let ci = usize::from(two_ctx && ctx != 0);
        counts[ci][s as usize] += 1;
        n_ctx[ci] += 1;
        extra += nb as u64;
    }
    let mut entropy_fp16 = extra << 16;
    let (mut used_tot, mut span_hdr) = (0u64, 0u64);
    for (ci, n) in n_ctx.into_iter().enumerate() {
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

/// Fast `Q6..=Q9` sampled entropy `(npostfix, ndirect)` selector (zero Huffman builds).
pub(crate) fn select_distance_params_from_samples(
    raw_dists: &[(u8, u32)],
    raw_seen: usize,
    max_raw_d: u32,
    large_window: bool,
) -> (u8, u8) {
    if raw_dists.len() < 48 {
        return (0, 0);
    }
    #[rustfmt::skip]
    const CANDIDATES: [(u8, u8); 15] = [
        (0, 0), (0, 4), (0, 8), (0, 12), (0, 15),
        (1, 0), (1, 4), (1, 8), (1, 16), (1, 30),
        (2, 0), (2, 8), (2, 16), (2, 32), (2, 60),
    ];
    let mut counts = [[0u32; MAX_HUFFMAN_ALPHABET]; 2];
    let base_cost =
        prescreen_dist_params(raw_dists, raw_seen, large_window, true, 0, 0, &mut counts);
    let (mut best_cost, mut best_pair) = (base_cost, (0u8, 0u8));
    for &(np, nd) in &CANDIDATES[1..] {
        if large_window
            && max_raw_d > 0
            && (encode_raw_distance_params(max_raw_d, np, nd).0 as usize)
                >= max_dist_symbol(np, nd, true)
        {
            continue;
        }
        let c = prescreen_dist_params(raw_dists, raw_seen, large_window, true, np, nd, &mut counts);
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

/// `Q10..=Q12` `(npostfix, ndirect)` selector: entropy prescreen -> top-K exact 4-context pricing.
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
    let mut cands: Vec<(u64, u8, u8)> = Vec::with_capacity(64);
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
            cands.push((score, np, nd));
        }
    }
    cands.sort_unstable_by_key(|x| x.0);
    let top_n = cands.len().min(match effort {
        12.. => 24,
        11 => 10,
        _ => 8,
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

#[derive(Clone, Copy)]
struct LitStream<'a> {
    data: &'a [u8],
    syms: &'a [u16],
    pos: &'a [u32],
}

/// Stage-4 Contextual Literal Viterbi Polish: refines a multi-block literal `BlockSplit` using
/// each block type's `ContextMode[bt]` and clustered `ContextMap` Huffman code lengths.
fn polish_contextual_lit_split(
    lits: LitStream<'_>,
    seed_split: &BlockSplit,
    enc: &LitEnc,
    viterbi_win: usize,
    hbuf: &mut [HuffmanCodeEntry],
) -> Option<(BlockSplit, u64)> {
    let (n, num_bt) = (lits.syms.len(), seed_split.num_types);
    if num_bt < 2 || n < viterbi_win * 2 || enc.tree_counts.is_empty() {
        return None;
    }

    let tree_cost_q4 = build_cost_tables_q4(enc.tree_counts.as_flattened(), MAX_LIT, hbuf);
    let mut mode_used = [false; 4];
    for &m in &enc.context_modes[..num_bt] {
        mode_used[m as usize] = true;
    }
    let luts = [
        ContextMode::LSB6,
        ContextMode::MSB6,
        ContextMode::UTF8,
        ContextMode::Signed,
    ]
    .map(context::context_luts);
    let ctx_by_pos: Vec<[u8; 4]> = lits
        .pos
        .iter()
        .map(|&sp| {
            std::array::from_fn(|m| {
                if mode_used[m] {
                    context::stream_byte_ctx(luts[m].0, luts[m].1, lits.data, sp as usize) as u8
                } else {
                    0
                }
            })
        })
        .collect();

    let (polished, sw) = viterbi_and_polish(
        n,
        num_bt,
        viterbi_win,
        viterbi_win * 2,
        Some(seed_split),
        hbuf,
        |ws, we, emit_out| {
            for (bt, slot) in emit_out.iter_mut().enumerate().take(num_bt) {
                let (m, map_row) = (
                    enc.context_modes[bt] as usize,
                    &enc.cmap[bt * 64..(bt + 1) * 64],
                );
                *slot = (ws..we)
                    .map(|idx| {
                        let tid = map_row[ctx_by_pos[idx][m] as usize] as usize;
                        tree_cost_q4[tid][lits.syms[idx] as usize] as u64
                    })
                    .sum();
            }
        },
        |bt, idx| {
            let tid =
                enc.cmap[bt * 64 + (ctx_by_pos[idx][enc.context_modes[bt] as usize] as usize)];
            tree_cost_q4[tid as usize][lits.syms[idx] as usize]
        },
    );
    (polished.num_types >= 2).then_some((polished, sw))
}

fn unpack_lit_and_iac_streams(
    data: &[u8],
    mb_start: usize,
    mb_end: usize,
    commands: &[OptimalCommand],
    trailing_literals: u32,
) -> (Vec<u16>, Vec<u32>, Vec<u16>, u64) {
    let total_lits = commands
        .iter()
        .map(|c| c.insert_len as usize)
        .sum::<usize>()
        + (trailing_literals as usize);
    let mut lit_syms = Vec::with_capacity(total_lits);
    let mut lit_stream_pos = Vec::with_capacity(total_lits);
    let mut iac_syms = Vec::with_capacity(commands.len() + usize::from(trailing_literals > 0));
    let (mut iac_extra_bits_total, mut stream_pos) = (0u64, mb_start);

    let mut push_cmd = |ins_len: u32, copy_len: u32, implicit_d0: bool, adv_len: usize| {
        let ins_end = stream_pos + (ins_len as usize);
        if stream_pos < ins_end {
            lit_syms.extend(data[stream_pos..ins_end].iter().map(|&b| b as u16));
            lit_stream_pos.extend(stream_pos as u32..ins_end as u32);
        }
        let (iac_sym, iac_eb, _) = encode_iac(ins_len, copy_len, implicit_d0);
        iac_syms.push(iac_sym);
        iac_extra_bits_total += iac_eb as u64;
        stream_pos = ins_end + adv_len;
    };
    for &c in commands {
        push_cmd(
            c.insert_len,
            c.copy_len,
            c.is_implicit_d0(),
            c.out_len as usize,
        );
    }
    if trailing_literals > 0 {
        push_cmd(trailing_literals, 2, false, 0);
    }
    debug_assert_eq!(stream_pos, mb_end);
    (lit_syms, lit_stream_pos, iac_syms, iac_extra_bits_total)
}

#[inline(always)]
fn compact_dist_feature(d_ctx: u8, dist_mode: u8, raw_dist: u32) -> u16 {
    let bucket: u16 = if dist_mode < 16 {
        dist_mode as u16
    } else {
        let dm1 = raw_dist.saturating_sub(1);
        if dm1 < 4 {
            16 + (dm1 as u16)
        } else {
            let lg = (31 - dm1.leading_zeros()) as u16;
            (16 + 2 * lg + (((dm1 >> (lg - 1)) & 1) as u16)).min(63)
        }
    };
    ((d_ctx as u16) << 6) | bucket
}

fn unpack_dist_streams(
    commands: &[OptimalCommand],
    npostfix: u8,
    ndirect: u8,
    need_feat: bool,
) -> (Vec<u16>, Vec<u8>, Vec<u16>, u64) {
    let mut dist_syms = Vec::with_capacity(commands.len());
    let mut dist_ctxs = Vec::with_capacity(commands.len());
    let mut dist_feat = Vec::with_capacity(if need_feat { commands.len() } else { 0 });
    let mut dist_extra_bits_total = 0u64;
    for &cmd in commands {
        if cmd.emits_dist_sym() {
            let d_ctx = copy_code_only(cmd.copy_len).min(3) as u8;
            dist_ctxs.push(d_ctx);
            dist_syms.push(if cmd.dist_mode < 16 {
                cmd.dist_mode as u16
            } else {
                let (s, eb, _) = encode_raw_distance_params(cmd.dist, npostfix, ndirect);
                dist_extra_bits_total += eb as u64;
                s
            });
            if need_feat {
                dist_feat.push(compact_dist_feature(d_ctx, cmd.dist_mode, cmd.dist));
            }
        }
    }
    (dist_syms, dist_ctxs, dist_feat, dist_extra_bits_total)
}

fn encode_iac_split(
    split: &BlockSplit,
    iac_syms: &[u16],
    hb: &mut [HuffmanCodeEntry],
) -> (Vec<[u32; MAX_IAC]>, u64) {
    let mut flat = vec![0u32; split.num_types * MAX_IAC];
    count_split_syms_flat(split, iac_syms, MAX_IAC, &mut flat);
    let bits = price_flat_trees(&flat, MAX_IAC, hb);
    (flat.as_chunks::<MAX_IAC>().0.to_vec(), bits)
}

fn encode_dist_split(
    split: &BlockSplit,
    dist_syms: &[u16],
    dist_ctxs: &[u8],
    dist_alpha: usize,
    max_dist_trees: usize,
    hb: &mut [HuffmanCodeEntry],
) -> ((Vec<u8>, Vec<Vec<u32>>), u64) {
    if dist_syms.is_empty() {
        let mut dummy = vec![0u32; dist_alpha];
        dummy[0] = 1;
        let bits = price_cmap_and_flat(&[0], &dummy, dist_alpha, hb);
        return ((vec![0u8; 4], vec![dummy]), bits);
    }
    let n_slots = split.num_types * 4;
    let (mut d_flat, mut d_tots, mut idx) =
        (vec![0u32; n_slots * dist_alpha], vec![0u32; n_slots], 0);
    for (&bt, &blen) in split.types.iter().zip(split.lengths.iter()) {
        for k in idx..(idx + blen) {
            let slot = (bt as usize) * 4 + (dist_ctxs[k] as usize);
            d_flat[slot * dist_alpha + (dist_syms[k] as usize)] += 1;
            d_tots[slot] += 1;
        }
        idx += blen;
    }
    let (best_flat, cmap, bits) =
        optimize_flat_clustering(&d_flat, &d_tots, dist_alpha, max_dist_trees, 10, false, hb);
    let trees = best_flat
        .chunks_exact(dist_alpha)
        .map(<[u32]>::to_vec)
        .collect();
    ((cmap, trees), bits)
}

struct LitEnc {
    context_modes: Vec<ContextMode>,
    cmap: Vec<u8>,
    tree_counts: Vec<[u32; MAX_LIT]>,
}

fn encode_lit_split(
    lits: LitStream<'_>,
    split: &BlockSplit,
    max_lit_trees: usize,
    effort: u8,
    hb: &mut [HuffmanCodeEntry],
) -> (LitEnc, u64) {
    if lits.syms.is_empty() {
        let mut dummy = [0u32; MAX_LIT];
        dummy[0] = 1;
        return (
            LitEnc {
                context_modes: vec![ContextMode::UTF8],
                cmap: vec![0u8; 64],
                tree_counts: vec![dummy],
            },
            2 + price_cmap_and_flat(&[0], &dummy, MAX_LIT, hb),
        );
    }

    let (num_bt, is_q11, is_q12) = (split.num_types, effort >= 11, effort >= 12);
    let (mut pos_by_bt, mut idx) = (vec![Vec::new(); num_bt], 0usize);
    for (&bt, &blen) in split.types.iter().zip(split.lengths.iter()) {
        pos_by_bt[bt as usize].extend_from_slice(&lits.pos[idx..idx + blen]);
        idx += blen;
    }

    let mut cmodes = Vec::with_capacity(num_bt);
    let mut ranked_by_bt = Vec::with_capacity(if is_q11 { num_bt } else { 0 });
    let mut ctx_flat = vec![0u32; num_bt * 64 * MAX_LIT];
    let mut ctx_tots = vec![0u32; num_bt * 64];
    for (bt, pos) in pos_by_bt.iter().enumerate() {
        let mode = if pos.len() < 32 {
            if is_q11 {
                ranked_by_bt.push(context::ALL_CONTEXT_MODES.map(|m| (m, u64::MAX)));
            }
            ContextMode::UTF8
        } else {
            let ranked = context::rank_sampled_ctx_modes::<8>(
                &context::ALL_CONTEXT_MODES,
                lits.data,
                pos.len(),
                |i| pos[i] as usize,
            );
            if is_q11 {
                ranked_by_bt.push(ranked);
            }
            ranked[0].0
        };
        cmodes.push(mode);
        fill_block_ctx_histograms_flat(
            lits.data,
            pos,
            mode,
            &mut ctx_flat[bt * 64 * MAX_LIT..(bt + 1) * MAX_LIT * 64],
            &mut ctx_tots[bt * 64..(bt + 1) * 64],
        );
    }

    let (mut best_flat, mut best_cmap, mut best_cbits) = optimize_flat_clustering(
        &ctx_flat,
        &ctx_tots,
        MAX_LIT,
        max_lit_trees,
        effort,
        true,
        hb,
    );

    if is_q11 {
        let mut probes: Vec<(usize, ContextMode, u8)> = Vec::new();
        if num_bt == 1 {
            if pos_by_bt[0].len() >= 128 {
                let r = &ranked_by_bt[0];
                let lim = r[0].1
                    + (r[0].1 * if is_q12 { 15 } else { 7 }) / 100
                    + (if is_q12 { 128 } else { 64 } << 16);
                for &(m, sc) in r.iter().take(if is_q12 { 4 } else { 2 }).skip(1) {
                    if sc <= lim {
                        probes.push((0, m, effort));
                    }
                }
            }
        } else {
            let pct = if is_q12 { 12 } else { 5 };
            let mut bts: Vec<usize> = (0..num_bt)
                .filter(|&bt| {
                    let r = &ranked_by_bt[bt];
                    pos_by_bt[bt].len() >= 160
                        && r[1].1 <= r[0].1 + (r[0].1 * pct) / 100 + (48 << 16)
                })
                .collect();
            bts.sort_by_key(|&bt| Reverse(pos_by_bt[bt].len()));
            for &bt in bts.iter().take(if is_q12 { 6 } else { 3 }) {
                probes.push((bt, ranked_by_bt[bt][1].0, if is_q12 { 11 } else { 9 }));
            }
        }
        if !probes.is_empty() {
            let (mut saved_h, mut saved_t) = (vec![0u32; 64 * MAX_LIT], [0u32; 64]);
            for (bt, alt_mode, p_eff) in probes {
                let (hr, tr) = (
                    bt * 64 * MAX_LIT..(bt + 1) * 64 * MAX_LIT,
                    bt * 64..(bt + 1) * 64,
                );
                saved_h.copy_from_slice(&ctx_flat[hr.clone()]);
                saved_t.copy_from_slice(&ctx_tots[tr.clone()]);
                fill_block_ctx_histograms_flat(
                    lits.data,
                    &pos_by_bt[bt],
                    alt_mode,
                    &mut ctx_flat[hr.clone()],
                    &mut ctx_tots[tr.clone()],
                );
                let (f, cm, cb) = optimize_flat_clustering(
                    &ctx_flat,
                    &ctx_tots,
                    MAX_LIT,
                    max_lit_trees,
                    p_eff,
                    true,
                    hb,
                );
                if cb < best_cbits {
                    (best_cbits, best_flat, best_cmap, cmodes[bt]) = (cb, f, cm, alt_mode);
                } else {
                    ctx_flat[hr].copy_from_slice(&saved_h);
                    ctx_tots[tr].copy_from_slice(&saved_t);
                }
            }
        }
    }

    (
        LitEnc {
            context_modes: cmodes,
            cmap: best_cmap,
            tree_counts: best_flat.as_chunks::<MAX_LIT>().0.to_vec(),
        },
        (2 * num_bt as u64) + best_cbits,
    )
}

struct StreamPlan<E> {
    split: BlockSplit,
    switch_bits: u64,
    content_bits: u64,
    enc: E,
}

impl<E> StreamPlan<E> {
    fn from_split(
        mut split: BlockSplit,
        hb: &mut [HuffmanCodeEntry],
        encode: impl FnOnce(&BlockSplit, &mut [HuffmanCodeEntry]) -> (E, u64),
    ) -> Self {
        let switch_bits = populate_and_price_block_switches(&mut split, hb);
        let (enc, content_bits) = encode(&split, hb);
        Self {
            split,
            switch_bits,
            content_bits,
            enc,
        }
    }

    fn eval_multi(
        cand: BlockSplit,
        hb: &mut [HuffmanCodeEntry],
        encode: impl FnOnce(&BlockSplit, &mut [HuffmanCodeEntry]) -> (E, u64),
    ) -> Option<Self> {
        (cand.num_types >= 2).then(|| Self::from_split(cand, hb, encode))
    }

    #[inline]
    fn total_bits(&self) -> u64 {
        self.switch_bits + self.content_bits
    }

    fn try_replace(&mut self, cand: Option<Self>) {
        if let Some(c) = cand.filter(|c| c.total_bits() < self.total_bits()) {
            *self = c;
        }
    }
}

fn maybe_par4<A: Send, B: Send, C: Send, D: Send>(
    parallel: bool,
    hb: &mut [HuffmanCodeEntry],
    fa: impl FnOnce(&mut [HuffmanCodeEntry]) -> A,
    fb: impl FnOnce(&mut [HuffmanCodeEntry]) -> B + Send,
    fc: impl FnOnce(&mut [HuffmanCodeEntry]) -> C + Send,
    fd: impl FnOnce(&mut [HuffmanCodeEntry]) -> D + Send,
) -> (A, B, C, D) {
    if parallel && !cfg!(target_family = "wasm") {
        std::thread::scope(|s| {
            let hb_b = s.spawn(|| fb(&mut vec![HuffmanCodeEntry::default(); MAX_HUFFMAN_ALPHABET]));
            let hb_c = s.spawn(|| fc(&mut vec![HuffmanCodeEntry::default(); MAX_HUFFMAN_ALPHABET]));
            let hb_d = s.spawn(|| fd(&mut vec![HuffmanCodeEntry::default(); MAX_HUFFMAN_ALPHABET]));
            (
                fa(hb),
                hb_b.join().unwrap(),
                hb_c.join().unwrap(),
                hb_d.join().unwrap(),
            )
        })
    } else {
        (fa(hb), fb(hb), fc(hb), fd(hb))
    }
}

/// Unified `MetablockPlan` engine for both `Q8–Q9` (`fixed_dist = Some(..)`) and `Q10–Q12`
/// (`fixed_dist = None`, `1T == MT` deterministic).
#[allow(clippy::too_many_arguments)]
fn try_plan_metablock(
    data: &[u8],
    mb_start: usize,
    mb_end: usize,
    commands: &[OptimalCommand],
    trailing_literals: u32,
    large_window: bool,
    fixed_dist: Option<(u8, u8)>,
    base_max_lit_trees: usize,
    effort: u8,
    num_threads: usize,
) -> Option<MetablockPlan> {
    let mb_len = mb_end - mb_start;
    let fast_only = fixed_dist.is_some();
    if fast_only && mb_len < 32_768 {
        return None;
    }
    assert!(mb_len >= 1);

    let (is_q11, is_q12) = (effort >= 11, effort >= 12);
    let mut hbuf = vec![HuffmanCodeEntry::default(); MAX_HUFFMAN_ALPHABET];

    // Rebase `data` to `mb_start.saturating_sub(2)` so literal positions fit in `u32`
    // (`<= 16 MiB + 2 << u32::MAX`) while preserving the 2-byte literal context lookbehind
    // `(p1, p2)` across arbitrary 64-bit `>= 4 GiB` stream offsets.
    let ctx_base = mb_start.saturating_sub(2);
    let mb_data = &data[ctx_base..mb_end];
    let (lit_syms, lit_stream_pos, iac_syms, iac_extra_bits) = unpack_lit_and_iac_streams(
        mb_data,
        mb_start - ctx_base,
        mb_end - ctx_base,
        commands,
        trailing_literals,
    );
    let lits = LitStream {
        data: mb_data,
        syms: &lit_syms,
        pos: &lit_stream_pos,
    };
    let (npostfix, ndirect) = fixed_dist.unwrap_or_else(|| {
        select_optimal_distance_params(commands, large_window, effort, &mut hbuf)
    });
    let dist_alpha = dist_alphabet_size(npostfix, ndirect, large_window);
    let (dist_syms, dist_ctxs, dist_feat, dist_extra_bits) =
        unpack_dist_streams(commands, npostfix, ndirect, !fast_only);

    let (lit_max_t, lit_micro, lit_gain) = if fast_only {
        let s = (lit_syms.len() / 16_384).clamp(6, 36);
        (
            if effort == 9 { s.max(8) } else { s.min(32) },
            1024,
            if effort == 9 { 96 } else { 112 },
        )
    } else if is_q11 {
        (
            (lit_syms.len() / 2048).clamp(2, if is_q12 { 36 } else { 24 }),
            256,
            68,
        )
    } else {
        (8, 256, 72)
    };
    let lit_spec = SplitSpec::new(
        MAX_LIT,
        lit_max_t,
        lit_micro,
        lit_gain,
        !fast_only && is_q11,
    );
    let (iac_max_t, iac_micro, iac_gain) = if fast_only {
        (if effort == 9 { 6 } else { 4 }, 512, 144)
    } else if is_q11 {
        (if is_q12 { 32 } else { 24 }, 128, 112)
    } else {
        (6, 128, 112)
    };
    let iac_spec = SplitSpec::new(MAX_IAC, iac_max_t, iac_micro, iac_gain, is_q11);
    let dist_fine_cap = if is_q12 {
        24
    } else if is_q11 {
        16
    } else {
        4
    };
    let (dist_max_t, dist_micro, dist_gain) = if fast_only {
        (4, 512, 112)
    } else {
        (dist_fine_cap, 128, 68)
    };
    let dist_spec = SplitSpec::new(dist_alpha, dist_max_t, dist_micro, dist_gain, is_q11);

    let (pre_lf, pre_ic, pre_dc) = if fast_only {
        let lf = split_symbol_stream(&lit_syms, lit_spec, &mut hbuf);
        let ic = split_symbol_stream(&iac_syms, iac_spec, &mut hbuf);
        let dc = split_symbol_stream(&dist_syms, dist_spec, &mut hbuf);
        if lf.num_types == 1 && ic.num_types == 1 && dc.num_types == 1 {
            return None;
        }
        (Some(lf), Some(ic), Some(dc))
    } else {
        (None, None, None)
    };

    let enc_lit = |sp: &BlockSplit, hb: &mut [HuffmanCodeEntry]| {
        let max_t = if fast_only {
            (base_max_lit_trees * sp.num_types).min(if effort == 9 { 224 } else { 144 })
        } else if is_q11 {
            64
        } else {
            16
        };
        encode_lit_split(lits, sp, max_t, if fast_only { 8 } else { effort }, hb)
    };
    let eval_lit_cand =
        |cand: BlockSplit, hb: &mut [HuffmanCodeEntry]| StreamPlan::eval_multi(cand, hb, enc_lit);

    let (
        (mut lit_plan, coarse_lit),
        (fine_nt, fine_eval),
        ctx_eval,
        ((iac_plan, iac_1b), (dist_plan, dist_1b)),
    ) = maybe_par4(
        !fast_only && num_threads >= 2 && iac_syms.len() >= 512,
        &mut hbuf,
        |hb| {
            let base =
                StreamPlan::from_split(BlockSplit::single_block(lit_syms.len()), hb, enc_lit);
            let coarse = (!fast_only && is_q11).then(|| {
                split_symbol_stream(&lit_syms, SplitSpec::new(MAX_LIT, 8, 512, 160, true), hb)
            });
            (base, coarse)
        },
        |hb| {
            let cand = pre_lf.unwrap_or_else(|| split_symbol_stream(&lit_syms, lit_spec, hb));
            (cand.num_types, eval_lit_cand(cand, hb))
        },
        |hb| {
            if !fast_only && is_q11 && lit_syms.len() >= 1024 {
                let mode = choose_context_mode_for_positions(lits.data, &lit_stream_pos);
                let (lut0, lut1) = context::context_luts(mode);
                let ctx_syms: Vec<u16> = lits
                    .pos
                    .iter()
                    .zip(lits.syms)
                    .map(|(&sp, &b)| {
                        ((context::stream_byte_ctx(lut0, lut1, lits.data, sp as usize) >> 4) as u16)
                            << 8
                            | b
                    })
                    .collect();
                let max_t = (lit_syms.len() / 2048).clamp(2, if is_q12 { 24 } else { 16 });
                let cand =
                    split_symbol_stream(&ctx_syms, SplitSpec::new(1024, max_t, 512, 136, true), hb);
                eval_lit_cand(cand, hb)
            } else {
                None
            }
        },
        |hb| {
            let mut iac =
                StreamPlan::from_split(BlockSplit::single_block(iac_syms.len()), hb, |s, h| {
                    encode_iac_split(s, &iac_syms, h)
                });
            let iac_1b = iac.total_bits();
            let try_iac =
                |iac: &mut StreamPlan<_>, cand: BlockSplit, hb: &mut [HuffmanCodeEntry]| {
                    iac.try_replace(StreamPlan::eval_multi(cand, hb, |s, h| {
                        encode_iac_split(s, &iac_syms, h)
                    }));
                };
            let ic = pre_ic.unwrap_or_else(|| split_symbol_stream(&iac_syms, iac_spec, hb));
            let ic_nt = ic.num_types;
            try_iac(&mut iac, ic, hb);
            if !fast_only && is_q11 && ic_nt > 6 {
                let coarse =
                    split_symbol_stream(&iac_syms, SplitSpec::new(MAX_IAC, 8, 256, 200, true), hb);
                if coarse.num_types != ic_nt {
                    try_iac(&mut iac, coarse, hb);
                }
            }
            if !fast_only && is_q12 && iac_syms.len() >= 1024 {
                let ultra =
                    split_symbol_stream(&iac_syms, SplitSpec::new(MAX_IAC, 32, 64, 92, true), hb);
                try_iac(&mut iac, ultra, hb);
            }

            let dist_trees = |nt: usize| match () {
                _ if fast_only => {
                    if nt == 1 {
                        2
                    } else {
                        (2 * nt).clamp(2, 6)
                    }
                }
                _ if is_q12 => 32,
                _ if is_q11 => 16,
                _ => 8,
            };
            let mut dist =
                StreamPlan::from_split(BlockSplit::single_block(dist_syms.len()), hb, |s, h| {
                    encode_dist_split(s, &dist_syms, &dist_ctxs, dist_alpha, dist_trees(1), h)
                });
            let dist_1b = dist.total_bits();
            let try_dist =
                |dist: &mut StreamPlan<_>, cand: BlockSplit, hb: &mut [HuffmanCodeEntry]| {
                    let d_cap = dist_trees(cand.num_types);
                    dist.try_replace(StreamPlan::eval_multi(cand, hb, |s, h| {
                        encode_dist_split(s, &dist_syms, &dist_ctxs, dist_alpha, d_cap, h)
                    }));
                };
            let dc = pre_dc.unwrap_or_else(|| split_symbol_stream(&dist_syms, dist_spec, hb));
            let dc_nt = dc.num_types;
            try_dist(&mut dist, dc, hb);
            if !fast_only {
                if is_q11 {
                    let dc_coarse = split_symbol_stream(
                        &dist_syms,
                        SplitSpec::new(dist_alpha, 6, 256, 144, true),
                        hb,
                    );
                    if dc_coarse.num_types != dc_nt {
                        try_dist(&mut dist, dc_coarse, hb);
                    }
                }
                if dist_feat.len() >= if is_q11 { 256 } else { 512 } {
                    let df = split_symbol_stream(
                        &dist_feat,
                        SplitSpec::new(
                            256,
                            dist_fine_cap,
                            if is_q11 { 128 } else { 256 },
                            if is_q11 { 84 } else { 92 },
                            is_q11,
                        ),
                        hb,
                    );
                    try_dist(&mut dist, df, hb);
                }
            }
            ((iac, iac_1b), (dist, dist_1b))
        },
    );

    let single_total = lit_plan.total_bits() + iac_1b + dist_1b;
    let coarse_eval = coarse_lit
        .filter(|c| c.num_types != fine_nt)
        .and_then(|c| eval_lit_cand(c, &mut hbuf));
    lit_plan.try_replace(fine_eval);
    lit_plan.try_replace(coarse_eval);
    lit_plan.try_replace(ctx_eval);

    if is_q11 && lit_plan.split.num_types >= 2 && lit_syms.len() >= 512 {
        let ctx_viterbi_win = if is_q12 { 16 } else { 32 };
        for _ in 0..if is_q12 { 2 } else { 1 } {
            let Some((p_sp, p_sw)) = polish_contextual_lit_split(
                lits,
                &lit_plan.split,
                &lit_plan.enc,
                ctx_viterbi_win,
                &mut hbuf,
            ) else {
                break;
            };
            let (p_enc, p_cbits) = enc_lit(&p_sp, &mut hbuf);
            if p_sw + p_cbits < lit_plan.total_bits() {
                lit_plan = StreamPlan {
                    split: p_sp,
                    switch_bits: p_sw,
                    content_bits: p_cbits,
                    enc: p_enc,
                };
            } else {
                break;
            }
        }
    }

    if fast_only {
        let multi_total = lit_plan.total_bits() + iac_plan.total_bits() + dist_plan.total_bits();
        if (lit_plan.split.num_types == 1
            && iac_plan.split.num_types == 1
            && dist_plan.split.num_types == 1)
            || multi_total + 128 >= single_total
        {
            return None;
        }
    }

    let mnibbles = u64::from(mb_len > (1 << 16)) + u64::from(mb_len > (1 << 20));
    let exact_compressed_bits = (26 + 4 * mnibbles)
        + lit_plan.total_bits()
        + iac_plan.total_bits()
        + iac_extra_bits
        + dist_plan.total_bits()
        + dist_extra_bits;
    Some(MetablockPlan {
        npostfix,
        ndirect,
        lit_split: lit_plan.split,
        iac_split: iac_plan.split,
        dist_split: dist_plan.split,
        context_modes: lit_plan.enc.context_modes,
        lit_cmap: lit_plan.enc.cmap,
        lit_tree_counts: lit_plan.enc.tree_counts,
        iac_tree_counts: iac_plan.enc,
        dist_cmap: dist_plan.enc.0,
        dist_alpha,
        dist_tree_counts: dist_plan.enc.1,
        exact_compressed_bits,
    })
}

/// Effort- and thread-aware `MetablockPlan` builder (`effort` in `{10, 11, 12}`): produces
/// **100% bit-for-bit identical** output across all `num_threads` (`1, 2, 4, 8, ...`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_metablock_plan_with_effort(
    data: &[u8],
    mb_start: usize,
    mb_end: usize,
    commands: &[OptimalCommand],
    trailing_literals: u32,
    large_window: bool,
    effort: u8,
    num_threads: usize,
) -> MetablockPlan {
    try_plan_metablock(
        data,
        mb_start,
        mb_end,
        commands,
        trailing_literals,
        large_window,
        None,
        0,
        effort,
        num_threads,
    )
    .unwrap()
}

/// Fast single-pass intra-metablock `BlockSplit` planner for `Q8–Q9` (`C::CTX_BLOCK_SPLIT`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_build_fast_metablock_plan(
    data: &[u8],
    mb_start: usize,
    mb_end: usize,
    commands: &[OptimalCommand],
    trailing_literals: u32,
    npostfix: u8,
    ndirect: u8,
    large_window: bool,
    base_max_lit_trees: usize,
    is_q9: bool,
) -> Option<MetablockPlan> {
    try_plan_metablock(
        data,
        mb_start,
        mb_end,
        commands,
        trailing_literals,
        large_window,
        Some((npostfix, ndirect)),
        base_max_lit_trees,
        if is_q9 { 9 } else { 8 },
        1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_symbol_stream_homogeneous_and_multi() {
        let mut hbuf = vec![HuffmanCodeEntry::default(); MAX_HUFFMAN_ALPHABET];
        let homo: Vec<u16> = (0..4096).map(|i| ((i * 13 + 7) & 15) as u16).collect();
        let spec = SplitSpec::new(256, 16, 256, 64, true);
        assert_eq!(split_symbol_stream(&homo, spec, &mut hbuf).num_types, 1);

        let mut multi = Vec::with_capacity(8192);
        let mut rng = 0xCAFE_BABE_1234_5678u64;
        for base in [0u16, 128, 0, 200] {
            for _ in 0..2048 {
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
        for extreme_d in [0x4000_0026u32, 0x6000_0001, 0x7FFF_FFF1, 0x7FFF_FFFC] {
            let mut samples: Vec<(u8, u32)> = (0..256u32)
                .map(|i| ((i & 1) as u8, 4 + (i % 16) * 4))
                .collect();
            let (np_base, nd_base) =
                select_distance_params_from_samples(&samples, samples.len(), 64, true);
            assert_ne!((np_base, nd_base), (0, 0));

            // Even if `extreme_d` was unsampled (via `max_raw_d`) or sampled, selected `(np, nd)`
            // must keep `encode_raw_distance_params(extreme_d, np, nd).0 < max_dist_symbol`.
            let (np_unsampled, nd_unsampled) =
                select_distance_params_from_samples(&samples, samples.len() + 1, extreme_d, true);
            assert!(
                (encode_raw_distance_params(extreme_d, np_unsampled, nd_unsampled).0 as usize)
                    < max_dist_symbol(np_unsampled, nd_unsampled, true)
            );
            samples.push((1, extreme_d));
            let (np_sampled, nd_sampled) =
                select_distance_params_from_samples(&samples, samples.len(), extreme_d, true);
            assert!(
                (encode_raw_distance_params(extreme_d, np_sampled, nd_sampled).0 as usize)
                    < max_dist_symbol(np_sampled, nd_sampled, true)
            );

            let mut cmds: Vec<OptimalCommand> = (0..128u32)
                .map(|i| OptimalCommand {
                    insert_len: 1,
                    copy_len: 5,
                    out_len: 5,
                    dist: 8 + (i % 16) * 8,
                    dist_mode: crate::metablock::DIST_MODE_RAW,
                })
                .collect();
            cmds.push(OptimalCommand {
                insert_len: 1,
                copy_len: 5,
                out_len: 5,
                dist: extreme_d,
                dist_mode: crate::metablock::DIST_MODE_STATIC_DICT,
            });
            for effort in [10u8, 11, 12] {
                let (np, nd) = select_optimal_distance_params(&cmds, true, effort, &mut hbuf);
                for c in &cmds {
                    let s = encode_raw_distance_params(c.dist, np, nd).0 as usize;
                    assert!(
                        s < max_dist_symbol(np, nd, true),
                        "effort={effort} d={:#x} selected ({np}, {nd}) -> sym={s} >= {}",
                        c.dist,
                        max_dist_symbol(np, nd, true)
                    );
                }
            }
        }
    }

    #[cfg(all(target_os = "linux", target_pointer_width = "64", not(miri)))]
    #[test]
    fn test_metablock_plan_and_emit_across_4gib_boundary() {
        use crate::emit::write_compressed_metablock;
        use crate::metablock::{self, test_helpers::decompress_brotli};
        use lsb_bitwriter::BitWriter;

        const TWO_32: usize = 1usize << 32;
        // Demand-zeroed virtual mapping (`calloc` / `MAP_ANONYMOUS`) faults only the touched 16 KiB
        // around `2^32` while exercising true 64-bit `mb_start..mb_end` straddling and `>= 2^32`.
        let mut data = vec![0u8; TWO_32 + 8192];
        for (i, slot) in data[(TWO_32 - 4096)..(TWO_32 + 8192)]
            .iter_mut()
            .enumerate()
        {
            let idx = (TWO_32 - 4096) + i;
            *slot =
                b"High64BitOffsetBrotliMetablockContextTest! "[idx % 43] ^ ((idx >> 8) as u8 & 7);
        }

        for (mb_start, mb_end) in [(TWO_32 - 2048, TWO_32 + 2048), (TWO_32, TWO_32 + 4096)] {
            let mb_len = (mb_end - mb_start) as u32;
            let plan = build_metablock_plan_with_effort(
                &data,
                mb_start,
                mb_end,
                &[],
                mb_len,
                false,
                12,
                2,
            );
            // Compare against an identical metablock placed at low offset `2..2 + mb_len` with the
            // same 2-byte literal context prefix `(data[mb_start - 2], data[mb_start - 1])`.
            let low_slice = &data[mb_start - 2..mb_end];
            let low_plan = build_metablock_plan_with_effort(
                low_slice,
                2,
                2 + mb_len as usize,
                &[],
                mb_len,
                false,
                12,
                1,
            );
            assert_eq!(plan.exact_compressed_bits, low_plan.exact_compressed_bits);

            let mut out = vec![0u8; 16_384];
            let mut bw = BitWriter::new(&mut out);
            metablock::write_window_bits(24, false, &mut bw);
            // Prepend a 2-byte uncompressed metablock carrying the exact 2-byte context prefix so
            // standalone decompression verifies every literal tree & context ID at `>= 2^32`.
            metablock::write_uncompressed_metablock(&data[mb_start - 2..mb_start], &mut bw);
            write_compressed_metablock(&data, mb_start, mb_end, &[], mb_len, &plan, true, &mut bw);
            let encoded = bw.finalize();
            assert_eq!(decompress_brotli(encoded), &data[mb_start - 2..mb_end]);
        }
    }
}
