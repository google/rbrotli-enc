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

//! Subquadratic $O(N \log N)$ Pareto-frontier match generator for `Q10`..=`Q12`.

#[cfg(target_arch = "x86_64")]
use crate::constants::X64V3;
use crate::constants::{try_avx2, Scalar, SimdBackend, HASH_MUL32, STD_MAX_RAW_STATIC_DIST};
use crate::hashtable::{
    common_prefix_at, common_prefix_len, common_suffix_len, find_closest_2b, FastLdmSpan,
};
use crate::map_to_sym::copy_code_only;
use crate::static_dict;
use hugepage_buffer::HugePageBuf;
#[cfg(target_arch = "x86_64")]
use simd_tokens::{simd_entry, SimdToken};

use super::ChunkEnv;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MatchInterval {
    pub(super) l_lo: u32,
    pub(super) l_hi: u32,
    pub(super) dist: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct StaticMatchCand {
    pub(super) out_len: u8,
    pub(super) base_len: u8,
    pub(super) raw_dist: u32,
}

#[derive(Default)]
pub(super) struct ChunkMatches {
    pub(super) intervals: Vec<MatchInterval>,
    pub(super) pos_offsets: Vec<u32>,
    pub(super) static_cands: Vec<StaticMatchCand>,
    pub(super) static_offsets: Vec<u32>,
}

impl ChunkMatches {
    #[inline(always)]
    fn clear(&mut self, chunk_len: usize) {
        self.intervals.clear();
        self.pos_offsets.clear();
        self.pos_offsets.reserve(chunk_len + 1);
        self.static_cands.clear();
        self.static_offsets.clear();
    }

    #[inline(always)]
    fn finish_empty(&mut self, chunk_len: usize, large_window: bool) {
        self.pos_offsets.resize(chunk_len + 1, 0);
        if !large_window {
            self.static_offsets.resize(chunk_len + 1, 0);
        }
    }

    #[inline(always)]
    fn begin_pos(&mut self, large_window: bool) {
        self.pos_offsets.push(self.intervals.len() as u32);
        if !large_window {
            self.static_offsets.push(self.static_cands.len() as u32);
        }
    }

    #[inline(always)]
    pub(super) fn at(&self, local_pos: usize) -> &[MatchInterval] {
        let s = self.pos_offsets[local_pos] as usize;
        let e = self.pos_offsets[local_pos + 1] as usize;
        &self.intervals[s..e]
    }

    #[inline(always)]
    pub(super) fn static_at(&self, local_pos: usize) -> &[StaticMatchCand] {
        if self.static_offsets.is_empty() {
            return &[];
        }
        let s = self.static_offsets[local_pos] as usize;
        let e = self.static_offsets[local_pos + 1] as usize;
        &self.static_cands[s..e]
    }

    #[inline(always)]
    fn push_pareto(
        &mut self,
        steps: impl IntoIterator<Item = (u32, u32)>,
        min_floor: u32,
        mut prev_hi: u32,
    ) -> u32 {
        for (l, d) in steps {
            let l_lo = (prev_hi + 1).max(min_floor).max(min_len_for_dist(d));
            if l_lo <= l {
                self.intervals.push(MatchInterval {
                    l_lo,
                    l_hi: l,
                    dist: d,
                });
                prev_hi = l;
            }
        }
        prev_hi
    }

    #[inline(always)]
    fn push_static_dedup(&mut self, sc_start: usize, out_len: u32, base_len: u32, raw_dist: u32) {
        let (ol, bl, cc) = (out_len as u8, base_len as u8, copy_code_only(base_len));
        for ex in &mut self.static_cands[sc_start..] {
            if ex.out_len == ol && copy_code_only(ex.base_len as u32) == cc {
                if raw_dist < ex.raw_dist {
                    (ex.base_len, ex.raw_dist) = (bl, raw_dist);
                }
                return;
            }
        }
        self.static_cands.push(StaticMatchCand {
            out_len: ol,
            base_len: bl,
            raw_dist,
        });
    }
}

const LDM_WAYS: usize = 8;
pub(super) type ChunkLdmMatches = Vec<FastLdmSpan>;

#[inline(always)]
fn read_u48(data: &[u8], off: usize) -> u64 {
    let [b0, b1, b2, b3, b4, b5]: [u8; 6] = data[off..off + 6].try_into().unwrap();
    u64::from_le_bytes([b0, b1, b2, b3, b4, b5, 0, 0])
}

#[inline(always)]
fn mix_ldm6(w6: u64) -> u64 {
    let z = (w6 ^ (w6 >> 23)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    z ^ (z >> 29)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn emit_ldm_pareto(
    cldm: &[FastLdmSpan],
    cur: &mut usize,
    active: &mut [(u32, u32); LDM_WAYS],
    n_act: &mut usize,
    lp: u32,
    (min_d_excl, max_d, min_gt): (usize, usize, u32),
    prev_hi: u32,
    out: &mut ChunkMatches,
) -> u32 {
    let min_end = lp.saturating_add(5);
    let retain =
        |act: &mut [(u32, u32); LDM_WAYS], n: &mut usize, keep: &dyn Fn(u32, u32) -> bool| {
            let mut dst = 0;
            for i in 0..*n {
                if keep(act[i].0, act[i].1) {
                    act[dst] = act[i];
                    dst += 1;
                }
            }
            *n = dst;
        };
    if *n_act > 0 {
        retain(active, n_act, &|e, _| e >= min_end);
    }
    while let Some(&sp) = cldm.get(*cur).filter(|s| s.start <= lp) {
        *cur += 1;
        if sp.end < min_end
            || active[..*n_act]
                .iter()
                .any(|&(ek, dk)| dk <= sp.dist && ek >= sp.end)
        {
            continue;
        }
        retain(active, n_act, &|es, ds| !(ds >= sp.dist && es <= sp.end));
        let ins = active[..*n_act].partition_point(|x| x.1 < sp.dist);
        if *n_act < LDM_WAYS {
            active.copy_within(ins..*n_act, ins + 1);
            active[ins] = (sp.end, sp.dist);
            *n_act += 1;
        } else if ins == LDM_WAYS {
            active[LDM_WAYS - 1] = (sp.end, sp.dist);
        } else if ins <= LDM_WAYS - 2 {
            active.copy_within(ins..LDM_WAYS - 2, ins + 1);
            active[ins] = (sp.end, sp.dist);
        }
    }
    let mut best_len = min_gt.max(4);
    out.push_pareto(
        active[..*n_act].iter().filter_map(|&(end_l, dist)| {
            let (d, len) = (dist as usize, end_l.saturating_sub(lp));
            (d > min_d_excl && d <= max_d && len > best_len).then(|| {
                best_len = len;
                (len, dist)
            })
        }),
        5,
        prev_hi,
    )
}

/// THP-backed 8-way SplitMix64 Long-Distance Matcher for `Q10..=Q12` (even buckets: `6 B` anchors;
/// odd buckets: `1-in-4` `12 B` phrase anchors resilient to Zipf eviction up to `1 GiB`).
#[derive(Default)]
pub(super) struct SparseLdm {
    buckets: HugePageBuf<[u32; LDM_WAYS]>,
    bucket_mask: usize,
    sync_mask: u64,
    indexed_end: usize,
    base: usize,
    dirty_to: usize,
}

impl SparseLdm {
    #[inline]
    pub(super) fn reset_stream(&mut self) {
        (self.bucket_mask, self.indexed_end, self.base) = (0, 0, 0);
    }

    fn shift_down(&mut self, mut drop_len: usize) {
        while drop_len > 0 {
            let step = drop_len.min((u32::MAX as usize) - 8);
            let (shift, min_p1) = (step as u32, step as u32 + 6);
            for b in &mut self.buckets[..] {
                for s in b {
                    *s = if *s >= min_p1 { *s - shift } else { 0 };
                }
            }
            self.indexed_end = self.indexed_end.saturating_sub(step);
            drop_len -= step;
        }
    }

    pub(super) fn on_compact(&mut self, drop_len: usize) {
        if self.bucket_mask != 0 && drop_len > 0 {
            let via = drop_len.min(self.base);
            self.base -= via;
            self.shift_down(drop_len - via);
        }
    }

    #[inline(always)]
    fn scan_loop<S: SimdBackend>(
        &mut self,
        simd: S,
        env: ChunkEnv<'_>,
        (scan_start, capped_end, local_window_cap, do_probe): (usize, usize, usize, bool),
        out: &mut ChunkLdmMatches,
    ) {
        let (data, chunk_start, chunk_end, wsize) = (env.data, env.start, env.end, env.wsize);
        let buf_start = chunk_start - chunk_start.min(wsize).min(local_window_cap);
        let (sync_mask, sync12_mask, bmask) = (
            self.sync_mask,
            (self.sync_mask + 1) * 4 - 1,
            self.bucket_mask,
        );
        let (mut prev_w6, mut skip6, mut skip12) = (u64::MAX, 0usize, 0usize);
        let (mut recent, mut r_idx) = ([FastLdmSpan::default(); 8], 0usize);

        for p in scan_start..capped_end {
            let w6 = read_u48(data, p - 5);
            if w6 == prev_w6 {
                continue;
            }
            prev_w6 = w6;
            let h = mix_ldm6(w6);
            if (h & sync_mask) != 0 {
                continue;
            }
            let mut step = |bkt: usize, is_12: bool| {
                let skip = if is_12 { skip12 } else { skip6 };
                if do_probe && p >= chunk_start && p >= skip {
                    let (min_cand, local_p) = (p.saturating_sub(wsize), (p - chunk_start) as u32);
                    let mut best_end = local_p;
                    for cand_p1 in self.buckets[bkt] {
                        if cand_p1 == 0 || (cand_p1 as usize - 1) < min_cand {
                            break;
                        }
                        let cand = cand_p1 as usize - 1;
                        if cand >= p || (cand >= buf_start && p - cand <= local_window_cap) {
                            continue;
                        }
                        let dist = (p - cand) as u32;
                        let eq = if is_12 {
                            cand >= 11 && data[cand - 11..=cand] == data[p - 11..=p]
                        } else {
                            cand >= 5 && read_u48(data, cand - 5) == w6
                        };
                        if !eq
                            || recent
                                .iter()
                                .any(|r| r.dist == dist && local_p >= r.start && local_p < r.end)
                        {
                            continue;
                        }
                        if !is_12 && best_end > local_p {
                            let req_p = chunk_start + (best_end as usize);
                            if req_p >= chunk_end {
                                break;
                            }
                            let cc = cand + (req_p - p);
                            if cc >= data.len() || data[cc] != data[req_p] {
                                continue;
                            }
                        }
                        let (max_back, cap) =
                            ((p - chunk_start).min(cand), 5 + 6 * usize::from(is_12));
                        let kb = cap.min(max_back);
                        let back = kb
                            + common_suffix_len(
                                &data[cand - max_back..cand - kb],
                                &data[p - max_back..p - kb],
                            );
                        let fwd =
                            common_prefix_at(simd, data, cand + 1, p + 1, chunk_end - (p + 1));
                        let (start, end) = (
                            (p - back - chunk_start) as u32,
                            (p + 1 + fwd - chunk_start) as u32,
                        );
                        let min_req = if is_12 {
                            5
                        } else {
                            6 + u32::from(dist > (64 << 20))
                        };
                        if end - start >= min_req {
                            let sp = FastLdmSpan { start, end, dist };
                            out.push(sp);
                            recent[r_idx & 7] = sp;
                            r_idx = r_idx.wrapping_add(1);
                            best_end = best_end.max(end);
                            let end_sp = chunk_start + end as usize;
                            if end >= local_p + 24 {
                                skip6 = skip6.max(end_sp.saturating_sub(4));
                            }
                            if end >= local_p + 32 {
                                skip12 = skip12.max(end_sp.saturating_sub(8));
                            }
                        }
                    }
                }
                if p >= self.indexed_end {
                    let (p1, b) = ((p + 1) as u32, &mut self.buckets[bkt]);
                    if !is_12 && b[0] != 0 && p1 >= b[0] && p1 - b[0] <= 16 {
                        b[0] = p1;
                    } else {
                        *b = [p1, b[0], b[1], b[2], b[3], b[4], b[5], b[6]];
                    }
                }
            };
            step((((h >> 32) as usize) & bmask) & !1, false);
            if p >= 11 && (h & sync12_mask) == 0 {
                let lo8 = u64::from_le_bytes(data[p - 11..p - 3].try_into().unwrap());
                step((((mix_ldm6(lo8 ^ h) >> 32) as usize) & bmask) | 1, true);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    #[inline(never)]
    fn scan_avx2(
        &mut self,
        env: ChunkEnv<'_>,
        bounds: (usize, usize, usize, bool),
        out: &mut ChunkLdmMatches,
        simd: X64V3,
        _tok: impl SimdToken,
    ) {
        self.scan_loop(simd, env, bounds, out);
    }

    pub(super) fn extract(
        &mut self,
        env: ChunkEnv<'_>,
        eff_win: usize,
        local_window_cap: usize,
        effort: u8,
        out: &mut ChunkLdmMatches,
    ) {
        out.clear();
        if eff_win.min(env.wsize).min(env.data.len()) <= local_window_cap || env.end <= 6 {
            return;
        }
        let is_lw = env.wsize > (1 << 24);
        let n_bkt = (eff_win.next_power_of_two() >> if effort < 11 && !is_lw { 3 } else { 2 })
            .clamp(1 << 20, 1 << 22);
        let base_sync = (((eff_win.saturating_sub(1) >> 26) + 1).next_power_of_two() - 1) as u64;
        let sync_mask = if effort < 11 && !is_lw {
            base_sync.max(3)
        } else {
            base_sync
        };
        if self.buckets.len() != n_bkt || self.bucket_mask == 0 || self.sync_mask != sync_mask {
            let reallocated = n_bkt > self.buckets.capacity();
            self.buckets.resize_or_grow_zeroed(n_bkt);
            if reallocated {
                self.dirty_to = 0;
            } else if self.dirty_to > 0 {
                let clr = n_bkt.min(self.dirty_to);
                self.buckets[..clr].fill([0; LDM_WAYS]);
                if n_bkt >= self.dirty_to {
                    self.dirty_to = 0;
                }
            }
            (
                self.bucket_mask,
                self.sync_mask,
                self.indexed_end,
                self.base,
            ) = (n_bkt - 1, sync_mask, 0, 0);
        }
        if env.end.saturating_sub(self.base) >= (u32::MAX as usize) - 1 {
            let delta = env
                .start
                .saturating_sub(env.wsize)
                .saturating_sub(self.base);
            if delta > 0 {
                self.shift_down(delta);
                self.base += delta;
            }
        }
        let env = ChunkEnv::new(
            &env.data[self.base..],
            env.start - self.base,
            env.end - self.base,
            env.wsize,
            env.dict,
        );
        let capped_end = env.end.min(env.data.len()).min((u32::MAX as usize) - 1);
        let buf_start = env.start - env.start.min(env.wsize).min(local_window_cap);
        let do_probe = !(buf_start == 0 && capped_end <= local_window_cap);
        let mut scan_start = self.indexed_end.max(env.start.saturating_sub(env.wsize));
        if do_probe {
            scan_start = scan_start.min(env.start);
        }
        scan_start = scan_start.max(5);
        if scan_start >= capped_end {
            return;
        }
        self.dirty_to = self.dirty_to.max(n_bkt);
        let bounds = (scan_start, capped_end, local_window_cap, do_probe);
        match try_avx2(true) {
            #[cfg(target_arch = "x86_64")]
            Some(tok) => self.scan_avx2(env, bounds, out, tok, tok),
            _ => self.scan_loop(Scalar, env, bounds, out),
        }
        self.indexed_end = self.indexed_end.max(capped_end);
        if out.len() > 1 {
            out.sort_unstable_by_key(|s| (s.start, s.dist));
            out.dedup();
        }
    }
}

/// Exact 4-byte hash chain over an attached custom dictionary.
pub(crate) struct DictChain {
    head: Vec<u32>,
    prev: Vec<u32>,
}

impl DictChain {
    pub(crate) fn new(dict_bytes: &[u8]) -> Option<Self> {
        let n = dict_bytes.len();
        if n < 4 {
            return None;
        }
        let (mut head, mut prev) = (vec![0u32; 1 << 16], vec![0u32; n]);
        for off in 0..=(n - 4) {
            let w4 = u32::from_le_bytes(dict_bytes[off..off + 4].try_into().unwrap());
            let h = (w4.wrapping_mul(HASH_MUL32) >> 16) as usize;
            prev[off] = std::mem::replace(&mut head[h], (off + 1) as u32);
        }
        Some(Self { head, prev })
    }

    #[inline]
    fn find_pareto_matches_asc(
        &self,
        env: ChunkEnv<'_>,
        stream_pos: usize,
        min_strictly_gt: usize,
        out_asc: &mut [(u32, u32)],
    ) -> usize {
        let target = &env.data[stream_pos..env.end];
        if target.len() < 4 || stream_pos >= env.wsize || target.len() <= min_strictly_gt {
            return 0;
        }
        let w4 = u32::from_le_bytes(target[..4].try_into().unwrap());
        let h = (w4.wrapping_mul(HASH_MUL32) >> 16) as usize;
        let (full_len, act_len) = (self.prev.len(), env.dict.len());
        let (mut cur, mut best_len, mut n_asc, mut probes) =
            (self.head[h], min_strictly_gt.max(3), 0usize, 0usize);

        while cur != 0 && probes < 64 && n_asc < out_asc.len() {
            let off = (cur - 1) as usize;
            cur = self.prev[off];
            let (dict_rem, dist) = (full_len - off, stream_pos + full_len - off);
            if dist > env.wsize || dict_rem > act_len {
                break;
            }
            probes += 1;
            let (act_off, max_l) = (act_len - dict_rem, dict_rem.min(target.len()));
            if max_l <= best_len || env.dict[act_off + best_len] != target[best_len] {
                continue;
            }
            let mlen = common_prefix_len(&env.dict[act_off..], &target[..max_l], max_l);
            if mlen > best_len {
                best_len = mlen;
                out_asc[n_asc] = (mlen as u32, dist as u32);
                n_asc += 1;
                if best_len == target.len() {
                    break;
                }
            }
        }
        n_asc
    }
}

#[inline(always)]
fn min_len_for_dist(dist: u32) -> u32 {
    2 + u32::from(dist > 64) + u32::from(dist > 16_384)
}

#[derive(Default)]
pub(super) struct MatchGenScratch {
    buckets: Vec<u32>,
    bucket_shift: u32,
    forest: Vec<u32>,
    ring_mask: usize,
    comp_cap: usize,
    ht3: Vec<[u32; 4]>,
    ht3_shift: u32,
    inc: (usize, usize, usize, usize),
    prev_rep_dist: usize,
    prev_rep_end: usize,
}

impl MatchGenScratch {
    pub(super) fn reset_h10(&mut self) {
        (self.comp_cap, self.prev_rep_end) = (0, 0);
    }

    fn prepare_h10(
        &mut self,
        data: &[u8],
        (buf_start, chunk_start, ring_cap, comp_cap): (usize, usize, usize, usize),
        (bucket_bits, ht3_bits): (usize, usize),
    ) -> usize {
        assert!(ring_cap.is_power_of_two() && ring_cap >= 64 && comp_cap >= 4);
        let (n_buckets, n_ht3, cur_id) = (
            1usize << bucket_bits,
            1usize << ht3_bits,
            data.as_ptr() as usize,
        );
        if self.buckets.len() == n_buckets
            && self.ht3.len() == n_ht3
            && self.ring_mask + 1 == ring_cap
            && self.comp_cap == comp_cap
            && (self.inc.0, self.inc.1, self.inc.3) == (cur_id, data.len(), chunk_start)
            && self.inc.2 <= buf_start
        {
            return chunk_start;
        }
        self.buckets.clear();
        self.buckets.resize(n_buckets, 0);
        self.bucket_shift = (32 - bucket_bits) as u32;
        if self.forest.len() < 2 * ring_cap {
            self.forest.resize(2 * ring_cap, 0);
        }
        self.ht3.clear();
        self.ht3.resize(n_ht3, [0; 4]);
        self.ht3_shift = (32 - ht3_bits) as u32;
        (self.ring_mask, self.comp_cap) = (ring_cap - 1, comp_cap);
        self.inc = (cur_id, data.len(), buf_start, buf_start);
        (self.prev_rep_dist, self.prev_rep_end) = (0, 0);
        buf_start
    }

    #[inline(always)]
    fn insert_and_find_3b(&mut self, data: &[u8], pos: usize, max_d3: usize) -> u32 {
        let [b0, b1, b2, ..] = data[pos..] else {
            return u32::MAX;
        };
        let h3 = (u32::from_le_bytes([b0, b1, b2, 0]).wrapping_mul(HASH_MUL32) >> self.ht3_shift)
            as usize;
        let (row, new_p1) = (&mut self.ht3[h3], (pos + 1) as u32);
        for i in 0..4 {
            let p1 = row[i];
            let d = pos.wrapping_sub((p1 as usize).wrapping_sub(1));
            if p1 == 0 || d > max_d3 {
                break;
            }
            if data[pos - d..pos - d + 3] == [b0, b1, b2] {
                row.copy_within(..i, 1);
                row[0] = new_p1;
                return d as u32;
            }
        }
        *row = [new_p1, row[0], row[1], row[2]];
        u32::MAX
    }

    #[inline(always)]
    fn store_and_find_bst<S: SimdBackend>(
        &mut self,
        simd: S,
        data: &[u8],
        pos: usize,
        (emit_limit, max_dist, comp_cap, max_depth): (usize, usize, usize, usize),
        emit: bool,
        out_asc: &mut [(u32, u32)],
    ) -> usize {
        debug_assert_eq!(comp_cap, self.comp_cap);
        let data_len = data.len();
        if data_len - pos < 4 {
            self.prev_rep_end = 0;
            return 0;
        }
        let w4_be = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap());
        let h4 = (w4_be.swap_bytes().wrapping_mul(HASH_MUL32) >> self.bucket_shift) as usize;
        let prev_root = self.buckets[h4];
        self.buckets[h4] = (pos + 1) as u32;

        let my_node = (pos & self.ring_mask) << 1;
        if prev_root == 0 || max_dist == 0 {
            self.forest[my_node] = 0;
            self.forest[my_node + 1] = 0;
            self.prev_rep_end = 0;
            return 0;
        }
        let max_emit_len = emit_limit.saturating_sub(pos);

        if pos + comp_cap < self.prev_rep_end {
            let rep_d = self.prev_rep_dist;
            if rep_d > 0 && rep_d <= max_dist && (prev_root as usize) + rep_d == pos + 1 {
                let lim_end = if rep_d == 1 {
                    data_len
                } else {
                    emit_limit.min(data_len)
                };
                if self.prev_rep_end < lim_end
                    && data[self.prev_rep_end] == data[self.prev_rep_end - rep_d]
                {
                    self.prev_rep_end += 1;
                }
                let cn = ((pos - rep_d) & self.ring_mask) << 1;
                (self.forest[my_node], self.forest[my_node + 1]) =
                    (self.forest[cn], self.forest[cn + 1]);
                let elen = (self.prev_rep_end - pos).min(max_emit_len);
                if emit && elen >= 4 && !out_asc.is_empty() {
                    out_asc[0] = (elen as u32, rep_d as u32);
                    return 1;
                }
                return 0;
            }
        }

        let max_stream_len = data_len - pos;
        let max_key_len = max_stream_len.min(comp_cap);
        let (mut left_r, mut right_r) = (my_node, my_node + 1);
        let (mut min_l, mut min_r, mut best_len, mut n_out) = (0usize, 0usize, 3usize, 0usize);
        macro_rules! push_bst {
            ($elen:expr, $dist:expr) => {{
                out_asc[n_out.min(out_asc.len() - 1)] = ($elen as u32, $dist as u32);
                n_out = (n_out + 1).min(out_asc.len());
            }};
        }

        let (mut cand_p1, mut depth) = (prev_root, 0usize);
        while cand_p1 != 0 {
            let cand = (cand_p1 - 1) as usize;
            let dist = pos - cand;
            if dist > max_dist || depth >= max_depth {
                break;
            }
            let (cand_node, skip) = ((cand & self.ring_mask) << 1, min_l.min(min_r));
            let cw4 = if skip < 4 {
                u32::from_be_bytes(data[cand..cand + 4].try_into().unwrap())
            } else {
                w4_be
            };
            let (mlen_key, go_right) = if cw4 != w4_be {
                (((cw4 ^ w4_be).leading_zeros() >> 3) as usize, cw4 > w4_be)
            } else {
                let s4 = skip.max(4);
                let mk = s4
                    + crate::hashtable::common_prefix_at(
                        simd,
                        data,
                        cand + s4,
                        pos + s4,
                        max_key_len.saturating_sub(s4),
                    );
                if mk == comp_cap {
                    (self.forest[left_r], self.forest[right_r]) =
                        (self.forest[cand_node], self.forest[cand_node + 1]);
                    if emit || dist == 1 {
                        let check_lim = if dist == 1 {
                            max_stream_len
                        } else {
                            max_emit_len.max(comp_cap)
                        };
                        let known_len = if dist == self.prev_rep_dist {
                            self.prev_rep_end.clamp(pos + comp_cap, pos + check_lim) - pos
                        } else {
                            comp_cap
                        };
                        let full_len = known_len
                            + crate::hashtable::common_prefix_at(
                                simd,
                                data,
                                cand + known_len,
                                pos + known_len,
                                check_lim.saturating_sub(known_len),
                            );
                        if pos + full_len > self.prev_rep_end {
                            (self.prev_rep_dist, self.prev_rep_end) = (dist, pos + full_len);
                        }
                        let elen = full_len.min(max_emit_len);
                        if emit && elen > best_len {
                            push_bst!(elen, dist);
                        }
                    }
                    return n_out;
                }
                let elen = mk.min(max_emit_len);
                if emit && elen > best_len {
                    best_len = elen;
                    push_bst!(elen, dist);
                }
                (mk, mk == max_key_len || data[cand + mk] > data[pos + mk])
            };
            if go_right {
                (self.forest[right_r], right_r, cand_p1, min_r) =
                    (cand_p1, cand_node, self.forest[cand_node], mlen_key);
            } else {
                (self.forest[left_r], left_r, cand_p1, min_l) =
                    (cand_p1, cand_node + 1, self.forest[cand_node + 1], mlen_key);
            }
            depth += 1;
        }
        (self.forest[left_r], self.forest[right_r]) = (0, 0);
        n_out
    }

    #[inline(always)]
    fn index_only_at<S: SimdBackend>(
        &mut self,
        simd: S,
        data: &[u8],
        pos: usize,
        p4: (usize, usize, usize, usize),
        insert_3b: bool,
    ) {
        if insert_3b {
            let _ = self.insert_and_find_3b(data, pos, 0);
        }
        let _ = self.store_and_find_bst(simd, data, pos, p4, false, &mut []);
    }
}

pub(super) struct MatchGenCfg<'a> {
    pub(super) env: ChunkEnv<'a>,
    pub(super) local_window_cap: usize,
    pub(super) large_window: bool,
    pub(super) effort: u8,
    pub(super) full_dict_len: usize,
    pub(super) dict_chain: Option<&'a DictChain>,
    pub(super) ldm: &'a ChunkLdmMatches,
}

#[inline(always)]
fn build_chunk_matches_impl<S: SimdBackend>(
    simd: S,
    cfg: MatchGenCfg<'_>,
    scratch: &mut MatchGenScratch,
    out: &mut ChunkMatches,
) {
    let env = cfg.env;
    let (data, chunk_start, chunk_end, wsize) = (env.data, env.start, env.end, env.wsize);
    assert!(chunk_start <= chunk_end && chunk_end <= data.len());
    let chunk_len = env.len();
    out.clear(chunk_len);
    if chunk_len < 2 {
        out.finish_empty(chunk_len, cfg.large_window);
        return;
    }

    let has_ldm = !cfg.ldm.is_empty();
    let (is_q11, is_q12) = (cfg.effort >= 11, cfg.effort >= 12);
    let eff_prefix = chunk_start.min(wsize).min(cfg.local_window_cap);
    let buf_start = chunk_start - eff_prefix;
    let ring_cap = (eff_prefix + chunk_len)
        .max(64)
        .next_power_of_two()
        .min((cfg.local_window_cap + 1).next_power_of_two())
        .min(4 << 20);
    let max_bst_window = if is_q12 {
        wsize.min(ring_cap - 1)
    } else {
        wsize.min(cfg.local_window_cap).min(ring_cap - 1)
    };
    let (raw_bb, raw_hb, comp_cap, max_depth, seed_depth, max_bst_steps) = if is_q12 {
        (20usize, 17usize, 2048usize, 2048usize, 1024usize, 48usize)
    } else if is_q11 {
        (19, 17, 512, 128, 64, 16)
    } else if cfg.large_window {
        (17, 14, 192, 40, 18, 16)
    } else {
        (17, 14, 160, 32, 12, 16)
    };
    let cap_lg = ring_cap.trailing_zeros() as usize;
    let bucket_bits = raw_bb.min((cap_lg + 1).max(11));
    let ht3_bits = raw_hb.min(if is_q11 {
        cap_lg.saturating_sub(2).clamp(10, 14)
    } else {
        cap_lg.saturating_sub(4).clamp(10, 13)
    });
    let enable_long_skip = !is_q11;
    let dict_ldm_len_cap = if is_q11 { 4096usize } else { 128usize };
    let max_lz_for_static = [24, 28, 37][usize::from(is_q11) + usize::from(is_q12)];

    let h10_base = if chunk_end >= u32::MAX as usize {
        buf_start & !(ring_cap.max(16_384) - 1)
    } else {
        0
    };
    let h10_data = &data[h10_base
        ..data
            .len()
            .min(h10_base.saturating_add((u32::MAX as usize) - 1))];
    let (h10_buf_start, h10_chunk_start, h10_chunk_end) = (
        buf_start - h10_base,
        chunk_start - h10_base,
        chunk_end - h10_base,
    );
    let seed_from = scratch.prepare_h10(
        h10_data,
        (h10_buf_start, h10_chunk_start, ring_cap, comp_cap),
        (bucket_bits, ht3_bits),
    );
    if seed_from < h10_chunk_start {
        let ht3_seed_from = h10_chunk_start.saturating_sub(16_384).max(seed_from);
        let deep_seed_from = if is_q11 {
            seed_from
        } else {
            h10_chunk_start.saturating_sub(262_144).max(seed_from)
        };
        let mid_seed_from = h10_chunk_start.saturating_sub(524_288).max(seed_from);
        for p in seed_from..h10_chunk_start {
            let sd = if p >= deep_seed_from {
                seed_depth
            } else if cfg.large_window || p >= mid_seed_from {
                8
            } else {
                4
            };
            scratch.index_only_at(
                simd,
                h10_data,
                p,
                (
                    h10_chunk_end,
                    (p - h10_buf_start).min(max_bst_window),
                    comp_cap,
                    sd,
                ),
                p >= ht3_seed_from,
            );
        }
        scratch.inc.3 = h10_chunk_start;
    }

    out.intervals.reserve(chunk_len * 2);
    if !cfg.large_window {
        out.static_offsets.reserve(chunk_len + 1);
        out.static_cands.reserve(chunk_len / 2);
    }

    let (mut asc_bst, mut dict_steps) = ([(0u32, 0u32); 64], [(0u32, 0u32); 16]);
    let (mut ldm_cur, mut n_ldm, mut act_ldm) = (0usize, 0usize, [(0u32, 0u32); LDM_WAYS]);
    let (mut long_match_start, mut long_match_end, mut long_match_dist) = (0usize, 0usize, 0u32);

    for local_pos in 0..chunk_len {
        out.begin_pos(cfg.large_window);
        let stream_pos = chunk_start + local_pos;
        let h10_pos = h10_chunk_start + local_pos;
        let max_local_d = (stream_pos - buf_start).min(max_bst_window);

        let skip_q10_interior =
            enable_long_skip && local_pos >= long_match_start + 3 && local_pos + 4 < long_match_end;
        if skip_q10_interior || chunk_len - local_pos < 2 {
            if skip_q10_interior {
                out.push_pareto(
                    [((long_match_end - local_pos) as u32, long_match_dist)],
                    0,
                    1,
                );
            }
            let sd = if skip_q10_interior { 4 } else { 8 };
            scratch.index_only_at(
                simd,
                h10_data,
                h10_pos,
                (h10_chunk_end, max_local_d, comp_cap, sd),
                !skip_q10_interior,
            );
            continue;
        }

        let n_bst = scratch.store_and_find_bst(
            simd,
            h10_data,
            h10_pos,
            (h10_chunk_end, max_local_d, comp_cap, max_depth),
            true,
            &mut asc_bst,
        );
        let min_d4 = if n_bst > 0 {
            asc_bst[0].1 as usize
        } else {
            usize::MAX
        };
        let emit_d3 = if chunk_len - local_pos >= 3 {
            max_local_d.min(16_384).min(min_d4.saturating_sub(1))
        } else {
            0
        };
        let d3 = scratch.insert_and_find_3b(h10_data, h10_pos, emit_d3);
        let max_d2 = max_local_d
            .min(64)
            .min(min_d4.min(d3 as usize).saturating_sub(1));
        let d2 = find_closest_2b(simd, h10_data, h10_pos, max_d2);

        let bst_slice = if !is_q11 && n_bst > 4 {
            let mut dst = n_bst - 1;
            for i in (1..dst).rev() {
                if asc_bst[dst].1 > asc_bst[i].1 * 5 / 4 && n_bst - dst < 15 {
                    dst -= 1;
                    asc_bst[dst] = asc_bst[i];
                }
            }
            asc_bst[dst - 1] = asc_bst[0];
            &asc_bst[dst - 1..n_bst]
        } else {
            if n_bst > max_bst_steps {
                asc_bst.copy_within(n_bst - max_bst_steps / 2..n_bst, max_bst_steps / 2);
            }
            &asc_bst[..n_bst.min(max_bst_steps)]
        };

        let mut longest_in_stream = if let Some(&(l, _)) = bst_slice.last() {
            l as usize
        } else if d3 != u32::MAX {
            3
        } else {
            usize::from(d2 != u32::MAX) * 2
        };

        let mut prev_hi = out.push_pareto((d2 != u32::MAX).then_some((2, d2)), 0, 1);
        prev_hi = out.push_pareto(
            (d3 != u32::MAX).then_some((3, d3)),
            if is_q11 { 2 } else { 3 },
            prev_hi,
        );
        prev_hi = out.push_pareto(bst_slice.iter().copied(), 0, prev_hi);

        if longest_in_stream < dict_ldm_len_cap && has_ldm && stream_pos > max_local_d {
            prev_hi = emit_ldm_pareto(
                cfg.ldm,
                &mut ldm_cur,
                &mut act_ldm,
                &mut n_ldm,
                local_pos as u32,
                (max_local_d, stream_pos.min(wsize), longest_in_stream as u32),
                prev_hi,
                out,
            );
            longest_in_stream = longest_in_stream.max(prev_hi as usize);
        }
        let n_dict = cfg
            .dict_chain
            .filter(|_| longest_in_stream < dict_ldm_len_cap && stream_pos < wsize)
            .map_or(0, |dc| {
                dc.find_pareto_matches_asc(env, stream_pos, longest_in_stream, &mut dict_steps)
            });
        prev_hi = out.push_pareto(dict_steps[..n_dict].iter().copied(), 4, prev_hi);

        if enable_long_skip && prev_hi > 1 {
            let top_iv = out.intervals[out.intervals.len() - 1];
            if top_iv.l_hi >= 128
                && (top_iv.dist as usize) <= stream_pos
                && (local_pos + 4 >= long_match_end
                    || local_pos + (top_iv.l_hi as usize) > long_match_end + 8)
            {
                (long_match_start, long_match_end) = (local_pos, local_pos + top_iv.l_hi as usize);
                long_match_dist = top_iv.dist;
            }
        }

        let longest_lz = prev_hi as usize;
        if !cfg.large_window
            && chunk_end - stream_pos >= 4
            && longest_lz < max_lz_for_static
            && (is_q11 || stream_pos < 524_288 || (stream_pos < 1_048_576 && longest_lz < 6))
        {
            let base_dist = (stream_pos.min(wsize) as u64) + (cfg.full_dict_len as u64) + 1;
            if base_dist <= STD_MAX_RAW_STATIC_DIST as u64 {
                let min_dict_out = longest_lz.saturating_sub(1).max(4);
                let slice = &data[stream_pos..chunk_end];
                if is_q11 {
                    let sc_start = out.static_cands.len();
                    static_dict::for_each_dict121_match(
                        slice,
                        base_dist,
                        STD_MAX_RAW_STATIC_DIST,
                        min_dict_out,
                        |ol, bl, rd| out.push_static_dedup(sc_start, ol, bl, rd),
                    );
                } else if let Some(sm) = static_dict::lookup_simd(
                    simd,
                    false,
                    slice,
                    (
                        base_dist,
                        STD_MAX_RAW_STATIC_DIST,
                        min_dict_out as u32,
                        180,
                        320,
                        16,
                    ),
                ) {
                    out.static_cands.push(StaticMatchCand {
                        out_len: sm.out_len,
                        base_len: sm.base_len,
                        raw_dist: sm.raw_dist,
                    });
                }
            }
        }
    }

    scratch.inc.3 = h10_chunk_end;
    out.begin_pos(cfg.large_window);
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
#[inline(never)]
fn build_chunk_matches_avx2(
    cfg: MatchGenCfg<'_>,
    scratch: &mut MatchGenScratch,
    out: &mut ChunkMatches,
    simd: X64V3,
    _tok: impl SimdToken,
) {
    build_chunk_matches_impl(simd, cfg, scratch, out);
}

#[cfg_attr(target_arch = "x86_64", cold)]
#[inline(never)]
fn build_chunk_matches_scalar(
    cfg: MatchGenCfg<'_>,
    scratch: &mut MatchGenScratch,
    out: &mut ChunkMatches,
) {
    build_chunk_matches_impl(Scalar, cfg, scratch, out);
}

pub(super) fn build_chunk_matches(
    cfg: MatchGenCfg<'_>,
    scratch: &mut MatchGenScratch,
    out: &mut ChunkMatches,
) {
    match try_avx2(true) {
        #[cfg(target_arch = "x86_64")]
        Some(tok) => build_chunk_matches_avx2(cfg, scratch, out, tok, tok),
        _ => build_chunk_matches_scalar(cfg, scratch, out),
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::test_helpers::{append_lcg_bytes, miri_or};
    use super::*;

    pub(crate) fn build_test_matches(env: ChunkEnv<'_>, effort: u8) -> ChunkMatches {
        let dc = DictChain::new(env.dict);
        let mut scratch = MatchGenScratch::default();
        let cldm = ChunkLdmMatches::default();
        let mut cm = ChunkMatches::default();
        build_chunk_matches(
            MatchGenCfg {
                env,
                local_window_cap: env.wsize,
                large_window: false,
                effort,
                full_dict_len: env.dict.len(),
                dict_chain: dc.as_ref(),
                ldm: &cldm,
            },
            &mut scratch,
            &mut cm,
        );
        cm
    }

    fn assert_pareto_frontier(ivs: &[MatchInterval]) {
        for w in ivs.windows(2) {
            assert!(w[0].l_lo <= w[0].l_hi && w[0].l_hi < w[1].l_lo && w[0].dist < w[1].dist);
        }
    }

    #[test]
    fn test_pareto_frontier_oracle_2b_parity_and_rle_grafting() {
        let mut data = Vec::new();
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        for idx in 0..miri_or(5usize, 160) {
            match idx % 5 {
                0 => data.extend_from_slice(b"the quick brown fox "),
                1 => data.extend_from_slice(b"the quick brown dog "),
                2 => data.extend_from_slice(b"the quiet green fox "),
                3 => data.extend_from_slice(&[b'A'; 17]),
                _ => {
                    for _ in 0..9 {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        data.push(b'a' + (((rng >> 58) as u8) & 7));
                    }
                }
            }
        }
        let chunk_start = miri_or(32usize, 300);

        for pos in 0..(data.len() - 1) {
            for max_d2 in [0usize, 1, 31, 32, 33, 64] {
                let exp = (1..=max_d2.min(pos))
                    .find(|&d| data[pos - d..pos - d + 2] == data[pos..pos + 2])
                    .map_or(u32::MAX, |d| d as u32);
                assert_eq!(find_closest_2b(Scalar, &data, pos, max_d2), exp);
                #[cfg(target_arch = "x86_64")]
                if let Some(tok) = X64V3::try_new() {
                    assert_eq!(find_closest_2b(tok, &data, pos, max_d2), exp);
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(tok) = X64V3::try_new() {
            let (mut s_a, mut s_p) = (MatchGenScratch::default(), MatchGenScratch::default());
            for &eff in miri_or(&[10u8, 12][..], &[10u8, 11, 12][..]) {
                let env = ChunkEnv::new(&data, chunk_start, data.len(), 1024, &[]);
                let ldm = ChunkLdmMatches::default();
                let (mut cm_a, mut cm_p) = (ChunkMatches::default(), ChunkMatches::default());
                build_chunk_matches_avx2(
                    MatchGenCfg {
                        env,
                        local_window_cap: 1024,
                        large_window: false,
                        effort: eff,
                        full_dict_len: 0,
                        dict_chain: None,
                        ldm: &ldm,
                    },
                    &mut s_a,
                    &mut cm_a,
                    tok,
                    tok,
                );
                build_chunk_matches_scalar(
                    MatchGenCfg {
                        env,
                        local_window_cap: 1024,
                        large_window: false,
                        effort: eff,
                        full_dict_len: 0,
                        dict_chain: None,
                        ldm: &ldm,
                    },
                    &mut s_p,
                    &mut cm_p,
                );
                assert_eq!(cm_a.intervals, cm_p.intervals);
                assert_eq!(cm_a.pos_offsets, cm_p.pos_offsets);
                assert_eq!(cm_a.static_cands, cm_p.static_cands);
                assert_eq!(cm_a.static_offsets, cm_p.static_offsets);
            }
        }

        let (wsize, oracle_end) = (1024usize, data.len());
        let cm = build_test_matches(
            ChunkEnv::new(&data, chunk_start, oracle_end, wsize, &[]),
            12,
        );
        for local_pos in 0..(oracle_end - chunk_start) {
            let sp = chunk_start + local_pos;
            let ivs = cm.at(local_pos);
            assert_pareto_frontier(ivs);
            let mut exact = vec![u32::MAX; oracle_end - sp + 1];
            for cand in sp.saturating_sub(wsize)..sp {
                let d = (sp - cand) as u32;
                let mlen = common_prefix_len(
                    &data[cand..cand + (oracle_end - sp)],
                    &data[sp..oracle_end],
                    usize::MAX,
                );
                for slot in exact.iter_mut().take(mlen + 1).skip(2) {
                    *slot = (*slot).min(d);
                }
            }
            if ivs.len() < 16 {
                for iv in ivs {
                    for l in iv.l_lo..=iv.l_hi {
                        assert_eq!(iv.dist, exact[l as usize]);
                    }
                }
            }
        }

        // Exercise > comp_cap (512) RLE run + > comp_cap subtree grafting in O(N).
        let (rle_len, phrase_len) = miri_or((532usize, 520usize), (2048, 600));
        let rle_pos = data.len();
        data.extend(std::iter::repeat_n(b'Z', rle_len));
        let mut phrase = Vec::new();
        append_lcg_bytes(&mut phrase, phrase_len, 0xF00D_CAFE_1234_5678, 1);
        data.extend_from_slice(&phrase);
        data.extend_from_slice(b"__gap1__");
        data.extend_from_slice(&phrase);
        data.extend_from_slice(b"__gap2__");
        let p3_pos = data.len();
        data.extend_from_slice(&phrase);
        let rle_env = ChunkEnv::new(&data, 0, data.len(), 8192, &[]);
        let (mut rle_sc, mut cm_rle, rle_ldm) = (
            MatchGenScratch::default(),
            ChunkMatches::default(),
            ChunkLdmMatches::default(),
        );
        build_chunk_matches(
            MatchGenCfg {
                env: rle_env,
                local_window_cap: 8192,
                large_window: cfg!(miri),
                effort: 11,
                full_dict_len: 0,
                dict_chain: None,
                ldm: &rle_ldm,
            },
            &mut rle_sc,
            &mut cm_rle,
        );
        for lp in 0..data.len() {
            assert_pareto_frontier(cm_rle.at(lp));
        }
        let min_rle = (rle_len - 8) as u32;
        assert!(cm_rle
            .at(rle_pos + 1)
            .last()
            .is_some_and(|iv| iv.dist == 1 && iv.l_hi >= min_rle));
        assert!(cm_rle
            .at(p3_pos)
            .last()
            .is_some_and(|iv| iv.l_hi == phrase_len as u32));
    }

    #[test]
    fn test_dict_chain_and_sparse_ldm() {
        let dict = b"PREFIX_DICT_ANCHOR_STRING_WITH_TAIL_XYZ";
        let small = b"XYZ_NOT_CROSSING_AND_PREFIX_DICT_ANCHOR_STRING_HERE";
        let env = ChunkEnv::new(small, 0, small.len(), 1008, dict);
        let cm = build_test_matches(env, 11);
        assert!(cm.at(21).iter().any(|iv| iv.l_hi >= 25 && iv.dist > 21));
        assert_eq!(env.match_len(0, 3), 3);

        if !cfg!(miri) {
            let gap = 20 << 20;
            let mut head = Vec::new();
            append_lcg_bytes(&mut head, 6144, 0x517C_C1B7_2722_0A95, 1);
            let mut data = vec![0u8; gap + 4096];
            data[..2048].copy_from_slice(&head[..2048]);
            data[gap..].copy_from_slice(&head[2048..]);
            data.copy_within(256..256 + 1024, gap + 512);
            let (mut ldm, mut cldm) = (SparseLdm::default(), ChunkLdmMatches::default());
            ldm.extract(
                ChunkEnv::new(&data, 0, 2048, 1 << 26, &[]),
                1 << 26,
                1 << 20,
                11,
                &mut cldm,
            );
            ldm.extract(
                ChunkEnv::new(&data, gap, data.len(), 1 << 26, &[]),
                1 << 26,
                1 << 20,
                11,
                &mut cldm,
            );
            let exp_d = (gap + 512 - 256) as u32;
            assert!(cldm
                .iter()
                .any(|sp| sp.dist == exp_d && sp.start <= 512 && sp.end >= 1536));
        }
    }

    #[test]
    fn test_q10_long_skip_preserves_bst_comp_cap_invariant() {
        let mut data = Vec::new();
        append_lcg_bytes(&mut data, miri_or(1024, 4096), 0x1234_5678_9ABC_DEF0, 1);

        // Node C (pos 100) and Node B (pos 300) agree for 48 bytes (`0x04` at byte 32),
        // then diverge so C sits in B's subtree.
        let pat_b = data[300..364].to_vec();
        data[100..148].copy_from_slice(&pat_b[..48]);
        data[132] = 4;
        data[332] = 4;
        data[148] = 10;
        data[348] = 200;

        // Trigger `skip_q10_interior` (>= 128B match) at pos 600..760, and inside the
        // skipped interior (pos 620) place 32 bytes matching B[..32] followed by `0x01`
        // at byte 32 (`1 != 4`), which previously caused a premature `mlen_key == 32`
        // BST splice when `index_only_at` passed `32` instead of `comp_cap`.
        let long_blk = data[400..560].to_vec();
        data[600..760].copy_from_slice(&long_blk);
        data[420..452].copy_from_slice(&pat_b[..32]);
        data[620..652].copy_from_slice(&pat_b[..32]);
        data[452] = 1;
        data[652] = 1;

        // Query Q (pos 900): matches 40 bytes of `pat_b` (`0x04` at byte 32) and 60 bytes of C.
        data.copy_within(100..160, 900);

        let lgwin = miri_or(12u32, 22);
        let env = ChunkEnv::new(&data, 0, data.len(), (1usize << lgwin) - 16, &[]);
        let ldm = ChunkLdmMatches::default();
        let mut scratch = MatchGenScratch::default();
        let mut cm = ChunkMatches::default();
        build_chunk_matches(
            MatchGenCfg {
                env,
                local_window_cap: (1 << 20) - 1,
                large_window: false,
                effort: 10,
                full_dict_len: 0,
                dict_chain: None,
                ldm: &ldm,
            },
            &mut scratch,
            &mut cm,
        );
        for pos in 0..data.len() {
            for iv in cm.at(pos) {
                let d = iv.dist as usize;
                let hi = iv.l_hi as usize;
                assert_eq!(
                    &data[pos..pos + hi],
                    &data[pos - d..pos - d + hi],
                    "invalid match emitted at pos={pos} dist={d} len={hi}"
                );
            }
        }
        if !cfg!(miri) {
            let mut enc = crate::Encoder::with_lgwin(10, lgwin);
            let comp = enc.compress(&data, None).unwrap();
            assert_eq!(
                crate::metablock::test_helpers::decompress_brotli(comp),
                data
            );
        }
    }
}
