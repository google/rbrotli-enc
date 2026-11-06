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

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn test_dict_chain_and_sparse_ldm() {
        let dc = DictChain::new(b"ABCD_ABCD_XYZ").unwrap();
        assert!(dc.head[0] != 0 || dc.prev.len() == 13);
        assert_eq!(
            ChunkEnv::new(b"XYZ_REST", 0, 8, 1008, b"TAIL_XYZ").match_len(0, 3),
            3
        );
        if !cfg!(miri) {
            let gap = 20 << 20;
            let mut head = Vec::new();
            super::super::test_helpers::append_lcg_bytes(&mut head, 6144, 0x517C_C1B7_2722_0A95, 1);
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
}
