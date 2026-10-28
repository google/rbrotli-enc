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

use crate::constants::{try_avx2, OptAvx2, HASH_MUL32, STD_MAX_RAW_STATIC_DIST};
use crate::map_to_sym::copy_code_only;
use crate::static_dict;
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::_mm256_loadu_si256;
#[cfg(target_arch = "x86_64")]
use simd_tokens::{simd_entry, SimdToken};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
    fn begin_pos(&mut self, large_window: bool) -> usize {
        let iv_start = self.intervals.len();
        self.pos_offsets.push(iv_start as u32);
        if !large_window {
            self.static_offsets.push(self.static_cands.len() as u32);
        }
        iv_start
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
const HT3_RING_BITS: usize = 14;
const HT3_RING_SIZE: usize = 1 << HT3_RING_BITS;
const HT3_RING_MASK: usize = HT3_RING_SIZE - 1;
const MAX_HT3_HOPS: usize = 16;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct LdmSpan {
    pub(super) start_local: u32,
    pub(super) end_local: u32,
    pub(super) dist: u32,
}

pub(super) type ChunkLdmMatches = Vec<LdmSpan>;

#[derive(Default)]
struct LdmActiveTracker {
    cursor: usize,
    n_act: usize,
    active: [(u32, u32); LDM_WAYS],
}

impl LdmActiveTracker {
    #[inline(always)]
    fn retain_active(&mut self, mut keep: impl FnMut(u32, u32) -> bool) {
        let mut dst = 0usize;
        for src in 0..self.n_act {
            let (e, d) = self.active[src];
            if keep(e, d) {
                self.active[dst] = (e, d);
                dst += 1;
            }
        }
        self.n_act = dst;
    }

    #[inline(always)]
    fn advance_to(&mut self, local_pos: u32, cldm: &[LdmSpan]) {
        let min_end = local_pos.saturating_add(5);
        if self.n_act > 0 {
            self.retain_active(|e, _| e >= min_end);
        }
        while self.cursor < cldm.len() {
            let sp = cldm[self.cursor];
            if sp.start_local > local_pos {
                break;
            }
            self.cursor += 1;
            if sp.end_local >= min_end {
                self.insert_pareto(sp.end_local, sp.dist);
            }
        }
    }

    #[inline(always)]
    fn insert_pareto(&mut self, end_new: u32, dist_new: u32) {
        if self.active[..self.n_act]
            .iter()
            .any(|&(ek, dk)| dk <= dist_new && ek >= end_new)
        {
            return;
        }
        self.retain_active(|es, ds| !(ds >= dist_new && es <= end_new));
        let mut ins = 0usize;
        while ins < self.n_act && self.active[ins].1 < dist_new {
            ins += 1;
        }
        if self.n_act < LDM_WAYS {
            self.active.copy_within(ins..self.n_act, ins + 1);
            self.active[ins] = (end_new, dist_new);
            self.n_act += 1;
        } else if ins == LDM_WAYS {
            self.active[LDM_WAYS - 1] = (end_new, dist_new);
        } else if ins <= LDM_WAYS - 2 {
            self.active.copy_within(ins..(LDM_WAYS - 2), ins + 1);
            self.active[ins] = (end_new, dist_new);
        }
    }

    #[inline(always)]
    fn collect_asc(
        &self,
        local_pos: u32,
        (min_dist_excl, max_dist): (usize, usize),
        min_strictly_gt: u32,
        out_asc: &mut [(u32, u32)],
    ) -> usize {
        let mut best_len = min_strictly_gt.max(4);
        let mut n_asc = 0usize;
        for &(end_l, dist) in &self.active[..self.n_act] {
            let d = dist as usize;
            if d > min_dist_excl && d <= max_dist {
                let len = end_l.saturating_sub(local_pos);
                if len > best_len && n_asc < out_asc.len() {
                    best_len = len;
                    out_asc[n_asc] = (len, dist);
                    n_asc += 1;
                }
            }
        }
        n_asc
    }
}

#[inline(always)]
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    crate::hashtable::common_prefix_len(a, b, usize::MAX)
}

#[inline(always)]
fn common_suffix_len(a: &[u8], b: &[u8]) -> usize {
    let max_len = a.len().min(b.len());
    let (a_end, b_end) = (a.len(), b.len());
    let mut i = 0usize;
    while i + 8 <= max_len {
        let wa = u64::from_le_bytes(a[a_end - i - 8..a_end - i].try_into().unwrap());
        let wb = u64::from_le_bytes(b[b_end - i - 8..b_end - i].try_into().unwrap());
        let diff = wa ^ wb;
        if diff != 0 {
            return i + (diff.leading_zeros() >> 3) as usize;
        }
        i += 8;
    }
    while i < max_len && a[a_end - 1 - i] == b[b_end - 1 - i] {
        i += 1;
    }
    i
}

#[inline(always)]
fn read_u48_le(data: &[u8], off: usize) -> u64 {
    if off + 8 <= data.len() {
        u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) & 0x0000_FFFF_FFFF_FFFF
    } else {
        let lo4 = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as u64;
        let hi2 = u16::from_le_bytes(data[off + 4..off + 6].try_into().unwrap()) as u64;
        lo4 | (hi2 << 32)
    }
}

#[inline(always)]
fn mix_ldm6(w6: u64) -> u64 {
    let z = (w6 ^ (w6 >> 23)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    z ^ (z >> 29)
}

#[inline(always)]
fn extend_ldm_anchor(env: ChunkEnv<'_>, p: usize, cand: usize, cap: usize) -> (u32, u32) {
    let max_back = (p - env.start).min(cand);
    let known_back = cap.min(max_back);
    let extra_back = if max_back > known_back {
        common_suffix_len(
            &env.data[cand - max_back..cand - known_back],
            &env.data[p - max_back..p - known_back],
        )
    } else {
        0
    };
    let (back, rem_fwd) = (known_back + extra_back, env.end - (p + 1));
    let fwd = common_prefix_len(
        &env.data[cand + 1..cand + 1 + rem_fwd],
        &env.data[p + 1..env.end],
    );
    (
        (p - back - env.start) as u32,
        (p + 1 + fwd - env.start) as u32,
    )
}

/// Single-table SplitMix64 8-way Long-Distance Matcher (even buckets: 6B anchors; odd buckets:
/// 1-in-4 content-defined 12B phrase anchors immune to Zipf eviction up to `1 GiB`).
#[derive(Default)]
pub(super) struct SparseLdmTable {
    buckets: Vec<[u32; LDM_WAYS]>,
    bucket_mask: usize,
    sync_mask: u64,
    indexed_end: usize,
    base: usize,
}

impl SparseLdmTable {
    #[inline]
    pub(super) fn reset_stream(&mut self) {
        self.bucket_mask = 0;
        self.indexed_end = 0;
        self.base = 0;
    }

    fn shift_down(&mut self, mut drop_len: usize) {
        while drop_len > 0 {
            let step = drop_len.min((u32::MAX as usize) - 8);
            let shift = step as u32;
            let min_valid_p1 = shift + 6;
            for b in &mut self.buckets {
                for slot in b {
                    *slot = if *slot >= min_valid_p1 {
                        *slot - shift
                    } else {
                        0
                    };
                }
            }
            self.indexed_end = self.indexed_end.saturating_sub(step);
            drop_len -= step;
        }
    }

    pub(super) fn extract_chunk_matches(
        &mut self,
        env: ChunkEnv<'_>,
        eff_win: usize,
        local_window_cap: usize,
        out: &mut ChunkLdmMatches,
    ) {
        out.clear();
        if eff_win <= local_window_cap || env.wsize <= local_window_cap || env.end <= 6 {
            return;
        }

        let n_buckets = (eff_win.next_power_of_two() >> 2).clamp(1 << 21, 1 << 22);
        let sync_mask = (((eff_win.saturating_sub(1) >> 26) + 1).next_power_of_two() - 1) as u64;
        let sync12_mask = (sync_mask + 1) * 4 - 1;
        if self.buckets.len() != n_buckets {
            self.buckets = vec![[0u32; LDM_WAYS]; n_buckets];
            self.bucket_mask = 0;
        } else if self.bucket_mask == 0 || self.sync_mask != sync_mask {
            self.buckets.fill([0; LDM_WAYS]);
        }
        if self.bucket_mask == 0 || self.sync_mask != sync_mask {
            self.bucket_mask = n_buckets - 1;
            self.sync_mask = sync_mask;
            self.indexed_end = 0;
            self.base = 0;
        }

        // Rebase `u32` bucket slots whenever 64-bit stream offsets approach `2^32`.
        if env.end.saturating_sub(self.base) >= (u32::MAX as usize) - 1 {
            let new_base = env.start.saturating_sub(env.wsize);
            if new_base > self.base {
                let delta = new_base - self.base;
                self.shift_down(delta);
                self.base = new_base;
            }
        }
        let env = ChunkEnv::new(
            &env.data[self.base..],
            env.start - self.base,
            env.end - self.base,
            env.wsize,
            env.dict,
        );
        let (data, chunk_start, chunk_end, wsize) = (env.data, env.start, env.end, env.wsize);
        let capped_end = chunk_end.min(data.len()).min((u32::MAX as usize) - 1);
        let buf_start = chunk_start - chunk_start.min(wsize).min(local_window_cap);
        let do_probe = !(buf_start == 0 && capped_end <= local_window_cap);
        let mut scan_start = self.indexed_end.max(chunk_start.saturating_sub(wsize));
        if do_probe {
            scan_start = scan_start.min(chunk_start);
        }
        scan_start = scan_start.max(5);
        if scan_start >= capped_end {
            return;
        }

        let mut prev_w6 = if scan_start >= 6 {
            read_u48_le(data, scan_start - 6)
        } else {
            u64::MAX
        };
        let (mut skip6, mut skip12) = (0usize, 0usize);
        let (mut recent, mut r_idx) = ([LdmSpan::default(); 8], 0usize);

        for p in scan_start..capped_end {
            let w6 = read_u48_le(data, p - 5);
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
                            cand >= 5 && read_u48_le(data, cand - 5) == w6
                        };
                        if !eq
                            || recent.iter().any(|r| {
                                r.dist == dist && local_p >= r.start_local && local_p < r.end_local
                            })
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
                        let (start_local, end_local) =
                            extend_ldm_anchor(env, p, cand, if is_12 { 11 } else { 5 });
                        let min_req = if is_12 {
                            5
                        } else if dist > (64 << 20) {
                            7
                        } else {
                            6
                        };
                        if end_local - start_local >= min_req {
                            let sp = LdmSpan {
                                start_local,
                                end_local,
                                dist,
                            };
                            out.push(sp);
                            recent[r_idx & 7] = sp;
                            r_idx = r_idx.wrapping_add(1);
                            best_end = best_end.max(end_local);
                            if end_local >= local_p + 24 {
                                skip6 =
                                    skip6.max((chunk_start + end_local as usize).saturating_sub(4));
                            }
                            if end_local >= local_p + 32 {
                                skip12 = skip12
                                    .max((chunk_start + end_local as usize).saturating_sub(8));
                            }
                        }
                    }
                }
                if p >= self.indexed_end {
                    let p1 = (p + 1) as u32;
                    let b = &mut self.buckets[bkt];
                    if !is_12 && b[0] != 0 && p1 >= b[0] && p1 - b[0] <= 16 {
                        b[0] = p1;
                    } else {
                        *b = [p1, b[0], b[1], b[2], b[3], b[4], b[5], b[6]];
                    }
                }
            };
            step((((h >> 32) as usize) & self.bucket_mask) & !1, false);
            if p >= 11 && (h & sync12_mask) == 0 {
                let lo8 = u64::from_le_bytes(data[p - 11..p - 3].try_into().unwrap());
                let h12 = mix_ldm6(lo8 ^ h);
                step((((h12 >> 32) as usize) & self.bucket_mask) | 1, true);
            }
        }

        self.indexed_end = self.indexed_end.max(capped_end);
        if out.len() > 1 {
            out.sort_unstable_by_key(|s| (s.start_local, s.dist));
            out.dedup();
        }
    }
}

/// Exact 4-byte hash chain over an attached custom dictionary's active window slice.
pub(super) struct DictChain {
    head: Vec<u32>,
    prev: Vec<u32>,
}

impl DictChain {
    pub(super) fn new(dict_active: &[u8]) -> Option<Self> {
        let n = dict_active.len();
        if n < 4 {
            return None;
        }
        let mut head = vec![0u32; 1 << 16];
        let mut prev = vec![0u32; n];
        for off in 0..=(n - 4) {
            let w4 = u32::from_le_bytes(dict_active[off..off + 4].try_into().unwrap());
            let h = (w4.wrapping_mul(HASH_MUL32) >> 16) as usize;
            prev[off] = head[h];
            head[h] = (off + 1) as u32;
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
        let dict_len = env.dict.len();
        let (mut cur, mut best_len, mut n_asc, mut probes) =
            (self.head[h], min_strictly_gt.max(3), 0usize, 0usize);

        while cur != 0 && probes < 64 && n_asc < out_asc.len() {
            let off = (cur - 1) as usize;
            cur = self.prev[off];
            let dict_rem = dict_len - off;
            let dist = stream_pos + dict_rem;
            if dist > env.wsize {
                break;
            }
            probes += 1;
            let max_l = dict_rem.min(target.len());
            if max_l <= best_len || env.dict[off + best_len] != target[best_len] {
                continue;
            }
            let mlen = common_prefix_len(&env.dict[off..off + max_l], &target[..max_l]);
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

impl ChunkEnv<'_> {
    /// Exact match length at `local_pos` inside `self.data[self.start..self.end]` for `dist`.
    #[inline(always)]
    pub(super) fn match_len(&self, local_pos: usize, dist: u32) -> u32 {
        let sp = self.start + local_pos;
        let d = dist as usize;
        if d == 0 || d > self.max_lz_dist(sp) as usize || sp >= self.end {
            return 0;
        }
        let rem = self.end - sp;
        if d <= sp {
            common_prefix_len(&self.data[sp - d..sp - d + rem], &self.data[sp..self.end]) as u32
        } else {
            let dict_rem = d - sp;
            if dict_rem > self.dict.len() {
                return 0;
            }
            let off = self.dict.len() - dict_rem;
            let max_l = dict_rem.min(rem);
            common_prefix_len(&self.dict[off..off + max_l], &self.data[sp..sp + max_l]) as u32
        }
    }
}

#[inline(always)]
fn min_len_for_dist(dist: u32) -> u32 {
    if dist <= 64 {
        2
    } else if dist <= 16_384 {
        3
    } else {
        4
    }
}

#[inline(always)]
fn find_closest_2b_scalar(data: &[u8], pos: usize, max_d2: usize) -> u32 {
    let (b0, b1) = (data[pos], data[pos + 1]);
    (1..=max_d2)
        .find(|&d| data[pos - d] == b0 && data[pos - d + 1] == b1)
        .map_or(u32::MAX, |d| d as u32)
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[target_feature(enable = "avx2")]
fn find_closest_2b_avx2(_token: impl SimdToken, data: &[u8], pos: usize, max_d2: usize) -> u32 {
    if pos < 64 {
        return find_closest_2b_scalar(data, pos, max_d2);
    }
    let win: &[u8; 65] = data[pos - 64..=pos].try_into().unwrap();
    let (vb0, vb1) = (
        _mm256_set1_epi8(win[64] as i8),
        _mm256_set1_epi8(data[pos + 1] as i8),
    );
    let hit = |lo: usize, base: usize| -> Option<usize> {
        let w0: &[u8; 32] = win[lo..lo + 32].try_into().unwrap();
        let w1: &[u8; 32] = win[lo + 1..lo + 33].try_into().unwrap();
        let m0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(w0), vb0)) as u32;
        let m1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(w1), vb1)) as u32;
        let h = m0 & m1;
        (h != 0).then(|| base + h.leading_zeros() as usize)
    };
    let d = hit(32, 1).or_else(|| (max_d2 > 32).then(|| hit(0, 33)).flatten());
    d.filter(|&d| d <= max_d2).map_or(u32::MAX, |d| d as u32)
}

#[inline(always)]
fn find_closest_2b(data: &[u8], pos: usize, max_d2: usize, avx2: OptAvx2) -> u32 {
    if max_d2 == 0 {
        return u32::MAX;
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(tok) = avx2 {
        return find_closest_2b_avx2(tok, data, pos, max_d2);
    }
    let _ = avx2;
    find_closest_2b_scalar(data, pos, max_d2)
}

#[derive(Default)]
pub(super) struct MatchGenScratch {
    pub(super) ldm: SparseLdmTable,
    buckets: Vec<u32>,
    bucket_shift: u32,
    forest: Vec<u32>,
    ring_cap: usize,
    ring_mask: usize,
    ht3_head: Vec<u32>,
    ht3_shift: u32,
    ht3_prev: Vec<u32>,
    data_id: usize,
    data_len: usize,
    indexed_start: usize,
    indexed_end: usize,
    prev_rep_dist: usize,
    prev_rep_end: usize,
}

impl MatchGenScratch {
    #[inline]
    pub(super) fn reset_stream(&mut self) {
        self.ldm.reset_stream();
        self.reset_h10();
    }

    #[inline]
    pub(super) fn reset_h10(&mut self) {
        self.data_id = 0;
        self.data_len = 0;
        self.indexed_start = 0;
        self.indexed_end = 0;
        self.prev_rep_dist = 0;
        self.prev_rep_end = 0;
    }

    fn prepare_h10(
        &mut self,
        data: &[u8],
        (buf_start, chunk_start, ring_cap): (usize, usize, usize),
        (bucket_bits, ht3_bits): (usize, usize),
    ) -> usize {
        assert!(ring_cap.is_power_of_two() && ring_cap >= 64);
        let (n_buckets, n_ht3, cur_id) = (
            1usize << bucket_bits,
            1usize << ht3_bits,
            data.as_ptr() as usize,
        );
        if self.buckets.len() == n_buckets
            && self.ht3_head.len() == n_ht3
            && self.ring_cap == ring_cap
            && self.data_id == cur_id
            && self.data_len == data.len()
            && self.indexed_start <= buf_start
            && self.indexed_end == chunk_start
        {
            return chunk_start;
        }
        let reset_zeroed = |v: &mut Vec<u32>, n: usize| {
            if v.len() != n {
                *v = vec![0u32; n];
            } else {
                v.fill(0);
            }
        };
        reset_zeroed(&mut self.buckets, n_buckets);
        self.bucket_shift = (32 - bucket_bits) as u32;
        if self.forest.len() < 2 * ring_cap {
            self.forest.resize(2 * ring_cap, 0);
        }
        reset_zeroed(&mut self.ht3_head, n_ht3);
        self.ht3_shift = (32 - ht3_bits) as u32;
        reset_zeroed(&mut self.ht3_prev, HT3_RING_SIZE);
        self.ring_cap = ring_cap;
        self.ring_mask = ring_cap - 1;
        self.data_id = cur_id;
        self.data_len = data.len();
        self.indexed_start = buf_start;
        self.indexed_end = buf_start;
        self.prev_rep_dist = 0;
        self.prev_rep_end = 0;
        buf_start
    }

    #[inline(always)]
    fn insert_and_find_3b(&mut self, data: &[u8], pos: usize, max_d3: usize) -> u32 {
        if data.len() - pos < 3 {
            return u32::MAX;
        }
        let (b0, b1, b2) = (data[pos], data[pos + 1], data[pos + 2]);
        let w3 = (b0 as u32) | ((b1 as u32) << 8) | ((b2 as u32) << 16);
        let h3 = (w3.wrapping_mul(HASH_MUL32) >> self.ht3_shift) as usize;
        let prev_p1 = self.ht3_head[h3];
        self.ht3_head[h3] = (pos + 1) as u32;
        self.ht3_prev[pos & HT3_RING_MASK] = prev_p1;
        if max_d3 == 0 {
            return u32::MAX;
        }
        let ring_lim = max_d3.min(HT3_RING_MASK);
        let (mut p1, mut hops) = (prev_p1, 0usize);
        while p1 != 0 && hops < MAX_HT3_HOPS {
            let cand = (p1 - 1) as usize;
            let dist = pos - cand;
            if dist > ring_lim {
                break;
            }
            if data[cand] == b0 && data[cand + 1] == b1 && data[cand + 2] == b2 {
                return dist as u32;
            }
            p1 = self.ht3_prev[cand & HT3_RING_MASK];
            hops += 1;
        }
        if max_d3 >= 16_384 && pos >= 16_384 {
            let c = pos - 16_384;
            if data[c] == b0 && data[c + 1] == b1 && data[c + 2] == b2 {
                return 16_384;
            }
        }
        u32::MAX
    }

    #[inline(always)]
    fn store_and_find_bst(
        &mut self,
        data: &[u8],
        pos: usize,
        (emit_limit, max_dist, comp_cap, max_depth): (usize, usize, usize, usize),
        emit: bool,
        out_asc: &mut [(u32, u32)],
    ) -> usize {
        let data_len = data.len();
        if data_len - pos < 4 {
            self.prev_rep_end = 0;
            return 0;
        }
        let w4 = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        let h4 = (w4.wrapping_mul(HASH_MUL32) >> self.bucket_shift) as usize;
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
        let (mut left_reroute, mut right_reroute) = (my_node, my_node + 1);
        let (mut min_len_left, mut min_len_right, mut best_len, mut n_out) =
            (0usize, 0usize, 3usize, 0usize);
        let mut push_bst = |elen: usize, dist: usize| {
            out_asc[n_out.min(out_asc.len() - 1)] = (elen as u32, dist as u32);
            n_out = (n_out + 1).min(out_asc.len());
        };

        let (mut cand_p1, mut depth) = (prev_root, 0usize);
        while cand_p1 != 0 {
            let cand = (cand_p1 - 1) as usize;
            let dist = pos - cand;
            if dist > max_dist || depth >= max_depth {
                (self.forest[left_reroute], self.forest[right_reroute]) = (0, 0);
                return n_out;
            }
            let cand_node = (cand & self.ring_mask) << 1;
            let skip = min_len_left.min(min_len_right);
            let mlen_key = skip
                + common_prefix_len(
                    &data[cand + skip..cand + max_key_len],
                    &data[pos + skip..pos + max_key_len],
                );

            if mlen_key == comp_cap {
                (self.forest[left_reroute], self.forest[right_reroute]) =
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
                        + common_prefix_len(
                            &data[cand + known_len..cand + check_lim],
                            &data[pos + known_len..pos + check_lim],
                        );
                    if pos + full_len > self.prev_rep_end {
                        (self.prev_rep_dist, self.prev_rep_end) = (dist, pos + full_len);
                    }
                    let elen = full_len.min(max_emit_len);
                    if emit && elen > best_len {
                        push_bst(elen, dist);
                    }
                }
                return n_out;
            }

            let elen = mlen_key.min(max_emit_len);
            if emit && elen > best_len {
                best_len = elen;
                push_bst(elen, dist);
            }
            if mlen_key == max_key_len || data[cand + mlen_key] > data[pos + mlen_key] {
                self.forest[right_reroute] = cand_p1;
                right_reroute = cand_node;
                cand_p1 = self.forest[cand_node];
                min_len_right = mlen_key;
            } else {
                self.forest[left_reroute] = cand_p1;
                left_reroute = cand_node + 1;
                cand_p1 = self.forest[cand_node + 1];
                min_len_left = mlen_key;
            }
            depth += 1;
        }
        (self.forest[left_reroute], self.forest[right_reroute]) = (0, 0);
        n_out
    }

    #[inline(always)]
    fn index_only_at(
        &mut self,
        data: &[u8],
        pos: usize,
        p4: (usize, usize, usize, usize),
        insert_3b: bool,
    ) {
        if insert_3b {
            let _ = self.insert_and_find_3b(data, pos, 0);
        }
        let _ = self.store_and_find_bst(data, pos, p4, false, &mut []);
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

pub(super) fn build_chunk_matches(
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
    let (bucket_bits, ht3_bits, comp_cap, max_depth, seed_depth, max_bst_steps) = if is_q12 {
        (20usize, 17usize, 2048usize, 2048usize, 1024usize, 48usize)
    } else if is_q11 {
        (19, 17, 512, 128, 64, 16)
    } else {
        (17, 14, 256, 64, 32, 16)
    };
    let enable_long_skip = !is_q11;
    let dict_ldm_len_cap = if is_q11 { 4096usize } else { 128usize };
    let max_lz_for_static = if is_q12 {
        37
    } else if is_q11 {
        28
    } else {
        24
    };

    // Rebase H10's `u32` `(pos + 1)` nodes when 64-bit stream offsets reach `2^32 - 1`.
    let h10_align = ring_cap.max(HT3_RING_SIZE);
    let h10_base = if chunk_end >= u32::MAX as usize {
        buf_start & !(h10_align - 1)
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
        (h10_buf_start, h10_chunk_start, ring_cap),
        (bucket_bits, ht3_bits),
    );
    if seed_from < h10_chunk_start {
        let ht3_seed_from = h10_chunk_start.saturating_sub(16_384).max(seed_from);
        for p in seed_from..h10_chunk_start {
            let md = (p - h10_buf_start).min(max_bst_window);
            scratch.index_only_at(
                h10_data,
                p,
                (h10_chunk_end, md, comp_cap, seed_depth),
                p >= ht3_seed_from,
            );
        }
        scratch.indexed_end = h10_chunk_start;
    }

    let avx2 = try_avx2(true);
    out.intervals.reserve(chunk_len * 2);
    if !cfg.large_window {
        out.static_offsets.reserve(chunk_len + 1);
        out.static_cands.reserve(chunk_len / 2);
    }

    let (mut asc_bst, mut pruned_bst) = ([(0u32, 0u32); 64], [(0u32, 0u32); 64]);
    let (mut ldm_steps, mut dict_steps) = ([(0u32, 0u32); LDM_WAYS], [(0u32, 0u32); 16]);
    let mut ldm_tracker = LdmActiveTracker::default();
    let (mut long_match_start, mut long_match_end, mut long_match_dist) = (0usize, 0usize, 0u32);

    for local_pos in 0..chunk_len {
        let iv_start = out.begin_pos(cfg.large_window);
        let stream_pos = chunk_start + local_pos;
        let h10_pos = h10_chunk_start + local_pos;
        let max_local_d = (stream_pos - buf_start).min(max_bst_window);
        if has_ldm {
            ldm_tracker.advance_to(local_pos as u32, cfg.ldm);
        }

        let skip_q10_interior =
            enable_long_skip && local_pos >= long_match_start + 3 && local_pos + 4 < long_match_end;
        if skip_q10_interior || chunk_len - local_pos < 2 {
            if skip_q10_interior {
                let rem_l = (long_match_end - local_pos) as u32;
                let l_lo = min_len_for_dist(long_match_dist);
                if l_lo <= rem_l {
                    out.intervals.push(MatchInterval {
                        l_lo,
                        l_hi: rem_l,
                        dist: long_match_dist,
                    });
                }
            }
            scratch.index_only_at(
                h10_data,
                h10_pos,
                (h10_chunk_end, max_local_d, comp_cap, 16),
                true,
            );
            continue;
        }

        let n_bst = scratch.store_and_find_bst(
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
        let d2 = find_closest_2b(h10_data, h10_pos, max_d2, avx2);

        let n_use_bst = if !is_q11 && n_bst > 4 {
            let mut rev_kept = [(0u32, 0u32); 16];
            rev_kept[0] = asc_bst[n_bst - 1];
            let (mut n_rev, mut kept_next_d) = (1usize, asc_bst[n_bst - 1].1);
            for &(l, d) in asc_bst[1..n_bst - 1].iter().rev() {
                if kept_next_d > d.saturating_add(d >> 2) && n_rev + 1 < rev_kept.len() {
                    rev_kept[n_rev] = (l, d);
                    n_rev += 1;
                    kept_next_d = d;
                }
            }
            rev_kept[n_rev] = asc_bst[0];
            n_rev += 1;
            pruned_bst[..n_rev].copy_from_slice(&rev_kept[..n_rev]);
            pruned_bst[..n_rev].reverse();
            n_rev
        } else if n_bst <= max_bst_steps {
            pruned_bst[..n_bst].copy_from_slice(&asc_bst[..n_bst]);
            n_bst
        } else {
            let half = max_bst_steps / 2;
            pruned_bst[..half].copy_from_slice(&asc_bst[..half]);
            pruned_bst[half..max_bst_steps].copy_from_slice(&asc_bst[n_bst - half..n_bst]);
            max_bst_steps
        };

        let mut longest_in_stream = if n_use_bst > 0 {
            pruned_bst[n_use_bst - 1].0 as usize
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
        prev_hi = out.push_pareto(pruned_bst[..n_use_bst].iter().copied(), 0, prev_hi);

        let mut n_ldm = 0usize;
        if longest_in_stream < dict_ldm_len_cap && has_ldm && stream_pos > max_local_d {
            let local_win_start = stream_pos - max_local_d;
            if stream_pos.saturating_sub(wsize) < local_win_start {
                n_ldm = ldm_tracker.collect_asc(
                    local_pos as u32,
                    (stream_pos - local_win_start, stream_pos.min(wsize)),
                    longest_in_stream as u32,
                    &mut ldm_steps,
                );
                if n_ldm > 0 {
                    longest_in_stream = ldm_steps[n_ldm - 1].0 as usize;
                }
            }
        }
        let n_dict = cfg
            .dict_chain
            .filter(|_| longest_in_stream < dict_ldm_len_cap && stream_pos < wsize)
            .map_or(0, |dc| {
                dc.find_pareto_matches_asc(env, stream_pos, longest_in_stream, &mut dict_steps)
            });

        prev_hi = out.push_pareto(ldm_steps[..n_ldm].iter().copied(), 5, prev_hi);
        prev_hi = out.push_pareto(dict_steps[..n_dict].iter().copied(), 4, prev_hi);

        if enable_long_skip && out.intervals.len() > iv_start {
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

        let longest_lz = if out.intervals.len() > iv_start {
            prev_hi as usize
        } else {
            0
        };
        if !cfg.large_window && chunk_end - stream_pos >= 4 && longest_lz < max_lz_for_static {
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
                } else if let Some(sm) = static_dict::lookup_with_mode::<false>(
                    slice,
                    base_dist,
                    STD_MAX_RAW_STATIC_DIST,
                    min_dict_out as u32,
                    180,
                    320,
                    16,
                ) {
                    out.static_cands.push(StaticMatchCand {
                        out_len: sm.out_len as u8,
                        base_len: sm.base_len as u8,
                        raw_dist: sm.raw_dist,
                    });
                }
            }
        }
    }

    scratch.indexed_end = chunk_end;
    let _ = out.begin_pos(cfg.large_window);
}

#[cfg(test)]
pub(super) mod tests {
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
        for idx in 0..160usize {
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

        #[cfg(target_arch = "x86_64")]
        if let Some(tok) = simd_tokens::Avx2Token::try_new() {
            for pos in 0..(data.len() - 1) {
                for max_d2 in [1usize, 31, 32, 33, 64] {
                    let eff_d2 = max_d2.min(pos);
                    assert_eq!(
                        find_closest_2b_avx2(tok, &data, pos, eff_d2),
                        find_closest_2b_scalar(&data, pos, eff_d2)
                    );
                }
            }
        }

        let (wsize, chunk_start, oracle_end) = (1024usize, 300usize, data.len());
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
                let mlen =
                    common_prefix_len(&data[cand..cand + (oracle_end - sp)], &data[sp..oracle_end]);
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

        // Exercise > comp_cap (512) RLE run + 600B subtree grafting in O(N).
        let rle_pos = data.len();
        data.extend(std::iter::repeat_n(b'Z', 2048));
        let mut phrase = Vec::new();
        super::super::test_helpers::append_lcg_bytes(&mut phrase, 600, 0xF00D_CAFE_1234_5678, 1);
        data.extend_from_slice(&phrase);
        data.extend_from_slice(b"__gap1__");
        data.extend_from_slice(&phrase);
        data.extend_from_slice(b"__gap2__");
        let p3_pos = data.len();
        data.extend_from_slice(&phrase);
        let cm_rle = build_test_matches(ChunkEnv::new(&data, 0, data.len(), 8192, &[]), 11);
        for lp in 0..data.len() {
            assert_pareto_frontier(cm_rle.at(lp));
        }
        assert!(cm_rle
            .at(rle_pos + 1)
            .last()
            .is_some_and(|iv| iv.dist == 1 && iv.l_hi >= 2040));
        assert!(cm_rle.at(p3_pos).last().is_some_and(|iv| iv.l_hi == 600));
    }

    #[test]
    fn test_dict_chain_and_sparse_ldm() {
        let dict = b"PREFIX_DICT_ANCHOR_STRING_WITH_TAIL_XYZ";
        let small = b"XYZ_NOT_CROSSING_AND_PREFIX_DICT_ANCHOR_STRING_HERE";
        let env = ChunkEnv::new(small, 0, small.len(), 1008, dict);
        let cm = build_test_matches(env, 11);
        assert!(cm.at(21).iter().any(|iv| iv.l_hi >= 25 && iv.dist > 21));
        assert_eq!(env.match_len(0, 3), 3);

        let rle = vec![0xAAu8; (2 << 20) + 4096];
        let (mut ldm, mut cldm) = (SparseLdmTable::default(), ChunkLdmMatches::default());
        let rle_env = ChunkEnv::new(&rle, 1 << 20, rle.len(), 4 << 20, &[]);
        ldm.extract_chunk_matches(rle_env, rle.len(), 1 << 20, &mut cldm);
        let non_empty: usize = ldm.buckets.iter().flatten().filter(|&&x| x != 0).count();
        assert!(non_empty <= 1);

        let gap = 20 << 20;
        let mut head = Vec::new();
        super::super::test_helpers::append_lcg_bytes(&mut head, 6144, 0x517C_C1B7_2722_0A95, 1);
        let mut data = vec![0u8; gap + 4096];
        data[..2048].copy_from_slice(&head[..2048]);
        data[gap..].copy_from_slice(&head[2048..]);
        data.copy_within(256..256 + 128, gap + 512);
        ldm.reset_stream();
        let d_env = ChunkEnv::new(&data, gap, data.len(), 1 << 26, &[]);
        ldm.extract_chunk_matches(d_env, 1 << 26, 1 << 20, &mut cldm);
        let exp_d = (gap + 512 - 256) as u32;
        assert!(cldm
            .iter()
            .any(|sp| sp.dist == exp_d && sp.start_local <= 512 && sp.end_local >= 640));

        let (local_cap, len) = (1 << 20, (1 << 20) + (128 << 10));
        let mut single = Vec::new();
        super::super::test_helpers::append_lcg_bytes(&mut single, len, 0xCAFE_BABE_9876_5432, 1);
        single.copy_within(512..512 + 256, local_cap + 65_536);
        ldm.reset_stream();
        let s_env = ChunkEnv::new(&single, 0, len, 4 << 20, &[]);
        ldm.extract_chunk_matches(s_env, 4 << 20, local_cap, &mut cldm);
        let exp_s = (local_cap + 65_536 - 512) as u32;
        assert!(cldm.iter().any(|sp| sp.dist == exp_s));
    }
}
