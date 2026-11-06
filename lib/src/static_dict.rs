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

//! Unified RFC 7932 built-in static dictionary (`kBrotliDictionary` + all 121 transforms).

use crate::constants::{dispatch_simd, SimdBackend};
#[cfg(target_arch = "x86_64")]
use simd_tokens::{simd_entry, SimdToken};
use std::sync::OnceLock;

const RFC7932_DICT: &[u8; 122_784] = include_bytes!("rfc7932_dict.bin");

const OFFSETS_BY_LEN: [usize; 25] = [
    0, 0, 0, 0, 0, 4096, 9216, 21504, 35840, 44032, 53248, 63488, 74752, 87040, 93696, 100864,
    104704, 106752, 108928, 113536, 115968, 118528, 119872, 121280, 122016,
];

const SIZE_BITS_BY_LEN: [u8; 25] = [
    0, 0, 0, 0, 10, 10, 11, 11, 10, 10, 10, 10, 10, 9, 9, 8, 7, 7, 8, 7, 7, 6, 6, 5, 5,
];

#[rustfmt::skip]
const PREFIX_SUFFIX_BLOB: [u8; 208] = [
    0x00, 0x20, 0x00, 0x2c, 0x20, 0x00, 0x20, 0x6f, 0x66, 0x20, 0x74, 0x68, 0x65, 0x20, 0x00, 0x20,
    0x6f, 0x66, 0x20, 0x00, 0x73, 0x20, 0x00, 0x2e, 0x00, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x00, 0x20,
    0x69, 0x6e, 0x20, 0x00, 0x22, 0x00, 0x20, 0x74, 0x6f, 0x20, 0x00, 0x22, 0x3e, 0x00, 0x0a, 0x00,
    0x2e, 0x20, 0x00, 0x5d, 0x00, 0x20, 0x66, 0x6f, 0x72, 0x20, 0x00, 0x20, 0x61, 0x20, 0x00, 0x20,
    0x74, 0x68, 0x61, 0x74, 0x20, 0x00, 0x27, 0x00, 0x20, 0x77, 0x69, 0x74, 0x68, 0x20, 0x00, 0x20,
    0x66, 0x72, 0x6f, 0x6d, 0x20, 0x00, 0x20, 0x62, 0x79, 0x20, 0x00, 0x28, 0x00, 0x2e, 0x20, 0x54,
    0x68, 0x65, 0x20, 0x00, 0x20, 0x6f, 0x6e, 0x20, 0x00, 0x20, 0x61, 0x73, 0x20, 0x00, 0x20, 0x69,
    0x73, 0x20, 0x00, 0x69, 0x6e, 0x67, 0x20, 0x00, 0x0a, 0x09, 0x00, 0x3a, 0x00, 0x65, 0x64, 0x20,
    0x00, 0x3d, 0x22, 0x00, 0x20, 0x61, 0x74, 0x20, 0x00, 0x6c, 0x79, 0x20, 0x00, 0x2c, 0x00, 0x3d,
    0x27, 0x00, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x00, 0x2e, 0x20, 0x54, 0x68, 0x69, 0x73, 0x20, 0x00,
    0x20, 0x6e, 0x6f, 0x74, 0x20, 0x00, 0x65, 0x72, 0x20, 0x00, 0x61, 0x6c, 0x20, 0x00, 0x66, 0x75,
    0x6c, 0x20, 0x00, 0x69, 0x76, 0x65, 0x20, 0x00, 0x6c, 0x65, 0x73, 0x73, 0x20, 0x00, 0x65, 0x73,
    0x74, 0x20, 0x00, 0x69, 0x7a, 0x65, 0x20, 0x00, 0xc2, 0xa0, 0x00, 0x6f, 0x75, 0x73, 0x20, 0x00,
];

#[rustfmt::skip]
const RAW_TRANSFORMS: [(u8, u8, u8); 121] = [
    (0,0,0), (0,0,1), (1,0,1), (0,12,0), (0,10,1), (0,0,9), (1,0,0), (20,0,1), (0,0,15), (0,10,0),
    (0,0,25), (0,13,0), (0,1,0), (3,0,1), (0,0,3), (1,10,1), (0,0,31), (0,0,38), (12,0,1), (0,0,36),
    (0,0,23), (0,0,43), (0,0,46), (0,3,0), (0,0,51), (0,0,53), (0,14,0), (0,2,0), (0,0,59), (0,0,63),
    (1,10,0), (0,0,48), (23,0,0), (1,0,3), (0,15,0), (0,0,72), (0,0,70), (0,0,79), (0,0,86), (0,16,0),
    (0,17,0), (9,0,0), (0,4,0), (0,0,93), (0,11,0), (0,0,100), (0,0,105), (0,0,110), (0,7,0), (0,1,115),
    (0,0,120), (0,0,123), (1,0,48), (0,0,125), (0,20,0), (0,18,0), (0,6,0), (0,0,91), (0,10,3), (0,8,0),
    (0,0,132), (0,0,137), (9,0,15), (0,5,0), (0,9,0), (1,10,3), (0,10,36), (23,0,91), (0,11,1), (0,10,43),
    (0,0,129), (1,0,23), (146,0,0), (9,0,6), (0,10,70), (0,0,152), (0,0,141), (23,0,1), (0,10,91), (0,10,23),
    (0,0,160), (1,0,129), (0,0,166), (1,11,1), (0,0,170), (1,11,0), (0,0,143), (0,11,36), (0,10,48), (1,0,91),
    (0,0,174), (1,10,48), (0,0,179), (0,0,184), (0,11,70), (0,0,190), (1,10,23), (0,11,43), (1,0,143), (0,10,141),
    (0,0,195), (0,11,23), (200,0,0), (1,0,141), (0,10,129), (0,11,129), (0,0,203), (0,11,3), (0,10,143), (1,10,141),
    (1,11,129), (1,11,3), (0,11,141), (0,11,91), (0,11,48), (1,11,23), (0,11,143), (1,11,48), (1,10,129), (1,11,143),
    (1,10,143),
];

const DISTINCT_PREFIX_IDS: [u8; 9] = [0, 1, 9, 20, 3, 12, 23, 146, 200];
const CORE_OPS: [u8; 3] = [0, 10, 11]; // Identity, UppercaseFirst, UppercaseAll
const CSR_BUCKETS_LOG2: usize = 14;
const CSR_BUCKETS: usize = 1 << CSR_BUCKETS_LOG2;
const CSR_WORDS: usize = 13_504;
const DICT_HASH_MUL32: u32 = 0x1E35_A7BD;
const FAST_TID_MASK: u128 = 0x47_9AC6_763C_EA7A_F37B_C777;

#[inline(always)]
const fn cstr_slice(id: u8) -> &'static [u8] {
    let start = id as usize;
    let mut end = start;
    while PREFIX_SUFFIX_BLOB[end] != 0 {
        end += 1;
    }
    PREFIX_SUFFIX_BLOB.split_at(start).1.split_at(end - start).0
}

#[inline(always)]
const fn hash_norm4_u32(w: u32) -> (usize, u8) {
    let a = w & 0x7f7f_7f7f;
    let az = a.wrapping_add(0x3f3f_3f3f) & !a.wrapping_add(0x2525_2525) & !w & 0x8080_8080;
    let ge = w & (w << 1) & 0x8080_8080;
    let c2 = ge & !(w << 2);
    let e0 = ge ^ c2;
    let ng = ge ^ 0x8080_8080;
    let a1 = (ng & 0x80) << 8;
    let a2 = ((c2 & 0x80) << 16) | ((a1 & ng) << 8);
    let a3 = (e0 << 24) | ((a1 & c2) << 16) | ((a2 & ng) << 8);
    let act = 0x80 | a1 | a2 | a3;
    let n = w | ((act & az) >> 2) | ((act & c2) << 6) | (((act & e0) << 9).wrapping_mul(5));
    let h = n.wrapping_mul(DICT_HASH_MUL32);
    ((h >> (32 - CSR_BUCKETS_LOG2)) as usize, (h >> 10) as u8)
}

#[inline(always)]
const fn hash_norm4(b0: u8, b1: u8, b2: u8, b3: u8) -> (usize, u8) {
    hash_norm4_u32(u32::from_le_bytes([b0, b1, b2, b3]))
}

#[inline(always)]
fn dict_word(wlen: usize, word_idx: usize) -> &'static [u8] {
    let off = OFFSETS_BY_LEN[wlen] + word_idx * wlen;
    &RFC7932_DICT[off..off + wlen]
}

#[inline(always)]
fn encode_raw_dist(word_idx: usize, base_len: usize, transform_id: usize) -> u32 {
    ((transform_id as u32) << SIZE_BITS_BY_LEN[base_len]) | (word_idx as u32)
}

#[inline(always)]
fn match_op01(b0: u8, w0: u8) -> Option<usize> {
    if b0 == w0 {
        Some(0)
    } else if w0.is_ascii_lowercase() && b0 == w0 - 0x20 {
        Some(1)
    } else {
        None
    }
}

struct BaseCsr {
    offsets: [u16; CSR_BUCKETS + 1],
    entries: [u32; CSR_WORDS],
}

impl BaseCsr {
    #[inline(always)]
    fn bucket(&self, b: usize) -> &[u32] {
        &self.entries[self.offsets[b] as usize..self.offsets[b + 1] as usize]
    }
}

const fn build_base_csr() -> BaseCsr {
    let d = RFC7932_DICT;
    let mut offsets = [0u16; CSR_BUCKETS + 1];
    let mut cursors = [0u16; CSR_BUCKETS + 1];
    let mut entries = [0u32; CSR_WORDS];
    let mut pass = 0usize;
    while pass < 2 {
        let mut base_len = 24usize;
        while base_len >= 4 {
            let (nwords, mut wi) = (1usize << SIZE_BITS_BY_LEN[base_len], 0usize);
            while wi < nwords {
                let off = OFFSETS_BY_LEN[base_len] + wi * base_len;
                let (b, tag_k4) = hash_norm4(d[off], d[off + 1], d[off + 2], d[off + 3]);
                if pass == 0 {
                    offsets[b + 1] += 1;
                } else {
                    let b4 = if base_len >= 5 { d[off + 4] as u32 } else { 0 };
                    entries[cursors[b] as usize] = (b4 << 24)
                        | ((tag_k4 as u32) << 16)
                        | ((base_len as u32) << 11)
                        | (wi as u32);
                    cursors[b] += 1;
                }
                wi += 1;
            }
            base_len -= 1;
        }
        if pass == 0 {
            let mut b = 0usize;
            while b < CSR_BUCKETS {
                offsets[b + 1] += offsets[b];
                b += 1;
            }
            assert!(offsets[CSR_BUCKETS] as usize == CSR_WORDS);
            cursors = offsets;
        }
        pass += 1;
    }
    BaseCsr { offsets, entries }
}

#[derive(Clone, Copy)]
struct SuffixRule(u64, u8, u8); // (pat, tid, len)

impl SuffixRule {
    #[inline(always)]
    fn matches(self, next8: u64) -> bool {
        ((next8 ^ self.0) << ((8 - self.2 as u32) * 8)) == 0
    }
}

#[derive(Clone, Copy)]
struct SuffixSet(u8, u8, u8, u8, u128); // (empty_tid, start, fast_end, end, b0_mask)

const fn build_suffix_tables() -> ([SuffixRule; 93], [[SuffixSet; 3]; 9], [u8; 10]) {
    let (mut rules, mut sets) = (
        [SuffixRule(0, 0, 0); 93],
        [[SuffixSet(u8::MAX, 0, 0, 0, 0); 3]; 9],
    );
    let (mut omit_last, mut tid) = ([u8::MAX; 10], 0usize);
    while tid < 121 {
        let (_, o, s) = RAW_TRANSFORMS[tid];
        if o >= 1 && o <= 9 && s == 0 {
            omit_last[o as usize] = tid as u8;
        }
        tid += 1;
    }

    let (mut out, mut keys, mut pid_pos) = (0usize, [0u32; 93], 0usize);
    while pid_pos < 9 {
        let (pid, mut op_idx) = (DISTINCT_PREFIX_IDS[pid_pos], 0usize);
        while op_idx < 3 {
            let (op, set_s, mut empty_tid) = (CORE_OPS[op_idx], out, u8::MAX);
            let (mut fast_cnt, mut b0_mask, mut t) = (0u8, 0u128, 0usize);
            while t < 121 {
                let (p, o, sid) = RAW_TRANSFORMS[t];
                if p == pid && o == op {
                    if sid == 0 && empty_tid == u8::MAX {
                        empty_tid = t as u8;
                    } else if sid != 0 {
                        let sfx = cstr_slice(sid);
                        let (len, mut pat_b, mut k) = (sfx.len(), [0u8; 8], 0usize);
                        assert!(sfx[0] < 128);
                        while k < len {
                            pat_b[k] = sfx[k];
                            k += 1;
                        }
                        let fast = ((FAST_TID_MASK >> t) & 1) != 0;
                        b0_mask |= 1u128 << sfx[0];
                        fast_cnt += fast as u8;
                        let r = SuffixRule(u64::from_le_bytes(pat_b), t as u8, len as u8);
                        let r_key = ((!fast as u32) << 16) | (((8 - len) as u32) << 8) | (t as u32);
                        let mut pos = out;
                        while pos > set_s && keys[pos - 1] > r_key {
                            (rules[pos], keys[pos]) = (rules[pos - 1], keys[pos - 1]);
                            pos -= 1;
                        }
                        (rules[pos], keys[pos]) = (r, r_key);
                        out += 1;
                    }
                }
                t += 1;
            }
            sets[pid_pos][op_idx] = SuffixSet(
                empty_tid,
                set_s as u8,
                set_s as u8 + fast_cnt,
                out as u8,
                b0_mask,
            );
            op_idx += 1;
        }
        pid_pos += 1;
    }
    assert!(out == 93);
    (rules, sets, omit_last)
}

static BASE_CSR: BaseCsr = build_base_csr();
const SUFFIX_TABLES: ([SuffixRule; 93], [[SuffixSet; 3]; 9], [u8; 10]) = build_suffix_tables();
static SUFFIX_RULES: [SuffixRule; 93] = SUFFIX_TABLES.0;
static SUFFIX_SETS: [[SuffixSet; 3]; 9] = SUFFIX_TABLES.1;
static OMIT_LAST_TIDS: [u8; 10] = SUFFIX_TABLES.2;

#[inline(always)]
fn load_prefix_u64(rest: &[u8]) -> u64 {
    if let Some(c8) = rest.first_chunk::<8>() {
        u64::from_le_bytes(*c8)
    } else {
        let mut tmp = [0u8; 8];
        tmp[..rest.len()].copy_from_slice(rest);
        u64::from_le_bytes(tmp)
    }
}

impl SuffixSet {
    #[inline(always)]
    fn has_any(&self) -> bool {
        self.0 != u8::MAX || self.1 < self.3
    }

    #[inline(always)]
    fn rules(&self, rest: &[u8], end: u8) -> (&'static [SuffixRule], u8, u64) {
        match rest.first() {
            Some(&b0) if b0 < 128 && ((self.4 >> b0) & 1) != 0 => (
                &SUFFIX_RULES[self.1 as usize..end as usize],
                b0,
                load_prefix_u64(rest),
            ),
            _ => (&[], 0, 0),
        }
    }

    #[inline(always)]
    fn upgrade_fast(&self, rest: &[u8]) -> (u32, usize) {
        let (rules, b0, next8) = self.rules(rest, self.2);
        for &r in rules {
            if (r.0 as u8) == b0 && r.matches(next8) {
                return (r.1 as u32, r.2 as usize);
            }
        }
        (self.0 as u32, 0)
    }

    #[inline(always)]
    fn for_each_match(
        &self,
        rest: &[u8],
        base_out: usize,
        min_out: usize,
        mut f: impl FnMut(u8, usize),
    ) {
        if base_out >= min_out && self.0 != u8::MAX {
            f(self.0, base_out);
        }
        let (rules, b0, next8) = self.rules(rest, self.3);
        for &r in rules {
            let out_l = base_out + r.2 as usize;
            if (r.0 as u8) == b0 && out_l >= min_out && r.matches(next8) {
                f(r.1, out_l);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StaticDictMatch {
    pub(crate) base_len: u8,
    pub(crate) out_len: u8,
    pub(crate) raw_dist: u32,
    pub(crate) gain: i32,
}

const _: () = assert!(core::mem::size_of::<Option<StaticDictMatch>>() <= 16);

#[inline(always)]
fn eq_chunk<const K: usize>(a: &[u8], b: &[u8], o: usize) -> bool {
    a[o..o + K].first_chunk::<K>() == b[o..o + K].first_chunk::<K>()
}

#[inline(always)]
fn eq_tail_from_1(sub: &[u8], word: &[u8], n: usize) -> bool {
    match n {
        4 => eq_chunk::<3>(sub, word, 1),
        5..=8 => eq_chunk::<4>(sub, word, 1) && eq_chunk::<4>(sub, word, n - 4),
        _ => {
            eq_chunk::<8>(sub, word, 1)
                && eq_chunk::<8>(sub, word, n - 8)
                && (n <= 16 || eq_chunk::<8>(sub, word, 9))
        }
    }
}

#[inline(always)]
fn can_len4_reach_min_out(b4: Option<u8>, needed_suffix: usize) -> bool {
    needed_suffix == 0
        || b4.is_some_and(|c| {
            matches!(c, b' ' | b',' | b'.' | b'"' | b'=' | b'\n' | b'e' | b'l')
                || (needed_suffix == 1 && matches!(c, b']' | b'\'' | b':' | b'('))
        })
}

pub(crate) type LookupArgs = (u64, u32, u32, i32, i32, i32);

#[inline(always)]
fn probe_base_csr<const EXHAUSTIVE: bool>(
    sub: &[u8],
    pid_pos: usize,
    plen: usize,
    min_out: u32,
    (base_dist, max_raw_dist, _, lit_mult, cmd_off, min_gain): LookupArgs,
    best: &mut Option<StaticDictMatch>,
    mut emit: impl FnMut(usize, usize, u8, usize),
) {
    let Some(first4) = sub.first_chunk::<4>() else {
        return;
    };
    let (b_idx, tag_k4) = hash_norm4_u32(u32::from_le_bytes(*first4));
    let opt_b4 = sub.get(4).copied();
    let tag5 = opt_b4.map_or(u16::MAX, |b4| ((b4 as u16) << 8) | (tag_k4 as u16));

    for &entry in BASE_CSR.bucket(b_idx) {
        let (wlen, widx) = (((entry >> 11) & 0x1F) as usize, (entry & 0x7FF) as usize);
        if !EXHAUSTIVE {
            let ok = if wlen >= 5 {
                (entry >> 16) as u16 == tag5 && sub.len() >= wlen
            } else {
                (entry >> 16) as u8 == tag_k4 && {
                    let eff = best.map_or(min_out, |b| min_out.max(b.out_len as u32 + 1));
                    can_len4_reach_min_out(opt_b4, (eff as usize).saturating_sub(4 + plen))
                }
            };
            if !ok {
                continue;
            }
            let word = dict_word(wlen, widx);
            let Some(op_idx) = eq_tail_from_1(sub, word, wlen)
                .then(|| match_op01(first4[0], word[0]))
                .flatten()
            else {
                continue;
            };
            let set = &SUFFIX_SETS[pid_pos][op_idx];
            if set.0 == u8::MAX {
                continue;
            }
            let (tid, slen0) = set.upgrade_fast(&sub[wlen..]);
            let copy_extra_q5 = (crate::map_to_sym::COPY_EXTRA
                [crate::map_to_sym::copy_code_only(wlen as u32)]
                << 5) as i32;
            for &(cand_tid, slen) in
                &[(tid, slen0), (set.0 as u32, 0)][..if slen0 > 0 { 2 } else { 1 }]
            {
                let out_len = (plen + wlen + slen) as u32;
                let raw64 = base_dist + encode_raw_dist(widx, wlen, cand_tid as usize) as u64;
                if out_len < min_out || raw64 > max_raw_dist as u64 {
                    continue;
                }
                let raw_dist = raw64 as u32;
                let dist_bits_q5 = ((31 ^ (raw_dist | 1).leading_zeros()) << 5) as i32;
                let gain = lit_mult * (out_len as i32) - (cmd_off + dist_bits_q5 + copy_extra_q5);
                if gain >= min_gain && best.is_none_or(|b| gain > b.gain) {
                    *best = Some(StaticDictMatch {
                        base_len: wlen as u8,
                        out_len: out_len as u8,
                        raw_dist,
                        gain,
                    });
                }
            }
        } else {
            if (entry >> 16) as u8 != tag_k4 {
                continue;
            }
            let (word, min_o) = (dict_word(wlen, widx), min_out as usize);
            let mut lcp = 0usize;
            if word.starts_with(first4) {
                lcp = 4;
                while lcp < wlen.min(sub.len()) && word[lcp] == sub[lcp] {
                    lcp += 1;
                }
                if lcp == wlen {
                    SUFFIX_SETS[pid_pos][0].for_each_match(
                        &sub[wlen..],
                        plen + wlen,
                        min_o,
                        |t, l| emit(wlen, widx, t, l),
                    );
                }
                if pid_pos == 0 {
                    for m in wlen.saturating_sub(9).max(min_o)..=lcp.min(wlen - 1) {
                        emit(wlen, widx, OMIT_LAST_TIDS[wlen - m], m);
                    }
                    if wlen >= 5
                        && lcp >= wlen - 1
                        && wlen + 3 >= min_o
                        && sub[wlen - 1..].starts_with(b"ing ")
                    {
                        emit(wlen, widx, 49, wlen + 3);
                    }
                }
            }
            if sub.len() >= wlen && lcp < wlen {
                let mut upper_buf = [0u8; 24];
                for (op_idx, upper_all) in [(1, false), (2, true)] {
                    let sset = &SUFFIX_SETS[pid_pos][op_idx];
                    if sset.has_any() {
                        apply_upper_rfc7932(&mut upper_buf[..wlen], word, upper_all);
                        if upper_buf[..wlen] == sub[..wlen] {
                            sset.for_each_match(&sub[wlen..], plen + wlen, min_o, |t, l| {
                                emit(wlen, widx, t, l)
                            });
                        }
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn lookup_impl(
    slice: &[u8],
    base_dist: u64,
    mrd_mo: u64,
    lm_co: u64,
    mg_fast: u64,
) -> Option<StaticDictMatch> {
    let fast_only_empty_prefix = (mg_fast & 1) != 0;
    let cfg: LookupArgs = (
        base_dist,
        mrd_mo as u32,
        (mrd_mo >> 32) as u32,
        lm_co as i32,
        (lm_co >> 32) as i32,
        (mg_fast >> 1) as i32,
    );
    let (_, max_raw_dist, min_out_len, lit_mult, cmd_off, min_gain) = cfg;
    if slice.len() < 4 || base_dist > max_raw_dist as u64 {
        return None;
    }
    let min_dist_q5 = ((31 ^ ((base_dist as u32) | 1).leading_zeros()) << 5) as i32;
    let min_needed = min_gain + cmd_off + min_dist_q5;
    let mut req_len = 4;
    while req_len <= 8 && req_len * lit_mult < min_needed {
        req_len += 1;
    }
    let min_out = min_out_len.max(req_len as u32);
    if slice.len() < min_out as usize {
        return None;
    }
    let mut best = None;
    probe_base_csr::<false>(slice, 0, 0, min_out, cfg, &mut best, |_, _, _, _| {});
    if !fast_only_empty_prefix && slice[0] == b' ' && slice.len() >= 5 {
        probe_base_csr::<false>(&slice[1..], 1, 1, min_out, cfg, &mut best, |_, _, _, _| {});
        if slice.len() >= 9 && slice.first_chunk::<5>() == Some(b" the ") {
            probe_base_csr::<false>(&slice[5..], 2, 5, min_out, cfg, &mut best, |_, _, _, _| {});
        }
    }
    best
}

#[cfg_attr(target_arch = "x86_64", cold)]
#[inline(never)]
fn lookup_with_mode(
    s: &[u8],
    bd: u64,
    mrd_mo: u64,
    lm_co: u64,
    mg_fast: u64,
) -> Option<StaticDictMatch> {
    lookup_impl(s, bd, mrd_mo, lm_co, mg_fast)
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
#[inline(never)]
fn lookup_avx2(
    s: &[u8],
    bd: u64,
    mrd_mo: u64,
    lm_co: u64,
    mg_fast: u64,
    _tok: impl SimdToken,
) -> Option<StaticDictMatch> {
    lookup_impl(s, bd, mrd_mo, lm_co, mg_fast)
}

#[inline(always)]
pub(crate) fn lookup_simd<S: SimdBackend>(
    simd: S,
    fast: bool,
    s: &[u8],
    (bd, mrd, mo, lm, co, mg): LookupArgs,
) -> Option<StaticDictMatch> {
    let mrd_mo = (mrd as u64) | ((mo as u64) << 32);
    let lm_co = (lm as u32 as u64) | ((co as u32 as u64) << 32);
    let mg_fast = ((mg as u32 as u64) << 1) | (fast as u64);
    dispatch_simd!(
        simd, S => |tok| lookup_avx2(s, bd, mrd_mo, lm_co, mg_fast, tok),
        lookup_with_mode(s, bd, mrd_mo, lm_co, mg_fast)
    )
}

#[inline]
fn to_upper_case_rfc7932(p: &mut [u8]) -> usize {
    if p[0] < 0xc0 {
        p[0] ^= u8::from(p[0].is_ascii_lowercase()) << 5;
        1
    } else if p.len() >= 2 && p[0] < 0xe0 {
        p[1] ^= 32;
        2
    } else {
        if p.len() >= 3 {
            p[2] ^= 5;
        }
        3.min(p.len())
    }
}

#[inline]
fn apply_upper_rfc7932(dst: &mut [u8], word: &[u8], upper_all: bool) {
    dst.copy_from_slice(word);
    if upper_all {
        let mut i = 0;
        while i < dst.len() {
            i += to_upper_case_rfc7932(&mut dst[i..]);
        }
    } else {
        to_upper_case_rfc7932(dst);
    }
}

pub(crate) const FAST_DICT_LOG2: usize = 11;
pub(crate) const FAST_DICT_SLOTS: usize = 1 << FAST_DICT_LOG2;

#[derive(Clone, Copy)]
#[repr(C)]
struct FastDictCold {
    w_tail: u64,
    dict_idx: u32,
    gain_base_q5: i16,
    base_len: u16,
}

#[repr(C, align(64))]
pub(crate) struct FastDictTable {
    head: [u64; FAST_DICT_SLOTS],
    cold: [FastDictCold; FAST_DICT_SLOTS],
}

fn expand_transform_rfc7932(out: &mut [u8; 64], bl: usize, wi: usize, tid: usize) -> usize {
    let (pid, op, sid) = RAW_TRANSFORMS[tid];
    let (pfx, sfx, word) = (cstr_slice(pid), cstr_slice(sid), dict_word(bl, wi));
    out[..pfx.len()].copy_from_slice(pfx);
    let mid = match op {
        0 => word,
        1..=9 => &word[..bl.saturating_sub(op as usize)],
        10 | 11 => {
            apply_upper_rfc7932(&mut out[pfx.len()..pfx.len() + bl], word, op == 11);
            &[][..]
        }
        _ => &word[(op as usize - 11).min(bl)..],
    };
    let pos = pfx.len() + if matches!(op, 10 | 11) { bl } else { mid.len() };
    if !matches!(op, 10 | 11) {
        out[pfx.len()..pos].copy_from_slice(mid);
    }
    out[pos..pos + sfx.len()].copy_from_slice(sfx);
    pos + sfx.len()
}

pub(crate) fn fast_dict() -> &'static FastDictTable {
    static FAST: OnceLock<Box<FastDictTable>> = OnceLock::new();
    FAST.get_or_init(|| {
        let mut table = Box::new(FastDictTable {
            head: [0x00FF_FFFF_FFFF_FFFF; FAST_DICT_SLOTS],
            cold: [FastDictCold {
                w_tail: 0,
                dict_idx: 0,
                gain_base_q5: 0,
                base_len: 0,
            }; FAST_DICT_SLOTS],
        });
        let mut best_scores = [0u64; FAST_DICT_SLOTS];
        let mut buf = [0u8; 64];

        for (tid, &(pid, op, sid)) in RAW_TRANSFORMS.iter().enumerate() {
            let aff_len = cstr_slice(pid).len() + cstr_slice(sid).len();
            if aff_len > 8 {
                continue;
            }
            let omit = match op {
                1..=9 => op as usize,
                12..=20 => (op - 11) as usize,
                _ => 0,
            };
            let min_bl = (5usize.saturating_sub(aff_len) + omit).max(4).max(omit + 3);
            let max_bl = (12 - aff_len + omit).min(15);
            if min_bl > max_bl {
                continue;
            }
            for (base_len, &size_bits) in SIZE_BITS_BY_LEN
                .iter()
                .enumerate()
                .take(max_bl + 1)
                .skip(min_bl)
            {
                let nwords = 1usize << size_bits;
                let limit = if cfg!(miri) {
                    nwords.min(4)
                } else if tid < 32 {
                    nwords
                } else {
                    nwords.min(512)
                };
                let cc = crate::map_to_sym::copy_code_only(base_len as u32);
                let copy_extra_q5 = (crate::map_to_sym::COPY_EXTRA[cc] << 5) as i32;
                for word_idx in 0..limit {
                    let out_len = expand_transform_rfc7932(&mut buf, base_len, word_idx, tid);
                    if !(5..=12).contains(&out_len) {
                        continue;
                    }
                    let dict_idx = encode_raw_dist(word_idx, base_len, tid);
                    let est_raw = 8192u32.saturating_add(dict_idx);
                    let dist_bits_q5 = ((31 ^ (est_raw | 1).leading_zeros()) << 5) as i32;
                    let gain_base = (out_len as i32) * 168 - 352 - copy_extra_q5;
                    let gain_q5 = gain_base - dist_bits_q5;
                    if gain_q5 < 64 {
                        continue;
                    }
                    let score =
                        ((gain_q5 as u64) << 16) / ((word_idx as u64 + 24) * (tid as u64 + 3));
                    let k4 = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
                    let slot = (k4.wrapping_mul(DICT_HASH_MUL32) >> (32 - FAST_DICT_LOG2)) as usize;
                    if score > best_scores[slot] {
                        best_scores[slot] = score;
                        let mut b12 = [0u8; 12];
                        b12[..out_len].copy_from_slice(&buf[..out_len]);
                        let w0 = u64::from_le_bytes(*b12.first_chunk::<8>().unwrap());
                        let w_tail = u64::from_le_bytes(*b12.last_chunk::<8>().unwrap());
                        table.head[slot] = (w0 & 0x00FF_FFFF_FFFF_FFFF) | ((out_len as u64) << 56);
                        table.cold[slot] = FastDictCold {
                            w_tail,
                            dict_idx,
                            gain_base_q5: gain_base as i16,
                            base_len: base_len as u16,
                        };
                    }
                }
            }
        }
        table
    })
}

#[inline(always)]
pub(crate) fn lookup_fast(
    table: &FastDictTable,
    word64: u64,
    hash32_4b: u32,
    next4: &[u8; 4],
    (base_dist, max_raw_dist): (u64, u32),
    (lit_mult, cmd_off, min_gain): (i32, i32, i32),
) -> Option<StaticDictMatch> {
    let slot = (hash32_4b >> (32 - FAST_DICT_LOG2)) as usize;
    let h = table.head[slot];
    if ((word64 ^ h) << 24) != 0 {
        return None;
    }
    let out_len = (h >> 56) as usize;
    if out_len == 0 {
        return None;
    }
    let cold = &table.cold[slot];
    if out_len <= 7 {
        if ((word64 ^ h) << ((8 - out_len) * 8)) != 0 {
            return None;
        }
    } else if out_len == 8 {
        if (word64 >> 32) != (cold.w_tail & 0xFFFF_FFFF) {
            return None;
        }
    } else {
        let tail_in = (word64 >> 32) | ((u32::from_le_bytes(*next4) as u64) << 32);
        if ((tail_in ^ cold.w_tail) << ((12 - out_len) * 8)) != 0 {
            return None;
        }
    }
    let raw64 = base_dist + cold.dict_idx as u64;
    if raw64 > max_raw_dist as u64 {
        return None;
    }
    let raw_dist = raw64 as u32;
    let dist_bits_q5 = ((31 ^ (raw_dist | 1).leading_zeros()) << 5) as i32;
    let gain = if lit_mult == 168 && cmd_off == 352 {
        (cold.gain_base_q5 as i32) - dist_bits_q5
    } else {
        let copy_extra_q5 = (out_len as i32) * 168 - 352 - (cold.gain_base_q5 as i32);
        lit_mult * (out_len as i32) - (cmd_off + dist_bits_q5 + copy_extra_q5)
    };
    (gain >= min_gain).then_some(StaticDictMatch {
        base_len: cold.base_len as u8,
        out_len: out_len as u8,
        raw_dist,
        gain,
    })
}

const EXT_BUCKETS_LOG2: usize = 15;
const EXT_BUCKETS: usize = 1 << EXT_BUCKETS_LOG2;

#[inline(always)]
fn hash4_raw(k4: u32) -> usize {
    (k4.wrapping_mul(DICT_HASH_MUL32) >> (32 - EXT_BUCKETS_LOG2)) as usize
}

struct Dict121Ext {
    offsets: Box<[u16]>,
    entries: Box<[u32]>,
}

impl Dict121Ext {
    #[inline(always)]
    fn bucket(&self, h: usize) -> &[u32] {
        &self.entries[self.offsets[h] as usize..self.offsets[h + 1] as usize]
    }
}

fn dict121_ext() -> &'static Dict121Ext {
    static EXT: OnceLock<Dict121Ext> = OnceLock::new();
    EXT.get_or_init(|| {
        let pack = |wi, bl, ol, tid| {
            (wi as u32) | ((bl as u32) << 11) | ((ol as u32) << 16) | ((tid as u32) << 21)
        };
        let mut items = Vec::with_capacity(64_000);
        for wi in 0..if cfg!(miri) { 8 } else { 1024usize } {
            let w = dict_word(4, wi);
            let k4 = u32::from_le_bytes([w[0], w[1], w[2], b'i']);
            items.push((hash4_raw(k4) as u16, pack(wi, 4, 7, 49)));
        }
        for (tid, &(_, op, _)) in RAW_TRANSFORMS
            .iter()
            .enumerate()
            .filter(|&(_, &(_, op, _))| op >= 12)
        {
            let skip = (op - 11) as usize;
            for (bl, &sb) in SIZE_BITS_BY_LEN.iter().enumerate().skip(skip + 4) {
                let nwords = 1usize << sb;
                let limit = if cfg!(miri) { nwords.min(4) } else { nwords };
                for wi in 0..limit {
                    let s = &dict_word(bl, wi)[skip..];
                    let k4 = u32::from_le_bytes(*s.first_chunk::<4>().unwrap());
                    items.push((hash4_raw(k4) as u16, pack(wi, bl, bl - skip, tid)));
                }
            }
        }
        let sub = |e: u32| {
            let (wi, bl, ol) = (
                (e & 0x7FF) as usize,
                ((e >> 11) & 0x1F) as usize,
                ((e >> 16) & 0x1F) as usize,
            );
            let w = dict_word(bl, wi);
            if (e >> 21) == 49 {
                &w[..3]
            } else {
                &w[bl - ol..]
            }
        };
        items.sort_unstable_by(|&(ha, a), &(hb, b)| {
            (ha, (a >> 11) & 0x1F, sub(a), a >> 21, a & 0x7FF).cmp(&(
                hb,
                (b >> 11) & 0x1F,
                sub(b),
                b >> 21,
                b & 0x7FF,
            ))
        });
        items.dedup_by(|&mut (hl, l), &mut (hk, k)| {
            (hl, (l >> 11) & 0x3FF, sub(l)) == (hk, (k >> 11) & 0x3FF, sub(k))
        });

        let (mut offsets, mut entries) =
            (vec![0u16; EXT_BUCKETS + 1], Vec::with_capacity(items.len()));
        for (h, e) in items {
            offsets[h as usize + 1] += 1;
            entries.push(e);
        }
        for b in 0..EXT_BUCKETS {
            offsets[b + 1] += offsets[b];
        }
        Dict121Ext {
            offsets: offsets.into_boxed_slice(),
            entries: entries.into_boxed_slice(),
        }
    })
}

pub(crate) fn warm_dict121() {
    let _ = dict121_ext();
}

#[inline(never)]
pub(crate) fn for_each_dict121_match(
    slice: &[u8],
    base_dist: u64,
    max_raw_dist: u32,
    min_out_len: usize,
    mut on_match: impl FnMut(u32, u32, u32),
) {
    let (min_out, max_raw) = (min_out_len.max(4), max_raw_dist as u64);
    if slice.len() < min_out || base_dist > max_raw {
        return;
    }
    let mut emit = |base_len: usize, word_idx: usize, tid: u8, out_len: usize| {
        let raw = base_dist + encode_raw_dist(word_idx, base_len, tid as usize) as u64;
        if raw <= max_raw {
            on_match(out_len as u32, base_len as u32, raw as u32);
        }
    };
    let (mut best, cfg) = (None, (0, 0, 0, 0, 0, 0));
    probe_base_csr::<true>(slice, 0, 0, min_out as u32, cfg, &mut best, &mut emit);
    if matches!(slice[0], b' ' | b's' | b',' | b'e' | b'.' | 0xc2) {
        for (pid_pos, &pid) in DISTINCT_PREFIX_IDS.iter().enumerate().skip(1) {
            let pfx = cstr_slice(pid);
            if pfx[0] == slice[0] && slice.starts_with(pfx) {
                probe_base_csr::<true>(
                    &slice[pfx.len()..],
                    pid_pos,
                    pfx.len(),
                    min_out as u32,
                    cfg,
                    &mut best,
                    &mut emit,
                );
            }
        }
    }
    let raw_k4 = u32::from_le_bytes(*slice.first_chunk::<4>().unwrap());
    for &e in dict121_ext().bucket(hash4_raw(raw_k4)) {
        let (wi, bl, ol, tid) = (
            (e & 0x7FF) as usize,
            ((e >> 11) & 0x1F) as usize,
            ((e >> 16) & 0x1F) as usize,
            (e >> 21) as u8,
        );
        let w = dict_word(bl, wi);
        let eq = (min_out..=slice.len()).contains(&ol)
            && if tid == 49 {
                w[..3] == slice[..3] && &slice[3..7] == b"ing "
            } else {
                w[bl - ol..] == slice[..ol]
            };
        if eq {
            emit(bl, wi, tid, ol);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::constants::STD_MAX_RAW_STATIC_DIST;
    use brotli_decompressor::transform::TransformDictionaryWord;

    fn verify_word(expected: &[u8], out_len: u32, base_len: u32, word_id: u32) {
        let (len, shift) = (base_len as usize, SIZE_BITS_BY_LEN[base_len as usize]);
        let (widx, tid) = (
            (word_id & ((1 << shift) - 1)) as usize,
            (word_id >> shift) as i32,
        );
        assert!(tid < 121);
        let mut dec = [0u8; 64];
        let n = TransformDictionaryWord(&mut dec, dict_word(len, widx), len as i32, tid) as usize;
        assert_eq!((n, &dec[..n]), (out_len as usize, &expected[..n]));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_all_121_transforms_discovered_and_verified() {
        let mut buf = [0u8; 64];
        for (base_len, &sb) in SIZE_BITS_BY_LEN.iter().enumerate().skip(4) {
            for word_idx in 0..(1usize << sb) {
                let w = dict_word(base_len, word_idx);
                let h0 = hash_norm4(w[0], w[1], w[2], w[3]);
                for tid in [0i32, 9, 44] {
                    let tlen = TransformDictionaryWord(&mut buf, w, base_len as i32, tid) as usize;
                    let phrase = &buf[..tlen];
                    assert_eq!(hash_norm4(phrase[0], phrase[1], phrase[2], phrase[3]), h0);
                    let mut found = false;
                    for_each_dict121_match(phrase, 1, STD_MAX_RAW_STATIC_DIST, 4, |ol, bl, rd| {
                        verify_word(phrase, ol, bl, rd - 1);
                        found |= ol as usize == tlen;
                    });
                    assert!(
                        found,
                        "missing base_len={base_len} word_idx={word_idx} tid={tid}"
                    );
                }
            }
        }
        for word_idx in [0usize, 10] {
            let base_word = dict_word(14, word_idx);
            for tid in 0..121i32 {
                let tlen = TransformDictionaryWord(&mut buf, base_word, 14, tid) as usize;
                let (phrase, mut found) = (&buf[..tlen], false);
                for_each_dict121_match(phrase, 1, STD_MAX_RAW_STATIC_DIST, 4, |ol, bl, rd| {
                    verify_word(phrase, ol, bl, rd - 1);
                    found |= ol as usize == tlen;
                });
                assert!(found, "missing tid={tid} word_idx={word_idx}");
            }
        }
        let mut found_timing = false;
        for_each_dict121_match(b"timing ", 1, STD_MAX_RAW_STATIC_DIST, 4, |ol, bl, rd| {
            verify_word(b"timing ", ol, bl, rd - 1);
            found_timing |= ol == 7 && bl == 4 && ((rd - 1) >> 10) == 49;
        });
        assert!(found_timing);
    }

    #[test]
    fn test_fast_lookup_parity_and_decoder_reference() {
        #[rustfmt::skip]
        let phrases: &[&[u8]] = &[
            b"time", b"time ", b"time, ", b"time. The ", b"time\">", b"time=\"", b"time\n\t",
            b"worked ", b"Information, ", b" information. ", b" Information, ",
            b" the government of the ",
        ];
        let lk = |s: &[u8], b: u64| {
            lookup_simd(
                crate::constants::Scalar,
                false,
                s,
                (b, STD_MAX_RAW_STATIC_DIST, 4, 168, 352, 48),
            )
        };
        for &phrase in phrases {
            let m = lk(phrase, 1).unwrap();
            verify_word(phrase, m.out_len as u32, m.base_len as u32, m.raw_dist - 1);
            if phrase[0] != b' ' {
                let mf = lookup_simd(
                    crate::constants::Scalar,
                    true,
                    phrase,
                    (1, STD_MAX_RAW_STATIC_DIST, 4, 168, 352, 48),
                )
                .unwrap();
                assert_eq!((mf.out_len, mf.raw_dist), (m.out_len, m.raw_dist));
            }
        }
        assert!(lk(b"time", 100_000).is_none());
        let m = lk(b"information,x", 100_000).unwrap();
        assert_eq!(m.out_len, 12);
        verify_word(
            b"information,x",
            m.out_len as u32,
            m.base_len as u32,
            m.raw_dist - 100_000,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_fast_dict_all_entries_match_rfc7932_decoder() {
        let fd = fast_dict();
        let mut populated = 0usize;
        for slot in 0..FAST_DICT_SLOTS {
            let h = fd.head[slot];
            let out_len = (h >> 56) as u32;
            if out_len == 0 {
                continue;
            }
            populated += 1;
            assert!((5..=12).contains(&out_len));
            let cold = &fd.cold[slot];
            let base_len = cold.base_len as u32;
            let dict_idx = cold.dict_idx;
            let mut phrase = [0u8; 16];
            phrase[..7].copy_from_slice(&h.to_le_bytes()[..7]);
            phrase[4..12].copy_from_slice(&cold.w_tail.to_le_bytes());
            verify_word(&phrase[..out_len as usize], out_len, base_len, dict_idx);

            let w64 = u64::from_le_bytes(*phrase.first_chunk::<8>().unwrap());
            let h32 = (w64 as u32).wrapping_mul(DICT_HASH_MUL32);
            assert_eq!((h32 >> (32 - FAST_DICT_LOG2)) as usize, slot);
            let hit = lookup_fast(
                fd,
                w64,
                h32,
                phrase[8..12].first_chunk::<4>().unwrap(),
                (1, STD_MAX_RAW_STATIC_DIST),
                (168, 352, 48),
            )
            .expect("populated fast_dict slot must match its own phrase");
            assert_eq!(
                (hit.out_len as u32, hit.base_len as u32, hit.raw_dist),
                (out_len, base_len, dict_idx + 1)
            );
        }
        assert!(
            populated >= 1950,
            "unexpectedly low slot occupancy: {populated}"
        );
        // Verify branchless `hash_norm4` matches sequential RFC 7932 UTF-8 folding across boundaries.
        let ref_norm4 = |b0: u8, b1: u8, b2: u8, b3: u8| {
            let (mut b, mut i) = ([b0, b1, b2, b3], 0usize);
            while i < 4 {
                if b[i] < 0xc0 {
                    b[i] = b[i].to_ascii_lowercase();
                    i += 1;
                } else if b[i] < 0xe0 {
                    if i < 3 {
                        b[i + 1] |= 0x20;
                    }
                    i += 2;
                } else {
                    if i < 2 {
                        b[i + 2] |= 0x05;
                    }
                    i += 3;
                }
            }
            u32::from_le_bytes(b).wrapping_mul(DICT_HASH_MUL32)
        };
        let bytes = [
            0u8, 0x20, b'@', b'A', b'M', b'Z', b'[', b'`', b'a', b'z', b'{', 0x7f, 0x80, 0xbf,
            0xc0, 0xdf, 0xe0, 0xff,
        ];
        for b0 in bytes {
            for b1 in bytes {
                for b2 in bytes {
                    for b3 in bytes {
                        let ref_h = ref_norm4(b0, b1, b2, b3);
                        let expected = (
                            (ref_h >> (32 - CSR_BUCKETS_LOG2)) as usize,
                            (ref_h >> 10) as u8,
                        );
                        assert_eq!(hash_norm4(b0, b1, b2, b3), expected);
                    }
                }
            }
        }
    }
}
