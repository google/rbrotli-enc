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

//! Unified RFC 7932 built-in static dictionary (`kBrotliDictionary` + all 121 transforms).

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
const fn hash_norm4(b0: u8, b1: u8, b2: u8, b3: u8) -> (usize, u8) {
    let h = u32::from_le_bytes([b0.to_ascii_lowercase(), b1, b2, b3]).wrapping_mul(DICT_HASH_MUL32);
    ((h >> (32 - CSR_BUCKETS_LOG2)) as usize, (h >> 10) as u8)
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
        if let Some(&b0) = rest
            .first()
            .filter(|&&b| b < 128 && ((self.4 >> b) & 1) != 0)
        {
            (
                &SUFFIX_RULES[self.1 as usize..end as usize],
                b0,
                load_prefix_u64(rest),
            )
        } else {
            (&[], 0, 0)
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
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StaticDictMatch {
    pub(crate) base_len: u32,
    pub(crate) out_len: u32,
    pub(crate) raw_dist: u32,
    pub(crate) gain: i32,
}

#[inline(always)]
fn eq_tail_from_1(sub: &[u8], word: &[u8], n: usize) -> bool {
    let eq4 = |o: usize| sub[o..o + 4].first_chunk::<4>() == word[o..o + 4].first_chunk::<4>();
    let eq8 = |o: usize| sub[o..o + 8].first_chunk::<8>() == word[o..o + 8].first_chunk::<8>();
    if n == 4 {
        sub[1..4].first_chunk::<3>() == word[1..4].first_chunk::<3>()
    } else if n <= 8 {
        eq4(1) && eq4(n - 4)
    } else {
        eq8(1) && eq8(n - 8) && (n <= 16 || eq8(9))
    }
}

#[inline(always)]
fn can_len4_reach_min_out(b4: Option<u8>, needed_suffix: usize) -> bool {
    match (needed_suffix, b4) {
        (0, _) => true,
        (_, None) => false,
        (2.., Some(c)) => matches!(c, b' ' | b',' | b'.' | b'"' | b'=' | b'\n' | b'e' | b'l'),
        (1, Some(c)) => matches!(
            c,
            b' ' | b',' | b'.' | b'"' | b'=' | b'\n' | b'e' | b'l' | b']' | b'\'' | b':' | b'('
        ),
    }
}

type LookupArgs = (u64, u32, u32, i32, i32, i32);

struct ProbeCtx {
    cfg: LookupArgs,
    min_out: u32,
    best: Option<StaticDictMatch>,
}

impl ProbeCtx {
    #[inline(always)]
    fn eval_entry(&mut self, entry: u32, sub: &[u8], pid_pos: usize, plen: usize) {
        let base_len = ((entry >> 11) & 0x1F) as usize;
        if sub.len() < base_len {
            return;
        }
        let word_idx = (entry & 0x7FF) as usize;
        let word = dict_word(base_len, word_idx);
        let Some(op_idx) = eq_tail_from_1(sub, word, base_len)
            .then(|| match_op01(sub[0], word[0]))
            .flatten()
        else {
            return;
        };
        let set = &SUFFIX_SETS[pid_pos][op_idx];
        if set.0 == u8::MAX {
            return;
        }
        let (base_dist, max_raw_dist, _, lit_mult, cmd_off, min_gain) = self.cfg;
        let (tid, slen0) = set.upgrade_fast(&sub[base_len..]);
        let cc = crate::map_to_sym::copy_code_only(base_len as u32);
        let copy_extra_q5 = (crate::map_to_sym::COPY_EXTRA[cc] << 5) as i32;
        let cands = [(tid, slen0), (set.0 as u32, 0)];
        for &(cand_tid, slen) in &cands[..if slen0 > 0 { 2 } else { 1 }] {
            let out_len = (plen + base_len + slen) as u32;
            let raw64 = base_dist + encode_raw_dist(word_idx, base_len, cand_tid as usize) as u64;
            if out_len < self.min_out || raw64 > max_raw_dist as u64 {
                continue;
            }
            let raw_dist = raw64 as u32;
            let dist_bits_q5 = ((31 ^ (raw_dist | 1).leading_zeros()) << 5) as i32;
            let gain = lit_mult * (out_len as i32) - (cmd_off + dist_bits_q5 + copy_extra_q5);
            if gain >= min_gain && self.best.is_none_or(|b| gain > b.gain) {
                self.best = Some(StaticDictMatch {
                    base_len: base_len as u32,
                    out_len,
                    raw_dist,
                    gain,
                });
            }
        }
    }

    #[inline(always)]
    fn probe_at(&mut self, sub: &[u8], pid_pos: usize, plen: usize) {
        let Some(first4) = sub.first_chunk::<4>() else {
            return;
        };
        let (b_idx, tag_k4) = hash_norm4(first4[0], first4[1], first4[2], first4[3]);
        let opt_b4 = sub.get(4).copied();
        let tag5 = opt_b4.map_or(u16::MAX, |b4| ((b4 as u16) << 8) | (tag_k4 as u16));
        for &entry in BASE_CSR.bucket(b_idx) {
            let base_len = ((entry >> 11) & 0x1F) as usize;
            if base_len >= 5 {
                if (entry >> 16) as u16 == tag5 {
                    self.eval_entry(entry, sub, pid_pos, plen);
                }
            } else if (entry >> 16) as u8 == tag_k4 {
                let eff_min = self
                    .best
                    .map_or(self.min_out, |b| self.min_out.max(b.out_len + 1));
                if can_len4_reach_min_out(opt_b4, (eff_min as usize).saturating_sub(4 + plen)) {
                    self.eval_entry(entry, sub, pid_pos, plen);
                }
            }
        }
    }
}

#[inline(always)]
fn lookup_impl<const FAST_ONLY_EMPTY_PREFIX: bool>(
    slice: &[u8],
    cfg: LookupArgs,
) -> Option<StaticDictMatch> {
    let (base_dist, max_raw_dist, min_out_len, lit_mult, cmd_off, min_gain) = cfg;
    if slice.len() < 4 || base_dist > max_raw_dist as u64 {
        return None;
    }
    let min_dist_q5 = ((31 ^ ((base_dist as u32) | 1).leading_zeros()) << 5) as i32;
    let min_needed = min_gain + cmd_off + min_dist_q5;
    let min_out =
        min_out_len.max((4..=8).find(|&n| n * lit_mult >= min_needed).unwrap_or(9) as u32);
    if slice.len() < min_out as usize {
        return None;
    }
    let mut ctx = ProbeCtx {
        cfg,
        min_out,
        best: None,
    };
    ctx.probe_at(slice, 0, 0);
    if !FAST_ONLY_EMPTY_PREFIX && slice[0] == b' ' && slice.len() >= 5 {
        ctx.probe_at(&slice[1..], 1, 1);
        if slice.len() >= 9 && slice.first_chunk::<5>() == Some(b" the ") {
            ctx.probe_at(&slice[5..], 2, 5);
        }
    }
    ctx.best
}

macro_rules! def_lookup {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[inline(never)]
        pub(crate) fn $name<const FAST: bool>(
            s: &[u8], bd: u64, mrd: u32, mo: u32, lm: i32, co: i32, mg: i32,
        ) -> Option<StaticDictMatch> {
            lookup_impl::<FAST>(s, (bd, mrd, mo, lm, co, mg))
        }
    };
}
def_lookup!(lookup_with_mode);
def_lookup!(
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    lookup_avx2
);

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

const EXT_BUCKETS_LOG2: usize = 15;
const EXT_BUCKETS: usize = 1 << EXT_BUCKETS_LOG2;

#[inline(always)]
fn hash4_raw(k4: u32) -> usize {
    (k4.wrapping_mul(DICT_HASH_MUL32) >> (32 - EXT_BUCKETS_LOG2)) as usize
}

struct ExtCsr {
    offsets: Box<[u32]>,
    entries: Box<[u64]>,
}

impl ExtCsr {
    fn from_sorted(sorted: Vec<(u16, u64)>) -> Self {
        let mut offsets = vec![0u32; EXT_BUCKETS + 1];
        let mut entries = Vec::with_capacity(sorted.len());
        for (h, e) in sorted {
            offsets[h as usize + 1] += 1;
            entries.push(e);
        }
        for b in 0..EXT_BUCKETS {
            offsets[b + 1] += offsets[b];
        }
        Self {
            offsets: offsets.into_boxed_slice(),
            entries: entries.into_boxed_slice(),
        }
    }

    #[inline(always)]
    fn bucket(&self, h: usize) -> &[u64] {
        &self.entries[self.offsets[h] as usize..self.offsets[h + 1] as usize]
    }
}

struct Dict121Ext {
    csr: ExtCsr,
    len4_stem3: [(u32, u16); 1024],
    prefixes: [&'static [u8]; 9],
}

fn dict121_ext() -> &'static Dict121Ext {
    static EXT: OnceLock<Dict121Ext> = OnceLock::new();
    EXT.get_or_init(|| {
        let k4_at = |s: &[u8]| u32::from_le_bytes(*s.first_chunk::<4>().unwrap());
        let mut items = Vec::with_capacity(96_000);
        for (tid, &(_, op, _)) in RAW_TRANSFORMS.iter().enumerate() {
            if op < 12 {
                continue;
            }
            let skip = (op - 11) as usize;
            for base_len in (skip + 4)..=24usize {
                let out_len = base_len - skip;
                for word_idx in 0..(1usize << SIZE_BITS_BY_LEN[base_len]) {
                    let off = OFFSETS_BY_LEN[base_len] + word_idx * base_len + skip;
                    let k4 = k4_at(&RFC7932_DICT[off..]);
                    let pk = (off as u64)
                        | ((word_idx as u64) << 20)
                        | ((base_len as u64) << 32)
                        | ((out_len as u64) << 40)
                        | ((tid as u64) << 48);
                    items.push((hash4_raw(k4) as u16, pk));
                }
            }
        }
        let omit_sub =
            |e: u64| &RFC7932_DICT[(e & 0xF_FFFF) as usize..][..((e >> 40) & 0xFF) as usize];
        items.sort_unstable_by(|&(ha, a), &(hb, b)| {
            ha.cmp(&hb)
                .then_with(|| ((a >> 32) as u8).cmp(&((b >> 32) as u8)))
                .then_with(|| omit_sub(a).cmp(omit_sub(b)))
                .then_with(|| ((a >> 48) as u8).cmp(&((b >> 48) as u8)))
                .then_with(|| ((a >> 20) & 0xFFF).cmp(&((b >> 20) & 0xFFF)))
        });
        items.dedup_by(|&mut (hl, l), &mut (hk, k)| {
            hl == hk && ((l >> 32) & 0xFFFF) == ((k >> 32) & 0xFFFF) && omit_sub(l) == omit_sub(k)
        });

        let mut buf = [0u8; 24];
        for base_len in (4..=24usize).rev() {
            for word_idx in 0..(1usize << SIZE_BITS_BY_LEN[base_len]) {
                let word = dict_word(base_len, word_idx);
                for (op_idx, upper_all) in [(1u64, false), (2u64, true)] {
                    if !upper_all && word[0] < 0xc0 {
                        continue;
                    }
                    apply_upper_rfc7932(&mut buf[..base_len], word, upper_all);
                    if &buf[..base_len] != word {
                        let k4 = k4_at(&buf);
                        let pk = (1u64 << 63)
                            | (k4 as u64)
                            | ((word_idx as u64) << 32)
                            | ((base_len as u64) << 48)
                            | (op_idx << 56);
                        items.push((hash4_raw(k4) as u16, pk));
                    }
                }
            }
        }
        items.sort_by_key(|&(h, _)| h);

        let mut len4_stem3: [(u32, u16); 1024] = std::array::from_fn(|wi| {
            let w = dict_word(4, wi);
            (u32::from_le_bytes([w[0], w[1], w[2], 0]), wi as u16)
        });
        len4_stem3.sort_unstable();

        Dict121Ext {
            csr: ExtCsr::from_sorted(items),
            len4_stem3,
            prefixes: std::array::from_fn(|i| cstr_slice(DISTINCT_PREFIX_IDS[i])),
        }
    })
}

pub(crate) fn warm_dict121() {
    let _ = dict121_ext();
}

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
    let ext = dict121_ext();
    let mut emit = |base_len: usize, word_idx: u16, tid: u8, out_len: usize| {
        let raw = base_dist + encode_raw_dist(word_idx as usize, base_len, tid as usize) as u64;
        if raw <= max_raw {
            on_match(out_len as u32, base_len as u32, raw as u32);
        }
    };
    let emit_suffixes = |emit: &mut dyn FnMut(usize, u16, u8, usize),
                         (pid_pos, op_idx, plen): (usize, usize, usize),
                         wlen: usize,
                         word_idx: u16,
                         rem: &[u8]| {
        let sset = &SUFFIX_SETS[pid_pos][op_idx];
        let base_out = plen + wlen;
        if base_out >= min_out && sset.0 != u8::MAX {
            emit(wlen, word_idx, sset.0, base_out);
        }
        let (rules, b0, next8) = sset.rules(&rem[wlen..], sset.3);
        for &r in rules {
            let out_l = base_out + (r.2 as usize);
            if (r.0 as u8) == b0 && out_l >= min_out && r.matches(next8) {
                emit(wlen, word_idx, r.1, out_l);
            }
        }
    };

    let (mut active_prefixes, mut num_prefixes) = ([(0usize, 0usize); 3], 1usize);
    if matches!(slice[0], b' ' | b's' | b',' | b'e' | b'.' | 0xc2) {
        for (pid_pos, &pfx) in ext.prefixes.iter().enumerate().skip(1) {
            if pfx[0] == slice[0] && slice.starts_with(pfx) {
                active_prefixes[num_prefixes] = (pid_pos, pfx.len());
                num_prefixes += 1;
            }
        }
    }

    let mut upper_buf = [0u8; 24];
    for &(pid_pos, plen) in &active_prefixes[..num_prefixes] {
        let rem = &slice[plen..];
        let Some(first4) = rem.first_chunk::<4>() else {
            continue;
        };

        let (b_idx, tag_k4) = hash_norm4(first4[0], first4[1], first4[2], first4[3]);
        for &entry in BASE_CSR.bucket(b_idx) {
            if (entry >> 16) as u8 != tag_k4 {
                continue;
            }
            let (wlen, word_idx) = (((entry >> 11) & 0x1F) as usize, (entry & 0x7FF) as u16);
            let word = dict_word(wlen, word_idx as usize);
            let Some(op_idx) = (word[1..4] == first4[1..4])
                .then(|| match_op01(first4[0], word[0]))
                .flatten()
            else {
                continue;
            };
            let (max_cmp, mut lcp) = (wlen.min(rem.len()), 4usize);
            while lcp < max_cmp && word[lcp] == rem[lcp] {
                lcp += 1;
            }
            if lcp == wlen {
                emit_suffixes(&mut emit, (pid_pos, op_idx, plen), wlen, word_idx, rem);
            }
            if pid_pos == 0 && op_idx == 0 {
                for m in wlen.saturating_sub(9).max(min_out)..=lcp.min(wlen - 1) {
                    emit(wlen, word_idx, OMIT_LAST_TIDS[wlen - m], m);
                }
                if wlen >= 5
                    && lcp >= wlen - 1
                    && wlen + 3 >= min_out
                    && rem[wlen - 1..].starts_with(b"ing ")
                {
                    emit(wlen, word_idx, 49, wlen + 3);
                }
            }
        }

        if pid_pos == 0 && rem.len() >= 7 && min_out <= 7 && &rem[3..7] == b"ing " {
            let k3 = u32::from_le_bytes([rem[0], rem[1], rem[2], 0]);
            let idx = ext.len4_stem3.partition_point(|&(k, _)| k < k3);
            if idx < ext.len4_stem3.len() && ext.len4_stem3[idx].0 == k3 {
                emit(4, ext.len4_stem3[idx].1, 49, 7);
            }
        }
        let raw_k4 = u32::from_le_bytes(*first4);
        let want_up = SUFFIX_SETS[pid_pos][1].has_any() || SUFFIX_SETS[pid_pos][2].has_any();
        if pid_pos == 0 || want_up {
            for &e in ext.csr.bucket(hash4_raw(raw_k4)) {
                if (e >> 63) != 0 {
                    let (wlen, op_idx) = (((e >> 48) & 0xFF) as usize, ((e >> 56) & 3) as usize);
                    if want_up
                        && (e as u32) == raw_k4
                        && rem.len() >= wlen
                        && SUFFIX_SETS[pid_pos][op_idx].has_any()
                    {
                        let widx = (e >> 32) as u16;
                        apply_upper_rfc7932(
                            &mut upper_buf[..wlen],
                            dict_word(wlen, widx as usize),
                            op_idx == 2,
                        );
                        if upper_buf[4..wlen] == rem[4..wlen] {
                            emit_suffixes(&mut emit, (pid_pos, op_idx, plen), wlen, widx, rem);
                        }
                    }
                } else if pid_pos == 0 {
                    let (off, olen) = ((e & 0xF_FFFF) as usize, ((e >> 40) & 0xFF) as usize);
                    if (min_out..=rem.len()).contains(&olen)
                        && RFC7932_DICT[off..off + olen] == rem[..olen]
                    {
                        emit(
                            ((e >> 32) & 0xFF) as usize,
                            ((e >> 20) & 0xFFF) as u16,
                            (e >> 48) as u8,
                            olen,
                        );
                    }
                }
            }
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
    fn test_all_121_transforms_discovered_and_verified() {
        for word_idx in [0usize, 10] {
            let base_word = dict_word(14, word_idx);
            for tid in 0..121i32 {
                let mut buf = [0u8; 64];
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
            lookup_with_mode::<false>(s, b, STD_MAX_RAW_STATIC_DIST, 4, 168, 352, 48)
        };
        for &phrase in phrases {
            let m = lk(phrase, 1).unwrap();
            verify_word(phrase, m.out_len, m.base_len, m.raw_dist - 1);
            if phrase[0] != b' ' {
                let mf =
                    lookup_with_mode::<true>(phrase, 1, STD_MAX_RAW_STATIC_DIST, 4, 168, 352, 48)
                        .unwrap();
                assert_eq!((mf.out_len, mf.raw_dist), (m.out_len, m.raw_dist));
            }
        }
        assert!(lk(b"time", 100_000).is_none());
        let m = lk(b"information,x", 100_000).unwrap();
        assert_eq!(m.out_len, 12);
        verify_word(
            b"information,x",
            m.out_len,
            m.base_len,
            m.raw_dist - 100_000,
        );
    }
}
