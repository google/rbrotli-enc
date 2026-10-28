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

//! RFC 7932 literal context lookup tables, context-mode selection, and JSD histogram clustering.

use crate::constants::MAX_LIT;
use crate::metablock::ContextMode;
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::_mm256_loadu_si256;
#[cfg(target_arch = "x86_64")]
use simd_tokens::{simd_entry, Avx2Token, SimdToken};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct LiteralHistogram {
    pub(crate) data: [u32; MAX_LIT],
    pub(crate) total: u32,
}

impl Default for LiteralHistogram {
    fn default() -> Self {
        Self {
            data: [0; MAX_LIT],
            total: 0,
        }
    }
}

impl LiteralHistogram {
    #[inline]
    pub(crate) fn reset(&mut self) {
        self.data.fill(0);
        self.total = 0;
    }

    #[inline]
    pub(crate) fn merge_from(&mut self, other: &Self) {
        if other.total == 0 {
            return;
        }
        self.total += other.total;
        for (d, &s) in self.data.iter_mut().zip(other.data.iter()) {
            *d += s;
        }
    }
}

macro_rules! lut256 {
    (|$i:ident| $body:expr) => {{
        let mut t = [0u8; 256];
        let mut $i = 0usize;
        while $i < 256 {
            t[$i] = $body;
            $i += 1;
        }
        t
    }};
}

const ZERO_LUT: [u8; 256] = [0u8; 256];
const LSB6_LUT0: [u8; 256] = lut256!(|i| (i as u8) & 0x3F);
const MSB6_LUT0: [u8; 256] = lut256!(|i| (i as u8) >> 2);

#[rustfmt::skip]
const UTF8_LUT0: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    8, 12, 16, 12, 12, 20, 12, 16, 24, 28, 12, 12, 32, 12, 36, 12, 44, 44, 44, 44, 44, 44, 44, 44,
    44, 44, 32, 32, 24, 40, 28, 12, 12, 48, 52, 52, 52, 48, 52, 52, 52, 48, 52, 52, 52, 52, 52, 48,
    52, 52, 52, 52, 52, 48, 52, 52, 52, 52, 52, 24, 12, 28, 12, 12, 12, 56, 60, 60, 60, 56, 60, 60,
    60, 56, 60, 60, 60, 60, 60, 56, 60, 60, 60, 60, 60, 56, 60, 60, 60, 60, 60, 24, 12, 28, 12, 0,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
    2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
];

#[rustfmt::skip]
const UTF8_LUT1: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
    1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
];

#[rustfmt::skip]
const SIGNED_LUT1: [u8; 256] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7,
];

const SIGNED_LUT0: [u8; 256] = lut256!(|i| SIGNED_LUT1[i] << 3);

#[inline(always)]
pub(crate) fn context_luts(mode: ContextMode) -> (&'static [u8; 256], &'static [u8; 256]) {
    match mode {
        ContextMode::LSB6 => (&LSB6_LUT0, &ZERO_LUT),
        ContextMode::MSB6 => (&MSB6_LUT0, &ZERO_LUT),
        ContextMode::UTF8 => (&UTF8_LUT0, &UTF8_LUT1),
        ContextMode::Signed => (&SIGNED_LUT0, &SIGNED_LUT1),
    }
}

#[inline(always)]
pub(crate) fn stream_byte_ctx(
    lut0: &[u8; 256],
    lut1: &[u8; 256],
    data: &[u8],
    pos: usize,
) -> usize {
    let p1 = if pos >= 1 { data[pos - 1] } else { 0 };
    let p2 = if pos >= 2 { data[pos - 2] } else { 0 };
    (lut0[p1 as usize] | lut1[p2 as usize]) as usize
}

/// Computes `out_ctx[k]` for stream positions `start .. start + out_ctx.len()` inside `data`,
/// using `(prev1, prev2) = (0, 0)` at stream offset `0` and `(data[0], 0)` at stream offset `1`.
pub(crate) fn compute_contexts_in_stream(
    mode: ContextMode,
    data: &[u8],
    start: usize,
    out_ctx: &mut [u8],
) {
    assert!(start + out_ctx.len() <= data.len());
    let (lut0, lut1) = context_luts(mode);
    let mut k = 0usize;
    while k < out_ctx.len() && start + k < 2 {
        out_ctx[k] = stream_byte_ctx(lut0, lut1, data, start + k) as u8;
        k += 1;
    }
    if k < out_ctx.len() {
        let base = start + k;
        let (mut p2, mut p1) = (data[base - 2], data[base - 1]);
        for (&b, slot) in data[base..start + out_ctx.len()]
            .iter()
            .zip(&mut out_ctx[k..])
        {
            *slot = lut0[p1 as usize] | lut1[p2 as usize];
            (p2, p1) = (p1, b);
        }
    }
}

#[inline(always)]
fn approx_nlog2_q8_scalar(prob: f32) -> i32 {
    (127 << 8) - ((prob.to_bits() >> 15) as i32)
}

#[inline(always)]
fn jsd_accum_scalar_q8(a: &[u32], b: &[u32], inv_a: f32, inv_b: f32, inv_tot: f32) -> i64 {
    let mut dist_q8 = 0i64;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        if (av | bv) == 0 {
            continue;
        }
        let totv = (av as u64) + (bv as u64);
        let nbits_a = approx_nlog2_q8_scalar((av as f32) * inv_a) as i64;
        let nbits_b = approx_nlog2_q8_scalar((bv as f32) * inv_b) as i64;
        let nbits_tot = approx_nlog2_q8_scalar((totv as f32) * inv_tot) as i64;
        dist_q8 += nbits_tot * (totv as i64) - nbits_a * (av as i64) - nbits_b * (bv as i64);
    }
    dist_q8
}

#[inline]
fn histogram_distance_slice_portable(a: &[u32], total_a: u32, b: &[u32], total_b: u32) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    if total_a == 0 || total_b == 0 {
        return 0;
    }
    let total_ab = (total_a as u64) + (total_b as u64);
    let dist_q8 = jsd_accum_scalar_q8(
        a,
        b,
        1.0f32 / (total_a as f32),
        1.0f32 / (total_b as f32),
        1.0f32 / (total_ab as f32),
    );
    ((dist_q8 >> 8).clamp(0, i32::MAX as i64)) as i32
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
fn hsum_epi32_avx2(mut v: __m256i) -> i32 {
    v = _mm256_add_epi32(v, _mm256_shuffle_epi32::<0b10110001>(v));
    v = _mm256_add_epi32(v, _mm256_shuffle_epi32::<0b01001110>(v));
    v = _mm256_add_epi32(v, _mm256_permute4x64_epi64::<0b01001110>(v));
    _mm256_extract_epi32::<0>(v)
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[target_feature(enable = "avx2")]
fn histogram_distance_slice_avx2(
    a: &[u32],
    total_a: u32,
    b: &[u32],
    total_b: u32,
    _token: impl SimdToken,
) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    if total_a == 0 || total_b == 0 {
        return 0;
    }
    let inv_a_s = 1.0f32 / (total_a as f32);
    let inv_b_s = 1.0f32 / (total_b as f32);
    let inv_tot_s = 1.0f32 / ((total_a + total_b) as f32);
    let inv_a = _mm256_set1_ps(inv_a_s);
    let inv_b = _mm256_set1_ps(inv_b_s);
    let inv_total = _mm256_set1_ps(inv_tot_s);
    let mut total_distance0 = _mm256_setzero_si256();
    let mut total_distance1 = _mm256_setzero_si256();
    let mut total_distance2 = _mm256_setzero_si256();
    let approx_nlog2_q8 = |x| {
        _mm256_sub_epi32(
            _mm256_set1_epi32(127 << 8),
            _mm256_srli_epi32::<15>(_mm256_castps_si256(x)),
        )
    };
    let (a_chunks, a_rem) = a.as_chunks::<8>();
    let (b_chunks, b_rem) = b.as_chunks::<8>();
    for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
        let av = _mm256_loadu_si256(ac);
        let bv = _mm256_loadu_si256(bc);
        let totv = _mm256_add_epi32(av, bv);
        let proba = _mm256_mul_ps(_mm256_cvtepi32_ps(av), inv_a);
        let probb = _mm256_mul_ps(_mm256_cvtepi32_ps(bv), inv_b);
        let probtot = _mm256_mul_ps(_mm256_cvtepi32_ps(totv), inv_total);
        total_distance0 = _mm256_add_epi32(
            _mm256_mullo_epi32(approx_nlog2_q8(probtot), totv),
            total_distance0,
        );
        total_distance1 = _mm256_add_epi32(
            _mm256_mullo_epi32(approx_nlog2_q8(proba), av),
            total_distance1,
        );
        total_distance2 = _mm256_add_epi32(
            _mm256_mullo_epi32(approx_nlog2_q8(probb), bv),
            total_distance2,
        );
    }
    let dist_q8 = hsum_epi32_avx2(_mm256_sub_epi32(
        total_distance0,
        _mm256_add_epi32(total_distance1, total_distance2),
    ))
    .wrapping_add(jsd_accum_scalar_q8(a_rem, b_rem, inv_a_s, inv_b_s, inv_tot_s) as i32);
    (dist_q8 >> 8).max(0)
}

#[inline]
pub(crate) fn histogram_distance(a: &LiteralHistogram, b: &LiteralHistogram) -> i32 {
    histogram_distance_slice(&a.data, a.total, &b.data, b.total)
}

/// Fast JSD bit-cost divergence over arbitrary-length `[u32]` histograms (`Literal`, `IAC`, `Distance`).
#[inline]
pub(crate) fn histogram_distance_slice(a: &[u32], total_a: u32, b: &[u32], total_b: u32) -> i32 {
    // 32-bit Q8 JSD lane sums (`dist_q8 <= 256 * (total_a + total_b)`) fit in signed `i32` without
    // wraparound for `total_a + total_b <= 8_000_000`; larger merged metablocks (e.g. Q12 `16 MiB`)
    // use 64-bit portable accumulation.
    #[cfg(target_arch = "x86_64")]
    if (total_a as u64) + (total_b as u64) <= 8_000_000 {
        if let Some(token) = Avx2Token::try_new() {
            return histogram_distance_slice_avx2(a, total_a, b, total_b, token);
        }
    }
    histogram_distance_slice_portable(a, total_a, b, total_b)
}

/// Asymptotic Miller–Madow / chi-square finite-sample JSD expectation under $P_A = P_B$:
/// $\frac{K_{\text{union}} - 1}{2 \ln 2} \approx \frac{23}{32} (K_{\text{union}} - 1)$ bits.
#[inline]
pub(crate) fn jsd_null_bias(a: &[u32], b: &[u32]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let k_union: u32 = a
        .iter()
        .zip(b)
        .map(|(&ca, &cb)| u32::from((ca | cb) != 0))
        .sum();
    ((k_union.saturating_sub(1) as i32) * 23) >> 5
}

/// Null-bias-corrected empirical Jensen–Shannon divergence (`bits`, `>= 0`) over arbitrary slices.
#[inline]
pub(crate) fn excess_jsd_slice(a: &[u32], tot_a: u32, b: &[u32], tot_b: u32) -> i32 {
    let raw = histogram_distance_slice(a, tot_a, b, tot_b);
    if raw <= 0 {
        0
    } else {
        (raw - jsd_null_bias(a, b)).max(0)
    }
}

fn canonicalize_cmap_with(
    cmap: &mut [u8],
    first_nonempty: Option<usize>,
    mut on_new_id: impl FnMut(usize),
) -> usize {
    if let Some(first_nz) = first_nonempty {
        let first_cl = cmap[first_nz];
        cmap[..first_nz].fill(first_cl);
    }
    let mut remap = [u16::MAX; 256];
    let mut num_distinct = 0usize;
    for entry in cmap.iter_mut() {
        let old = *entry as usize;
        if remap[old] == u16::MAX {
            remap[old] = num_distinct as u16;
            num_distinct += 1;
            on_new_id(old);
        }
        *entry = remap[old] as u8;
    }
    num_distinct
}

/// Propagates `cmap[first_nz]` across leading empty slots `cmap[..first_nz]` (when provided)
/// and renumbers `cmap` IDs in order of first appearance, returning the number of distinct IDs.
pub(crate) fn canonicalize_cmap_ids(cmap: &mut [u8], first_nonempty: Option<usize>) -> usize {
    canonicalize_cmap_with(cmap, first_nonempty, |_| {})
}

/// Rational `[2/2]` Padé approximation of `log2(x)` for positive normal `f32` (`x > 0`),
/// adapted from `jxl-rs` (`fast_math::fast_log2f`). Max error `< 2e-6` bits; zero lookup tables.
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub(crate) fn fast_log2f(x: f32) -> f32 {
    const LOG2F_P: [f32; 3] = [
        -1.8503833400518310e-6,
        1.4287160470083755,
        7.4245873327820566e-1,
    ];
    const LOG2F_Q: [f32; 3] = [
        9.9032814277590719e-1,
        1.0096718572241148,
        1.7409343003366853e-1,
    ];
    let x_bits = x.to_bits() as i32;
    let exp_shifted = x_bits.wrapping_sub(0x3f2aaaab) >> 23;
    let m = f32::from_bits(x_bits.wrapping_sub(exp_shifted << 23) as u32) - 1.0;
    let yp = (LOG2F_P[2] * m + LOG2F_P[1]) * m + LOG2F_P[0];
    let yq = (LOG2F_Q[2] * m + LOG2F_Q[1]) * m + LOG2F_Q[0];
    yp / yq + (exp_shifted as f32)
}

/// Computes `x * log2(x)` in `FP16` fixed-point bits (`65536 == 1.0 bit`, `< 2e-6` bits/sym error).
#[inline(always)]
pub(crate) fn xlog2x_fp16(x: u32) -> u64 {
    if x <= 1 {
        0
    } else {
        ((x as f64) * (fast_log2f(x as f32) as f64) * 65536.0 + 0.5) as u64
    }
}

/// Flat-stride farthest-first histogram clustering with optional online center updates
/// (`ONLINE_MERGE`) and optional first-appearance canonicalization (`CANONICALIZE`).
pub(crate) fn cluster_flat<const ONLINE_MERGE: bool, const CANONICALIZE: bool>(
    hists: &[u32],
    totals: &[u32],
    alpha: usize,
    max_clusters: usize,
    min_gain: i32,
    seed_dist: impl Fn(&[u32], u32, &[u32], u32) -> i32,
    assign_dist: impl Fn(&[u32], u32, &[u32], u32) -> i32,
) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let n = totals.len();
    assert!(n > 0 && hists.len() == n * alpha);
    let max_clusters = max_clusters.clamp(1, 256);
    let mut used = vec![false; n];
    let mut cmap = vec![0u8; n];
    let cap = max_clusters.min(n);
    let mut seeds = Vec::with_capacity(cap * alpha);
    let mut seed_tots = Vec::with_capacity(cap);

    let seed0 = (0..n).max_by_key(|&i| totals[i]).unwrap_or(0);
    seeds.extend_from_slice(&hists[seed0 * alpha..(seed0 + 1) * alpha]);
    seed_tots.push(totals[seed0]);
    used[seed0] = true;
    cmap[seed0] = 0;

    if seed_tots[0] == 0 {
        return (seeds, seed_tots, cmap);
    }

    let mut hd: Vec<i32> = totals
        .iter()
        .map(|&t| if t > 0 { i32::MAX } else { 0 })
        .collect();
    hd[seed0] = 0;

    while seed_tots.len() < max_clusters {
        let last = seed_tots.len() - 1;
        let (newest, newest_tot) = (&seeds[last * alpha..(last + 1) * alpha], seed_tots[last]);
        for i in 0..n {
            if !used[i] && totals[i] > 0 {
                hd[i] = hd[i].min(seed_dist(
                    newest,
                    newest_tot,
                    &hists[i * alpha..(i + 1) * alpha],
                    totals[i],
                ));
            }
        }
        let (best_idx, gain) = hd
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|(_, g)| *g)
            .unwrap();
        if gain < min_gain || used[best_idx] || totals[best_idx] == 0 {
            break;
        }
        cmap[best_idx] = seed_tots.len() as u8;
        seeds.extend_from_slice(&hists[best_idx * alpha..(best_idx + 1) * alpha]);
        seed_tots.push(totals[best_idx]);
        used[best_idx] = true;
        hd[best_idx] = 0;
    }

    let k = seed_tots.len();
    let frozen = (!ONLINE_MERGE).then(|| (seeds.clone(), seed_tots.clone()));
    for i in 0..n {
        if used[i] {
            continue;
        }
        let ti = totals[i];
        if ti == 0 {
            cmap[i] = if i > 0 { cmap[i - 1] } else { 0 };
            continue;
        }
        let hi = &hists[i * alpha..(i + 1) * alpha];
        let best_c = if k == 1 {
            0
        } else {
            let (rh, rt) = match &frozen {
                Some((fh, ft)) => (&fh[..], &ft[..]),
                None => (&seeds[..], &seed_tots[..]),
            };
            (0..k)
                .min_by_key(|&c| assign_dist(&rh[c * alpha..(c + 1) * alpha], rt[c], hi, ti))
                .unwrap_or(0)
        };
        cmap[i] = best_c as u8;
        seed_tots[best_c] += ti;
        for (d, &s) in seeds[best_c * alpha..(best_c + 1) * alpha]
            .iter_mut()
            .zip(hi)
        {
            *d += s;
        }
    }

    if CANONICALIZE {
        let mut rh = Vec::with_capacity(seeds.len());
        let mut rt = Vec::with_capacity(seed_tots.len());
        canonicalize_cmap_with(&mut cmap, totals.iter().position(|&t| t > 0), |old| {
            rh.extend_from_slice(&seeds[old * alpha..(old + 1) * alpha]);
            rt.push(seed_tots[old]);
        });
        (rh, rt, cmap)
    } else {
        (seeds, seed_tots, cmap)
    }
}

#[inline]
pub(crate) fn cluster_flat_jsd(
    hists: &[u32],
    totals: &[u32],
    alpha: usize,
    max_clusters: usize,
    min_gain: i32,
) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    cluster_flat::<true, true>(
        hists,
        totals,
        alpha,
        max_clusters,
        min_gain,
        histogram_distance_slice,
        histogram_distance_slice,
    )
}

/// Farthest-first histogram clustering with nearest-center merge and first-appearance
/// canonical cluster numbering (minimizing `ContextMap` MTF + zero-RLE header bits).
pub(crate) fn cluster_histograms_64(
    histograms: &[LiteralHistogram; 64],
    max_clusters: usize,
    min_histogram_gain: i32,
) -> (Vec<LiteralHistogram>, [u8; 64]) {
    let mut flat = Vec::with_capacity(64 * MAX_LIT);
    let totals: [u32; 64] = std::array::from_fn(|i| {
        flat.extend_from_slice(&histograms[i].data);
        histograms[i].total
    });
    let (cl_h, cl_t, cmap_vec) =
        cluster_flat_jsd(&flat, &totals, MAX_LIT, max_clusters, min_histogram_gain);
    let clusters = cl_h
        .as_chunks::<MAX_LIT>()
        .0
        .iter()
        .zip(cl_t)
        .map(|(&data, total)| LiteralHistogram { data, total })
        .collect();
    let mut cmap = [0u8; 64];
    cmap.copy_from_slice(&cmap_vec);
    (clusters, cmap)
}

/// Deterministic integer Shannon entropy (`H(h) * h.total`, in bits) of a 256-bin histogram.
#[inline]
pub(crate) fn histogram_entropy_bits(h: &LiteralHistogram) -> u64 {
    if h.total <= 1 {
        return 0;
    }
    let mut sum = xlog2x_fp16(h.total);
    for &c in &h.data {
        if c > 1 {
            sum = sum.saturating_sub(xlog2x_fp16(c));
        }
    }
    sum >> 16
}

pub(crate) const ALL_CONTEXT_MODES: [ContextMode; 4] = [
    ContextMode::UTF8,
    ContextMode::Signed,
    ContextMode::MSB6,
    ContextMode::LSB6,
];

/// Scores each `ContextMode` in `modes` (`<= 4`) by empirical conditional entropy (+ `1.5` bits MDL
/// penalty per populated `(ctx, byte)` bin) across up to `MAX_RANGES` (`<= 8`) 1 KiB windows,
/// returning `(mode, bits_fp16)` sorted ascending.
pub(crate) fn rank_sampled_ctx_modes<const MAX_RANGES: usize>(
    modes: &[ContextMode],
    data: &[u8],
    total_len: usize,
    pos_at: impl Fn(usize) -> usize,
) -> [(ContextMode, u64); 4] {
    const WIN: usize = 1024;
    assert!((1..=8).contains(&MAX_RANGES) && !modes.is_empty() && modes.len() <= 4);
    let mut ranges = [(0usize, 0usize); MAX_RANGES];
    let num_ranges = if total_len <= MAX_RANGES * WIN {
        ranges[0] = (0, total_len);
        1
    } else if MAX_RANGES == 1 {
        let mid = (total_len - WIN) / 2;
        ranges[0] = (mid, mid + WIN);
        1
    } else {
        let span = (total_len - WIN) as u64;
        let denom = (MAX_RANGES - 1) as u64;
        for (r, slot) in ranges.iter_mut().enumerate() {
            let s = ((span * (r as u64)) / denom) as usize;
            *slot = (s, s + WIN);
        }
        MAX_RANGES
    };

    let mut counts = [0u16; 64 * MAX_LIT];
    let mut dirty = [0u16; 8 * 1024];
    let mut out = [(ContextMode::UTF8, u64::MAX); 4];
    for (slot_out, &mode) in out.iter_mut().zip(modes.iter()) {
        let mut ctx_totals = [0u32; 64];
        let (lut0, lut1) = context_luts(mode);
        let mut distinct_bins = 0usize;
        for &(s, e) in &ranges[..num_ranges] {
            for i in s..e {
                let pos = pos_at(i);
                let ctx = stream_byte_ctx(lut0, lut1, data, pos);
                let idx = (ctx << 8) | (data[pos] as usize);
                if counts[idx] == 0 {
                    dirty[distinct_bins] = idx as u16;
                    distinct_bins += 1;
                }
                counts[idx] += 1;
                ctx_totals[ctx] += 1;
            }
        }
        // `1.5 * 65536 == 98_304` (`FP16` bits per populated `(ctx, byte)` bin).
        let mut bits_fp16 = (distinct_bins as u64) * 98_304;
        for &tot in &ctx_totals {
            if tot > 1 {
                bits_fp16 += xlog2x_fp16(tot);
            }
        }
        for &idx in &dirty[..distinct_bins] {
            let c = counts[idx as usize] as u32;
            counts[idx as usize] = 0;
            if c > 1 {
                bits_fp16 = bits_fp16.saturating_sub(xlog2x_fp16(c));
            }
        }
        *slot_out = (mode, bits_fp16);
    }
    out[..modes.len()].sort_by_key(|x| x.1);
    out
}

#[inline]
pub(crate) fn best_sampled_ctx_mode<const MAX_RANGES: usize>(
    modes: &[ContextMode],
    data: &[u8],
    total_len: usize,
    pos_at: impl Fn(usize) -> usize,
) -> ContextMode {
    rank_sampled_ctx_modes::<MAX_RANGES>(modes, data, total_len, pos_at)[0].0
}

pub(crate) fn choose_context_mode_modes(data: &[u8], modes: &[ContextMode]) -> ContextMode {
    if data.len() < 64 {
        return ContextMode::UTF8;
    }
    best_sampled_ctx_mode::<4>(modes, data, data.len(), |pos| pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_context_modes_exhaustive_65536_pairs() {
        let mut stream = Vec::with_capacity(256 * 256 * 3);
        for p2 in 0..=255u8 {
            for p1 in 0..=255u8 {
                stream.extend_from_slice(&[p2, p1, p1.wrapping_add(p2)]);
            }
        }
        for mode in ALL_CONTEXT_MODES {
            let (lut0, lut1) = context_luts(mode);
            let expected: Vec<u8> = (0..stream.len())
                .map(|pos| stream_byte_ctx(lut0, lut1, &stream, pos) as u8)
                .collect();
            let mut actual = vec![0u8; stream.len()];
            compute_contexts_in_stream(mode, &stream, 0, &mut actual);
            assert_eq!(actual, expected);
            for start in [0, 1, 2, 3, 17, 63] {
                for len in [0, 1, 2, 15, 16, 17, 31, 32, 33, 79] {
                    let mut sub = vec![0u8; len];
                    compute_contexts_in_stream(mode, &stream, start, &mut sub);
                    assert_eq!(&sub[..], &expected[start..start + len]);
                }
            }
        }
    }

    #[test]
    fn test_histogram_distance_mode_and_log2f() {
        let mut hists = [LiteralHistogram::default(); 64];
        for (i, h) in hists[..32].iter_mut().enumerate() {
            for b in b'a'..=b'z' {
                h.data[b as usize] = 40 + i as u32;
                h.total += 40 + i as u32;
            }
        }
        for h in &mut hists[32..] {
            for b in 0..16usize {
                h.data[b * 16] = 50;
                h.total += 50;
            }
        }
        let d_port = histogram_distance_slice_portable(
            &hists[0].data,
            hists[0].total,
            &hists[40].data,
            hists[40].total,
        );
        assert_eq!(d_port, histogram_distance(&hists[0], &hists[40]));
        assert!(d_port > 1000);

        let (s0, s1) = (&hists[0].data[..79], &hists[40].data[..79]);
        let (t0, t1) = (s0.iter().sum(), s1.iter().sum());
        assert_eq!(
            histogram_distance_slice(s0, t0, s1, t1),
            histogram_distance_slice_portable(s0, t0, s1, t1)
        );

        let (clusters, cmap) = cluster_histograms_64(&hists, 16, 256);
        assert_eq!(clusters.len(), 2);
        assert!(cmap[..32].iter().all(|&c| c == 0) && cmap[32..].iter().all(|&c| c == 1));

        let text =
            b"The quick brown fox jumps over the lazy dog. Function definitions and UTF-8 prose! "
                .repeat(64);
        assert_eq!(
            choose_context_mode_modes(&text, &ALL_CONTEXT_MODES[..2]),
            ContextMode::UTF8
        );
        let utf16: Vec<u8> = (0..2048u16)
            .flat_map(|i| (0x0410u16 + (i % 32)).to_le_bytes())
            .collect();
        assert_eq!(
            choose_context_mode_modes(&utf16, &ALL_CONTEXT_MODES[..2]),
            ContextMode::Signed
        );

        for exp in -20..=24 {
            let base = (2.0f64).powi(exp) as f32;
            for step in 0..64 {
                let x = base * (1.0 + (step as f32) / 64.0);
                assert!((fast_log2f(x) - (x as f64).log2() as f32).abs() < 5e-6);
            }
        }
    }

    #[test]
    fn test_canonicalize_cmap_256_clusters_and_large_jsd_no_overflow() {
        // 256 distinct cluster IDs with repeated 256th ID (`remap[old] == 255`) must not collide
        // with the unvisited sentinel or wrap `num_distinct` past 256.
        let mut cmap: Vec<u8> = (0..=255u8).rev().collect();
        cmap.extend_from_slice(&[0, 128, 255, 0]);
        let n_distinct = canonicalize_cmap_ids(&mut cmap, None);
        assert_eq!(n_distinct, 256);
        for (i, &c) in cmap[..256].iter().enumerate() {
            assert_eq!(c, i as u8);
        }
        assert_eq!(&cmap[256..], &[255, 127, 0, 255]);

        // Disjoint 5M + 5M (`10M > 8.38M` Q8 `i32` limit) and `2.5B + 2.5B > u32::MAX` histograms
        // must produce positive non-overflowing JSD bit distances.
        let (mut a, mut b) = ([0u32; 16], [0u32; 16]);
        a[0] = 5_000_000;
        b[1] = 5_000_000;
        let d_10m = histogram_distance_slice(&a, a[0], &b, b[1]);
        assert!(
            (9_900_000..=10_100_000).contains(&d_10m),
            "unexpected 10M disjoint JSD: {d_10m}"
        );
        a[0] = 2_500_000_000;
        b[1] = 2_500_000_000;
        assert_eq!(histogram_distance_slice(&a, a[0], &b, b[1]), i32::MAX);

        // `MAX_RANGES == 1` with `total_len > 1024` must not divide by zero.
        let prose =
            b"ContextMode single-window sampling test payload for MAX_RANGES == 1. ".repeat(64);
        let ranked =
            rank_sampled_ctx_modes::<1>(&ALL_CONTEXT_MODES[..2], &prose, prose.len(), |i| i);
        assert!(ranked[0].1 < u64::MAX);
    }
}
