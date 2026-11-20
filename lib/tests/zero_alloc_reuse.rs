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

use rbrotli_enc_lib::Encoder;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    mem::MaybeUninit,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct CountingAlloc;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Delegates directly to `std::alloc::System` while counting active allocations.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: Caller upholds `GlobalAlloc::alloc` contract; forwards directly to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: Caller upholds `GlobalAlloc::alloc_zeroed` contract; forwards to `System`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Caller upholds `GlobalAlloc::dealloc` contract; forwards to `System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: Caller upholds `GlobalAlloc::realloc` contract; forwards to `System`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn make_payload(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let phrase = b"Structured telemetry record status=OK region=us-central1 shard=";
    while out.len() < len {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.extend_from_slice(phrase);
        out.push(b'0' + ((rng >> 56) as u8 % 10));
        out.push(b'\n');
        out.extend_from_slice(b"{\"id\":");
        out.extend_from_slice(&[((rng >> 48) as u8 & 7) + b'a'; 8]);
        out.extend_from_slice(b",\"ok\":true}\n");
    }
    out.truncate(len);
    out
}

fn measure_allocs(f: impl FnOnce()) -> usize {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    f();
    COUNTING.store(false, Ordering::SeqCst);
    ALLOC_COUNT.load(Ordering::SeqCst)
}

#[test]
fn test_zero_alloc_steady_state_reuse() {
    let p512 = make_payload(512);
    let p1k = make_payload(1024);
    let p16k = make_payload(16 * 1024);
    let p64k = make_payload(64 * 1024);
    let inputs: [&[u8]; 4] = [&p512, &p1k, &p16k, &p64k];
    let qualities: &[u32] = &[9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    let lgwins: &[u32] = &[16, 22, 24];

    let max_out = Encoder::max_required_size_for(9, 64 * 1024);
    let mut out_buf = vec![MaybeUninit::uninit(); max_out];

    // Warm up a single reusable `Encoder` across all qualities, window sizes, and payload tiers.
    let mut enc = Encoder::with_lgwin(9, 24);
    for &q in qualities {
        enc.set_quality(q);
        for &lw in lgwins {
            enc.set_lgwin(lw);
            for &inp in &inputs {
                assert!(enc.compress(inp, None).is_some());
                assert!(enc.compress(inp, Some(&mut out_buf)).is_some());
            }
        }
    }

    // Warm up thread-local one-shot encoder across the same matrix.
    for &q in qualities {
        for &lw in lgwins {
            for &inp in &inputs {
                assert!(Encoder::compress_into_tls(q, lw, inp, None, &mut out_buf).is_some());
            }
        }
    }

    // Steady-state warm reuse on `Encoder` (`out_buf = None` and caller buffer) must not allocate.
    for &q in qualities {
        enc.set_quality(q);
        for &lw in lgwins {
            enc.set_lgwin(lw);
            for &inp in &inputs {
                let n_none = measure_allocs(|| {
                    let out = enc.compress(inp, None).unwrap();
                    assert!(!out.is_empty());
                });
                assert_eq!(
                    n_none,
                    0,
                    "Encoder::compress(None) allocated at q={q}, lw={lw}, len={}",
                    inp.len()
                );

                let n_buf = measure_allocs(|| {
                    let out = enc.compress(inp, Some(&mut out_buf)).unwrap();
                    assert!(!out.is_empty());
                });
                assert_eq!(
                    n_buf,
                    0,
                    "Encoder::compress(Some) allocated at q={q}, lw={lw}, len={}",
                    inp.len()
                );
            }
        }
    }

    // Steady-state warm reuse on `Encoder::compress_into_tls` must also perform zero allocations.
    for &q in qualities {
        for &lw in lgwins {
            for &inp in &inputs {
                let n_tls = measure_allocs(|| {
                    let out = Encoder::compress_into_tls(q, lw, inp, None, &mut out_buf).unwrap();
                    assert!(!out.is_empty());
                });
                assert_eq!(
                    n_tls,
                    0,
                    "compress_into_tls allocated at q={q}, lw={lw}, len={}",
                    inp.len()
                );
            }
        }
    }
}
