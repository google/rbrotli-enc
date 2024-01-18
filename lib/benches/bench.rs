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

use std::{fs, time::Duration};

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion, Throughput,
};
use rbrotli_enc_lib::Encoder;

fn bench_impl(c: &mut BenchmarkGroup<WallTime>, files: &[&str]) {
    let mut encoder = Encoder::new();
    for file in files {
        let f = fs::read(&format!("data/{}", file)).unwrap();
        c.throughput(Throughput::Bytes(f.len() as u64));
        c.bench_with_input(file.to_owned(), &f, |b, f| {
            b.iter(|| encoder.compress(f).unwrap())
        });
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    {
        let mut large_files = c.benchmark_group("large_files");
        bench_impl(
            large_files
                .sample_size(20)
                .sampling_mode(criterion::SamplingMode::Flat)
                .measurement_time(Duration::from_secs(120)),
            &["enwik8", "large-js-corpus.js"],
        );
    }
    {
        let mut small_files = c.benchmark_group("small_files");
        bench_impl(
            small_files
                .sample_size(1000)
                .measurement_time(Duration::from_secs(60)),
            &[
                "brotlidump.py",
                "jquery-3.7.1.min.js",
                "bootstrap.min.css",
                "apple_home.html",
            ],
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
