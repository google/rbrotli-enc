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

use clap::Parser;
use color_eyre::eyre::Result;
use rbrotli_enc_lib::Encoder;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Sets a custom config file
    #[arg(value_name = "INPUT_FILE")]
    input: PathBuf,

    #[arg(value_name = "OUTPUT_FILE")]
    output: PathBuf,

    #[clap(short, default_value = "1")]
    num_reps: usize,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    let input = fs::read(args.input)?;

    let mut encoder = Encoder::new();

    let start = Instant::now();
    let mut compressed = encoder.compress(&input).unwrap();
    for _ in 1..args.num_reps {
        compressed = encoder.compress(&input).unwrap();
    }
    let stop = Instant::now();

    eprintln!("Compressed {} to {} bytes", input.len(), compressed.len());
    let secs = (stop - start).as_secs_f64();
    eprintln!(
        "{:8.5} seconds, {} reps, {:5.3} MB/s",
        secs,
        args.num_reps,
        (input.len() * args.num_reps) as f64 / secs * 1e-6
    );

    fs::write(args.output, compressed)?;

    Ok(())
}
