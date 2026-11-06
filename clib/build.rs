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

extern crate cbindgen;

use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let header_path = crate_dir.join("rbrotli_enc.h");
    let mut buf = Vec::new();
    cbindgen::generate(&crate_dir)
        .expect("Unable to generate bindings")
        .write(&mut buf);
    let existing = fs::read(&header_path).ok();
    if existing.as_deref() != Some(buf.as_slice()) {
        if let Err(err) = fs::write(&header_path, &buf) {
            if existing.is_some() {
                println!(
                    "cargo:warning=Skipping rbrotli_enc.h update on read-only source tree: {err}"
                );
            } else {
                panic!("Unable to create rbrotli_enc.h: {err}");
            }
        }
    }
}
