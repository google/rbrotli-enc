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
    if fs::read(&header_path).ok().as_deref() != Some(buf.as_slice()) {
        fs::write(&header_path, &buf).expect("Unable to update rbrotli_enc.h");
    }
}
