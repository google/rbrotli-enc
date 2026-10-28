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
#[cfg(feature = "color-eyre")]
use color_eyre::eyre::{bail, Result};
#[cfg(not(feature = "color-eyre"))]
use eyre::{bail, Result};
use rbrotli_enc_lib::{Encoder, PreparedDictionary, StreamingWriter};
use std::{
    fs,
    io::{self, Write},
    mem::MaybeUninit,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input file to compress ('-' for stdin)
    #[arg(value_name = "INPUT_FILE")]
    input: PathBuf,

    /// Output file for compressed Brotli stream ('-' for stdout)
    #[arg(value_name = "OUTPUT_FILE")]
    output: PathBuf,

    #[clap(short, long, default_value = "1")]
    num_reps: usize,

    #[clap(
        short,
        long,
        default_value = "5",
        value_parser = clap::value_parser!(u32).range(0..=12)
    )]
    quality: u32,

    #[clap(
        short = 'w',
        long,
        default_value = "24",
        value_parser = clap::value_parser!(u32).range(10..=30)
    )]
    lgwin: u32,

    /// Number of worker threads for Q10..=Q12 (1 = single-threaded, 0 = available parallelism)
    #[clap(short = 'j', long = "threads", default_value = "1")]
    threads: usize,

    /// Optional raw LZ77 prefix dictionary file
    #[clap(short = 'd', long)]
    dictionary: Option<PathBuf>,

    /// Compress incrementally via StreamingWriter (automatic when INPUT_FILE or OUTPUT_FILE is '-')
    #[clap(short = 's', long = "stream")]
    stream: bool,
}

fn is_same_regular_file(a: &Path, b: &Path) -> bool {
    let (Ok(ma), Ok(mb)) = (fs::metadata(a), fs::metadata(b)) else {
        return false;
    };
    if !ma.is_file() || !mb.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ma.dev() == mb.dev() && ma.ino() == mb.ino()
    }
    #[cfg(not(unix))]
    {
        fs::canonicalize(a)
            .ok()
            .zip(fs::canonicalize(b).ok())
            .is_some_and(|(ca, cb)| ca == cb)
    }
}

fn run_cli(args: Args) -> Result<()> {
    if args.num_reps == 0 {
        bail!("--num-reps must be at least 1");
    }
    let use_stdin = args.input.as_os_str() == "-";
    let use_stdout = args.output.as_os_str() == "-";
    if !use_stdin && !use_stdout && is_same_regular_file(&args.input, &args.output) {
        bail!(
            "Refusing to overwrite input file '{}' in place",
            args.input.display()
        );
    }

    let in_meta = if use_stdin {
        None
    } else {
        let m = fs::metadata(&args.input)?;
        if m.is_dir() {
            bail!("Input path '{}' is a directory", args.input.display());
        }
        Some(m)
    };
    let use_stream =
        args.stream || use_stdin || use_stdout || in_meta.as_ref().is_some_and(|m| !m.is_file());
    if use_stream && args.num_reps > 1 {
        bail!("--num-reps > 1 is only supported for one-shot regular-file compression");
    }

    let prepared_dict = args
        .dictionary
        .as_ref()
        .map(|p| fs::read(p).map(|b| PreparedDictionary::with_lgwin(&b, args.quality, args.lgwin)))
        .transpose()?;

    let (in_bytes, out_bytes, reps, secs) = if use_stream {
        let reader: Box<dyn io::Read> = if use_stdin {
            Box::new(io::stdin().lock())
        } else {
            Box::new(fs::File::open(&args.input)?)
        };
        // Only remove `args.output` on mid-stream error if `File::create` created a brand-new
        // file (never delete a pre-existing regular file, symlink, FIFO, or device node).
        let created_new_output = !use_stdout && fs::symlink_metadata(&args.output).is_err();
        let raw_writer: Box<dyn Write> = if use_stdout {
            Box::new(io::BufWriter::new(io::stdout().lock()))
        } else {
            Box::new(io::BufWriter::new(fs::File::create(&args.output)?))
        };
        let mut enc_writer =
            StreamingWriter::with_threads(raw_writer, args.quality, args.lgwin, args.threads);
        if let Some(ref dict) = prepared_dict {
            enc_writer.set_dictionary(dict)?;
        }
        let start = Instant::now();
        let copy_res = (|| -> io::Result<u64> {
            let n = io::copy(
                &mut io::BufReader::with_capacity(65_536, reader),
                &mut enc_writer,
            )?;
            enc_writer.try_finish()?;
            enc_writer.get_mut().flush()?;
            Ok(n)
        })();
        let in_bytes = match copy_res {
            Ok(n) => n,
            Err(err) => {
                drop(enc_writer.abandon());
                if created_new_output {
                    let _ = fs::remove_file(&args.output);
                }
                return Err(err.into());
            }
        };
        let secs = start.elapsed().as_secs_f64();
        (in_bytes, enc_writer.total_out(), 1, secs)
    } else {
        let input = fs::read(&args.input)?;
        let mut encoder = Encoder::with_threads(args.quality, args.lgwin, args.threads);
        let max_req = encoder.max_required_size(input.len());
        if max_req > isize::MAX as usize {
            bail!(
                "Input size {} exceeds one-shot encoder buffer limit; re-run with --stream",
                input.len()
            );
        }
        let mut outbuf = vec![MaybeUninit::uninit(); max_req];
        let reps = args.num_reps;
        let start = Instant::now();
        let mut compressed: &[u8] = &[];
        for _ in 0..reps {
            let opt = match prepared_dict {
                Some(ref dict) => encoder.compress_with_dict(&input, dict, Some(&mut outbuf)),
                None => encoder.compress(&input, Some(&mut outbuf)),
            };
            let Some(c) = opt else {
                bail!(
                    "One-shot compression failed for {} bytes; re-run with --stream",
                    input.len()
                );
            };
            compressed = c;
        }
        let secs = start.elapsed().as_secs_f64();
        let out_len = compressed.len() as u64;
        fs::write(&args.output, compressed)?;
        (input.len() as u64, out_len, reps, secs)
    };

    let mb_per_s = if secs > 0.0 {
        (in_bytes as f64 * reps as f64) / secs * 1e-6
    } else {
        0.0
    };
    eprintln!("Compressed {in_bytes} to {out_bytes} bytes");
    eprintln!("{secs:8.5} seconds, {reps} reps, {mb_per_s:5.3} MB/s");
    Ok(())
}

fn main() -> Result<()> {
    #[cfg(feature = "color-eyre")]
    color_eyre::install()?;
    run_cli(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_arg_ranges_and_same_file_guard() {
        assert!(Args::try_parse_from(["rbrotli-enc", "in", "out", "-q", "12", "-w", "30"]).is_ok());
        assert!(Args::try_parse_from(["rbrotli-enc", "in", "out", "-q", "0", "-w", "10"]).is_ok());
        assert!(Args::try_parse_from(["rbrotli-enc", "in", "out", "-q", "13"]).is_err());
        assert!(Args::try_parse_from(["rbrotli-enc", "in", "out", "-w", "9"]).is_err());
        assert!(Args::try_parse_from(["rbrotli-enc", "in", "out", "-w", "31"]).is_err());

        let tmp = std::env::temp_dir().join(format!("rbrotli_cli_same_{}", std::process::id()));
        fs::write(&tmp, b"payload").unwrap();
        assert!(is_same_regular_file(&tmp, &tmp));
        let _ = fs::remove_file(&tmp);
        assert!(!is_same_regular_file(&tmp, &tmp));
    }

    #[test]
    fn test_cli_preserves_existing_output_on_invalid_input_and_rejects_bad_reps() {
        let dir = std::env::temp_dir().join(format!("rbrotli_cli_guard_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let existing_out = dir.join("important_existing.br");
        fs::write(&existing_out, b"DO_NOT_DELETE_ME").unwrap();

        // Streaming compression with a directory as INPUT_FILE must fail BEFORE touching `existing_out`.
        let err_dir = run_cli(
            Args::try_parse_from([
                "rbrotli-enc",
                "-s",
                dir.to_str().unwrap(),
                existing_out.to_str().unwrap(),
            ])
            .unwrap(),
        );
        assert!(err_dir.is_err());
        assert_eq!(fs::read(&existing_out).unwrap(), b"DO_NOT_DELETE_ME");

        // `--num-reps 0` and `--stream --num-reps 2` must return clean errors.
        let valid_in = dir.join("in.txt");
        let fresh_out = dir.join("out.br");
        fs::write(&valid_in, b"valid cli test input").unwrap();
        assert!(run_cli(
            Args::try_parse_from([
                "rbrotli-enc",
                "-n",
                "0",
                valid_in.to_str().unwrap(),
                fresh_out.to_str().unwrap(),
            ])
            .unwrap(),
        )
        .is_err());
        assert!(run_cli(
            Args::try_parse_from([
                "rbrotli-enc",
                "-s",
                "-n",
                "2",
                valid_in.to_str().unwrap(),
                fresh_out.to_str().unwrap(),
            ])
            .unwrap(),
        )
        .is_err());
        assert!(run_cli(
            Args::try_parse_from([
                "rbrotli-enc",
                "-s",
                valid_in.to_str().unwrap(),
                fresh_out.to_str().unwrap(),
            ])
            .unwrap(),
        )
        .is_ok());
        assert!(fresh_out.is_file());
        let _ = fs::remove_dir_all(&dir);
    }
}
