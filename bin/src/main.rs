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

#[cfg(unix)]
fn fd_metadata(fd: std::os::fd::BorrowedFd<'_>) -> Option<fs::Metadata> {
    fs::File::from(fd.try_clone_to_owned().ok()?)
        .metadata()
        .ok()
}

fn is_same_regular_meta(a: &Path, ma: &fs::Metadata, b: &Path, mb: &fs::Metadata) -> bool {
    if !ma.is_file() || !mb.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let _ = (a, b);
        ma.dev() == mb.dev() && ma.ino() == mb.ino()
    }
    #[cfg(not(unix))]
    {
        if a.as_os_str() == "-" || b.as_os_str() == "-" {
            return false;
        }
        fs::canonicalize(a)
            .ok()
            .zip(fs::canonicalize(b).ok())
            .is_some_and(|(ca, cb)| ca == cb)
    }
}

fn load_dictionary(path: &Path, quality: u32, lgwin: u32) -> Result<PreparedDictionary> {
    use io::Read as _;
    let file = fs::File::open(path)?;
    let meta = file.metadata().ok();
    let max_len = PreparedDictionary::MAX_RAW_LEN;
    if let Some(ref m) = meta {
        if m.is_dir() {
            bail!("Dictionary path '{}' is a directory", path.display());
        }
        if m.is_file() && m.len() > max_len as u64 {
            bail!(
                "Dictionary '{}' exceeds maximum raw Brotli dictionary size ({} bytes)",
                path.display(),
                max_len
            );
        }
    }
    let cap = meta
        .as_ref()
        .filter(|m| m.is_file())
        .and_then(|m| usize::try_from(m.len()).ok())
        .map_or(0, |len| len.saturating_add(1).min(max_len + 1));
    let mut buf = Vec::with_capacity(cap);
    file.take(max_len as u64 + 1).read_to_end(&mut buf)?;
    if buf.len() > max_len {
        bail!(
            "Dictionary '{}' exceeds maximum raw Brotli dictionary size ({} bytes)",
            path.display(),
            max_len
        );
    }
    Ok(PreparedDictionary::with_lgwin(&buf, quality, lgwin))
}

fn run_cli(args: Args) -> Result<()> {
    if args.num_reps == 0 {
        bail!("--num-reps must be at least 1");
    }
    let use_stdin = args.input.as_os_str() == "-";
    let use_stdout = args.output.as_os_str() == "-";
    let in_meta = if use_stdin {
        #[cfg(unix)]
        {
            use std::os::fd::AsFd as _;
            let sm = fd_metadata(io::stdin().as_fd());
            if sm.as_ref().is_some_and(|m| m.is_dir()) {
                bail!("Standard input is a directory");
            }
            sm
        }
        #[cfg(not(unix))]
        {
            None
        }
    } else {
        let m = fs::metadata(&args.input)?;
        if m.is_dir() {
            bail!("Input path '{}' is a directory", args.input.display());
        }
        Some(m)
    };
    let out_meta = if use_stdout {
        #[cfg(unix)]
        {
            use std::os::fd::AsFd as _;
            fd_metadata(io::stdout().as_fd())
        }
        #[cfg(not(unix))]
        {
            None
        }
    } else {
        fs::metadata(&args.output).ok()
    };
    if out_meta.as_ref().is_some_and(|m| m.is_dir()) {
        bail!(
            "{} is a directory",
            if use_stdout {
                "Standard output".to_string()
            } else {
                format!("Output path '{}'", args.output.display())
            }
        );
    }
    if let (Some(ref im), Some(ref om)) = (&in_meta, &out_meta) {
        if is_same_regular_meta(&args.input, im, &args.output, om) {
            bail!(
                "Refusing to overwrite input file '{}' in place",
                if use_stdin {
                    args.output.display()
                } else {
                    args.input.display()
                }
            );
        }
    }
    let non_regular_in = in_meta.as_ref().is_some_and(|m| !m.is_file());
    let non_regular_out = out_meta.as_ref().is_some_and(|m| !m.is_file());
    if (args.stream || use_stdin || use_stdout || non_regular_in) && args.num_reps > 1 {
        bail!("--num-reps > 1 is only supported for one-shot regular-file compression");
    }
    let use_stream = args.stream
        || use_stdin
        || use_stdout
        || non_regular_in
        || (non_regular_out && args.num_reps == 1);

    let prepared_dict = args
        .dictionary
        .as_ref()
        .map(|p| load_dictionary(p, args.quality, args.lgwin))
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
        let (in_bytes, out_bytes) = io::copy(
            &mut io::BufReader::with_capacity(65_536, reader),
            &mut enc_writer,
        )
        .and_then(|n| {
            enc_writer.try_finish()?;
            enc_writer.get_mut().flush()?;
            Ok((n, enc_writer.total_out()))
        })
        .inspect_err(|_| {
            drop(enc_writer.abandon());
            if created_new_output {
                let _ = fs::remove_file(&args.output);
            }
        })?;
        (in_bytes, out_bytes, 1, start.elapsed().as_secs_f64())
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
        let created_new_output = fs::symlink_metadata(&args.output).is_err();
        fs::write(&args.output, compressed).inspect_err(|_| {
            if created_new_output {
                let _ = fs::remove_file(&args.output);
            }
        })?;
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

        #[cfg(not(miri))]
        {
            let pid = std::process::id();
            let tmp = std::env::temp_dir().join(format!("rbrotli_cli_same_{pid}"));
            let other = std::env::temp_dir().join(format!("rbrotli_cli_other_{pid}"));
            fs::write(&tmp, b"payload").unwrap();
            fs::write(&other, b"other").unwrap();
            let m = fs::metadata(&tmp).unwrap();
            let mo = fs::metadata(&other).unwrap();
            assert!(is_same_regular_meta(&tmp, &m, &tmp, &m));
            assert!(!is_same_regular_meta(&tmp, &m, &other, &mo));
            let _ = fs::remove_file(&tmp);
            let _ = fs::remove_file(&other);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
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
        // Output directory is rejected up front before compression.
        assert!(run_cli(
            Args::try_parse_from([
                "rbrotli-enc",
                "-n",
                "3",
                valid_in.to_str().unwrap(),
                dir.to_str().unwrap(),
            ])
            .unwrap(),
        )
        .is_err());
        assert!(load_dictionary(&dir, 5, 24).is_err());
        #[cfg(unix)]
        {
            use std::os::fd::AsFd as _;
            let f = fs::File::open(&valid_in).unwrap();
            let fm = fd_metadata(f.as_fd()).unwrap();
            assert!(is_same_regular_meta(
                Path::new("-"),
                &fm,
                &valid_in,
                &fs::metadata(&valid_in).unwrap(),
            ));
            // Sparse oversized file (> `MAX_RAW_LEN` without writing 128 MiB of disk blocks).
            let over_dict = dir.join("oversized.dict");
            let of = fs::File::create(&over_dict).unwrap();
            of.set_len(PreparedDictionary::MAX_RAW_LEN as u64 + 1)
                .unwrap();
            drop(of);
            assert!(load_dictionary(&over_dict, 5, 24).is_err());
            // `/dev/null` works for both single-run auto-streaming and `-n 2` one-shot benchmarking.
            for reps in ["1", "2"] {
                assert!(run_cli(
                    Args::try_parse_from([
                        "rbrotli-enc",
                        "-n",
                        reps,
                        valid_in.to_str().unwrap(),
                        "/dev/null",
                    ])
                    .unwrap(),
                )
                .is_ok());
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
