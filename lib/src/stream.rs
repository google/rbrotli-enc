// Copyright 2026 Google LLC
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

//! Incremental streaming Brotli encoder ([`StreamingEncoder`] and [`StreamingWriter`]) across
//! `Q0..=Q12`, supporting `Process` / `Flush` (`SyncFlush`) / `Finish`, custom dictionaries,
//! Large Window (`lgwin = 10..=30`), bounded-memory backpressure, and `1T == MT` determinism.

use std::io::{self, Write};
use std::mem::MaybeUninit;

use lsb_bitwriter::{BitWriter, OutSlice};

use crate::compress::build_stream_engine;
use crate::constants::{window_size, LARGE_MAX_WBITS, MIN_WBITS, STD_MAX_WBITS, WBITS};
use crate::hashtable::PreparedDictionary;
use crate::metablock;

/// Operation requested for [`StreamingEncoder::compress_stream`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamOp {
    /// Ingest input and emit compressed bytes opportunistically once internal chunk/metablock
    /// thresholds are reached.
    Process,
    /// Emit all buffered input as complete metablocks followed by a byte-aligned RFC 7932 §9.2
    /// empty metadata metablock so a streaming decoder can immediately recover all input bytes
    /// fed so far while keeping the sliding window and distance ring intact.
    Flush,
    /// Emit all remaining input and close the Brotli stream (`ISLAST = 1`, zero-padded final byte).
    Finish,
}

/// Terminal state of a single [`StreamingEncoder::compress_stream`] invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamStatus {
    /// All currently staged output has been drained; feed more input or advance `StreamOp`.
    Ok,
    /// Staged output (or unconsumed input for an active `Flush`/`Finish`) remains because `output`
    /// filled up; call `compress_stream` again with the same `StreamOp` and remaining input slice.
    HasMoreOutput,
    /// The final Brotli metablock (`ISLAST = 1`) has been emitted and all output bytes have been
    /// drained to the caller.
    Finished,
}

/// Progress report from a single [`StreamingEncoder::compress_stream`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamResult {
    pub bytes_consumed: usize,
    pub bytes_written: usize,
    pub status: StreamStatus,
}

/// State-machine misuse errors returned by [`StreamingEncoder`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamError {
    /// `compress_stream` was called with non-empty input or non-`Finish` op after `Finished`.
    AlreadyFinished,
    /// `StreamOp` or remaining input slice changed before an in-progress `Flush` or `Finish`
    /// finished draining (`StreamStatus::Ok` / `StreamStatus::Finished`).
    OpChangedWhileFlushing,
    /// [`StreamingEncoder::set_dictionary`] was called after input was ingested or output emitted.
    DictionaryAfterStart,
}

impl core::fmt::Display for StreamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AlreadyFinished => f.write_str("Brotli stream is already finished"),
            Self::OpChangedWhileFlushing => {
                f.write_str("stream operation or remaining input changed while flushing/finishing")
            }
            Self::DictionaryAfterStart => {
                f.write_str("cannot attach dictionary after streaming compression has started")
            }
        }
    }
}

impl std::error::Error for StreamError {}

impl From<StreamError> for io::Error {
    fn from(err: StreamError) -> Self {
        io::Error::new(io::ErrorKind::InvalidInput, err)
    }
}

/// Internal per-tier engine hook implemented by `Q0`, `Q1..=Q9`, and `Q10..=Q12` backends.
pub(crate) trait StreamEngine: Send {
    fn step_ingest_cap(&self) -> usize;
    fn reset_stream(&mut self);
    fn attach_dictionary(&mut self, dict: &PreparedDictionary);
    fn unemitted_start(&self) -> usize;
    fn on_compact(&mut self, drop_len: usize);
    fn step_max_bytes(&self, op: StreamOp, in_len: usize) -> Option<usize>;
    fn emit_step(&mut self, op: StreamOp, in_buf: &[u8], bw: &mut BitWriter<'_>) -> bool;
}

#[derive(Default)]
struct StreamState {
    out_pos: usize,
    carry_bits: u8,
    carry_byte: u8,
    header_written: bool,
    is_flushed: bool,
    eof_emitted: bool,
    finished: bool,
    active_op: Option<(StreamOp, usize)>,
    total_in: u64,
    total_out: u64,
}

/// Incremental streaming Brotli encoder supporting `Q0..=Q12`, custom dictionaries, Large Window
/// (`lgwin = 10..=30`), `SyncFlush`, and `1T == MT` deterministic multithreading.
pub struct StreamingEncoder {
    wbits: usize,
    engine: Box<dyn StreamEngine>,
    in_buf: Vec<u8>,
    out_buf: Vec<u8>,
    scratch_bw: Vec<u8>,
    st: StreamState,
}

impl StreamingEncoder {
    /// Creates a streaming encoder for `quality` with the default window (`lgwin = 24`, 1 thread).
    pub fn new(quality: u32) -> Self {
        Self::with_lgwin(quality, WBITS as u32)
    }

    /// Creates a streaming encoder for `(quality, lgwin)` running single-threaded.
    pub fn with_lgwin(quality: u32, lgwin: u32) -> Self {
        Self::with_threads(quality, lgwin, 1)
    }

    /// Creates a streaming encoder configured for `(quality, lgwin, num_threads)`.
    pub fn with_threads(quality: u32, lgwin: u32, num_threads: usize) -> Self {
        let wbits = (lgwin as usize).clamp(MIN_WBITS, LARGE_MAX_WBITS);
        StreamingEncoder {
            wbits,
            engine: build_stream_engine(quality, wbits, num_threads),
            in_buf: Vec::new(),
            out_buf: Vec::new(),
            scratch_bw: Vec::new(),
            st: StreamState::default(),
        }
    }

    /// Attaches a pre-indexed raw LZ77 dictionary before any input has been ingested or output
    /// emitted.
    pub fn set_dictionary(&mut self, dict: &PreparedDictionary) -> Result<(), StreamError> {
        let s = &self.st;
        if s.total_in > 0 || s.header_written || s.finished || s.active_op.is_some() {
            return Err(StreamError::DictionaryAfterStart);
        }
        if !dict.is_empty() {
            self.engine.attach_dictionary(dict);
        }
        Ok(())
    }

    /// Resets stream framing, sliding-window buffers, attached dictionary, and per-stream parser
    /// state while retaining allocated hash tables, DP scratch pools, and buffer capacities.
    pub fn reset(&mut self) {
        self.engine.reset_stream();
        self.in_buf.clear();
        self.out_buf.clear();
        self.st = StreamState::default();
    }

    /// Compresses `input` into `output` according to `op`.
    pub fn compress_stream(
        &mut self,
        op: StreamOp,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<StreamResult, StreamError> {
        self.compress_stream_into(op, input, OutSlice::from_slice(output))
    }

    /// Non-destructive view of currently staged compressed output bytes.
    #[inline]
    pub fn peek_output(&self) -> &[u8] {
        &self.out_buf[self.st.out_pos..]
    }

    /// Zero-copy drain of up to `max_len` staged compressed output bytes (`usize::MAX` for all).
    pub fn take_output(&mut self, max_len: usize) -> &[u8] {
        let n = (self.out_buf.len() - self.st.out_pos).min(max_len);
        let start = self.st.out_pos;
        self.st.out_pos += n;
        self.st.total_out += n as u64;
        if self.st.out_pos == self.out_buf.len() {
            if self.st.eof_emitted {
                self.st.finished = true;
                self.st.active_op = None;
            } else if matches!(self.st.active_op, Some((StreamOp::Flush, 0))) && self.st.is_flushed
            {
                self.st.active_op = None;
            }
        }
        &self.out_buf[start..start + n]
    }

    #[inline]
    pub fn has_more_output(&self) -> bool {
        self.st.out_pos < self.out_buf.len()
    }

    /// Returns `true` when all input ingested so far has been flushed to a byte-aligned
    /// `SyncFlush` boundary and drained from the internal staging buffer.
    #[inline]
    pub fn is_flushed(&self) -> bool {
        self.st.is_flushed
            && !self.has_more_output()
            && self.engine.unemitted_start() == self.in_buf.len()
    }

    #[inline]
    pub fn is_finished(&self) -> bool {
        self.st.finished
    }

    #[inline]
    pub fn total_in(&self) -> u64 {
        self.st.total_in
    }

    #[inline]
    pub fn total_out(&self) -> u64 {
        self.st.total_out
    }

    fn emit_step(&mut self, op: Option<StreamOp>, max_bytes: usize) {
        if self.st.out_pos == self.out_buf.len() {
            self.out_buf.clear();
            self.st.out_pos = 0;
        }
        let cap = max_bytes.saturating_add(64);
        if self.scratch_bw.len() < cap {
            self.scratch_bw.resize(cap, 0);
        }
        let mut bw = BitWriter::new_with_carry(
            &mut self.scratch_bw[..cap],
            self.st.carry_bits,
            self.st.carry_byte,
        );
        if !self.st.header_written {
            metablock::write_window_bits(self.wbits, self.wbits > STD_MAX_WBITS, &mut bw);
            self.st.header_written = true;
        }
        match op {
            Some(op) => {
                let closed = self.engine.emit_step(op, &self.in_buf, &mut bw);
                if op == StreamOp::Finish && !closed {
                    metablock::write_empty_last_metablock(&mut bw);
                }
            }
            None => metablock::write_empty_metadata_metablock(&mut bw),
        }
        if op == Some(StreamOp::Finish) {
            self.out_buf.extend_from_slice(bw.finalize());
            (self.st.carry_bits, self.st.carry_byte) = (0, 0);
        } else {
            let (written, tail_nbits, tail_bits) = bw.finalize_unpadded();
            self.out_buf.extend_from_slice(written);
            (self.st.carry_bits, self.st.carry_byte) = (tail_nbits as u8, tail_bits as u8);
        }
    }

    fn maybe_compact(&mut self) {
        let mb_start = self.engine.unemitted_start();
        let wsize = window_size(self.wbits);
        if mb_start >= wsize + wsize.clamp(4_096, 4 << 20) {
            let drop_len = mb_start - wsize;
            self.in_buf.copy_within(drop_len.., 0);
            self.in_buf.truncate(self.in_buf.len() - drop_len);
            self.engine.on_compact(drop_len);
        }
    }

    /// Compresses `input` into an uninitialized `output` buffer according to `op`.
    pub fn compress_stream_uninit(
        &mut self,
        op: StreamOp,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
    ) -> Result<StreamResult, StreamError> {
        self.compress_stream_into(op, input, OutSlice::from_uninit(output))
    }

    fn compress_stream_into(
        &mut self,
        op: StreamOp,
        input: &[u8],
        mut output: OutSlice<'_>,
    ) -> Result<StreamResult, StreamError> {
        let mk_res = |bytes_consumed, bytes_written, status| {
            Ok(StreamResult {
                bytes_consumed,
                bytes_written,
                status,
            })
        };
        if self.st.finished {
            return if op == StreamOp::Finish && input.is_empty() {
                mk_res(0, 0, StreamStatus::Finished)
            } else {
                Err(StreamError::AlreadyFinished)
            };
        }
        if let Some((prev_op, expected_rem)) = self.st.active_op {
            if op != prev_op || input.len() != expected_rem {
                return Err(StreamError::OpChangedWhileFlushing);
            }
        }

        let (mut consumed, mut written) = (0usize, 0usize);
        let step_cap = self.engine.step_ingest_cap();

        loop {
            // 1. Drain any staged compressed bytes first.
            if self.has_more_output() {
                let chunk = self.take_output(output.len() - written);
                output.write_at(written, chunk);
                written += chunk.len();
                if self.has_more_output() {
                    if op != StreamOp::Process {
                        self.st.active_op = Some((op, input.len() - consumed));
                    }
                    return mk_res(consumed, written, StreamStatus::HasMoreOutput);
                }
                self.out_buf.clear();
                self.st.out_pos = 0;
            }

            // 2. Once EOF has been emitted and fully drained, the stream is finished.
            if self.st.finished {
                return mk_res(consumed, written, StreamStatus::Finished);
            }

            // 3. Ingest a bounded slice of `input` into `in_buf`.
            if consumed < input.len() {
                let buffered_unemitted = self.in_buf.len() - self.engine.unemitted_start();
                let room = step_cap
                    .saturating_sub(buffered_unemitted)
                    .max(usize::from(buffered_unemitted == 0));
                let take = (input.len() - consumed).min(room);
                if take > 0 {
                    self.in_buf
                        .extend_from_slice(&input[consumed..consumed + take]);
                    consumed += take;
                    self.st.total_in += take as u64;
                    self.st.is_flushed = false;
                }
            }

            // 4. Advance the active engine step if enough input is buffered.
            //    When `op` is `Flush` or `Finish` and all `input` has been ingested
            //    (`consumed == input.len()`), fall through to stage the remaining tail +
            //    `SyncFlush` / `ISLAST = 1` trailer into `out_buf` before returning so zero-copy
            //    `take_output` callers (`output.is_empty()`) never stall mid-flush with
            //    `bytes_consumed == input.len()` and an unemitted trailer.
            let progressed = if let Some(max_bytes) = self
                .engine
                .step_max_bytes(StreamOp::Process, self.in_buf.len())
            {
                self.emit_step(Some(StreamOp::Process), max_bytes);
                self.maybe_compact();
                if self.has_more_output() && (op == StreamOp::Process || consumed < input.len()) {
                    continue;
                }
                true
            } else {
                false
            };

            if consumed < input.len() {
                debug_assert!(
                    self.in_buf.len() - self.engine.unemitted_start() < step_cap || progressed
                );
                continue;
            }

            // All `input` for this call has been ingested (`consumed == input.len()`).
            match op {
                StreamOp::Process => {
                    self.st.active_op = None;
                    return mk_res(consumed, written, StreamStatus::Ok);
                }
                StreamOp::Flush => {
                    if self.engine.unemitted_start() < self.in_buf.len() || !self.st.is_flushed {
                        if let Some(max_bytes) = self
                            .engine
                            .step_max_bytes(StreamOp::Flush, self.in_buf.len())
                        {
                            self.emit_step(Some(StreamOp::Flush), max_bytes);
                        }
                        if !self.st.is_flushed {
                            self.emit_step(None, 32);
                            debug_assert_eq!(self.st.carry_bits, 0);
                            self.st.is_flushed = true;
                        }
                        self.maybe_compact();
                        if self.has_more_output() {
                            continue;
                        }
                    }
                    self.st.active_op = None;
                    return mk_res(consumed, written, StreamStatus::Ok);
                }
                StreamOp::Finish => {
                    let max_bytes = self
                        .engine
                        .step_max_bytes(StreamOp::Finish, self.in_buf.len())
                        .unwrap_or(64);
                    self.emit_step(Some(StreamOp::Finish), max_bytes);
                    self.st.eof_emitted = true;
                    debug_assert!(self.has_more_output());
                }
            }
        }
    }
}

/// [`std::io::Write`] adapter around [`StreamingEncoder`].
///
/// Calling [`Write::flush`] emits a byte-aligned Brotli `SyncFlush` metablock and flushes the
/// underlying writer `W`. Calling [`StreamingWriter::finish`] (or dropping the writer) emits the
/// final Brotli metablock (`ISLAST = 1`).
pub struct StreamingWriter<W: Write> {
    inner: Option<W>,
    enc: StreamingEncoder,
}

impl<W: Write> StreamingWriter<W> {
    pub fn new(writer: W, quality: u32) -> Self {
        Self::with_lgwin(writer, quality, WBITS as u32)
    }

    pub fn with_lgwin(writer: W, quality: u32, lgwin: u32) -> Self {
        Self::with_threads(writer, quality, lgwin, 1)
    }

    pub fn with_threads(writer: W, quality: u32, lgwin: u32, num_threads: usize) -> Self {
        StreamingWriter {
            inner: Some(writer),
            enc: StreamingEncoder::with_threads(quality, lgwin, num_threads),
        }
    }

    pub fn set_dictionary(&mut self, dict: &PreparedDictionary) -> Result<(), StreamError> {
        self.enc.set_dictionary(dict)
    }

    #[inline]
    pub fn total_in(&self) -> u64 {
        self.enc.total_in()
    }

    #[inline]
    pub fn total_out(&self) -> u64 {
        self.enc.total_out()
    }

    pub fn get_ref(&self) -> &W {
        self.inner
            .as_ref()
            .expect("StreamingWriter inner writer still present")
    }

    pub fn get_mut(&mut self) -> &mut W {
        self.inner
            .as_mut()
            .expect("StreamingWriter inner writer still present")
    }

    fn drain_staged(enc: &mut StreamingEncoder, writer: &mut W) -> io::Result<()> {
        while enc.has_more_output() {
            let staged = enc.peek_output();
            match writer.write(staged) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write staged Brotli bytes",
                    ));
                }
                Ok(n) => {
                    let _ = enc.take_output(n);
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn drive(&mut self, op: StreamOp, mut rem: &[u8]) -> io::Result<&mut W> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| io::Error::other("StreamingWriter already finished"))?;
        Self::drain_staged(&mut self.enc, writer)?;
        loop {
            let done = match op {
                StreamOp::Process => rem.is_empty(),
                StreamOp::Flush => {
                    rem.is_empty() && (self.enc.st.is_flushed || self.enc.is_finished())
                }
                StreamOp::Finish => self.enc.is_finished(),
            };
            if done
                && !self.enc.has_more_output()
                && (op == StreamOp::Process || self.enc.st.header_written)
            {
                return Ok(writer);
            }
            let res = self.enc.compress_stream(op, rem, &mut [])?;
            rem = &rem[res.bytes_consumed..];
            Self::drain_staged(&mut self.enc, writer)?;
        }
    }

    /// Emits the final Brotli metablock (`ISLAST = 1`) and writes all remaining compressed bytes
    /// to `W` without consuming `self`. Idempotent if already finished.
    pub fn try_finish(&mut self) -> io::Result<()> {
        self.drive(StreamOp::Finish, &[]).map(|_| ())
    }

    /// Finishes the Brotli stream (`ISLAST = 1`), writes all remaining compressed bytes to `W`,
    /// and returns `W`.
    pub fn finish(mut self) -> io::Result<W> {
        let res = self.try_finish();
        let inner = self
            .inner
            .take()
            .expect("StreamingWriter inner writer still present");
        res.map(|()| inner)
    }

    /// Extracts the underlying writer `W` without emitting a final `ISLAST = 1` Brotli trailer,
    /// ensuring an aborted stream after an upstream read error is not sealed as valid EOF.
    pub fn abandon(mut self) -> W {
        self.inner
            .take()
            .expect("StreamingWriter inner writer still present")
    }
}

impl<W: Write> Write for StreamingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| io::Error::other("StreamingWriter already finished"))?;
        Self::drain_staged(&mut self.enc, writer)?;
        let res = self.enc.compress_stream(StreamOp::Process, buf, &mut [])?;
        if let Err(e) = Self::drain_staged(&mut self.enc, writer) {
            if res.bytes_consumed > 0 {
                return Ok(res.bytes_consumed);
            }
            return Err(e);
        }
        Ok(res.bytes_consumed)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.drive(StreamOp::Flush, &[])?.flush()
    }
}

impl<W: Write> Drop for StreamingWriter<W> {
    fn drop(&mut self) {
        if self.inner.is_some() && !self.enc.is_finished() {
            let _ = self.try_finish();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metablock::test_helpers::{decompress_brotli, decompress_brotli_with_dict};
    use crate::q11::test_helpers::{append_lcg_bytes, dbg_or};
    use std::io::Read;

    fn make_mixed_corpus(len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        let phrase = b"Information about international government policy and university research! ";
        for blk_idx in 0u64.. {
            if out.len() >= len {
                break;
            }
            let blk = (len - out.len()).min(16_384);
            if blk_idx % 4 == 2 {
                append_lcg_bytes(&mut out, blk, 0xD1B5_4A32_D192_ED03 ^ blk_idx, 1);
            } else {
                let base = out.len();
                out.extend(
                    (0..blk).map(|i| phrase[i % phrase.len()] ^ (((base + i) >> 13) as u8 & 3)),
                );
            }
        }
        out
    }

    fn pump_op(
        enc: &mut StreamingEncoder,
        op: StreamOp,
        mut rem: &[u8],
        buf: &mut [u8],
        out: &mut Vec<u8>,
    ) {
        let done = if op == StreamOp::Finish {
            StreamStatus::Finished
        } else {
            StreamStatus::Ok
        };
        loop {
            let r = enc.compress_stream(op, rem, buf).unwrap();
            rem = &rem[r.bytes_consumed..];
            out.extend_from_slice(&buf[..r.bytes_written]);
            if r.status == done {
                break;
            }
        }
    }

    #[test]
    fn test_empty_streams_and_reset_all_qualities() {
        let sample = make_mixed_corpus(dbg_or(32_768, 96_000));
        for q in 0..=12 {
            let (mut enc, mut out) = (StreamingEncoder::with_lgwin(q, 22), [0u8; 64]);
            let res = enc
                .compress_stream(StreamOp::Finish, &[], &mut out)
                .unwrap();
            assert_eq!(res.status, StreamStatus::Finished);
            assert!(decompress_brotli(&out[..res.bytes_written]).is_empty());
            assert_eq!(
                enc.compress_stream(StreamOp::Process, b"x", &mut out),
                Err(StreamError::AlreadyFinished)
            );

            enc.reset();
            let mut staged = Vec::new();
            for op in [StreamOp::Process, StreamOp::Flush, StreamOp::Finish] {
                pump_op(&mut enc, op, &[], &mut out, &mut staged);
            }
            assert!(decompress_brotli(&staged).is_empty());

            enc.reset();
            let mut comp = Vec::new();
            pump_op(
                &mut enc,
                StreamOp::Finish,
                &sample,
                &mut [0u8; 4096],
                &mut comp,
            );
            assert_eq!(decompress_brotli(&comp), sample);
        }
    }

    #[test]
    fn test_sync_flush_immediate_prefix_recovery_and_idempotency() {
        let chunk =
            b"The quick brown fox jumps over the lazy dog! Repeated sync-flush prefix data. "
                .repeat(60);
        let expected = [chunk.as_slice(), chunk.as_slice()].concat();
        for q in [0, 1, 4, 6, 9, 10, 11, 12] {
            let (mut enc, mut comp, mut buf) =
                (StreamingEncoder::with_lgwin(q, 20), Vec::new(), [0u8; 512]);
            pump_op(&mut enc, StreamOp::Flush, &chunk, &mut buf, &mut comp);
            let len_flush1 = comp.len();

            let mut prefix = vec![0u8; chunk.len()];
            brotli_decompressor::Decompressor::new(comp.as_slice(), 4096)
                .read_exact(&mut prefix)
                .unwrap();
            assert_eq!(prefix, chunk);

            for _ in 0..2 {
                let r = enc.compress_stream(StreamOp::Flush, &[], &mut buf).unwrap();
                assert_eq!((r.status, r.bytes_written), (StreamStatus::Ok, 0));
            }

            pump_op(&mut enc, StreamOp::Finish, &chunk, &mut buf, &mut comp);
            assert!(comp.len() - len_flush1 < chunk.len() / 10, "q={q}");
            assert_eq!(decompress_brotli(&comp), expected);
        }
    }

    #[test]
    fn test_one_byte_input_and_one_byte_output_backpressure() {
        let data = make_mixed_corpus(dbg_or(6_500, 18_000));
        for q in [0, 2, 5, 8, 10, 11] {
            let (mut enc, mut comp, mut out1) =
                (StreamingEncoder::with_lgwin(q, 18), Vec::new(), [0u8; 1]);
            for (idx, b) in data.iter().enumerate() {
                let op = if idx == data.len() / 3 || idx == (data.len() * 2) / 3 {
                    StreamOp::Flush
                } else {
                    StreamOp::Process
                };
                pump_op(&mut enc, op, core::slice::from_ref(b), &mut out1, &mut comp);
            }
            pump_op(&mut enc, StreamOp::Finish, &[], &mut out1, &mut comp);
            assert_eq!(decompress_brotli(&comp), data, "1B/1B mismatch at q={q}");
        }
    }

    #[test]
    fn test_sliding_window_compaction_small_lgwin_and_dictionary() {
        let dict_raw = b"custom-sliding-dictionary-header-for-compaction-test-9876543210-abcdef";
        let payload = [
            dict_raw.as_slice(),
            &make_mixed_corpus(dbg_or(28_000, 280_000)),
        ]
        .concat();

        for &q in &[0u32, 1, 5, 8, 9, 10, 11, 12] {
            let dict = PreparedDictionary::with_lgwin(dict_raw, q, 10);
            let (mut enc, mut comp, mut out) =
                (StreamingEncoder::with_lgwin(q, 10), Vec::new(), [0u8; 2048]);
            enc.set_dictionary(&dict).unwrap();
            for (ci, chunk) in payload.chunks(4096).enumerate() {
                let op = if ci % 5 == 4 {
                    StreamOp::Flush
                } else {
                    StreamOp::Process
                };
                pump_op(&mut enc, op, chunk, &mut out, &mut comp);
            }
            pump_op(&mut enc, StreamOp::Finish, &[], &mut out, &mut comp);
            assert_eq!(
                enc.set_dictionary(&dict),
                Err(StreamError::DictionaryAfterStart)
            );
            assert_eq!(
                decompress_brotli_with_dict(&comp, dict_raw),
                payload,
                "q={q}"
            );
            enc.reset();
            comp.clear();
            pump_op(
                &mut enc,
                StreamOp::Finish,
                &payload[..8192],
                &mut out,
                &mut comp,
            );
            assert_eq!(decompress_brotli(&comp), &payload[..8192], "reset q={q}");
        }
    }

    #[test]
    fn test_large_window_streaming_above_16mib() {
        let mut anchor = Vec::new();
        append_lcg_bytes(&mut anchor, 65_536, 0x517C_C1B7_2722_0A95, 1);
        let filler: Vec<u8> = (0..65_536)
            .map(|i| ((i * 7 + (i >> 8)) & 0xFF) as u8)
            .collect();

        for &q in dbg_or(&[0u32, 5][..], &[0, 5, 10][..]) {
            let mut writer = StreamingWriter::with_lgwin(Vec::new(), q, 26);
            writer.write_all(&anchor).unwrap();
            (0..272).for_each(|_| writer.write_all(&filler).unwrap());
            writer.flush().unwrap();
            let size_before_repeat = writer.get_ref().len();
            writer.write_all(&anchor).unwrap();
            let comp = writer.finish().unwrap();
            assert!(comp.len() - size_before_repeat < 1024, "q={q}");
            let plain = decompress_brotli(&comp);
            assert_eq!(plain.len(), anchor.len() * 2 + 272 * filler.len());
            assert_eq!(
                (&plain[..anchor.len()], &plain[plain.len() - anchor.len()..]),
                (&anchor[..], &anchor[..])
            );
        }
    }

    #[test]
    fn test_streaming_q10_q11_q12_mt_matches_1t_bit_for_bit() {
        let (corpus_len, lgwin, chunk_sz, q_lens) = dbg_or(
            (
                180_000,
                14,
                16_384,
                [(10u32, 180_000), (11, 150_000), (12, 140_000)],
            ),
            (
                (8 << 20) + 700_000,
                22,
                131_072,
                [
                    (10, (8 << 20) + 700_000),
                    (11, (4 << 20) + 300_000),
                    (12, 1_500_000),
                ],
            ),
        );
        let corpus = make_mixed_corpus(corpus_len);
        let split = corpus.len() - corpus.len() / 5;

        for (q, slice_len) in q_lens {
            let data = &corpus[..slice_len];
            let mid = data.len() - (data.len() / 7).min(split);
            let run = |threads: usize| -> Vec<u8> {
                let mut w = StreamingWriter::with_threads(Vec::new(), q, lgwin, threads);
                data[..mid]
                    .chunks(chunk_sz)
                    .for_each(|c| w.write_all(c).unwrap());
                w.flush().unwrap();
                data[mid..]
                    .chunks(chunk_sz)
                    .for_each(|c| w.write_all(c).unwrap());
                w.finish().unwrap()
            };
            let out_1t = run(1);
            assert_eq!(out_1t, run(4), "1T != 4T streaming mismatch at q={q}");
            assert_eq!(decompress_brotli(&out_1t), data);
        }
    }

    #[test]
    fn test_streaming_writer_would_block_recovery_post_finish_flush_and_abandon() {
        struct FlakyNonBlockingVec {
            out: Vec<u8>,
            step: usize,
            flushes: usize,
        }
        impl Write for FlakyNonBlockingVec {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.step += 1;
                if self.step % 2 == 1 {
                    Err(io::Error::new(io::ErrorKind::WouldBlock, "try again"))
                } else {
                    let n = buf.len().min(7);
                    self.out.extend_from_slice(&buf[..n]);
                    Ok(n)
                }
            }
            fn flush(&mut self) -> io::Result<()> {
                self.step += 1;
                if self.step.is_multiple_of(3) {
                    Err(io::Error::new(io::ErrorKind::WouldBlock, "flush blocked"))
                } else {
                    self.flushes += 1;
                    Ok(())
                }
            }
        }

        let payload = make_mixed_corpus(dbg_or(20_000, 150_000));
        for q in [0u32, 5, 10] {
            let mut sw = StreamingWriter::with_lgwin(
                FlakyNonBlockingVec {
                    out: Vec::new(),
                    step: 0,
                    flushes: 0,
                },
                q,
                18,
            );
            for (ci, mut slice) in payload.chunks(4099).enumerate() {
                while !slice.is_empty() {
                    match sw.write(slice) {
                        Ok(n) => slice = &slice[n..],
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
                        Err(e) => panic!("unexpected write error: {e}"),
                    }
                }
                if ci == 1 {
                    loop {
                        match sw.flush() {
                            Ok(()) => break,
                            Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
                            Err(e) => panic!("unexpected flush error: {e}"),
                        }
                    }
                }
            }
            loop {
                match sw.try_finish() {
                    Ok(()) => break,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
                    Err(e) => panic!("unexpected try_finish error: {e}"),
                }
            }
            // Post-finish `flush()` must delegate cleanly to `W::flush()` without `AlreadyFinished`.
            loop {
                match sw.flush() {
                    Ok(()) => break,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
                    Err(e) => panic!("unexpected post-finish flush error: {e}"),
                }
            }
            let inner = sw.finish().unwrap();
            assert!(inner.flushes >= 1);
            assert_eq!(decompress_brotli(&inner.out), payload, "q={q}");
        }

        // `abandon()` must NOT emit an `ISLAST = 1` trailer on partial data.
        let mut aborted = StreamingWriter::new(Vec::new(), 5);
        aborted.write_all(&payload[..1024]).unwrap();
        let raw_partial = aborted.abandon();
        let mut dec_out = Vec::new();
        assert!(
            brotli_decompressor::BrotliDecompress(&mut raw_partial.as_slice(), &mut dec_out)
                .is_err()
        );
    }

    #[test]
    fn test_q10_sparse_ldm_compaction_anchor_underflow_regression() {
        // `lgwin = 21` (`wsize = 2_097_136 > 1 MiB` local_window_cap) activates `SparseLdmTable`
        // (`sync_mask == 0`) and compacts after 8 waves (`4_194_304` B) with `drop_len = 2_097_168`.
        // Repeating a 4 KiB tile keeps debug H10 fast while planting unique 32-byte anchors right
        // across `drop_len - 5 .. drop_len + 5` (`p1 - drop_len in 1..=5`) and post-compaction.
        let tile: Vec<u8> = (0..4096usize)
            .map(|i| ((i.wrapping_mul(131) ^ (i >> 5)) & 0xFF) as u8)
            .collect();
        let mut data = tile.repeat(1056); // 4,325,376 bytes (> 4_194_304 compaction threshold)
        let anchor = *b"SPARSE_LDM_COMPACTION_ANCHOR_32!";
        for off in [2_097_163usize, 2_097_165, 2_097_168, 4_194_320] {
            data[off..off + anchor.len()].copy_from_slice(&anchor);
        }
        let mut sw = StreamingWriter::with_threads(Vec::new(), 10, 21, 4);
        for chunk in data.chunks(131_072) {
            sw.write_all(chunk).unwrap();
        }
        let comp = sw.finish().unwrap();
        assert_eq!(decompress_brotli(&comp), data);
    }

    #[test]
    fn test_zero_copy_flush_and_finish_at_step_ingest_cap() {
        let first = make_mixed_corpus(dbg_or(196_608, 393_216)); // >= Q1 192 KiB step_ingest_cap
        let second = make_mixed_corpus(dbg_or(24_576, 98_304));
        for &q in dbg_or(&[0u32, 1, 5][..], &[0u32, 1, 5, 10, 11][..]) {
            let mut enc = StreamingEncoder::with_lgwin(q, 18);
            let mut wire = Vec::new();
            let mut rem = first.as_slice();
            loop {
                let r = enc.compress_stream(StreamOp::Flush, rem, &mut []).unwrap();
                rem = &rem[r.bytes_consumed..];
                wire.extend_from_slice(enc.take_output(usize::MAX));
                if rem.is_empty() && !enc.has_more_output() && enc.is_flushed() {
                    break;
                }
            }
            assert!(enc.is_flushed());
            // Streaming prefix decoder must immediately recover 100% of `first` before `second`.
            let mut prefix_out = vec![0u8; first.len() + 64];
            let mut dec = brotli_decompressor::Decompressor::new(wire.as_slice(), 4096);
            let mut got = 0usize;
            while got < first.len() {
                let n = dec.read(&mut prefix_out[got..]).unwrap();
                assert!(n > 0, "q={q}: zero-copy Flush left prefix incomplete");
                got += n;
            }
            assert_eq!(&prefix_out[..first.len()], first.as_slice(), "q={q}");

            // Subsequent `Process` and zero-copy `Finish` must succeed without `OpChangedWhileFlushing`.
            let mut rem2 = second.as_slice();
            while !rem2.is_empty() {
                let r = enc
                    .compress_stream(StreamOp::Process, rem2, &mut [])
                    .unwrap();
                rem2 = &rem2[r.bytes_consumed..];
                wire.extend_from_slice(enc.take_output(usize::MAX));
            }
            while !enc.is_finished() {
                let r = enc.compress_stream(StreamOp::Finish, &[], &mut []).unwrap();
                assert_eq!(r.bytes_consumed, 0);
                wire.extend_from_slice(enc.take_output(usize::MAX));
            }
            let mut expected = first.clone();
            expected.extend_from_slice(&second);
            assert_eq!(decompress_brotli(&wire), expected, "q={q}");
        }
    }

    #[test]
    fn test_streaming_writer_retryable_timed_out_no_duplicate_bytes() {
        struct TimedOutSink {
            out: Vec<u8>,
            calls: usize,
        }
        impl Write for TimedOutSink {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.calls += 1;
                if self.calls % 2 == 1 {
                    return Err(io::Error::new(io::ErrorKind::TimedOut, "transient timeout"));
                }
                let n = buf.len().min(53);
                self.out.extend_from_slice(&buf[..n]);
                Ok(n)
            }
            fn flush(&mut self) -> io::Result<()> {
                self.calls += 1;
                if self.calls.is_multiple_of(3) {
                    return Err(io::Error::new(io::ErrorKind::TimedOut, "flush timeout"));
                }
                Ok(())
            }
        }

        let payload = make_mixed_corpus(200_000);
        let mut sw = StreamingWriter::with_lgwin(
            TimedOutSink {
                out: Vec::new(),
                calls: 0,
            },
            1,
            18,
        );
        for mut slice in payload.chunks(65_536) {
            while !slice.is_empty() {
                match sw.write(slice) {
                    Ok(n) => slice = &slice[n..],
                    Err(e) if e.kind() == io::ErrorKind::TimedOut => continue,
                    Err(e) => panic!("unexpected write error: {e}"),
                }
            }
        }
        loop {
            match sw.try_finish() {
                Ok(()) => break,
                Err(e) if e.kind() == io::ErrorKind::TimedOut => continue,
                Err(e) => panic!("unexpected finish error: {e}"),
            }
        }
        let sink = sw.finish().unwrap();
        assert_eq!(decompress_brotli(&sink.out), payload);
    }

    #[test]
    fn test_streaming_compaction_stream_base_crosses_static_dict_limit() {
        // `q = 5, lgwin = 14` (`wsize = 16,368 << STATIC_DICT_LIMIT = 262,144`): streaming `320 KiB`
        // triggers multiple sliding-window compactions while crossing `STATIC_DICT_LIMIT`.
        let payload = make_mixed_corpus(327_680);
        let mut sw = StreamingWriter::with_lgwin(Vec::new(), 5, 14);
        for chunk in payload.chunks(8192) {
            sw.write_all(chunk).unwrap();
        }
        let comp = sw.finish().unwrap();
        assert_eq!(decompress_brotli(&comp), payload);
    }
}
