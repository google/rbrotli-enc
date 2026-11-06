#![allow(unused_imports)]
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
use crate::constants::{
    try_avx2, window_size, OptAvx2, INITIAL_LAST_DISTANCES, LARGE_MAX_WBITS, MIN_WBITS,
    STD_MAX_WBITS, WBITS,
};
use crate::fast_onepass::fast_twopass_max_output_size_for;
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
pub(crate) trait StreamEngine: Send + Sync {
    fn step_ingest_cap(&self) -> usize;
    fn retain_window(&self, wsize: usize) -> usize {
        wsize
    }
    fn reset_stream(&mut self);
    fn unemitted_start(&self) -> usize;
    fn on_compact(&mut self, drop_len: usize);
    fn step_max_bytes(&self, op: StreamOp, in_len: usize) -> Option<usize>;
    fn emit_step(
        &mut self,
        op: StreamOp,
        in_buf: &[u8],
        dict: Option<&PreparedDictionary>,
        bw: &mut BitWriter<'_>,
    ) -> bool;
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
    dict: Option<PreparedDictionary>,
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
            dict: None,
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
        if dict.is_empty() {
            self.dict = None;
            self.engine.reset_stream();
        } else {
            self.dict = Some(dict.clone());
        }
        Ok(())
    }

    /// Resets stream framing, sliding-window buffers, attached dictionary, and per-stream parser
    /// state while retaining allocated hash tables, DP scratch pools, and buffer capacities.
    pub fn reset(&mut self) {
        self.dict = None;
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
                let closed = self
                    .engine
                    .emit_step(op, &self.in_buf, self.dict.as_ref(), &mut bw);
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
        let wsize = self.engine.retain_window(window_size(self.wbits));
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SinkHealth {
    Ok,
    RetryableErr,
    FatalErr,
}

/// [`std::io::Write`] adapter around [`StreamingEncoder`].
///
/// Calling [`Write::flush`] emits a byte-aligned Brotli `SyncFlush` metablock and flushes the
/// underlying writer `W`. Calling [`StreamingWriter::finish`] emits the final Brotli metablock
/// (`ISLAST = 1`) and surfaces any final write errors; dropping an unfinished writer best-effort
/// finishes the stream only when not unwinding from a panic and no prior fatal or unrecovered sink
/// error occurred (call [`StreamingWriter::abandon`] on upstream read errors to avoid sealing a
/// truncated stream).
pub struct StreamingWriter<W: Write> {
    inner: Option<W>,
    enc: StreamingEncoder,
    pending_err: Option<io::Error>,
    health: SinkHealth,
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
            pending_err: None,
            health: SinkHealth::Ok,
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

    #[inline]
    fn note_err(&mut self, err: &io::Error) {
        if matches!(
            err.kind(),
            io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
        ) {
            if self.health == SinkHealth::Ok {
                self.health = SinkHealth::RetryableErr;
            }
        } else {
            self.health = SinkHealth::FatalErr;
        }
    }

    #[inline]
    fn note_ok(&mut self) {
        if self.health == SinkHealth::RetryableErr {
            self.health = SinkHealth::Ok;
        }
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

    fn step_once(&mut self, op: StreamOp, buf: &[u8]) -> io::Result<usize> {
        if let Some(err) = self.pending_err.take() {
            self.note_err(&err);
            return Err(err);
        }
        let Some(writer) = self.inner.as_mut() else {
            self.health = SinkHealth::FatalErr;
            return Err(io::Error::other("StreamingWriter already finished"));
        };
        if let Err(err) = Self::drain_staged(&mut self.enc, writer) {
            self.note_err(&err);
            return Err(err);
        }
        if buf.is_empty()
            && self.enc.st.header_written
            && (self.enc.is_finished() || (op == StreamOp::Flush && self.enc.st.is_flushed))
        {
            self.note_ok();
            return Ok(0);
        }
        let n = match self.enc.compress_stream(op, buf, &mut []) {
            Ok(res) => res.bytes_consumed,
            Err(err) => {
                let io_err = io::Error::from(err);
                self.note_err(&io_err);
                return Err(io_err);
            }
        };
        match Self::drain_staged(&mut self.enc, writer) {
            Ok(()) => {
                self.note_ok();
                Ok(n)
            }
            Err(e) if n == 0 => {
                self.note_err(&e);
                Err(e)
            }
            Err(e) => {
                if !matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) {
                    self.health = SinkHealth::FatalErr;
                    self.pending_err = Some(e);
                }
                Ok(n)
            }
        }
    }

    fn drive(&mut self, op: StreamOp) -> io::Result<&mut W> {
        if let Some(err) = self.pending_err.take() {
            self.note_err(&err);
            return Err(err);
        }
        while !(self.enc.st.header_written
            && !self.enc.has_more_output()
            && (self.enc.is_finished() || (op == StreamOp::Flush && self.enc.st.is_flushed)))
        {
            self.step_once(op, &[])?;
        }
        self.note_ok();
        self.inner.as_mut().ok_or_else(|| {
            self.health = SinkHealth::FatalErr;
            io::Error::other("StreamingWriter already finished")
        })
    }

    /// Emits the final Brotli metablock (`ISLAST = 1`) and writes all remaining compressed bytes
    /// to `W` without consuming `self`. Idempotent if already finished.
    pub fn try_finish(&mut self) -> io::Result<()> {
        self.drive(StreamOp::Finish).map(|_| ())
    }

    /// Finishes the Brotli stream (`ISLAST = 1`), writes all remaining compressed bytes to `W`,
    /// and returns `W`.
    pub fn finish(mut self) -> io::Result<W> {
        let res = self.try_finish();
        let inner = self.abandon();
        res.map(|()| inner)
    }

    /// Extracts the underlying writer `W` without emitting a final `ISLAST = 1` Brotli trailer,
    /// ensuring an aborted stream after an upstream read error is not sealed as valid EOF.
    pub fn abandon(mut self) -> W {
        self.pending_err = None;
        self.health = SinkHealth::Ok;
        self.inner
            .take()
            .expect("StreamingWriter inner writer still present")
    }
}

impl<W: Write> Write for StreamingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            Ok(0)
        } else {
            self.step_once(StreamOp::Process, buf)
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        let res = self.drive(StreamOp::Flush).and_then(|w| w.flush());
        match &res {
            Ok(()) => self.note_ok(),
            Err(err) => self.note_err(err),
        }
        res
    }
}

impl<W: Write> Drop for StreamingWriter<W> {
    fn drop(&mut self) {
        if self.inner.is_some()
            && self.pending_err.is_none()
            && self.health == SinkHealth::Ok
            && !self.enc.is_finished()
            && !std::thread::panicking()
        {
            let _ = self.try_finish();
        }
    }
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<StreamingEncoder>();
    assert_send_sync::<StreamingWriter<Vec<u8>>>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metablock::test_helpers::{decompress_brotli, decompress_brotli_with_dict};
    use crate::q11::test_helpers::{append_lcg_bytes, dbg_or, miri_or};
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
        let sample = make_mixed_corpus(miri_or(128, dbg_or(32_768, 96_000)));
        let lgwin = miri_or(12, 22);
        let qs: &[u32] = miri_or(&[5], &[3, 4, 5, 6, 7, 8, 9]);
        for &q in qs {
            let (mut enc, mut out) = (StreamingEncoder::with_lgwin(q, lgwin), [0u8; 64]);
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
                .repeat(miri_or(32, 60));
        let expected = [chunk.as_slice(), chunk.as_slice()].concat();
        let lgwin = miri_or(12, 20);
        for &q in miri_or(&[4u32][..], &[4, 6, 9][..]) {
            let (mut enc, mut comp, mut buf) = (
                StreamingEncoder::with_lgwin(q, lgwin),
                Vec::new(),
                [0u8; 512],
            );
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
        let data = make_mixed_corpus(miri_or(96, dbg_or(6_500, 18_000)));
        let lgwin = miri_or(12, 18);
        for &q in miri_or(&[5u32][..], &[5, 8][..]) {
            let (mut enc, mut comp, mut out1) =
                (StreamingEncoder::with_lgwin(q, lgwin), Vec::new(), [0u8; 1]);
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
    #[cfg_attr(miri, ignore)]
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

    #[test]
    fn test_streaming_writer_panic_unwind_and_deferred_fatal_error() {
        use std::sync::{Arc, Mutex};

        #[derive(Clone, Default)]
        struct SharedVec(Arc<Mutex<Vec<u8>>>);
        impl Write for SharedVec {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        // 1. Dropping `StreamingWriter` during panic unwind must NOT seal a truncated stream with
        //    an `ISLAST = 1` Brotli EOF trailer.
        let sink = SharedVec::default();
        let sink_clone = sink.clone();
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let mut sw = StreamingWriter::with_lgwin(sink_clone, 5, miri_or(12, 16));
            sw.write_all(b"partial plaintext before panic").unwrap();
            panic!("simulated upstream panic");
        }));
        assert!(unwound.is_err());
        let wire = sink.0.lock().unwrap().clone();
        let mut decoded = Vec::new();
        assert!(
            brotli_decompressor::BrotliDecompress(&mut wire.as_slice(), &mut decoded).is_err()
                || decoded.is_empty() && wire.is_empty(),
            "panic drop must not emit a valid EOF-terminated Brotli stream"
        );
        assert!(wire.is_empty());

        // 2 & 3: Dropping `StreamingWriter` after a surfaced error (deferred `BrokenPipe` returned
        // via `write_all` `?`, `n == 0` `Flush` write failure, `W::flush()` failure, or unretried
        // `TimedOut` abort) must NEVER emit `ISLAST = 1` sealing a truncated prefix as EOF, and a
        // one-shot deferred `BrokenPipe` during `Process` must surface on `finish()` / next write.
        let assert_not_valid_eof = |wire: &[u8], ctx: &str| {
            let mut decoded = Vec::new();
            assert!(
                wire.is_empty()
                    || brotli_decompressor::BrotliDecompress(&mut &*wire, &mut decoded).is_err(),
                "{ctx}: Drop sealed truncated stream ({} wire -> {} decoded bytes)",
                wire.len(),
                decoded.len(),
            );
        };

        struct FailCountShared {
            fail_on: io::ErrorKind,
            fail_flush: bool,
            fails_left: usize,
            out: Arc<Mutex<Vec<u8>>>,
        }
        impl Write for FailCountShared {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                if !self.fail_flush && self.fails_left > 0 {
                    self.fails_left -= 1;
                    return Err(io::Error::new(self.fail_on, "injected write failure"));
                }
                self.out.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                if self.fail_flush && self.fails_left > 0 {
                    self.fails_left -= 1;
                    return Err(io::Error::new(self.fail_on, "injected flush failure"));
                }
                Ok(())
            }
        }

        if !cfg!(miri) {
            struct FailOnceSink {
                failed: bool,
                out: Vec<u8>,
            }
            impl Write for FailOnceSink {
                fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                    if !self.failed {
                        self.failed = true;
                        return Err(io::Error::new(io::ErrorKind::BrokenPipe, "sink broke once"));
                    }
                    self.out.extend_from_slice(buf);
                    Ok(buf.len())
                }
                fn flush(&mut self) -> io::Result<()> {
                    Ok(())
                }
            }

            for chunk_len in [524_288usize, 1_048_576] {
                let chunk = make_mixed_corpus(chunk_len);
                let mut sw = StreamingWriter::with_lgwin(
                    FailOnceSink {
                        failed: false,
                        out: Vec::new(),
                    },
                    4,
                    18,
                );
                let err = sw
                    .write_all(&chunk)
                    .and_then(|()| sw.finish().map(|_| ()))
                    .expect_err("deferred BrokenPipe must surface");
                assert_eq!(err.kind(), io::ErrorKind::BrokenPipe);
            }

            // 3a: Deferred `BrokenPipe` (`n > 0` consumed, surfaced on next chunk, then `sw` dropped).
            // 3b: Surfaced `TimedOut` aborting `write_all` (`fails_left = 2`: opportunistic drain +
            //     next chunk's pre-drain), then `sw` dropped while sink accepts subsequent writes.
            let big = make_mixed_corpus(1_048_576);
            for (kind, fails_left) in [(io::ErrorKind::BrokenPipe, 1), (io::ErrorKind::TimedOut, 2)]
            {
                let wire = Arc::new(Mutex::new(Vec::new()));
                let err = {
                    let mut sw = StreamingWriter::with_lgwin(
                        FailCountShared {
                            fail_on: kind,
                            fail_flush: false,
                            fails_left,
                            out: Arc::clone(&wire),
                        },
                        4,
                        18,
                    );
                    sw.write_all(&big)
                        .expect_err("sink error must abort write_all")
                };
                assert_eq!(err.kind(), kind);
                assert_not_valid_eof(&wire.lock().unwrap(), &format!("write_all {kind:?}"));
            }
        }

        // 3c: `n == 0` write failure during `flush()` and 3d: `W::flush()` failure after emitting a
        // sync-flush metablock — dropping `sw` must leave the stream unsealed (`ISLAST = 0`).
        for fail_flush in [false, true] {
            let wire = Arc::new(Mutex::new(Vec::new()));
            let err = {
                let mut sw = StreamingWriter::with_lgwin(
                    FailCountShared {
                        fail_on: io::ErrorKind::BrokenPipe,
                        fail_flush,
                        fails_left: 1,
                        out: Arc::clone(&wire),
                    },
                    5,
                    miri_or(12, 16),
                );
                sw.write_all(b"first half before failed flush").unwrap();
                sw.flush().expect_err("flush failure must surface")
            };
            assert_eq!(err.kind(), io::ErrorKind::BrokenPipe);
            let captured = wire.lock().unwrap().clone();
            if fail_flush {
                assert!(!captured.is_empty(), "sync-flush prefix should be present");
            }
            assert_not_valid_eof(&captured, &format!("flush(fail_flush={fail_flush})"));
        }
    }
}
