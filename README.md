# A fast, safe Rust brotli encoder
This repository contains a fast Rust implementation of a Brotli encoder.

The code is divided in multiple crates:

 - `bounded-utils`, `safe-arch`, `lsb-bitwriter` and `hugepage-buffer` contain
   code that is meant to be usable by other libraries. They provide safe
   abstractions that help implementing high performance code.
 - `lib` contains the main encoder library. It is entirely written using safe
   code.
 - `bin` contains the main Rust binary for this library.
 - `clib` contains C bindings for the library.

## Performance
At the moment, quality 5 is the recommended setting. It achieves roughly the
same compression ratio as quality 5 of the C Brotli encoder, but with
1.5-2x the performance depending on the machine.

A mode roughly equivalent to C-Brotli's quality 2 in compression ratio and speed
is also implemented, accessible by specifying qualities 3 or lower.
