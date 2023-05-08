# NNDescent-rs

A highly optimised Rust port of the incredibly `pynndescent` library.

This library does not support all the features of `pynndescent` but tries to include as many as reasonable, see the
list further down of supported and unsupported features.

#### A word a warning
This crate uses *a lot* of unsafe! 🦀

Also, this implementation without all the optimisation features enabled, **is not faster than `pynndescent`**.
Something which quite frankly amazed me was how well numba (the JIT used by `pynndescent`) optimises the
code (which behind the scenes calls LLVM JIT.) this means the compiler can apply all the optimisations
of `fma` and auto-vectorisation as if you were running `-C target-cpu=native` and more, so without fast math
(which makes up at least 50% of the performance) your code will not beat Python, please bear that in mind.

### Core metrics supported
- Cosine
- Dot
- Euclidean
- Squared Euclidean
- Cosine w/ Corrections (Known as `AlternativeCosine` and is the same behaviour if you selected `cosine` with pynndescent.)
- Dot w/ Corrections (Known as `AlternativeDot` and is the same behaviour if you selected `dot` with pynndescent.)
- Custom callback

### Optimisations

In order to unlock the full set of optimisations you need to use the `fast-math` feature and the `nightly` toolchain, 
this is unfortunately unavoidable as currently Rust provides not way to 
insert `fma` instructions without using `core_intrinsics`.

This also requires that you pass the `-C target-feature=+fma` feature to the compiler via the `RUSTFLAGS` env var,
if you don't, the compiler will try it's best to do the same job without the actual instructions, leading to
slower code compared to if you never enabled the `fast-math` feature at all.

### Features
- ✔️ Dot product distance
- ✔️ Cosine distance
- ✔️ Euclidean distance
- ✔️ Custom distance callback (Both euclidean and angular trees supported)
- ✔️ High memory implementation
- ✔️ Initial indexing
- ❌ Searching (WIP)
- ❌ Low memory implementation (WIP)
- ❌ Parrallisation (WIP)
