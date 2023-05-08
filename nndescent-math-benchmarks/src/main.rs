#![feature(core_intrinsics)]

use std::arch::x86_64::*;
use std::intrinsics::{fadd_fast, fmul_fast};
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use hdf5::{Dataset, File};
use nndescent_math::{FastMathAlgorithms, Vector};

macro_rules! timeit {
    ($cmd:ident, $data:expr, $runs:expr) => {{
        let start = Instant::now();
        for _ in 0..$runs {
            for parts in $data.windows(2) {
                let left = &parts[0];
                let right = &parts[1];
                std::hint::black_box(left.$cmd(right));
            }
        }
        println!(
            "{:<25} Took: {:?}, {:?}/iter",
            stringify!($cmd),
            start.elapsed(),
            start.elapsed() / $runs
        );
    }};
}

macro_rules! timeit_direct {
    ($cmd:ident, $data:expr, $runs:expr) => {{
        let start = Instant::now();
        for _ in 0..$runs {
            for parts in $data.windows(2) {
                let left = &parts[0];
                let right = &parts[1];
                std::hint::black_box(unsafe { $cmd(left, right) });
            }
        }
        println!(
            "{:<25} Took: {:?}, {:?}/iter",
            stringify!($cmd),
            start.elapsed(),
            start.elapsed() / $runs
        );
    }};
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let dataset = read_json_dataset("../datasets/gist-960-euclidean")?;

    nndescent_math::log_selected();

    timeit_direct!(dot_and_norms, dataset, 5);
    timeit_direct!(dot, dataset, 5);
    timeit_direct!(dot_std, dataset, 5);

    println!("Correct optimised:");
    timeit!(dot, dataset, 5);
    timeit!(dot_adjusted, dataset, 5);
    timeit!(alternative_dot_adjusted, dataset, 5);
    timeit!(cosine, dataset, 5);
    timeit!(alternative_cosine, dataset, 5);

    Ok(())
}

fn read_json_dataset(path: impl AsRef<Path>) -> Result<Vec<Vector<FastMathAlgorithms>>> {
    let path = path.as_ref().with_extension("hdf5");
    let file = File::open(path)?; // open for reading

    let ds: Dataset = file.dataset("train")?; // open the dataset
    let entries = ds.read_2d::<f32>()?;

    let mut resulting_entries = Vec::with_capacity(entries.len());
    for entry in entries.rows() {
        resulting_entries.push(unsafe { Vector::new(entry.to_vec()) });
    }
    Ok(resulting_entries)
}

#[inline(always)]
/// Calculates the dot product and standard L2 squared norms of two
/// vector in one pass.
unsafe fn dot_and_norms(left: &[f32], right: &[f32]) -> (f32, f32, f32) {
    let mut res = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;

    for k in 0..left.len() {
        res = fadd_fast(
            res,
            fmul_fast(*left.get_unchecked(k), *right.get_unchecked(k)),
        );
        left_norm = fadd_fast(
            left_norm,
            fmul_fast(*left.get_unchecked(k), *left.get_unchecked(k)),
        );
        right_norm = fadd_fast(
            right_norm,
            fmul_fast(*right.get_unchecked(k), *right.get_unchecked(k)),
        );
    }

    (res, left_norm, right_norm)
}

#[inline(always)]
/// Calculates the dot product.
unsafe fn dot(left: &[f32], right: &[f32]) -> f32 {
    let mut res = 0.0;

    for k in 0..left.len() {
        res = fadd_fast(
            res,
            fmul_fast(*left.get_unchecked(k), *right.get_unchecked(k)),
        );
    }

    res
}

#[inline(always)]
/// Calculates the dot product.
unsafe fn dot_std(left: &[f32], right: &[f32]) -> f32 {
    let mut res = 0.0;

    for k in 0..left.len() {
        res += *left.get_unchecked(k) * *right.get_unchecked(k);
    }

    res
}
