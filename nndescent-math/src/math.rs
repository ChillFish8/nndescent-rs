pub trait Math {
    unsafe fn add(a: f32, b: f32) -> f32;

    unsafe fn sub(a: f32, b: f32) -> f32;

    unsafe fn mul(a: f32, b: f32) -> f32;

    unsafe fn div(a: f32, b: f32) -> f32;
}

#[derive(Debug)]
pub struct StdMath;
impl Math for StdMath {
    #[inline(always)]
    unsafe fn add(a: f32, b: f32) -> f32 {
        a + b
    }

    #[inline(always)]
    unsafe fn sub(a: f32, b: f32) -> f32 {
        a - b
    }

    #[inline(always)]
    unsafe fn mul(a: f32, b: f32) -> f32 {
        a * b
    }

    #[inline(always)]
    unsafe fn div(a: f32, b: f32) -> f32 {
        a / b
    }
}

#[cfg(all(
    feature = "fast-math",
    any(target_feature = "fma", feature = "no-feature-check",)
))]
pub use fast_math::FastMath;

#[cfg(all(
    feature = "fast-math",
    any(target_feature = "fma", feature = "no-feature-check",)
))]
mod fast_math {
    use core::intrinsics::{fadd_fast, fdiv_fast, fmul_fast, fsub_fast};

    use super::Math;

    #[derive(Debug)]
    pub struct FastMath;

    impl Math for FastMath {
        #[inline(always)]
        unsafe fn add(a: f32, b: f32) -> f32 {
            fadd_fast(a, b)
        }

        #[inline(always)]
        unsafe fn sub(a: f32, b: f32) -> f32 {
            fsub_fast(a, b)
        }

        #[inline(always)]
        unsafe fn mul(a: f32, b: f32) -> f32 {
            fmul_fast(a, b)
        }

        #[inline(always)]
        unsafe fn div(a: f32, b: f32) -> f32 {
            fdiv_fast(a, b)
        }
    }
}
