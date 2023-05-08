#![cfg_attr(feature = "fast-math", feature(core_intrinsics))]

use std::ops::{Deref, DerefMut};

use crate::scalar::ScalarAlgorithms;

pub mod math;

mod scalar;

#[cfg(all(
    feature = "fast-math",
    any(target_feature = "fma", feature = "no-feature-check",)
))]
pub type FastMathAlgorithms = ScalarAlgorithms<math::FastMath>;
pub type StandardAlgorithms = ScalarAlgorithms<math::StdMath>;

#[cfg(all(
    feature = "fast-math",
    not(target_feature = "fma"),
    not(feature = "no-feature-check"),
))]
compile_error!("In order to enable the \"fast-math\" feature you must target \"fma\" feature via compiler flags.");

pub(crate) const EPS: f32 = 1e-8;

pub trait DistanceAlgorithms: Default {
    unsafe fn norm_squared(&self, arr: &[f32]) -> f32;

    unsafe fn norm(&self, arr: &[f32]) -> f32;

    unsafe fn squared_euclidean(&self, left: &[f32], right: &[f32]) -> f32;

    unsafe fn euclidean(&self, left: &[f32], right: &[f32]) -> f32;

    unsafe fn dot(&self, left: &[f32], right: &[f32]) -> f32;

    unsafe fn dot_adjusted(&self, left: &[f32], right: &[f32]) -> f32;

    unsafe fn alternative_dot_adjusted(&self, left: &[f32], right: &[f32]) -> f32;

    unsafe fn cosine(&self, left: &[f32], right: &[f32]) -> f32;

    unsafe fn alternative_cosine(&self, left: &[f32], right: &[f32]) -> f32;

    unsafe fn angular_hyperplane<A: DistanceAlgorithms>(
        &self,
        left: &[f32],
        right: &[f32],
    ) -> Vector<A>;
}

#[derive(Debug)]
pub enum AutoAlgorithms {
    Scalar(StandardAlgorithms),
    #[cfg(all(
        feature = "fast-math",
        any(target_feature = "fma", feature = "no-feature-check")
    ))]
    FastMathScalar(FastMathAlgorithms),
}

macro_rules! select_method {
    ($slf:expr, $method:ident, $($arg:expr $(,)?)*) => {{
        match $slf {
            Self::Scalar(a) => a.$method($($arg,)*),
            #[cfg(all(
                feature = "fast-math",
                any(
                    target_feature = "fma",
                    feature = "no-feature-check"
                )
            ))]
            Self::FastMathScalar(a) => a.$method($($arg,)*),
        }
    }};
}

#[cfg(all(feature = "tracing", target_arch = "x86_64"))]
pub fn log_selected() {
    #[cfg(feature = "no-feature-check")]
    tracing::warn!("The compiler has likely not optimised the fast-math and simd methods well. Please disable this feature.");

    #[cfg(all(feature = "fast-math", target_feature = "fma",))]
    if std::arch::is_x86_feature_detected!("fma") {
        tracing::debug!("Using fast-math impl");
        return;
    }

    tracing::debug!("Using scalar impl");
}

#[cfg(all(feature = "tracing", not(target_arch = "x86_64")))]
pub fn log_selected() {
    tracing::debug!("Using scalar impl")
}

impl Default for AutoAlgorithms {
    #[cfg(target_arch = "x86_64")]
    fn default() -> Self {
        #[cfg(all(feature = "fast-math", target_feature = "fma",))]
        if std::arch::is_x86_feature_detected!("fma") {
            return Self::FastMathScalar(Default::default());
        }

        Self::Scalar(Default::default())
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn default() -> Self {
        Self::Scalar(Default::default())
    }
}

impl DistanceAlgorithms for AutoAlgorithms {
    #[inline]
    unsafe fn norm_squared(&self, arr: &[f32]) -> f32 {
        select_method!(self, norm_squared, arr)
    }

    #[inline]
    unsafe fn norm(&self, arr: &[f32]) -> f32 {
        select_method!(self, norm, arr)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, left: &[f32], right: &[f32]) -> f32 {
        select_method!(self, squared_euclidean, left, right)
    }

    #[inline]
    unsafe fn euclidean(&self, left: &[f32], right: &[f32]) -> f32 {
        select_method!(self, euclidean, left, right)
    }

    #[inline]
    unsafe fn dot(&self, left: &[f32], right: &[f32]) -> f32 {
        select_method!(self, dot, left, right)
    }

    #[inline]
    unsafe fn dot_adjusted(&self, left: &[f32], right: &[f32]) -> f32 {
        select_method!(self, dot_adjusted, left, right)
    }

    #[inline]
    unsafe fn alternative_dot_adjusted(&self, left: &[f32], right: &[f32]) -> f32 {
        select_method!(self, alternative_dot_adjusted, left, right)
    }

    #[inline]
    unsafe fn cosine(&self, left: &[f32], right: &[f32]) -> f32 {
        select_method!(self, cosine, left, right)
    }

    #[inline]
    unsafe fn alternative_cosine(&self, left: &[f32], right: &[f32]) -> f32 {
        select_method!(self, alternative_cosine, left, right)
    }

    #[inline]
    unsafe fn angular_hyperplane<A: DistanceAlgorithms>(
        &self,
        left: &[f32],
        right: &[f32],
    ) -> Vector<A> {
        select_method!(self, angular_hyperplane, left, right)
    }
}

#[derive(Debug)]
pub struct Vector<A: DistanceAlgorithms> {
    inner: Vec<f32>,
    algorithm: A,
}

impl<A: DistanceAlgorithms> Clone for Vector<A> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            algorithm: A::default(),
        }
    }
}

impl<A: DistanceAlgorithms> Deref for Vector<A> {
    type Target = Vec<f32>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<A: DistanceAlgorithms> DerefMut for Vector<A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<A: DistanceAlgorithms> PartialEq<Self> for Vector<A> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<A: DistanceAlgorithms> PartialEq<Vec<f32>> for Vector<A> {
    fn eq(&self, other: &Vec<f32>) -> bool {
        &self.inner == other
    }
}

impl<A: DistanceAlgorithms> Vector<A> {
    #[inline]
    /// Create a new pre-computed wrapper where the norm
    /// is automatically calculated via [DistanceAlgorithms::norm()]
    ///
    /// ## Safety:
    /// This system assumes all the values within the vector are
    /// finite. If they are not finite this can become UB with
    /// "fast-math" or "SIMD" optimisations enabled.
    pub unsafe fn new(inner: Vec<f32>) -> Self {
        Self {
            inner,
            algorithm: A::default(),
        }
    }

    #[inline]
    /// Create a new computed vector from a given element and size.
    pub fn from_element(size: usize, element: f32) -> Self {
        unsafe { Self::new(vec![element; size]) }
    }

    #[inline]
    /// Create a new computed vector from a given element and size.
    pub fn zeroes(size: usize) -> Self {
        unsafe { Self::new(vec![0.0f32; size]) }
    }

    #[inline]
    /// Consumer the wrapper and return the inner type.
    pub fn into_inner(self) -> Vec<f32> {
        self.inner
    }

    #[inline]
    /// Get the raw f32 slice of the inner vec.
    pub fn data(&self) -> &[f32] {
        self.inner.as_slice()
    }

    #[inline]
    /// Gets the standard l2 norm of the vector.
    pub fn norm(&self) -> f32 {
        unsafe { self.algorithm.norm(&self.inner) }
    }

    #[inline]
    /// Gets the standard l2 squared norm of the vector.
    pub fn norm_squared(&self) -> f32 {
        unsafe { self.algorithm.norm_squared(&self.inner) }
    }

    #[inline]
    pub fn euclidean(&self, right: &Self) -> f32 {
        assert_len(self.data(), right.data());
        unsafe { self.algorithm.euclidean(self.data(), right.data()) }
    }

    #[inline]
    pub fn squared_euclidean(&self, right: &Self) -> f32 {
        assert_len(self.data(), right.data());
        unsafe { self.algorithm.squared_euclidean(self.data(), right.data()) }
    }

    #[inline]
    pub fn cosine(&self, right: &Self) -> f32 {
        assert_len(self.data(), right.data());
        unsafe { self.algorithm.cosine(self.data(), right.data()) }
    }

    #[inline]
    pub fn alternative_cosine(&self, right: &Self) -> f32 {
        assert_len(self.data(), right.data());
        unsafe { self.algorithm.alternative_cosine(self.data(), right.data()) }
    }

    #[inline]
    /// Calculates the dot product between the self anf the right vector.
    ///
    /// This does not additional adjustments.
    pub fn dot(&self, right: &Self) -> f32 {
        assert_len(self.data(), right.data());
        unsafe { self.algorithm.dot(self.data(), right.data()) }
    }

    #[inline]
    pub fn dot_adjusted(&self, right: &Self) -> f32 {
        assert_len(self.data(), right.data());
        unsafe { self.algorithm.dot_adjusted(self.data(), right.data()) }
    }

    #[inline]
    pub fn alternative_dot_adjusted(&self, right: &Self) -> f32 {
        assert_len(self.data(), right.data());
        unsafe {
            self.algorithm
                .alternative_dot_adjusted(self.data(), right.data())
        }
    }

    #[inline]
    pub fn angular_hyperplane(&self, right: &Self) -> Self {
        assert_len(self.data(), right.data());
        unsafe { self.algorithm.angular_hyperplane(self.data(), right.data()) }
    }
}

#[inline(always)]
pub(crate) fn assert_len(left: &[f32], right: &[f32]) {
    #[cfg(feature = "debug-assert")]
    debug_assert_eq!(
        left.len(),
        right.len(),
        "The lengths of the two arrays must be the same."
    );
    #[cfg(not(feature = "debug-assert"))]
    assert_eq!(
        left.len(),
        right.len(),
        "The lengths of the two arrays must be the same."
    );
}

#[cfg(any(test, feature = "test-suite"))]
pub mod test_suite {
    use super::*;

    unsafe fn test_euclidean_distance<A: DistanceAlgorithms>() {
        let left = Vector::<A>::new(vec![1.0, 2.0, 3.0]);
        let right = Vector::<A>::new(vec![0.1, 0.2, 0.3]);
        let distance = left.euclidean(&right);
        assert_eq!(distance, 3.3674917, "Euclidean values should match");
    }

    unsafe fn test_squared_euclidean<A: DistanceAlgorithms>() {
        let left = Vector::<A>::new(vec![1.0, 2.0, 3.0]);
        let right = Vector::<A>::new(vec![0.1, 0.2, 0.3]);
        let distance = left.squared_euclidean(&right);
        assert_eq!(distance, 11.34, "Squared euclidean values should match");
    }

    unsafe fn test_cosine<A: DistanceAlgorithms>() {
        let left = Vector::<A>::new(vec![0.5, 0.5, 0.3]);
        let right = Vector::<A>::new(vec![0.1, 0.2, 0.3]);
        let distance = left.cosine(&right);
        assert_eq!(distance, 0.16493326, "Cosine values should match");
    }

    unsafe fn test_alternative_cosine<A: DistanceAlgorithms>() {
        let left = Vector::<A>::new(vec![0.5, 0.5, 0.3]);
        let right = Vector::<A>::new(vec![0.1, 0.2, 0.3]);
        let distance = left.alternative_cosine(&right);
        assert_eq!(
            distance, 0.26003656,
            "Alternative cosine values should match"
        );
    }

    unsafe fn test_dot<A: DistanceAlgorithms>() {
        let left = Vector::<A>::new(vec![1.0, 2.0, 3.0, 4.0]);
        let right = Vector::<A>::new(vec![1.2, 2.3, 3.0, 5.0]);
        let distance = left.dot(&right);
        assert_eq!(distance, 34.8, "Dot values should match")
    }

    unsafe fn test_dot_adjusted<A: DistanceAlgorithms>() {
        let left = Vector::<A>::new(vec![0.5, 0.5, 0.3]);
        let right = Vector::<A>::new(vec![0.1, 0.2, 0.3]);
        let distance = left.dot_adjusted(&right);
        assert_eq!(distance, 0.76, "Dot values should match")
    }

    unsafe fn test_alternative_dot_adjusted<A: DistanceAlgorithms>() {
        let left = Vector::<A>::new(vec![0.5, 0.5, 0.3]);
        let right = Vector::<A>::new(vec![0.1, 0.2, 0.3]);
        let distance = left.alternative_dot_adjusted(&right);
        assert_eq!(distance, 2.0588937, "Alternative dot values should match")
    }

    pub fn test_impl<A: DistanceAlgorithms>() {
        unsafe {
            test_euclidean_distance::<A>();
            test_squared_euclidean::<A>();
            test_cosine::<A>();
            test_alternative_cosine::<A>();
            test_dot::<A>();
            test_dot_adjusted::<A>();
            test_alternative_dot_adjusted::<A>();
        }
    }

    #[test]
    fn test_runtime_selection() {
        test_impl::<AutoAlgorithms>()
    }
}
