use std::marker::PhantomData;
use crate::{DistanceAlgorithms, Vector, EPS};
use crate::math::Math;

#[derive(Debug)]
pub struct ScalarAlgorithms<M: Math>(PhantomData<M>);

impl<M: Math> Default for ScalarAlgorithms<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<M: Math> DistanceAlgorithms for ScalarAlgorithms<M> {
    #[inline]
    unsafe fn norm_squared(&self, arr: &[f32]) -> f32 {
         self.dot(arr, arr)
    }

    #[inline]
    unsafe fn norm(&self, arr: &[f32]) -> f32 {
        self.norm_squared(arr).sqrt()
    }

    #[inline]
    unsafe fn squared_euclidean(&self, left: &[f32], right: &[f32]) -> f32 {
         let mut res = 0.0;

         for k in 0..left.len() {
             let delta = M::sub(*left.get_unchecked(k), *right.get_unchecked(k));
             res = M::add(res, M::mul(delta, delta));
         }

        res
    }

    #[inline]
    unsafe fn euclidean(&self, left: &[f32], right: &[f32]) -> f32 {
        self.squared_euclidean(left, right).sqrt()
    }

    #[inline]
    unsafe fn dot(&self, left: &[f32], right: &[f32]) -> f32 {
        let mut res = 0.0;

        for i in 0..left.len() {
            res = M::add(res, M::mul(*left.get_unchecked(i), *right.get_unchecked(i)));
        }

        res
    }

    #[inline]
    unsafe fn dot_adjusted(&self, left: &[f32], right: &[f32]) -> f32 {
        let res = self.dot(left, right);

        if res <= 0.0 {
            1.0
        } else {
            M::sub(1.0, res)
        }
    }

    #[inline]
    unsafe fn alternative_dot_adjusted(&self, left: &[f32], right: &[f32]) -> f32 {
        let res = self.dot(left, right);

        if res <= 0.0 {
            f32::MAX
        } else {
            -res.log2()
        }
    }

    #[inline]
    unsafe fn cosine(
        &self,
        left: &[f32],
        right: &[f32],
    ) -> f32 {
        let (result, left_norm, right_norm) = dot_and_norms::<M>(left, right);

        if left_norm == 0.0 && right_norm == 0.0 {
            0.0
        } else if left_norm == 0.0 || right_norm == 0.0 {
            1.0
        } else {
            M::sub(1.0, M::div(result, M::mul(left_norm, right_norm).sqrt()))
        }
    }

    #[inline]
    unsafe fn alternative_cosine(
        &self,
        left: &[f32],
        right: &[f32],
    ) -> f32 {
        let (result, left_norm, right_norm) = dot_and_norms::<M>(left, right);

        if left_norm == 0.0 && right_norm == 0.0 {
            0.0
        } else if left_norm == 0.0 || right_norm == 0.0 || result <= 0.0 {
            f32::MAX
        } else {
            M::div(M::mul(left_norm, right_norm).sqrt(), result).log2()
        }
    }

    #[inline]
    unsafe fn angular_hyperplane<A: DistanceAlgorithms>(
        &self,
        left: &[f32],
        right: &[f32],
    ) -> Vector<A> {
        let data = angular_hyperplane::<M>(left, right);
        Vector::new(data)
    }
}

#[inline]
unsafe fn angular_hyperplane<M: Math>(
    left: &[f32],
    right: &[f32],
) -> Vec<f32> {
    let (mut left_norm, mut right_norm) = dual_norms::<M>(left, right);

    if left_norm.abs() < EPS {
        left_norm = 1.0;
    }

    if right_norm.abs() < EPS {
        right_norm = 1.0;
    }

    let mut hyperplane_norm_squared = 0.0;
    let mut hyperplane_array = Vec::with_capacity(left.len());
    for i in 0..left.len() {
        let l = *left.get_unchecked(i);
        let r = *right.get_unchecked(i);
        let value = M::sub(M::div(l, left_norm),  M::div(r, right_norm));
        hyperplane_norm_squared = M::add(hyperplane_norm_squared, M::mul(value, value));
        hyperplane_array.push(value);
    }

    let hyperplane_norm_raw = hyperplane_norm_squared.sqrt();
    let mut hyperplane_norm = hyperplane_norm_raw;
    if hyperplane_norm.abs() < EPS {
        hyperplane_norm = 1.0;
    }

    for v in hyperplane_array.iter_mut() {
        (*v) = M::div(*v, hyperplane_norm);
    }

    hyperplane_array
}

#[inline]
/// Calculates the dot product and standard L2 squared norms of two
/// vector in one pass.
unsafe fn dot_and_norms<M: Math>(left: &[f32], right: &[f32]) -> (f32, f32, f32) {
    let mut res = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;

    for k in 0..left.len() {
        res = M::add(res, M::mul(*left.get_unchecked(k), *right.get_unchecked(k)));
        left_norm = M::add(left_norm, M::mul(*left.get_unchecked(k), *left.get_unchecked(k)));
        right_norm = M::add(right_norm, M::mul(*right.get_unchecked(k), *right.get_unchecked(k)));
    }

    (res, left_norm, right_norm)
}

#[inline]
/// Calculates the L2 squared norm of two vectors in a single pass.
unsafe fn dual_norms<M: Math>(left: &[f32], right: &[f32]) -> (f32, f32) {
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;

    for k in 0..left.len() {
        left_norm = M::add(left_norm, M::mul(*left.get_unchecked(k), *left.get_unchecked(k)));
        right_norm = M::add(right_norm, M::mul(*right.get_unchecked(k), *right.get_unchecked(k)));
    }

    ( left_norm, right_norm)
}

#[cfg(test)]
mod tests {
    use crate::math::StdMath;
    use super::*;

    #[test]
    fn test_scalar_std_impl() {
        crate::test_suite::test_impl::<ScalarAlgorithms<StdMath>>();
    }

    #[cfg(all(
        feature = "fast-math",
        any(
            target_feature = "fma",
            feature = "no-feature-check",
        )
    ))]
    #[test]
    fn test_scalar_fast_math_impl() {
        crate::test_suite::test_impl::<ScalarAlgorithms<crate::math::FastMath>>();
    }

}