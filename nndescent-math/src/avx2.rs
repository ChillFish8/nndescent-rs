use std::marker::PhantomData;
use std::arch::x86_64::*;
use std::mem;

use crate::{DistanceAlgorithms, Vector, EPS};
use crate::math::Math;

#[derive(Debug)]
pub struct Avx2Algorithms<M: Math>(PhantomData<M>);

impl<M: Math> Default for Avx2Algorithms<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<M: Math> DistanceAlgorithms for Avx2Algorithms<M> {
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
        let mut acc = _mm256_setzero_ps();

        let mut i = 0;
        while left.len() - i >= 8 {
            let left = load_reg(left, i);
            let right = load_reg(right, i);
            let delta = _mm256_sub_ps(left, right);
            let squared = _mm256_mul_ps(delta, delta);
            acc = _mm256_add_ps(acc, squared);

            i += 8;
        }

        let mut res = sum(acc);

        for k in i..left.len() {
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
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;

        while left.len() - i >= 8 {
            let left = load_reg(left, i);
            let right = load_reg(right, i);

            let res = _mm256_mul_ps(left, right);
            acc = _mm256_add_ps(acc, res);

            i += 8;
        }

        let mut res = sum(acc);

        for k in i..left.len() {
            res = M::add(res, M::mul(*left.get_unchecked(k), *right.get_unchecked(k)));
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

    let mut hyperplane_array = Vec::with_capacity(left.len());
    let mut norm_acc = _mm256_setzero_ps();
    let left_norm_reg = _mm256_set1_ps(left_norm);
    let right_norm_reg = _mm256_set1_ps(right_norm);

    let mut i = 0;
    while left.len() - i >= 8 {
        let left = load_reg(left, i);
        let right = load_reg(right,i);

        let left_normalized = _mm256_div_ps(left, left_norm_reg);
        let right_normalized = _mm256_div_ps(right, right_norm_reg);
        let delta = _mm256_sub_ps(left_normalized, right_normalized);
        let values = mem::transmute::<_, [f32; 8]>(delta);

        hyperplane_array.push(values[0]);
        hyperplane_array.push(values[1]);
        hyperplane_array.push(values[2]);
        hyperplane_array.push(values[3]);
        hyperplane_array.push(values[4]);
        hyperplane_array.push(values[5]);
        hyperplane_array.push(values[6]);
        hyperplane_array.push(values[7]);

        let squared = _mm256_mul_ps(delta, delta);
        norm_acc = _mm256_add_ps(norm_acc, squared);

        i += 8;
    }

    let mut hyperplane_norm_squared = sum(norm_acc);
    for k in i..left.len() {
        let l = *left.get_unchecked(k);
        let r = *right.get_unchecked(k);

        let value = M::sub(M::div(l, left_norm),  M::div(r, right_norm));
        hyperplane_norm_squared = M::add(hyperplane_norm_squared, M::mul(value, value));

        hyperplane_array.push(value);
    }

    let hyperplane_norm_raw = hyperplane_norm_squared.sqrt();
    let mut hyperplane_norm = hyperplane_norm_raw;
    if hyperplane_norm.abs() < EPS {
        hyperplane_norm = 1.0;
    }
    let hyperplane_norm_reg = _mm256_set1_ps(hyperplane_norm);

    let mut i = 0;
    while hyperplane_array.len() - i >= 8 {
        let values = load_reg(&hyperplane_array, i);
        let results = _mm256_div_ps(values, hyperplane_norm_reg);
        let values = mem::transmute::<_, [f32; 8]>(results);

        (*hyperplane_array.get_unchecked_mut(i)) = values[0];
        (*hyperplane_array.get_unchecked_mut(i + 1)) = values[1];
        (*hyperplane_array.get_unchecked_mut(i + 2)) = values[2];
        (*hyperplane_array.get_unchecked_mut(i + 3)) = values[3];
        (*hyperplane_array.get_unchecked_mut(i + 4)) = values[4];
        (*hyperplane_array.get_unchecked_mut(i + 5)) = values[5];
        (*hyperplane_array.get_unchecked_mut(i + 6)) = values[6];
        (*hyperplane_array.get_unchecked_mut(i + 6)) = values[7];

        i += 1;
    }

    for k in i..hyperplane_array.len() {
        let v = hyperplane_array.get_unchecked(k);
        (*hyperplane_array.get_unchecked_mut(k)) = M::div(*v, hyperplane_norm);
    }

    hyperplane_array
}

#[inline]
/// Calculates the dot product and standard L2 squared norms of two
/// vector in one pass.
unsafe fn dot_and_norms<M: Math>(left: &[f32], right: &[f32]) -> (f32, f32, f32) {
    let mut acc = _mm256_setzero_ps();
    let mut left_acc = _mm256_setzero_ps();
    let mut right_acc = _mm256_setzero_ps();
    let mut i = 0;

    while left.len() - i >= 8 {
        let left = load_reg(left, i);
        let left_res = _mm256_mul_ps(left, left);

        let right = load_reg(right, i);
        let right_res = _mm256_mul_ps(right, right);

        let res = _mm256_mul_ps(left, right);

        acc = _mm256_add_ps(acc, res);
        left_acc = _mm256_add_ps(left_acc, left_res);
        right_acc = _mm256_add_ps(right_acc, right_res);

        i += 8;
    }

    let mut res = sum(acc);
    let mut left_norm = sum(left_acc);
    let mut right_norm = sum(right_acc);

    for k in i..left.len() {
        res = M::add(res, M::mul(*right.get_unchecked(k), *left.get_unchecked(k)));
        left_norm = M::add(left_norm, M::mul(*left.get_unchecked(k), *left.get_unchecked(k)));
        right_norm = M::add(right_norm, M::mul(*right.get_unchecked(k), *right.get_unchecked(k)));
    }

    (res, left_norm, right_norm)
}

/// Calculates the L2 squared norm of two vectors in a single pass.
unsafe fn dual_norms<M: Math>(left: &[f32], right: &[f32]) -> (f32, f32) {
    let mut left_acc = _mm256_setzero_ps();
    let mut right_acc = _mm256_setzero_ps();
    let mut i = 0;

    while left.len() - i >= 8 {
        let left = load_reg(left, i);
        let left_res = _mm256_mul_ps(left, left);

        let right = load_reg(right, i);
        let right_res = _mm256_mul_ps(right, right);

        left_acc = _mm256_add_ps(left_acc, left_res);
        right_acc = _mm256_add_ps(right_acc, right_res);

        i += 8;
    }

    let mut left_norm = sum(left_acc);
    let mut right_norm = sum(right_acc);

    for k in i..left.len() {
        left_norm = M::add(left_norm, M::mul(*left.get_unchecked(k), *left.get_unchecked(k)));
        right_norm = M::add(right_norm, M::mul(*right.get_unchecked(k), *right.get_unchecked(k)));
    }

    (left_norm, right_norm)
}

unsafe fn sum(reg: __m256) -> f32 {
    let mut low = _mm256_castps256_ps128(reg);
    let high = _mm256_extractf128_ps::<1>(reg);
    low = _mm_add_ps(low, high);

    let mut shuffle = _mm_movehdup_ps(low);
    let mut sums = _mm_add_ps(low, shuffle);
    shuffle = _mm_movehl_ps(sums, shuffle);
    sums = _mm_add_ps(sums, shuffle);
    _mm_cvtss_f32(sums)
}

#[inline(always)]
unsafe fn load_reg(arr: &[f32], offset: usize) -> __m256 {
    _mm256_set_ps(
        *arr.get_unchecked(offset),
        *arr.get_unchecked(offset + 1),
        *arr.get_unchecked(offset + 2),
        *arr.get_unchecked(offset + 3),
        *arr.get_unchecked(offset + 4),
        *arr.get_unchecked(offset + 5),
        *arr.get_unchecked(offset + 6),
        *arr.get_unchecked(offset + 7),
    )
}

#[cfg(test)]
mod tests {
    use crate::math::StdMath;
    use super::*;

    #[test]
    fn test_scalar_std_impl() {
        crate::test_suite::test_impl::<Avx2Algorithms<StdMath>>();
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
        crate::test_suite::test_impl::<Avx2Algorithms<crate::math::FastMath>>();
    }

}