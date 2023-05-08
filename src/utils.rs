use std::fmt::{Debug, Formatter};

use bumpalo::collections::Vec as BumpVec;
use rand::{Rng, SeedableRng};
use nndescent_math::{DistanceAlgorithms, AutoAlgorithms};

use crate::{Array, BuildError};

pub type RngState = (i64, i64, i64);

/// Generate the initial random state from entropy.
pub fn create_rand_state() -> RngState {
    let mut rng = rand::rngs::StdRng::from_entropy();
    (rng.gen(), rng.gen(), rng.gen())
}

/// Generate the initial random state from entropy.
pub fn create_rand_state_from_rng<R: Rng>(rng: &mut R) -> RngState {
    (rng.gen(), rng.gen(), rng.gen())
}

#[inline(always)]
/// A fast (pseudo)-random number generator.
///
/// Returns a (pseudo)-random int32 value
pub fn tau_rand_int(state: &mut RngState) -> i32 {
    state.0 = (((state.0 & 4294967294) << 12) & 0xFFFFFFFF)
        ^ ((((state.0 << 13) & 0xFFFFFFFF) ^ state.0) >> 19);
    state.1 = (((state.1 & 4294967288) << 4) & 0xFFFFFFFF)
        ^ ((((state.1 << 2) & 0xFFFFFFFF) ^ state.1) >> 25);
    state.2 = (((state.2 & 4294967280) << 17) & 0xFFFFFFFF)
        ^ ((((state.2 << 3) & 0xFFFFFFFF) ^ state.2) >> 11);

    (state.0 ^ state.1 ^ state.2) as i32
}

#[inline(always)]
/// A fast (pseudo)-random number generator.
///
/// Returns a (pseudo)-random float32 in the interval [0, 1]
pub fn tau_rand(state: &mut RngState) -> f32 {
    let integer = tau_rand_int(state);
    (integer as f32 / 0x7FFFFFFF as f32).abs()
}

pub fn verify_and_normalize(input: Vec<Vec<f32>>, normalize: bool) -> Result<Vec<Array>, BuildError> {
    let selector = AutoAlgorithms::default();
    let mut arrays = Vec::with_capacity(input.len());
    for mut array in input {
        if array.iter().any(|v| !v.is_finite()) {
            return Err(BuildError::DataNoteFinite);
        }

        if normalize {
            let norm = unsafe { selector.norm(&array) };
            for element in array.iter_mut() {
                (*element) /= norm;
            }
        }

        // SAFETY: This is safe as we are ensuring the values are finite
        //         beforing passing it in.
        arrays.push(unsafe { Array::new(array) })
    }
    Ok(arrays)
}

macro_rules! unsafe_get_swap_point {
    ($p:expr, $i:expr, $size:expr, $priorities:expr) => {{
        let ic1 = 2 * $i + 1;
        let ic2 = ic1 + 1;

        if ic1 >= $size {
            break;
        } else if (ic2 >= $size)
            || ($priorities.get_unchecked(ic1) >= $priorities.get_unchecked(ic2))
        {
            if *$priorities.get_unchecked(ic1) > $p {
                ic1
            } else {
                break;
            }
        } else if $p < *$priorities.get_unchecked(ic2) {
            ic2
        } else {
            break;
        }
    }};
}

pub type Indices = Vec<Option<usize>>;
pub type BumpIndices<'b> = BumpVec<'b, Option<usize>>;

pub struct Heap {
    pub indices: Vec<Indices>,
    pub distances: Vec<Array>,
    pub flags: Vec<Vec<bool>>, // TODO: Consider using bitvec here.
}

impl Debug for Heap {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Indices:")?;
        for indices in self.indices.iter() {
            writeln!(f, "{indices:?}")?;
        }

        writeln!(f, "Distances:")?;
        for distances in self.distances.iter() {
            writeln!(f, "{distances:?}")?;
        }

        writeln!(f, "Flags:")?;
        for flags in self.flags.iter() {
            writeln!(f, "{flags:?}")?;
        }

        Ok(())
    }
}

impl Heap {
    /// Create a new heap.
    pub fn new(n_points: usize, size: usize) -> Self {
        let indices = vec![vec![None; size]; n_points];
        let distances = vec![Array::from_element(size, f32::INFINITY); n_points];
        let flags = vec![vec![false; size]; n_points];
        Self {
            indices,
            distances,
            flags,
        }
    }

    #[inline]
    pub fn distance_thresholds(&self, dist_thresholds: &mut Vec<f32>) {
        dist_thresholds.clear();
        for row in self.distances.iter() {
            if let Some(col) = row.first() {
                dist_thresholds.push(*col);
            }
        }
    }

    #[inline]
    pub fn checked_flagged_heap_push(
        &mut self,
        point: usize,
        dist: f32,
        n: usize,
        flag: bool,
    ) -> bool {
        // Ensure we wont go out of bounds
        assert!(self.distances.len() > point);
        assert!(self.indices.len() > point);
        assert!(self.flags.len() > point);

        unsafe { self.checked_flagged_heap_push_inner(point, dist, n, flag) }
    }

    #[inline(always)]
    unsafe fn checked_flagged_heap_push_inner(
        &mut self,
        point: usize,
        dist: f32,
        n: usize,
        flag: bool,
    ) -> bool {
        let distances = self.distances.get_unchecked_mut(point);
        let indices = self.indices.get_unchecked_mut(point);
        let flags = self.flags.get_unchecked_mut(point);

        if dist >= *distances.get_unchecked(0) {
            return false;
        }

        let size = distances.len();

        // Break if we already have this element.
        if indices.iter().any(|v| *v == Some(n)) {
            return false;
        }

        (*distances.get_unchecked_mut(0)) = dist;
        (*indices.get_unchecked_mut(0)) = Some(n);
        (*flags.get_unchecked_mut(0)) = flag;

        // Descend the heap, swapping values until the max heap criterion is met
        let mut i = 0;
        loop {
            let i_swap = unsafe_get_swap_point!(dist, i, size, distances);

            (*distances.get_unchecked_mut(i)) = *distances.get_unchecked(i_swap);
            (*indices.get_unchecked_mut(i)) = *indices.get_unchecked(i_swap);
            (*flags.get_unchecked_mut(i)) = *flags.get_unchecked(i_swap);

            i = i_swap
        }

        (*distances.get_unchecked_mut(i)) = dist;
        (*indices.get_unchecked_mut(i)) = Some(n);
        (*flags.get_unchecked_mut(i)) = flag;

        true
    }
}

/// Given two arrays representing a heap (indices and distances), reorder the
/// arrays by increasing distance. This is effectively just the second half of
/// heap sort (the first half not being required since we already have the
/// graph_data in a heap).
pub fn deheap_sort(indices: &mut [Indices], distances: &mut [Array]) {
    for (indices, distances) in indices.iter_mut().zip(distances) {
        for j in 0..indices.len() {
            indices.swap(0, j);
            distances.swap(0, j);

            siftdown(indices, distances, 0);
        }
    }
}

#[inline]
fn siftdown(indices: &mut Indices, distances: &mut Array, mut pos: usize) {
    while (pos * 2 + 1) < indices.len() {
        let left_child = pos * 2 + 1;
        let right_child = left_child + 1;
        let mut swap = pos;

        if indices[swap] < indices[left_child] {
            swap = left_child;
        }

        if right_child < indices.len() && (indices[swap] < indices[right_child]) {
            swap = right_child;
        }

        if swap == pos {
            break;
        }

        indices.swap(swap, pos);
        distances.swap(swap, pos);
        pos = swap;
    }
}

/// Inserts a value into the heap and re-orders the heap.
pub fn checked_heap_push(
    priorities: &mut Array,
    indices: &mut [Option<usize>],
    p: f32,
    n: usize,
) -> bool {
    assert!(!priorities.is_empty());
    assert!(!indices.is_empty());

    unsafe { checked_heap_push_inner(priorities, indices, p, n) }
}

#[inline(always)]
unsafe fn checked_heap_push_inner(
    priorities: &mut Array,
    indices: &mut [Option<usize>],
    p: f32,
    n: usize,
) -> bool {
    if p >= *priorities.get_unchecked(0) {
        return false;
    }

    let size = priorities.len();

    // Break if we already have this element.
    if indices.iter().any(|v| *v == Some(n)) {
        return false;
    }

    (*priorities.get_unchecked_mut(0)) = p;
    (*indices.get_unchecked_mut(0)) = Some(n);

    // Descend the heap, swapping values until the max heap criterion is met
    let mut i = 0;
    loop {
        let i_swap = unsafe_get_swap_point!(p, i, size, priorities);

        (*priorities.get_unchecked_mut(i)) = *priorities.get_unchecked(i_swap);
        (*indices.get_unchecked_mut(i)) = *indices.get_unchecked(i_swap);

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];

        i = i_swap
    }

    (*priorities.get_unchecked_mut(i)) = p;
    (*indices.get_unchecked_mut(i)) = Some(n);

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn array(v: impl Into<Vec<f32>>) -> Array {
        unsafe { Array::new(v.into()) }
    }

    #[test]
    fn test_normalize() {
        let data = vec![vec![1.2, 3.4, 11.2]];
        let data = verify_and_normalize(data, true).unwrap();
        assert_eq!(
            data,
            [array(vec![0.10198832, 0.28896692, 0.95189095]),]
        )
    }

    #[test]
    fn test_rng_behaviour() {
        let value = tau_rand_int(&mut (1i64, 34i64, 22i64));
        assert_eq!(value, 2097664);

        let value = tau_rand(&mut (1i64, 34i64, 22i64));
        assert_eq!(value, 0.0009768009);
    }

    #[test]
    fn test_heap_logic() {
        let mut heap = Heap::new(3, 3); // 3 x 3 heap
        let mut distances = Vec::new();

        heap.distance_thresholds(&mut distances);
        assert_eq!(distances, [f32::INFINITY, f32::INFINITY, f32::INFINITY]);

        heap.checked_flagged_heap_push(0, 1.0, 0, true);
        heap.checked_flagged_heap_push(0, 3.0, 2, true);
        heap.checked_flagged_heap_push(0, 4.0, 3, true);

        heap.distance_thresholds(&mut distances);
        assert_eq!(distances, [4.0, f32::INFINITY, f32::INFINITY]);

        let mut heap = Heap::new(3, 3); // 3 x 3 heap

        heap.checked_flagged_heap_push(0, 3.3000002, 1, true);
        heap.checked_flagged_heap_push(0, 3.3000002, 1, true);
        heap.checked_flagged_heap_push(0, 3.3000002, 1, true);
        heap.checked_flagged_heap_push(0, 3.3000002, 1, true);
        heap.checked_flagged_heap_push(0, 5.1800003, 2, true);

        heap.checked_flagged_heap_push(1, 5.4100003, 2, true);
        heap.checked_flagged_heap_push(1, 3.3000002, 0, true);
        heap.checked_flagged_heap_push(1, 3.3000002, 0, true);
        heap.checked_flagged_heap_push(1, 5.4100003, 2, true);
        heap.checked_flagged_heap_push(1, 3.71, 1, true);

        heap.checked_flagged_heap_push(2, 5.1800003, 0, true);
        heap.checked_flagged_heap_push(2, 9.29, 2, true);
        heap.checked_flagged_heap_push(2, 9.29, 2, true);
        heap.checked_flagged_heap_push(2, 9.29, 2, true);
        heap.checked_flagged_heap_push(2, 5.1800003, 0, true);

        assert_eq!(
            heap.indices,
            [
                vec![None, Some(1), Some(2)],
                vec![Some(2), Some(1), Some(0)],
                vec![None, Some(0), Some(2)],
            ]
        );
        assert_eq!(
            heap.distances
                .iter()
                .map(|v| v.as_ref())
                .collect::<Vec<_>>(),
            [
                vec![f32::INFINITY, 3.3000002, 5.1800003],
                vec![5.4100003, 3.71, 3.3000002],
                vec![f32::INFINITY, 5.1800003, 9.29],
            ]
        );
        assert_eq!(
            heap.flags,
            [
                vec![false, true, true],
                vec![true, true, true],
                vec![false, true, true],
            ]
        );
    }

    #[test]
    fn test_siftdown() {
        let mut heap = Heap::new(3, 1); // 3 x 1 heap
        heap.indices[1] = vec![None, Some(1), None];
        heap.distances[1] = array(vec![f32::INFINITY, 3.3, f32::INFINITY]);

        siftdown(&mut heap.indices[1], &mut heap.distances[1], 0);
        assert_eq!(heap.indices[1], [Some(1), None, None]);
        assert_eq!(
            heap.distances[1],
            array(vec![3.3, f32::INFINITY, f32::INFINITY])
        );

        let mut heap = Heap::new(3, 1); // 3 x 1 heap
        heap.indices[1] = vec![None, None, Some(1)];
        heap.distances[1] = array(vec![f32::INFINITY, f32::INFINITY, 3.3]);

        siftdown(&mut heap.indices[1], &mut heap.distances[1], 0);
        assert_eq!(heap.indices[1], [Some(1), None, None]);
        assert_eq!(
            heap.distances[1],
            array(vec![3.3, f32::INFINITY, f32::INFINITY])
        );
    }

    #[test]
    fn test_deheap_sort() {
        let mut heap = Heap::new(3, 1); // 3 x 1 heap
        heap.indices[1] = vec![None, Some(1), None];
        heap.distances[1] = array(vec![f32::INFINITY, 3.3, f32::INFINITY]);

        deheap_sort(&mut heap.indices, &mut heap.distances);

        assert_eq!(heap.indices[1], [Some(1), None, None]);
        assert_eq!(
            heap.distances[1],
            array(vec![3.3, f32::INFINITY, f32::INFINITY])
        );
    }
}
