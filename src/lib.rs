use std::cmp;
use std::fmt::{Debug, Formatter};
use std::time::Instant;

use bumpalo::collections::Vec as BumpVec;
use hashbrown::HashSet;
use nndescent_math::AutoAlgorithms;
use rand::rngs::StdRng;
use rand::SeedableRng;
use smallvec::SmallVec;

use crate::distances::Distance;
use crate::rp_trees::{init_rp_tree, make_forest, rptree_leaf_array, LeafArray};
use crate::utils::{
    checked_heap_push,
    deheap_sort,
    tau_rand,
    tau_rand_int,
    BumpIndices,
    Heap,
    Indices,
    RngState,
};

mod distances;
mod rp_trees;
mod utils;

pub type Array = nndescent_math::Vector<AutoAlgorithms>;
const DELTA: f64 = 0.001;

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("The index requires at least one vector to be provided in order to build the index.")]
    NotEnoughData,
    #[error("Cannot build index using non-finite data.")]
    DataNoteFinite,
}

#[derive(Copy, Clone)]
pub enum Metric {
    Euclidean,
    SquaredEuclidean,
    Cosine,
    AlternativeCosine,
    Dot,
    AlternativeDot,
    Custom(fn(&Array, &Array) -> f32),
}

impl Debug for Metric {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Metric::Euclidean => write!(f, "Euclidean"),
            Metric::SquaredEuclidean => write!(f, "SquaredEuclidean"),
            Metric::Cosine => write!(f, "Cosine"),
            Metric::AlternativeCosine => write!(f, "AlternativeCosine"),
            Metric::Dot => write!(f, "Dot"),
            Metric::AlternativeDot => write!(f, "AlternativeDot"),
            Metric::Custom(_) => write!(f, "Custom"),
        }
    }
}

impl Metric {
    pub(crate) fn should_normalize(&self) -> bool {
        matches!(self, Self::Dot)
    }

    #[inline]
    pub(crate) fn angular_trees(&self) -> bool {
        matches!(
            self,
            Metric::Dot
                | Metric::AlternativeDot
                | Metric::Cosine
                | Metric::AlternativeCosine
        )
    }

    #[inline]
    pub(crate) fn as_distance(&self) -> Distance {
        match *self {
            Metric::Cosine => Distance::Cosine,
            Metric::AlternativeCosine => Distance::AlternativeCosine,
            Metric::Dot => Distance::Dot,
            Metric::AlternativeDot => Distance::AlternativeDot,
            Metric::Euclidean => Distance::Euclidean,
            Metric::SquaredEuclidean => Distance::SquaredEuclidean,
            Metric::Custom(cb) => Distance::Custom(cb),
        }
    }
}
/// NNDescent for fast approximate nearest neighbor queries.
///
/// NDescent is very flexible and supports a wide variety of distances, including
/// non-metric distances. NNDescent also scales well against high dimensional
/// graph_data in many cases. This implementation provides a straightforward
/// interface, with access to some tuning parameters.
///
/// This is a Rust port of the original Python implementation: https://github.com/lmcinnes/pynndescent
pub struct NNDescentBuilder {
    metric: Metric,
    data: Vec<Vec<f32>>,
    compressed: bool,
    rng_seed: Option<StdRng>,
    n_neighbors: usize,
    n_trees: Option<usize>,
}

impl Default for NNDescentBuilder {
    fn default() -> Self {
        Self {
            metric: Metric::Cosine,
            data: Vec::new(),
            compressed: false,
            rng_seed: None,
            n_neighbors: 30,
            n_trees: None,
        }
    }
}

impl NNDescentBuilder {
    /// Set the data to index.
    pub fn data(mut self, data: Vec<Vec<f32>>) -> Self {
        self.data = data;
        self
    }

    /// Enable index compression.
    pub fn compressed(mut self) -> Self {
        self.compressed = true;
        self
    }

    /// The RNG seed to use when generating the index.
    pub fn seed(mut self, seed: [u8; 32]) -> Self {
        self.rng_seed = Some(StdRng::from_seed(seed));
        self
    }

    ///  The number of neighbors to use in k-neighbor graph graph_data structure
    ///  used for fast approximate nearest neighbor search. Larger values
    ///  will result in more accurate search results at the cost of
    ///  computation time.
    ///
    ///  Defaults to `30`
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// This implementation uses random projection forests for initializing the index
    /// build process. This parameter controls the number of trees in that forest. A
    /// larger number will result in more accurate neighbor computation at the cost
    /// of performance. The default of None means a value will be chosen based on the
    /// size of the graph_data.
    pub fn n_trees(mut self, n: usize) -> Self {
        self.n_trees = Some(n);
        self
    }

    /// Set the given distance metric.
    ///
    /// Defaults to consine
    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Attempts to build the nndescent instance.
    pub fn try_build(self) -> Result<NNDescent, BuildError> {
        NNDescent::try_from_builder(self)
    }
}

pub struct NNDescent {
    pub metric: Metric,
    pub n_trees: usize,
    pub n_iters: usize,
    pub n_neighbors: usize,
    pub angular_trees: bool,
    pub random_state: RngState,
    pub search_random_state: RngState,
    pub neighbor_graph: (Vec<Indices>, Vec<Array>),
}

impl NNDescent {
    pub fn try_from_builder(mut builder: NNDescentBuilder) -> Result<Self, BuildError> {
        #[cfg(feature = "tracing")]
        nndescent_math::log_selected();

        if builder.data.is_empty() {
            return Err(BuildError::NotEnoughData);
        }
        let data = utils::verify_and_normalize(
            builder.data,
            builder.metric.should_normalize(),
        )?;

        let n_trees = builder.n_trees.unwrap_or_else(|| {
            let num_entries = data.len() as f32;
            let n_trees = 5 + (num_entries.powf(0.25).round() as usize);
            cmp::min(32, n_trees) // Only so many trees are useful
        });
        let n_iters = cmp::max(5, (data.len() as f32).log2().round() as usize);

        #[cfg(feature = "tracing")]
        tracing::debug!(metric = ?builder.metric, "Creating NN graphs for metric");

        let mut random_state = builder
            .rng_seed
            .as_mut()
            .map(utils::create_rand_state_from_rng)
            .unwrap_or_else(utils::create_rand_state);
        let mut search_random_state = builder
            .rng_seed
            .as_mut()
            .map(utils::create_rand_state_from_rng)
            .unwrap_or_else(utils::create_rand_state);

        for _ in 0..10 {
            tau_rand_int(&mut search_random_state);
        }
        #[cfg(feature = "tracing")]
        tracing::debug!("Rng is ready");

        let start = Instant::now();
        let forest = make_forest(
            &data,
            builder.n_neighbors,
            n_trees,
            &mut random_state,
            builder.metric.angular_trees(),
        );
        #[cfg(feature = "tracing")]
        tracing::debug!(elapsed = ?start.elapsed(), "Forest has been made");

        let start = Instant::now();
        let leaf_array = rptree_leaf_array(&forest);
        #[cfg(feature = "tracing")]
        tracing::debug!(elapsed = ?start.elapsed(), "Leaf array is built");

        let effective_max_candidates = cmp::min(60, builder.n_neighbors);

        let neighbor_graph = nn_descent(
            &data,
            builder.n_neighbors,
            &mut random_state,
            effective_max_candidates,
            builder.metric.as_distance(),
            n_iters,
            leaf_array,
        );

        Ok(Self {
            metric: builder.metric,
            n_trees,
            n_neighbors: builder.n_neighbors,
            n_iters,
            angular_trees: builder.metric.angular_trees(),
            random_state,
            search_random_state,
            neighbor_graph,
        })
    }

    fn init_search_graph(&mut self) {}
}

#[allow(unused_variables)]
fn nn_descent(
    data: &[Array],
    n_neighbors: usize,
    random_state: &mut RngState,
    max_candidates: usize,
    dist: Distance,
    n_iters: usize,
    leaf_array: Option<LeafArray>,
) -> (Vec<Indices>, Vec<Array>) {
    let mut current_graph = Heap::new(data.len(), n_neighbors);

    let start = Instant::now();
    init_rp_tree(data, dist, &mut current_graph, &leaf_array);
    #[cfg(feature = "tracing")]
    tracing::debug!(elapsed = ?start.elapsed(), "RP tree created");

    let start = Instant::now();
    init_random(n_neighbors, data, &mut current_graph, dist, random_state);
    #[cfg(feature = "tracing")]
    tracing::debug!(elapsed = ?start.elapsed(), "Random graph init complete");

    let start = Instant::now();
    nn_descent_high_memory_internal(
        &mut current_graph,
        data,
        n_neighbors,
        random_state,
        max_candidates,
        dist,
        n_iters,
    );

    #[cfg(feature = "tracing")]
    tracing::debug!(elapsed = ?start.elapsed(), "Completed build");

    let start = Instant::now();
    let mut indices = current_graph.indices;
    let mut distances = current_graph.distances;
    deheap_sort(&mut indices, &mut distances);

    #[cfg(feature = "tracing")]
    tracing::debug!(elapsed = ?start.elapsed(), "Graph unpack complete");

    (indices, distances)
}

fn nn_descent_high_memory_internal(
    current_graph: &mut Heap,
    data: &[Array],
    n_neighbors: usize,
    random_state: &mut RngState,
    max_candidates: usize,
    dist: Distance,
    n_iters: usize,
) {
    const BLOCK_SIZE: usize = 16384;

    let n_vertices = data.len();
    let n_blocks = n_vertices / BLOCK_SIZE;
    let mut bump = bumpalo::Bump::with_capacity(500 << 20);

    let mut distance_thresholds = Vec::new();
    let mut updates = SmallVec::new();
    let mut in_graph = Vec::with_capacity(current_graph.indices.len());
    for indices in current_graph.indices.iter() {
        let mut set = HashSet::with_capacity(indices.len());
        for val in indices {
            set.insert(*val);
        }
        in_graph.push(set);
    }

    #[allow(unused_variables)]
    let mut total_changes = 0;
    #[allow(unused_variables)]
    for n in 0..n_iters {
        #[cfg(feature = "tracing")]
        tracing::debug!(n = n, n_iters = n_iters, "Beginning iteration");

        let mut c = 0;
        {
            let (new_candidate_neighbors, old_candidate_neighbors) =
                build_candidates(&bump, current_graph, max_candidates, random_state);

            for i in 0..n_blocks + 1 {
                let block_start = i * BLOCK_SIZE;
                let block_end = cmp::min(n_vertices, (i + 1) * BLOCK_SIZE);

                let new_candidate_block =
                    &new_candidate_neighbors[block_start..block_end];
                let old_candidate_block =
                    &old_candidate_neighbors[block_start..block_end];
                current_graph.distance_thresholds(&mut distance_thresholds);

                generate_graph_updates(
                    &mut updates,
                    new_candidate_block,
                    old_candidate_block,
                    &distance_thresholds,
                    data,
                    dist,
                );

                c += apply_graph_updates_high_memory(
                    current_graph,
                    &mut updates,
                    &mut in_graph,
                );
            }
            total_changes += c;
        }

        bump.reset();

        if c as f64 <= DELTA * n_neighbors as f64 * data.len() as f64 {
            #[cfg(feature = "tracing")]
            tracing::debug!(
                exit_at_n = n + 1,
                total_changes = total_changes,
                "Stopping threshold met"
            );
            return;
        }
    }
}

fn apply_graph_updates_high_memory(
    current_graph: &mut Heap,
    updates: &mut SmallVec<[(usize, usize, f32); 24]>,
    in_graph: &mut [HashSet<Option<usize>>],
) -> usize {
    let mut n_changes = 0;

    for (p, q, d) in updates.drain(..) {
        unsafe {
            let q_in_p = in_graph.get_unchecked(p).contains(&Some(q));
            let p_in_q = in_graph.get_unchecked(q).contains(&Some(p));

            if q_in_p && p_in_q {
                continue;
            }

            if !q_in_p {
                let added = current_graph.checked_flagged_heap_push(p, d, q, true);

                if added {
                    in_graph.get_unchecked_mut(p).insert(Some(q));
                    n_changes += 1;
                }
            }

            if !(p == q || p_in_q) {
                let added = current_graph.checked_flagged_heap_push(p, d, q, true);

                if added {
                    in_graph.get_unchecked_mut(q).insert(Some(p));
                    n_changes += 1;
                }
            }
        }
    }

    n_changes
}

// TODO: This can be made into an iterator if someone has the will to do so.
fn generate_graph_updates(
    updates: &mut SmallVec<[(usize, usize, f32); 24]>,
    new_candidate_block: &[BumpIndices<'_>],
    old_candidate_block: &[BumpIndices<'_>],
    distance_thresholds: &[f32],
    data: &[Array],
    dist: Distance,
) {
    let iter = new_candidate_block.iter().zip(old_candidate_block);
    for (new_block, old_block) in iter {
        let max_candidates = new_block.len();

        for (j, p) in new_block.iter().enumerate() {
            let p = match p {
                None => continue,
                Some(p) => *p,
            };

            for q in &new_block[j..max_candidates] {
                let q = match q {
                    None => continue,
                    Some(q) => *q,
                };

                let d = dist.get_distance(&data[p], &data[q]);
                if d <= distance_thresholds[p] || d <= distance_thresholds[q] {
                    updates.push((p, q, d));
                }
            }

            for q in &old_block[..max_candidates] {
                let q = match q {
                    None => continue,
                    Some(q) => *q,
                };

                let d = dist.get_distance(&data[p], &data[q]);
                if d <= distance_thresholds[p] || d <= distance_thresholds[q] {
                    updates.push((p, q, d));
                }
            }
        }
    }
}

fn build_candidates<'b>(
    bump: &'b bumpalo::Bump,
    current_graph: &mut Heap,
    max_candidates: usize,
    random_state: &mut RngState,
) -> (BumpVec<'b, BumpIndices<'b>>, BumpVec<'b, BumpIndices<'b>>) {
    let n_vertices = current_graph.indices.len();

    let mut new_candidate_indices = BumpVec::new_in(bump);
    new_candidate_indices.reserve(n_vertices);
    for _ in 0..n_vertices {
        let mut inner = BumpVec::new_in(bump);
        inner.reserve(max_candidates);
        for _ in 0..max_candidates {
            inner.push(None);
        }
        new_candidate_indices.push(inner);
    }
    let mut new_candidate_priority = BumpVec::new_in(bump);
    new_candidate_priority.reserve(n_vertices);
    for _ in 0..n_vertices {
        new_candidate_priority.push(Array::from_element(max_candidates, f32::INFINITY));
    }

    let mut old_candidate_indices = BumpVec::new_in(bump);
    old_candidate_indices.reserve(n_vertices);
    for _ in 0..n_vertices {
        let mut inner = BumpVec::new_in(bump);
        inner.reserve(max_candidates);
        for _ in 0..max_candidates {
            inner.push(None);
        }
        old_candidate_indices.push(inner);
    }
    let mut old_candidate_priority = BumpVec::new_in(bump);
    old_candidate_priority.reserve(n_vertices);
    for _ in 0..n_vertices {
        old_candidate_priority.push(Array::from_element(max_candidates, f32::INFINITY));
    }

    for (i, (indices, flags)) in current_graph
        .indices
        .iter()
        .zip(current_graph.flags.iter())
        .enumerate()
    {
        for (&idx, &isn) in indices.iter().zip(flags) {
            let idx = match idx {
                None => continue,
                Some(idx) => idx,
            };

            let d = tau_rand(random_state);

            if isn {
                checked_heap_push(
                    &mut new_candidate_priority[i],
                    &mut new_candidate_indices[i],
                    d,
                    idx,
                );
                checked_heap_push(
                    &mut new_candidate_priority[idx],
                    &mut new_candidate_indices[idx],
                    d,
                    i,
                );
            } else {
                checked_heap_push(
                    &mut old_candidate_priority[i],
                    &mut old_candidate_indices[i],
                    d,
                    idx,
                );
                checked_heap_push(
                    &mut old_candidate_priority[idx],
                    &mut old_candidate_indices[idx],
                    d,
                    i,
                );
            }
        }
    }

    let iter = current_graph
        .indices
        .iter()
        .zip(new_candidate_indices.iter())
        .zip(current_graph.flags.iter_mut());

    for ((indices, new_indices), flags) in iter {
        for (idx, flag) in indices.iter().zip(flags.iter_mut()) {
            for new_idx in new_indices.iter() {
                if new_idx == idx {
                    (*flag) = false;
                    break;
                }
            }
        }
    }

    (new_candidate_indices, old_candidate_indices)
}

fn init_random(
    n_neighbors: usize,
    data: &[Array],
    heap: &mut Heap,
    dist: Distance,
    random_state: &mut RngState,
) {
    for (i, array) in data.iter().enumerate() {
        let indices = &heap.indices[i];
        if indices[0].is_none() {
            let count = indices.iter().filter(|v| v.is_some()).count();
            let n = n_neighbors - count;

            for _ in 0..n {
                let rng_val = tau_rand_int(random_state);
                let idx = rng_val.unsigned_abs() as usize % data.len();
                let d = dist.get_distance(&data[idx], array);
                heap.checked_flagged_heap_push(i, d, idx, true);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;

    fn array(v: impl Into<Vec<f32>>) -> Array {
        unsafe { Array::new(v.into()) }
    }

    /// Just does left.dot(right)
    fn custom_dist(left: &Array, right: &Array) -> f32 {
        left.dot(right)
    }

    fn to_vec_from_bump<'b, V, T>(v: V) -> Vec<Vec<T>>
    where
        T: 'b + Clone,
        V: AsRef<[BumpVec<'b, T>]>,
    {
        let mut data = Vec::new();
        let slice = v.as_ref();
        for nested in slice {
            data.push(nested.to_vec());
        }

        data
    }

    #[test]
    fn test_init_random() {
        let data = [
            array(vec![1.2, 1.0, 0.8]),
            array(vec![1.5, 1.1, 0.5]),
            array(vec![2.2, 1.1, 1.8]),
        ];

        let mut heap = Heap::new(3, 3);
        init_random(
            3,
            &data,
            &mut heap,
            Distance::Custom(custom_dist),
            &mut (1i64, 34i64, 22i64),
        );
        assert_eq!(
            heap.indices,
            [
                vec![None, Some(1), None],
                vec![None, Some(1), Some(2)],
                vec![None, Some(0), Some(2)],
            ]
        );
        assert_eq!(
            heap.distances
                .iter()
                .map(|v| v.as_ref())
                .collect::<Vec<_>>(),
            [
                vec![f32::INFINITY, 3.3000002, f32::INFINITY],
                vec![f32::INFINITY, 3.71, 5.4100003],
                vec![f32::INFINITY, 5.1800003, 9.29],
            ]
        );
        assert_eq!(
            heap.flags,
            [
                vec![false, true, false],
                vec![false, true, true],
                vec![false, true, true],
            ]
        );

        let mut heap = Heap::new(3, 3);
        init_random(
            1,
            &data,
            &mut heap,
            Distance::Custom(custom_dist),
            &mut (1i64, 34i64, 22i64),
        );
        assert_eq!(
            heap.indices,
            [
                vec![None, Some(1), None],
                vec![None, Some(1), None],
                vec![None, Some(1), None],
            ]
        );
        assert_eq!(
            heap.distances
                .iter()
                .map(|v| v.as_ref())
                .collect::<Vec<_>>(),
            [
                vec![f32::INFINITY, 3.3000002, f32::INFINITY],
                vec![f32::INFINITY, 3.71, f32::INFINITY],
                vec![f32::INFINITY, 5.4100003, f32::INFINITY],
            ]
        );
        assert_eq!(
            heap.flags,
            [
                vec![false, true, false],
                vec![false, true, false],
                vec![false, true, false],
            ]
        );

        let mut heap = Heap::new(3, 3);
        init_random(
            5,
            &data,
            &mut heap,
            Distance::Custom(custom_dist),
            &mut (1i64, 34i64, 22i64),
        );
        assert_eq!(
            heap.indices,
            [
                vec![None, Some(1), Some(2)],
                vec![Some(2), Some(1), Some(0)],
                vec![Some(2), Some(0), Some(1)],
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
                vec![9.29, 5.1800003, 5.4100003],
            ]
        );
        assert_eq!(
            heap.flags,
            [
                vec![false, true, true],
                vec![true, true, true],
                vec![true, true, true],
            ]
        );
    }

    #[test]
    fn test_build_candidates() {
        let data = [
            array(vec![1.2, 1.0, 0.8]),
            array(vec![1.5, 1.1, 0.5]),
            array(vec![2.2, 1.1, 1.8]),
        ];
        let mut heap = Heap::new(3, 3);
        init_random(
            5,
            &data,
            &mut heap,
            Distance::Custom(custom_dist),
            &mut (1i64, 34i64, 22i64),
        );

        let bump = bumpalo::Bump::new();
        let (left, right) =
            build_candidates(&bump, &mut heap, 3, &mut (1i64, 34i64, 22i64));

        assert_eq!(
            to_vec_from_bump(left),
            [
                vec![None, Some(1), Some(2)],
                vec![Some(2), Some(0), Some(1)],
                vec![Some(2), Some(0), Some(1)],
            ]
        );
        assert_eq!(
            to_vec_from_bump(right),
            [
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, None],
            ]
        );
        // Check it reset the flags.
        assert_eq!(
            heap.flags,
            [
                vec![false, false, false],
                vec![false, false, false],
                vec![false, false, false],
            ]
        );

        let mut heap = Heap::new(3, 3);
        init_random(
            1,
            &data,
            &mut heap,
            Distance::Custom(custom_dist),
            &mut (1i64, 34i64, 22i64),
        );

        let bump = bumpalo::Bump::new();
        let (left, right) =
            build_candidates(&bump, &mut heap, 3, &mut (1i64, 34i64, 22i64));

        assert_eq!(
            to_vec_from_bump(left),
            [
                vec![None, Some(1), None],
                vec![Some(2), Some(0), Some(1)],
                vec![None, Some(1), None],
            ]
        );
        assert_eq!(
            to_vec_from_bump(right),
            [
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, None],
            ]
        );
        // Check it reset the flags.
        assert_eq!(
            heap.flags,
            [
                vec![false, false, false],
                vec![false, false, false],
                vec![false, false, false],
            ]
        );

        let bump = bumpalo::Bump::new();
        let (left, right) =
            build_candidates(&bump, &mut heap, 3, &mut (1i64, 34i64, 22i64));
        assert_eq!(
            to_vec_from_bump(left),
            [
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, None],
            ]
        );
        assert_eq!(
            to_vec_from_bump(right),
            [
                vec![None, Some(1), None],
                vec![Some(2), Some(0), Some(1)],
                vec![None, Some(1), None],
            ]
        );
    }

    #[test]
    fn test_generate_graph_updates() {
        let data = [
            array(vec![1.2, 3.4, 11.2, 4.6432, 5.2]),
            array(vec![1.23, 3.8, 12.2, 4.2, 0.3452]),
            array(vec![3.8, 6.4, 9.2, 3.0, 56.2]),
        ];
        let mut state = (1i64, 34i64, 22i64);
        let mut heap = Heap::new(3, 3);
        init_random(
            3,
            &data,
            &mut heap,
            Distance::Custom(custom_dist),
            &mut state,
        );

        let bump = bumpalo::Bump::new();
        let (new_candidates, old_candidates) =
            build_candidates(&bump, &mut heap, 3, &mut state);
        let mut distance_thresholds = Vec::new();
        let mut updates = SmallVec::new();
        heap.distance_thresholds(&mut distance_thresholds);
        generate_graph_updates(
            &mut updates,
            &new_candidates,
            &old_candidates,
            &distance_thresholds,
            &data,
            Distance::Custom(custom_dist),
        );

        let expected: SmallVec<[(usize, usize, f32); 24]> = smallvec![
            (1, 1, 182.55205),
            (1, 2, 173.23424),
            (2, 2, 3307.4802),
            (2, 2, 3307.4802),
            (2, 0, 435.52957),
            (2, 1, 173.23424),
            (0, 0, 187.0393),
            (0, 1, 172.33247),
            (1, 1, 182.55205),
            (0, 0, 187.0393),
            (0, 1, 172.33247),
            (0, 2, 435.52957),
            (1, 1, 182.55205),
            (1, 2, 173.23424),
            (2, 2, 3307.4802),
        ];

        assert_eq!(updates, expected);
    }
}
