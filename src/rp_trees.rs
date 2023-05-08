use std::cmp;
use std::time::Instant;

use bumpalo::collections::Vec as BumpVec;
use smallvec::SmallVec;

use crate::utils::{tau_rand_int, Heap, RngState};
use crate::{Array, Distance};

const EPS: f32 = 1e-8;

pub type LeafArray = SmallVec<[Vec<usize>; 64]>;

pub(crate) fn init_rp_tree(
    data: &[Array],
    dist: Distance,
    graph: &mut Heap,
    leaf_array: &Option<LeafArray>,
) {
    const BLOCK_SIZE: usize = 65536;

    let leaf_array = leaf_array.as_deref().unwrap_or(&[]);
    let n_leaves = leaf_array.len();
    let n_blocks = n_leaves / BLOCK_SIZE;

    let mut distance_thresholds = Vec::new();
    let mut updates = SmallVec::new();
    // TODO: This is bugged because our leaf array is not populated.
    for i in 0..n_blocks + 1 {
        let block_start = i * BLOCK_SIZE;
        let block_end = cmp::min(n_leaves, (i + 1) * BLOCK_SIZE);

        let leaf_block = &leaf_array[block_start..block_end];
        graph.distance_thresholds(&mut distance_thresholds);

        // TODO: Go back to the iterator once it works
        generate_leaf_updates(
            &mut updates,
            leaf_block,
            &distance_thresholds,
            data,
            dist,
        );
        for (point_a, point_b, distance) in updates.drain(..) {
            graph.checked_flagged_heap_push(point_a, distance, point_b, true);
            graph.checked_flagged_heap_push(point_b, distance, point_a, true);
        }
    }
}

fn generate_leaf_updates(
    updates: &mut SmallVec<[(usize, usize, f32); 24]>,
    leaf_blocks: &[Vec<usize>],
    distance_thresholds: &[f32],
    data: &[Array],
    distance: Distance,
) {
    for block in leaf_blocks {
        for (i, &p) in block.iter().enumerate() {
            unsafe {
                for &q in block.get_unchecked(i + 1..) {
                    let d = distance
                        .get_distance(data.get_unchecked(p), data.get_unchecked(q));

                    if d < *distance_thresholds.get_unchecked(p)
                        || d < *distance_thresholds.get_unchecked(q)
                    {
                        updates.push((p, q, d));
                    }
                }
            }
        }
    }
}

/// Builds the RP tree leaf array.
///
/// This differs from the python implementation as we do not create and
/// populate the array all at once, instead we just reserve some of the allocations
/// and then push data to the vector as it's more efficient this way.
pub(crate) fn rptree_leaf_array(forest: &[FlatTree]) -> Option<LeafArray> {
    if forest.is_empty() {
        return None;
    }

    let mut leaves = LeafArray::with_capacity(forest.len());
    for tree in forest {
        let iter = tree.indices.iter().zip(tree.children.iter());
        for (indices, children) in iter {
            if children.is_some() {
                continue;
            }

            if let Some(indices) = indices {
                leaves.push(indices.clone());
            }
        }
    }

    Some(leaves)
}

/// Builds a new FlatTree forest.
///
/// This does not build in parallel but probably can
/// in future editions easily using rayon.
pub(crate) fn make_forest(
    data: &[Array],
    n_neighbors: usize,
    n_trees: usize,
    random_state: &mut RngState,
    angular: bool,
) -> SmallVec<[FlatTree; 64]> {
    #[cfg(feature = "tracing")]
    tracing::debug!(
        n_trees = n_trees,
        n_neighbors = n_neighbors,
        angular = angular,
        "Building RP forest"
    );

    let leaf_size = cmp::max(10, n_neighbors);

    // We use a bump allocator here to avoid doing constant allocations
    // for each tree and it's indices.
    let mut bump = bumpalo::Bump::with_capacity(512 << 10);
    let mut forest = SmallVec::with_capacity(n_trees);
    for _ in 0..n_trees {
        let tree = make_dense_tree(&bump, data, leaf_size, random_state, angular);
        forest.push(tree);
        bump.reset();
    }

    forest
}

#[derive(Debug, Default)]
pub struct FlatTree {
    pub hyperplanes: Vec<Option<Array>>,
    pub offsets: Vec<Option<f64>>,
    pub children: Vec<Option<(usize, usize)>>,
    pub indices: Vec<Option<Vec<usize>>>,
    pub leaf_size: usize,
}

#[inline]
fn make_dense_tree(
    bump: &bumpalo::Bump,
    data: &[Array],
    leaf_size: usize,
    random_state: &mut RngState,
    angular: bool,
) -> FlatTree {
    let indices = BumpVec::from_iter_in(0..data.len(), bump);
    let mut tree = FlatTree::default();

    if angular {
        make_angular_tree(
            bump,
            data,
            leaf_size,
            100,
            random_state,
            &mut tree,
            &indices,
        );
    } else {
        make_euclidean_tree(
            bump,
            data,
            leaf_size,
            100,
            random_state,
            &mut tree,
            &indices,
        );
    }

    tree
}

#[inline]
fn make_angular_tree(
    bump: &bumpalo::Bump,
    data: &[Array],
    leaf_size: usize,
    max_depth: usize,
    random_state: &mut RngState,
    tree: &mut FlatTree,
    indices: &[usize],
) {
    if indices.len() > leaf_size && max_depth > 0 {
        let (left_indices, right_indices, hyperplane, offset) =
            angular_random_projection_split(bump, data, indices, random_state);

        make_angular_tree(
            bump,
            data,
            leaf_size,
            max_depth - 1,
            random_state,
            tree,
            &left_indices,
        );

        let left_node_num = tree.indices.len() - 1;

        make_angular_tree(
            bump,
            data,
            leaf_size,
            max_depth - 1,
            random_state,
            tree,
            &right_indices,
        );

        let right_node_num = tree.indices.len() - 1;

        tree.hyperplanes.push(Some(hyperplane));
        tree.offsets.push(Some(offset));
        tree.children.push(Some((left_node_num, right_node_num)));
        tree.indices.push(None);
    } else {
        tree.hyperplanes.push(None);
        tree.offsets.push(None);
        tree.children.push(None);
        tree.indices.push(Some(indices.to_vec()));
    }
}

#[inline]
fn make_euclidean_tree(
    bump: &bumpalo::Bump,
    data: &[Array],
    leaf_size: usize,
    max_depth: usize,
    random_state: &mut RngState,
    tree: &mut FlatTree,
    indices: &[usize],
) {
    if indices.len() > leaf_size && max_depth > 0 {
        let (left_indices, right_indices, hyperplane, offset) =
            euclidean_random_projection_split(bump, data, indices, random_state);

        make_euclidean_tree(
            bump,
            data,
            leaf_size,
            max_depth - 1,
            random_state,
            tree,
            &left_indices,
        );

        let left_node_num = tree.indices.len() - 1;

        make_euclidean_tree(
            bump,
            data,
            leaf_size,
            max_depth - 1,
            random_state,
            tree,
            &right_indices,
        );

        let right_node_num = tree.indices.len() - 1;

        tree.hyperplanes.push(Some(hyperplane));
        tree.offsets.push(Some(offset));
        tree.children.push(Some((left_node_num, right_node_num)));
        tree.indices.push(None);
    } else {
        tree.hyperplanes.push(None);
        tree.offsets.push(None);
        tree.children.push(None);
        tree.indices.push(Some(indices.to_vec()));
    }
}

/// Given a set of `graph_indices` for graph_data points from `graph_data`, create
/// a random hyperplane to split the graph_data, returning two arrays graph_indices
/// that fall on either side of the hyperplane. This is the basis for a
/// random projection tree, which simply uses this splitting recursively.
/// This particular split uses cosine distance to determine the hyperplane
/// and which side each graph_data sample falls on.
fn angular_random_projection_split<'b>(
    bump: &'b bumpalo::Bump,
    data: &[Array],
    indices: &[usize],
    random_state: &mut RngState,
) -> (BumpVec<'b, usize>, BumpVec<'b, usize>, Array, f64) {
    let left_index = tau_rand_int(random_state).unsigned_abs() as usize % indices.len();
    let mut right_index =
        tau_rand_int(random_state).unsigned_abs() as usize % indices.len();

    if left_index == right_index {
        right_index += 1;
    }
    right_index %= indices.len();

    let left = unsafe { *indices.get_unchecked(left_index) };
    let right = unsafe { *indices.get_unchecked(right_index) };
    let left_data = unsafe { data.get_unchecked(left) };
    let right_data = unsafe { data.get_unchecked(right) };

    let hyperplane_array = left_data.angular_hyperplane(right_data);

    let (indices_left, indices_right) =
        select_side(bump, data, random_state, indices, &hyperplane_array, 0.0);

    (indices_left, indices_right, hyperplane_array, 0.0)
}

/// Given a set of `graph_indices` for graph_data points from `graph_data`, create
/// a random hyperplane to split the graph_data, returning two arrays graph_indices
/// that fall on either side of the hyperplane. This is the basis for a
/// random projection tree, which simply uses this splitting recursively.
/// This particular split uses euclidean distance to determine the hyperplane
/// and which side each graph_data sample falls on.
fn euclidean_random_projection_split<'b>(
    bump: &'b bumpalo::Bump,
    data: &[Array],
    indices: &[usize],
    random_state: &mut RngState,
) -> (BumpVec<'b, usize>, BumpVec<'b, usize>, Array, f64) {
    let left_index = tau_rand_int(random_state) as usize % indices.len();
    let mut right_index = tau_rand_int(random_state) as usize % indices.len();

    if left_index == right_index {
        right_index += 1;
    }
    right_index %= indices.len();

    let left = unsafe { *indices.get_unchecked(left_index) }; // TODO: Remove
    let right = unsafe { *indices.get_unchecked(right_index) };
    let left_data = unsafe { data.get_unchecked(left) };
    let right_data = unsafe { data.get_unchecked(right) };

    // TODO: Make this part of pyke-math like the angular system.
    let mut hyperplane_offset = 0.0;
    let mut hyperplane_array = Vec::with_capacity(left_data.len());
    for (l, r) in left_data.iter().zip(right_data.iter()) {
        let value = l - r;
        hyperplane_offset -= value * (l + r) / 2.0;
        hyperplane_array.push(value);
    }
    let hyperplane_array = unsafe { Array::new(hyperplane_array) };

    let (indices_left, indices_right) = select_side(
        bump,
        data,
        random_state,
        indices,
        &hyperplane_array,
        hyperplane_offset,
    );

    (
        indices_left,
        indices_right,
        hyperplane_array,
        hyperplane_offset as f64,
    )
}

#[inline]
fn select_side<'b>(
    bump: &'b bumpalo::Bump,
    data: &[Array],
    random_state: &mut RngState,
    indices: &[usize],
    hyperplane_array: &Array,
    hyperplane_offset: f32,
) -> (BumpVec<'b, usize>, BumpVec<'b, usize>) {
    // TODO: We may be able to restructure this so we're not doing the separate iterations.

    // For each point compute the margin (project into normal vector)
    // If we are on lower side of the hyperplane put in one pile, otherwise
    // put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    let mut n_left = 0;
    let mut n_right = 0;
    let mut side = BumpVec::new_in(bump);
    side.reserve(indices.len());
    for pos in indices.iter().copied() {
        // TODO: Check if this is correct.
        let margin =
            hyperplane_offset + hyperplane_array.dot(unsafe { data.get_unchecked(pos) });

        let side_flag = if margin.abs() < EPS {
            let side_flag = (tau_rand_int(random_state) % 2) as u8;
            if side_flag == 0 {
                n_left += 1;
            } else {
                n_right += 1;
            }
            side_flag
        } else if margin > 0.0 {
            n_left += 1;
            0
        } else {
            n_right += 1;
            1
        };

        side.push(side_flag);
    }

    // If all points end up on one side, something went wrong numerically
    // In this case, assign points randomly; they are likely very close anyway
    if n_left == 0 || n_right == 0 {
        n_left = 0;
        n_right = 0;

        for side_mut in side.iter_mut() {
            (*side_mut) = (tau_rand_int(random_state) % 2) as u8;
            if *side_mut == 0 {
                n_left += 1;
            } else {
                n_right += 1;
            }
        }
    }

    // Now that we have the counts allocate arrays
    let mut indices_left = BumpVec::new_in(bump);
    indices_left.reserve(n_left);
    let mut indices_right = BumpVec::new_in(bump);
    indices_right.reserve(n_right);

    // Populate the arrays with graph_indices according to which side they fell on
    for (i, v) in side.into_iter().enumerate() {
        unsafe {
            if v == 0 {
                indices_left.push(*indices.get_unchecked(i));
            } else {
                indices_right.push(*indices.get_unchecked(i));
            }
        }
    }

    (indices_left, indices_right)
}

#[cfg(test)]
mod tests {
    use bumpalo::Bump;
    use smallvec::smallvec;

    use super::*;

    fn array(v: impl Into<Vec<f32>>) -> Array {
        unsafe { Array::new(v.into()) }
    }

    #[test]
    fn test_make_angular_tree() {
        let data = &[
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.0),
        ];
        let bump = Bump::new();
        let indices = (0..data.len()).collect::<Vec<_>>();
        let mut tree = FlatTree::default();
        make_angular_tree(
            &bump,
            data,
            3,
            100,
            &mut (1i64, 34i64, 22i64),
            &mut tree,
            &indices,
        );
        assert_eq!(
            tree.hyperplanes,
            [None, None, Some(Array::from_element(5, 0.0f32))]
        );
        assert_eq!(tree.offsets, [None, None, Some(0.0f64)]);
        assert_eq!(tree.children, [None, None, Some((0, 1))]);
        assert_eq!(tree.indices, [Some(vec![0, 1, 2]), Some(vec![3]), None]);

        let data = &[
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.0),
        ];
        let bump = Bump::new();
        let indices = (0..data.len()).collect::<Vec<_>>();
        let mut tree = FlatTree::default();
        make_angular_tree(
            &bump,
            data,
            1,
            100,
            &mut (1i64, 34i64, 22i64),
            &mut tree,
            &indices,
        );
        assert_eq!(
            tree.hyperplanes,
            [
                None,
                None,
                None,
                Some(Array::from_element(5, 0.0f32)),
                None,
                Some(Array::from_element(5, 0.0f32)),
                Some(Array::from_element(5, 0.0f32)),
                None,
                Some(Array::from_element(5, 0.0f32)),
                None,
                Some(Array::from_element(5, 0.0f32)),
            ]
        );
        assert_eq!(
            tree.offsets,
            [
                None,
                None,
                None,
                Some(0.0f64),
                None,
                Some(0.0),
                Some(0.0),
                None,
                Some(0.0),
                None,
                Some(0.0)
            ]
        );
        assert_eq!(
            tree.children,
            [
                None,
                None,
                None,
                Some((1, 2)),
                None,
                Some((3, 4)),
                Some((0, 5)),
                None,
                Some((6, 7)),
                None,
                Some((8, 9))
            ]
        );
        assert_eq!(
            tree.indices,
            [
                Some(vec![]),
                Some(vec![0]),
                Some(vec![2]),
                None,
                Some(vec![]),
                None,
                None,
                Some(vec![1]),
                None,
                Some(vec![3]),
                None,
            ]
        );

        let data = &[
            array(vec![0.20f32, 0.45f32, 1.2f32, 0.1f32, 4.0f32]),
            Array::from_element(5, 1.2),
            Array::from_element(5, 3.0),
            Array::from_element(5, 0.9),
        ];
        let bump = Bump::new();
        let indices = (0..data.len()).collect::<Vec<_>>();
        let mut tree = FlatTree::default();
        make_angular_tree(
            &bump,
            data,
            3,
            100,
            &mut (1i64, 34i64, 22i64),
            &mut tree,
            &indices,
        );
        assert_eq!(
            tree.hyperplanes,
            [
                None,
                None,
                Some(array(vec![
                    -0.46625,
                    -0.39691254,
                    -0.1889002,
                    -0.493985,
                    0.5876792
                ]))
            ]
        );
        assert_eq!(tree.offsets, [None, None, Some(0.0)]);
        assert_eq!(tree.children, [None, None, Some((0, 1))]);
        assert_eq!(tree.indices, [Some(vec![0]), Some(vec![1, 2, 3]), None]);
    }

    #[test]
    fn test_make_euclidian_tree() {
        let data = &[
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.0),
        ];

        let bump = Bump::new();
        let indices = (0..data.len()).collect::<Vec<_>>();
        let mut tree = FlatTree::default();
        make_euclidean_tree(
            &bump,
            data,
            3,
            100,
            &mut (1i64, 34i64, 22i64),
            &mut tree,
            &indices,
        );
        assert_eq!(
            tree.hyperplanes,
            [None, None, Some(Array::from_element(5, 0.0f32))]
        );
        assert_eq!(tree.offsets, [None, None, Some(0.0f64)]);
        assert_eq!(tree.children, [None, None, Some((0, 1))]);
        assert_eq!(tree.indices, [Some(vec![0, 1, 2]), Some(vec![3]), None]);

        let data = &[
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.0),
        ];

        let bump = Bump::new();
        let indices = (0..data.len()).collect::<Vec<_>>();
        let mut tree = FlatTree::default();
        make_euclidean_tree(
            &bump,
            data,
            1,
            100,
            &mut (1i64, 34i64, 22i64),
            &mut tree,
            &indices,
        );
        assert_eq!(
            tree.hyperplanes,
            [
                None,
                None,
                None,
                Some(Array::from_element(5, 0.0f32)),
                None,
                Some(Array::from_element(5, 0.0f32)),
                Some(Array::from_element(5, 0.0f32)),
                None,
                Some(Array::from_element(5, 0.0f32)),
                None,
                Some(Array::from_element(5, 0.0f32)),
            ]
        );
        assert_eq!(
            tree.offsets,
            [
                None,
                None,
                None,
                Some(0.0f64),
                None,
                Some(0.0),
                Some(0.0),
                None,
                Some(0.0),
                None,
                Some(0.0)
            ]
        );
        assert_eq!(
            tree.children,
            [
                None,
                None,
                None,
                Some((1, 2)),
                None,
                Some((3, 4)),
                Some((0, 5)),
                None,
                Some((6, 7)),
                None,
                Some((8, 9))
            ]
        );
        assert_eq!(
            tree.indices,
            [
                Some(vec![]),
                Some(vec![0]),
                Some(vec![2]),
                None,
                Some(vec![]),
                None,
                None,
                Some(vec![1]),
                None,
                Some(vec![3]),
                None,
            ]
        );

        let data = &[
            Array::from_element(5, 1.0),
            Array::from_element(5, 1.2),
            Array::from_element(5, 3.0),
            Array::from_element(5, 0.9),
        ];
        let indices = (0..data.len()).collect::<Vec<_>>();
        let mut tree = FlatTree::default();
        make_euclidean_tree(
            &bump,
            data,
            3,
            100,
            &mut (1i64, 34i64, 22i64),
            &mut tree,
            &indices,
        );
        assert_eq!(
            tree.hyperplanes,
            [None, None, Some(Array::from_element(5, -0.20000005))]
        );
        assert_eq!(tree.offsets, [None, None, Some(1.100000262260437)]);
        assert_eq!(tree.children, [None, None, Some((0, 1))]);
        assert_eq!(tree.indices, [Some(vec![0, 3]), Some(vec![1, 2]), None]);
    }

    #[test]
    fn test_leaf_array_construction() {
        let forest = vec![
            FlatTree::default(),
            FlatTree::default(),
            FlatTree::default(),
        ];
        let arr = rptree_leaf_array(&forest);
        assert_eq!(arr, Some(SmallVec::new()), "Leaf array should be empty.");

        let forest = vec![
            FlatTree {
                hyperplanes: vec![],
                offsets: vec![],
                children: vec![Some((0, 1)), None],
                indices: vec![None, Some(vec![0, 1])],
                leaf_size: 2,
            },
            FlatTree {
                hyperplanes: vec![],
                offsets: vec![],
                children: vec![Some((0, 1)), None],
                indices: vec![None, Some(vec![0, 1])],
                leaf_size: 2,
            },
            FlatTree {
                hyperplanes: vec![],
                offsets: vec![],
                children: vec![Some((0, 1)), None],
                indices: vec![None, Some(vec![0, 1])],
                leaf_size: 2,
            },
        ];
        let arr = rptree_leaf_array(&forest);
        assert_eq!(
            arr,
            Some(smallvec![vec![0, 1], vec![0, 1], vec![0, 1],]),
            "Leaf should match.",
        );

        let forest = vec![
            FlatTree {
                hyperplanes: vec![],
                offsets: vec![],
                children: vec![Some((0, 1)), None],
                indices: vec![None, Some(vec![0, 1])],
                leaf_size: 3,
            },
            FlatTree {
                hyperplanes: vec![],
                offsets: vec![],
                children: vec![Some((0, 1)), None],
                indices: vec![None, Some(vec![0, 1])],
                leaf_size: 3,
            },
            FlatTree {
                hyperplanes: vec![],
                offsets: vec![],
                children: vec![Some((0, 1)), None],
                indices: vec![None, Some(vec![0, 1])],
                leaf_size: 3,
            },
        ];
        let arr = rptree_leaf_array(&forest);
        assert_eq!(
            arr,
            Some(smallvec![vec![0, 1], vec![0, 1], vec![0, 1],]),
            "Leaf should match.",
        );

        let forest = vec![
            FlatTree {
                hyperplanes: vec![],
                offsets: vec![],
                children: vec![Some((0, 1)), Some((2, 3)), None],
                indices: vec![None, None, Some(vec![0, 1])],
                leaf_size: 3,
            },
            FlatTree {
                hyperplanes: vec![],
                offsets: vec![],
                children: vec![Some((0, 1)), Some((4, 5)), None],
                indices: vec![None, None, Some(vec![0, 1])],
                leaf_size: 3,
            },
            FlatTree {
                hyperplanes: vec![],
                offsets: vec![],
                children: vec![Some((0, 1)), Some((8, 9)), None],
                indices: vec![None, None, Some(vec![0, 2])],
                leaf_size: 3,
            },
        ];
        let arr = rptree_leaf_array(&forest);
        assert_eq!(
            arr,
            Some(smallvec![vec![0, 1], vec![0, 1], vec![0, 2],]),
            "Leaf should match.",
        );
    }
}
