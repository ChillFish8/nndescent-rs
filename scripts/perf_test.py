import time

import h5py
import numpy as np
from urllib.request import urlretrieve
import os
import numba

FLOAT32_MAX = np.finfo(np.float32).max

def get_ann_benchmark_data(dataset_name):
    if not os.path.exists(f"../datasets/{dataset_name}.hdf5"):
        print(f"Dataset {dataset_name} is not cached; downloading now ...")
        urlretrieve(f"http://ann-benchmarks.com/{dataset_name}.hdf5", f"../datasets/{dataset_name}.hdf5")
    hdf5_file = h5py.File(f"../datasets/{dataset_name}.hdf5", "r")
    return np.array(hdf5_file['train']), np.array(hdf5_file['test']), hdf5_file.attrs['distance']


fmnist_train, fmnist_test, _ = get_ann_benchmark_data('gist-960-euclidean')


@numba.njit(fastmath=True)
def euclidean(x, y):
    r"""Standard euclidean distance.

    .. math::
        D(x, y) = \\sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def squared_euclidean(x, y):
    r"""Squared euclidean distance.

    .. math::
        D(x, y) = \sum_i (x_i - y_i)^2
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


@numba.njit(fastmath=True)
def cosine(x, y):
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(x.shape[0]):
        result += x[i] * y[i]
        norm_x += x[i] ** 2
        norm_y += y[i] ** 2

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return 1.0
    else:
        return 1.0 - (result / np.sqrt(norm_x * norm_y))

@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "norm_x": numba.types.float32,
        "norm_y": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def alternative_cosine(x, y):
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dim = x.shape[0]
    for i in range(dim):
        result += x[i] * y[i]
        norm_x += x[i] * x[i]
        norm_y += y[i] * y[i]

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return FLOAT32_MAX
    elif result <= 0.0:
        return FLOAT32_MAX
    else:
        result = np.sqrt(norm_x * norm_y) / result
        return np.log2(result)

@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def dot(x, y):
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        result += x[i] * y[i]

    if result <= 0.0:
        return 1.0
    else:
        return 1.0 - result


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def alternative_dot(x, y):
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        result += x[i] * y[i]

    if result <= 0.0:
        return FLOAT32_MAX
    else:
        return -np.log2(result)


def run_benchmark(dist):
    window_size = 2

    for i in range(len(fmnist_train) - window_size + 1):
        left, right = fmnist_train[i: i + window_size]
        dist(left, right)


def benchmark(method, runs=5, warmup=False):
    start = time.perf_counter()

    for _ in range(runs):
        run_benchmark(method)

    elapsed = ((time.perf_counter() - start) / runs) * 1000

    if not warmup:
        print(f"{str(method).replace('CPUDispatcher', ''):<55} Took: {elapsed.__trunc__()}ms")


benchmark(dot, warmup=True)
benchmark(alternative_dot, warmup=True)
benchmark(cosine, warmup=True)
benchmark(alternative_cosine, warmup=True)
benchmark(euclidean, warmup=True)
benchmark(squared_euclidean, warmup=True)

benchmark(dot)
benchmark(alternative_dot)
benchmark(cosine)
benchmark(alternative_cosine)
benchmark(euclidean)
benchmark(squared_euclidean)
