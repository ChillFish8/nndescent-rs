import numba
import pynndescent
import numpy as np
import h5py
from urllib.request import urlretrieve
import os


def get_ann_benchmark_data(dataset_name):
    if not os.path.exists(f"{dataset_name}.hdf5"):
        print(f"Dataset {dataset_name} is not cached; downloading now ...")
        urlretrieve(f"http://ann-benchmarks.com/{dataset_name}.hdf5", f"{dataset_name}.hdf5")
    hdf5_file = h5py.File(f"{dataset_name}.hdf5", "r")
    return np.array(hdf5_file['train']), np.array(hdf5_file['test']), hdf5_file.attrs['distance']


fmnist_train, fmnist_test, _ = get_ann_benchmark_data('fashion-mnist-784-euclidean')
