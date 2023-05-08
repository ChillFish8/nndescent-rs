import numpy as np
import h5py
import pynndescent
import time
from urllib.request import urlretrieve
import os

def get_ann_benchmark_data(dataset_name):
    if not os.path.exists(f"../datasets/{dataset_name}.hdf5"):
        print(f"Dataset {dataset_name} is not cached; downloading now ...")
        urlretrieve(f"http://ann-benchmarks.com/{dataset_name}.hdf5", f"../datasets/{dataset_name}.hdf5")
    hdf5_file = h5py.File(f"../datasets/{dataset_name}.hdf5", "r")
    return np.array(hdf5_file['train']), np.array(hdf5_file['test']), hdf5_file.attrs['distance']


fmnist_train, fmnist_test, _ = get_ann_benchmark_data('fashion-mnist-784-euclidean')


print(len(fmnist_train))

start = time.perf_counter()
index = pynndescent.NNDescent(
    fmnist_train,
    verbose=True,
    low_memory=False,
    metric="dot",
    n_jobs=1
)

end = time.perf_counter() - start
print(f"Took: {end}s {len(fmnist_train) / end} v/s")
