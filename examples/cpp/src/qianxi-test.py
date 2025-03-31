import numpy as np
import cupy as cp
from cuvs.neighbors import ivf_flat
from cuvs.neighbors import cagra
import time
import os
import glob
import faiss
 
def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size
 
keys = np.load("/data/ruler/key_states_layer_10.npy")
print(keys[0][0].shape)
key = keys[0][0].astype(np.float16)
column_means = np.mean(key, axis = 0)
key = key - column_means
dataset = cp.asarray(key)

k = 100
prefix = '/data/ruler/query_states_layer_10_gen_token'
file_paths = glob.glob(f"{prefix}*.npy")
data_list = []
for file_path in file_paths:
    one_query = np.load(file_path)
    data_list.append(one_query[0][0])
query = np.concatenate(data_list, axis=0).astype(np.float16)
print(query.shape)
queries = cp.asarray(query)

#faiss.omp_set_num_threads(1)
#flat_index = faiss.index_factory(key.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
#flat_index.add(key)
#gt_D, gt_I = flat_index.search(query, k)
 
#start = time.time()
#ivf_index = faiss.index_factory(key.shape[1], "IVF2048,Flat", faiss.METRIC_INNER_PRODUCT)
#ivf_index.train(key)
#ivf_index.add(key)
#end = time.time()
#print("ivf build time:", end - start)
#ivf_index.nprobe = 100
#ivf_D, ivf_I = ivf_index.search(query, k)
#print("ivf recall: ", compute_recall(ivf_I, gt_I))
 
cagra_build_params = cagra.IndexParams(metric="inner_product")
cagra_index = cagra.build(cagra_build_params, dataset)
distances, neighbors = cagra.search(cagra.SearchParams(), cagra_index, queries, 10)
 
index_params = ivf_flat.IndexParams(metric="inner_product", n_lists=2048, kmeans_n_iters=10)
start = time.time()
index = ivf_flat.build(index_params, dataset)
end = time.time()
print("cuvs build time", end - start)
 
search_params = ivf_flat.SearchParams(n_probes=100)
distances, neighbors = ivf_flat.search(search_params, index, queries, k)
neighbors = cp.asarray(neighbors)
neighbors = cp.asnumpy(neighbors)
#print("cuvs recall: ", compute_recall(neighbors, gt_I))