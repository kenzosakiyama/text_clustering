from typing import Tuple, Dict
from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index
from argparse import ArgumentParser
import numpy as np
import pickle
import json

np.random.seed(1234)

def load_parameters_from_json(json_file_path: str) -> Tuple[Dict, Dict]:
    
    with open(json_file_path, 'r') as f:
        params = json.load(f)

    umap_params = params["umap"]
    hdbscan_params = params["hdbscan"]

    return umap_params, hdbscan_params

def show_clusters(clusters: np.array) -> None:

    unique, counts = np.unique(clusters, return_counts=True)
    sorted_idxs = np.argsort(counts)[::-1]
    print(f"- {len(unique) - 1} identified clusters.")
    for i in sorted_idxs:
        print(f"\t- {unique[i]:2d}: {counts[i]}")

def split_largest_cluster(clusters: np.array,
                          umap_embeddings: np.array,
                          clustering_model: HDBSCAN,
                          n_times: int) -> np.array:
    
    clusters = clusters.copy()

    for i in range(n_times):

        unique, counts = np.unique(clusters, return_counts=True)
        n_clusters = len(unique) - 1
        print(f"- Staring with {n_clusters} (excluding outliers).")

        # Find the two largest clusters
        sorted_cluster_ids = np.argsort(counts)[::-1]
        largest_cluster = unique[sorted_cluster_ids[0]]
        second_largest_cluster = unique[sorted_cluster_ids[1]]

        # If the largest cluster is the outlier cluster, use the second largest.
        if largest_cluster == -1:
            print(f" - Warning: the current largest cluster is the outlier cluster (-1). Using the second largest cluster ({second_largest_cluster}) instead.")
            largest_cluster = second_largest_cluster

        # Filtering examples from the largest cluster
        cluster_mask = clusters == largest_cluster
        cluster_embeddings = umap_embeddings[cluster_mask]
        new_clusters = clustering_model.fit_predict(cluster_embeddings)

        # Fixing cluster ids. Keeping outlier id (-1).
        new_clusters[new_clusters >= 0] += n_clusters + i # adding the current number divided clusters
        # Updating clusters with the new ids.
        clusters[cluster_mask] = new_clusters

        # Showing split result.
        print(f"- After splitting the largest cluster ({largest_cluster}), {len(np.unique(new_clusters))} clusters were created.")
        show_clusters(new_clusters)

        split_dbcv = validity_index(cluster_embeddings.astype(np.float64), new_clusters)
        print(f"- DBCV for the splitted clusters: {split_dbcv}")

    return clusters

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--params", type=str, required=False, help="Arquivo .json com os parametros de configuração do UMAP e HDBSCAN.")
    parser.add_argument("--embeddings", type=str, required=True, help="Arquivo .pkl com os embeddings a serem utilizados.")
    parser.add_argument("--output_file", type=str, default="clusters.pkl", help="Arquivo .pkl para armazenar os clusters. Associa-se um cluster para cada exemplo do arquivo de embeddings.")
    parser.add_argument("--divide_largest_cluster", type=int, default=0, help="Isola-se o maior cluster e repete-se a clusterização divide_largest_cluster vezes.")

    args = parser.parse_args()

    with open(args.embeddings, "rb") as f:
        text_vectors = pickle.load(f)

    print(f"- Loaded vectors of shape: {text_vectors.shape}, from {args.embeddings}.")
    print(f"- Using parameters from {args.params}.")
    print(f"- Saving clusters to: {args.output_file}.")

    umap_params, hdbscan_params = load_parameters_from_json(args.params)

    umap_reducer = UMAP(**umap_params)
    print(f"- Running UMAP: {umap_reducer}.")
    reduced_embeddings = umap_reducer.fit_transform(text_vectors)

    hdbscan_clustering = HDBSCAN(**hdbscan_params)
    print(f"- Running HDBSCAN: {hdbscan_clustering}.")
    cluster_labels = hdbscan_clustering.fit_predict(reduced_embeddings)

    DBCV = validity_index(reduced_embeddings.astype(np.float64), cluster_labels)
    print(f"- DBCV: {DBCV}.")

    show_clusters(cluster_labels)

    if args.divide_largest_cluster > 0:
        print(f"- Splitting the largest cluster. Repeating process {args.divide_largest_cluster} times.")
        cluster_labels = split_largest_cluster(
            cluster_labels,
            reduced_embeddings,
            hdbscan_clustering,
            args.divide_largest_cluster
        )
        print(f"Final clusters:")
        show_clusters(cluster_labels)
        new_DBCV = validity_index(reduced_embeddings.astype(np.float64), cluster_labels)
        print(f"- DBCV, considering the new clusters: {new_DBCV}.")

    print(f"- Saving clusters to: {args.output_file}.")
    with open(args.output_file, "wb") as f:
        pickle.dump(cluster_labels, f)

    

