from typing import Tuple, Dict
from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index
from argparse import ArgumentParser
import numpy as np
import pickle
import json

np.random.seed(1234)
umap_params = {"n_components": 110, "n_neighbors": 50, 'min_dist': 0.0, 'metric': 'cosine', "transform_seed":1234, 'random_state': 1234, "low_memory":True}
hdbscan_params = {"min_cluster_size": 65, "min_samples": 55, 'metric': 'euclidean', 'cluster_selection_method': 'eom'}

def load_parameters_from_json(json_file_path: str) -> Tuple[Dict, Dict]:
    
    with open(json_file_path, 'r') as f:
        params = json.load(f)

    umap_params = params["umap"]
    hdbscan_params = params["hdbscan"]

    return umap_params, hdbscan_params

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--params", type=str, required=False, help="Arquivo .json com os parametros de configuração do UMAP e HDBSCAN.")
    parser.add_argument("--embeddings", type=str, required=True, help="Arquivo .pkl com os embeddings a serem utilizados.")
    parser.add_argument("--output_file", type=str, default="clusters.pkl", help="Arquivo .pkl para armazenar os clusters. Associa-se um cluster para cada exemplo do arquivo de embeddings.")

    args = parser.parse_args()

    with open(args.embeddings, "rb") as f:
        text_vectors = pickle.load(f)

    print(f"- Loaded vectors of shape: {text_vectors.shape}, from {args.embeddings}.")
    print(f"- LOading parameters from {args.params}.")
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

    unique, counts = np.unique(cluster_labels, return_counts=True)
    sorted_idxs = np.argsort(counts)[::-1]
    print(f"- {len(unique) - 1} identified clusters.")
    for i in sorted_idxs:
        print(f"\t- {unique[i]:2d}: {counts[i]}")

    print(f"- Saving clusters to: {args.output_file}.")
    with open(args.output_file, "wb") as f:
        pickle.dump(cluster_labels, f)

    # df.to_csv(args.output_file)

