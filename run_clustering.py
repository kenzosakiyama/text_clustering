from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index
from argparse import ArgumentParser
import numpy as np
import pickle
import pandas as pd

np.random.seed(1234)
umap_params = {"n_components": 185, "n_neighbors": 70, 'min_dist': 0.0, 'metric': 'cosine', "transform_seed":1234, 'random_state': 1234}
hdbscan_params = {"min_cluster_size": 65, "min_samples": 100, 'metric': 'euclidean', 'cluster_selection_method': 'eom'}

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--embeddings_file", type=str, required=True, help="Arquivo .pkl com os embeddings a serem utilizados.")
    parser.add_argument("--output_file", type=str, required=True, help="Arquivo .csv para adicionar coluna com os identificadores dos clusters.")
    parser.add_argument("--params_file", type=str, required=False, help="Arquivo .json com os parametros de configuração do UMAP e HDBSCAN.")

    args = parser.parse_args()

    df = pd.read_csv(args.output_file, index_col=0)

    with open(args.embeddings_file, "rb") as f:
        text_vectors = pickle.load(f)

    assert len(df) == text_vectors.shape[0], "O número de embeddings não corresponde ao número de linhas do arquivo de saída."

    print(f"- Loaded texts: {len(df)}.")
    print(f"- Loaded vectors of shape: {text_vectors.shape}, from {args.embeddings_file}.")
    print(f"- Saving clusters to: {args.output_file}.")

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

    df["cluster"] = cluster_labels
    print(df.head())

    # df.to_csv(args.output_file)

