from typing import Tuple, Dict
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from yellowbrick.cluster.elbow import KElbowVisualizer
import numpy as np
import pickle
import json

np.random.seed(1234)

K_RANGES = (3, 12)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--embeddings", type=str, required=True, help="Arquivo .pkl com os embeddings a serem utilizados.")
    parser.add_argument("--metric", type=str, default="distortion", help="Métrica utilizada para escolha do K.")
    parser.add_argument("--plot_file", type=str, default="elbow.png", help="Caminho para salvar uma imagem contendo a visualização da regra do joelho.")
    parser.add_argument("--output_file", type=str, default="clusters.pkl", help="Arquivo .pkl para armazenar os clusters. Associa-se um cluster para cada exemplo do arquivo de embeddings.")

    args = parser.parse_args()

    with open(args.embeddings, "rb") as f:
        text_vectors = pickle.load(f)

    print(f"- Loaded vectors of shape: {text_vectors.shape}, from {args.embeddings}.")
    print(f"- Saving clusters to: {args.output_file}.")

    clustering_model = KMeans(random_state=1234, n_init=200, max_iter=500)
    print(f"- Running elbow rule, using {args.metric} to estimate K.")
    model = KElbowVisualizer(clustering_model, k=K_RANGES, metric="silhouette")
    model.fit(text_vectors)

    model.show(args.plot_file)

    best_k = model.elbow_value_

    final_model = KMeans(n_clusters=best_k, random_state=1234)
    cluster_labels = final_model.fit_predict(text_vectors)

    print(f"- Saving clusters to: {args.output_file}.")
    with open(args.output_file, "wb") as f:
        pickle.dump(cluster_labels, f)

    

