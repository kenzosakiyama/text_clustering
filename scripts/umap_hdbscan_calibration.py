import optuna
from argparse import ArgumentParser
from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index
import numpy as np
import pickle
from functools import partial
import pandas as pd
from typing import Iterable, Tuple

SEED = 1234
N_CORES = 6

# TODO: transformar isto num json?
# Range values for calibration
# (UMAP)
MIN_NEIGHBORS = 10
MAX_NEIGHBORS = 70
STEP_NEIGHB = 10
MIN_COMPONENTS = 5
MAX_COMPONENTS = 200
STEP_COMPONENTS = 5
# (HDBSCAN)
MIN_CLUSTER_SIZE = 10
MAX_CLUSTER_SIZE = 100
STEP_CLUSTER = 5
MIN_SAMPLES = 5
MAX_SAMPLES = 200
STEP_SAMPLES = 5

np.random.seed(SEED)

def run_clustering(text_vectors: np.array,
                   n_neighbors: int, 
                   n_components: int, 
                   min_cluster_size: int, 
                   min_samples: int) -> Tuple[Iterable[int], np.array]:

    umap_reducer = UMAP(
        n_neighbors=n_neighbors, 
        n_components=n_components, 
        min_dist=0.0,
        metric='cosine',
        low_memory=True,
        random_state=SEED,
        transform_seed=SEED,
        n_jobs=N_CORES
    )

    hdbscan_clustering = HDBSCAN(
        min_cluster_size=min_cluster_size if type(min_cluster_size) == int else min_cluster_size.item(),
        min_samples=min_samples if type(min_samples) == int else min_samples.item(),
        metric='euclidean',
        cluster_selection_method='eom',
        core_dist_n_jobs=N_CORES,
    )

    # Aplicando o UMAP
    print(f"- Running UMAP: {umap_reducer}.")
    reduced_embeddings = umap_reducer.fit_transform(text_vectors)

    # Aplicando o HDBSCAN
    print(f"- Running HDBSCAN: {hdbscan_clustering}.")
    cluster_labels = hdbscan_clustering.fit_predict(reduced_embeddings)

    # Retorna os rótulos dos clusters e as embeddings UMAP geradas
    return cluster_labels, reduced_embeddings

def objective(trial: optuna.Trial, 
              text_vectors: np.array) -> float:

    # Preparando UMAP
    n_neighbors = trial.suggest_int("umap_n_neighbors", MIN_NEIGHBORS, MAX_NEIGHBORS, STEP_NEIGHB)
    n_components = trial.suggest_int("umap_n_components", MIN_COMPONENTS, MAX_COMPONENTS, STEP_COMPONENTS)

    # Preparando HDBSCAN
    min_cluster_size = trial.suggest_int("hdbscan_min_cluster_size", MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE, STEP_CLUSTER)
    min_samples = trial.suggest_int("hdbscan_min_samples", MIN_SAMPLES, MAX_SAMPLES, STEP_SAMPLES)

    cluster_labels, reduced_embeddings = run_clustering(
        text_vectors, 
        n_neighbors, 
        n_components, 
        min_cluster_size, 
        min_samples
    )

    # Calculando DBCV
    score = validity_index(reduced_embeddings.astype(np.float64), cluster_labels)

    return score

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--output_file", type=str, required=True, help="Arquivo .csv de saida com os resultados da validação cruzada.")
    parser.add_argument("--embeddings_file", type=str, required=True, help="Arquivo .pkl com os embeddings a serem utilizados.")

    parser.add_argument("--n_jobs", type=int, default=1, help="Número cores a utilizar.")
    parser.add_argument("--iterations", type=int, default=100, help="Número de iterações para o Random Search.")
    parser.add_argument("--cluster_file", type=str, default="cluster.pkl", help="Arquivo .pkl contendo os cluster para os melhores parâmetros.")

    args = parser.parse_args()

    with open(args.embeddings_file, "rb") as f:
        text_vectors = pickle.load(f)

    print(f"- Loaded vectors of shape: {text_vectors.shape}, from {args.embeddings_file}.")
    print(f"- Saving parameters to: {args.output_file}.")

    # Mostrando ranges
    print(f"- UMAP ranges:")
    print(f"\t- n_neighbors: [{MIN_NEIGHBORS} ... {MAX_NEIGHBORS}] -> {STEP_NEIGHB} step.")
    print(f"\t- n_components: [{MIN_COMPONENTS} ... {MAX_COMPONENTS}] -> {STEP_COMPONENTS} step.")
    print(f"- HDBSCAN ranges:")
    print(f"\t- min_cluster_size: [{MIN_CLUSTER_SIZE} ... {MAX_CLUSTER_SIZE}] -> {STEP_CLUSTER} step.")
    print(f"\t- min_samples: [{MIN_SAMPLES} ... {MAX_SAMPLES}] -> {STEP_SAMPLES} steps.")

    # Definindo função objetivo
    objective_func = partial(objective, text_vectors=text_vectors)

    # Criando Study com o Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_func, n_trials=args.iterations, n_jobs=args.n_jobs)

    print(f"\nBest DBCV: {study.best_value}")
    print(f"Best params: {study.best_params}")

    # Serializar trials para um csv
    df = study.trials_dataframe()
    df.sort_values(by='value', ascending=False, inplace=True)
    print(df.head(5))

    df.to_csv(args.output_file)

    print("- Clustering with the best parameters.")
    cluster_labels, reduced_embeddings = run_clustering(
        text_vectors,
        study.best_params["umap_n_neighbors"],
        study.best_params["umap_n_components"],
        study.best_params["hdbscan_min_cluster_size"],
        study.best_params["hdbscan_min_samples"]
    )

    final_dbcv = validity_index(reduced_embeddings.astype(np.float64), cluster_labels)

    print(f"Final DBCV: {final_dbcv}")

    print(f"- Saving clusters to: {args.cluster_file}.")
    with open(args.cluster_file, "wb") as f:
        pickle.dump(cluster_labels, f)