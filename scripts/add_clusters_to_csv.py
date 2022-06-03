import pandas as pd
import numpy as np
import pickle
from argparse import ArgumentParser

# Script auxiliar para adicionar clusters obtidos em outro arquivo.

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--csv", type=str, required=True, help="Arquivo .csv, no qual será adicionada uma coluna com os clusters.")
    parser.add_argument("--clusters", type=str, required=True, help="Arquivo .pkl contendo os clusters.")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    with open(args.clusters, "rb") as f:
        clusters = pickle.load(f)

    print(f"- Adding clusters to: {args.csv}.")
    print(f"- Loaded clusters of shape: {clusters.shape}, from {args.clusters}.")
    print(f"- Number of examples in the csv file: {len(df)}.")

    assert len(df) == len(clusters), "The number of examples in the csv file must be equal to the number of clusters."

    df["cluster"] = clusters

    print(df["cluster"].value_counts())

    df.to_csv(args.csv, index=False)