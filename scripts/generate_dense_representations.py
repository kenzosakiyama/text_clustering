from argparse import ArgumentParser
import pandas as pd
import os
import pickle
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from transformers import data

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--source_file", type=str, required=True, help="Arquivo .csv de entrada. As representações serão armazenadas na mesma pasta do arquivo de origem.")

    parser.add_argument("--text_column", type=str, required=True, help="Coluna de texto a ser processada do arquivo de entrada.")
    parser.add_argument("--encoder", type=str, required=True, help="Enconder (SentenceTransformer) a ser utilizado para geração das representações. Especificar um checkpoint valido.")
    parser.add_argument("--bs", type=int, default=512, help="Batch size para geração das representações.")


    args = parser.parse_args()
    print(f"Loading data from {args.source_file}")
    data_df = pd.read_csv(args.source_file, index_col=0)
    print(data_df.tail())

    print(f"- Number of texts to encode: {len(data_df)}")
    print(f"- Using {args.encoder} representation to encode '{args.text_column}'.")

    model = SentenceTransformer(args.encoder, device="cuda")
    text_vectors = model.encode(data_df[args.text_column].values, show_progress_bar=True, batch_size=args.bs)

    # Formatando nome do arquivo de saida
    output_file_path = f"{args.source_file.replace('.csv', '_' + args.encoder.replace('/', '-'))}.pkl"

    print(f"{text_vectors.shape} vectors for {len(data_df)} documents.")

    print(data_df.tail(3))

    print(f"- Saving representations to {output_file_path}")
    with open(output_file_path, "wb") as f:
        pickle.dump(text_vectors, f)


