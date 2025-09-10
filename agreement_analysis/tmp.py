import sys

import pandas as pd


def sort_csv_by_index(file_path):
    # Read CSV with index
    df = pd.read_csv(file_path, index_col=0)

    # Sort by index
    df_sorted = df.sort_index()

    # Overwrite original file
    df_sorted.to_csv(file_path)


if __name__ == "__main__":
    sort_csv_by_index("dataset_descriptions.csv")
