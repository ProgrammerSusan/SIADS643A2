import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import argparse


def load_data(path: str, columns: list[str]) -> pd.DataFrame:
    """
    Load data from passed in CSV path and select columns for k-means clustering

    input: str, list[str]

    output: pd.DataFrame
    """
    df = pd.read_csv(path)
    return df[columns]


def scale_data(df: pd.DataFrame):
    """
    Scale data for k-means clustering

    input: pd.DataFrame

    output: np.ndarray
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df)


def create_kmeans_clusters(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Apply k-means clustering to data

    input: pd.DataFrame, int

    output: np.ndarray
    """
    k = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = k.fit_predict(data)
    return labels


def parse_args():
    parser = argparse.ArgumentParser(description="K-means clustering script")
    parser.add_argument('path', help="CSV file path")
    parser.add_argument('columns', help="Columns to use for clustering", nargs='+')
    parser.add_argument('clusters', help="Number of clusters", type=int)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    df = load_data(args.path, args.columns)
    scaled = scale_data(df)
    labels = create_kmeans_clusters(scaled, args.clusters)

    print("Labels: ")
    print(labels)







