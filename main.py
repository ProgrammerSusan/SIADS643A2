import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_data(path: str, columns: List[str]) -> pd.DataFrame:
    """
    Load data from passed in CSV path and select columns for k-means clustering

    input: str, List[str]

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
    k = KMeans(n_clusters=n_clusters, random_state=42)
    classification = k.fit_predict(data)
    return classification


def parse_args():

if __name__ == "__main__":







