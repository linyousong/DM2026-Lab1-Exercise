import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE


def compare_tsne():
    # 1. Load dataset fixture directly
    df = pd.read_csv("agent_dev/sample_fixture.csv")
    texts = df["text"].fillna("").astype(str).tolist()

    # 2. Build DTM using CountVectorizer defaults
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(texts).toarray()

    # 3. Compute actual t-SNE coordinates dynamically via scikit-learn
    tsne_model = TSNE(n_components=2, perplexity=2, random_state=42)
    actual = tsne_model.fit_transform(dtm)

    # 4. Known Answer #8 (Desired)
    desired = np.array([
        [1330.9991, 32.1039],     # Doc 0 (catA)
        [951.2459, 393.5588],     # Doc 1 (catA)
        [530.8209, 682.9791],     # Doc 2 (catA)
        [-1558.6448, -314.1625],  # Doc 3 (catB)
        [-915.5153, -346.3907],   # Doc 4 (catB)
        [-280.0313, -1261.2115],  # Doc 5 (catB)
        [-280.0313, -1261.2115],  # Doc 6 (catB, duplicate of doc 5)
        [-701.3674, 1465.3973],   # Doc 7 (catA, empty text)
    ])

    datasets = {"Known Answer #8 (Desired)": desired, "Current Backend (Actual)": actual}

    for name, data in datasets.items():
        print(f"=== {name} ===")
        # Rows 0-6 non-empty documents: catA = [0, 1, 2], catB = [3, 4, 5, 6]
        catA = data[[0, 1, 2]]
        catB = data[[3, 4, 5, 6]]

        centroid_A = catA.mean(axis=0)
        centroid_B = catB.mean(axis=0)
        centroid_dist = np.linalg.norm(centroid_A - centroid_B)

        # Mean within-class distance
        dist_matrix_A = squareform(pdist(catA))
        dist_matrix_B = squareform(pdist(catB))

        # Upper triangle off-diagonals for average pairwise distance
        mean_within_A = dist_matrix_A[np.triu_indices(len(catA), k=1)].mean()
        mean_within_B = dist_matrix_B[np.triu_indices(len(catB), k=1)].mean()
        mean_within_all = (mean_within_A + mean_within_B) / 2.0

        # Pairwise distance between rows 0, 1, 2
        d01 = dist_matrix_A[0, 1]
        d02 = dist_matrix_A[0, 2]
        d12 = dist_matrix_A[1, 2]

        print(f"catA Centroid (Rows 0-2): [{centroid_A[0]:.4f}, {centroid_A[1]:.4f}]")
        print(f"catB Centroid (Rows 3-6): [{centroid_B[0]:.4f}, {centroid_B[1]:.4f}]")
        print(f"Centroid Distance (catA vs catB): {centroid_dist:.4f}")
        print(f"Mean Within-Class Distance: catA = {mean_within_A:.4f}, catB = {mean_within_B:.4f}, overall = {mean_within_all:.4f}")
        print(f"catA Pairwise Distances: d(0,1) = {d01:.4f}, d(0,2) = {d02:.4f}, d(1,2) = {d12:.4f}")
        print(f"Ordering 0-1 < 0-2 preserved? {d01 < d02} ({d01:.4f} < {d02:.4f})")

        # Row 7 position relative to centroids
        row7 = data[7]
        dist_7_to_A = np.linalg.norm(row7 - centroid_A)
        dist_7_to_B = np.linalg.norm(row7 - centroid_B)
        print(f"Row 7 Coordinates (Empty text): [{row7[0]:.4f}, {row7[1]:.4f}]")
        print(f"Row 7 Distance to catA Centroid: {dist_7_to_A:.4f}")
        print(f"Row 7 Distance to catB Centroid: {dist_7_to_B:.4f}\n")


if __name__ == "__main__":
    compare_tsne()
