import logging
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from safetensors.torch import load_file
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProfileLoader:
    """Loads the 'profile' part of user or item embeddings."""

    def __init__(self, datadir: str):
        self.base_path = os.path.join(os.getcwd(), 'data', datadir, 'LLMEmb_output')

    def load_profiles(self, filename: str, normalize: bool = True):
        path = os.path.join(self.base_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        raw_data = load_file(path)
        profiles = {}

        for key, tensor in raw_data.items():
            if key.endswith('_profile'):
                arr = tensor.float().numpy().astype(np.float32)

                if np.isnan(arr).any(): continue

                # L2 Normalization
                if normalize:
                    norm = np.linalg.norm(arr)
                    if norm > 0: arr /= norm

                profiles[key.replace('_profile', '')] = arr

        if not profiles:
            raise ValueError(f"No profile embeddings found in {path}. Check your keys.")

        ids = list(profiles.keys())
        matrix = np.vstack([profiles[i] for i in ids])
        logger.info(f"Loaded {len(ids)} profiles from {filename} (ignored reasoning).")
        return matrix, ids


class Visualizer:
    def __init__(self, loader: ProfileLoader):
        self.loader = loader

    def plot_tsne(self, matrix, title):
        """Standard 2D Projection."""
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='random')
        proj = tsne.fit_transform(matrix)

        plt.figure(figsize=(10, 8))
        plt.scatter(proj[:, 0], proj[:, 1], alpha=0.6, s=10, c='tab:blue')
        plt.title(f"t-SNE Projection: {title}")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_xsimgcl_density(self, matrix, title):
        """Density of angles on the 2D plane."""
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='random')
        proj = tsne.fit_transform(matrix)

        # Center data to focus on angular distribution rather than absolute position
        proj -= proj.mean(axis=0)
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10), gridspec_kw={'height_ratios': [2, 1]})

        ax1.scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.5, c='tab:blue')
        ax1.set_title(f"2D Space: {title}")
        ax1.axis('equal')

        # KDE Density of Angles
        sns.kdeplot(angles, ax=ax2, fill=True, color='orange', bw_adjust=0.5)
        ax2.set_xlim(-np.pi, np.pi)
        ax2.set_title("XSimGCL Metric: Angular Density (Hypersphere Uniformity)")
        ax2.set_xlabel("Angle (radians)")

        plt.tight_layout()
        plt.show()


class MetricsCalculator:
    @staticmethod
    def calculate(matrix):
        """Calculates MAD and Uniformity."""
        # MAD (Higher implies profiles are unique and distinct)
        dists = pdist(matrix, metric='cosine')
        mad = np.mean(dists)

        # Uniformity (Negative implies even distribution on sphere)
        sq_euclidean = pdist(matrix, metric='sqeuclidean')
        uniformity = np.log(np.mean(np.exp(-2 * sq_euclidean)))

        print(f"\n" + "=" * 50)
        print(f"EMBEDDING EVALUATION METRICS")
        print(f"=" * 50)
        print(f"MAD:         {mad:.4f}")
        print(f"Uniformity:      {uniformity:.4f}")
        print(f"Avg Pairwise Similarity:    {1 - mad:.4f}")
        print(f"=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, choices=['tsne', 'density', 'metrics'])
    parser.add_argument('--filename', type=str, required=True, help="e.g. user_last_embeddings.safetensors")
    args = parser.parse_args()

    loader = ProfileLoader(args.dataset)
    viz = Visualizer(loader)

    matrix, _ = loader.load_profiles(args.filename)

    if args.task == 'tsne':
        viz.plot_tsne(matrix, args.filename)
    elif args.task == 'density':
        viz.plot_xsimgcl_density(matrix, args.filename)
    elif args.task == 'metrics':
        MetricsCalculator.calculate(matrix)


if __name__ == "__main__":
    main()

# python run_visualize.py --dataset DATASET_NAME --task metrics/density/tsne --filename user_last_embeddings.safetensors/item_last_embeddings.safetensors