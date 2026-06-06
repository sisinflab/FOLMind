import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm


class SignificanceTester:
    def __init__(self, gt_path: str, model_a_path: str, model_b_path: str, k: int = 10):
        self.gt_path = gt_path
        self.model_a_path = model_a_path
        self.model_b_path = model_b_path
        self.k = k

        self.gt_dict = {}
        self.recs_a = {}
        self.recs_b = {}

    def _load_ground_truth(self):
        """Loads Ground Truth file (User, Item). Assumes headers exist or infers col 0,1."""
        print(f"--> Loading Ground Truth: {os.path.basename(self.gt_path)}")
        sep = '\t' if self.gt_path.endswith('.tsv') else ','

        try:
            df = pd.read_csv(self.gt_path, sep=sep)
            # Normalize column names if possible, otherwise take first two
            cols = df.columns
            if 'userId' in cols and 'itemId' in cols:
                u_col, i_col = 'userId', 'itemId'
            else:
                u_col, i_col = cols[0], cols[1]

            self.gt_dict = df.groupby(u_col)[i_col].apply(set).to_dict()
            print(f"    Loaded {len(self.gt_dict)} users from GT.")
        except Exception as e:
            print(f"Error loading GT: {e}")
            sys.exit(1)

    def _load_recs(self, path: str):
        """Loads recommendations TSV (User \t Item \t Score)."""
        recs = {}
        print(f"--> Loading Recs: {os.path.basename(path)}")
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2: continue
                    try:
                        u, i = int(parts[0]), int(parts[1])
                        if u not in recs:
                            recs[u] = []
                        recs[u].append(i)
                    except ValueError:
                        continue
        except FileNotFoundError:
            print(f"CRITICAL: File not found: {path}")
            sys.exit(1)

        return recs

    def calculate_user_metric(self, pred_items: list, ground_truth: set, metric: str) -> float:
        """Calculates metric @ K for a single user."""
        # Cutoff
        top_k = pred_items[:self.k]

        if metric == 'recall':
            hits = len(set(top_k).intersection(ground_truth))
            return hits / len(ground_truth) if len(ground_truth) > 0 else 0.0

        elif metric == 'ndcg':
            dcg = 0.0
            idcg = 0.0

            # DCG
            for i, item in enumerate(top_k):
                if item in ground_truth:
                    dcg += 1.0 / np.log2(i + 2)

            # IDCG
            num_relevant = min(len(ground_truth), self.k)
            for i in range(num_relevant):
                idcg += 1.0 / np.log2(i + 2)

            return dcg / idcg if idcg > 0 else 0.0

        return 0.0

    def run(self, metric_name='ndcg', test_type='wilcoxon'):
        self._load_ground_truth()
        self.recs_a = self._load_recs(self.model_a_path)
        self.recs_b = self._load_recs(self.model_b_path)

        # Identify Common Users
        users_gt = set(self.gt_dict.keys())
        users_a = set(self.recs_a.keys())
        users_b = set(self.recs_b.keys())

        common_users = sorted(list(users_gt & users_a & users_b))

        if not common_users:
            print("ERROR: No common users found between Ground Truth and both models.")
            print(f"GT Users: {len(users_gt)}, Model A: {len(users_a)}, Model B: {len(users_b)}")
            sys.exit(1)

        print(f"\nComputing {metric_name}@{self.k} for {len(common_users)} common users...")

        scores_a = []
        scores_b = []

        for u in tqdm(common_users):
            s_a = self.calculate_user_metric(self.recs_a[u], self.gt_dict[u], metric_name)
            s_b = self.calculate_user_metric(self.recs_b[u], self.gt_dict[u], metric_name)
            scores_a.append(s_a)
            scores_b.append(s_b)

        # Perform Statistical Test
        vec_a = np.array(scores_a)
        vec_b = np.array(scores_b)

        mean_a = np.mean(vec_a)
        mean_b = np.mean(vec_b)

        print(f"\nResults for {metric_name}@{self.k}")
        print("-" * 40)
        print(f"Model A Mean: {mean_a:.5f}")
        print(f"Model B Mean: {mean_b:.5f}")

        p_value = 1.0
        if test_type == 'wilcoxon':
            # Wilcoxon Signed-Rank Test
            stat, p_value = stats.wilcoxon(vec_a, vec_b)
        elif test_type == 'ttest':
            # Paired T-Test
            stat, p_value = stats.ttest_rel(vec_a, vec_b)

        print(f"Test Type:    {test_type.upper()}")
        print(f"p-value:      {p_value:.10f}")

        alpha = 0.05
        sig = "SIGNIFICANT" if p_value < alpha else "NOT SIGNIFICANT"
        winner = "Model B" if mean_b > mean_a else "Model A"

        print(f"Conclusion:   {sig} difference at alpha={alpha}")
        if sig == "SIGNIFICANT":
            print(f"Winner:       {winner}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Significance Testing for Recommender Systems")

    parser.add_argument('--gt', type=str, required=True, help="Path to Ground Truth CSV/TSV (Test set)")
    parser.add_argument('--base', type=str, required=True, help="Path to Baseline recommendations .tsv")
    parser.add_argument('--new', type=str, required=True, help="Path to New Model recommendations .tsv")
    parser.add_argument('--metric', type=str, default='ndcg', choices=['ndcg', 'recall'], help="Metric to calculate")
    parser.add_argument('--k', type=int, default=10, help="Cutoff K")
    parser.add_argument('--test', type=str, default='wilcoxon', choices=['wilcoxon', 'ttest'],
                        help="Statistical test type")

    args = parser.parse_args()

    tester = SignificanceTester(args.gt, args.base, args.new, args.k)
    tester.run(args.metric, args.test)


#   python run_significance.py \
#   --gt data/DATASET_DIR/splitting/test/ratings.csv \
#   --base results/neural_models_results/NeuMF/xxx/recommendations.tsv \
#   --new results/neural_models_results/NewNeuMF/xxy/recommendations.tsv \
#   --metric ndcg \
#   --k 10 \
#   --test wilcoxon