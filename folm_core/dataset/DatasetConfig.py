import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Dict, Any, Set
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    strategy: str
    train_path: Optional[str] = None
    valid_path: Optional[str] = None
    test_path: Optional[str] = None
    split_params: Dict = Field(default_factory=dict)  # false positive error PyCharm
    subset_interactions: Optional[Dict[str, Any]] = Field(default=None)
    stratification_params: Optional[Dict[str, Any]] = Field(default=None)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        try:
            return cls(**config_dict)
        except ValidationError as e:
            logger.error(f"Config validation error: {e}")
            raise

    def split_data(self, data_df: pd.DataFrame, data_dir: str, filename: str) -> Set[Any]:
        """
        Splits the data into train, validation, and test subdirectories under data_dir.
        Returns a set of the item IDs that remain after filtering.
        """

        df = data_df.copy()

        #apply stratification
        if self.stratification_params:
            logger.info("Stratification is configured. Subsampling the dataset first.")
            df = self._perform_stratification(df)

        # apply minimum interactions filter (optional)
        df = self._apply_min_interactions(df)

        # create subdirectories
        split_dir = os.path.join(data_dir, 'splitting')
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'train'), exist_ok=True)

        # conditionally create validation and test folders if required in config
        if self.split_params.get('validation_size'):
            os.makedirs(os.path.join(split_dir, 'validation'), exist_ok=True)
        if self.split_params.get('test_size'):
            os.makedirs(os.path.join(split_dir, 'test'), exist_ok=True)

        if self.strategy == "random_holdout":
            self._split_random_holdout(df, split_dir, filename, **self.split_params)
        elif self.strategy == "temporal_hold_out":
            self._split_temporal_hold_out(df, split_dir, filename, **self.split_params)
        elif self.strategy == "fixed":
            self._load_fixed(split_dir)
        else:
            raise ValueError(f"Invalid split strategy: {self.strategy}")

        return set(df['itemId'].unique())

    def _perform_stratification(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs stratification based on stratification_params.
        """
        params = self.stratification_params
        target_users = params.get('target_users')
        if not target_users:
            raise ValueError("'target_users' must be specified in stratification_params.")

        random_state = params.get('random_state', 42)
        k_min_safeguard = params.get('k_min_safeguard', 0)

        logger.info(f"Stratifying to {target_users} target users.")
        RNG = np.random.default_rng(random_state)
        inter = data_df.copy()

        # Compute global item popularity
        item_pop = inter.groupby("itemId").size().rename("item_cnt")
        q1, q2 = np.quantile(item_pop, [1 / 3, 2 / 3])

        def pop_bucket(c):
            if c <= q1: return "tail"
            if c <= q2: return "mid"
            return "head"

        item_pop_bucket = item_pop.map(pop_bucket).rename("item_bucket")
        inter = inter.merge(item_pop_bucket.reset_index(), on="itemId", how="left")

        # User features
        u_grp = inter.groupby("userId")
        user_df = pd.DataFrame({"userId": u_grp.size().index, "n_inter": u_grp.size().values})
        expo = (inter.groupby(["userId", "item_bucket"]).size().unstack(fill_value=0))
        for col in ["head", "mid", "tail"]:
            if col not in expo.columns: expo[col] = 0
        expo = expo.div(expo.sum(axis=1).replace(0, 1), axis=0)
        expo.columns = [f"exp_{c}" for c in expo.columns]
        user_df = user_df.merge(expo.reset_index(), on="userId", how="left")

        # Build strata
        user_df["act_decile"] = pd.qcut(user_df["n_inter"], 10, labels=False, duplicates="drop")
        user_df["head_share"] = user_df["exp_head"].fillna(0.0)
        user_df["head_decile"] = pd.qcut(user_df["head_share"], 10, labels=False, duplicates="drop")
        user_df["stratum"] = user_df["act_decile"].astype(str) + "_" + user_df["head_decile"].astype(str)

        # Allocation
        strata_counts = user_df["stratum"].value_counts()
        total_users = len(user_df)
        if target_users >= total_users:
            logger.warning(f"Target users ({target_users}) >= total users ({total_users}). Using all users.")
            alloc = strata_counts
        else:
            alloc = (strata_counts / strata_counts.sum() * target_users).round().astype(int)

        # Sample users
        samples = []
        for s, k in alloc.items():
            if k == 0: continue
            cand = user_df.loc[user_df["stratum"] == s, "userId"].values
            if k >= len(cand):
                samples.append(cand)
            else:
                samples.append(cand[RNG.choice(len(cand), size=int(k), replace=False)])

        if not samples: return pd.DataFrame()

        sampled_users = pd.Series(np.concatenate(samples), name="userId").drop_duplicates()
        inter_sub = inter[inter["userId"].isin(sampled_users)]

        # Coverage safeguard
        item_counts_sub = inter_sub["itemId"].value_counts()
        bad_items = item_counts_sub[item_counts_sub < k_min_safeguard].index.tolist()
        if len(bad_items) > 0:
            logger.info(
                f"Safeguard found {len(bad_items)} items with < {k_min_safeguard} interactions. Adding users...")
            full_u2items = inter.groupby("userId")["itemId"].apply(set)
            todo = set(bad_items)
            extra_users = []
            cand_users = set(inter["userId"].unique()) - set(sampled_users.values)

            while todo and cand_users:
                best_u, best_cover_len = None, 0
                scan_candidates = list(cand_users)
                if len(scan_candidates) > 10000:
                    scan_candidates = RNG.choice(scan_candidates, size=10000, replace=False).tolist()
                for u in scan_candidates:
                    cover = full_u2items.get(u, set()) & todo
                    if len(cover) > best_cover_len:
                        best_cover_len = len(cover)
                        best_u = u
                if best_u is None or best_cover_len == 0: break
                extra_users.append(best_u)
                cand_users.remove(best_u)
                todo -= full_u2items[best_u]

            if extra_users:
                sampled_users = pd.concat([sampled_users, pd.Series(extra_users, name="userId")]).drop_duplicates()
                inter_sub = inter[inter["userId"].isin(sampled_users)]

        logger.info(f"Stratification complete. Subsampled dataset has {len(inter_sub)} interactions.")
        return inter_sub.drop(columns=["item_cnt", "item_bucket"], errors="ignore")

    def _apply_min_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame iteratively to produce a k-core.
        This process is repeated until the dataset stabilizes (no more users or items are removed).
        """
        if not self.subset_interactions:
            return df

        # Get k values for users and items from the config
        min_user_interactions = self.subset_interactions.get('min_user_interactions')
        min_item_interactions = self.subset_interactions.get('min_item_interactions')

        # Check if any filtering is needed
        if not min_user_interactions and not min_item_interactions:
            return df

        logger.info("Starting iterative k-core filtering...")
        if min_user_interactions:
            logger.info(f"  - Minimum user interactions: {min_user_interactions}")
        if min_item_interactions:
            logger.info(f"  - Minimum item interactions: {min_item_interactions}")

        # Start the iterative filtering loop
        while True:
            # Store the number of rows before this iteration
            initial_rows = df.shape[0]

            # --- Item filtering ---
            # Apply only if the parameter is specified
            if min_item_interactions:
                item_counts = df['itemId'].value_counts()
                logger.info(f"  - N. items: {len(item_counts)}")
                valid_items = item_counts[item_counts >= min_item_interactions].index
                df = df[df['itemId'].isin(valid_items)]

            # --- User filtering ---
            # Apply only if the parameter is specified
            if min_user_interactions:
                user_counts = df['userId'].value_counts()
                logger.info(f"  - N. users: {len(user_counts)}")
                valid_users = user_counts[user_counts >= min_user_interactions].index
                df = df[df['userId'].isin(valid_users)]

            # Check for stability
            final_rows = df.shape[0]

            logger.info(f"  Iteration: {initial_rows} -> {final_rows} interactions remaining.")

            if initial_rows == final_rows:
                # If no rows were removed in a full pass, the dataset is stable.
                break

        # Log final counts for verification
        final_user_counts = df['userId'].value_counts()
        final_item_counts = df['itemId'].value_counts()
        logger.info(f"Final stats: {len(final_user_counts)} users, {len(final_item_counts)} items.")


        return df



    def _split_random_holdout(self, data_df, base_dir, filename, test_size=0.2, validation_size=None, random_state=42):
        # validation frac
        total_val_frac = validation_size if validation_size else 0

        # group by user
        grouped = data_df.groupby('userId')
        train_dfs = []
        val_dfs = []
        test_dfs = []

        # process user
        for user_id, user_data in grouped:
            n = len(user_data)
            user_data = user_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

            # count n_interacts
            n_test = max(1, int(round(test_size * n))) if test_size > 0 else 0
            n_val = max(1, int(round(total_val_frac * n))) if validation_size else 0

            n_train = n - n_val - n_test

            # splitting
            train_part = user_data.iloc[:n_train]
            val_part = user_data.iloc[n_train:n_train + n_val] if n_val > 0 else None
            test_part = user_data.iloc[n_train + n_val:] if n_test > 0 else None

            train_dfs.append(train_part)
            if val_part is not None:
                val_dfs.append(val_part)
            if test_part is not None:
                test_dfs.append(test_part)

        # combine users splits
        train_df = pd.concat(train_dfs)
        val_df = pd.concat(val_dfs) if val_dfs else None
        test_df = pd.concat(test_dfs) if test_dfs else None

        # save results
        train_df.to_csv(os.path.join(base_dir, 'train', filename + '.csv'), sep=',', index=False)
        if val_df is not None:
            val_df.to_csv(os.path.join(base_dir, 'validation', filename + '.csv'), sep=',', index=False)
        test_df.to_csv(os.path.join(base_dir, 'test', filename + '.csv'), sep=',', index=False)

    def _split_temporal_hold_out(self, data_df, base_dir, filename, test_size=0.2, validation_size=None,
                                 timestamp_col="timestamp"):
        data_df = data_df.sort_values(by=timestamp_col)
        n = len(data_df)
        test_start = int(n * (1 - test_size))
        train_val = data_df.iloc[:test_start]
        test = data_df.iloc[test_start:]
        if validation_size:
            val_start = int(len(train_val) * (1 - validation_size))
            train = train_val.iloc[:val_start]
            val = train_val.iloc[val_start:]
        else:
            train, val = train_val, None

        train.to_csv(os.path.join(base_dir, 'train', filename + '.csv'), sep=',', index=False)
        if val is not None:
            val.to_csv(os.path.join(base_dir, 'validation', filename + '.csv'), sep=',', index=False)
        test.to_csv(os.path.join(base_dir, 'test', filename + '.csv'), sep=',', index=False)

    def _load_fixed(self, base_dir: str):
        self.train_path = os.path.join(base_dir, self.train_path)
        if self.valid_path:
            self.valid_path = os.path.join(base_dir, self.valid_path)
        if self.test_path:
            self.test_path = os.path.join(base_dir, self.test_path)