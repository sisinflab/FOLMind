import os
import re
import time
import pickle
import logging
import numpy as np
import pandas as pd
import optuna
import torch
import random
import csv
from optuna.samplers import TPESampler
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from .models import get_model_class, get_model_config_class, BaseModelConfig
from folm_core.evaluator.Evaluator import Evaluator
from .Folminder import Folminder
from .Refiner import Refiner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.enable_default_handler()


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_matrix(df, user_map=None, item_map=None):
    if user_map is None:
        users = sorted(pd.to_numeric(df['userId'], errors='coerce').dropna().unique())
        user_map = {int(u): i for i, u in enumerate(users)}
    if item_map is None:
        items = sorted(pd.to_numeric(df['itemId'], errors='coerce').dropna().unique())
        item_map = {int(m): i for i, m in enumerate(items)}

    mat = np.zeros((len(user_map), len(item_map)), dtype=np.float32)

    df_copy = df.copy()
    df_copy['userId'] = pd.to_numeric(df_copy['userId'], errors='coerce')
    df_copy['itemId'] = pd.to_numeric(df_copy['itemId'], errors='coerce')
    df_copy.dropna(subset=['userId', 'itemId'], inplace=True)
    df_copy = df_copy.astype({'userId': int, 'itemId': int})

    # Determine if feedback is explicit or implicit
    rating_col = None
    if 'rating' in df_copy.columns:
        rating_col = 'rating'

    for _, row in df_copy.iterrows():
        if row.userId in user_map and row.itemId in item_map:
            user_idx = user_map[row.userId]
            item_idx = item_map[row.itemId]

            if rating_col:
                rating = pd.to_numeric(row.get(rating_col), errors='coerce')
                if pd.notna(rating):
                    mat[user_idx, item_idx] = float(rating)
            else:
                # implicit feedback
                mat[user_idx, item_idx] = 1.0

    return mat, user_map, item_map


def store_recommendation(recs: List[Tuple[int, int, float]], path: str):
    """
    Saves recommendations to a TSV file.
    Format: user_id \t item_id \t score
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for user_id, item_id, score in recs:
            writer.writerow([user_id, item_id, score])

    logger.info(f"Recommendations saved to {path}")


class Recommender:
    def __init__(self, datadir: str, recs_config: Any, recs_dir: str):
        self.datadir = datadir
        self.config = recs_config
        self.recs_dir = recs_dir
        self.eval_conf = self.config.evaluation
        set_global_seeds(self.config.seed)
        os.makedirs(os.path.join(os.getcwd(), 'data', self.datadir, 'recs'), exist_ok=True)

        self.results_dir = os.path.join(os.getcwd(), 'results', self.datadir)
        os.makedirs(self.results_dir, exist_ok=True)

        self.user_map, self.item_map = {}, {}
        self.train_mat, self.valid_mat, self.test_mat = None, None, None
        self.filters = None
        logger.info(msg="** Recommender initialized.")

    def run(self):
        self.load_splits()

        for model_name, model_params in self.config.models.items():
            config_class = get_model_config_class(model_name)
            model_cfg = config_class(**model_params)

            logger.info(f"\n{'=' * 20} PROCESSING MODEL: {model_name.upper()} {'=' * 20}")

            if model_cfg.hpo.enabled:
                self._run_hpo_for_model(model_name, model_cfg)
            else:
                logger.info("HPO disabled. Running single experiment with fixed parameters.")
                self._run_single_experiment(model_name, model_cfg)

    def _prepare_instantiation_params(self, model_cfg: BaseModelConfig, params_override: Dict = None) -> Dict:
        """
        Prepare params
        """
        instantiation_params = model_cfg.dict()
        if not params_override:
            return instantiation_params

        folm_fields = model_cfg.folm.__class__.__fields__.keys()
        for key, value in params_override.items():
            if key in folm_fields:
                instantiation_params['folm'][key] = value
            else:
                instantiation_params[key] = value
        return instantiation_params

    def _run_single_experiment(self, model_name: str, model_cfg: BaseModelConfig, params_override: Dict = None):
        instantiation_params = self._prepare_instantiation_params(model_cfg, params_override)

        # params logging
        run_params_for_logging = {k: v for k, v in instantiation_params.items() if
                                  k not in ['meta', 'hpo', 'folm', 'embedding_init']}
        if instantiation_params['folm']['refined']:
            run_params_for_logging['folm'] = instantiation_params['folm']
        self._print_parameters(run_params_for_logging, model_name)

        model_class = get_model_class(model_name)
        model = model_class(datadir=self.datadir, **instantiation_params)

        start_time = time.time()

        fit_kwargs = {}

        current_folm_config = model_cfg.folm.__class__(**instantiation_params['folm'])
        if current_folm_config.refined:

            self.load_precomputed_similarities(model_cfg=model_cfg)
            model.fit(self.train_mat, **fit_kwargs)
            folminder = Folminder(self.filters, folm_config=current_folm_config, model=model)
            pred_matrix = folminder.apply()
        else:
            model.fit(self.train_mat, **fit_kwargs)
            pred_matrix = model.compute_predictions()

        execution_time = time.time() - start_time
        logger.info(f"Training and prediction took: < {execution_time:.3f}s >")

        validation_results = self._call_eval(pred_matrix, model_name, "VALIDATION")
        test_results = self._call_eval(pred_matrix, model_name, "TEST")

        self._save_experiment_results(
            model_name=model_name,
            dataset_name=self.datadir,
            params=run_params_for_logging,
            validation_results=validation_results,
            test_results=test_results,
            execution_time_seconds=execution_time
        )

        if model_cfg.meta.save_recs:
            self._save_recommendations(pred_matrix, model_name)

    def _run_hpo_for_model(self, model_name: str, model_cfg: BaseModelConfig):
        hpo_conf = model_cfg.hpo
        logger.info(f"Starting HPO for {model_name} with {hpo_conf.n_trials} trials.")

        def objective(trial: optuna.trial.Trial) -> float:
            params = {}
            # base model params
            for p_name, p_conf in hpo_conf.search_space.items():
                params[p_name] = self._suggest_from_config(trial, p_name, p_conf)

            # folm params (if defined)
            folm_conf = model_cfg.folm
            if folm_conf.refined and folm_conf.search_space:
                for p_name, p_conf in folm_conf.search_space.items():
                    params[p_name] = self._suggest_from_config(trial, p_name, p_conf)

            logger.info(f"--- HPO Trial #{trial.number} - Params: {params} ---")

            try:

                instantiation_params = self._prepare_instantiation_params(model_cfg, params)

                model_class = get_model_class(model_name)
                model = model_class(datadir=self.datadir, **instantiation_params)

                # fit and predict
                current_folm_config = model_cfg.folm.__class__(**instantiation_params['folm'])
                if current_folm_config.refined:
                    self.load_precomputed_similarities(model_cfg=model_cfg)
                    model.fit(self.train_mat)
                    folminder = Folminder(self.filters, folm_config=current_folm_config, model=model)
                    pred_matrix = folminder.apply()
                else:
                    model.fit(self.train_mat)
                    pred_matrix = model.compute_predictions()

                # evaluate
                metric_to_optimize = hpo_conf.metric_to_optimize
                cutoff = int(metric_to_optimize.split('@')[1])
                metric_name_str = metric_to_optimize.split('@')[0]

                eval_config = {"cutoffs": [cutoff], "accuracy_metrics": [metric_name_str]}
                evaluator = Evaluator(train_matrix=self.train_mat, test_matrix=self.valid_mat, **eval_config)
                results = evaluator.evaluate(pred_matrix)

                return results.get(metric_to_optimize, -1.0)
            except Exception as e:
                logger.error(f"Trial #{trial.number} failed: {e}", exc_info=True)
                return -1.0

        sampler = TPESampler(seed=self.config.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=hpo_conf.n_trials)

        best_params = study.best_trial.params
        logger.info(f"\n--- HPO for {model_name} COMPLETED ---")
        logger.info(f"Best score ({hpo_conf.metric_to_optimize}): {study.best_value:.5f}")
        logger.info(f"Best parameters found: {best_params}")

        df_results = study.trials_dataframe()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hpo_trials_{model_name}_{timestamp}.csv"

        # Save to original recs directory
        output_path_recs = os.path.join(self.recs_dir, filename)
        df_results.to_csv(output_path_recs, index=False)
        logger.info(f"HPO trial results saved to: {output_path_recs}")

        # Save to results directory
        output_path_results = os.path.join(self.results_dir, filename)
        df_results.to_csv(output_path_results, index=False)

        logger.info(f"\n--- RUNNING FINAL EVALUATION FOR {model_name} WITH BEST PARAMS ---")
        self._run_single_experiment(model_name, model_cfg, params_override=best_params)

    def _find_data_file(self, split='train'):
        data_dir = os.path.join('data', self.datadir, 'splitting', split)
        for filename in os.listdir(data_dir):
            if filename.lower().endswith('.csv'):
                path = os.path.join(data_dir, filename)
                logger.info(f"Data file found for '{split}' split: {filename}")
                return path
        return None

    def load_splits(self):
        logger.info("Loading data splits...")
        train_path = self._find_data_file('train')
        valid_path = self._find_data_file('validation')
        test_path = self._find_data_file('test')

        if not train_path:
            raise FileNotFoundError("Primary training data file not found.")

        all_dfs, loaded_dfs = [], {}
        path_map = {'train': train_path, 'validation': valid_path, 'test': test_path}

        for split, path in path_map.items():
            if path:
                try:
                    df = pd.read_csv(path, sep='\t' if '.tsv' in path else ',',
                                     header=None if '.tsv' in path else 'infer')
                    loaded_dfs[split] = df
                    all_dfs.append(df)
                except Exception as e:
                    logger.warning(f"Could not load or parse {path}: {e}")

        if not all_dfs: raise ValueError("No data could be loaded.")

        full_df = pd.concat(all_dfs, ignore_index=True)
        all_users = sorted(pd.to_numeric(full_df['userId'], errors='coerce').dropna().unique())
        all_items = sorted(pd.to_numeric(full_df['itemId'], errors='coerce').dropna().unique())

        self.user_map = {int(u): i for i, u in enumerate(all_users)}
        self.item_map = {int(m): i for i, m in enumerate(all_items)}

        if 'train' in loaded_dfs:
            ratings_df = loaded_dfs['train']
            self.train_mat, _, _ = build_matrix(ratings_df, self.user_map, self.item_map)

        if 'validation' in loaded_dfs:
            self.valid_mat, _, _ = build_matrix(loaded_dfs['validation'], self.user_map, self.item_map)
        if 'test' in loaded_dfs:
            self.test_mat, _, _ = build_matrix(loaded_dfs['test'], self.user_map, self.item_map)

        logger.info(f"Data loading complete. Framework has {len(self.user_map)} users and {len(self.item_map)} items.")

    def load_precomputed_similarities(self, model_cfg: BaseModelConfig):
        """
        Loads pre-computed similarity data. If the data does not exist, it triggers the
        Refiner to generate it.
        """
        folm_conf = model_cfg.folm

        filters_path = os.path.join(self.recs_dir,
                                    f'refined_similarities_{folm_conf.entity_emb}_{folm_conf.type_emb}_{folm_conf.info_emb}_{folm_conf.emb_source_model}.pkl')

        if not os.path.exists(filters_path):
            logger.info(f"Refined similarities file not found. Generating: {filters_path}")

            refiner = Refiner(dataset_dir=self.datadir, recs_dir=self.recs_dir, model_cfg=model_cfg)
            refiner.compose_refining()

        with open(filters_path, 'rb') as f:
            refined_data = pickle.load(f)

        self.filters = self._normalize_filters(refined_data, entity_type=folm_conf.entity_emb)

    def _normalize_filters(self, refined_data: dict, entity_type: str) -> dict:
        processed = {}
        id_map = self.user_map if entity_type == 'user' else self.item_map

        # Define separate regex patterns
        user_pattern = re.compile(r'user_(\d+)')
        # For items, we expect the key to be the ID itself, so we just check if it's a digit string

        for raw_entity_id, data in refined_data.items():
            raw_id = None

            # Select the correct parsing logic based on entity_type
            if entity_type == 'user':
                m = user_pattern.search(str(raw_entity_id))
                if m:
                    raw_id = int(m.group(1))
            elif entity_type == 'item':
                # For items, the key is the ID. Ensure it's a valid number.
                if str(raw_entity_id).isdigit():
                    raw_id = int(raw_entity_id)

            if raw_id is None or raw_id not in id_map:
                continue

            entity_idx = id_map[raw_id]
            sim_idxs, sim_scores = [], []

            for sim_str, score in zip(data.get("similar_entities", []), data.get("similarity_scores", [])):
                sim_raw_id = None
                if entity_type == 'user':
                    mm = user_pattern.search(str(sim_str))
                    if mm:
                        sim_raw_id = int(mm.group(1))
                elif entity_type == 'item':
                    if str(sim_str).isdigit():
                        sim_raw_id = int(sim_str)

                if sim_raw_id is not None and sim_raw_id in id_map:
                    sim_idxs.append(id_map[sim_raw_id])
                    sim_scores.append(float(score))

            processed[entity_idx] = {"similar_entities": sim_idxs, "similarity_scores": sim_scores}

        logger.info(f"Normalized {len(processed)} {entity_type} similarities for refinement.")
        return processed

    def _call_eval(self, pred_matrix, name, mode) -> Optional[Dict[str, float]]:
        test_matrix = self.valid_mat if mode == 'VALIDATION' else self.test_mat
        if test_matrix is None:
            logger.info(f"No {mode} data available, skipping evaluation.")
            return
        evaluator = Evaluator(train_matrix=self.train_mat, test_matrix=test_matrix, **self.eval_conf)
        results = evaluator.evaluate(pred_matrix)
        logger.info(f"\n- {mode} Evaluation results for {name}:\n{pd.Series(results)}")
        return results

    def _save_recommendations(self, pred_matrix: np.ndarray, name: str):
        """
        Extracts Top-K items and scores for each user and saves them to a TSV file.
        Format: user_idx \t item_idx \t score
        """
        recs_list = []
        top_k = self.config.top_k
        n_users = pred_matrix.shape[0]

        logger.info(f"Generating top-{top_k} recommendations for {name}...")

        for u_idx in range(n_users):
            # If user has no training data (and likely no test data in index), skip or handle
            if u_idx >= self.train_mat.shape[0]:
                continue

            scores = pred_matrix[u_idx].copy()

            # Mask items already seen in training set (set to -infinity)
            scores[self.train_mat[u_idx].nonzero()] = -np.inf

            # Efficient Top-K extraction
            if top_k < len(scores):
                # argpartition moves the k-th largest element to position -k.
                # everything after -k is larger than everything before.
                unsorted_top_k = np.argpartition(-scores, top_k)[:top_k]
                # sort just the top k
                sorted_top_k = unsorted_top_k[np.argsort(-scores[unsorted_top_k])]
            else:
                # if fewer items than k, sort all
                sorted_top_k = np.argsort(-scores)

            # Build list of tuples
            for i_idx in sorted_top_k:
                score_val = scores[i_idx]
                # Do not recommend masked items
                if score_val == -np.inf:
                    continue
                recs_list.append((u_idx, i_idx, float(score_val)))

        # Save using the unified TSV function
        out_path = os.path.join(self.recs_dir, f"{name}_recommendations.tsv")
        store_recommendation(recs_list, out_path)

    def _save_experiment_results(self, model_name: str, dataset_name: str, params: dict,
                                 validation_results: Optional[dict], test_results: Optional[dict],
                                 execution_time_seconds: Optional[float] = None):
        """
        Saves the parameters and results of an experiment to a text file.
        The file is saved in 'results/<dataset_name>/<model_name>.txt'.
        """
        results_dir = os.path.join(os.getcwd(), 'results', dataset_name)
        os.makedirs(results_dir, exist_ok=True)

        file_path = os.path.join(results_dir, f"{model_name}.txt")

        with open(file_path, 'w') as f:
            f.write(f"Experiment Results for Model: {model_name}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if execution_time_seconds is not None:
                f.write(f"Execution Time: {execution_time_seconds:.3f} seconds\n")
            f.write("=" * 40 + "\n\n")

            f.write("Parameters:\n")
            for key, value in params.items():
                if isinstance(value, dict):
                    f.write(f"  {key}:\n")
                    for sub_key, sub_value in value.items():
                        f.write(f"    {sub_key}: {sub_value}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n" + "-" * 40 + "\n\n")

            f.write("Validation Results:\n")
            if validation_results:
                for metric, score in validation_results.items():
                    f.write(f"  {metric}: {score:.5f}\n")
            else:
                f.write("  No validation data provided.\n")
            f.write("\n" + "-" * 40 + "\n\n")

            f.write("Test Results:\n")
            if test_results:
                for metric, score in test_results.items():
                    f.write(f"  {metric}: {score:.5f}\n")
            else:
                f.write("  No test data provided.\n")
            f.write("\n" + "=" * 40 + "\n")

        logger.info(f"Experiment results saved to: {file_path}")

    @staticmethod
    def _suggest_from_config(trial: optuna.trial.Trial, param_name: str, param_config: dict):
        param_type = param_config.get("type")
        if param_type == "categorical":
            return trial.suggest_categorical(param_name, param_config["values"])
        elif param_type == "int":
            return trial.suggest_int(param_name, param_config["low"], param_config["high"])
        elif param_type == "float":
            return trial.suggest_float(param_name, param_config["low"], param_config["high"],
                                       log=param_config.get("log", False))
        else:
            raise ValueError(f"Unsupported parameter type in HPO config: {param_type}")

    @staticmethod
    def _print_parameters(params: dict, name: str):
        logger.info("\n*********************************************")
        logger.info(f"Running {name} with parameters:")
        for k, v in params.items():
            if isinstance(v, dict):
                logger.info(f"\t{k}:")
                for sub_k, sub_v in v.items():
                    logger.info(f"\t  {sub_k}: {sub_v}")
            else:
                logger.info(f"\t{k}: {v}")
        logger.info("*********************************************")