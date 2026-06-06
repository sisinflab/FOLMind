import os
import yaml
import copy
import logging
import argparse
from typing import Dict, Any, Optional
from importlib import import_module
import torch
import optuna
import pandas as pd
from safetensors.torch import load_file
from torch.utils.data import DataLoader
import warnings
import numpy as np
import random
#
# framework imports
from neural_recommender import Trainer, Evaluator, AppConfig, OptunaStudyConfig, OptunaSearchManager, ExperimentManager
from neural_recommender.utils import (
    get_dataset,
    build_interactions,
    infer_emb_dim,
    EarlyStopping,
    set_global_seed
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)


class WorkerSeeder:
    """
    A callable class to seed DataLoader workers.
    This preserves the logic `seed + worker_id` while being pickleable.
    """

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)


# # # # # # #  Optuna helper functions  # # # # # # # #
def set_nested_value(d: Dict, key_path: str, v: Any):
    """
    Sets a value in a nested dictionary using a dot-separated key path.
    Example: set_nested_value(config, 'model.params.dropout', 0.5)
    """
    keys = key_path.split('.')
    current_level = d
    for key_ in keys[:-1]:
        current_level = current_level.setdefault(key_, {})
    current_level[keys[-1]] = v


def validate_configuration_logic(cfg: AppConfig):
    """
    Checks for logical inconsistencies in the configuration before running the experiment.
    """
    logger.info("Validating configuration logic...")

    is_maximizing = cfg.eval.validation.metrics and len(cfg.eval.validation.metrics) > 0

    if is_maximizing:
        logger.info("--> Mode: Metric Maximization.")
        if cfg.scheduler and cfg.scheduler.name == 'ReduceLROnPlateau':
            scheduler_mode = cfg.scheduler.params.get('mode', 'min')
            if scheduler_mode == 'min':
                raise ValueError(
                    "** [CRITICAL LOGICAL ERROR] - The Trainer is maximizing a metric, but the "
                    "ReduceLROnPlateau scheduler is configured to 'min'. It should be 'max'."
                )

        # ensure Optuna direction is 'maximize'
        if cfg.optuna and cfg.optuna.direction == 'minimize':
            raise ValueError(
                "** [CRITICAL LOGICAL ERROR] - The Trainer is maximizing a metric, but the "
                "Optuna study direction is set to 'minimize'. It should be 'maximize'."
            )
    else:
        logger.info("--> Mode: Loss Minimization.")
        # ensure Optuna direction is 'minimize'
        if cfg.optuna and cfg.optuna.direction == 'maximize':
            raise ValueError(
                "** [CRITICAL LOGICAL ERROR] - The Trainer is minimizing loss, but the "
                "Optuna study direction is set to 'maximize'. It should be 'minimize'."
            )
    logger.info("** Configuration logic is valid.")


def find_csv(dataset_path: str) -> str:
    """
    Finds the common CSV filename present in train, validation, and test directories
    """
    splitting_path = os.path.join(dataset_path, 'splitting')
    dirs_to_check = {
        'train': os.path.join(splitting_path, 'train'),
        'validation': os.path.join(splitting_path, 'validation'),
        'test': os.path.join(splitting_path, 'test')
    }

    for dir_name, dir_path in dirs_to_check.items():
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"The '{dir_name}' directory does not exist at: {dir_path}")

    try:
        train_csvs = {f for f in os.listdir(dirs_to_check['train']) if f.endswith('.csv')}
        val_csvs = {f for f in os.listdir(dirs_to_check['validation']) if f.endswith('.csv')}
        test_csvs = {f for f in os.listdir(dirs_to_check['test']) if f.endswith('.csv')}
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Error listing files. Ensure directories exist and have read permissions. Original error: {e}")

    # intersection
    common_files = train_csvs.intersection(val_csvs, test_csvs)

    if len(common_files) == 1:
        return common_files.pop()

    if len(common_files) == 0:
        raise FileNotFoundError(
            "No common CSV file found across the train, validation, and test directories."
        )

    raise ValueError(
        f"Ambiguous setup: Multiple common CSV files found: {list(common_files)}. "
        "Please ensure only one CSV file is shared across all splits."
    )


# ------------------------------------------------
def run_experiment(cfg: AppConfig, exp_dir: str, initial_best_score: Optional[float] = None) -> float:
    """
    Runs a single, complete training and evaluation experiment based on a given configuration.

    It is designed to be called either once for a single run or multiple times by the
    Optuna objective function.

    Args:
        cfg: A validated AppConfig object containing all experiment parameters.
        exp_dir: directory of the single run experiment or Optuna trial.
        initial_best_score: initial score in the hyperparam search of past trials
    """
    # seed
    seed = getattr(cfg.data, "seed", None)
    if seed is not None:
        set_global_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    worker_seeder = WorkerSeeder(seed)

    # data loading and processing
    base_path = os.getcwd()
    dataset_path = os.path.join(base_path, 'data', cfg.data.dataset_dir)

    logger.info(f"Loading data from: {dataset_path}")

    _name = find_csv(dataset_path=dataset_path)
    logger.info(f"Loading ratings from: **{_name}** ")
    train_df = pd.read_csv(os.path.join(dataset_path, 'splitting', 'train', _name))
    val_df = pd.read_csv(os.path.join(dataset_path, 'splitting', 'validation', _name))
    test_df = pd.read_csv(os.path.join(dataset_path, 'splitting', 'test', _name))

    all_users = sorted(set(train_df.userId) | set(val_df.userId) | set(test_df.userId))
    user2idx = {u: i for i, u in enumerate(all_users)}
    n_users = len(all_users)

    # pre-trained embeddings
    user_emb_dir = os.path.join(dataset_path, f'{cfg.data.user_emb_source}_output')
    item_emb_dir = os.path.join(dataset_path, f'{cfg.data.item_emb_source}_output')
    logger.info(f"Loading user embeddings from: {user_emb_dir}")
    logger.info(f"Loading item embeddings from: {item_emb_dir}")

    ud = load_file(os.path.join(user_emb_dir,
                                f'user_{cfg.data.user_emb_type}_embeddings_{cfg.data.user_emb_model}.safetensors'))
    it = load_file(os.path.join(item_emb_dir,
                                f'item_{cfg.data.item_emb_type}_embeddings_{cfg.data.item_emb_model}.safetensors'))

    user_data = {uid: vec for k, vec in ud.items() for _, uid, t in [k.split('_')] if t == cfg.data.user_info_type}
    item_data = {int(k): v for k, v in it.items()}

    emb_items = sorted(item_data.keys())
    item2idx = {i: j for j, i in enumerate(emb_items)}
    n_items = len(emb_items)

    # build I-M
    logger.info("Building interaction matrices...")
    train_mat = build_interactions(
        train_df, user2idx, item2idx,
        binary=cfg.data.binarize,
        rating_col=cfg.data.rating_col,
        pos_threshold=cfg.data.pos_threshold
    )
    val_mat = build_interactions(
        val_df, user2idx, item2idx,
        binary=cfg.data.binarize,
        rating_col=cfg.data.rating_col,
        pos_threshold=cfg.data.pos_threshold
    )
    test_mat = build_interactions(
        test_df, user2idx, item2idx,
        binary=cfg.data.binarize,
        rating_col=cfg.data.rating_col,
        pos_threshold=cfg.data.pos_threshold
    )

    # create Dataset and DataLoader
    ds = get_dataset(name=cfg.data.dataset_type, config=cfg.data, interactions=train_mat)
    train_loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=4, worker_init_fn=worker_seeder,
                              generator=g, persistent_workers=False)

    # prepare pre-trained Embedding tensors
    user_emb_tensor = torch.stack([torch.as_tensor(user_data[str(u)], dtype=torch.float32) for u in all_users])
    item_emb_tensor = torch.stack([torch.as_tensor(item_data[i], dtype=torch.float32) for i in emb_items])
    u_emb_dim, i_emb_dim = infer_emb_dim(user_emb_tensor, item_emb_tensor)

    # initialize Model, Criterion, and Optimizer
    logger.info(f"** Initializing model: {cfg.model.name}")
    ModelClass = getattr(import_module(f"neural_recommender.models.{cfg.model.name}"), cfg.model.name)

    model_params = cfg.model.params.copy()

    # ------------------------------------------------------------- FAMILIES LOGIC
    # get family logic -> (defaults to 'standard' if not defined)
    family = getattr(ModelClass, 'model_family', 'standard')

    if family == 'graph':
        logger.info(f"-> Model identified as Graph-based (FAMILY: {family}). Injecting interaction matrix.")
        # graph models need the interaction matrix and explicit device for sparse construction
        model_params['interaction_matrix'] = train_mat
        model_params['device'] = cfg.model.device

    # NOTE: the code is modular and allows to add any family and any kind of
    # input (for the specific family) in this section
    # -------------------------------------------------------------

    model = ModelClass(
        user_emb_tensor=user_emb_tensor, item_emb_tensor=item_emb_tensor,
        u_emb_dim=u_emb_dim, i_emb_dim=i_emb_dim, **model_params
    ).to(cfg.model.device)

    # model = ModelClass(
    #     user_emb_tensor=user_emb_tensor, item_emb_tensor=item_emb_tensor,
    #     u_emb_dim=u_emb_dim, i_emb_dim=i_emb_dim, **cfg.model.params
    # ).to(cfg.model.device)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # MODEL CHECKPOINT LOGIC # # # # # # # # #
    if cfg.checkpoint and cfg.checkpoint.load_from_dir:
        checkpoint_path = os.path.join(os.getcwd(), cfg.checkpoint.load_from_dir)
        if os.path.exists(checkpoint_path):
            logger.info(f"** Loading model checkpoint from: {checkpoint_path}")
            # load state dict, making sure to map to the correct device
            model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.model.device))
            logger.info("** Model state loaded successfully.")
        else:
            logger.error(f"Checkpoint file not found at: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    logger.info(f"** Initializing optimizer: {cfg.optimizer.name}")
    Opt = getattr(torch.optim, cfg.optimizer.name)
    optimizer = Opt(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    logger.info(f"** Initializing criterion: {cfg.criterion.name}Loss")
    CritClass = getattr(import_module(f"neural_recommender.criterions.{cfg.criterion.name}"),
                        f"{cfg.criterion.name}Loss")
    criterion = CritClass(**cfg.criterion.params)

    scheduler = None
    if cfg.scheduler:
        logger.info(f"** Initializing scheduler: {cfg.scheduler.name}")
        Sch = getattr(torch.optim.lr_scheduler, cfg.scheduler.name)
        scheduler = Sch(optimizer, **cfg.scheduler.params)

    # setup Evaluators and Early Stopping
    early_stopper = EarlyStopping(patience=cfg.early_stopping.patience, monitor=cfg.early_stopping.monitor)

    evaluator_val = Evaluator(
        train_matrix=train_mat,
        test_matrix=val_mat,
        cutoff=cfg.eval.validation.cutoff,
        simple_metrics=cfg.eval.validation.metrics
    )
    evaluator_test = Evaluator(
        train_matrix=train_mat,
        test_matrix=test_mat,
        cutoff=cfg.eval.test.cutoff,
        simple_metrics=cfg.eval.test.metrics
    )

    # initialize and Run the Trainer
    trainer = Trainer(
        config=cfg,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        eval_every=cfg.eval.validation.eval_every,
        n_users=n_users,
        n_items=n_items,
        train_loader=train_loader,
        scheduler=scheduler,
        early_stopper=early_stopper,
        val_evaluator=evaluator_val,
        test_evaluator=evaluator_test,
        exp_dir=exp_dir,
        initial_best_score=initial_best_score,
    )

    # check the mode if a checkpoint was loaded
    if cfg.checkpoint and cfg.checkpoint.mode == 'evaluate_only':
        logger.info("Checkpoint mode is 'evaluate_only'. Skipping training.")

        # Call cold start method in Trainer.py
        final_metrics = trainer.evaluate_on_test()
        logger.info(f"Final evaluation metrics reported to run_experiment: {final_metrics}")

        return 0.0

    # if no checkpoint or if mode is 'fine_tune', run the full training process
    logger.info("Starting full training process...")

    best_score = trainer.train()

    logger.info("=" * 20 + " EXPERIMENT FINISHED " + "=" * 20)
    return best_score


# Optuna Objective Function
def objective(trial: optuna.trial.Trial, base_config_dict: Dict, optuna_config: OptunaStudyConfig,
              manager: OptunaSearchManager) -> float:
    """
    The objective function for an Optuna study. It configures and runs an experiment for one trial.
    """
    trial_config_dict = copy.deepcopy(base_config_dict)

    logger.info(f"\n--- Starting Optuna Trial #{trial.number} ---")
    for hp in optuna_config.search_space:
        if hp.type == 'categorical':
            val = trial.suggest_categorical(hp.param_name, **hp.params)
        elif hp.type == 'float':
            val = trial.suggest_float(hp.param_name, **hp.params)
        elif hp.type == 'int':
            val = trial.suggest_int(hp.param_name, **hp.params)

            # console log for immediate feedback
            logger.info(f"  - Suggesting {hp.param_name} = {val}")
            set_nested_value(trial_config_dict, hp.param_name, val)

    # log the chosen parameters to the dedicated optuna_study.log file
    manager.log_trial_start(trial.number, trial.params)

    try:
        config = AppConfig(**trial_config_dict)
    except Exception as e:
        logger.warning(f"Skipping trial due to invalid config: {e}")
        raise optuna.exceptions.TrialPruned()

    # get the dedicated directory for this trial from the manager
    trial_exp_dir = manager.setup_trial_directory(trial.number)

    # get the current best score between all trails from the manager
    current_best_from_study = manager.best_score

    # run the experiment, passing in the best score so far
    score = run_experiment(config, trial_exp_dir, initial_best_score=current_best_from_study)

    # log the result with the manager, which will handle saving the best artifacts
    manager.log_trial_result(trial.number, score, trial.params)

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOLMind Neural Recommender System Training and Hyperparameter Search")
    parser.add_argument(
        '--config',
        type=str,
        default='ml1m_optuna_neural_baseline.yml',  # example/default
        help='Name of the configuration YAML file located in ./neural_configs.'
    )
    args = parser.parse_args()

    config_path = os.path.join(os.getcwd(), 'neural_configs', args.config)
    logger.info(f"** Loading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    app_config = AppConfig(**config_dict)

    # validate direction for optuna and scheduler option
    validate_configuration_logic(app_config)

    # if "optuna" section is present in the file:
    if app_config.optuna:
        # # # # # # # #  OPTUNA HYPERPARAMETER SEARCH  # # # # # # # #
        logger.info(f"Starting Optuna search defined in '{args.config}'")

        # initialize the OPTUNA search manager
        optuna_manager = OptunaSearchManager(base_config=app_config)

        study = optuna.create_study(
            direction=app_config.optuna.direction,
            study_name=app_config.optuna.study_name
        )

        study.optimize(
            lambda trial: objective(trial, config_dict, app_config.optuna, optuna_manager),
            n_trials=app_config.optuna.n_trials
        )

        logger.info("=" * 80)
        logger.info("Optuna search finished.")
        logger.info(f"Best trial: #{study.best_trial.number}")
        logger.info(f" -> Manager logged best trial: #{optuna_manager.best_trial_number}")
        logger.info(f" -> Value ({app_config.early_stopping.monitor}): {study.best_trial.value:.5f}")
        logger.info(" -> Best parameters found:")
        for key, value in study.best_trial.params.items():
            logger.info(f"    - {key}: {value}")
        logger.info(f"\nAll results for this study are centralized in: {optuna_manager.study_dir}")

    else:
        # # # # # # # #  SINGLE EXPERIMENT RUN  # # # # # # # #
        logger.info(f"No 'optuna' section found in '{args.config}'. Starting a single experiment run.")

        # use ExperimentManager for a single run
        exp_manager = ExperimentManager(config=app_config)
        run_experiment(app_config, exp_dir=exp_manager.exp_dir)

# Examples:
# python run_neural_experiments.py --config CONFIGNAME.yml
# (with nohup)
# nohup python run_neural_experiments.py --config CONFIGNAME.yml > CONFIGNAME.log 2>&1 &

# NOFOL
# nohup python run_neural_experiments.py --config genomeNOFOL_optunaNewNMF_LLMEmb_QWEN4.yml > genomeNOFOL_optunaNewNMF_LLMEmb_QWEN4.log 2>&1 &
# nohup python run_neural_experiments.py --config ml1mNOFOL_optunaNewNMF_LLMEmb_QWEN4.yml > ml1mNOFOL_optunaNewNMF_LLMEmb_QWEN4.log 2>&1 &

# FOL
# nohup python run_neural_experiments.py --config genome_optunaNewNMF_LLMEmb_QWEN4.yml > genome_optunaNewNMF_LLMEmb_QWEN4.log 2>&1 &
# nohup python run_neural_experiments.py --config ml1m_optunaNewNMF_LLMEmb_QWEN4.yml > ml1m_optunaNewNMF_LLMEmb_QWEN4.log 2>&1 &

# baselines
# nohup python run_neural_experiments.py --config genome_optuna_neural_baseline.yml > genome_optuna_neural_baseline.log 2>&1 &
# nohup python run_neural_experiments.py --config ml1m_optuna_neural_baseline.yml > ml1m_optuna_neural_baseline.log 2>&1 &