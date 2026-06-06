import os
import torch
import logging
import json
import hashlib
import yaml
import shutil
from collections import OrderedDict
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.relevant_config = None
        self.save_dir = os.path.join(os.getcwd(), 'results', 'neural_models_results')
        os.makedirs(self.save_dir, exist_ok=True)
        self._setup_experiment_directory()

    def _generate_config_hash(self):
        """Create MD5 hash from relevant configuration sections"""
        # convert config object to dictionary using dot-access
        config_dict = self._config_to_dict()

        # select sections that define experiment uniqueness
        self.relevant_config = OrderedDict()
        self.relevant_config['data'] = config_dict['data']
        self.relevant_config['model'] = {
            'name': config_dict['model']['name'],
            'params': config_dict['model']['params']
        }
        self.relevant_config['criterion'] = config_dict['criterion']
        self.relevant_config['optimizer'] = config_dict['optimizer']
        self.relevant_config['scheduler'] = config_dict['scheduler']
        self.relevant_config['train'] = {
            'epochs': config_dict['train']['epochs'],
            'batch_size': config_dict['train']['batch_size'],
            'evaluate_cutoff': config_dict['train']['evaluate_cutoff']
        }

        canonical_str = json.dumps(self.relevant_config, sort_keys=True, indent=None, separators=(',', ':'))
        return hashlib.md5(canonical_str.encode('utf-8')).hexdigest()[:16]

    def _config_to_dict(self):
        """Convert AppConfig object to nested dictionary with universal handling"""
        relevant_sections = ['data', 'model', 'criterion', 'optimizer', 'scheduler', 'train']
        config_dict = {}

        def to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: to_dict(v) for k, v in vars(obj).items()}
            elif isinstance(obj, (list, tuple)):
                return [to_dict(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            else:
                return obj

        for section in relevant_sections:
            section_obj = getattr(self.config, section)
            config_dict[section] = to_dict(section_obj)

        return config_dict

    def _setup_logging(self, filename='training.log'):
        """Sets up a dedicated log file for the current experiment."""
        log_file_path = os.path.join(self.exp_dir, filename)
        root_logger = logging.getLogger()
        self.log_filename = filename

        # remove any existing file handlers to prevent log duplication
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                root_logger.removeHandler(handler)

        # create a new file handler
        file_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # new handler to the root logger
        root_logger.addHandler(file_handler)
        logger.info(f"** Logging will be saved to: '{log_file_path}'")

    def _setup_experiment_directory(self):
        """
        Generates a hash, creates the experiment directory, and handles
        warnings for existing directories and hash collisions.
        """
        self.experiment_hash = self._generate_config_hash()

        # Use the full hash for the directory name
        self.exp_dir = os.path.join(self.save_dir, self.config.model.name, self.experiment_hash)
        os.makedirs(os.path.join(self.save_dir, self.config.model.name), exist_ok=True)
        config_path = os.path.join(self.exp_dir, 'config.json')

        # check if the directory already exists
        if os.path.isdir(self.exp_dir):
            logger.warning(f"** [!!] Directory already exists: '{self.exp_dir}'")

            # check for the config file to detect a collision
            if os.path.isfile(config_path):
                # load the old configuration
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)

                # compare the old config with the current relevant config
                if saved_config == self.relevant_config:
                    logger.warning("** [!!] This is a re-run of an identical experiment. Results may be overwritten.")
                else:
                    # HASH COLLISION [!!!]
                    logger.error("** [!!] HASH COLLISION DETECTED ")
                    logger.error(f"** [!!] A different configuration produced the same hash: {self.experiment_hash}")
                    # aborting
                    raise SystemExit("** [!!] CRITICAL: Hash collision detected. Aborting experiment.")
            else:
                logger.warning(f"Directory exists but contains no 'config.json'. Cannot check for collisions.")

        # if the directory doesn't exist, create it and save the config
        else:
            logger.info(f"** Creating new experiment directory: '{self.exp_dir}'")
            os.makedirs(self.exp_dir)
            # save config for future checks and reference
            with open(config_path, 'w') as f:
                json.dump(self.relevant_config, f, indent=4, sort_keys=True)

        # setup the dedicated log file for this experiment
        if self.config.checkpoint and (self.config.checkpoint.mode is 'evaluate_only'):
            self._setup_logging(filename='eval_only_logging.log')
        elif self.config.checkpoint and (self.config.checkpoint.mode is 'fine_tune'):
            self._setup_logging(filename='fine_tune_logging.log')
        else:
            self._setup_logging()

        logger.info(f"** Experiment directory located at: '{self.exp_dir}' ")


class OptunaSearchManager:
    """Manages an entire Optuna hyperparameter search study."""

    def __init__(self, base_config):
        self.base_config = base_config
        self.study_name = self.base_config.optuna.study_name
        self.direction = self.base_config.optuna.direction

        os.makedirs(os.path.join(os.getcwd(), 'results', 'optuna_neural_results'), exist_ok=True)

        self.study_dir = os.path.join(os.getcwd(), 'results', 'optuna_neural_results', self.study_name)
        os.makedirs(self.study_dir, exist_ok=True)
        self.trials_dir = os.path.join(self.study_dir, 'trials')
        os.makedirs(self.trials_dir, exist_ok=True)

        self.best_score = float('-inf') if self.direction == "maximize" else float('inf')
        self.best_trial_number = -1

        self._setup_study_logging()
        logger.info(f"Optuna study '{self.study_name}' initialized. Results will be saved in: {self.study_dir}")

    def _setup_study_logging(self):
        log_file = os.path.join(self.study_dir, 'optuna_study.log')
        # configure a handler specifically for the study log
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # get the logger for this specific manager and add the handler
        study_logger = logging.getLogger(__name__)
        study_logger.propagate = False
        # prevent propagation to root logger to avoid duplicate messages in console
        if study_logger.hasHandlers():
            study_logger.handlers.clear()
        study_logger.addHandler(file_handler)
        study_logger.setLevel(logging.INFO)

    def setup_trial_directory(self, trial_number: int) -> str:
        """Creates a dedicated directory for a single trial."""
        trial_dir = os.path.join(self.trials_dir, f"trial_{trial_number}")
        os.makedirs(trial_dir, exist_ok=True)
        return trial_dir

    @staticmethod
    def log_trial_start(trial_number: int, params: Dict[str, Any]):
        """
        Logs the beginning of a trial and its parameters to the study log.
        """
        logger.info(f"--- Starting Optuna Trial #{trial_number} ---")
        logger.info("  Suggested Parameters:")
        for key, value in params.items():
            logger.info(f"    - {key}: {value}")
        logger.info("-" * (25 + len(str(trial_number))))

    def log_trial_result(self, trial_number: int, trial_score: float, trial_params: Dict[str, Any]):
        """
        Logs the result and saves artifacts if it's the best.
        """
        logger.info(
            f"Trial {trial_number} finished with score: {trial_score:.5f}. Best score so far: {self.best_score:.5f}")

        is_better = (self.direction == 'maximize' and trial_score > self.best_score) or \
                    (self.direction == 'minimize' and trial_score < self.best_score)

        if is_better:
            self.best_score = trial_score
            self.best_trial_number = trial_number
            logger.info(f"** Better score found: [Trial #{trial_number} with score {self.best_score:.5f}]")

            best_config_path = os.path.join(self.study_dir, 'best_run_config.yml')
            best_model_path = os.path.join(self.study_dir, 'best_model_checkpoint.pt')
            best_log_path = os.path.join(self.study_dir, 'best_run_log.log')
            best_recs_path = os.path.join(self.study_dir, 'best_recommendations.tsv')

            trial_dir = os.path.join(self.trials_dir, f"trial_{trial_number}")
            source_model_path = os.path.join(trial_dir, 'model_checkpoint.pt')
            source_log_path = os.path.join(trial_dir, 'training.log')
            source_recs_path = os.path.join(trial_dir, 'recommendations.tsv')

            if os.path.exists(source_model_path):
                shutil.copy(source_model_path, best_model_path)
                logger.info(f"-> Saved best model checkpoint to {best_model_path}")

            if os.path.exists(source_log_path):
                shutil.copy(source_log_path, best_log_path)
                logger.info(f"-> Saved best run log to {best_log_path}")

            if os.path.exists(source_recs_path):
                shutil.copy(source_recs_path, best_recs_path)
                logger.info(f"-> Saved best recommendations to {best_recs_path}")

            with open(best_config_path, 'w') as f:
                yaml.dump(trial_params, f, indent=4)
            logger.info(f"-> Saved best run config to {best_config_path}")
