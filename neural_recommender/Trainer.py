import os
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import numpy as np
from tqdm import tqdm
import logging
import time
from .utils import store_recommendation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
            self,
            config,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.Module,
            eval_every: int,
            n_users: int,
            n_items: int,
            train_loader: DataLoader,
            exp_dir: str,
            scheduler: Optional[torch.optim.lr_scheduler] = None,
            early_stopper: Optional[Any] = None,
            device: Optional[str] = None,
            val_evaluator: Optional[Any] = None,
            test_evaluator: Optional[Any] = None,
            initial_best_score: Optional[float] = None
    ):
        self.config = config
        self.exp_dir = exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        # logger
        self.logger = None
        self._setup_logging()  # create an isolated logger
        # device and model
        self.device = device or self.config.model.device if self.config.model.device else 'cpu'
        self.logger.info(f" ** Running on {self.device}")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.scheduler = scheduler
        self.early_stopper = early_stopper

        # full ranking evaluator
        self.evaluator = val_evaluator
        self.test_evaluator = test_evaluator or val_evaluator
        self.eval_batch = self.config.train.eval_batch
        self.n_users = n_users
        self.n_items = n_items
        self.eval_every = eval_every

        # setup model checkpoint path
        self.save_path = os.path.join(self.exp_dir, "model_checkpoint.pt")
        # determine the optimization mode
        self.maximize_metric = self.evaluator is not None
        if self.maximize_metric:
            self.logger.info("** Trainer is in METRIC MAXIMIZATION mode.")
        else:
            self.logger.info("** Trainer is in LOSS MINIMIZATION mode (no evaluator provided).")

        self.trial_best_score = None  # best score found within this specific trial

        if initial_best_score is None:
            # this is a single run or the first Optuna trial. Set default based on mode
            self.overall_best_score = float('-inf') if self.maximize_metric else float('inf')
        else:
            # in this case a score was passed from the Optuna manager
            self.overall_best_score = initial_best_score

        self.logger.info(f"** Trainer initialized [Maximize: {self.maximize_metric}]. "
                         f"Will save checkpoint if score is better than: {self.overall_best_score:.5f}")

    def _setup_logging(self):
        """Dedicated, isolated log file for the current trial / single experiment."""
        self.logger = logging.getLogger(self.exp_dir)

        # prevent log messages from propagating to the root logger
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

        # clear any handlers that might exist from a previous, unrelated run
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # create a file handler that points to this trial's log file
        log_file_path = os.path.join(self.exp_dir, 'training.log')
        file_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # add the new handler directly to this trial's logger
        self.logger.addHandler(file_handler)
        self.logger.info(f"** Individual run logging will be saved to: '{log_file_path}'")

    def train(self) -> float:
        """Run the full training and validation loop and return the best score"""
        monitor = self.config.early_stopping.monitor
        trigger_max_scheduler = False

        for epoch in range(1, self.config.train.epochs + 1):
            start_time = time.time()
            train_loss = self._train_one_epoch(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f} — current LR = {current_lr:.8f} "
                             f"| epoch executed in {(time.time() - start_time):.2f} seconds")

            current_score = None
            if self.maximize_metric:
                if epoch == 1 or epoch % self.eval_every == 0 or epoch == self.config.train.epochs:
                    self.model.eval()
                    metrics = self._eval_model(evaluator=self.evaluator)
                    current_score = metrics.get(monitor)
                    trigger_max_scheduler = True
            else:  # minimization mode
                current_score = train_loss

            if current_score is not None:
                # check if this is the best score for this trial
                is_trial_best = False
                if self.trial_best_score is None:
                    is_trial_best = True
                elif self.maximize_metric and current_score > self.trial_best_score:
                    is_trial_best = True
                elif not self.maximize_metric and current_score < self.trial_best_score:
                    is_trial_best = True

                if is_trial_best:
                    self.trial_best_score = current_score

                # check best score
                is_overall_best = False
                if self.maximize_metric and current_score > self.overall_best_score:
                    is_overall_best = True
                elif not self.maximize_metric and current_score < self.overall_best_score:
                    is_overall_best = True

                if is_overall_best:
                    self.overall_best_score = current_score
                    torch.save(self.model.state_dict(), self.save_path)
                    metric_name = monitor if self.maximize_metric else "loss"
                    self.logger.info(
                        f"** New best model found and saved with {metric_name} = {current_score:.5f} **")

                # early stopping (always uses the validation metric)
                if self.maximize_metric and self.early_stopper and self.early_stopper.step(current_score):
                    self.logger.info(f"** Early stopping triggered at epoch {epoch}")
                    break

            # scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # pull “mode” out of the dict
                    mode = self.config.scheduler.params.get('mode', 'min')
                    if mode == 'max' and trigger_max_scheduler:
                        # metric to maximize
                        trigger_max_scheduler = False  # set again to False waiting the next current metric update
                        self.scheduler.step(current_score)
                    elif mode == 'min':
                        # treat everything else as a loss‐minimizer
                        self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()

            self.logger.info(msg="\n" + "=" * 70 + "\n")

        # test + final validation eval with best model
        if os.path.exists(self.save_path):
            self.logger.info(f"Training complete. Loading best model for final evaluation.")
            self.model.load_state_dict(torch.load(self.save_path))
            self._eval_model(evaluator=self.evaluator) # sanity check
            self.evaluate_on_test() # evaluate and save metrics if save_recs = True (default)
        else:
            self.logger.warning('** [!!] No new overall best model was found in this trial. SKIPPING FINAL '
                                'EVALUATION. **')

        # return the best score achieved in this trial
        default_return = float('inf') if not self.maximize_metric else -1.0
        # return the best score achieved in this trial for Optuna to process
        return self.trial_best_score if self.trial_best_score is not None else default_return

    def _train_one_epoch(self, epoch: int) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc=f"Training {epoch}th epoch..", leave=False):
            batch = self._to_device(batch)

            self.optimizer.zero_grad()
            outputs = self.model(*batch.values())
            loss = self.criterion(*outputs)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def _to_device(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Recursively move batch data to the Trainer's device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return {f"arg{i}": v.to(self.device) for i, v in enumerate(batch)}
        else:
            raise ValueError("Unsupported batch type for device transfer.")

    def _eval_model(self, evaluator, return_scores: bool = False):
        all_scores = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        with torch.no_grad():
            start_time = time.time()
            for start in tqdm(range(0, self.n_users, self.eval_batch), desc=f"Evaluating..",
                              leave=False):
                end = min(start + self.eval_batch, self.n_users)
                u_idx = torch.arange(start, end, device=self.device) \
                    .unsqueeze(1).expand(-1, self.n_items).reshape(-1)
                i_idx = torch.arange(self.n_items, device=self.device).repeat(end - start)
                scores = self.model.score(u_idx, i_idx)
                all_scores[start:end] = scores.view(end - start, self.n_items).cpu().numpy()
        metrics = evaluator.evaluate(all_scores)
        self.logger.info(f" || Evaluation executed in {(time.time() - start_time):.2f} seconds ||"
                         f"\n{metrics}")
        if return_scores:
            return metrics, all_scores
        return metrics

    def evaluate_on_test(self) -> dict:
        """
        WRAPPER METHOD
        Runs a one-time evaluation on the test set using the current model state.
        This method is designed to be called after a model has been loaded for a
        cold-start evaluation.
        """
        self.logger.info("=" * 20 + " RUNNING EVALUATION-ONLY MODE " + "=" * 20)

        # Ensure the model is in evaluation mode
        self.model.eval()

        # check if save_recs = True (default)
        should_save = self.config.eval.test.save_recs

        if should_save:
            test_metrics, all_scores = self._eval_model(evaluator=self.test_evaluator, return_scores=True)

            # determine max K for saving (largest cutoff defined)
            # defaulting to max cutoff or 50 if missing
            max_k = max(self.config.eval.test.cutoff) if self.config.eval.test.cutoff else 50

            self.logger.info(f"Generating top-{max_k} recommendations for storage...")
            recs_list = self.test_evaluator.get_recommendations(all_scores, k=max_k)

            # define output path (standard name to be picked up by Manager)
            rec_path = os.path.join(self.exp_dir, "recommendations.tsv")
            store_recommendation(recs_list, rec_path)
            self.logger.info(f"Recommendations saved to: {rec_path}")

        else:
            test_metrics = self._eval_model(evaluator=self.test_evaluator, return_scores=False)

        self.logger.info("=" * 20 + " EVALUATION-ONLY FINISHED " + "=" * 20)
        return test_metrics

