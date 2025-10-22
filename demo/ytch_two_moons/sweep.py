"""Optuna hyperparameter sweep for two moons classifier."""

import hydra
import optuna
from dataclasses import replace
from hydra.core.config_store import ConfigStore
from mlflow import set_experiment, start_run
from omegaconf import DictConfig, OmegaConf

from .config import OptunaSweepConfig
from .__main__ import run_training

cs = ConfigStore.instance()
cs.store(name="optuna_sweep", node=OptunaSweepConfig)


@hydra.main(version_base=None, config_name="optuna_sweep")
def main(cfg: DictConfig) -> None:
    """Run Optuna hyperparameter sweep."""
    from typing import cast

    # Convert DictConfig to proper dataclass instances
    sweep_cfg = cast(OptunaSweepConfig, OmegaConf.to_object(cfg))
    base_cfg = sweep_cfg.base

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function - returns final accuracy."""
        # Suggest hyperparameters
        lr = trial.suggest_float(
            "lr", sweep_cfg.optuna.lr_min, sweep_cfg.optuna.lr_max, log=True
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            sweep_cfg.optuna.weight_decay_min,
            sweep_cfg.optuna.weight_decay_max,
            log=True,
        )

        if sweep_cfg.optuna.sweep_smart_output_init:
            smart_output_init = trial.suggest_categorical(
                "smart_output_init", [True, False]
            )
        else:
            smart_output_init = base_cfg.model.smart_output_init

        # Create config with suggested hyperparameters
        trial_cfg = replace(
            base_cfg,
            training=replace(base_cfg.training, base_lr=lr, weight_decay=weight_decay),
            model=replace(base_cfg.model, smart_output_init=smart_output_init),
        )

        # Run training with MLflow logging but no visualization (incompatible with multiprocessing)
        acc = float("nan")
        with start_run(run_name=f"trial_{trial.number}"):
            acc = run_training(trial_cfg, log_mlflow=True, enable_viz=False)
        return acc

    _ = set_experiment("two_moons_hp_sweep")
    study = optuna.create_study(direction="maximize", study_name="two_moons_hp_sweep")
    study.optimize(
        objective,
        n_trials=sweep_cfg.optuna.n_trials,
        n_jobs=sweep_cfg.optuna.n_jobs,
        show_progress_bar=True,
    )

    print("\nBest trial:")
    print(f"  Accuracy: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")


if __name__ == "__main__":
    main()
