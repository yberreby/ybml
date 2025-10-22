from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    noise_std: float = 0.1
    seed: int = 42
    grid_samples: int = 1000


@dataclass
class TrainingConfig:
    batch_size: int = 256
    n_steps: int = 5000
    base_lr: float = 1e-4
    weight_decay: float = 0.0


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    smart_output_init: bool = True  # Data-dependent output init for balanced logits


@dataclass
class LoggingConfig:
    experiment_name: str = "two_moons"
    plot_every_n_steps: int = 10


@dataclass
class VizConfig:
    grid_resolution: int = 200
    alpha_points: float = 0.6
    alpha_surface: float = 0.3
    figsize_width: int = 10
    figsize_height: int = 8
    dpi: int = 100
    plot_padding: float = 0.5


@dataclass
class FormattingConfig:
    loss_precision: int = 4
    acc_precision: int = 4
    lr_precision: int = 6


@dataclass
class OptunaConfig:
    n_trials: int = 50
    n_jobs: int = -1  # -1 means use all CPUs
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    weight_decay_min: float = 1e-6
    weight_decay_max: float = 1e-2
    sweep_smart_output_init: bool = True


@dataclass
class TwoMoonsConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    formatting: FormattingConfig = field(default_factory=FormattingConfig)


@dataclass
class OptunaSweepConfig:
    """Config for Optuna hyperparameter sweep."""

    base: TwoMoonsConfig = field(default_factory=TwoMoonsConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
