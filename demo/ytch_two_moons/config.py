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


@dataclass
class ModelConfig:
    hidden_dim: int = 64


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
class TwoMoonsConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    formatting: FormattingConfig = field(default_factory=FormattingConfig)
