# ybml - @yberreby's ML Library

Utilities for my own usage; may be broken at any time.

Targeting **PyTorch** (`ytch`), **JAX** (`yjax`), and **MLX** (`ymlx`).
Core, framework-independent code is in `ymc`.

## Testing
- `uv run just`: linting, type checking, formatting, testing, coverage.

## End-to-end demo(s)

- `uv run just demo ytch_two_moons`: utterly toy task; uses MLFlow.
- `uv run just demo single_img_deep_coord_mlp`: NeRF-style coordinate MLP mapping `(x,y)` to `(r,g,b)` to learn an image with a few twists.

See `./justfile` for up-to-date commands.


## Module tree

Run `uv run just inspect` to print the module tree, e.g. (this sample output may become out of date):

```
ymc
├── constants
└── lr
    ├── get_linear_scaled_lr()
    └── get_warmup_steps_for_adam_beta2()
ytch
├── attention
│   └── mh
│       ├── from_multihead()
│       └── to_multihead()
├── constants
├── correctness
│   ├── gradients
│   │   └── assert_gradients_flow()
│   └── shapes
│       └── assert_shape()
├── device
│   └── get_sensible_device()
├── lr
│   └── warmup
│       └── get_linear_warmup_scheduler()
├── magic
│   └── dyniso
│       └── ortho_block_init_()
├── metrics
│   ├── compute_grad_norm()
│   └── print_grad_norms()
├── model
│   └── count_parameters()
├── nn
│   ├── elementwise_affine
│   │   └── ElementwiseAffine
│   ├── rff
│   │   └── RandomFourierFeaturesND
│   └── skip
│       └── Skip
└── train
    ├── Trainer
    └── train()
```
