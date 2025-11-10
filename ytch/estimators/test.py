import pytest
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from .bptt import get_loss_bptt
from .reinforce import (
    get_loss_reinforce,
    get_loss_rloo_stepwise,
    get_loss_rloo_scalar,
)


@pytest.mark.parametrize(
    "loss_fn",
    [
        get_loss_bptt,
        get_loss_reinforce,
        get_loss_rloo_stepwise,
        get_loss_rloo_scalar,
    ],
)
def test_all_losses_accept_same_shape(loss_fn):
    """Smoke test that all loss functions accept [Groups, N, G] shape."""
    groups, n, g = 2, 4, 3  # 2 groups, 4 samples per group, 3 timesteps

    # Create dummy data with correct shape
    per_glimpse_reward = torch.randn(groups, n, g)
    log_probs = torch.randn(groups, n, g)

    # Should not crash
    loss = loss_fn(per_glimpse_reward, log_probs)

    # Basic checks
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss}"


# Wrappers for loss functions to handle test's 2D inputs
def _reinforce_wrapper(rewards, log_probs):
    """Wrapper that reshapes 2D inputs to 3D for new API."""
    return get_loss_reinforce(rewards.unsqueeze(0), log_probs.unsqueeze(0))


def _rloo_stepwise_wrapper(rewards, log_probs):
    """Wrapper that reshapes 2D inputs to 3D for new API."""
    return get_loss_rloo_stepwise(rewards.unsqueeze(0), log_probs.unsqueeze(0))


def _rloo_scalar_wrapper(rewards, log_probs):
    """Wrapper that reshapes 2D inputs to 3D for new API."""
    return get_loss_rloo_scalar(rewards.unsqueeze(0), log_probs.unsqueeze(0))


# All REINFORCE variants to test
REINFORCE_VARIANTS = [
    _reinforce_wrapper,
    _rloo_stepwise_wrapper,
    _rloo_scalar_wrapper,
]

# Test configuration
BATCH = 10000
DIM = 32
STD = 0.1
SEED = 42
CONVERGENCE_TIMESTEPS = 5
CONVERGENCE_TOL = 0.2
SPARSE_DIFF_TOL = 0.01
BIAS_DIFF_TOL = 0.5
FLOAT_TOL = 1e-6


# Utilities
def compute_gradient(loss, param):
    """Compute gradient and return it, properly handling zero_grad."""
    if param.grad is not None:
        param.grad.zero_()
    loss.backward()
    assert param.grad is not None, "Gradient should exist after backward()"
    grad = param.grad.clone()
    param.grad.zero_()
    return grad


def setup_policy(batch, timesteps, dim=1, seed=None, init_means=None):
    """Setup MVN policy with actions, rewards, log_probs."""
    if seed is not None:
        torch.manual_seed(seed)

    # Policy parameters
    if init_means is None:
        init_means = torch.full((timesteps, dim), 0.0, requires_grad=True)
    else:
        # Ensure init_means has requires_grad=True
        if not init_means.requires_grad:
            init_means = init_means.detach().requires_grad_(True)
    policy_mean = nn.Parameter(init_means)
    cov = torch.eye(dim) * (STD**2)

    # Expand mean to batch x timesteps x dim
    mean_expanded = policy_mean.unsqueeze(0).expand(batch, -1, -1)

    # Sample actions from MVN (one distribution per timestep)
    actions_list = []
    log_probs_list = []
    for t in range(timesteps):
        dist = MultivariateNormal(mean_expanded[:, t], cov)
        action = dist.rsample()
        actions_list.append(action)
        log_probs_list.append(dist.log_prob(action.detach()))

    actions = torch.stack(actions_list, dim=1)  # batch x timesteps x dim
    log_probs = torch.stack(log_probs_list, dim=1)  # batch x timesteps

    # Compute rewards (negative squared distance to target=1)
    target = 1.0
    rewards = -(actions - target).pow(2).mean(dim=-1)  # batch x timesteps

    return policy_mean, actions, rewards, log_probs


def test_bptt_gradients_flow():
    """BPTT: gradients flow through rewards."""
    batch, timesteps = 4, 3

    # Test 1: mean below target
    policy_mean, _, rewards, _ = setup_policy(
        batch, timesteps, seed=SEED, init_means=torch.full((timesteps, 1), 0.0)
    )
    loss = get_loss_bptt(rewards.unsqueeze(0))  # Add Groups dimension
    grad = compute_gradient(loss, policy_mean)
    assert (grad < 0).all(), f"BPTT should push mean up when below target, got {grad}"

    # Test 2: mean above target
    policy_mean, _, rewards, _ = setup_policy(
        batch, timesteps, seed=SEED, init_means=torch.full((timesteps, 1), 2.0)
    )
    loss = get_loss_bptt(rewards.unsqueeze(0))  # Add Groups dimension
    grad = compute_gradient(loss, policy_mean)
    assert (grad > 0).all(), f"BPTT should push mean down when above target, got {grad}"


def test_sparse_reward_gradient():
    """Sparse terminal reward affects all timestep gradients in vanilla REINFORCE."""
    torch.manual_seed(SEED)

    # Different log_probs for each timestep (3 timesteps)
    param = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    # Episode with only terminal reward
    rewards_sparse = torch.tensor([[0.0, 0.0, 1.0]])  # Only terminal
    log_probs = param.unsqueeze(0) - torch.tensor([[1.0, 2.0, 3.0]])

    loss = _reinforce_wrapper(rewards_sparse, log_probs)
    loss.backward()
    assert param.grad is not None
    grad_sparse = param.grad.clone()
    print(f"Sparse rewards gradient: {grad_sparse}")
    param.grad = None

    # Dense rewards (same total)
    rewards_dense = torch.tensor([[0.2, 0.3, 0.5]])
    loss = _reinforce_wrapper(rewards_dense, log_probs)
    loss.backward()
    assert param.grad is not None
    grad_dense = param.grad.clone()
    print(f"Dense rewards gradient: {grad_dense}")

    # With sparse rewards, all actions get credit for terminal reward
    # With dense rewards, credit assignment is different
    diff = (grad_sparse - grad_dense).norm()
    print(f"Gradient difference norm: {diff:.4f}")
    assert diff > SPARSE_DIFF_TOL, f"{diff=:.4f} <= {SPARSE_DIFF_TOL=}"


def test_stepwise_rloo_convergence():
    """Stepwise RLOO converges to BPTT at large batch."""
    timesteps = CONVERGENCE_TIMESTEPS

    # Generate initial means once
    torch.manual_seed(SEED - 1)  # Different seed for init means
    init_means_base = torch.randn(timesteps, DIM)

    # BPTT setup with cloned init_means
    init_means_bptt = init_means_base.clone()
    policy_mean_bptt, actions_bptt, rewards_bptt, _ = setup_policy(
        BATCH, timesteps, dim=DIM, seed=SEED, init_means=init_means_bptt
    )

    print(
        f"BPTT - Mean reward: {rewards_bptt.mean():.4f}, std: {rewards_bptt.std():.4f}"
    )

    # Compute BPTT gradient
    loss_bptt = get_loss_bptt(rewards_bptt.unsqueeze(0))  # Add Groups dimension
    grad_bptt = compute_gradient(loss_bptt, policy_mean_bptt)
    print(f"BPTT grad norm: {grad_bptt.norm():.4f}")

    # REINFORCE setup with cloned init_means (same seed for same samples)
    init_means_reinforce = init_means_base.clone()
    policy_mean_reinforce, actions_reinforce, rewards_reinforce, log_probs_reinforce = (
        setup_policy(
            BATCH, timesteps, dim=DIM, seed=SEED, init_means=init_means_reinforce
        )
    )

    print(
        f"Stepwise RLOO - Mean reward: {rewards_reinforce.mean():.4f}, std: {rewards_reinforce.std():.4f}"
    )

    # Compute REINFORCE gradient with stepwise RLOO baseline
    loss_reinforce = _rloo_stepwise_wrapper(rewards_reinforce, log_probs_reinforce)
    grad_reinforce = compute_gradient(loss_reinforce, policy_mean_reinforce)
    print(f"Stepwise RLOO grad norm: {grad_reinforce.norm():.4f}")

    # Verify same samples were used
    assert torch.allclose(actions_bptt, actions_reinforce, atol=FLOAT_TOL), (
        "Actions should be identical"
    )
    assert torch.allclose(rewards_bptt, rewards_reinforce, atol=FLOAT_TOL), (
        "Rewards should be identical"
    )

    # Compare gradients
    relative_diff = (grad_bptt - grad_reinforce).norm() / grad_bptt.norm()
    print(f"Relative diff: {relative_diff:.4f}")
    assert relative_diff < CONVERGENCE_TOL, (
        f"{relative_diff=:.4f} >= {CONVERGENCE_TOL=}"
    )


def test_incorrect_detachment():
    """Not detaching actions in REINFORCE causes bias."""
    timesteps = 1
    torch.manual_seed(SEED)

    # Setup policy
    policy_mean = nn.Parameter(torch.zeros(timesteps, DIM))
    cov = torch.eye(DIM) * (STD**2)
    mean_expanded = policy_mean.expand(BATCH, -1, -1)

    # Sample actions
    dist = MultivariateNormal(mean_expanded[:, 0], cov)
    actions = dist.rsample()
    rewards = -(actions - 1.0).pow(2).sum(dim=-1, keepdim=True)  # batch x 1

    # Correct REINFORCE (detached)
    log_probs_correct = dist.log_prob(actions.detach()).unsqueeze(1)
    loss_correct = _reinforce_wrapper(rewards, log_probs_correct)
    grad_correct = compute_gradient(loss_correct, policy_mean)
    print(f"Correct detachment grad norm: {grad_correct.norm():.4f}")

    # Incorrect (no detach - creates pathwise leak)
    log_probs_wrong = dist.log_prob(actions).unsqueeze(1)  # Bug: no detach
    loss_wrong = _reinforce_wrapper(rewards, log_probs_wrong)
    grad_wrong = compute_gradient(loss_wrong, policy_mean)
    print(f"Incorrect (no detach) grad norm: {grad_wrong.norm():.4f}")

    # Should have very different gradients
    relative_diff = (grad_correct - grad_wrong).norm() / grad_correct.norm()
    print(f"Relative difference: {relative_diff:.4f}")
    assert relative_diff > BIAS_DIFF_TOL, f"{relative_diff=:.4f} <= {BIAS_DIFF_TOL=}"


def test_bptt_single_timestep():
    """BPTT works with single timestep episodes."""
    batch, timesteps = 10, 1
    policy_mean, _, rewards, _ = setup_policy(
        batch, timesteps, seed=SEED, init_means=torch.full((timesteps, 1), 0.0)
    )

    loss = get_loss_bptt(rewards.unsqueeze(0))  # Add Groups dimension
    grad = compute_gradient(loss, policy_mean)

    # With mean=0 < target=1, should push up (negative gradient)
    assert (grad < 0).all(), f"Single timestep BPTT should push mean up, got {grad}"


def test_reinforce_single_timestep():
    """Vanilla REINFORCE works with single timestep episodes."""
    batch, timesteps = 10, 1
    policy_mean, _, rewards, log_probs = setup_policy(
        batch, timesteps, seed=SEED, init_means=torch.full((timesteps, 1), 0.0)
    )

    loss = _reinforce_wrapper(rewards, log_probs)
    grad = compute_gradient(loss, policy_mean)

    # With mean=0 < target=1, should push up (positive gradient for REINFORCE)
    assert (grad > 0).all(), (
        f"Single timestep REINFORCE should push mean up, got {grad}"
    )


@pytest.mark.parametrize("loss_fn", REINFORCE_VARIANTS)
def test_batch_size_one(loss_fn):
    """All REINFORCE variants should handle batch_size=1 gracefully."""
    batch, timesteps = 1, 3
    policy_mean, _, rewards, log_probs = setup_policy(
        batch, timesteps, seed=SEED, init_means=torch.full((timesteps, 1), 0.5)
    )

    # Should not crash and should produce valid gradient
    loss = loss_fn(rewards, log_probs)
    grad = compute_gradient(loss, policy_mean)

    assert grad is not None, f"{loss_fn.__name__} failed with batch_size=1"
    assert not torch.isnan(grad).any(), (
        f"{loss_fn.__name__} produced NaN with batch_size=1"
    )


def test_grouped_shape_equivalence():
    """Grouped shape with Groups=1 should give same result as flat batch."""
    batch, timesteps = 100, 3

    # Setup identical policies
    torch.manual_seed(SEED)
    init_means = torch.randn(timesteps, 1)
    policy_mean, _, rewards, log_probs = setup_policy(
        batch, timesteps, seed=SEED, init_means=init_means.clone()
    )

    # Test each loss function
    # Vanilla REINFORCE
    loss_flat = _reinforce_wrapper(rewards, log_probs)
    loss_grouped = get_loss_reinforce(rewards.unsqueeze(0), log_probs.unsqueeze(0))
    assert torch.allclose(loss_flat, loss_grouped, atol=FLOAT_TOL), (
        f"REINFORCE: flat={loss_flat:.6f}, grouped={loss_grouped:.6f}"
    )

    # Stepwise RLOO
    loss_flat = _rloo_stepwise_wrapper(rewards, log_probs)
    loss_grouped = get_loss_rloo_stepwise(rewards.unsqueeze(0), log_probs.unsqueeze(0))
    assert torch.allclose(loss_flat, loss_grouped, atol=FLOAT_TOL), (
        f"Stepwise RLOO: flat={loss_flat:.6f}, grouped={loss_grouped:.6f}"
    )

    # Scalar RLOO
    loss_flat = _rloo_scalar_wrapper(rewards, log_probs)
    loss_grouped = get_loss_rloo_scalar(rewards.unsqueeze(0), log_probs.unsqueeze(0))
    assert torch.allclose(loss_flat, loss_grouped, atol=FLOAT_TOL), (
        f"Scalar RLOO: flat={loss_flat:.6f}, grouped={loss_grouped:.6f}"
    )
