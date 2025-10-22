# YBML Project Guidelines

Machine learning library with PyTorch (ytch), JAX (yjax), and MLX (ymlx) implementations.

## Core Principles

- **NO BRANCHING** - Linear code. Let it crash with full context if things don't make sense, don't hide errors.
- **DRY obsessively** - Extract duplication / high-level operations immediately
- **Strong typing** - dataclass/NamedTuple or the likes for structured data
- **Self-documenting code** - Clear names over comments

## Project-Specific Patterns

### Type Safety

**DON'T** use union types with bool - footguns:
```python
# BAD - unclear semantics
def fn(grad_clip: bool | ZClip = True): ...
```

**DO** use clear, single-purpose types:
```python
# GOOD - explicit optional
def fn(grad_clip: ZClip | None = None): ...

# Or if default needed, construct explicitly
def fn(grad_clip: ZClip | None = None):
    if grad_clip is None:
        grad_clip = ZClip()  # Clear default construction
```


## Module Structure

Inspect it with `uv run just inspect`.

### Tests

- Minimal smoke tests by default
- Basic: shape correctness, value correctness, gradient flow...
- Gradients: use `ytch.correctness.assert_gradients_flow`
- Keep them fast

Example:
```python
def test_skip_shape_and_value():
    block = nn.Linear(DIM, DIM)
    layer = Skip(block)
    x = torch.randn(BATCH_SIZE, DIM)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x + block(x))

def test_skip_gradients():
    assert_gradients_flow(Skip(nn.Linear(DIM, DIM)), torch.randn(BATCH_SIZE, DIM, requires_grad=True))
```


## Remember

This is a library of **clean, reusable primitives**. Each module should:
- Do one thing well
- Have zero branching (or minimal, justified)
- Be obviously correct
- Compose cleanly with other modules

If you're adding complexity, step back and ask: "Can I split this into two simple things instead?"
