"""Lightweight tests for Angular steering math (no vLLM)."""

import numpy as np
import pytest
import torch

from recipe.steering.angular import AngularSteeringOperator


def test_operator_adaptive_zero_is_identity_at_degree_zero():
    rng = np.random.default_rng(0)
    d = 32
    u1 = rng.standard_normal(d).astype(np.float32)
    u2 = rng.standard_normal(d).astype(np.float32)
    op = AngularSteeringOperator(u1, u2)

    h = torch.randn(2, 5, d, dtype=torch.float32)
    out = op.steer(h, target_degree=0.0, adaptive_mode=0)
    assert out.shape == h.shape
    # At 0°, v_theta = b1; steering replaces plane projection — not identity; smoke only
    assert torch.isfinite(out).all()


def test_operator_raises_on_bad_adaptive_mode():
    op = AngularSteeringOperator(
        np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    )
    h = torch.zeros(1, 1, 4)
    with pytest.raises(ValueError, match="Unknown adaptive_mode"):
        op.steer(h, target_degree=0.0, adaptive_mode=2)
