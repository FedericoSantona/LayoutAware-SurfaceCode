"""Tests for :mod:`transpile.config` helpers.

Focus areas:
- ``test_basis_set`` asserts the config exposes basis gates as a set for quick membership checks.
- ``test_seed_stream_sequence`` confirms deterministic seed iteration with offsets.
"""

import pytest

from transpile.config import TranspileConfig


@pytest.mark.usefixtures("simple_target")
def test_basis_set(simple_target):
    cfg = TranspileConfig(target=simple_target, basis=("rz", "sx", "x", "cx"))
    assert cfg.basis_set() == {"rz", "sx", "x", "cx"}


@pytest.mark.usefixtures("simple_target")
def test_seed_stream_sequence(simple_target):
    cfg = TranspileConfig(target=simple_target, seeds=3, seed_offset=5)
    assert list(cfg.seed_stream()) == [5, 6, 7]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv", "-rA"]))
