"""Tests for observability helpers."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from observability import get_bool_env, get_float_env


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure test env is isolated."""
    for key in (
        "TEST_BOOL",
        "TEST_FLOAT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_get_bool_env_defaults_when_missing():
    """Missing boolean env returns the provided default."""
    assert get_bool_env("TEST_BOOL", True) is True
    assert get_bool_env("TEST_BOOL", False) is False


def test_get_bool_env_parses_common_truthy_values(monkeypatch):
    """Truthy env values are recognized."""
    monkeypatch.setenv("TEST_BOOL", "yes")
    assert get_bool_env("TEST_BOOL", False) is True


def test_get_bool_env_parses_common_falsy_values(monkeypatch):
    """Falsy env values stay false."""
    monkeypatch.setenv("TEST_BOOL", "0")
    assert get_bool_env("TEST_BOOL", True) is False


def test_get_float_env_defaults_when_missing():
    """Missing float env returns the provided default."""
    assert get_float_env("TEST_FLOAT", 0.5) == 0.5


def test_get_float_env_parses_numbers(monkeypatch):
    """Float env values are parsed."""
    monkeypatch.setenv("TEST_FLOAT", "0.25")
    assert get_float_env("TEST_FLOAT") == 0.25
