"""Tests for MODEL_NAME normalization (repo:hash splitting)."""

import os
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from normalize_model import normalize_model_env


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure MODEL_NAME and MODEL_REVISION are clean before each test."""
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_REVISION", raising=False)


def test_no_model_name(monkeypatch):
    """No-op when MODEL_NAME is not set."""
    normalize_model_env()
    assert os.getenv("MODEL_NAME") is None
    assert os.getenv("MODEL_REVISION") is None


def test_plain_model_name(monkeypatch):
    """Plain repo name without hash is left unchanged."""
    monkeypatch.setenv("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B")
    normalize_model_env()
    assert os.environ["MODEL_NAME"] == "Qwen/Qwen3.5-35B-A3B"
    assert os.getenv("MODEL_REVISION") is None


def test_model_name_with_hash(monkeypatch):
    """repo:hash is split into MODEL_NAME + MODEL_REVISION."""
    monkeypatch.setenv("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B:abc1234")
    normalize_model_env()
    assert os.environ["MODEL_NAME"] == "Qwen/Qwen3.5-35B-A3B"
    assert os.environ["MODEL_REVISION"] == "abc1234"


def test_model_name_with_full_hash(monkeypatch):
    """Full 40-char SHA hash is recognized."""
    sha = "a" * 40
    monkeypatch.setenv("MODEL_NAME", f"org/model:{sha}")
    normalize_model_env()
    assert os.environ["MODEL_NAME"] == "org/model"
    assert os.environ["MODEL_REVISION"] == sha


def test_explicit_revision_takes_priority(monkeypatch):
    """When MODEL_REVISION is already set, the hash from MODEL_NAME is ignored."""
    monkeypatch.setenv("MODEL_NAME", "org/model:abc1234")
    monkeypatch.setenv("MODEL_REVISION", "main")
    normalize_model_env()
    assert os.environ["MODEL_NAME"] == "org/model"
    assert os.environ["MODEL_REVISION"] == "main"


def test_short_suffix_not_treated_as_hash(monkeypatch):
    """Suffix shorter than 7 chars is not treated as a commit hash."""
    monkeypatch.setenv("MODEL_NAME", "org/model:v1")
    normalize_model_env()
    assert os.environ["MODEL_NAME"] == "org/model:v1"
    assert os.getenv("MODEL_REVISION") is None


def test_non_hex_suffix_not_treated_as_hash(monkeypatch):
    """Non-hex suffix is not treated as a commit hash."""
    monkeypatch.setenv("MODEL_NAME", "org/model:gptq-int4")
    normalize_model_env()
    assert os.environ["MODEL_NAME"] == "org/model:gptq-int4"
    assert os.getenv("MODEL_REVISION") is None
