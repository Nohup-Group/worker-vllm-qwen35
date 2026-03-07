"""Tests for startup validation logging."""

import os
import sys
import logging
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from validate_model import validate_and_log


def test_validate_logs_model_info(monkeypatch, caplog):
    """Validate that startup logging includes model configuration."""
    monkeypatch.setenv("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
    monkeypatch.setenv("MODEL_REVISION", "main")
    monkeypatch.setenv("QUANTIZATION", "gptq")

    with caplog.at_level(logging.INFO):
        validate_and_log()

    log_text = caplog.text
    assert "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" in log_text
    assert "Worker Startup Validation" in log_text
