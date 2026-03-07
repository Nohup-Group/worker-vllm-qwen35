"""Startup validation for the vLLM worker.

Logs model name, revision, and supported architectures so operators can
quickly verify that the correct model will load before vLLM engine init.
"""

import os
import logging


def validate_and_log():
    """Log model configuration and check architecture support."""
    model_name = os.getenv("MODEL_NAME", "(not set)")
    model_revision = os.getenv("MODEL_REVISION", "(not set)")
    quantization = os.getenv("QUANTIZATION", "(not set)")
    gpu_memory = os.getenv("GPU_MEMORY_UTILIZATION", "0.90")

    logging.info("=" * 60)
    logging.info("Worker Startup Validation")
    logging.info("=" * 60)
    logging.info(f"  MODEL_NAME:             {model_name}")
    logging.info(f"  MODEL_REVISION:         {model_revision}")
    logging.info(f"  QUANTIZATION:           {quantization}")
    logging.info(f"  GPU_MEMORY_UTILIZATION: {gpu_memory}")

    # Check vLLM version and Qwen3.5 support
    try:
        import vllm

        logging.info(f"  vLLM version:           {vllm.__version__}")
    except ImportError:
        logging.warning("  vLLM: NOT INSTALLED")
        return

    # Check if Qwen3.5 MoE architecture is registered
    try:
        from vllm.model_executor.models import ModelRegistry

        supported = ModelRegistry.is_text_generation_model(
            "Qwen3_5MoeForConditionalGeneration"
        )
        if supported:
            logging.info(
                "  Qwen3_5MoeForConditionalGeneration: SUPPORTED"
            )
        else:
            logging.warning(
                "  Qwen3_5MoeForConditionalGeneration: NOT SUPPORTED — "
                "vLLM version may be too old"
            )
    except Exception as e:
        # ModelRegistry API may differ between versions, just log
        logging.info(f"  Architecture check skipped: {e}")

    # Check transformers version
    try:
        import transformers

        logging.info(f"  transformers version:   {transformers.__version__}")
    except ImportError:
        logging.warning("  transformers: NOT INSTALLED")

    logging.info("=" * 60)
