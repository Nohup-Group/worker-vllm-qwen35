"""Sentry and logging setup for the worker runtime."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

logger = logging.getLogger(__name__)

try:
    import sentry_sdk
    from sentry_sdk.integrations.asyncio import enable_asyncio_integration
    from sentry_sdk.integrations.logging import LoggingIntegration
except ImportError:  # pragma: no cover - production image installs sentry-sdk
    sentry_sdk = None
    enable_asyncio_integration = None
    LoggingIntegration = None

_SENTRY_INITIALIZED = False
_ASYNCIO_ENABLED = False

_STATIC_TAG_ENV_VARS = {
    "service": "SERVICE_NAME",
    "model_name": "MODEL_NAME",
    "model_revision": "MODEL_REVISION",
    "quantization": "QUANTIZATION",
    "runpod_endpoint_id": "RUNPOD_ENDPOINT_ID",
    "runpod_pod_id": "RUNPOD_POD_ID",
}


def get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_float_env(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


def configure_logging() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, force=True)
    return level


def sentry_enabled() -> bool:
    return _SENTRY_INITIALIZED and sentry_sdk is not None


def init_sentry() -> bool:
    global _SENTRY_INITIALIZED

    if _SENTRY_INITIALIZED:
        return True

    if not os.getenv("SENTRY_DSN"):
        return False

    if sentry_sdk is None or LoggingIntegration is None:
        logger.warning("Sentry disabled because sentry-sdk is not installed")
        return False

    logging_integration = LoggingIntegration(
        level=logging.INFO,
        event_level=None,
        sentry_logs_level=logging.INFO,
    )

    options: dict[str, Any] = {
        "dsn": os.getenv("SENTRY_DSN"),
        "environment": os.getenv("SENTRY_ENVIRONMENT")
        or os.getenv("ENVIRONMENT")
        or "production",
        "release": os.getenv("SENTRY_RELEASE"),
        "server_name": os.getenv("RUNPOD_POD_ID") or os.getenv("HOSTNAME"),
        "sample_rate": get_float_env("SENTRY_ERROR_SAMPLE_RATE", 1.0),
        "enable_logs": get_bool_env("SENTRY_ENABLE_LOGS", True),
        "send_default_pii": get_bool_env("SENTRY_SEND_DEFAULT_PII", False),
        "include_local_variables": get_bool_env(
            "SENTRY_INCLUDE_LOCAL_VARIABLES", False
        ),
        "max_request_body_size": os.getenv("SENTRY_MAX_REQUEST_BODY_SIZE", "never"),
        "shutdown_timeout": get_float_env("SENTRY_SHUTDOWN_TIMEOUT", 2.0),
        "keep_alive": get_bool_env("SENTRY_KEEP_ALIVE", True),
        "project_root": "/src",
        "integrations": [logging_integration],
    }

    traces_sample_rate = get_float_env("SENTRY_TRACES_SAMPLE_RATE")
    if traces_sample_rate is not None:
        options["traces_sample_rate"] = traces_sample_rate

    profiles_sample_rate = get_float_env("SENTRY_PROFILES_SAMPLE_RATE")
    if profiles_sample_rate is not None:
        options["profiles_sample_rate"] = profiles_sample_rate

    profile_session_sample_rate = get_float_env(
        "SENTRY_PROFILE_SESSION_SAMPLE_RATE"
    )
    if profile_session_sample_rate is not None:
        options["profile_session_sample_rate"] = profile_session_sample_rate
        options["profile_lifecycle"] = os.getenv("SENTRY_PROFILE_LIFECYCLE", "trace")

    sentry_sdk.init(**options)
    _SENTRY_INITIALIZED = True

    with sentry_sdk.configure_scope() as scope:
        for tag_name, env_var in _STATIC_TAG_ENV_VARS.items():
            value = os.getenv(env_var)
            if value:
                scope.set_tag(tag_name, value)
        scope.set_tag("service", os.getenv("SERVICE_NAME", "worker-vllm-qwen35"))

    logger.info("Sentry observability enabled")
    return True


def enable_asyncio_observability() -> bool:
    global _ASYNCIO_ENABLED

    if _ASYNCIO_ENABLED or not sentry_enabled() or enable_asyncio_integration is None:
        return False

    enable_asyncio_integration(
        task_spans=get_bool_env("SENTRY_ASYNCIO_TASK_SPANS", False)
    )
    _ASYNCIO_ENABLED = True
    logger.info("Sentry asyncio integration enabled")
    return True


def start_transaction(name: str, op: str):
    if not sentry_enabled():
        return nullcontext()
    return sentry_sdk.start_transaction(name=name, op=op)


def start_span(op: str, description: str | None = None):
    if not sentry_enabled():
        return nullcontext()
    return sentry_sdk.start_span(op=op, description=description)


@contextmanager
def push_job_scope(job_input: Any, engine_name: str) -> Iterator[None]:
    if not sentry_enabled():
        yield
        return

    with sentry_sdk.push_scope() as scope:
        scope.set_tag("engine", engine_name)
        scope.set_tag("request_id", job_input.request_id)
        scope.set_tag("stream", str(bool(job_input.stream)).lower())
        scope.set_tag("route", job_input.openai_route or "native")
        scope.set_context(
            "job",
            {
                "route": job_input.openai_route or "native",
                "stream": bool(job_input.stream),
                "use_openai_format": bool(job_input.use_openai_format),
                "max_batch_size": job_input.max_batch_size,
                "batch_size_growth_factor": job_input.batch_size_growth_factor,
                "min_batch_size": job_input.min_batch_size,
            },
        )
        yield


def capture_exception(error: Exception) -> None:
    if sentry_enabled():
        sentry_sdk.capture_exception(error)


def flush_observability() -> None:
    if sentry_enabled():
        sentry_sdk.flush(timeout=get_float_env("SENTRY_SHUTDOWN_TIMEOUT", 2.0))
