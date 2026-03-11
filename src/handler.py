import sys
import logging
import multiprocessing
import traceback
import runpod
from runpod import RunPodLogger
from observability import (
    capture_exception,
    configure_logging,
    enable_asyncio_observability,
    flush_observability,
    init_sentry,
    push_job_scope,
    start_transaction,
)

log = RunPodLogger()
logger = logging.getLogger(__name__)

vllm_engine = None
openai_engine = None


async def handler(job):
    enable_asyncio_observability()

    from utils import JobInput

    try:
        job_input = JobInput(job["input"])
    except Exception as e:
        log.error(f"Invalid inference input: {e}")
        logger.warning("Invalid inference input: %s", e)
        yield {"error": str(e)}
        return

    engine = openai_engine if job_input.openai_route else vllm_engine
    engine_name = "openai" if job_input.openai_route else "vllm"
    transaction_name = job_input.openai_route or "generate"

    with push_job_scope(job_input, engine_name):
        with start_transaction(
            name=f"{engine_name}:{transaction_name}",
            op="inference.request",
        ) as transaction:
            try:
                results_generator = engine.generate(job_input)
                async for batch in results_generator:
                    yield batch
                if transaction is not None:
                    transaction.set_status("ok")
            except Exception as e:
                error_str = str(e)
                full_traceback = traceback.format_exc()

                log.error(f"Error during inference: {error_str}")
                log.error(f"Full traceback:\n{full_traceback}")
                logger.error("Error during inference: %s", error_str)
                logger.error("Full traceback:\n%s", full_traceback)
                capture_exception(e)

                if transaction is not None:
                    transaction.set_status("internal_error")

                # CUDA errors = worker is broken, exit to let RunPod spin up a healthy one
                if "CUDA" in error_str or "cuda" in error_str:
                    log.error("Terminating worker due to CUDA/GPU error")
                    logger.error("Terminating worker due to CUDA/GPU error")
                    flush_observability()
                    sys.exit(1)

                yield {"error": error_str}


# Only run in main process to prevent re-initialization when vLLM spawns worker subprocesses
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    configure_logging()

    try:
        init_sentry()
        with start_transaction(name="worker.startup", op="startup") as transaction:
            # Normalize MODEL_NAME="repo:hash" -> MODEL_NAME + MODEL_REVISION
            from normalize_model import normalize_model_env
            normalize_model_env()

            # Log startup configuration and validate Qwen3.5 support
            from validate_model import validate_and_log
            validate_and_log()

            from engine import vLLMEngine, OpenAIvLLMEngine

            vllm_engine = vLLMEngine()
            openai_engine = OpenAIvLLMEngine(vllm_engine)
            log.info("vLLM engines initialized successfully")
            logger.info("vLLM engines initialized successfully")
            if transaction is not None:
                transaction.set_status("ok")
    except Exception as e:
        capture_exception(e)
        log.error(f"Worker startup failed: {e}\n{traceback.format_exc()}")
        logger.error("Worker startup failed: %s\n%s", e, traceback.format_exc())
        flush_observability()
        sys.exit(1)

    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda x: vllm_engine.max_concurrency if vllm_engine else 1,
            "return_aggregate_stream": True,
        }
    )
