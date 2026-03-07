"""Normalize MODEL_NAME and MODEL_REVISION environment variables.

Handles the "repo:commit_hash" syntax that some users pass to MODEL_NAME,
splitting it into separate MODEL_NAME and MODEL_REVISION env vars so that
vLLM and huggingface-hub receive clean values.

Call normalize_model_env() early in startup, before engine_args are parsed.
"""

import os
import logging


def normalize_model_env():
    """Split MODEL_NAME="org/model:hash" into MODEL_NAME + MODEL_REVISION."""
    model_name = os.getenv("MODEL_NAME", "")
    if not model_name:
        return

    # Check for "org/model:commit_hash" pattern
    # Only split on the LAST colon to avoid breaking Windows paths or other edge cases
    # A valid HF commit hash is 7-40 hex chars
    if ":" in model_name:
        parts = model_name.rsplit(":", 1)
        candidate_hash = parts[1]
        # Validate it looks like a commit hash (hex, 7-40 chars)
        if 7 <= len(candidate_hash) <= 40 and all(
            c in "0123456789abcdefABCDEF" for c in candidate_hash
        ):
            repo = parts[0]
            logging.info(
                f"Normalized MODEL_NAME: '{model_name}' -> repo='{repo}', revision='{candidate_hash}'"
            )
            os.environ["MODEL_NAME"] = repo
            # Only set MODEL_REVISION if not already explicitly set
            if not os.getenv("MODEL_REVISION"):
                os.environ["MODEL_REVISION"] = candidate_hash
                logging.info(f"Set MODEL_REVISION={candidate_hash} from MODEL_NAME")
            else:
                logging.info(
                    f"MODEL_REVISION already set to '{os.getenv('MODEL_REVISION')}', "
                    f"ignoring hash from MODEL_NAME"
                )
