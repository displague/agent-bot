# logging_setup.py

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from config import APP_LOG_PATH


def _resolve_log_level():
    raw = os.getenv("AGENTBOT_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, raw, logging.INFO)


def setup_logging():
    """Configures the logging system."""

    os.makedirs("logs", exist_ok=True)
    os.makedirs("compressed_logs", exist_ok=True)
    os.makedirs("index", exist_ok=True)

    logging.basicConfig(
        level=_resolve_log_level(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                APP_LOG_PATH,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            ),
            # logging.StreamHandler(sys.stdout),  # Uncomment if you want logs to also go to console
        ],
    )
    logger = logging.getLogger("autonomous_system")

    # Redirect stderr to a log file to capture llama_cpp output
    stderr_fileno = sys.stderr.fileno()
    stderr_backup = os.dup(stderr_fileno)
    original_stderr = sys.stderr

    def redirect_stderr():
        nonlocal stderr_backup
        f = open("logs/llm_stderr.log", "w")
        os.dup2(f.fileno(), stderr_fileno)
        sys.stderr = f
        return stderr_backup, f

    def restore_stderr(backup, f):
        os.dup2(backup, stderr_fileno)
        os.close(backup)
        f.close()
        sys.stderr = original_stderr

    return redirect_stderr, restore_stderr, logger
