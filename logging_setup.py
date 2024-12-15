# logging_setup.py

import logging
import os
import sys
from config import HARD_LOG_PATH


def setup_logging():
    """Configures the logging system."""

    os.makedirs("logs", exist_ok=True)
    os.makedirs("compressed_logs", exist_ok=True)
    os.makedirs("index", exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(HARD_LOG_PATH),
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
