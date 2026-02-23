import os
import signal
import subprocess
import sys


def force_exit_now(exit_code: int = 130) -> None:
    """Force-kill the current process tree and exit immediately."""
    pid = os.getpid()
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
    os._exit(exit_code)

