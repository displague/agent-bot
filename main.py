import os
import sys
import argparse
import shutil

# Disable brittle optimizations early by default
os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("NO_CUDA_GRAPH", "1")

import asyncio
import signal
from pathlib import Path

# Ensure PersonaPlex/Moshi is in path
project_root = Path(__file__).parent.absolute()
personaplex_moshi = project_root / "personaplex" / "moshi"
if personaplex_moshi.exists() and str(personaplex_moshi) not in sys.path:
    sys.path.insert(0, str(personaplex_moshi))

# Import modules
from logging_setup import setup_logging
from config import (
    DEV_DISABLE_AUTONOMOUS,
    INTERACTION_LOG_PATH,
    INTERACTION_LOG_WARN_BYTES,
    MODEL_PATH,
    SHUTDOWN_GRACE_SECONDS,
)
from index_manager import IndexManager
from interaction_log_manager import InteractionLogManager
from llama_model_manager import LlamaModelManager
from event_scheduler import EventScheduler
from simple_renderer import SimpleRenderer
from interaction_processor import InteractionProcessor
from thought_generator import ThoughtGenerator
from event_compressor import EventCompressor
from runtime_manager import RuntimeManager
from voice_loop import VoiceLoop
from utils import PersonaPlexManager, AudioMultiplexer, RollingAudioBuffer
from process_utils import force_exit_now

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional at runtime
    load_dotenv = None


def _reset_logs():
    """Clear all log directories."""
    log_dirs = ["logs", "compressed_logs", "hf_offload", "offload_weights"]
    for d in log_dirs:
        if os.path.exists(d):
            try:
                # Keep the directory but empty it
                for item in os.listdir(d):
                    item_path = os.path.join(d, item)
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                print(f"Cleared log directory: {d}")
            except Exception as e:
                print(f"Error clearing {d}: {e}")


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_environment():
    if load_dotenv is None:
        return
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


_interrupt_count = 0


def _install_sigint_handler():
    if os.name not in {"nt", "posix"}:
        return

    def _handle_sigint(_signum, _frame):
        global _interrupt_count
        _interrupt_count += 1
        if _interrupt_count >= 2:
            print("Force quitting immediately (second Ctrl+C)...", flush=True)
            force_exit_now(130)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)


def _ensure_utf8_mode_on_windows():
    """Relaunch under UTF-8 mode to avoid cp1252 decode issues on Windows."""
    if os.name != "nt":
        return
    if sys.flags.utf8_mode:
        return
    if os.getenv("AGENTBOT_UTF8_REEXEC") == "1":
        return
    env = os.environ.copy()
    env["AGENTBOT_UTF8_REEXEC"] = "1"
    args = [sys.executable, "-X", "utf8", *sys.argv]
    os.execvpe(sys.executable, args, env)


def _resolve_ui_mode() -> str:
    mode = os.getenv("AGENTBOT_UI_MODE", "auto").strip().lower()
    if mode not in {"auto", "simple", "curses"}:
        return "auto"
    return mode


def _check_curses_available():
    try:
        import curses  # noqa: PLC0415
    except Exception as exc:
        return False, None, f"import failed: {exc}"
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False, curses, "stdin/stdout is not a TTY"
    return True, curses, ""


def _show_startup_status(stdscr, lines):
    if stdscr is None:
        for line in lines:
            print(line, flush=True)
        return
    stdscr.clear()
    max_y, max_x = stdscr.getmaxyx()
    for idx, line in enumerate(lines):
        if idx >= max_y - 1:
            break
        stdscr.addstr(idx, 0, line[: max_x - 1])
        stdscr.clrtoeol()
    stdscr.refresh()


def _interaction_log_health_message():
    try:
        if os.path.exists(INTERACTION_LOG_PATH):
            size = os.path.getsize(INTERACTION_LOG_PATH)
            if size >= INTERACTION_LOG_WARN_BYTES:
                size_mb = size / (1024 * 1024)
                return (
                    f"Warning: interaction log is large ({size_mb:.1f} MB): "
                    f"{INTERACTION_LOG_PATH}"
                )
    except Exception:
        return None
    return None


async def main(stdscr=None, renderer_name="auto", renderer_reason="", dev_mode=False):
    # Initialize logging
    redirect_stderr, restore_stderr, logger = setup_logging()
    backup, f = redirect_stderr()  # Get backup and file object from redirect_stderr
    logger.info("Starting application")

    # Initialize other components
    runtime_manager = RuntimeManager(llm_workers=1, io_workers=2)
    state = {
        "unprocessed_interactions": 0,
        "ongoing_thoughts": 0,
        "next_event": "Not scheduled",
        "is_sleeping": False,
        "is_processing": False,
    }
    interaction_queue = asyncio.Queue()

    startup_lines = [
        "Agent-Bot: interactive autonomous assistant with voice and scheduled tasks.",
        f"Renderer: {renderer_name}" + (f" ({renderer_reason})" if renderer_reason else ""),
        "Initializing systems...",
    ]
    if dev_mode:
        startup_lines.append("Development mode: autonomous background tasks disabled.")
    log_warning = _interaction_log_health_message()
    if log_warning:
        startup_lines.append(log_warning)
    _show_startup_status(stdscr, startup_lines)

    def _status_callback(message: str):
        # If this message is a completion (contains 'loaded' or 'Model ready'), 
        # try to replace the matching 'Loading' line if it's the last one.
        if ("loaded" in message or "ready" in message) and startup_lines:
            last = startup_lines[-1]
            # Simple heuristic: if the last line started the load for this component, replace it
            if "Loading" in last or "starting" in last:
                # E.g. 'Stage: Loading tokenizer' vs 'Stage: Tokenizer loaded in 2.6s'
                startup_lines[-1] = f"Stage: {message}"
            else:
                startup_lines.append(f"Stage: {message}")
        else:
            startup_lines.append(f"Stage: {message}")
        _show_startup_status(stdscr, startup_lines)

    index_manager = IndexManager()
    interaction_log_manager = InteractionLogManager()
    
    audio_multiplexer = AudioMultiplexer()
    audio_multiplexer.start()
    
    rolling_buffer = RollingAudioBuffer(audio_multiplexer)
    await rolling_buffer.start()

    personaplex_manager = PersonaPlexManager(status_callback=_status_callback)
    voice_loop = VoiceLoop(state, interaction_log_manager, personaplex_manager=personaplex_manager, audio_multiplexer=audio_multiplexer)

    llama_manager = None
    if not skip_reasoning:
        llama_manager = LlamaModelManager(
            model_path=MODEL_PATH,
            llm_executor=runtime_manager.llm_executor,
            status_callback=_status_callback,
            voice_loop=rolling_buffer,
        )
    else:
        _show_startup_status(stdscr, startup_lines + ["Stage: Skipping LLM load (deep reasoning disabled)"])

    # Explicitly load PersonaPlex models during startup to show progress
    await asyncio.to_thread(personaplex_manager.load)

    event_scheduler = EventScheduler(
        state, interaction_log_manager, index_manager, runtime_manager=runtime_manager
    )
    interaction_processor = InteractionProcessor(
        interaction_queue, state, llama_manager, interaction_log_manager, index_manager,
        voice_loop=voice_loop, personaplex_manager=personaplex_manager
    )
    if stdscr is not None:
        from tui_renderer import TUIRenderer  # noqa: PLC0415

        ui_renderer = TUIRenderer(
            stdscr,
            state,
            interaction_queue,
            interaction_log_manager,
            functional_agent=interaction_processor.functional_agent,
            voice_loop=voice_loop,
        )
    else:
        ui_renderer = SimpleRenderer(
            state,
            interaction_queue,
            interaction_log_manager,
            functional_agent=interaction_processor.functional_agent,
            voice_loop=voice_loop,
        )
    thought_generator = ThoughtGenerator(
        state, llama_manager, interaction_log_manager, event_scheduler,
        interaction_processor=interaction_processor
    )
    event_compressor = EventCompressor(
        llama_manager,
        event_scheduler,
        io_executor=runtime_manager.io_executor,
        state=state,
    )

    ui_task = runtime_manager.register_task(asyncio.create_task(ui_renderer.start()))
    background_tasks = [
        runtime_manager.register_task(asyncio.create_task(event_scheduler.start())),
        runtime_manager.register_task(asyncio.create_task(interaction_processor.start())),
    ]
    if not dev_mode:
        background_tasks.append(
            runtime_manager.register_task(asyncio.create_task(thought_generator.start()))
        )
        background_tasks.append(
            runtime_manager.register_task(asyncio.create_task(event_compressor.start()))
        )

    try:
        await asyncio.wait_for(ui_task, timeout=None)
    except (Exception, asyncio.CancelledError) as e:
        if not isinstance(e, asyncio.CancelledError):
            logger.exception(f"An error occurred: {e}")
            _show_startup_status(
                stdscr,
                [
                    "Fatal runtime error in UI loop.",
                    str(e),
                    "Check logs/app.log and logs/llm_stderr.log for details.",
                ],
            )
            print(
                f"Fatal runtime error: {e}. Check logs/app.log and logs/llm_stderr.log.",
                flush=True,
            )
    finally:
        logger.info("Starting shutdown sequence")
        
        # Start hard-kill watchdog: if we don't finish in 10s, force exit.
        def _watchdog():
            time.sleep(10)
            print("\nShutdown watchdog triggered: force killing process tree...", flush=True)
            force_exit_now(130)
        
        import threading
        import time
        watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        watchdog_thread.start()

        ui_task.cancel()
        interaction_processor.request_stop()
        thought_generator.request_stop()
        
        # Stop auditory backbone
        if 'rolling_buffer' in locals():
            await rolling_buffer.stop()
        if 'audio_multiplexer' in locals():
            audio_multiplexer.stop()
        
        # Shut down scheduler with a timeout
        try:
            await asyncio.wait_for(event_scheduler.shutdown(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("Event scheduler shutdown timed out")

        # Cancel all background tasks
        for task in background_tasks:
            task.cancel()
        
        if background_tasks:
            done, pending = await asyncio.wait(
                background_tasks, timeout=SHUTDOWN_GRACE_SECONDS
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*background_tasks, return_exceptions=True)

        await runtime_manager.cancel_all_tasks(timeout_seconds=SHUTDOWN_GRACE_SECONDS)
        runtime_manager.shutdown_executors()
        if personaplex_manager:
            personaplex_manager.shutdown()
        restore_stderr(backup, f)
        logger.info("Application stopped")


if __name__ == "__main__":
    _ensure_utf8_mode_on_windows()
    _install_sigint_handler()
    _load_environment()

    parser = argparse.ArgumentParser(description="Agent-Bot")
    parser.add_argument("--reset-logs", action="store_true", help="Clear all logs on startup")
    parser.add_argument("--dev", action="store_true", help="Enable dev mode (disables autonomous tasks)")
    parser.add_argument("--skip-deep-reasoning", action="store_true", help="Bypass the deep reasoning LLM phase")
    args_parsed = parser.parse_args()

    if args_parsed.reset_logs:
        _reset_logs()

    redirect_stderr, restore_stderr, logger = setup_logging()
    backup, f = redirect_stderr()  # Store the file object
    try:
        ui_mode = _resolve_ui_mode()
        dev_mode = args_parsed.dev or _env_enabled("AGENTBOT_DEV_MODE", default=DEV_DISABLE_AUTONOMOUS)
        
        # Merge CLI and config for deep reasoning toggle
        import config
        if args_parsed.skip_deep_reasoning:
            config.SKIP_DEEP_REASONING = True
        skip_reasoning = getattr(config, "SKIP_DEEP_REASONING", False)
        
        curses_ok, curses_mod, reason = _check_curses_available()

        if ui_mode == "simple":
            asyncio.run(
                main(
                    None,
                    renderer_name="simple",
                    renderer_reason="forced by AGENTBOT_UI_MODE=simple",
                    dev_mode=dev_mode,
                )
            )
        elif ui_mode == "curses":
            if not curses_ok:
                raise RuntimeError(
                    f"Curses mode requested but unavailable: {reason}. "
                    "Use AGENTBOT_UI_MODE=simple."
                )
            curses_mod.wrapper(
                lambda stdscr: asyncio.run(
                    main(
                        stdscr,
                        renderer_name="curses",
                        renderer_reason="forced by AGENTBOT_UI_MODE=curses",
                        dev_mode=dev_mode,
                    )
                )
            )
        else:
            if curses_ok:
                curses_mod.wrapper(
                    lambda stdscr: asyncio.run(
                        main(
                            stdscr,
                            renderer_name="curses",
                            renderer_reason="auto",
                            dev_mode=dev_mode,
                        )
                    )
                )
            else:
                asyncio.run(
                    main(
                        None,
                        renderer_name="simple",
                        renderer_reason=f"auto fallback: {reason}",
                        dev_mode=dev_mode,
                    )
                )
    except KeyboardInterrupt:
        print(
            "Interrupted by Ctrl+C. Shutting down... (Press Ctrl+C again to force quit)",
            flush=True,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        restore_stderr(backup, f)  # Pass the file object to restore_stderr
