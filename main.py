import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Import modules
from logging_setup import setup_logging
from config import DEV_DISABLE_AUTONOMOUS, MAX_WORKERS, MODEL_PATH
from index_manager import IndexManager
from interaction_log_manager import InteractionLogManager
from llama_model_manager import LlamaModelManager
from event_scheduler import EventScheduler
from simple_renderer import SimpleRenderer
from functional_agent import FunctionalAgent
from interaction_processor import InteractionProcessor
from thought_generator import ThoughtGenerator
from event_compressor import EventCompressor


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    _, max_x = stdscr.getmaxyx()
    for idx, line in enumerate(lines):
        stdscr.addstr(idx, 0, line[: max_x - 1])
        stdscr.clrtoeol()
    stdscr.refresh()


async def main(stdscr=None, renderer_name="auto", renderer_reason="", dev_mode=False):
    # Initialize logging
    redirect_stderr, restore_stderr, logger = setup_logging()
    backup, f = redirect_stderr()  # Get backup and file object from redirect_stderr
    logger.info("Starting application")

    # Initialize other components
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    state = {
        "unprocessed_interactions": 0,
        "ongoing_thoughts": 0,
        "next_event": "Not scheduled",
        "is_sleeping": False,
    }
    interaction_queue = asyncio.Queue()

    startup_lines = [
        "Agent-Bot: interactive autonomous assistant with voice and scheduled tasks.",
        f"Renderer: {renderer_name}" + (f" ({renderer_reason})" if renderer_reason else ""),
        "Loading model... please wait.",
    ]
    if dev_mode:
        startup_lines.append("Development mode: autonomous background tasks disabled.")
    _show_startup_status(stdscr, startup_lines)

    llama_manager = LlamaModelManager(model_path=MODEL_PATH)

    _show_startup_status(
        stdscr,
        [
            "Agent-Bot: interactive autonomous assistant with voice and scheduled tasks.",
            f"Renderer: {renderer_name}" + (f" ({renderer_reason})" if renderer_reason else ""),
            "Model loaded. Starting UI...",
        ],
    )

    index_manager = IndexManager()
    interaction_log_manager = InteractionLogManager()
    event_scheduler = EventScheduler(state, interaction_log_manager, index_manager)
    interaction_processor = InteractionProcessor(
        interaction_queue, state, llama_manager, interaction_log_manager, index_manager
    )
    if stdscr is not None:
        from tui_renderer import TUIRenderer  # noqa: PLC0415

        ui_renderer = TUIRenderer(
            stdscr,
            state,
            interaction_queue,
            interaction_log_manager,
            functional_agent=interaction_processor.functional_agent,
        )
    else:
        ui_renderer = SimpleRenderer(
            state,
            interaction_queue,
            interaction_log_manager,
            functional_agent=interaction_processor.functional_agent,
        )
    thought_generator = ThoughtGenerator(
        state, llama_manager, interaction_log_manager, event_scheduler
    )
    event_compressor = EventCompressor(llama_manager, event_scheduler)

    # Start tasks
    tasks = [
        asyncio.create_task(ui_renderer.start()),
        asyncio.create_task(event_scheduler.start()),
        asyncio.create_task(interaction_processor.start()),
    ]
    if not dev_mode:
        tasks.append(asyncio.create_task(thought_generator.start()))
        tasks.append(asyncio.create_task(event_compressor.start()))

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        restore_stderr(backup, f)  # Pass backup and f to restore_stderr
        logger.info("Application stopped")
        # Perform cleanup, cancel tasks, etc.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    redirect_stderr, restore_stderr, logger = setup_logging()
    backup, f = redirect_stderr()  # Store the file object
    try:
        ui_mode = _resolve_ui_mode()
        dev_mode = _env_enabled("AGENTBOT_DEV_MODE", default=DEV_DISABLE_AUTONOMOUS)
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
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        restore_stderr(backup, f)  # Pass the file object to restore_stderr
