import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import modules
from logging_setup import setup_logging
from config import MAX_WORKERS, MODEL_PATH
from index_manager import IndexManager
from interaction_log_manager import InteractionLogManager
from llama_model_manager import LlamaModelManager
from event_scheduler import EventScheduler
from simple_renderer import SimpleRenderer
from functional_agent import FunctionalAgent
from interaction_processor import InteractionProcessor
from thought_generator import ThoughtGenerator
from event_compressor import EventCompressor


async def main(stdscr=None):
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

    if stdscr is not None:
        stdscr.clear()
        stdscr.addstr(0, 0, "Loading model... please wait.")
        stdscr.refresh()
    else:
        print("Loading model... please wait.", flush=True)

    llama_manager = LlamaModelManager(model_path=MODEL_PATH)

    if stdscr is not None:
        stdscr.clear()
        stdscr.addstr(0, 0, "Model loaded. Starting UI...")
        stdscr.refresh()
    else:
        print("Model loaded. Starting UI...", flush=True)

    index_manager = IndexManager()
    interaction_log_manager = InteractionLogManager()
    event_scheduler = EventScheduler(state, interaction_log_manager, index_manager)
    if stdscr is not None:
        from tui_renderer import TUIRenderer  # noqa: PLC0415

        ui_renderer = TUIRenderer(stdscr, state, interaction_queue, interaction_log_manager)
    else:
        ui_renderer = SimpleRenderer(state, interaction_queue, interaction_log_manager)
    interaction_processor = InteractionProcessor(
        interaction_queue, state, llama_manager, interaction_log_manager, index_manager
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
        asyncio.create_task(thought_generator.start()),
        asyncio.create_task(event_compressor.start()),
    ]

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
        try:
            import curses  # noqa: PLC0415
        except Exception:
            asyncio.run(main(None))
        else:
            curses.wrapper(lambda stdscr: asyncio.run(main(stdscr)))
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        restore_stderr(backup, f)  # Pass the file object to restore_stderr
