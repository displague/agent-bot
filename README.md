# Agent-Bot

An autonomous AI agent system that interacts via text or voice, processes interactions through multi-phase reasoning using a Llama model, and provides a terminal-based user interface (TUI). It continuously operates, simulating an autonomous agent capable of recording audio, transcribing speech, responding to queries, compressing historical data, and scheduling future actions.

## Features

- **Bidirectional Audio Interaction**: Uses NVIDIA PersonaPlex for real-time, full-duplex speech-to-speech conversations with persona control and voice conditioning.
- **Multi-Phase Reasoning**: Breaks down tasks into planning, execution, digestion, validation, and response phases using Llama model inference.
- **Autonomous Operation**: Generates periodic "thoughts" during active hours, compresses interaction logs hourly, and schedules events like reminders or training.
- **Logging and Indexing**: Maintains hard logs (JSONL), compressed summaries, and a keyword-based index for context retrieval.
- **TUI Interface**: Real-time interface using curses, showing status, thoughts, interaction history, and debug info. Supports text input or voice (via PersonaPlex integration).
- **Event Management**: Handles deferred topics, lookups, RAG completions, and training events.
- **State Management**: Tracks unprocessed interactions, ongoing thoughts, sleep status, and next events.

## Architecture

- `main.py`: Entry point; initializes and runs all components asynchronously using curses for TUI.
- `tui_renderer.py`: Manages the curses-based TUI; handles rendering, input (text/voice), scrolling, and screen switching.
- `interaction_processor.py`: Dequeues interactions, processes via `FunctionalAgent`, logs to files, indexes entries, and updates state.
- `functional_agent.py`: Orchestrates multi-phase processing using `LlamaModelManager`.
- `llama_model_manager.py`: Wraps `llama_cpp.Llama` for model inference; manages context, simulates function calls, and handles phases with prompts.
- `thought_generator.py`: Generates autonomous thoughts during active hours using `FunctionalAgent`.
- `event_scheduler.py`: Async queue for events; handles scheduling and execution.
- `event_compressor.py`: Hourly compression of logs using Llama to summarize interactions; saves compressed logs and triggers RAG events.
- `interaction_log_manager.py`: In-memory log with async locking; provides paginated display data.
- `index_manager.py`: Builds/saves a JSON index of interactions by keywords; supports context searches.
- `utils.py`: Utility functions; includes speech-to-speech functions via PersonaPlex.
- `config.py`: Constants for paths (logs, index), sleep times, workers, and model paths.
- `logging_setup.py`: Configures logging to files and stderr redirection.

## Setup

### Prerequisites

- Python 3.8+
- NVIDIA GPU (RTX 5080 recommended, 16GB+ VRAM)
- Opus audio codec: Install `libopus-dev` (Ubuntu/Debian) or equivalent.
- Hugging Face account with access to [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd agent-bot
   ```

2. Install Python dependencies:
   ```bash
   pip install sounddevice torch transformers llama-cpp-python
   ```

3. Install PersonaPlex:
   ```bash
   git clone https://github.com/NVIDIA/personaplex.git
   cd personaplex
   pip install moshi/.
   # For Blackwell GPUs:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   ```

4. Set up environment:
   - Create `.env` file with `HF_TOKEN=<your-huggingface-token>`
   - Ensure Opus binaries (`opusdec`, `opusenc`) are in PATH (e.g., `~/.local/bin`)

5. Download models:
   - Llama model: Place GGUF file at `model.bin` (update `config.py` if different)
   - PersonaPlex voices: Download embeddings (e.g., NATF2.pt) as needed

### Running

```bash
python main.py
```

Use Ctrl+V in TUI for voice input via PersonaPlex.

## Configuration

- `config.py`: Adjust paths, sleep hours (11 PM - 7 AM), worker count (5), model settings.
- Voice/Persona: Configure in `utils.py` or TUI for PersonaPlex prompts.

## Development

- Commit often with detailed messages.
- See [AGENT.md](AGENT.md) for collaboration guidelines.
- Skills in [.github/skills/](.github/skills/) for modular extensions.

## License

MIT License.