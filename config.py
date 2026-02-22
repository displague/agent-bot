# config.py

APP_LOG_PATH = "logs/app.log"
INTERACTION_LOG_PATH = "logs/hard_log.jsonl"
HARD_LOG_PATH = INTERACTION_LOG_PATH  # Backward-compat alias
PERSONAPLEX_SERVER_LOG_PATH = "logs/personaplex_server.log"
SMOKE_LOG_PATH = "logs/smoke_test.jsonl"
COMPRESSED_LOG_PATH = "compressed_logs/compressed_log.jsonl"
INDEX_PATH = "index/context_index.json"
DAILY_SLEEP_START = 23
DAILY_SLEEP_END = 7
MAX_WORKERS = 5
MODEL_PATH = "model.bin"  # Legacy fallback path for local GGUF.

# Model runtime settings
MODEL_DEFAULT_ALIAS = "gemma-it"
MODEL_LIST = {
    "gemma-it": {
        "backend": "transformers",
        "repo_id": "google/gemma-3n-E2B-it",
    },
    # Tiny model for lightweight/CPU-only testing.
    "tiny": {
        "backend": "llama_cpp",
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    },
    # Optional transformer backend model.
    "gpt-oss": {
        "backend": "transformers",
        "repo_id": "openai/gpt-oss-20b",
    },
}
MODEL_TRANSFORMERS_MAX_NEW_TOKENS = 256
MODEL_TRANSFORMERS_OFFLOAD_DIR = "hf_offload"

# PersonaPlex / Moshi settings
PERSONAPLEX_VOICE_PROMPT = "NATF2.pt"
PERSONAPLEX_TEXT_PROMPT = (
    "You are a wise and friendly teacher with multi-modal capabilities. "
    "You can hear your environment through a 10-second auditory memory and see "
    "the current screen. If you are unsure about a sound or want to investigate "
    "the recent past, use 'inspect_audio_snippet'. If you need to see what is on "
    "the screen, use 'inspect_current_screen'. Answer in a clear and engaging way."
)
PERSONAPLEX_VOICE_PROMPT_DIR = "personaplex"
PERSONAPLEX_DEVICE = "cuda"
PERSONAPLEX_CPU_OFFLOAD = True
PERSONAPLEX_USE_CUDA_GRAPHS = False
PERSONAPLEX_OFFLINE_TIMEOUT_SECONDS = 900
PERSONAPLEX_PYTHON_BIN = ""
PERSONAPLEX_VERBAL_FILLERS = [
    "Hmm, let me think about that for a second...",
    "One moment, I'm just checking on that...",
    "That's an interesting question, let me see...",
    "Oh, hold on, I think I know what you're talking about...",
    "Let me ponder that for a bit...",
]
SMOKE_MODEL_TIMEOUT_SECONDS = 120
INTERACTION_PROCESS_TIMEOUT_SECONDS = 300
LLM_DIAG_TIMEOUT_SECONDS = 40

# Local voice I/O settings
VOICE_SAMPLE_RATE = 24000
VOICE_CAPTURE_KEY_CODE = 18  # Ctrl+R
VOICE_CAPTURE_KEY_LABEL = "Ctrl+R"
PERSONAPLEX_SERVER_URL = "https://localhost:8998"
VOICE_AUTO_START_ON_LAUNCH = True
VOICE_MODE = "offline_continuous"
VOICE_ALWAYS_ON = True
VOICE_CHUNK_SECONDS = 0.4
VOICE_SILENCE_SECONDS = 0.9
VOICE_MIN_UTTERANCE_SECONDS = 0.6
VOICE_INTERJECT_POLICY = "hard"
VOICE_VAD_RMS_THRESHOLD = 0.012
VOICE_CONTROL_PREFIX = "\x1f"
VOICE_OFFLINE_INFER_TIMEOUT_SECONDS = 120
VOICE_STOP_GRACE_SECONDS = 2.0

# Development/runtime controls
DEV_DISABLE_AUTONOMOUS = False
INTERACTION_LOG_WARN_BYTES = 50 * 1024 * 1024
SHUTDOWN_GRACE_SECONDS = 5
THOUGHT_MAX_CONCURRENCY = 1
THOUGHT_MIN_INTERVAL_SECONDS = 2
EVENT_COMPRESS_MAX_ENTRIES = 500
