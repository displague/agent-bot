# config.py

HARD_LOG_PATH = "logs/hard_log.jsonl"
COMPRESSED_LOG_PATH = "compressed_logs/compressed_log.jsonl"
INDEX_PATH = "index/context_index.json"
DAILY_SLEEP_START = 23
DAILY_SLEEP_END = 7
MAX_WORKERS = 5
MODEL_PATH = "model.bin"  # Remember to put your actual model path here

# PersonaPlex / Moshi settings
PERSONAPLEX_VOICE_PROMPT = "NATF2.pt"
PERSONAPLEX_TEXT_PROMPT = (
    "You are a wise and friendly teacher. Answer questions or provide advice in a "
    "clear and engaging way."
)
PERSONAPLEX_VOICE_PROMPT_DIR = ""
PERSONAPLEX_DEVICE = "cuda"
PERSONAPLEX_CPU_OFFLOAD = False
PERSONAPLEX_OFFLINE_TIMEOUT_SECONDS = 900
PERSONAPLEX_PYTHON_BIN = ""

# Local voice I/O settings
VOICE_CAPTURE_SECONDS = 6
VOICE_SAMPLE_RATE = 24000
