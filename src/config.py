from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHROMA_DIR = BASE_DIR / "chroma_db"
PROMPT_FILE = BASE_DIR / "prompts" / "system_prompt.txt"

ORDERS_FILE = DATA_DIR / "dummy_orders.csv"
INVENTORY_FILE = DATA_DIR / "dummy_inventory.csv"
POLICY_FILE = DATA_DIR / "business_policy.txt"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
