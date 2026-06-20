from dotenv import load_dotenv

import os

#Load all variables from .env to the environment
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VOYAGEAI_API_KEY = os.getenv("VOYAGEAI_API_KEY")
RAWG_API_KEY = os.getenv("RAWG_API_KEY")
IGDB_CLIENT_ID = os.getenv("IGDB_CLIENT_ID")
IGDB_CLIENT_SECRET = os.getenv("IGDB_CLIENT_SECRET")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")

DATABASE_URL = os.getenv("DATABASE_URL")

required = {
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    "VOYAGEAI_API_KEY": VOYAGEAI_API_KEY,
    "RAWG_API_KEY": RAWG_API_KEY,
    "IGDB_CLIENT_ID": IGDB_CLIENT_ID,
    "IGDB_CLIENT_SECRET": IGDB_CLIENT_SECRET,
    "LANGSMITH_API_KEY": LANGSMITH_API_KEY,
    "LANGSMITH_PROJECT": LANGSMITH_PROJECT,
    "LANGCHAIN_TRACING_V2": LANGCHAIN_TRACING_V2,
    "LANGSMITH_ENDPOINT": LANGSMITH_ENDPOINT,
    "DATABASE_URL": DATABASE_URL,
}

missing = [key for key, value in required.items() if not value]

if missing:
    raise ValueError(f"Missing required environment variables: {missing}");