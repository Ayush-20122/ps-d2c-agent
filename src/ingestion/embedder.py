import voyageai
import time
from src.config import VOYAGEAI_API_KEY
from src.ingestion.chunker import GameChunk

# Initialize Voyage client once at module level
client = voyageai.Client(api_key=VOYAGEAI_API_KEY)

# Use voyage-4 — latest generation Voyage 4 family model
# Balanced general-purpose, 1024 default dimensions
# Shares embedding space with voyage-4-large and voyage-4-lite
# enabling asymmetric retrieval across the Voyage 4 family
EMBEDDING_MODEL = "voyage-4"
EMBEDDING_DIMENSIONS = 1024

def embed_text(text: str) -> list[float]:
    """
    Convert a single piece of text into a 1024-dimensional
    vector embedding using Voyage AI.
    """
    result = client.embed(
        texts = [text],
        model = EMBEDDING_MODEL,
        input_type = "query",
        output_dimension = EMBEDDING_DIMENSIONS
    )

    return result.embeddings[0]

def embed_chunks(chunks: list[GameChunk], batch_size: int = 10) -> list[tuple[GameChunk, list[float]]]:
    """
    Embed a list of GameChunks in batches to respect rate limits.
    Returns a list of (chunk, embedding) pairs.
    """
    results = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        print(f"Embedding Batch {i // batch_size + 1} ({len(batch)} chunks)...")

        # Embed entire batch in one API call — more efficient than one by one
        texts = [chunk.text for chunk in batch]
        response = client.embed(
            texts = texts,
            model = EMBEDDING_MODEL,
            input_type = "document",
            output_dimension = EMBEDDING_DIMENSIONS
        )

        for chunk, embedding in zip(batch, response.embeddings):
            results.append((chunk, embedding))

        time.sleep(0.5)

    return results

if __name__ == "__main__":
    from src.ingestion.chunker import chunk_rawg_game

    fake_game = {
        "id": 1,
        "name": "God of War",
        "rating": 4.7,
        "released": "2018-04-20",
        "playtime": 22,
        "genres": [{"name": "Action"}, {"name": "Adventure"}],
        "platforms": [{"platform": {"name": "PlayStation 4"}}]
    }

    print("Chunking game...")
    chunks = chunk_rawg_game(fake_game)

    print(f"Embedding {len(chunks)} chunks...\n")
    embedded = embed_chunks(chunks)

    for chunk, embedding in embedded:
        print(f"[{chunk.chunk_type}]\nText: {chunk.text}\nVector dims: {len(embedding)}\n")
        print(f"[{chunk.chunk_type}] → vector of {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}\n")