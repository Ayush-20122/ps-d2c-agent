from src.ingestion.embedder import embed_text
from src.ingestion.vector_store import search_similar_chunks

def retrieve(
    query: str,
    top_k: int = 5,
    source: str = None,
    chunk_type: str = None
) -> list[dict]:
    """
    Given a natural language query, embed it and retrieve the top_k most semantically similar chunks from pgvector.
    Optionally filter by source (rawg/igdb) or chunk_type.
    """
    query_embedding = embed_text(query)

    results = search_similar_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        source=source,
        chunk_type=chunk_type
    )

    return results

def format_context(results: list[dict]) -> str:
    """
    Format retrieved chunks into a clean context string that gets injected into Claude's prompt.
    Each chunk is clearly labeled with its source and type so Claude knows where the information came from.
    """
    if not results:
        return "No relevant game information found."
    
    context_parts = []

    for i, result in enumerate(results, start=1):
        part = (
            f"[Source {i}: {result['game_name']}] "
            f"({result['source'].upper()}, {result['chunk_type']}), "
            f"Similarity: {round(result['similarity'], 4)}]\n"
            f"{result['text']}"
        )
        context_parts.append(part)

    # Join all parts with a clear separator between chunks
    return "\n\n---\n\n".join(context_parts)

if __name__ == "__main__":
    queries = [
        "action adventure game on PS5",
        "open world RPG with good story",
        "multiplayer shooting game PlayStation",
        "who made God of War",
        "games similar to Dark Souls",
    ]

    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")

        results = retrieve(query, top_k=3)
        context = format_context(results)
        print(context)