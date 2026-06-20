import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from src.config import DATABASE_URL
from src.ingestion.chunker import GameChunk
from src.ingestion.embedder import EMBEDDING_DIMENSIONS

def get_connection():
    """
    Create and return a connection to the pgvector PostgreSQL database.
    """
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def setup_database():
    """
    Creates the game_chunks table and HNSW index if they don't exist.
    Safe to run multiple times — uses IF NOT EXISTS.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:

            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create the game_chunks table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS game_chunks (
                    id          SERIAL PRIMARY KEY,
                    game_id     TEXT NOT NULL,
                    game_name   TEXT NOT NULL,
                    source      TEXT NOT NULL,
                    chunk_type  TEXT NOT NULL,
                    text        TEXT NOT NULL,
                    metadata    JSONB,
                    embedding   vector({EMBEDDING_DIMENSIONS})   
                );
            """)

            # Add unique constraint to prevent duplicate chunks
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'unique_game_chunk'
                    ) THEN
                        ALTER TABLE game_chunks
                        ADD CONSTRAINT unique_game_chunk
                        UNIQUE (game_id, source, chunk_type);
                    END IF;
                END
                $$;
            """)

            # Create HNSW index for fast cosine similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS game_chunks_embedding_idx
                ON game_chunks
                USING hnsw (embedding vector_cosine_ops);
            """)

            conn.commit()
            print("Database setup complete.")

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def store_chunks(chunks_with_embeddings: list[tuple[GameChunk, list[float]]]):
    """
    Store a list of (GameChunk, embedding) pairs into pgvector.
    Uses INSERT ... ON CONFLICT ON CONSTRAINT unique_game_chunk DO NOTHING to idempotently skip duplicate (game_id, source, chunk_type) combinations.
    Commits all inserts in a single transaction for atomicity and performance.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for chunk, embedding in chunks_with_embeddings:
                cur.execute("""
                    INSERT INTO game_chunks
                        (game_id, game_name, source, chunk_type, text, metadata, embedding)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT ON CONSTRAINT unique_game_chunk DO NOTHING;
                """,(
                    chunk.game_id,
                    chunk.game_name,
                    chunk.source,
                    chunk.chunk_type,
                    chunk.text,
                    psycopg2.extras.Json(chunk.metadata),
                    embedding,
                ))
        conn.commit()
        print(f"Stored {len(chunks_with_embeddings)} chunks successfully.")

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def search_similar_chunks(
        query_embedding: list[float],
        top_k: int = 5,
        source: str = None,
        chunk_type: str = None
) -> list[dict]:
    """
    Find the top_k most similar chunks to a query embedding using cosine similarity via the HNSW index.
    Optionally filter by source (rawg/igdb) or chunk_type.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            
            # Build dynamic WHERE clause based on optional filters
            filters = []
            params = []

            if source:
                filters.append("source = %s")
                params.append(source)

            if chunk_type:
                filters.append("chunk_type = %s")
                params.append(chunk_type)

            where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

            # <=> is pgvector's cosine distance operator
            # cosine distance = 1 - cosine similarity
            # so ORDER BY ASC gives most similar first
            cur.execute(f"""
                SELECT
                    game_id,
                    game_name,
                    source,
                    chunk_type,
                    text,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM game_chunks
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, [query_embedding, *params, query_embedding, top_k])

            return cur.fetchall()
            
    except Exception as e:
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    from src.ingestion.chunker import chunk_rawg_game
    from src.ingestion.embedder import embed_chunks, embed_text

    print("Setting up database...")
    setup_database()

    fake_game = {
        "id": 1,
        "name": "God of War",
        "rating": 4.7,
        "released": "2018-04-20",
        "playtime": 22,
        "genres": [{"name": "Action"}, {"name": "Adventure"}],
        "platforms": [{"platform": {"name": "PlayStation 4"}}]
    }

    print("Chunking and embedding game...")
    chunks = chunk_rawg_game(fake_game)
    embedded = embed_chunks(chunks)

    print("Storing chunks in pgvector...")
    store_chunks(embedded)

    print("\nSearching for 'games on PS4'...")
    query_embedding = embed_text("games on PS4")
    results = search_similar_chunks(query_embedding, top_k=3)

    for result in results:
        print(f"\n[{result['chunk_type']}] {result['game_name']} "
              f"(similarity: {round(result['similarity'], 4)})")
        print(f"Text: {result['text']}")