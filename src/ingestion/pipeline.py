import time
from src.ingestion.rawg_fetcher import fetch_playstation_games, fetch_game_details
from src.ingestion.igdb_fetcher import get_access_token, fetch_playstation_games as fetch_igbd_games
from src.ingestion.chunker import chunk_rawg_game, chunk_igdb_game
from src.ingestion.embedder import embed_chunks
from src.ingestion.vector_store import setup_database, store_chunks

def run_rawg_ingestion(page_size: int = 40, pages: int = 5):
    """
    Fetch, chunk, embed and store PlayStation games from RAWG.
    Fetches `pages` pages of `page_size` games each.
    """
    print(f"\n{"="*50}")
    print("Starting RAWG Ingestion Pipeline...")
    print(f"{"="*50}")

    total_chunks = []
    skipped = 0

    for page in range(1, pages + 1):
        print(f"\nFetching RAWG page {page}/{pages}...")
        games = fetch_playstation_games(page_size=page_size, page=page)
        print(f"Fetched {len(games)} games.")

        for game in games:
            try:
                # fetch_game_details() gets richer data per game genres, platforms, descriptions since the list endpoint only returns minimal fields
                details = fetch_game_details(game["id"])
                chunks = chunk_rawg_game(details)

                # extend() adds each item from chunks individually into total_chunks,
                # unlike append() which would add the whole list as a single nested element
                total_chunks.extend(chunks)

            except Exception as e:
                # Skip games that return 404 or any other error
                # Log the skip so we have visibility into what was skipped
                skipped += 1
                print(f"Skipping game {game.get('name', game['id'])}: {e}")
                continue

        # Wait 1 second between pages to respect RAWG's rate limits
        # Only sleep if there are more pages remaining
        if page < pages:
            time.sleep(1)

    print(f"\nTotal RAWG chunks to embed: {len(total_chunks)}")
    embedded = embed_chunks(total_chunks)

    print("Storing RAWG chunks in pgvector...")
    store_chunks(embedded)

    print("RAWG Ingestion complete.")

def run_igdb_ingestion(limit: int = 100):
    """
    Fetch, chunk, embed and store PlayStation games from IGDB.
    Fetches up to `limit` games in a single request.
    """
    print(f"\n{'='*50}")
    print("Starting IGDB ingestion pipeline...")
    print(f"{'='*50}")

    # IGDB requires a fresh OAuth token on every run tokens expire after a few hours so we always fetch a new one
    print("Getting IGDB Access Token...")
    token = get_access_token()

    print(f"Fetching {limit} games from IGDB...")
    games = fetch_igbd_games(token, limit=limit)
    print(f"Fetched {len(games)} games.")

    total_chunks = []
    for game in games:
        chunks = chunk_igdb_game(game)
        total_chunks.extend(chunks)

    print(f"\nTotal IGDB chunks to embed: {len(total_chunks)}")
    embedded = embed_chunks(total_chunks)

    print("Storing IGDB chunks in pgvector...")
    store_chunks(embedded)
    print("IGDB ingestion complete.")

def run_full_pipeline():
    """
    Entry point for the complete ingestion pipeline.
    Runs in order:
    1. Sets up database schema (idempotent — safe to run repeatedly)
    2. Ingests from RAWG (200 games, ~600 chunks)
    3. Ingests from IGDB (100 games, ~300 chunks)
    Total: ~900 chunks stored in pgvector.
    """
    print("Setting up database...")
    setup_database()

    run_rawg_ingestion(page_size=40, pages=5)
    run_igdb_ingestion(limit=100)

    print(f"\n{'='*50}")
    print("Full ingestion pipeline complete.")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_full_pipeline()