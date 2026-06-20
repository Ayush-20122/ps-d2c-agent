from dataclasses import dataclass

@dataclass
class GameChunk:
    """
    Represents a single chunk of game data ready for embedding.
    Each chunk is a self-contained piece of text about one game.
    """
    game_id: str
    game_name: str
    source: str                # "rawg" or "igdb"
    chunk_type: str            # 'overview', 'genres', 'companies'
    text: str                  # the actual text that gets embeded
    metadata: dict             # extra info stored aongside the vector


def chunk_rawg_game(game: dict) -> list[GameChunk]:
    """
    Takes a raw RAWG API game object and splits it into multiple focused chunks for better retrieval precision.
    """
    chunks = []
    game_id = str(game.get("id",""))
    game_name = game.get("name", "Unknown")

    # Chunk 1: Overview - name, rating, release date
    overview_text = f"""
    Game: {game_name}
    Rating: {game.get('rating','N/A')} out of 5
    Released: {game.get('released', 'N/A')}
    Playime: {game.get('playtime', 'N/A')} hours average
    """.strip()

    chunks.append(GameChunk(
        game_id = game_id,
        game_name = game_name,
        source = "rawg",
        chunk_type = "overview",
        text = overview_text,
        metadata = {
            "rating": game.get("rating"),
            "released": game.get("released")
        }
    ))

    # Chunk 2: Genres
    genres = game.get("genres", [])
    if genres:
        genre_names = ", ".join([g["name"] for g in genres])
        genre_text = f"{game_name} belongs to the following genres: {genre_names}."

        chunks.append(GameChunk(
            game_id = game_id,
            game_name = game_name,
            source = "rawg",
            chunk_type = "genres",
            text = genre_text,
            metadata = {"genres": genre_names}
        ))

    # Chunk 3: Platforms
    platforms = game.get("platforms", [])
    if platforms:
        platform_names = ", ".join([p['platform']['name'] for p in platforms])
        platform_text = f"{game_name} is available on: {platform_names}."

        chunks.append(GameChunk(
            game_id = game_id,
            game_name = game_name,
            source = "rawg",
            chunk_type = "platforms",
            text = platform_text,
            metadata = {"platforms": platform_names}
        ))

    return chunks



def chunk_igdb_game(game: dict) -> list[GameChunk]:
    """
    Takes a raw IGDB API game object and splits it into multiple focused chunks for better retrieval precision.
    """
    chunks = []
    game_id = game.get("id", "")
    game_name = game.get("name", "Unknown")

    # Chunk 1: Overview - name, rating, summary
    summary = game.get("summary", "No summary available.")
    overview_text = f"""
    Game: {game_name}
    Rating: {round(game.get("rating", 0), 2)} out of 100
    Summary: {summary}
    """.strip()

    chunks.append(GameChunk(
        game_id = game_id,
        game_name = game_name,
        source = "igdb",
        chunk_type = "overview",
        text = overview_text,
        metadata = {
            "rating": game.get("rating"),
            "summary": summary
        }
    ))


    # Chunk 2: Genres
    genres = game.get("genres", [])
    if genres:
        genre_names = ", ".join([g["name"] for g in genres])
        genre_text = f"{game_name} belongs to the following genres: {genre_names}."

        chunks.append(GameChunk(
            game_id=game_id,
            game_name=game_name,
            source="igdb",
            chunk_type="genres",
            text=genre_text,
            metadata={"genres": genre_names}
        ))

    # Chunk 3: Companies
    companies = game.get("involved_companies", [])
    if companies:
        company_names = ", ".join([
            c["company"]["name"] for c in companies
            if "company" in c
        ])

        company_text = f"{game_name} was developed/published by: {company_names}."

        chunks.append(GameChunk(
            game_id = game_id,
            game_name = game_name,
            source = "igdb",
            chunk_type = "companies",
            text = company_text,
            metadata = {"companies": company_names}
        ))

    return chunks

if __name__ == "__main__":
    # Quick Test with fake data
    fake_rawg_game = {
        "id": 1,
        "name": "God of War",
        "rating": 4.7,
        "released": "2018-04-20",
        "playtime": 22,
        "genres": [{"name": "Action"}, {"name": "Adventure"}],
        "platforms": [{"platform": {"name": "PlayStation 4"}}]
    }

    fake_igdb_game = {
        "id": 1,
        "name": "God of War",
        "rating": 97.3,
        "summary": "A father and son journey through Norse mythology.",
        "genres": [{"name": "Action"}, {"name": "Adventure"}],
        "involved_companies": [{"company": {"name": "Santa Monica Studio"}}]
    }

    print("=== RAWG Chunks ===")
    for chunk in chunk_rawg_game(fake_rawg_game):
        print(f"\n[{chunk.chunk_type}]\n{chunk.text}")

    print("\n=== IGDB Chunks ===")
    for chunk in chunk_igdb_game(fake_igdb_game):
        print(f"\n[{chunk.chunk_type}]\n{chunk.text}")