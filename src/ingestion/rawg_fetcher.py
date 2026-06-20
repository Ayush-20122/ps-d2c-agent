import requests
from src.config import RAWG_API_KEY

BASE_URL = "https://api.rawg.io/api"

def fetch_playstation_games(page_size: int = 40, page: int = 1) -> list[dict]:
    """
    Fetch Playstation Games from RAWG API.
    Filters by Playstation Platforms only.
    """

    # Playstation Platform IDs on RAWG:
    # 187 = PS5, 18 = PS4, 16 = PS3, 15 = PS2, 27 = PS1
    response = requests.get(
        f"{BASE_URL}/games",
        params={
            "key": RAWG_API_KEY,
            "platforms": "187,18,16",
            "page_size": page_size,
            "page": page,
            "ordering": "-rating",
        }
    )

    response.raise_for_status()
    data = response.json()
    return data.get("results", [])

def fetch_game_details(game_id: int) -> dict:
    """
    Fetch detailed info for a single game by its RAWG ID.
    """

    response = requests.get(
        f"{BASE_URL}/games/{game_id}",
        params = {"key": RAWG_API_KEY}
    )

    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    print("Fetching top Playstation games....")
    games = fetch_playstation_games(page_size=5)

    for game in games:
        print(f"- {game['name']} | Rating {game.get('rating', 'N/A')}")