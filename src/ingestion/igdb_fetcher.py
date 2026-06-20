import requests
from src.config import IGDB_CLIENT_ID, IGDB_CLIENT_SECRET

TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
IGDB_BASE_URL = "https://api.igdb.com/v4"

def get_access_token() -> str:
    """
    Exchange Client ID + Client Secret for Twitch OAuth Access Token.
    This token is then used to authenticate all IGDB API calls.
    """

    response = requests.post(
        TWITCH_TOKEN_URL,
        params = {
            "client_id": IGDB_CLIENT_ID,
            "client_secret": IGDB_CLIENT_SECRET,
            "grant_type": "client_credentials",
        }
    )
    response.raise_for_status()
    return response.json()["access_token"]

def fetch_playstation_games(access_token: str, limit: int = 10) -> list[dict]:
    """
    Fetch top rated PlayStation games from IGDB.
    Platform ID 48 = PS4, 167 = PS5
    """

    headers = {
        "Client-ID": IGDB_CLIENT_ID,
        "Authorization": f"Bearer {access_token}"
    }

    # IGDB uses a custom query language called Apicalypse
    query = f"""
    fields name, summary, rating, genres.name, platforms.name, cover.url, involved_companies.company.name;
    where platforms = (48, 167) & rating != null & rating_count > 20;
    sort rating desc;
    limit {limit};
    """
    
    response = requests.post(
        f"{IGDB_BASE_URL}/games",
        headers = headers,
        data = query
    )
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    print("Getting IGDB Access Token....")
    token = get_access_token()
    print(f"Token Acquired....")

    print("Fetching top Playstation games from IGDB....")
    games = fetch_playstation_games(token, limit=5)

    for game in games:
        print(f"- {game['name']} | Rating: {round(game.get('rating', 0), 2)}")
