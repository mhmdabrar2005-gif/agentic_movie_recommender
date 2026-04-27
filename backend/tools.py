import os
import threading
from functools import lru_cache
import time
import requests
from langchain.tools import tool
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

@lru_cache(maxsize=1)
def get_vector_collection():
    """Initialize and cache Chroma collection + embedding model once per process."""
    client = chromadb.PersistentClient(path="./chroma_db")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_collection(name="movies", embedding_function=emb_fn)


def check_tmdb_health() -> dict:
    """Check whether TMDb endpoint is reachable from this machine."""
    start = time.perf_counter()

    if not TMDB_API_KEY:
        return {
            "ok": False,
            "status": "missing_api_key",
            "message": "TMDB_API_KEY is missing.",
            "elapsed_seconds": round(time.perf_counter() - start, 2),
        }

    endpoint = "https://api.themoviedb.org/3/configuration"
    params = {"api_key": TMDB_API_KEY}

    try:
        response = requests.get(
            endpoint,
            params=params,
        )
        elapsed = round(time.perf_counter() - start, 2)

        if response.status_code == 200:
            return {
                "ok": True,
                "status": "ok",
                "message": "TMDb is reachable.",
                "elapsed_seconds": elapsed,
            }

        return {
            "ok": False,
            "status": "http_error",
            "message": f"TMDb returned HTTP {response.status_code}.",
            "elapsed_seconds": elapsed,
        }
    except requests.exceptions.ConnectTimeout:
        return {
            "ok": False,
            "status": "connect_timeout",
            "message": "Connection to TMDb timed out.",
            "elapsed_seconds": round(time.perf_counter() - start, 2),
        }
    except requests.exceptions.ReadTimeout:
        return {
            "ok": False,
            "status": "read_timeout",
            "message": "TMDb response timed out.",
            "elapsed_seconds": round(time.perf_counter() - start, 2),
        }
    except Exception:
        return {
            "ok": False,
            "status": "unexpected_error",
            "message": "Unexpected error while reaching TMDb.",
            "elapsed_seconds": round(time.perf_counter() - start, 2),
        }

def _async_upsert_to_chroma(items, is_tv=False):
    """Background task to silently embed and save TMDb results to ChromaDB."""
    def task():
        try:
            collection = get_vector_collection()
            docs, metas, ids = [], [], []
            
            for item in items:
                item_id = str(item.get("id"))
                title = item.get("name") if is_tv else item.get("title")
                if not title: title = "Unknown"
                
                date = item.get("first_air_date") if is_tv else item.get("release_date")
                year = date[:4] if date else ""
                
                overview = item.get("overview", "")
                if not overview: continue
                    
                doc_text = f"{title} ({year}): {overview}"
                docs.append(doc_text)
                metas.append({"title": title, "id": item_id, "type": "tv" if is_tv else "movie", "source": "tmdb_auto"})
                ids.append(f"{'tv' if is_tv else 'movie'}_{item_id}")
                
            if docs:
                collection.upsert(documents=docs, metadatas=metas, ids=ids)
                print(f"\n[BACKGROUND] ✅ Auto-saved {len(docs)} {'TV shows' if is_tv else 'movies'} to local Vector DB!")
        except Exception as e:
            print(f"\n[BACKGROUND ERROR] ChromaDB auto-save failed: {e}")

    threading.Thread(target=task, daemon=True).start()

@tool
def search_tmdb(query: str, release_year: int | None = None) -> str:
    """Search TMDb for specific movies, actors, or directors."""
    print(f"\n[AGENT ACTION] 🛠️ Tool 'search_tmdb' invoked by LLM with query: '{query}'\n")
    if not TMDB_API_KEY:
        return "TMDb API key is missing. Set TMDB_API_KEY in your .env file."

    endpoint = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": query,
        "page": 1,
    }
    if release_year:
        params["primary_release_year"] = release_year

    try:
        response = requests.get(
            endpoint,
            params=params,
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            return "No movies found."

        # Filter out documentaries (genre ID 99) and behind-the-scenes
        filtered_results = []
        for m in results:
            genres = m.get("genre_ids", [])
            title = m.get("title", "").lower()
            if 99 in genres:
                continue
            if title.startswith("inside '") or title.startswith("the making of ") or "behind the scenes" in title:
                continue
            filtered_results.append(m)
            
        if not filtered_results:
            return "No valid movies found."

        # Save to DB in background
        _async_upsert_to_chroma(filtered_results[:3], is_tv=False)

        # Format top 3 results instantly
        return "\n".join(
            [f"- {m['title']} ({m.get('release_date', '')[:4]}): {m['overview']}" for m in filtered_results[:3]]
        )
    except requests.exceptions.ConnectTimeout:
        return "TMDb lookup failed: connection timed out. Check internet, firewall, or proxy settings."
    except requests.exceptions.ReadTimeout:
        return "TMDb lookup failed: response timed out. Try again in a moment."
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        return f"TMDb lookup failed: HTTP {status}."
    except Exception:
        return "TMDb lookup failed: unexpected network error."


@tool
def search_vector_db(mood: str) -> str:
    """Search local ChromaDB for movies and TV shows matching a vibe or mood."""
    print(f"\n[AGENT ACTION] 🛠️ Tool 'search_vector_db' invoked by LLM with mood: '{mood}'\n")
    try:
        collection = get_vector_collection()

        results = collection.query(query_texts=[mood], n_results=3)

        # Chroma query output is nested by query index: [[doc1, doc2, ...]]
        documents_nested = results.get("documents", [])
        metadatas_nested = results.get("metadatas", [])

        documents = documents_nested[0] if documents_nested else []
        metadatas = metadatas_nested[0] if metadatas_nested else []

        if not documents:
            return "No matches found."

        lines = []
        for idx, doc in enumerate(documents):
            meta = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
            title = meta.get("title", "Unknown title")
            item_type = meta.get("type", "movie")
            lines.append(f"- [{item_type.upper()}] {title}: {doc}")

        return "\n".join(lines)
    except Exception as exc:
        return f"Vector DB lookup failed: {exc}"

GENRE_MAP = {
    "action": 28, "adventure": 12, "animation": 16, "comedy": 35, 
    "crime": 80, "documentary": 99, "drama": 18, "family": 10751, 
    "fantasy": 14, "history": 36, "horror": 27, "music": 10402, 
    "mystery": 9648, "romance": 10749, "sci-fi": 878, "science fiction": 878, 
    "tv movie": 10770, "thriller": 53, "war": 10752, "western": 37
}

@tool
def get_trending_movies(time_window: str = "day", genre: str = None) -> str:
    """Get trending movies from TMDb. time_window can be 'day' or 'week'. If the user asks for a specific genre (like 'romance', 'sci-fi', 'action'), pass it to the genre parameter."""
    print(f"\n[AGENT ACTION] 🛠️ Tool 'get_trending_movies' invoked by LLM with time_window: '{time_window}', genre: '{genre}'\n")
    if not TMDB_API_KEY:
        return "TMDb API key is missing. Set TMDB_API_KEY in your .env file."

    if time_window not in ["day", "week"]:
        time_window = "day"

    params = {
        "api_key": TMDB_API_KEY,
    }

    if genre and genre.lower() in GENRE_MAP:
        # If a genre is requested, the generic trending endpoint doesn't support it.
        # We must use the discover endpoint sorted by popularity instead.
        endpoint = "https://api.themoviedb.org/3/discover/movie"
        params["sort_by"] = "popularity.desc"
        params["with_genres"] = GENRE_MAP[genre.lower()]
    else:
        # Standard global trending
        endpoint = f"https://api.themoviedb.org/3/trending/movie/{time_window}"

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            return "No trending movies found."

        # Save to DB in background
        _async_upsert_to_chroma(results[:5], is_tv=False)

        # Format top 5 results
        return "\n".join(
            [f"- {m.get('title', 'Unknown')} ({m.get('release_date', '')[:4]}): {m.get('overview', '')}" for m in results[:5]]
        )
    except Exception as exc:
        return f"TMDb trending lookup failed: {exc}"

TV_GENRE_MAP = {
    "action": 10759, "adventure": 10759, "animation": 16, "comedy": 35, 
    "crime": 80, "documentary": 99, "drama": 18, "family": 10751, 
    "kids": 10762, "mystery": 9648, "news": 10763, "reality": 10764, 
    "sci-fi": 10765, "science fiction": 10765, "fantasy": 10765, "soap": 10766, 
    "talk": 10767, "war": 10768, "politics": 10768, "western": 37
}

@tool
def search_tmdb_tv(query: str, first_air_year: int | None = None) -> str:
    """Search TMDb for specific TV shows."""
    print(f"\n[AGENT ACTION] 🛠️ Tool 'search_tmdb_tv' invoked by LLM with query: '{query}'\n")
    if not TMDB_API_KEY:
        return "TMDb API key is missing. Set TMDB_API_KEY in your .env file."

    endpoint = "https://api.themoviedb.org/3/search/tv"
    params = {
        "api_key": TMDB_API_KEY,
        "query": query,
        "page": 1,
    }
    if first_air_year:
        params["first_air_date_year"] = first_air_year

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            return "No TV shows found."

        # Save to DB in background
        _async_upsert_to_chroma(results[:3], is_tv=True)

        return "\n".join(
            [f"- {m.get('name', 'Unknown')} ({m.get('first_air_date', '')[:4]}): {m.get('overview', '')}" for m in results[:3]]
        )
    except Exception as exc:
        return f"TMDb TV lookup failed: {exc}"

@tool
def get_trending_tv_shows(time_window: str = "day", genre: str = None) -> str:
    """Get trending TV shows from TMDb. time_window can be 'day' or 'week'. If user asks for a specific genre, pass it."""
    print(f"\n[AGENT ACTION] 🛠️ Tool 'get_trending_tv_shows' invoked by LLM with time_window: '{time_window}', genre: '{genre}'\n")
    if not TMDB_API_KEY:
        return "TMDb API key is missing. Set TMDB_API_KEY in your .env file."

    if time_window not in ["day", "week"]:
        time_window = "day"

    params = {
        "api_key": TMDB_API_KEY,
    }

    if genre and genre.lower() in TV_GENRE_MAP:
        endpoint = "https://api.themoviedb.org/3/discover/tv"
        params["sort_by"] = "popularity.desc"
        params["with_genres"] = TV_GENRE_MAP[genre.lower()]
    else:
        endpoint = f"https://api.themoviedb.org/3/trending/tv/{time_window}"

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            return "No trending TV shows found."

        # Save to DB in background
        _async_upsert_to_chroma(results[:5], is_tv=True)

        return "\n".join(
            [f"- {m.get('name', 'Unknown')} ({m.get('first_air_date', '')[:4]}): {m.get('overview', '')}" for m in results[:5]]
        )
    except Exception as exc:
        return f"TMDb trending TV lookup failed: {exc}"

get_tools = [search_tmdb, search_vector_db, get_trending_movies, search_tmdb_tv, get_trending_tv_shows]