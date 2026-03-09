"""
Fetch movie poster URLs from the TMDB API by title.

Requires TMDB_API_KEY in .env (free at https://www.themoviedb.org/settings/api).
"""

import os
import re
import httpx
import dotenv

dotenv.load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"


async def fetch_poster_url(title: str, year: str | None = None) -> str | None:
    """Search TMDB for a movie by title and return its poster URL, or None."""
    if not TMDB_API_KEY:
        return None

    params = {"api_key": TMDB_API_KEY, "query": title}
    if year and year != "Unknown":
        params["year"] = year

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(TMDB_SEARCH_URL, params=params)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results and results[0].get("poster_path"):
                return TMDB_IMAGE_BASE + results[0]["poster_path"]
    except Exception:
        pass

    return None


def extract_titles_from_response(text: str) -> list[tuple[str, str | None]]:
    """
    Parse movie titles from the agent's markdown response.
    Matches patterns like:
      **Movie Title** (2010)
      **Movie Title**
      "Movie Title" (2010)
      "Movie Title"
    Returns list of (title, year_or_None) tuples.
    """
    seen = set()
    results = []

    # **Title** with optional (year)
    for match in re.finditer(r"\*\*(.+?)\*\*(?:\s*\((\d{4})\))?", text):
        title = match.group(1).strip()
        year = match.group(2)
        # Skip short/generic bold text that isn't a movie title
        if len(title) < 2 or title.lower() in ("for", "note", "tip", "warning"):
            continue
        key = title.lower()
        if key not in seen:
            seen.add(key)
            results.append((title, year))

    # "Title" with optional (year)
    for match in re.finditer(r'["""](.+?)["""](?:\s*\((\d{4})\))?', text):
        title = match.group(1).strip()
        year = match.group(2)
        if len(title) < 2:
            continue
        key = title.lower()
        if key not in seen:
            seen.add(key)
            results.append((title, year))

    return results


async def fetch_posters_for_response(text: str) -> list[tuple[str, str]]:
    """
    Given an agent response text, extract movie titles and fetch their poster URLs.
    Returns list of (title, poster_url) tuples for movies where a poster was found.
    """
    titles = extract_titles_from_response(text)
    posters = []
    for title, year in titles:
        url = await fetch_poster_url(title, year)
        if url:
            posters.append((title, url))
    return posters


async def insert_posters_into_text(text: str) -> str:
    """
    Insert markdown poster images inline into the response text,
    right after each movie title line.
    """
    titles = extract_titles_from_response(text)
    if not titles:
        return text

    for title, year in titles:
        url = await fetch_poster_url(title, year)
        if not url:
            continue
        poster_md = f"\n\n![{title}]({url})\n"

        # Build pattern to find this title in the text
        if year:
            patterns = [
                re.escape(f"**{title}**") + r"\s*\(" + re.escape(year) + r"\)",
                r'["""]' + re.escape(title) + r'["""][\s]*\(' + re.escape(year) + r'\)',
            ]
        else:
            patterns = [
                re.escape(f"**{title}**"),
                r'["""]' + re.escape(title) + r'["""]',
            ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                end = match.end()
                line_end = text.find("\n", end)
                if line_end == -1:
                    line_end = len(text)
                text = text[:line_end] + poster_md + text[line_end:]
                break

    return text
