"""Recipe video lookup through the YouTube Data API v3.

A search call costs 100 quota units (the free daily quota is 10,000), so results
are cached per query. Without an API key the app falls back to a search link.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import quote_plus

import requests

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


@dataclass(frozen=True)
class Video:
    title: str
    url: str
    channel: str
    thumbnail: str


def search_link(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"


@lru_cache(maxsize=256)
def search_recipe_videos(query: str, api_key: str, max_results: int = 3) -> tuple[Video, ...]:
    if not api_key:
        return ()
    params = {
        "part": "snippet", "q": query, "type": "video", "maxResults": max_results,
        "videoEmbeddable": "true", "safeSearch": "strict", "relevanceLanguage": "en",
        "key": api_key,
    }
    resp = requests.get(SEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    videos = []
    for item in resp.json().get("items", []):
        vid = item.get("id", {}).get("videoId")
        if not vid:
            continue
        snip = item["snippet"]
        videos.append(Video(
            title=snip.get("title", ""), url=f"https://www.youtube.com/watch?v={vid}",
            channel=snip.get("channelTitle", ""),
            thumbnail=snip.get("thumbnails", {}).get("medium", {}).get("url", "")))
    return tuple(videos)
