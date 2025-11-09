from __future__ import annotations
import os, json, time
from datetime import datetime
import feedparser

CACHE_DIR = os.path.join("data","news")
os.makedirs(CACHE_DIR, exist_ok=True)

DEFAULT_FEEDS = {
    "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "CNBC Top": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Yahoo Finance": "https://finance.yahoo.com/rss/topstories",
}

def pull_feeds(feeds: dict[str,str] | None=None, ttl_sec: int=1800) -> list[dict]:
    feeds = feeds or DEFAULT_FEEDS
    cache_file = os.path.join(CACHE_DIR, "latest.json")
    if os.path.exists(cache_file) and (time.time()-os.path.getmtime(cache_file) < ttl_sec):
        return json.load(open(cache_file))

    items = []
    for src, url in feeds.items():
        d = feedparser.parse(url)
        for e in d.entries:
            ts = None
            for k in ("published_parsed","updated_parsed"):
                if getattr(e, k, None):
                    ts = datetime(*getattr(e,k)[:6]).isoformat()
                    break
            items.append({
                "source": src,
                "title": getattr(e,"title",""),
                "link": getattr(e,"link",""),
                "time": ts
            })
    items.sort(key=lambda x: x["time"] or "", reverse=True)
    with open(cache_file,"w") as f:
        json.dump(items, f, indent=2)
    return items
