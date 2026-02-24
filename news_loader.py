"""
news_loader.py
---------------
Uses Perplexity Sonar API to fetch recent company news headlines and store them in ChromaDB
"""

from __future__ import annotations
import os
import uuid
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from perplexity import Perplexity
import chromadb
from chromadb.utils import embedding_functions
import pytz 


NY_TZ = pytz.timezone("America/New_York")



# Classes
@dataclass
class NewsItem:
    title: str
    url: str
    snippet: Optional[str]
    date: Optional[str]        # raw string from API; we’ll parse below
    source: Optional[str]      # domain (e.g., 'reuters.com')
    ticker: str                # stock ticker for traceability


class SonarNewsClient:
    "Using Perplexity Sonar API for finance queries"
    def __init__(self, model: str = "sonar-pro", max_results: int = 8):
        load_dotenv()
        if not os.getenv("PPLX_API_KEY"):
            raise RuntimeError("Missing PPLX_API_KEY in environment. Add it to your .env file.")
        self.client = Perplexity()
        self.model = model
        self.max_results = max_results

    def build_query(self, ticker: str, time_range: str = "1m") -> str:
        "Build natural-language search query for Sonar"
        return (
            f"Recent market-moving news for {ticker}. "
            f"Focus on earnings, guidance, M&A, regulation, product, supply chain, "
            f"and management commentary in the last {time_range}. "
            f"Return diverse, reputable sources."
        )

    def search_news(self, ticker: str, time_range: str = "1m") -> List[NewsItem]:
        "Send search query to Sonar API and parse results"
        query = self.build_query(ticker, time_range)
        search = self.client.search.create(
            query=query,
            max_results=self.max_results,
            max_tokens_per_page=1024,
        )

        items: List[NewsItem] = []
        for r in getattr(search, "results", []) or []:
            url = getattr(r, "url", None) or ""
            items.append(
                NewsItem(
                    title=getattr(r, "title", None) or "",
                    url=url,
                    snippet=getattr(r, "snippet", None),
                    date=getattr(r, "date", None),  # may be date-only or full ISO
                    source=url.split("/")[2] if url else None,
                    ticker=ticker.upper(),
                )
            )
        return items



# Building metadata

def now_ny() -> dt.datetime:
    """Current time in New York timezone."""
    return dt.datetime.now(NY_TZ)

def ensure_publish_ny(publish_raw: Optional[str]) -> dt.datetime:
    """
    Parse article's publish timestamp and convert to New York time.
    If missing time or only date provided, default time to 09:30:00 ET.
    """
    default_time = dt.time(hour=9, minute=30, second=0)  # market open ET
    if not publish_raw:
        # Use today's date as a fallback if nothing is provided
        base = now_ny().date()
        return NY_TZ.localize(dt.datetime.combine(base, default_time))

    # Try a few safe parses without adding new deps
    s = publish_raw.strip()
    try:
        # Full ISO with timezone (e.g., 2025-11-02T14:03:00Z or with offset)
        if s.endswith("Z"):
            ts = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            ts = dt.datetime.fromisoformat(s)
        # If naive, assume UTC then convert; if aware, convert to NY
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        ts_ny = ts.astimezone(NY_TZ)
        return ts_ny
    except Exception:
        # Date (YYYY-MM-DD) or unknown format → default 09:30
        try:
            d = dt.date.fromisoformat(s[:10])
            return NY_TZ.localize(dt.datetime.combine(d, default_time))
        except Exception:
            # Last resort: use today's date with 09:30
            return NY_TZ.localize(dt.datetime.combine(now_ny().date(), default_time))

def sanitize_for_key(s: str) -> str:
    """Make a URL/file-ish string safe for the composite key."""
    return s.replace("://", "_").replace("/", "_").replace("?", "_").replace("&", "_")

def build_metadata(
    item: NewsItem,
    company: Optional[str] = None,
    doctype: str = "news",
) -> Dict[str, Any]:
    "Build metadata matching the team’s desired fields + composite key."
    # Insert time (retrieval) in NY
    ts_insert_ny = now_ny()
    date_retrieved = ts_insert_ny.date().isoformat()
    time_retrieved = ts_insert_ny.strftime("%H:%M:%S")

    # Publish time in NY (with default 09:30 if time missing)
    ts_pub_ny = ensure_publish_ny(item.date)
    date_published = ts_pub_ny.date().isoformat()
    time_published = ts_pub_ny.strftime("%H:%M:%S")

    symbol = item.ticker
    company_name = company or item.ticker  # if you don’t have a lookup yet

    meta = {
        "symbol": symbol,
        "company": company_name,
        "date_retrieved": date_retrieved,
        "time_retrieved": time_retrieved,
        "date_published": date_published,
        "time_published": time_published,
        "source": item.source or "",
        "url": item.url,
        "title": item.title,
        "snippet": item.snippet,
        "doctype": doctype,
    }

    # Composite key Format: ticker_company_datetimeNY_dateofpublish_timeofpublish_typeofdoc_url
    composite = (
        f"{symbol}_{sanitize_for_key(company_name)}_"
        f"{date_retrieved}T{time_retrieved}_"
        f"{date_published}_{time_published}_"
        f"{doctype}_{sanitize_for_key(item.url)}"
    )
    meta["metadata_key"] = composite
    return meta



# Chroma upsert
def get_chroma_collection(
    collection_name: str = "news_data",
    persist_dir: str = "./chroma"
):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment for Chroma embeddings.")
    client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

def upsert_to_chroma(
    items: List[NewsItem],
    collection_name: str = "news_data",
    persist_dir: str = "./chroma",
    company: Optional[str] = None,
) -> int:
    if not items:
        return 0
    col = get_chroma_collection(collection_name, persist_dir)

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for it in items:
        # Embed title+snippet for retrieval
        text = it.title if not it.snippet else f"{it.title}\n\n{it.snippet}"
        documents.append(text)
        metadatas.append(build_metadata(it, company=company, doctype="news"))
        ids.append(str(uuid.uuid4()))  # Use UUIDs; keep the composite string inside metadata

    try:
        col.add(documents=documents, metadatas=metadatas, ids=ids)
        return len(ids)
    except Exception:
        # If any duplicate UUID collision (unlikely), try again fresh
        col.add(documents=documents, metadatas=metadatas, ids=[str(uuid.uuid4()) for _ in ids])
        return len(ids)



# Put it all together

def fetch_and_upsert_news(
    ticker: str,
    time_range: str = "1m",
    model: str = "sonar-pro",
    max_results: int = 8,
    collection_name: str = "news_data",
    persist_dir: str = "./chroma",
    company: Optional[str] = None,
) -> str:
    "Creates client, fetches news for given ticker, and upsert results to timestamped ChromaDB"
    client = SonarNewsClient(model=model, max_results=max_results)
    items = client.search_news(ticker, time_range=time_range)
    count = upsert_to_chroma(
        items=items,
        collection_name=collection_name,
        persist_dir=persist_dir,
        company=company,
    )
    return f"Upserted {count} item(s) into Chroma collection '{collection_name}'."



# Testing

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch company news via Sonar and store in ChromaDB (NY timestamps + composite metadata key).")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--time-range", default="1m", choices=["24h", "1w", "1m", "3m", "1y"])
    parser.add_argument("--model", default="sonar-pro")
    parser.add_argument("--max-results", type=int, default=8)
    parser.add_argument("--collection", default="news_data")
    parser.add_argument("--persist-dir", default="./chroma")
    parser.add_argument("--company", default=None, help="Optional company name to include in metadata_key")
    args = parser.parse_args()

    msg = fetch_and_upsert_news(
        ticker=args.ticker,
        time_range=args.time_range,
        model=args.model,
        max_results=args.max_results,
        collection_name=args.collection,
        persist_dir=args.persist_dir,
        company=args.company,
    )
    print(f"✅ {msg}")
    