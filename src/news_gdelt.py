"""
GDELT 2.0 DOC API news fetcher for Anomalyy.

Free, no API key, no auth. Indexes global news at 15-min granularity since
Feb 18, 2015. Replaces NewsAPI (which was 30-day free-tier capped) so the
sentiment-spike anomaly classifier works across the full 10-year window.

Returns dicts shape-compatible with NewsAPI articles, so the existing
`DataIngestion._process_and_store_news` (VADER scoring) consumes them unchanged.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT throttles unidentified `python-requests/X.Y.Z` clients harder. A real
# UA + contact email is the polite ask the docs recommend.
GDELT_HEADERS = {
    'User-Agent': 'Anomalyy/1.0 (CS210 student project; purabhsingh123@gmail.com)',
}

# Per-ticker keyword queries. Multi-keyword to avoid generic-name noise
# (e.g. "Apple" alone returns fruit/orchard hits).
TICKER_TO_QUERY = {
    'AAPL': '("Apple Inc" OR "AAPL")',
    'MSFT': '("Microsoft" OR "MSFT")',
    'TSLA': '("Tesla Inc" OR "TSLA")',
    'AMZN': '("Amazon.com" OR "AMZN")',
    '^GSPC': '("S&P 500" OR "SPX")',
}

# Major financial-news domains to suppress off-topic hits.
DEFAULT_DOMAINS = [
    'reuters.com', 'wsj.com', 'bloomberg.com', 'ft.com',
    'cnbc.com', 'marketwatch.com', 'barrons.com', 'forbes.com',
    'businessinsider.com', 'seekingalpha.com', 'finance.yahoo.com',
    'investors.com', 'fool.com', 'thestreet.com',
]

# GDELT 2.0 DOC index begins Feb 18, 2015. Earlier requests return empty.
GDELT_EARLIEST = datetime(2015, 2, 18)


def _build_query(keyword_query: str, domains: Optional[List[str]] = None) -> str:
    if not domains:
        return keyword_query
    domain_clause = ' OR '.join(f'domain:{d}' for d in domains)
    return f'{keyword_query} ({domain_clause})'


def _parse_seendate(s: str) -> str:
    try:
        dt = datetime.strptime(s, '%Y%m%dT%H%M%SZ')
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except (ValueError, TypeError):
        return s or ''


def _to_newsapi_shape(article: Dict) -> Dict:
    return {
        'title': article.get('title', ''),
        'description': '',  # GDELT DOC API doesn't carry article body/description
        'source': {'name': article.get('domain', 'unknown')},
        'url': article.get('url', ''),
        'publishedAt': _parse_seendate(article.get('seendate', '')),
    }


def _fetch_chunk(query: str, start: datetime, end: datetime,
                 max_records: int, timeout: float,
                 max_retries: int = 2, base_backoff: float = 10.0) -> List[Dict]:
    """
    One GDELT request with exponential backoff. GDELT's free DOC API rate-limits
    at roughly 1 req per 5s; bursts return 429 or HTML error pages instead of JSON.
    Retry on 429, 5xx, connection-reset, and non-JSON; give up cleanly otherwise.

    Tuned to fail-fast when persistently throttled (max 2 retries = ~30s wasted
    per chunk worst-case) so the pipeline doesn't stall for hours when GDELT's
    free service has IP-blocked us.
    """
    params = {
        'query': query,
        'mode': 'ArtList',
        'format': 'json',
        'maxrecords': max_records,
        'startdatetime': start.strftime('%Y%m%d%H%M%S'),
        'enddatetime': end.strftime('%Y%m%d%H%M%S'),
        'sort': 'DateDesc',
    }
    label = f"{start.date()}-{end.date()}"
    for attempt in range(max_retries):
        try:
            r = requests.get(GDELT_DOC_URL, params=params, headers=GDELT_HEADERS, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                wait = base_backoff * (2 ** attempt)
                logger.warning(f"GDELT HTTP {r.status_code} for {label}; backing off {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            try:
                data = r.json()
            except ValueError:
                wait = base_backoff * (2 ** attempt)
                logger.warning(f"GDELT non-JSON for {label} (likely throttled); backing off {wait:.0f}s")
                time.sleep(wait)
                continue
            return data.get('articles', []) or []
        except requests.HTTPError as e:
            logger.warning(f"GDELT HTTP {e.response.status_code} for {label} (no retry)")
            return []
        except requests.RequestException as e:
            wait = base_backoff * (2 ** attempt)
            logger.warning(f"GDELT request failed for {label}: {e}; backing off {wait:.0f}s")
            time.sleep(wait)
            continue
    logger.warning(f"GDELT giving up on {label} after {max_retries} retries")
    return []


def fetch_gdelt_articles(
    keyword_query: str,
    start: str,
    end: str,
    domains: Optional[List[str]] = None,
    chunk_days: int = 180,
    sleep_between: float = 5.0,
    max_per_chunk: int = 250,
    timeout: float = 30.0,
) -> List[Dict]:
    """
    Fetch news articles from GDELT 2.0 DOC API across [start, end].

    Loops in `chunk_days` windows because GDELT caps each request at 250
    articles and works best with shorter ranges. Returns NewsAPI-shape dicts
    so `_process_and_store_news` consumes them unchanged.

    Args:
        keyword_query: GDELT query, e.g. '("Apple Inc" OR "AAPL")'
        start, end:    'YYYY-MM-DD' inclusive bounds
        domains:       optional whitelist (defaults to major financial sites)
        chunk_days:    request window size
        sleep_between: pause between requests (be nice to the free service)
        max_per_chunk: maxrecords per request (GDELT caps at 250)
        timeout:       per-request HTTP timeout
    """
    # Domain whitelist intentionally not applied by default: GDELT's free DOC API
    # rejects long OR-clauses (we measured 14 domains failing). The quoted keyword
    # query ('"Apple Inc" OR "AAPL"') is already specific enough.
    query = _build_query(keyword_query, domains)
    start_dt = max(datetime.strptime(start, '%Y-%m-%d'), GDELT_EARLIEST)
    end_dt = datetime.strptime(end, '%Y-%m-%d')

    if start_dt >= end_dt:
        logger.warning(
            f"GDELT: empty window {start} to {end} after clamping to "
            f">= {GDELT_EARLIEST.date()}"
        )
        return []

    all_articles: List[Dict] = []
    seen_urls = set()
    cursor = start_dt

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        logger.info(f"GDELT fetch {cursor.date()} -> {chunk_end.date()}")
        chunk = _fetch_chunk(query, cursor, chunk_end, max_per_chunk, timeout)
        for article in chunk:
            url = article.get('url')
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            all_articles.append(_to_newsapi_shape(article))
        cursor = chunk_end
        time.sleep(sleep_between)

    logger.info(f"GDELT total: {len(all_articles)} unique articles")
    return all_articles


def fetch_for_ticker(ticker: str, start: str, end: str, **kwargs) -> List[Dict]:
    """Look up the per-ticker query and fetch."""
    query = TICKER_TO_QUERY.get(ticker)
    if not query:
        logger.warning(f"No GDELT query mapping for {ticker}; skipping")
        return []
    return fetch_gdelt_articles(query, start, end, **kwargs)
