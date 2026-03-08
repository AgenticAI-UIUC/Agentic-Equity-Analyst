"""
competitor_agent_ML.py
──────────────────────
ML-driven competitor identification via KMeans clustering over the full
S&P 500 financial + market-behaviour feature space.

Key differences from competitor_agent.py (GICS rule-based)
───────────────────────────────────────────────────────────
  • No hard GICS filter — peers emerge from financial similarity, which
    can cross sub-industry boundaries (e.g. AMZN and MSFT clustering
    together on cloud-scale metrics despite different GICS labels).

  • Richer feature space:
      log_revenue    — business scale
      log_market_cap — market size
      revenue_growth — growth trajectory
      net_margin     — profitability
      pe_ratio       — valuation
      volatility_2y  — return uncertainty (annualised, 2Y weekly data)
      beta           — systematic / market-factor exposure

  • Cluster count k is chosen automatically via silhouette score search.

  • Composite ranking within the cluster reuses the same four-dimension
    scoring (rev_size, rev_growth, margin, corr_dist) as the rule-based
    agent so outputs are directly comparable.

Pipeline
────────
  1. Fetch Wikipedia S&P 500 table (one HTTP call).
  2. Load cached feature matrix, or build it (~5–10 min on first run):
       a. Parallel yfinance financials fetch for all ~500 tickers.
       b. Batch 2Y weekly price download → volatility + beta per ticker.
       c. Log-transform skewed columns; merge GICS labels from Wikipedia.
  3. Impute missing values (median per column), StandardScale.
  4. Silhouette-score search over k ∈ [15, 40] → fit best KMeans.
  5. Assign target to its cluster; collect cluster-mates as candidates.
  6. Market cap soft filter (same 0.05x–20x band, relaxed if < 3 pass).
  7. Compute 2Y weekly return correlations (target vs. filtered peers).
  8. Composite scoring → top 3-5 peers.
  9. Formatted output: target profile, cluster diagnostics, scored table.

Caching
───────
  Feature matrix is saved to CACHE_FILE (parquet) and reloaded on
  subsequent calls. Set CACHE_MAX_AGE_DAYS to control staleness.
  Call rebuild_cache() or pass --rebuild at the CLI to force refresh.
"""

import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from langchain.tools import tool
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CACHE_FILE         = "sp500_features_cache.parquet"
CACHE_MAX_AGE_DAYS = 7

# Silhouette search range — trades off cluster granularity vs. size.
# 500 stocks / k=25 → ~20 companies per cluster on average.
K_RANGE   = range(15, 41)
K_DEFAULT = 25          # fallback if silhouette search fails

TOP_N_PEERS     = 5
MARKET_CAP_BAND = (0.05, 20.0)   # relative to target

WEIGHTS = {
    "rev_size"  : 0.25,
    "rev_growth": 0.25,
    "margin"    : 0.25,
    "corr_dist" : 0.25,
}

# Columns fed into KMeans (must be present in the feature DataFrame)
CLUSTER_FEATURES = [
    "log_revenue",
    "log_market_cap",
    "revenue_growth",
    "net_margin",
    "pe_ratio",
    "volatility_2y",
    "beta",
]

# ─── S&P 500 UNIVERSE ────────────────────────────────────────────────────────

def get_sp500_universe() -> pd.DataFrame:
    """
    Fetch the S&P 500 constituent table from Wikipedia in one HTTP call.
    Returns DataFrame: Symbol, Security, GICS Sector, GICS Sub-Industry.
    Ticker dots → hyphens (BRK.B → BRK-B).
    """
    url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text
    table = pd.read_html(StringIO(html))[0]
    table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
    return table[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]


# ─── FEATURE BUILDING ────────────────────────────────────────────────────────

def _fetch_one_financials(ticker: str) -> Optional[Dict]:
    """
    Fetch financial features for a single ticker via yfinance.
    Returns None on any failure.
    """
    try:
        t      = yf.Ticker(ticker)
        info   = t.info
        income = t.income_stmt

        if income is None or income.empty:
            return None

        rev_series = None
        for label in ("Total Revenue", "Revenue"):
            if label in income.index:
                rev_series = income.loc[label].dropna()
                break
        if rev_series is None or len(rev_series) < 1:
            return None

        revenue = float(rev_series.iloc[0])

        revenue_growth = None
        if len(rev_series) >= 2 and rev_series.iloc[1] != 0:
            revenue_growth = (rev_series.iloc[0] - rev_series.iloc[1]) / abs(rev_series.iloc[1])

        net_margin = None
        for label in ("Net Income", "Net Income Common Stockholders"):
            if label in income.index:
                ni_series = income.loc[label].dropna()
                if len(ni_series) >= 1 and revenue != 0:
                    net_margin = float(ni_series.iloc[0]) / revenue
                break

        market_cap = info.get("marketCap")
        pe_raw     = info.get("trailingPE") or info.get("forwardPE")

        # Clip P/E to a sane range — extreme values (negative or >200) add noise
        pe_ratio = None
        if pe_raw is not None:
            pe_ratio = float(np.clip(pe_raw, 0.0, 200.0)) if pe_raw > 0 else None

        return {
            "revenue"       : revenue,
            "revenue_growth": revenue_growth,
            "net_margin"    : net_margin,
            "market_cap"    : market_cap,
            "pe_ratio"      : pe_ratio,
        }
    except Exception:
        return None


def _compute_price_features(tickers: List[str]) -> pd.DataFrame:
    """
    Batch-download 2Y weekly prices for all tickers + SPY.
    Returns DataFrame indexed by ticker with columns:
        volatility_2y  — annualised return volatility
        beta           — Cov(stock, SPY) / Var(SPY) over 2Y weekly data
    """
    print("  Downloading 2Y weekly prices for volatility + beta…")
    raw = yf.download(
        tickers + ["SPY"],
        period="2y", interval="1wk",
        progress=False, auto_adjust=True,
    )
    prices  = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    returns = prices.pct_change().dropna(how="all")

    spy_ret = returns["SPY"] if "SPY" in returns.columns else None
    spy_var = float(spy_ret.var()) if spy_ret is not None else None

    rows = {}
    for ticker in tickers:
        if ticker not in returns.columns:
            continue
        stk = returns[ticker].dropna()
        if len(stk) < 10:
            continue

        volatility = float(stk.std() * math.sqrt(52))  # annualised

        beta = None
        if spy_ret is not None and spy_var and spy_var > 0:
            pair = pd.concat([stk, spy_ret], axis=1).dropna()
            if len(pair) >= 10:
                cov  = float(pair.iloc[:, 0].cov(pair.iloc[:, 1]))
                beta = round(float(np.clip(cov / spy_var, -1.0, 4.0)), 4)

        rows[ticker] = {
            "volatility_2y": round(volatility, 4),
            "beta"         : beta,
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def build_feature_matrix(sp500_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix for all S&P 500 tickers.
    This is slow (~5–10 min). Results are cached in CACHE_FILE.

    Steps:
      1. Parallel financial fetch (20 threads).
      2. Batch price download → volatility + beta.
      3. Log-transform revenue and market cap.
      4. Merge GICS labels from Wikipedia table.
    """
    tickers = sp500_df["Symbol"].tolist()

    # ── 1. Parallel financials ───────────────────────────────────────────── #
    print(f"  Fetching financials for {len(tickers)} tickers (parallel, 20 workers)…")
    fin_rows: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch_one_financials, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            ticker = futures[fut]
            done  += 1
            if done % 100 == 0:
                print(f"    {done}/{len(tickers)} financials done…")
            try:
                result = fut.result()
                if result:
                    fin_rows[ticker] = result
            except Exception:
                pass

    fin_df = pd.DataFrame.from_dict(fin_rows, orient="index")
    fin_df.index.name = "Symbol"

    # ── 2. Price features ────────────────────────────────────────────────── #
    price_df = _compute_price_features(tickers)
    price_df.index.name = "Symbol"

    # ── 3. Merge & derive log features ──────────────────────────────────── #
    merged = fin_df.join(price_df, how="outer")

    # Only keep beta from price computation (yfinance beta can be stale)
    merged["log_revenue"]    = np.log(merged["revenue"].clip(lower=1.0))
    merged["log_market_cap"] = np.log(merged["market_cap"].clip(lower=1.0))

    # ── 4. Attach GICS labels and names from Wikipedia ───────────────────── #
    wiki = sp500_df.set_index("Symbol")[["Security", "GICS Sector", "GICS Sub-Industry"]]
    merged = merged.join(wiki, how="left")
    merged.rename(columns={
        "Security"          : "name",
        "GICS Sector"       : "gics_sector",
        "GICS Sub-Industry" : "gics_sub_industry",
    }, inplace=True)

    merged["fetched_at"] = datetime.now().isoformat()

    print(f"  Feature matrix built: {len(merged)} rows, {merged.shape[1]} columns.")
    return merged


def load_or_build_features(sp500_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the cached feature matrix if it exists and is fresh.
    Otherwise build it from scratch and save.
    """
    if os.path.exists(CACHE_FILE):
        df = pd.read_parquet(CACHE_FILE)
        if "fetched_at" in df.columns:
            cache_time = pd.to_datetime(df["fetched_at"].iloc[0])
            age_days   = (datetime.now() - cache_time.to_pydatetime().replace(tzinfo=None)).days
            if age_days < CACHE_MAX_AGE_DAYS:
                print(f"[cache] Loaded {len(df)} rows from {CACHE_FILE}  (age: {age_days}d)")
                return df
            print(f"[cache] Cache is {age_days} days old (> {CACHE_MAX_AGE_DAYS}). Rebuilding…")
        else:
            print("[cache] Cache has no timestamp. Rebuilding…")
    else:
        print(f"[cache] {CACHE_FILE} not found. Building feature matrix (first run)…")

    df = build_feature_matrix(sp500_df)
    df.to_parquet(CACHE_FILE)
    print(f"[cache] Saved to {CACHE_FILE}.")
    return df


def rebuild_cache() -> pd.DataFrame:
    """Force a full rebuild of the feature matrix cache."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    sp500_df = get_sp500_universe()
    return load_or_build_features(sp500_df)


# ─── CLUSTERING ──────────────────────────────────────────────────────────────

def _prepare_cluster_matrix(
    feature_df: pd.DataFrame,
) -> Tuple[np.ndarray, pd.Index, SimpleImputer, StandardScaler]:
    """
    Select CLUSTER_FEATURES, drop rows with >50% missing, impute (median),
    and StandardScale. Returns (X, valid_index, imputer, scaler).
    """
    sub = feature_df[CLUSTER_FEATURES].copy()

    # Drop companies with more than half the cluster features missing
    threshold = math.ceil(len(CLUSTER_FEATURES) / 2)
    sub = sub.dropna(thresh=threshold)

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_imp    = imputer.fit_transform(sub)
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, sub.index, imputer, scaler


def find_optimal_k(X: np.ndarray) -> int:
    """
    Search K_RANGE for the k that maximises average silhouette score.
    Falls back to K_DEFAULT on any error.

    Silhouette score ∈ [-1, 1]:  higher = more cohesive, well-separated clusters.
    We sample up to 400 points for speed (full dataset only has ~500 anyway).
    """
    best_k, best_score = K_DEFAULT, -1.0

    for k in K_RANGE:
        if k >= X.shape[0]:
            break
        try:
            km     = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = km.fit_predict(X)
            score  = silhouette_score(X, labels, sample_size=min(400, X.shape[0]))
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            pass

    return best_k


# ─── COMPOSITE SCORING (within-cluster ranking) ───────────────────────────────

def _get_price_correlation(
    target_ticker: str, peer_tickers: List[str]
) -> Dict[str, float]:
    """2Y weekly Pearson correlation between target and each peer."""
    corr_map: Dict[str, float] = {}
    try:
        raw = yf.download(
            [target_ticker] + peer_tickers,
            period="2y", interval="1wk",
            progress=False, auto_adjust=True,
        )
        prices  = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        returns = prices.pct_change().dropna(how="all")

        if target_ticker not in returns.columns:
            return corr_map

        for peer in peer_tickers:
            if peer in returns.columns:
                pair = returns[[target_ticker, peer]].dropna()
                if len(pair) >= 10:
                    c = pair[target_ticker].corr(pair[peer])
                    if not math.isnan(c):
                        corr_map[peer] = round(float(c), 4)
    except Exception:
        pass
    return corr_map


def _minmax(values: List[Optional[float]]) -> List[float]:
    """Min-max normalise to [0, 1]; None → 0.5 (neutral)."""
    valid = [v for v in values if v is not None]
    if not valid:
        return [0.5] * len(values)
    mn, mx = min(valid), max(valid)
    if mn == mx:
        return [0.5] * len(values)
    return [(0.5 if v is None else (v - mn) / (mx - mn)) for v in values]


def _rank_peers(
    target: pd.Series,
    peers: pd.DataFrame,
    corr_map: Dict[str, float],
) -> List[Dict]:
    """
    Composite similarity scoring using cached financial data.
    Works directly from the feature DataFrame rows (no re-fetch needed).

    Returns list of dicts sorted by composite_score ascending.
    """
    raw: Dict[str, List] = {k: [] for k in WEIGHTS}

    tr = target.get("revenue")
    tg = target.get("revenue_growth")
    tm = target.get("net_margin")

    for ticker, p in peers.iterrows():
        pr = p.get("revenue")
        raw["rev_size"].append(
            abs(math.log(pr / tr)) if tr and pr and tr > 0 and pr > 0 else None
        )

        pg = p.get("revenue_growth")
        raw["rev_growth"].append(
            abs(pg - tg) if tg is not None and pg is not None else None
        )

        pm = p.get("net_margin")
        raw["margin"].append(
            abs(pm - tm) if tm is not None and pm is not None else None
        )

        raw["corr_dist"].append(1.0 - corr_map.get(ticker, 0.0))

    norm = {dim: _minmax(raw[dim]) for dim in raw}

    scored = []
    for i, (ticker, p) in enumerate(peers.iterrows()):
        score = sum(WEIGHTS[dim] * norm[dim][i] for dim in WEIGHTS)
        scored.append({
            "ticker"           : ticker,
            "name"             : p.get("name", ticker),
            "gics_sector"      : p.get("gics_sector"),
            "gics_sub_industry": p.get("gics_sub_industry"),
            "market_cap"       : p.get("market_cap"),
            "revenue"          : p.get("revenue"),
            "revenue_growth"   : p.get("revenue_growth"),
            "net_margin"       : p.get("net_margin"),
            "pe_ratio"         : p.get("pe_ratio"),
            "composite_score"  : round(score, 4),
            "corr_2y"          : corr_map.get(ticker),
            # Normalised component scores
            "_s_rev_size"      : round(norm["rev_size"][i], 3),
            "_s_rev_growth"    : round(norm["rev_growth"][i], 3),
            "_s_margin"        : round(norm["margin"][i], 3),
            "_s_corr_dist"     : round(norm["corr_dist"][i], 3),
        })

    return sorted(scored, key=lambda x: x["composite_score"])


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def _run_pipeline_ml(ticker: str):
    """
    Shared pipeline. Returns an error string or a result dict.
    """
    ticker = ticker.upper().strip()

    # ── 1. S&P 500 universe ──────────────────────────────────────────────── #
    sp500_df = get_sp500_universe()

    # ── 2. Load / build feature matrix ───────────────────────────────────── #
    feature_df = load_or_build_features(sp500_df)

    if ticker not in feature_df.index:
        return (
            f"[ERROR] {ticker} not found in S&P 500 feature cache. "
            f"Run rebuild_cache() if this is a recent addition."
        )

    # ── 3. Prepare cluster matrix ─────────────────────────────────────────── #
    X, valid_idx, imputer, scaler = _prepare_cluster_matrix(feature_df)

    if ticker not in valid_idx:
        return (
            f"[ERROR] {ticker} was excluded from clustering "
            f"(too many missing features). Check the cache."
        )

    # ── 4. Silhouette search → fit KMeans ─────────────────────────────────── #
    print(f"[ML] Searching optimal k in {list(K_RANGE)[0]}–{list(K_RANGE)[-1]}…")
    best_k      = find_optimal_k(X)
    km          = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    labels      = km.fit_predict(X)
    cluster_ser = pd.Series(labels, index=valid_idx, name="cluster")

    # ── 5. Target's cluster ───────────────────────────────────────────────── #
    target_cluster   = int(cluster_ser[ticker])
    cluster_members  = cluster_ser[cluster_ser == target_cluster].index.tolist()
    peers_in_cluster = [t for t in cluster_members if t != ticker]

    if not peers_in_cluster:
        return f"[ERROR] Cluster for {ticker} has no other members."

    # ── 6. Market cap soft filter ─────────────────────────────────────────── #
    target_row = feature_df.loc[ticker]
    mkt_cap    = target_row.get("market_cap")

    if not mkt_cap:
        return f"[ERROR] Could not determine market cap for {ticker}."

    low, high   = mkt_cap * MARKET_CAP_BAND[0], mkt_cap * MARKET_CAP_BAND[1]
    peers_df    = feature_df.loc[peers_in_cluster]
    cap_pass    = peers_df[
        peers_df["market_cap"].notna()
        & (peers_df["market_cap"] >= low)
        & (peers_df["market_cap"] <= high)
    ]
    filtered_df = cap_pass if len(cap_pass) >= 3 else peers_df
    cap_relaxed = len(cap_pass) < 3

    # ── 7. 2Y price correlation ───────────────────────────────────────────── #
    print(f"[ML] Computing 2Y correlations for {len(filtered_df)} cluster-mates…")
    corr_map = _get_price_correlation(ticker, filtered_df.index.tolist())

    # ── 8. Composite rank ─────────────────────────────────────────────────── #
    ranked    = _rank_peers(target_row, filtered_df, corr_map)
    top_peers = ranked[:TOP_N_PEERS]

    # ── 9. Cluster GICS diversity (how cross-industry is this cluster?) ──── #
    cluster_gics = feature_df.loc[cluster_members, "gics_sub_industry"].dropna()
    n_sub_industries = cluster_gics.nunique()
    n_sectors        = feature_df.loc[cluster_members, "gics_sector"].dropna().nunique()

    return {
        "ticker"           : ticker,
        "target_row"       : target_row,
        "best_k"           : best_k,
        "target_cluster"   : target_cluster,
        "cluster_size"     : len(cluster_members),
        "n_sub_industries" : n_sub_industries,
        "n_sectors"        : n_sectors,
        "cluster_members"  : cluster_members,
        "filtered_df"      : filtered_df,
        "cap_relaxed"      : cap_relaxed,
        "top_peers"        : top_peers,
    }


# ─── FORMATTED OUTPUT ────────────────────────────────────────────────────────

def _fmt_b(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"${float(val) / 1e9:,.1f}B"

def _fmt_pct(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{float(val) * 100:+.1f}%"


def find_competitors_ml(ticker: str) -> str:
    """
    Full ML pipeline with formatted output including cluster diagnostics
    and composite score breakdown per peer.
    """
    result = _run_pipeline_ml(ticker)
    if isinstance(result, str):   # error message
        return result

    tr  = result["target_row"]
    top = result["top_peers"]

    W     = 76
    dline = "═" * W
    line  = "─" * W

    def _row(*cols, widths):
        return "  " + "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    lines = [
        dline,
        f"  ML COMPETITOR ANALYSIS: {result['ticker']}  ({tr.get('name', '')})",
        dline,
        f"  GICS Sub-Industry : {tr.get('gics_sub_industry', '—')}",
        f"  GICS Sector       : {tr.get('gics_sector', '—')}",
        f"  Market Cap        : {_fmt_b(tr.get('market_cap'))}",
        f"  Revenue (TTM)     : {_fmt_b(tr.get('revenue'))}",
        f"  Rev Growth (YoY)  : {_fmt_pct(tr.get('revenue_growth'))}",
        f"  Net Margin        : {_fmt_pct(tr.get('net_margin'))}",
        f"  Beta (2Y weekly)  : {tr.get('beta', '—') if tr.get('beta') is not None else '—'}",
        f"  Volatility (ann.) : {_fmt_pct(tr.get('volatility_2y'))}",
        line,
        f"  KMeans k          : {result['best_k']}  (chosen by silhouette score)",
        f"  Cluster #         : {result['target_cluster']}",
        f"  Cluster size      : {result['cluster_size']} companies",
        f"  GICS diversity    : {result['n_sub_industries']} sub-industries"
        f" across {result['n_sectors']} sectors"
        + ("  ← cross-industry peers present" if result["n_sectors"] > 1 else ""),
        f"  After cap filter  : {len(result['filtered_df'])} companies"
        + ("  [relaxed]" if result["cap_relaxed"] else ""),
        line,
        "  Top Peers by Composite Similarity Score  (lower = more similar)",
        "",
        "  Score breakdown (each column normalised 0–1 across cluster peers):",
        "    RevSize  = |log(peer_rev / target_rev)|    revenue scale match",
        "    RevGrwth = |peer_YoY − target_YoY|         growth trajectory match",
        "    Margin   = |peer_margin − target_margin|   profitability match",
        "    CorrDist = 1 − Pearson(2Y weekly returns)  price co-movement match",
        "    Score    = equal-weighted average of all four (25% each)",
        line,
    ]

    # Table
    cols   = ["Rank", "Ticker", "Name", "Sector/Sub-Industry", "Score",
              "RevSize", "RevGrwth", "Margin", "CorrDist", "2Y Corr"]
    widths = [4, 7, 22, 30, 6, 7, 8, 7, 8, 7]
    lines.append(_row(*cols, widths=widths))
    lines.append("  " + "─" * (sum(widths) + 2 * len(widths)))

    for rank, p in enumerate(top, 1):
        gics_label = p.get("gics_sub_industry") or p.get("gics_sector") or "—"
        corr       = p.get("corr_2y")
        row = [
            rank,
            p["ticker"],
            (p.get("name") or "")[:21],
            gics_label[:29],
            p["composite_score"],
            p["_s_rev_size"],
            p["_s_rev_growth"],
            p["_s_margin"],
            p["_s_corr_dist"],
            f"{corr:.3f}" if corr is not None else "—",
        ]
        lines.append(_row(*row, widths=widths))

    lines.append(dline)
    return "\n".join(lines)


# ─── LANGCHAIN TOOL ───────────────────────────────────────────────────────────

@tool
def competitor_tool_ml(ticker: str) -> str:
    """
    Identifies the top 3-5 most similar S&P 500 competitors using ML clustering.

    Clusters all S&P 500 companies by financial + market-behaviour features
    (revenue, growth, margin, market cap, P/E, volatility, beta), then ranks
    the target's cluster-mates by composite similarity score.

    Advantage over GICS-based approach: can surface cross-industry competitors
    when the financials warrant it (e.g. AMZN and MSFT on cloud scale).

    Requires cached feature matrix (built automatically on first run, ~5–10 min).

    Returns competitor ticker symbols only, one per line.
    Takes one string argument: the stock ticker symbol (e.g., 'AAPL', 'NVDA').
    """
    result = _run_pipeline_ml(ticker)
    if isinstance(result, str):
        return result
    return "\n".join(p["ticker"] for p in result["top_peers"])


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if "--rebuild" in args:
        print("Forcing cache rebuild…")
        rebuild_cache()
        args = [a for a in args if a != "--rebuild"]

    ticker_arg = args[0] if args else "NVDA"
    print(find_competitors_ml(ticker_arg))
