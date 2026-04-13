import uuid, os, pytz, datetime, math
import numpy as np
from dotenv import load_dotenv
from typing import Dict, Any
import yfinance as yf
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from datetime import date, timedelta, timezone
from openai import OpenAI
from langchain.tools import tool 

BATCH = 300

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

_COLLECTION = None

def get_financial_collection():
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION
    
    db = os.getenv("CHROMADB")
    api_key = os.getenv("CHROMADB_API_KEY")
    tenant = os.getenv("CHROMADB_TENANT")
    
    if not all([db, api_key, tenant]):
        return None

    from langchain_chroma import Chroma
    _COLLECTION = Chroma(
        database=db,
        collection_name="financial_data",
        embedding_function=embeddings,
        chroma_cloud_api_key=api_key,
        tenant=tenant,
    )
    return _COLLECTION


def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

ny_tz = pytz.timezone("America/New_York")

def get_daily_yf(company: str, symbol: str):
    ticker = yf.Ticker(symbol)
    data = ticker.history(start= date.today() - timedelta(days=1), end=date.today(),
        interval="1m",
        auto_adjust=True,
        prepost=True
    )

    texts = []
    metadatas = []
    ids = []

    for ts_utc, row in data.iterrows():
        text = (
            f"timestamp_utc={ts_utc.isoformat()} "
            f"Open={row['Open']} High={row['High']} Low={row['Low']} Close={row['Close']} "
            f"Volume={row['Volume']} Dividends={row.get('Dividends', 0.0)} "
            f"StockSplits={row.get('Stock Splits', 0.0)}"
        )
        texts.append(text)

        ts_ny = datetime.datetime.now(ny_tz)
        meta = {
            "symbol": symbol,
            "company": company,
            "date_retrieved": date.today().isoformat(),
            "time_retrieved": datetime.datetime.now(timezone.utc).astimezone(ny_tz).strftime("%H:%M:%S"),
            "date_published": ts_ny.date().isoformat(), 
            "time_published": ts_ny.strftime("%H:%M:%S"),    
            "source": "yf",
            "url": "url",
        }
        metadatas.append(meta)
        ids.append(str(uuid.uuid4()))

    collection = get_financial_collection()
    if not collection:
        logging.error("ChromaDB not configured. Cannot add texts.")
        return

    for t_chunk, m_chunk, id_chunk in zip(chunked(texts, BATCH), chunked(metadatas, BATCH), chunked(ids, BATCH)):
        collection.add_texts(texts=t_chunk, metadatas=m_chunk, ids=id_chunk)

@tool
def get_daily_yf_tool(company: str, symbol: str):
    """
    Get financial ticker data for a given company within the last day
    Takes company name and its ticker as the arguments
    """
    return get_daily_yf(company, symbol)

@tool
def calculate_moving_average_tool(ticker: str, days: int = 365) -> str:
    """
    Calculate the moving average for a given stock ticker over a specified number of days.
    Defaults to 365 days if not specified.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
        days: Number of days for moving average calculation (default: 365)

    Returns:
        Formatted string with the moving average value
    """
    try:
        stock = yf.Ticker(ticker)
        end_date = date.today()
        # Add buffer days to account for weekends/holidays
        start_date = end_date - timedelta(days=days + 100)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return f"Error: No data found for ticker {ticker}"

        if len(hist) < days:
            actual_days = len(hist)
            moving_avg = hist['Close'].mean()
            return f"Warning: Only {actual_days} days of data available. Moving average for period of last {actual_days} days: ${moving_avg:.2f}"

        # Calculate moving average using the most recent 'days' closing prices
        moving_avg = hist['Close'].tail(days).mean()

        return f"Moving average for period of last {days} days: ${moving_avg:.2f}"

    except Exception as e:
        return f"Error calculating moving average for {ticker}: {str(e)}"

def get_normalized_technical_score(ticker: str) -> Dict[str, Any]:
    """
    Returns a normalized technical score (0.0-1.0) based on multi-horizon Z-scores.
    Uses 50d, 200d, and 365d moving averages blended: (0.5, 0.3, 0.2).
    """
    try:
        import yfinance as yf
        from datetime import date, timedelta
        stock = yf.Ticker(ticker)
        
        info = stock.info
        current_price = info.get("currentPrice")
        if not current_price:
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
            else:
                return {"score": 0.5, "confidence": 0.0, "error": "No price data"}
        
        # Get history for MA and StdDev calculations
        end_date = date.today()
        start_date = end_date - timedelta(days=500)
        full_hist = stock.history(start=start_date, end=end_date)
        
        if full_hist.empty or len(full_hist) < 365:
             return {
                 "score": 0.5, 
                 "confidence": 0.0, 
                 "error": f"Insufficient history (need 365 days, got {len(full_hist)})"
             }

        prices = full_hist['Close']
        
        def calc_z_sigmoid(window: int):
            if len(prices) < window: return 0.5, 0.0
            subset = prices.tail(window)
            ma = subset.mean()
            std = subset.std()
            if std == 0: return 0.5, 0.0
            z = (current_price - ma) / std
            # Map Z to (0,1) using Sigmoid
            normalized = 1 / (1 + math.exp(-z))
            return normalized, z, ma, std

        # Calculate for each horizon
        s_norm, s_z, s_ma, s_std = calc_z_sigmoid(50)
        m_norm, m_z, m_ma, m_std = calc_z_sigmoid(200)
        l_norm, l_z, l_ma, l_std = calc_z_sigmoid(365)
        
        # Weighted Composite Score
        composite_score = (0.5 * s_norm) + (0.3 * m_norm) + (0.2 * l_norm)
        
        # Volatility & Confidence
        # Volatility as coefficient of variation for the long horizon
        volatility = l_std / l_ma if l_ma != 0 else 0.5
        trend_strength = abs(l_z)
        
        # confidence = min(1.0, trend_strength / 3) * (1 - volatility)
        # We cap confidence but allow it to scale with deviation
        confidence = min(1.0, trend_strength / 3.0) * (1.0 - min(volatility * 2, 0.7))
        
        return {
            "score": round(composite_score, 3),
            "confidence": round(confidence, 3),
            "horizons": {
                "50d": {"score": round(s_norm, 3), "z": round(s_z, 3)},
                "200d": {"score": round(m_norm, 3), "z": round(m_z, 3)},
                "365d": {"score": round(l_norm, 3), "z": round(l_z, 3)}
            },
            "volatility": round(volatility, 3),
            "current_price": round(float(current_price), 2)
        }
    except Exception as e:
        return {"score": 0.5, "confidence": 0.0, "error": str(e)}
