import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def rating_to_label(score):
    """
    Convert Yahoo numeric rating to label.
    1.0 = Strong Buy
    2.0 = Buy
    3.0 = Hold
    4.0 = Sell
    5.0 = Strong Sell
    """

    if score is None:
        return "Unknown"

    if score <= 1.5:
        return "Strong Buy"
    elif score <= 2.5:
        return "Buy"
    elif score <= 3.5:
        return "Hold"
    elif score <= 4.5:
        return "Sell"
    else:
        return "Strong Sell"


def get_recent_changes(recommendations, days=100):
    """
    Extract recent upgrades/downgrades.
    """
    if recommendations is None or recommendations.empty:
        return []

    if not isinstance(recommendations.index, pd.DatetimeIndex):
        try:
            now = pd.Timestamp.now()
            recommendations.index = recommendations.index.map(lambda i: now - pd.DateOffset(months=int(i)))

        except:
            return []
        
    cutoff = datetime.now() - timedelta(days=days)
    recent = recommendations[
        recommendations.index >= cutoff
    ]

    changes = []

    # Need to implement

    return changes


def load_analyst_ratings(ticker):
    """
    Main function to load analyst ratings.
    """

    stock = yf.Ticker(ticker)

    info = stock.info
    recs = stock.recommendations

    try:
        consensus = info.get("recommendationMean")
        num_analysts = info.get("numberOfAnalystOpinions")

        data = {
            "ticker": ticker,
            "consensus_rating": consensus,
            "rating_label": rating_to_label(consensus),
            "price_target_avg": info.get("targetMeanPrice"),
            "price_target_high": info.get("targetHighPrice"),
            "price_target_low": info.get("targetLowPrice"),
            "num_analysts": num_analysts,
            "recent_changes": get_recent_changes(recs)
        }

        return data

    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e)
        }
    
data = load_analyst_ratings("NFLX")

print(data)