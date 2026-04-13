import logging
from typing import Dict, Any, List
import numpy as np

from social_sentiment_loader import get_normalized_sentiment_score
from dcf import get_normalized_valuation_score
from market_data_loader import get_normalized_technical_score
from analyst_ratings_loader import get_normalized_fundamental_score

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class SynthesisNode:
    def __init__(self):
        # Weights normalized to sum to 1.0
        self.weights = {
            "fundamentals": 0.40,
            "valuation": 0.30,
            "technicals": 0.15,
            "sentiment": 0.15
        }

    def calculate_synthesis(self, ticker: str) -> Dict[str, Any]:
        """
        Orchestrates the fetching, normalization, and weighted synthesis of signals.
        """
        ticker = ticker.upper()
        logging.info(f"Starting Weighted Signal Synthesis for {ticker}...")

        # 1. Fetch normalized scores
        signals = {
            "fundamentals": get_normalized_fundamental_score(ticker),
            "valuation": get_normalized_valuation_score(ticker),
            "technicals": get_normalized_technical_score(ticker),
            "sentiment": get_normalized_sentiment_score(ticker)
        }

        # 2. Weighted Calculation
        weighted_score = 0.0
        total_confidence = 0.0
        scores_list = []
        
        for key, weight in self.weights.items():
            sig = signals[key]
            score = sig["score"]
            conf = sig["confidence"]
            
            weighted_score += score * weight
            total_confidence += conf * weight
            scores_list.append(score)

        # 3. Conflict Resolution & Disagreement Map
        # Calculate max deviation and variance
        scores_array = np.array(scores_list)
        weights_array = np.array([self.weights[k] for k in signals.keys()])
        
        mean_score = weighted_score # Since weights sum to 1
        variance = np.average((scores_array - mean_score)**2, weights=weights_array)
        max_dev = np.max(np.abs(scores_array - mean_score))
        
        # Identify high conflict sources (deviation > 0.25)
        high_conflict = [
            k for k, s in signals.items() 
            if abs(s["score"] - mean_score) > 0.25
        ]

        # 4. Final Final Score & Confidence Adjustment
        # Confidence is penalized by variance
        # A variance of 0.25 (e.g., half 0 and half 1) would heavily penalize confidence
        final_confidence = total_confidence * (1 - min(variance * 4, 0.8))

        # 5. Rating Label
        rating = self._get_rating_label(weighted_score)

        return {
            "ticker": ticker,
            "final_score": round(weighted_score, 3),
            "rating": rating,
            "confidence": round(final_confidence, 3),
            "signal_breakdown": {k: v["score"] for k, v in signals.items()},
            "disagreement_map": {
                "max_deviation": round(float(max_dev), 3),
                "weighted_variance": round(float(variance), 3),
                "high_conflict_sources": high_conflict
            },
            "rationale": self._generate_rationale(ticker, rating, weighted_score, final_confidence, high_conflict, signals)
        }

    def _get_rating_label(self, score: float) -> str:
        if score >= 0.8: return "Strong Buy"
        if score >= 0.6: return "Buy"
        if score >= 0.4: return "Hold"
        if score >= 0.2: return "Sell"
        return "Strong Sell"

    def _generate_rationale(self, ticker, rating, score, confidence, high_conflict, signals) -> str:
        summary = f"The synthesis node issued a {rating} rating for {ticker} with a conviction score of {score:.2f}."
        
        if confidence < 0.4:
            summary += " Note: Confidence is LOW due to significant disagreement between data sources."
        elif confidence > 0.7:
            summary += " Confidence is HIGH as core signals are broadly aligned."
            
        if high_conflict:
            sources = ", ".join(high_conflict)
            summary += f" Key conflict(s) detected in: {sources}."
            
        # Add a bit of detail on primary driver
        primary = max(self.weights, key=lambda k: self.weights[k])
        summary += f" Fundamentals (weighted {self.weights[primary]*100:.0f}%) remained the primary anchor for this evaluation."
        
        return summary

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    
    node = SynthesisNode()
    result = node.calculate_synthesis(args.ticker)
    print(json.dumps(result, indent=2))
