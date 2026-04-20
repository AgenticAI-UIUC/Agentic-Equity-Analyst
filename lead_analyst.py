"""
Lead Analyst — Sequential LangGraph Orchestration
══════════════════════════════════════════════════

PROBLEM THIS SOLVES
───────────────────
The previous architecture funneled all 12 tools into a single ReAct agent (gpt-4o).
With no specialist boundaries, the agent could duplicate searches, skip sections, or
produce unbalanced reports depending on how the model chose to order its tool calls.

SOLUTION: SEQUENTIAL STATE GRAPH
──────────────────────────────────
A LangGraph StateGraph runs six specialist nodes in a fixed sequence. Each node:
  1. Calls only its designated tools (no overlap with other specialists)
  2. Synthesizes raw tool output into findings via a focused LLM call
  3. Writes those findings into the shared AnalysisState
  4. Passes state to the next node — which can read earlier findings for context

This guarantees complete coverage with no duplicated work.

FLOW
────
  START
    │
    ▼
  [fundamental_node]   ← analyze_filings, analyze_parser, analyze_financials
    │  writes: state["fundamental_findings"]
    ▼
  [technical_node]     ← calculate_rsi_tool, calculate_moving_average_tool,
    │                     calculate_trend_regime_tool, calculate_atr_tool
    │  reads:  fundamental_findings (price anchoring only)
    │  writes: state["technical_findings"]
    ▼
  [valuation_node]     ← valuation_tool, find_dcf_tool
    │  reads:  fundamental_findings (validates DCF assumptions)
    │  writes: state["valuation_findings"]
    ▼
  [market_intel_node]  ← analyze_news, analyze_social_sentiment_tool
    │  writes: state["market_intel_findings"]
    ▼
  [competitor_node]    ← advanced_comp_analysis (SWOT, metrics, white space)
    │  reads:  fundamental_findings (relative comparison anchor)
    │  writes: state["competitor_findings"]
    ▼
  [divergence_node]    ← analyze_divergence_tool
    │  reads:  technical_findings + valuation_findings (cross-reference)
    │  writes: state["divergence_findings"]
    ▼
  [synthesis_node]     ← reads ALL six findings, calls LLM once, no tools
    │  writes: state["final_report"]
    ▼
  END

KEY DESIGN DECISIONS
─────────────────────
- Each node has a tight system prompt with explicit "Do NOT" exclusions — this
  is the critical guard against duplication and role confusion.
- Direct tool calls (not a sub-agent loop) inside each node: deterministic,
  faster, and easier to debug than nested ReAct loops.
- Synthesis node uses no tools — Lead Analyst integrates, does not re-research.
- Competitor node runs after valuation so the SWOT has a margin/multiple
  baseline to anchor relative comparisons.
- Divergence node runs last (before synthesis) so it can reference both
  technical and valuation findings that are already in state.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from advanced_comp_analysis import advanced_comp_analysis
from analyst import (
    analyze_filings,
    analyze_financials,
    analyze_news,
    analyze_parser,
    analyze_social_sentiment_tool,
)
from dcf import find_dcf_tool
from divergence_analyzer import analyze_divergence_tool
from market_data_loader import (
    calculate_atr_tool,
    calculate_moving_average_tool,
    calculate_rsi_tool,
    calculate_trend_regime_tool,
)
from valuation_agent import valuation_tool

LLM = ChatOpenAI(model="gpt-4o", temperature=0.2, timeout=60)


# ── Shared state ───────────────────────────────────────────────────────────────

class AnalysisState(TypedDict):
    company: str
    ticker: str
    year: str
    fundamental_findings: str
    technical_findings: str
    valuation_findings: str
    market_intel_findings: str
    competitor_findings: str
    divergence_findings: str
    final_report: str


# ── Per-specialist system prompts ──────────────────────────────────────────────

FUNDAMENTAL_PROMPT = """\
You are the Fundamental Analyst on a sell-side equity research team.
Your job covers ONLY: revenue trends, earnings quality, balance sheet health, margins,
free cash flow, and SEC filing disclosures (10-K / 10-Q).

Do NOT: assess technical price action, compute DCF intrinsic value, interpret news
sentiment, or compare competitors. Those are handled by dedicated specialists.

Synthesize the raw tool outputs below into 4–5 concise paragraphs.
Write for a Lead Analyst who will integrate your findings with five other specialist reports.\
"""

TECHNICAL_PROMPT = """\
You are the Technical Analyst on a sell-side equity research team.
Your job covers ONLY: RSI momentum, moving average signals (50/200-day crossovers,
365-day MA), ATR volatility levels, and trend regime classification (bullish / bearish / neutral).

Do NOT: assess company fundamentals, run DCF analysis, interpret news, or compare competitors.

When interpreting indicators:
- State whether they confirm or contradict each other
- Classify trend direction (bullish / bearish) and market dynamics (slow / fast / volatile)
- Note overbought or oversold conditions with actionable timing implications

Use the fundamental context only to anchor price levels — do not repeat it.
Synthesize into 3–4 structured paragraphs ending with a unified technical verdict.\
"""

VALUATION_PROMPT = """\
You are the Valuation Analyst on a sell-side equity research team.
Your job covers ONLY: DCF intrinsic value vs current price, analyst price targets,
consensus ratings, and rating trend direction (upgrades / downgrades).

Do NOT: re-fetch or re-interpret fundamental filing data, assess technical indicators,
summarize news narratives, or discuss competitor valuations.

You may reference the fundamental context to validate DCF assumptions — but do not
repeat it; build on it. Quantify the gap between DCF fair value and Street consensus.
Explicitly state whether the stock is undervalued, fairly valued, or overvalued on each measure.
Synthesize into 3–4 paragraphs.\
"""

MARKET_INTEL_PROMPT = """\
You are the Market Intelligence Analyst on a sell-side equity research team.
Your job covers ONLY: recent news catalysts, social media sentiment, short-term narrative
shifts, regulatory or macro headlines, and retail/institutional investor tone.

Do NOT: restate financial fundamentals, run technical analysis, or compute valuation metrics.

Classify the current news and sentiment environment (positive / negative / mixed) and
identify the 2–3 most market-moving signals. Note any divergence between news tone and
social sentiment. Synthesize into 3–4 paragraphs.\
"""

COMPETITOR_PROMPT = """\
You are the Competitive Intelligence Analyst on a sell-side equity research team.
Your job covers ONLY: peer group financial comparisons, SWOT analysis relative to
competitors, hiring signals, and white-space strategic opportunities.

Do NOT: discuss absolute valuation, technical indicators, or restate news already covered.

Reference the fundamental findings to anchor relative comparisons (e.g., "our target's
margins exceed the peer median of X%"), but do not repeat fundamental detail.
Synthesize the SWOT and white-space data into 3–4 paragraphs, ending with a clear
competitive positioning verdict (leader / challenger / laggard in key dimensions).\
"""

DIVERGENCE_PROMPT = """\
You are the Divergence Analyst on a sell-side equity research team.
Your job covers ONLY: detecting and interpreting gaps between technical signals and
fundamental / valuation signals across 1-week, 1-month, and 3-month windows.

Do NOT: re-derive any individual indicators. Reference only the divergence analysis output
and connect it to the specialist findings already summarized below.

Classify the divergence type (bullish divergence, bearish divergence, confirmation, neutral)
and assess its persistence across time windows. Translate into a risk / opportunity flag
for the Lead Analyst's synthesis. Synthesize into 2–3 focused paragraphs.\
"""

SYNTHESIS_PROMPT = """\
You are the Lead Analyst on a sell-side equity research team.
You have received complete findings from six specialist analysts. Your job is to integrate
them into a single, professional equity research report.

Report structure (use these exact section headers):
1. Executive Summary (2–3 sentences: thesis, key catalysts, primary risk flag)
2. Fundamental Analysis
3. Technical Analysis
4. Valuation
5. Market Intelligence & Sentiment
6. Competitive Positioning
7. Divergence & Risk Signals
8. Investment Conclusion (Buy / Hold / Sell with price target range if supported by data)

Rules:
- Do not invent data not present in the specialist reports
- Highlight where specialists CONFIRM each other (increases conviction)
- Highlight where specialists CONTRADICT each other (flag as key risk or uncertainty)
- Keep each section tight: 3–5 sentences, lead with the most important finding
- Write in a professional, third-person sell-side tone\
"""


# ── Specialist nodes ───────────────────────────────────────────────────────────

def fundamental_node(state: AnalysisState) -> dict:
    company, ticker, year = state["company"], state["ticker"], state["year"]

    filings_result    = analyze_filings.invoke({"query": f"{company} {year} revenue earnings margins cash flow"})
    parser_result     = analyze_parser.invoke({"query": f"{company} {year} financial highlights balance sheet"})
    financials_result = analyze_financials.invoke({"query": f"{company} {ticker} financial data"})

    synthesis = LLM.invoke([
        SystemMessage(content=FUNDAMENTAL_PROMPT),
        HumanMessage(content=(
            f"Company: {company} | Ticker: {ticker} | Year: {year}\n\n"
            f"10-K/10-Q Filings:\n{filings_result}\n\n"
            f"Parsed Filing Sections:\n{parser_result}\n\n"
            f"Market Financial Data:\n{financials_result}"
        )),
    ])
    return {"fundamental_findings": synthesis.content}


def technical_node(state: AnalysisState) -> dict:
    ticker = state["ticker"]

    rsi_result    = calculate_rsi_tool.invoke({"ticker": ticker})
    ma_result     = calculate_moving_average_tool.invoke({"ticker": ticker, "days": 365})
    regime_result = calculate_trend_regime_tool.invoke({"ticker": ticker})
    atr_result    = calculate_atr_tool.invoke({"ticker": ticker})

    synthesis = LLM.invoke([
        SystemMessage(content=TECHNICAL_PROMPT),
        HumanMessage(content=(
            f"Ticker: {ticker}\n\n"
            f"Fundamental context (price anchoring only — do not repeat):\n"
            f"{state['fundamental_findings'][:600]}\n\n"
            f"RSI:\n{rsi_result}\n\n"
            f"365-Day Moving Average:\n{ma_result}\n\n"
            f"Trend Regime (50/200-day MA):\n{regime_result}\n\n"
            f"ATR Volatility:\n{atr_result}"
        )),
    ])
    return {"technical_findings": synthesis.content}


def valuation_node(state: AnalysisState) -> dict:
    company, ticker, year = state["company"], state["ticker"], state["year"]

    valuation_result = valuation_tool.invoke({"company": company, "year": year})
    dcf_result       = find_dcf_tool.invoke({"company": company, "year": year})

    synthesis = LLM.invoke([
        SystemMessage(content=VALUATION_PROMPT),
        HumanMessage(content=(
            f"Company: {company} | Ticker: {ticker} | Year: {year}\n\n"
            f"Fundamental context (anchor only — do not repeat):\n"
            f"{state['fundamental_findings'][:500]}\n\n"
            f"Valuation Tool Output (DCF + analyst ratings + Street targets):\n{valuation_result}\n\n"
            f"Raw DCF Calculation:\n{dcf_result}"
        )),
    ])
    return {"valuation_findings": synthesis.content}


def market_intel_node(state: AnalysisState) -> dict:
    company, ticker = state["company"], state["ticker"]

    news_result      = analyze_news.invoke({"query": f"{company} {ticker} recent news catalysts"})
    sentiment_result = analyze_social_sentiment_tool.invoke({"query": ticker})

    synthesis = LLM.invoke([
        SystemMessage(content=MARKET_INTEL_PROMPT),
        HumanMessage(content=(
            f"Company: {company} | Ticker: {ticker}\n\n"
            f"News Analysis:\n{news_result}\n\n"
            f"Social Sentiment:\n{sentiment_result}"
        )),
    ])
    return {"market_intel_findings": synthesis.content}


def competitor_node(state: AnalysisState) -> dict:
    company, ticker = state["company"], state["ticker"]

    comp_data = advanced_comp_analysis(ticker)
    comp_json = json.dumps(comp_data, indent=2)

    synthesis = LLM.invoke([
        SystemMessage(content=COMPETITOR_PROMPT),
        HumanMessage(content=(
            f"Company: {company} | Ticker: {ticker}\n\n"
            f"Fundamental context (anchor for relative comparisons):\n"
            f"{state['fundamental_findings'][:500]}\n\n"
            f"Competitor Analysis (metrics, SWOT, white space):\n{comp_json[:4000]}"
        )),
    ])
    return {"competitor_findings": synthesis.content}


def divergence_node(state: AnalysisState) -> dict:
    ticker = state["ticker"]

    divergence_result = analyze_divergence_tool.invoke({"ticker": ticker})

    synthesis = LLM.invoke([
        SystemMessage(content=DIVERGENCE_PROMPT),
        HumanMessage(content=(
            f"Ticker: {ticker}\n\n"
            f"Technical findings (cross-reference):\n{state['technical_findings'][:500]}\n\n"
            f"Valuation findings (cross-reference):\n{state['valuation_findings'][:400]}\n\n"
            f"Divergence Analysis Output:\n{divergence_result}"
        )),
    ])
    return {"divergence_findings": synthesis.content}


def synthesis_node(state: AnalysisState) -> dict:
    company, ticker, year = state["company"], state["ticker"], state["year"]

    all_findings = (
        f"FUNDAMENTAL ANALYSIS:\n{state['fundamental_findings']}\n\n"
        f"TECHNICAL ANALYSIS:\n{state['technical_findings']}\n\n"
        f"VALUATION ANALYSIS:\n{state['valuation_findings']}\n\n"
        f"MARKET INTELLIGENCE & SENTIMENT:\n{state['market_intel_findings']}\n\n"
        f"COMPETITIVE INTELLIGENCE:\n{state['competitor_findings']}\n\n"
        f"DIVERGENCE & RISK SIGNALS:\n{state['divergence_findings']}"
    )

    final = LLM.invoke([
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(content=(
            f"Company: {company} | Ticker: {ticker} | Coverage Year: {year}\n\n"
            f"{all_findings}"
        )),
    ])
    return {"final_report": final.content}


# ── Graph construction ─────────────────────────────────────────────────────────

_nodes = [
    ("fundamental",  fundamental_node),
    ("technical",    technical_node),
    ("valuation",    valuation_node),
    ("market_intel", market_intel_node),
    ("competitor",   competitor_node),
    ("divergence",   divergence_node),
    ("synthesis",    synthesis_node),
]

_workflow = StateGraph(AnalysisState)
for _name, _fn in _nodes:
    _workflow.add_node(_name, _fn)

_seq = [n for n, _ in _nodes]
_workflow.add_edge(START, _seq[0])
for _a, _b in zip(_seq, _seq[1:]):
    _workflow.add_edge(_a, _b)
_workflow.add_edge(_seq[-1], END)

lead_analyst_graph = _workflow.compile()


def run_lead_analyst(company: str, ticker: str, year: str) -> str:
    """Run the full sequential analyst graph and return the final report text."""
    result = lead_analyst_graph.invoke({
        "company":               company,
        "ticker":                ticker,
        "year":                  year,
        "fundamental_findings":  "",
        "technical_findings":    "",
        "valuation_findings":    "",
        "market_intel_findings": "",
        "competitor_findings":   "",
        "divergence_findings":   "",
        "final_report":          "",
    })
    return result["final_report"]
