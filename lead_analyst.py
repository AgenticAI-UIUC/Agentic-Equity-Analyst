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
  2. Synthesizes raw tool output into a detailed, evidence-rich report section
  3. Writes that section into the shared AnalysisState
  4. Passes state to the next node — which can read earlier findings for context

This guarantees complete coverage with no duplicated work.

REPORT STRUCTURE
────────────────
Each specialist node produces one full report section. The synthesis_node adds a
concise final assessment. run_lead_analyst() assembles everything into a single document.

  SECTION 1 — Fundamental Analysis       ← fundamental_node
  SECTION 2 — Technical Analysis         ← technical_node
  SECTION 3 — Valuation                  ← valuation_node
  SECTION 4 — Market Intelligence        ← market_intel_node
  SECTION 5 — Competitive Intelligence   ← competitor_node
  SECTION 6 — Divergence & Risk Signals  ← divergence_node
  SECTION 7 — Lead Analyst Final Assessment ← synthesis_node (conclusion only)

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
    │  writes: state["final_report"]  (Lead Analyst Final Assessment only)
    ▼
  END

KEY DESIGN DECISIONS
─────────────────────
- Each node has a tight system prompt with explicit "Do NOT" exclusions.
- Specialist nodes produce detailed, evidence-citing sections — NOT short summaries.
- valuation_tool already does a full LLM synthesis; valuation_node enhances it rather
  than re-summarizing from scratch.
- advanced_comp_analysis returns structured data (metrics table, source-backed SWOT,
  white space); competitor_node preserves that structure via _format_comp_data().
- analyze_divergence_tool already produces detailed output; divergence_node preserves
  key numerical values and divergence types rather than over-compressing them.
- Synthesis node writes the concluding section only — it does NOT repeat the specialist
  sections that readers have already seen.
- Context passed between nodes uses 1500–2000 char windows (not 400–600).
"""

from __future__ import annotations

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

Write a comprehensive fundamental analysis section (400–600 words) that:
- Cites specific numerical data points directly from the tool outputs provided
  (exact revenue figures, margin percentages, FCF amounts, debt ratios)
- Covers in this order:
    1. Revenue growth trajectory with specific numbers and year-over-year changes
    2. Profitability margins (gross / operating / net) with exact percentages
    3. Free cash flow quality and trend
    4. Balance sheet strength (cash position, debt levels, coverage ratios)
    5. Key risks or notable disclosures from SEC filings (specific items, management commentary)
- Uses precise language: do not say "margins improved" — say "gross margin expanded from X% to Y%"
- Write for a Lead Analyst who will integrate your section with five other specialist sections.\
"""

TECHNICAL_PROMPT = """\
You are the Technical Analyst on a sell-side equity research team.
Your job covers ONLY: RSI momentum, moving average signals (50/200-day crossovers,
365-day MA), ATR volatility levels, and trend regime classification.

Do NOT: assess company fundamentals, run DCF analysis, interpret news, or compare competitors.

Write a comprehensive technical analysis section (300–450 words) that:
- Cites every indicator value explicitly:
    • RSI: state the exact value and what it signals (e.g., "RSI at 67.3 indicates overbought territory")
    • Moving averages: state exact dollar values for the 50-day, 200-day, and 365-day MAs
    • Trend regime: state the classification (bullish/bearish/neutral) with the MA spread that supports it
    • ATR: state the exact ATR value and what it implies for expected daily price range
- Covers: trend direction & conviction, momentum state, volatility level, inter-indicator agreement
- Explicitly states whether indicators confirm or contradict each other
- Ends with a unified technical verdict: direction (bullish/bearish/neutral) +
  conviction level (high/medium/low) + expected price dynamics (volatile/trending/range-bound)
- Use the fundamental context only to anchor price levels — do not repeat it.\
"""

VALUATION_PROMPT = """\
You are the Valuation Analyst on a sell-side equity research team.
Your job covers ONLY: DCF intrinsic value vs current price, analyst price targets,
consensus ratings, and rating trend direction.

Do NOT: re-fetch or re-interpret fundamental filing data, assess technical indicators,
summarize news narratives, or discuss competitor valuations.

You are given two inputs: (1) a pre-written valuation memo from a specialist analyst,
and (2) a raw DCF calculation output. ENHANCE this memo — do NOT re-summarize from scratch.
All key metrics in the memo must be preserved in your output.

Write a comprehensive valuation section (350–500 words) that:
- Preserves all key numerical metrics from the valuation memo (P/E multiples, EV/EBITDA,
  price targets, consensus ratings, specific dollar values) — do not drop these
- Integrates the raw DCF output explicitly: state intrinsic_value, current_price,
  undervaluation_percent, and terminal_value with their actual numbers
- Quantifies the three-way gap: DCF intrinsic value vs. current market price vs. Street
  consensus target — express each difference as both an absolute dollar amount and a percentage
- Explicitly states whether the stock is undervalued / fairly valued / overvalued on EACH measure
- Notes the analyst rating trend direction and what recent upgrades/downgrades signal for conviction
- You may reference the fundamental context to validate DCF assumptions, but do not repeat it.\
"""

MARKET_INTEL_PROMPT = """\
You are the Market Intelligence Analyst on a sell-side equity research team.
Your job covers ONLY: recent news catalysts, social media sentiment, short-term narrative
shifts, regulatory or macro headlines, and retail/institutional investor tone.

Do NOT: restate financial fundamentals, run technical analysis, or compute valuation metrics.

Write a comprehensive market intelligence section (300–450 words) that:
- References specific headlines by source and approximate date wherever available
  (e.g., "According to [source] ([month/year]), the company announced...")
- Classifies the current news environment (positive / negative / mixed) with explicit reasoning
- Identifies the 2–3 most market-moving signals with supporting evidence from the data
- Directly compares news tone vs. social sentiment — note any divergence between institutional
  narrative (news coverage) and retail investor tone (social media / forums)
- Flags any regulatory, macro, or sector-level headwinds or tailwinds that could affect
  the near-term share price\
"""

COMPETITOR_PROMPT = """\
You are the Competitive Intelligence Analyst on a sell-side equity research team.
Your job covers ONLY: peer group financial comparisons, SWOT analysis relative to
competitors, hiring signals, and white-space strategic opportunities.

Do NOT: discuss absolute valuation multiples in isolation, technical indicators, or
restate news already covered by the Market Intelligence specialist.

You are given structured competitive data: a financial metrics comparison table, source-backed
SWOT analyses, and white space opportunities. PRESERVE this structured data in your output —
do not compress specific numbers or source-backed SWOT bullets into generic prose.

Write a comprehensive competitive intelligence section (450–600 words) that:
- Opens with a comparative metrics overview citing specific numbers from the table
  (e.g., "At a trailing P/E of X vs. peer median of Y, the target trades at a Z% premium")
- Uses the fundamental context to anchor relative margin/multiple comparisons with peers
- For the target company SWOT: present each bullet point with its source reference —
  do not drop source names from the data
- For white space opportunities: name each opportunity and explain the specific competitive
  gap it exploits, citing the evidence signal from the data
- Names the 'watch out' competitor and the specific signal that poses a threat
- Ends with a clear competitive positioning verdict across key dimensions
  (e.g., margin leadership, growth trajectory, technology positioning, market share)\
"""

DIVERGENCE_PROMPT = """\
You are the Divergence Analyst on a sell-side equity research team.
Your job covers ONLY: detecting and interpreting gaps between technical signals and
fundamental / valuation signals across 1-week, 1-month, and 3-month windows.

Do NOT: re-derive any individual indicators. Reference only the divergence analysis output
and connect it to the specialist findings already summarized below.

You are given a detailed divergence analysis report with specific scores, divergence values,
and signal classifications. PRESERVE key numerical values — do not drop RSI values, MA values,
divergence scores, or signal type labels.

Write a comprehensive divergence analysis section (300–400 words) that:
- For each time window (1 week, 1 month, 3 months): explicitly states the divergence type,
  combined score, and divergence value (e.g., "Over the 1-month window, D = +0.50 indicates
  technical signals outpacing fundamentals")
- Identifies whether divergence is strengthening, stable, or narrowing across windows
- Connects the divergence pattern to the specific technical and valuation findings already
  written (cross-reference by name: e.g., "aligns with the overbought RSI noted by Technical Analyst")
- Translates divergence into a clear risk/opportunity flag: state explicitly whether this
  is a BUY, SELL, or HOLD signal and why, supported by the divergence data\
"""

SYNTHESIS_PROMPT = """\
You are the Lead Analyst on a sell-side equity research team.
Six specialist sections (Fundamentals, Technical, Valuation, Market Intelligence,
Competitive Intelligence, Divergence) have already been written and appear earlier in the report.

Your job is to write the LEAD ANALYST FINAL ASSESSMENT — the concluding section only.
Do NOT repeat or re-summarize the specialist sections. Readers have already seen them in full.
Your value is integration and a definitive recommendation — not repetition.

Write the Lead Analyst Final Assessment (300–400 words) with these exact sub-sections:

## Investment Thesis
2–3 sentences: the core investment case (the "why"), the primary catalyst driving it,
and the single most important risk. Be direct and definitive.

## Signal Convergence
Bullet list of 3–4 specific points where two or more specialists CONFIRM each other.
Format: "[Specialist A finding with data] + [Specialist B finding with data] → [conclusion]"
Example: "Technical RSI at 67.3 (overbought) + Valuation 18% above DCF intrinsic value
→ near-term mean reversion risk is elevated"

## Key Divergences & Risk Flags
Bullet list of 2–3 specific contradictions between specialists, or areas of high uncertainty.
State the primary risk factor and what would need to change to resolve it.

## Investment Recommendation
One paragraph: Buy / Accumulate / Hold / Reduce / Sell — with:
- A price target range in dollars (if supported by valuation data)
- A time horizon (e.g., 12-month)
- The single key catalyst that would confirm the thesis
- The single key catalyst that would invalidate it

Write in a professional, definitive sell-side tone. No hedging.\
"""


# ── Competitor data formatter ──────────────────────────────────────────────────

def _format_comp_data(comp_data: dict, target: str) -> str:
    """Render the advanced_comp_analysis dict as structured readable text.

    Preserves the metrics table, source-backed SWOT, and white space data
    instead of letting the LLM receive raw JSON and immediately compress it.
    """
    lines: list[str] = []
    tickers = comp_data.get("tickers", [])
    metrics = comp_data.get("metrics", {})
    fields  = ["Price", "Mkt Cap ($B)", "P/E (TTM)", "Fwd P/E", "Rev Growth", "Profit Margin"]

    lines.append("FINANCIAL METRICS COMPARISON")
    col_w  = 16
    header = f"{'Ticker':<8}" + "".join(f"{f:>{col_w}}" for f in fields)
    lines.append(header)
    lines.append("─" * len(header))
    for t in tickers:
        m   = metrics.get(t, {})
        tag = "  ← TARGET" if t == target else ""
        row = f"{t:<8}" + "".join(f"{m.get(f, 'N/A'):>{col_w}}" for f in fields) + tag
        lines.append(row)
    lines.append("")

    # Target SWOT with sources preserved
    swot        = comp_data.get("swot", {})
    target_swot = swot.get(target, {})
    lines.append(f"{target} SWOT (SOURCE-BACKED)")
    for quadrant in ("strengths", "weaknesses", "opportunities", "threats"):
        lines.append(f"\n  {quadrant.upper()}:")
        for item in target_swot.get(quadrant, []):
            src = item.get("source_name", "")
            txt = item.get("text", "")
            lines.append(f"    • {txt}" + (f"  [{src}]" if src else ""))

    # Key competitor highlights (condensed to top strength + weakness per peer)
    lines.append("\nKEY COMPETITOR HIGHLIGHTS")
    for t in tickers:
        if t == target:
            continue
        c_swot  = swot.get(t, {})
        s_items = c_swot.get("strengths", [{}])
        w_items = c_swot.get("weaknesses", [{}])
        lines.append(f"\n  {t}:")
        if s_items:
            lines.append(f"    Strength : {s_items[0].get('text', 'N/A')}")
        if w_items:
            lines.append(f"    Weakness : {w_items[0].get('text', 'N/A')}")

    # White space opportunities with full detail
    ws = comp_data.get("white_space", {})
    lines.append("\nWHITE SPACE OPPORTUNITIES")
    for opp in ws.get("opportunities", []):
        lines.append(
            f"  ► {opp.get('name', '')}\n"
            f"    Gap    : {opp.get('gap', '')}\n"
            f"    Signal : {opp.get('signal', '')}\n"
            f"    Action : {opp.get('action', '')}"
        )
    watch = ws.get("watch_out", {})
    if watch.get("competitor"):
        lines.append(
            f"\n  ⚠  WATCH OUT — {watch['competitor']}\n"
            f"    Signal      : {watch.get('signal', '')}\n"
            f"    Description : {watch.get('description', '')}"
        )

    return "\n".join(lines)


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
            f"{state['fundamental_findings'][:2000]}\n\n"
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
            f"{state['fundamental_findings'][:2000]}\n\n"
            f"Pre-written Valuation Memo (enhance this — do not re-summarize from scratch):\n"
            f"{valuation_result}\n\n"
            f"Raw DCF Calculation Output:\n{dcf_result}"
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

    comp_data      = advanced_comp_analysis(ticker)
    comp_formatted = _format_comp_data(comp_data, ticker)

    synthesis = LLM.invoke([
        SystemMessage(content=COMPETITOR_PROMPT),
        HumanMessage(content=(
            f"Company: {company} | Ticker: {ticker}\n\n"
            f"Fundamental context (anchor for relative comparisons):\n"
            f"{state['fundamental_findings'][:2000]}\n\n"
            f"Structured Competitive Data:\n{comp_formatted}"
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
            f"Technical findings (cross-reference):\n{state['technical_findings'][:1500]}\n\n"
            f"Valuation findings (cross-reference):\n{state['valuation_findings'][:1500]}\n\n"
            f"Full Divergence Analysis Output:\n{divergence_result}"
        )),
    ])
    return {"divergence_findings": synthesis.content}


def synthesis_node(state: AnalysisState) -> dict:
    company, ticker, year = state["company"], state["ticker"], state["year"]

    all_findings = (
        f"FUNDAMENTAL ANALYSIS:\n{state['fundamental_findings']}\n\n"
        f"TECHNICAL ANALYSIS:\n{state['technical_findings']}\n\n"
        f"VALUATION:\n{state['valuation_findings']}\n\n"
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
    """Run the full sequential analyst graph and return a complete assembled report.

    The report consists of six detailed specialist sections followed by the
    Lead Analyst Final Assessment. The synthesis node writes only the concluding
    section — it does not re-summarize the specialist sections.
    """
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

    header  = f"EQUITY RESEARCH REPORT — {company} ({ticker}) | {year}"
    divider = "═" * max(len(header), 60)

    sections = [
        ("FUNDAMENTAL ANALYSIS",            result["fundamental_findings"]),
        ("TECHNICAL ANALYSIS",              result["technical_findings"]),
        ("VALUATION",                       result["valuation_findings"]),
        ("MARKET INTELLIGENCE & SENTIMENT", result["market_intel_findings"]),
        ("COMPETITIVE INTELLIGENCE",        result["competitor_findings"]),
        ("DIVERGENCE & RISK SIGNALS",       result["divergence_findings"]),
        ("LEAD ANALYST FINAL ASSESSMENT",   result["final_report"]),
    ]

    parts = [divider, header, divider, ""]
    for title, content in sections:
        parts.append("─" * 60)
        parts.append(title)
        parts.append("─" * 60)
        parts.append(content.strip())
        parts.append("")

    return "\n".join(parts)
