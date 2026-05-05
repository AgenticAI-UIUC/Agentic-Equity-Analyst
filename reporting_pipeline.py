"""High-level orchestration utilities for generating equity research reports."""

from __future__ import annotations

from typing import Optional

from analyst import analyze_filings, analyze_financials, analyze_news, analyze_parser
from deepagents import create_deep_agent
from dcf import find_dcf_tool
from divergence_analyzer import analyze_divergence_tool
from langchain import agents
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from market_data_loader import calculate_moving_average_tool, calculate_trend_regime_tool, calculate_rsi_tool, calculate_atr_tool, get_daily_yf_tool
from pdf_builder import report
from valuation_agent import valuation_tool
from pm_agent import PMAgent, format_ic_memo_text
from pm_schemas import PMReview, InvestmentCommitteeMemo, Verdict
from pm_routing import route_pm_issues
from pm_requery import requery_specialists, synthesize_new_evidence, format_requery_summary
from pm_patcher import patch_draft_with_new_evidence, validate_patch_quality

LLM_MANAGER = init_chat_model("gpt-5.1", model_provider="openai")
LLM_REPORTER = ChatOpenAI(model="gpt-4o", temperature=0.2, timeout=30)

MANAGER_PROMPT = (
    "You are a helpful professional equity analyst. "
    "Run the reporting tool once per request and return the tool's response."
)

REPORTING_PROMPT = """
You are a helpful professional financial analyst tasked with consulting a user.

You have access to these tools:
- analyze_filings: find specific financial metrics of a company in its 10-Q and 10-K filings.
- find_dcf_tool: run a Discounted Cash Flow analysis for a company and year.
- analyze_financials: retrieve financial ticker data for a company.
- valuation_tool: summarize equity research valuation commentary for a company and year.
- analyze_news: extract recent qualitative signals from news coverage.
- get_daily_yf_tool: fetch historical daily stock price data (OHLCV) from Yahoo Finance for up to 365 days of trading history.
- calculate_moving_average_tool: calculate the 365-day moving average for a stock ticker.
- calculate_trend_regime_tool: analyze the trend regime using 50-day and 200-day moving averages to determine if the stock is in a bullish, bearish, or neutral trend.
- calculate_rsi_tool: calculate the Relative Strength Index (RSI) to identify overbought, oversold, or neutral conditions with actionable trading advice.
- calculate_atr_tool: calculate the Average True Range (ATR) to measure market volatility and identify quiet or volatile market conditions.
- analyze_divergence_tool: detect divergence between technical indicators (RSI + Moving Averages) and fundamental signals (Analyst Ratings) across 1-week, 1-month, and 3-month periods to identify potential trading opportunities or risks.

CRITICAL - TECHNICAL ANALYSIS INTEGRATION:
When performing technical analysis, you MUST use multiple indicators together and synthesize them into unified insights:

1. ALWAYS run multiple technical indicators (RSI, ATR, Moving Averages, Trend Regime) when analyzing a stock's technical position.
   ALSO run the analyze_divergence_tool to identify divergence between technical and fundamental signals.

2. INTEGRATE the indicators to assess:
   - TREND DIRECTION (Bullish vs Bearish):
     * Compare trend regime (50/200 MA) with RSI momentum
     * Check if moving averages support or contradict RSI signals
     * Determine if indicators confirm each other or show divergence

   - MARKET DYNAMICS (Slow vs Fast Movement):
     * Use ATR to assess volatility levels
     * Correlate volatility with trend strength
     * Identify if high/low volatility supports or conflicts with trend signals

3. PROVIDE UNIFIED CONCLUSIONS:
   - Synthesize all indicators into clear statements like:
     * "Strong bullish case with high conviction" (when trend, MA, and RSI all align bullish)
     * "Bearish with caution" (when indicators show bearish trend but RSI is oversold)
     * "Slow, range-bound movement expected" (when trend is neutral and ATR is low)
     * "Fast, volatile bullish momentum" (when bullish trend combines with high ATR)

   - Highlight CONFIRMATION: "Multiple indicators confirm [direction]"
   - Flag DIVERGENCE: "RSI shows [X] while trend indicates [Y], suggesting [interpretation]"

4. COMPARE AND CONTRAST:
   - Explicitly state agreement or disagreement between indicators
   - Assess strength of conviction based on how many indicators align
   - Provide context: "The bullish trend is supported by [X, Y] but tempered by [Z]"

5. ACTIONABLE SYNTHESIS:
   - Give clear, integrated recommendations based on the full technical picture
   - Address both direction (buy/sell/hold) and timing (enter now vs wait)
   - Account for volatility in risk management advice

6. ALWAYS INCLUDE TECHNICAL ANALYSIS IN YOUR FINAL REPORT:
   - Your final report MUST contain a dedicated "Technical Analysis" section
   - This section should present all technical indicator results (Moving Averages, RSI, ATR, Trend Regime)
   - Provide the integrated synthesis and unified conclusions from these indicators
   - Do not omit technical analysis even if fundamental analysis is strong
   - Technical analysis should be a core component of every equity research report you generate

Return accurate, concise, data-driven guidance that integrates all technical indicators into cohesive insights.
"""

reporting_tools = [
    analyze_filings,
    analyze_parser,
    analyze_financials,
    analyze_news,
    valuation_tool,
    find_dcf_tool,
    get_daily_yf_tool,
    calculate_moving_average_tool,
    calculate_trend_regime_tool,
    calculate_rsi_tool,
    calculate_atr_tool,
    analyze_divergence_tool,
]

reporting_agent = agents.create_agent(
    model=LLM_REPORTER,
    system_prompt=REPORTING_PROMPT,
    tools=reporting_tools,
)


def _normalize_message_payload(message) -> str:
    """Best-effort conversion from LangChain message payloads into text."""

    content = getattr(message, "text", None) or getattr(message, "content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item))
            else:
                parts.append(str(item))
        content = "\n".join(parts)
    return str(content)


@tool
def create_report(request: str) -> str:
    """Invoke the reporting agent with a natural-language request."""

    res = reporting_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return _normalize_message_payload(res["messages"][-1])


def _summarize_prompt(prompt: str) -> str:
    """Use the manager LLM to lightly compress the user prompt for efficiency."""

    messages = [
        SystemMessage(
            content=(
                "You are a helpful summarizer. Focus on clarity and keep the prompt under 60 words."
            )
        ),
        HumanMessage(
            content=f"Summarize and reformulate the following request without losing intent: {prompt}"
        ),
    ]

    try:
        summary_message = LLM_MANAGER.invoke(messages)
        return _normalize_message_payload(summary_message)
    except Exception:
        return prompt


manager_agent = create_deep_agent(
    model=LLM_MANAGER,
    tools=[create_report],
    system_prompt=MANAGER_PROMPT,
)


def _invoke_manager(prompt: str) -> str:
    """Send the condensed instruction to the manager agent and return the report text."""

    response = manager_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return _normalize_message_payload(response["messages"][-1])


DEFAULT_PROMPT_TEMPLATE = (
    "Create a professional equity research style outlook for {company}{ticker_clause} covering {year}. "
    "Highlight financial performance, valuation, major risks, catalysts, and data-supported insights. "
    "\n\nIMPORTANT: Include a comprehensive TECHNICAL ANALYSIS section that integrates:\n"
    "- Moving Averages (365-day, 50-day, 200-day)\n"
    "- RSI (Relative Strength Index) with overbought/oversold analysis\n"
    "- ATR (Average True Range) for volatility assessment\n"
    "- Trend Regime analysis (bullish/bearish/neutral)\n"
    "Synthesize these indicators to provide a unified technical outlook on direction (bullish/bearish) "
    "and market dynamics (slow/fast movement, volatility). Highlight confirmations and divergences between indicators."
)


def build_prompt(
    company: str,
    year: str,
    ticker: Optional[str] = None,
    custom_prompt: Optional[str] = None,
) -> str:
    """Return either the user-provided prompt or a descriptive default template."""

    if custom_prompt and custom_prompt.strip():
        return custom_prompt

    ticker_clause = f" (ticker: {ticker.upper()})" if ticker else ""
    return DEFAULT_PROMPT_TEMPLATE.format(
        company=company,
        ticker_clause=ticker_clause,
        year=year,
    )


def generate_financial_report(
    *,
    company: str,
    ticker: Optional[str] = None,
    year: str,
    custom_prompt: Optional[str] = None,
    launch_ui: bool = False,
    file_path: str = "report.txt",
) -> str:
    """Run the end-to-end reporting pipeline and persist results to disk."""

    user_prompt = build_prompt(company, year, ticker, custom_prompt)
    condensed_prompt = _summarize_prompt(user_prompt)
    report_text = _invoke_manager(condensed_prompt)
    report(report_text, launch_ui=launch_ui, file_path=file_path)
    return report_text


def generate_financial_report_with_pm(
    *,
    company: str,
    ticker: Optional[str] = None,
    year: str,
    custom_prompt: Optional[str] = None,
    launch_ui: bool = False,
    file_path: str = "report.txt",
    pm_output_file: str = "pm_review.txt",
    ic_memo_file: str = "ic_memo.txt",
    enable_pm_challenge: bool = True,
) -> tuple[str, Optional[PMReview], Optional[InvestmentCommitteeMemo]]:
    """
    Run the end-to-end reporting pipeline WITH PM Challenge Loop (Phase 1).

    Phase 1: Generate report → PM critiques → output structured issues
    (No revision loop yet - that's Phase 2)

    Args:
        company: Company name
        ticker: Stock ticker
        year: Year for report
        custom_prompt: Optional custom prompt
        launch_ui: Whether to launch Streamlit UI
        file_path: Where to save the report
        pm_output_file: Where to save PM review
        ic_memo_file: Where to save IC memo
        enable_pm_challenge: Whether to run PM review

    Returns:
        Tuple of (report_text, pm_review, ic_memo)
    """

    # Phase 1: Generate draft report (existing pipeline)
    print("\n" + "=" * 80)
    print("PHASE 1: GENERATING EQUITY RESEARCH DRAFT")
    print("=" * 80)

    user_prompt = build_prompt(company, year, ticker, custom_prompt)
    condensed_prompt = _summarize_prompt(user_prompt)
    draft_report = _invoke_manager(condensed_prompt)

    print(f"\n✓ Draft report generated ({len(draft_report)} characters)")

    # If PM challenge is disabled, return early
    if not enable_pm_challenge:
        report(draft_report, launch_ui=launch_ui, file_path=file_path)
        return draft_report, None, None

    # Phase 1: PM Challenge (critique only, no revision yet)
    print("\n" + "=" * 80)
    print("PHASE 1: PM CHALLENGE - REVIEWING DRAFT")
    print("=" * 80)

    pm_agent = PMAgent()

    # Run PM review
    print(f"Running PM review for {company} ({ticker})...")
    pm_review = pm_agent.review_draft(
        draft=draft_report,
        company=company,
        ticker=ticker or company,
        specialist_outputs=None  # Phase 1: No specialist outputs yet
    )

    print(f"\n✓ PM Review Complete")
    print(f"  Verdict: {pm_review.verdict.value}")
    print(f"  Confidence: {pm_review.confidence_level.value}")
    print(f"  Thesis Coherence: {pm_review.thesis_coherence.score}/5")
    print(f"  Numerical Precision: {pm_review.numerical_precision.score}/5")
    print(f"  Qual/Quant Consistency: {pm_review.qual_quant_consistency.score}/5")
    print(f"  Blind Spots: {len(pm_review.blind_spots)}")
    print(f"  Rule Violations: {len(pm_review.rule_violations)}")
    print(f"  Contradictions: {len(pm_review.contradictions)}")

    # Save PM review to file
    pm_review_text = f"""PM REVIEW REPORT
{'=' * 80}

Company: {pm_review.company}
Ticker: {pm_review.ticker}
Review Date: {pm_review.review_date}

VERDICT: {pm_review.verdict.value}
CONFIDENCE: {pm_review.confidence_level.value}

DIMENSION SCORES:
- Thesis Coherence: {pm_review.thesis_coherence.score}/5
- Numerical Precision: {pm_review.numerical_precision.score}/5
- Qual/Quant Consistency: {pm_review.qual_quant_consistency.score}/5

EXECUTIVE SUMMARY:
{pm_review.executive_summary}

{'=' * 80}
STRENGTHS
{'=' * 80}

Thesis Coherence Strengths:
{chr(10).join('  • ' + s for s in pm_review.thesis_coherence.strengths) if pm_review.thesis_coherence.strengths else '  (none)'}

Numerical Precision Strengths:
{chr(10).join('  • ' + s for s in pm_review.numerical_precision.strengths) if pm_review.numerical_precision.strengths else '  (none)'}

Qual/Quant Consistency Strengths:
{chr(10).join('  • ' + s for s in pm_review.qual_quant_consistency.strengths) if pm_review.qual_quant_consistency.strengths else '  (none)'}

{'=' * 80}
ISSUES
{'=' * 80}

Thesis Coherence Issues:
{chr(10).join('  • ' + i for i in pm_review.thesis_coherence.issues) if pm_review.thesis_coherence.issues else '  (none)'}

Numerical Precision Issues:
{chr(10).join('  • ' + i for i in pm_review.numerical_precision.issues) if pm_review.numerical_precision.issues else '  (none)'}

Qual/Quant Consistency Issues:
{chr(10).join('  • ' + i for i in pm_review.qual_quant_consistency.issues) if pm_review.qual_quant_consistency.issues else '  (none)'}

{'=' * 80}
RULE VIOLATIONS ({len(pm_review.rule_violations)})
{'=' * 80}

{chr(10).join(f"[{rv.severity.value}] {rv.rule_id}: {rv.message}{chr(10)}  Action: {rv.required_action}{chr(10)}" for rv in pm_review.rule_violations) if pm_review.rule_violations else '(none)'}

{'=' * 80}
CONTRADICTIONS ({len(pm_review.contradictions)})
{'=' * 80}

{chr(10).join(f"[{c.severity.value}] {c.check_type}: {c.description}{chr(10)}  Recommendation: {c.recommendation}{chr(10)}" for c in pm_review.contradictions) if pm_review.contradictions else '(none)'}

{'=' * 80}
BLIND SPOTS ({len(pm_review.blind_spots)})
{'=' * 80}

{chr(10).join(f"[{bs.severity.value}] {bs.type.value} / {bs.subtype}:{chr(10)}  {bs.description}{chr(10)}  Evidence Gap: {bs.evidence_gap}{chr(10)}  Route To: {', '.join(bs.route_to_agent)}{chr(10)}  Action: {bs.required_action}{chr(10)}  Impact: {bs.impact_on_thesis}{chr(10)}" for bs in pm_review.blind_spots) if pm_review.blind_spots else '(none)'}

{'=' * 80}
REQUIRED OUTPUTS CHECKLIST
{'=' * 80}

{chr(10).join(f"  {'✓' if v else '✗'} {k}" for k, v in pm_review.required_outputs_present.items())}

{'=' * 80}
AGENTS TO RE-QUERY
{'=' * 80}

{chr(10).join(f"  • {agent}" for agent in pm_review.agents_to_requery) if pm_review.agents_to_requery else '  (none)'}

{'=' * 80}
"""

    with open(pm_output_file, "w", encoding="utf-8") as f:
        f.write(pm_review_text)

    print(f"\n✓ PM review saved to: {pm_output_file}")

    # Generate Investment Committee Memo
    print("\n" + "=" * 80)
    print("PHASE 1: GENERATING INVESTMENT COMMITTEE MEMO")
    print("=" * 80)

    ic_memo = pm_agent.generate_ic_memo(
        draft=draft_report,
        pm_review=pm_review,
        company=company,
        ticker=ticker or company
    )

    print(f"\n✓ IC Memo Generated")
    print(f"  Verdict: {ic_memo.verdict.value}")
    print(f"  Quality: {ic_memo.overall_thesis_quality}")
    print(f"  Decision Readiness: {ic_memo.decision_readiness_score}/100")
    print(f"  IC Recommendation: {ic_memo.ic_recommendation}")

    # Format and save IC memo
    ic_memo_text = format_ic_memo_text(ic_memo)

    with open(ic_memo_file, "w", encoding="utf-8") as f:
        f.write(ic_memo_text)

    print(f"\n✓ IC memo saved to: {ic_memo_file}")

    # Save the original draft report
    report(draft_report, launch_ui=launch_ui, file_path=file_path)

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  • Draft Report: {file_path}")
    print(f"  • PM Review: {pm_output_file}")
    print(f"  • IC Memo: {ic_memo_file}")
    print(f"\nNext: Phase 2 will add revision loop based on PM feedback")
    print("=" * 80)

    return draft_report, pm_review, ic_memo


def generate_financial_report_with_pm_revision(
    *,
    company: str,
    ticker: Optional[str] = None,
    year: str,
    custom_prompt: Optional[str] = None,
    launch_ui: bool = False,
    file_path: str = "report.txt",
    pm_output_file: str = "pm_review.txt",
    ic_memo_file: str = "ic_memo.txt",
    requery_summary_file: str = "pm_requery_summary.txt",
    max_iterations: int = 3,
) -> tuple[str, Optional[PMReview], Optional[InvestmentCommitteeMemo], int]:
    """
    Run the end-to-end reporting pipeline WITH PM Challenge Loop Phase 2 (REVISION).

    Phase 2: Generate report → PM critiques → Re-query specialists → Patch draft → Re-validate
    Iterates until PM approves or max iterations reached.

    Args:
        company: Company name
        ticker: Stock ticker
        year: Year for report
        custom_prompt: Optional custom prompt
        launch_ui: Whether to launch Streamlit UI
        file_path: Where to save the final report
        pm_output_file: Where to save PM review
        ic_memo_file: Where to save IC memo
        requery_summary_file: Where to save re-query summary
        max_iterations: Maximum PM revision iterations (default: 3)

    Returns:
        Tuple of (final_report_text, final_pm_review, ic_memo, iterations_count)
    """

    # Phase 2: Generate initial draft
    print("\n" + "=" * 80)
    print("PHASE 2: GENERATING INITIAL EQUITY RESEARCH DRAFT")
    print("=" * 80)

    user_prompt = build_prompt(company, year, ticker, custom_prompt)
    condensed_prompt = _summarize_prompt(user_prompt)
    current_draft = _invoke_manager(condensed_prompt)

    print(f"\n✓ Initial draft generated ({len(current_draft)} characters)")

    # Initialize PM agent
    pm_agent = PMAgent()

    # Phase 2: Iterative revision loop
    iteration = 0
    final_pm_review = None
    all_requery_summaries = []

    while iteration < max_iterations:
        iteration += 1

        print("\n" + "=" * 80)
        print(f"PHASE 2: PM REVIEW - ITERATION {iteration}/{max_iterations}")
        print("=" * 80)

        # Run PM review
        print(f"Running PM review for {company} ({ticker})...")
        pm_review = pm_agent.review_draft(
            draft=current_draft,
            company=company,
            ticker=ticker or company,
            specialist_outputs=None
        )

        print(f"\n✓ PM Review Complete (Iteration {iteration})")
        print(f"  Verdict: {pm_review.verdict.value}")
        print(f"  Confidence: {pm_review.confidence_level.value}")
        print(f"  Thesis Coherence: {pm_review.dimension_scores.thesis_coherence}/5")
        print(f"  Numerical Precision: {pm_review.dimension_scores.numerical_precision}/5")
        print(f"  Qual/Quant Consistency: {pm_review.dimension_scores.qual_quant_consistency}/5")
        print(f"  Blind Spots: {len(pm_review.blind_spots)}")
        print(f"  Rule Violations: {len(pm_review.rule_violations)}")

        final_pm_review = pm_review

        # Check if approved
        if pm_review.verdict == Verdict.APPROVE:
            print(f"\n🎉 PM APPROVED the report on iteration {iteration}!")
            break

        # If last iteration, stop here
        if iteration >= max_iterations:
            print(f"\n⚠ Maximum iterations ({max_iterations}) reached")
            print(f"Final verdict: {pm_review.verdict.value}")
            break

        # Phase 2: Route issues to specialists
        print("\n" + "=" * 80)
        print(f"PHASE 2: ROUTING PM ISSUES TO SPECIALISTS (Iteration {iteration})")
        print("=" * 80)

        agent_queries = route_pm_issues(
            blind_spots=pm_review.blind_spots,
            rule_violations=pm_review.rule_violations,
        )

        if not agent_queries:
            print("⚠ No specialist agents to re-query (PM found no fixable issues)")
            break

        print(f"\n✓ Routing complete:")
        for agent_name, queries in agent_queries.items():
            print(f"  • {agent_name}: {len(queries)} queries")

        # Phase 2: Re-query specialists
        print("\n" + "=" * 80)
        print(f"PHASE 2: RE-QUERYING SPECIALISTS (Iteration {iteration})")
        print("=" * 80)

        pm_context = f"PM Verdict: {pm_review.verdict.value}. Issues found: {pm_review.executive_summary}"

        requery_results = requery_specialists(
            agent_queries=agent_queries,
            company=company,
            ticker=ticker or company,
            year=year,
            pm_context=pm_context,
        )

        print(f"\n✓ Re-query complete:")
        for agent_name, results in requery_results.items():
            print(f"  • {agent_name}: {len(results)} results")

        # Save re-query summary
        requery_summary = format_requery_summary(agent_queries, requery_results)
        all_requery_summaries.append(f"\n\n{'=' * 80}\nITERATION {iteration}\n{'=' * 80}\n{requery_summary}")

        # Phase 2: Synthesize new evidence
        new_evidence = synthesize_new_evidence(requery_results)

        print(f"\n✓ New evidence synthesized ({len(new_evidence)} characters)")

        # Phase 2: Patch the draft
        print("\n" + "=" * 80)
        print(f"PHASE 2: PATCHING DRAFT WITH NEW EVIDENCE (Iteration {iteration})")
        print("=" * 80)

        print("Running intelligent patching...")
        revised_draft = patch_draft_with_new_evidence(
            original_draft=current_draft,
            new_evidence=new_evidence,
            pm_review=pm_review,
            company=company,
            ticker=ticker or company,
        )

        # Validate patch quality
        patch_validation = validate_patch_quality(
            original_draft=current_draft,
            patched_draft=revised_draft,
            pm_review=pm_review,
        )

        print(f"\n✓ Patching complete")
        print(f"  Length change: {patch_validation['length_change_pct']:.1f}%")
        print(f"  Structure preserved: {patch_validation['structure_preserved']}")
        print(f"  Issues addressed: {patch_validation['issues_addressed']}")

        # Update current draft for next iteration
        current_draft = revised_draft

    # Generate final IC memo
    print("\n" + "=" * 80)
    print("PHASE 2: GENERATING FINAL INVESTMENT COMMITTEE MEMO")
    print("=" * 80)

    ic_memo = pm_agent.generate_ic_memo(
        draft=current_draft,
        pm_review=final_pm_review,
        company=company,
        ticker=ticker or company
    )

    print(f"\n✓ IC Memo Generated")
    print(f"  Verdict: {ic_memo.verdict.value}")
    print(f"  Quality: {ic_memo.overall_thesis_quality}")
    print(f"  Decision Readiness: {ic_memo.decision_readiness_score}/100")

    # Save all outputs
    _save_phase2_outputs(
        final_draft=current_draft,
        pm_review=final_pm_review,
        ic_memo=ic_memo,
        requery_summaries=all_requery_summaries,
        file_path=file_path,
        pm_output_file=pm_output_file,
        ic_memo_file=ic_memo_file,
        requery_summary_file=requery_summary_file,
        launch_ui=launch_ui,
        company=company,
        ticker=ticker or company,
        iterations=iteration,
    )

    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)
    print(f"\nFinal Outputs:")
    print(f"  • Final Report: {file_path}")
    print(f"  • PM Review: {pm_output_file}")
    print(f"  • IC Memo: {ic_memo_file}")
    print(f"  • Re-query Summary: {requery_summary_file}")
    print(f"\nIterations: {iteration}/{max_iterations}")
    print(f"Final Verdict: {final_pm_review.verdict.value if final_pm_review else 'N/A'}")
    print("=" * 80)

    return current_draft, final_pm_review, ic_memo, iteration


def _save_phase2_outputs(
    final_draft: str,
    pm_review: PMReview,
    ic_memo: InvestmentCommitteeMemo,
    requery_summaries: List[str],
    file_path: str,
    pm_output_file: str,
    ic_memo_file: str,
    requery_summary_file: str,
    launch_ui: bool,
    company: str,
    ticker: str,
    iterations: int,
):
    """Save all Phase 2 outputs to disk."""

    # Save final report
    report(final_draft, launch_ui=launch_ui, file_path=file_path)
    print(f"\n✓ Final report saved to: {file_path}")

    # Save PM review
    pm_review_text = _format_pm_review_text(pm_review)
    with open(pm_output_file, "w", encoding="utf-8") as f:
        f.write(pm_review_text)
    print(f"✓ PM review saved to: {pm_output_file}")

    # Save IC memo
    ic_memo_text = format_ic_memo_text(ic_memo)
    with open(ic_memo_file, "w", encoding="utf-8") as f:
        f.write(ic_memo_text)
    print(f"✓ IC memo saved to: {ic_memo_file}")

    # Save re-query summaries
    if requery_summaries:
        requery_text = f"""PM CHALLENGE LOOP - RE-QUERY SUMMARY
{'=' * 80}

Company: {company} ({ticker})
Total Iterations: {iterations}

{"".join(requery_summaries)}

{'=' * 80}
END OF RE-QUERY SUMMARY
{'=' * 80}
"""
        with open(requery_summary_file, "w", encoding="utf-8") as f:
            f.write(requery_text)
        print(f"✓ Re-query summary saved to: {requery_summary_file}")


def _format_pm_review_text(pm_review: PMReview) -> str:
    """Format PM review as text for saving to file."""
    return f"""PM REVIEW REPORT
{'=' * 80}

Company: {pm_review.company}
Ticker: {pm_review.ticker}
Review Date: {pm_review.review_date}

VERDICT: {pm_review.verdict.value}
CONFIDENCE: {pm_review.confidence_level.value}

DIMENSION SCORES:
- Thesis Coherence: {pm_review.dimension_scores.thesis_coherence}/5
- Numerical Precision: {pm_review.dimension_scores.numerical_precision}/5
- Qual/Quant Consistency: {pm_review.dimension_scores.qual_quant_consistency}/5

EXECUTIVE SUMMARY:
{pm_review.executive_summary}

{'=' * 80}
STRENGTHS
{'=' * 80}

{chr(10).join('  • ' + s for s in pm_review.strengths) if pm_review.strengths else '  (none)'}

{'=' * 80}
ISSUES
{'=' * 80}

{chr(10).join('  • ' + i for i in pm_review.issues) if pm_review.issues else '  (none)'}

{'=' * 80}
RULE VIOLATIONS ({len(pm_review.rule_violations)})
{'=' * 80}

{chr(10).join(f"[{rv.severity.value}] {rv.violation_name}: {rv.explanation}{chr(10)}  Action: {rv.required_action}{chr(10)}" for rv in pm_review.rule_violations) if pm_review.rule_violations else '(none)'}

{'=' * 80}
CONTRADICTIONS ({len(pm_review.contradictions)})
{'=' * 80}

{chr(10).join(f"[{c.severity.value}] {c.contradiction_type}: {c.description}{chr(10)}  Recommendation: {c.recommendation}{chr(10)}" for c in pm_review.contradictions) if pm_review.contradictions else '(none)'}

{'=' * 80}
BLIND SPOTS ({len(pm_review.blind_spots)})
{'=' * 80}

{chr(10).join(f"[{bs.severity.value}] {bs.blind_spot_type.value}:{chr(10)}  {bs.description}{chr(10)}  Route To: {', '.join(bs.route_to_agent) if bs.route_to_agent else 'N/A'}{chr(10)}  Action: {bs.suggested_action}{chr(10)}  Impact: {bs.estimated_impact}{chr(10)}" for bs in pm_review.blind_spots) if pm_review.blind_spots else '(none)'}

{'=' * 80}
REQUIRED OUTPUTS CHECKLIST
{'=' * 80}

{chr(10).join(f"  {'✓' if v else '✗'} {k}" for k, v in pm_review.required_outputs.items()) if pm_review.required_outputs else '  (none)'}

{'=' * 80}
"""


__all__ = ["generate_financial_report", "generate_financial_report_with_pm", "generate_financial_report_with_pm_revision", "build_prompt"]
