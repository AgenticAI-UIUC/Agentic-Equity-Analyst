"""
PM Challenge Loop - Specialist Re-Querying

Calls specialist agents with targeted queries based on PM feedback.
Phase 2 implementation.
"""

from typing import Dict, List, Optional
from analyst import analyze_filings, analyze_financials, analyze_news, analyze_parser
from dcf import find_dcf_tool
from market_data_loader import calculate_moving_average_tool
from valuation_agent import valuation_tool


# ============================================================================
# SPECIALIST AGENT REGISTRY
# ============================================================================

# Map agent names to actual tool functions
AGENT_REGISTRY = {
    "analyze_filings": analyze_filings,
    "analyze_parser": analyze_parser,
    "analyze_financials": analyze_financials,
    "analyze_news": analyze_news,
    "find_dcf_tool": find_dcf_tool,
    "calculate_moving_average_tool": calculate_moving_average_tool,
    "valuation_tool": valuation_tool,
}


# ============================================================================
# RE-QUERYING FUNCTIONS
# ============================================================================

def requery_specialists(
    agent_queries: Dict[str, List[str]],
    company: str,
    ticker: str,
    year: str,
    pm_context: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Execute targeted queries to specialist agents based on PM feedback.

    Args:
        agent_queries: Dict from route_pm_issues() mapping agent_name -> queries
        company: Company name
        ticker: Stock ticker
        year: Analysis year
        pm_context: Optional PM feedback context to include in queries

    Returns:
        Dict mapping agent_name -> list of new evidence strings
    """
    results: Dict[str, List[str]] = {}

    for agent_name, queries in agent_queries.items():
        if agent_name not in AGENT_REGISTRY:
            print(f"Warning: Unknown agent '{agent_name}', skipping")
            continue

        agent_func = AGENT_REGISTRY[agent_name]
        results[agent_name] = []

        for query in queries:
            try:
                # Enhance query with PM context
                enhanced_query = _enhance_query_with_context(
                    query=query,
                    company=company,
                    ticker=ticker,
                    year=year,
                    pm_context=pm_context,
                )

                # Call the specialist agent
                result = _call_specialist_agent(
                    agent_func=agent_func,
                    agent_name=agent_name,
                    query=enhanced_query,
                    company=company,
                    ticker=ticker,
                    year=year,
                )

                if result:
                    results[agent_name].append(result)

            except Exception as e:
                error_msg = f"Error re-querying {agent_name}: {str(e)}"
                print(f"Warning: {error_msg}")
                results[agent_name].append(f"[{error_msg}]")

    return results


def _enhance_query_with_context(
    query: str,
    company: str,
    ticker: str,
    year: str,
    pm_context: Optional[str] = None,
) -> str:
    """
    Enhance query with company context and PM feedback.
    """
    enhanced = f"{company} ({ticker}) {year}: {query}"

    if pm_context:
        enhanced += f"\n\nPM Feedback Context: {pm_context}"

    return enhanced


def _call_specialist_agent(
    agent_func,
    agent_name: str,
    query: str,
    company: str,
    ticker: str,
    year: str,
) -> str:
    """
    Call a specialist agent with appropriate parameters.

    Different agents have different signatures, so we need to handle each case.
    """
    # Analyst tools (analyze_filings, analyze_parser, analyze_financials, analyze_news)
    # These take a single query parameter
    if agent_name in ["analyze_filings", "analyze_parser", "analyze_financials", "analyze_news"]:
        return agent_func.invoke({"query": query})

    # DCF tool
    elif agent_name == "find_dcf_tool":
        # find_dcf_tool(company: str, ticker: str, year: str) -> str
        # We'll call it with the standard params and append query context
        base_result = agent_func.invoke({
            "company": company,
            "ticker": ticker,
            "year": year,
        })
        return f"{base_result}\n\nTargeted Query: {query}"

    # Moving average tool
    elif agent_name == "calculate_moving_average_tool":
        # calculate_moving_average_tool(ticker: str, window: int) -> str
        # Extract window from query if possible, default to 50
        window = _extract_window_from_query(query)
        return agent_func.invoke({
            "ticker": ticker,
            "window": window,
        })

    # Valuation tool
    elif agent_name == "valuation_tool":
        # valuation_tool(query: str) -> str
        return agent_func.invoke({"query": query})

    else:
        return f"[Unknown agent type: {agent_name}]"


def _extract_window_from_query(query: str) -> int:
    """
    Try to extract moving average window from query text.
    Default to 50 if not found.
    """
    query_lower = query.lower()

    # Common moving average periods
    if "200" in query_lower or "200-day" in query_lower:
        return 200
    elif "100" in query_lower or "100-day" in query_lower:
        return 100
    elif "50" in query_lower or "50-day" in query_lower:
        return 50
    elif "20" in query_lower or "20-day" in query_lower:
        return 20
    else:
        return 50  # Default


# ============================================================================
# EVIDENCE SYNTHESIS
# ============================================================================

def synthesize_new_evidence(
    requery_results: Dict[str, List[str]],
) -> str:
    """
    Combine all new evidence from specialist re-queries into a single summary.

    This will be used to patch the original draft.

    Args:
        requery_results: Results from requery_specialists()

    Returns:
        Synthesized evidence string
    """
    if not requery_results:
        return ""

    sections = []

    for agent_name, evidence_list in requery_results.items():
        if not evidence_list:
            continue

        sections.append(f"## {agent_name.replace('_', ' ').title()}\n")

        for idx, evidence in enumerate(evidence_list, 1):
            if evidence.strip():
                sections.append(f"{evidence}\n")

    return "\n".join(sections)


def format_requery_summary(
    agent_queries: Dict[str, List[str]],
    requery_results: Dict[str, List[str]],
) -> str:
    """
    Create a human-readable summary of what was re-queried and what was found.

    Args:
        agent_queries: Original queries that were executed
        requery_results: Results from those queries

    Returns:
        Formatted summary string
    """
    lines = ["=" * 80]
    lines.append("SPECIALIST RE-QUERY SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    total_queries = sum(len(queries) for queries in agent_queries.values())
    total_agents = len(agent_queries)

    lines.append(f"Total Agents Queried: {total_agents}")
    lines.append(f"Total Queries Executed: {total_queries}")
    lines.append("")

    for agent_name in sorted(agent_queries.keys()):
        queries = agent_queries[agent_name]
        results = requery_results.get(agent_name, [])

        lines.append(f"### {agent_name}")
        lines.append(f"Queries: {len(queries)}")
        lines.append(f"Results: {len(results)}")
        lines.append("")

        for idx, query in enumerate(queries, 1):
            lines.append(f"  Query {idx}: {query[:100]}{'...' if len(query) > 100 else ''}")
            if idx <= len(results):
                result_preview = results[idx - 1][:150].replace('\n', ' ')
                lines.append(f"  Result: {result_preview}{'...' if len(results[idx - 1]) > 150 else ''}")
            lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)
