"""
PM Challenge Loop - Routing Table

Maps PM blind spot types and issues to specific specialist agents for re-querying.
Phase 2 implementation.
"""

from typing import Dict, List, Set
from pm_schemas import BlindSpotType, BlindSpot, RuleViolation

# ============================================================================
# ROUTING TABLE: Blind Spot Type → Specialist Agents
# ============================================================================

BLIND_SPOT_ROUTING: Dict[BlindSpotType, List[str]] = {
    BlindSpotType.ASSUMPTION_RISK: [
        "find_dcf_tool",
        "analyze_filings",
        "analyze_financials",
    ],
    BlindSpotType.NARRATIVE_DRIFT: [
        "find_dcf_tool",
        "analyze_filings",
    ],
    BlindSpotType.EVIDENCE_MISMATCH: [
        "analyze_filings",
        "analyze_financials",
        "analyze_news",
    ],
    BlindSpotType.QUAL_QUANT_CONTRADICTION: [
        "analyze_news",
        "analyze_filings",
        "find_dcf_tool",
    ],
    BlindSpotType.MISSING_SCENARIO: [
        "find_dcf_tool",
    ],
    BlindSpotType.CATALYST_AMBIGUITY: [
        "analyze_news",
        "analyze_filings",
    ],
    BlindSpotType.TEMPORAL_MISMATCH: [
        "analyze_news",
        "analyze_filings",
        "calculate_moving_average_tool",
    ],
}

# ============================================================================
# ROUTING TABLE: Rule Violation → Specialist Agents
# ============================================================================

# Map common rule names to specialist agents
RULE_VIOLATION_ROUTING: Dict[str, List[str]] = {
    "tri_scenario_valuation": ["find_dcf_tool"],
    "sensitivity_table": ["find_dcf_tool"],
    "invalidation_criteria": ["analyze_filings", "analyze_news"],
    "wacc_breakdown": ["find_dcf_tool"],
    "margin_bridge": ["analyze_financials", "analyze_filings"],
    "revenue_cagr": ["analyze_financials", "analyze_filings"],
    "terminal_growth": ["find_dcf_tool"],
    "catalyst_timeline": ["analyze_news", "analyze_filings"],
    "technical_context": ["calculate_moving_average_tool"],
}

# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_pm_issues(
    blind_spots: List[BlindSpot],
    rule_violations: List[RuleViolation],
) -> Dict[str, List[str]]:
    """
    Parse PM feedback and determine which specialist agents to call.

    Args:
        blind_spots: List of blind spots identified by PM
        rule_violations: List of rule violations identified by PM

    Returns:
        Dict mapping agent_name -> list of targeted queries to make
    """
    agent_queries: Dict[str, List[str]] = {}

    # Route blind spots
    for blind_spot in blind_spots:
        # Get agents from routing table
        agents = BLIND_SPOT_ROUTING.get(blind_spot.blind_spot_type, [])

        # If PM explicitly specified agents, use those instead
        if blind_spot.route_to_agent:
            agents = blind_spot.route_to_agent

        for agent_name in agents:
            if agent_name not in agent_queries:
                agent_queries[agent_name] = []

            # Create targeted query based on blind spot
            query = _create_targeted_query_from_blind_spot(blind_spot)
            if query and query not in agent_queries[agent_name]:
                agent_queries[agent_name].append(query)

    # Route rule violations
    for violation in rule_violations:
        # Try to match violation name to known rules
        agents = _get_agents_for_rule_violation(violation)

        for agent_name in agents:
            if agent_name not in agent_queries:
                agent_queries[agent_name] = []

            # Create targeted query based on rule violation
            query = _create_targeted_query_from_violation(violation)
            if query and query not in agent_queries[agent_name]:
                agent_queries[agent_name].append(query)

    return agent_queries


def _create_targeted_query_from_blind_spot(blind_spot: BlindSpot) -> str:
    """
    Create a targeted query string based on blind spot details.

    The query should be specific enough to get delta evidence without
    regenerating the entire report.
    """
    # Use the PM's suggested action as the base query
    query_parts = []

    if blind_spot.description:
        query_parts.append(blind_spot.description)

    if blind_spot.suggested_action:
        query_parts.append(f"Action needed: {blind_spot.suggested_action}")

    # Add context about expected impact
    if blind_spot.estimated_impact:
        query_parts.append(f"Focus on: {blind_spot.estimated_impact}")

    return " ".join(query_parts)


def _create_targeted_query_from_violation(violation: RuleViolation) -> str:
    """
    Create a targeted query string based on rule violation.
    """
    query_parts = [violation.violation_name]

    if violation.explanation:
        query_parts.append(violation.explanation)

    if violation.required_action:
        query_parts.append(f"Required: {violation.required_action}")

    return " ".join(query_parts)


def _get_agents_for_rule_violation(violation: RuleViolation) -> List[str]:
    """
    Determine which agents to call for a given rule violation.
    """
    # Check if violation name matches known rules
    for rule_key, agents in RULE_VIOLATION_ROUTING.items():
        if rule_key.lower() in violation.violation_name.lower():
            return agents

    # Fallback: try to infer from violation text
    violation_text = (violation.violation_name + " " + violation.explanation).lower()

    agents = []
    if any(keyword in violation_text for keyword in ["dcf", "valuation", "wacc", "terminal", "scenario"]):
        agents.append("find_dcf_tool")
    if any(keyword in violation_text for keyword in ["filing", "10-k", "10-q", "guidance", "management"]):
        agents.append("analyze_filings")
    if any(keyword in violation_text for keyword in ["news", "catalyst", "sentiment"]):
        agents.append("analyze_news")
    if any(keyword in violation_text for keyword in ["financial", "margin", "revenue", "earnings"]):
        agents.append("analyze_financials")
    if any(keyword in violation_text for keyword in ["technical", "moving average", "trend"]):
        agents.append("calculate_moving_average_tool")

    return agents if agents else ["analyze_filings"]  # Default fallback


def get_unique_agents_to_query(agent_queries: Dict[str, List[str]]) -> Set[str]:
    """
    Extract unique agent names from routing result.

    Args:
        agent_queries: Result from route_pm_issues()

    Returns:
        Set of unique agent names
    """
    return set(agent_queries.keys())


def merge_query_results(
    existing_content: str,
    agent_name: str,
    new_evidence: str,
) -> str:
    """
    Merge new evidence from specialist into existing content.

    This is a simple merge strategy for Phase 2. Phase 3 will add
    more sophisticated section-level patching.

    Args:
        existing_content: Current report section content
        agent_name: Name of the specialist agent
        new_evidence: New evidence from re-querying

    Returns:
        Updated content with new evidence incorporated
    """
    if not new_evidence or new_evidence.strip() == "":
        return existing_content

    # For now, append new evidence with a clear marker
    # Phase 3 will use LLM to intelligently merge
    marker = f"\n\n--- Updated {agent_name} Evidence ---\n"
    return existing_content + marker + new_evidence
