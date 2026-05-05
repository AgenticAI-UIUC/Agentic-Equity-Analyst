"""
PM Challenge Loop - Draft Patching

Updates specific sections of the draft report based on new evidence.
Phase 2 implementation.
"""

from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from pm_schemas import PMReview, BlindSpot, RuleViolation
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

# LLM for intelligent patching
PATCHER_LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,  # Lower temperature for more consistent patching
    timeout=45,
)


# ============================================================================
# SECTION IDENTIFICATION
# ============================================================================

def identify_sections_to_patch(
    pm_review: PMReview,
    draft_text: str,
) -> Dict[str, List[str]]:
    """
    Identify which sections of the draft need patching based on PM review.

    Args:
        pm_review: PM review with blind spots and violations
        draft_text: Original draft text

    Returns:
        Dict mapping section_name -> list of issues to fix in that section
    """
    section_issues: Dict[str, List[str]] = {}

    # Analyze blind spots
    for blind_spot in pm_review.blind_spots:
        section = _infer_section_from_blind_spot(blind_spot, draft_text)
        if section not in section_issues:
            section_issues[section] = []
        section_issues[section].append(blind_spot.description)

    # Analyze rule violations
    for violation in pm_review.rule_violations:
        section = _infer_section_from_violation(violation, draft_text)
        if section not in section_issues:
            section_issues[section] = []
        section_issues[section].append(violation.explanation)

    return section_issues


def _infer_section_from_blind_spot(blind_spot: BlindSpot, draft_text: str) -> str:
    """
    Infer which section of the draft a blind spot refers to.
    """
    description = blind_spot.description.lower()

    # Keywords to section mapping
    if any(kw in description for kw in ["dcf", "valuation", "price target", "wacc", "terminal"]):
        return "Valuation"
    elif any(kw in description for kw in ["revenue", "margin", "earnings", "financial"]):
        return "Financial Outlook"
    elif any(kw in description for kw in ["news", "catalyst", "sentiment"]):
        return "Catalysts"
    elif any(kw in description for kw in ["risk", "downside", "threat"]):
        return "Risks"
    elif any(kw in description for kw in ["technical", "moving average", "trend", "rsi"]):
        return "Technical Analysis"
    elif any(kw in description for kw in ["thesis", "investment", "recommendation"]):
        return "Investment Thesis"
    else:
        return "General"


def _infer_section_from_violation(violation: RuleViolation, draft_text: str) -> str:
    """
    Infer which section of the draft a rule violation refers to.
    """
    violation_name = violation.violation_name.lower()

    if "scenario" in violation_name or "sensitivity" in violation_name:
        return "Valuation"
    elif "invalidation" in violation_name or "catalyst" in violation_name:
        return "Catalysts"
    elif "technical" in violation_name:
        return "Technical Analysis"
    elif "margin" in violation_name or "revenue" in violation_name:
        return "Financial Outlook"
    else:
        return "General"


# ============================================================================
# DRAFT PATCHING
# ============================================================================

def patch_draft_with_new_evidence(
    original_draft: str,
    new_evidence: str,
    pm_review: PMReview,
    company: str,
    ticker: str,
) -> str:
    """
    Patch the original draft with new evidence from specialist re-queries.

    Uses an LLM to intelligently integrate new evidence without losing
    the structure and flow of the original report.

    Args:
        original_draft: Original equity research draft
        new_evidence: Synthesized new evidence from specialists
        pm_review: PM review identifying issues
        company: Company name
        ticker: Stock ticker

    Returns:
        Updated draft with patches applied
    """
    system_prompt = """You are an expert equity research editor.

Your task is to update an equity research report by incorporating new evidence that addresses specific issues identified by a Portfolio Manager review.

Guidelines:
1. PRESERVE the overall structure and format of the original report
2. UPDATE sections that have issues with the new evidence
3. MAINTAIN professional tone and style consistency
4. DO NOT remove good content - only fix identified issues
5. ENSURE numerical precision - if new DCF data is provided, use it
6. KEEP the report length similar (don't make it much longer)
7. INTEGRATE new evidence smoothly - don't just append it
8. FIX contradictions between sections
9. ADD missing required elements (scenarios, sensitivity, etc.)
10. MAINTAIN section headers and overall organization

Output the complete revised report."""

    issues_summary = _summarize_pm_issues(pm_review)

    human_prompt = f"""Original Report:
{'=' * 80}
{original_draft}
{'=' * 80}

PM Review Issues to Address:
{issues_summary}

New Evidence from Specialist Re-Queries:
{'=' * 80}
{new_evidence}
{'=' * 80}

Company: {company} ({ticker})

Please revise the report to address the PM issues using the new evidence. Output the complete updated report."""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = PATCHER_LLM.invoke(messages)
        return response.content

    except Exception as e:
        print(f"Warning: Patching failed with error: {e}")
        # Fallback: simple append
        return _fallback_patch(original_draft, new_evidence, pm_review)


def _summarize_pm_issues(pm_review: PMReview) -> str:
    """
    Create a concise summary of PM issues for the patcher prompt.
    """
    lines = []

    # Critical rule violations
    critical_violations = [v for v in pm_review.rule_violations if v.severity == "CRITICAL"]
    if critical_violations:
        lines.append("CRITICAL ISSUES:")
        for v in critical_violations:
            lines.append(f"  • {v.violation_name}: {v.explanation}")
            if v.required_action:
                lines.append(f"    Action: {v.required_action}")
        lines.append("")

    # Critical blind spots
    critical_blindspots = [b for b in pm_review.blind_spots if b.severity == "CRITICAL"]
    if critical_blindspots:
        lines.append("CRITICAL BLIND SPOTS:")
        for b in critical_blindspots:
            lines.append(f"  • {b.blind_spot_type.value}: {b.description}")
            if b.suggested_action:
                lines.append(f"    Action: {b.suggested_action}")
        lines.append("")

    # Contradictions
    if pm_review.contradictions:
        lines.append("CONTRADICTIONS:")
        for c in pm_review.contradictions:
            lines.append(f"  • {c.contradiction_type}: {c.description}")
            if c.recommendation:
                lines.append(f"    Fix: {c.recommendation}")
        lines.append("")

    # Missing outputs
    if pm_review.required_outputs:
        missing = [k for k, v in pm_review.required_outputs.items() if not v]
        if missing:
            lines.append("MISSING REQUIRED OUTPUTS:")
            for item in missing:
                lines.append(f"  • {item}")
            lines.append("")

    # Dimension scores
    lines.append("DIMENSION SCORES:")
    lines.append(f"  • Thesis Coherence: {pm_review.dimension_scores.thesis_coherence}/5")
    lines.append(f"  • Numerical Precision: {pm_review.dimension_scores.numerical_precision}/5")
    lines.append(f"  • Qual/Quant Consistency: {pm_review.dimension_scores.qual_quant_consistency}/5")

    return "\n".join(lines)


def _fallback_patch(
    original_draft: str,
    new_evidence: str,
    pm_review: PMReview,
) -> str:
    """
    Simple fallback patching strategy if LLM patching fails.

    Just appends new evidence with clear section markers.
    """
    sections = [original_draft]
    sections.append("\n\n" + "=" * 80)
    sections.append("UPDATED ANALYSIS (Addressing PM Feedback)")
    sections.append("=" * 80 + "\n")
    sections.append(new_evidence)

    return "\n".join(sections)


# ============================================================================
# PATCH VALIDATION
# ============================================================================

def validate_patch_quality(
    original_draft: str,
    patched_draft: str,
    pm_review: PMReview,
) -> Dict[str, any]:
    """
    Quick validation to ensure patching didn't break things.

    Args:
        original_draft: Original text
        patched_draft: Patched text
        pm_review: Original PM review

    Returns:
        Dict with validation metrics
    """
    return {
        "length_change_pct": _calculate_length_change(original_draft, patched_draft),
        "structure_preserved": _check_structure_preserved(original_draft, patched_draft),
        "issues_addressed": _check_issues_addressed(patched_draft, pm_review),
    }


def _calculate_length_change(original: str, patched: str) -> float:
    """Calculate percentage change in document length."""
    if not original:
        return 0.0
    return ((len(patched) - len(original)) / len(original)) * 100


def _check_structure_preserved(original: str, patched: str) -> bool:
    """
    Check if major section headers are preserved.
    """
    # Common section headers in equity research
    headers = [
        "Investment Thesis",
        "Financial Outlook",
        "Valuation",
        "Technical Analysis",
        "Risks",
        "Catalysts",
    ]

    original_lower = original.lower()
    patched_lower = patched.lower()

    # Check that headers present in original are still in patched
    for header in headers:
        if header.lower() in original_lower:
            if header.lower() not in patched_lower:
                return False

    return True


def _check_issues_addressed(patched: str, pm_review: PMReview) -> int:
    """
    Count how many PM issues appear to be addressed in the patched draft.

    This is a heuristic check - not perfect, but gives a sense of coverage.
    """
    patched_lower = patched.lower()
    addressed_count = 0

    # Check if missing required outputs now appear
    if pm_review.required_outputs:
        for output_name, was_present in pm_review.required_outputs.items():
            if not was_present:
                # Check if keywords from output name now appear
                keywords = output_name.lower().replace("_", " ").split()
                if any(kw in patched_lower for kw in keywords):
                    addressed_count += 1

    return addressed_count
