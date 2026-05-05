"""Portfolio Manager agent for challenging equity research drafts.

Phase 1: Basic PM agent that critiques drafts and outputs structured issues.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from pm_schemas import (
    PMReview,
    InvestmentCommitteeMemo,
    PMReviewDimension,
    RuleViolation,
    ContradictionFlag,
    BlindSpot,
    ThesisStrength,
    ThesisWeakness,
    ThesisInvalidator,
    Verdict,
    ConfidenceLevel,
    Severity,
    BlindSpotType,
    ICMemoVerdict,
)


# PM Agent System Prompt
PM_SYSTEM_PROMPT = """# ROLE

You are a Portfolio Manager conducting Investment Committee review of an equity research draft.

Your mandate: **Adversarial validation of thesis quality before capital deployment.**

You do NOT:
- Generate new research from scratch
- Rewrite the draft optimistically
- Duplicate the reporting agent's work

You DO:
- Attack the draft to find weaknesses
- Identify unsupported claims and contradictions
- Demand numerical rigor
- Decide if the thesis is ready for capital deployment

## CORE QUESTION

"Would I deploy institutional capital based on this thesis, or does it have blind spots that could cause losses?"

---

# VALIDATION FRAMEWORK

## A. Score 3 Key Dimensions (1=Weak, 5=Strong)

### 1. Thesis Coherence
- Is there ONE clear investment stance (Long/Short/Neutral)?
- Do all sections support that stance?
- Are there internal contradictions?
- Is recommendation consistent with price target?

### 2. Numerical Precision
- Are revenue growth, margins, WACC, terminal growth, share count explicit?
- Are conclusions tied to actual valuation outputs?
- Are ranges and sensitivities discussed?
- Is there a base/bull/bear scenario?

### 3. Qualitative-Quantitative Consistency
- Does news sentiment support or contradict forecast assumptions?
- Is management commentary aligned with modeled growth/margins?
- Are macro/regulatory/headline risks reflected in valuation?
- Do technicals contradict the fundamental thesis?

---

## B. Enforce Hard Rules (Binary Pass/Fail)

### Valuation Completeness
- V1: Base/Bull/Bear scenarios provided
- V2: Sensitivity table (≥2 variables) present
- V3: WACC stated + components shown
- V4: Terminal growth rate stated
- V5: Terminal growth ≤5% OR justified if higher
- V6: Share count/dilution explicit

### Assumption Transparency
- A1: Revenue growth assumption (CAGR or annual) stated
- A2: Margin assumptions (gross, operating, net) stated
- A3: If margin expansion is driver → operational bridge shown
- A4: Capex % of sales or path shown
- A5: Value drivers quantified (% from growth vs margin vs multiple)

### Qual-Quant Alignment
- Q1: If news negative → addressed in forecast or risk section
- Q2: If regulatory/litigation risk → downside scenario shown
- Q3: If management guided down → DCF reflects OR contrarian view justified
- Q4: If technicals bearish → catalyst for contrarian view stated

### Risk & Invalidation
- R1: Thesis invalidation criteria stated
- R2: If upside >30% → uncertainty acknowledged
- R3: If downside scenario exists → probability/condition stated

### Recommendation Logic
- L1: Recommendation matches valuation math
- L2: Price target stated
- L3: Upside/downside % calculated explicitly
- L4: Risk/reward asymmetry discussed

**If ANY critical rule fails → cannot APPROVE.**

---

## C. Identify Blind Spots (7 Categories)

When you find issues, classify them:

1. **assumption_risk** — Model inputs unsupported or unrealistic
2. **narrative_drift** — Conclusion tone stronger than evidence
3. **evidence_mismatch** — Claim uses weak/stale/irrelevant evidence
4. **qual_quant_contradiction** — Qualitative signals conflict with model
5. **missing_scenario** — Thesis fragile to one variable, no sensitivity
6. **catalyst_ambiguity** — Fair value identified but no path to realization
7. **temporal_mismatch** — Long-term vs near-term inconsistency

---

# DECISION LOGIC

## Verdict Determination

**APPROVE:**
- All dimension scores ≥3
- No CRITICAL rule violations
- All required numerical outputs present
- Contradictions are minor or justified
- Blind spots ≤2 MODERATE

**REVISE:**
- 1-2 dimension scores = 2
- ≤2 CRITICAL violations (fixable with targeted edits)
- 1-2 required outputs missing
- Some contradictions need reconciliation

**NEEDS_MORE_RESEARCH:**
- ≥1 dimension score = 1
- ≥3 CRITICAL violations
- ≥2 required outputs missing
- Major contradictions across multiple areas

## Confidence Level

**HIGH:** All scores ≥4, no contradictions, all required outputs present
**MEDIUM:** Scores average 3-4, minor contradictions addressed
**LOW:** Scores average <3, unresolved contradictions

---

# TONE & STYLE

- **Be adversarial, not accommodating.** Your job is to find problems.
- **Be specific, not vague.** Not "needs more evidence" but "DCF growth of 18% exceeds management guidance of 12% with no catalyst identified."
- **Be quantitative.** State the numerical gap: "Price target overstated by 15-20% if this assumption corrects."
- **Be actionable.** Don't just flag issues—route them to specific agents with specific queries.
- **Be decisive.** Don't hedge. If it's not ready, say NEEDS_MORE_RESEARCH with clear actions.

Remember: **Your job is to protect capital by catching blind spots before deployment, not to be optimistic or generous with approval.**
"""


class PMAgent:
    """Portfolio Manager agent for reviewing equity research drafts.

    Phase 1: Basic critique with structured output.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1):
        """Initialize PM agent with LLM."""
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=60  # PM review can take longer
        )
        self.system_prompt = PM_SYSTEM_PROMPT

    def review_draft(
        self,
        draft: str,
        company: str,
        ticker: str,
        specialist_outputs: Dict[str, Any] = None
    ) -> PMReview:
        """
        Review an equity research draft and return structured critique.

        Args:
            draft: The research report text
            company: Company name
            ticker: Stock ticker
            specialist_outputs: Optional dict of raw specialist tool outputs

        Returns:
            PMReview with structured issues and scores
        """

        # Build review prompt
        review_prompt = f"""Review the following equity research draft for {company} ({ticker}).

DRAFT REPORT:
{draft}

"""

        if specialist_outputs:
            review_prompt += f"""
SPECIALIST OUTPUTS (for cross-validation):
{json.dumps(specialist_outputs, indent=2, default=str)}
"""

        review_prompt += """

Perform a thorough PM review and return your assessment in structured JSON format with these fields:

{
  "verdict": "APPROVE" | "REVISE" | "NEEDS_MORE_RESEARCH",
  "confidence_level": "HIGH" | "MEDIUM" | "LOW",
  "thesis_coherence": {
    "score": 1-5,
    "issues": ["..."],
    "strengths": ["..."]
  },
  "numerical_precision": {
    "score": 1-5,
    "issues": ["..."],
    "strengths": ["..."]
  },
  "qual_quant_consistency": {
    "score": 1-5,
    "issues": ["..."],
    "strengths": ["..."]
  },
  "rule_violations": [
    {
      "rule_id": "V1",
      "severity": "CRITICAL",
      "message": "...",
      "required_action": "..."
    }
  ],
  "contradictions": [
    {
      "check_type": "dcf_vs_filings",
      "severity": "HIGH",
      "description": "...",
      "evidence_dcf": "...",
      "evidence_specialist": "...",
      "recommendation": "..."
    }
  ],
  "blind_spots": [
    {
      "type": "assumption_risk",
      "subtype": "1a",
      "severity": "CRITICAL",
      "description": "...",
      "evidence_gap": "...",
      "route_to_agent": ["find_dcf_tool", "analyze_filings"],
      "required_action": "...",
      "impact_on_thesis": "...",
      "affected_sections": ["valuation_section", "assumptions_table"]
    }
  ],
  "required_outputs_present": {
    "tri_scenario_valuation": true/false,
    "sensitivity_table": true/false,
    "assumptions_table": true/false,
    "value_bridge": true/false,
    "invalidation_criteria": true/false
  },
  "agents_to_requery": ["find_dcf_tool", "analyze_filings"],
  "executive_summary": "2-3 sentence summary",
  "rules_passed": 0,
  "rules_failed": 0,
  "blind_spot_summary": {"assumption_risk": 1, ...}
}

Be thorough and adversarial. Find the blind spots.
"""

        # Invoke LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=review_prompt)
        ]

        response = self.llm.invoke(messages)
        response_text = response.content

        # Parse JSON response
        try:
            # Extract JSON from response (may be wrapped in ```json blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            review_data = json.loads(response_text)

            # Build PMReview object
            pm_review = PMReview(
                company=company,
                ticker=ticker,
                verdict=Verdict(review_data.get("verdict", "NEEDS_MORE_RESEARCH")),
                confidence_level=ConfidenceLevel(review_data.get("confidence_level", "LOW")),
                thesis_coherence=PMReviewDimension(**review_data.get("thesis_coherence", {"score": 3, "issues": [], "strengths": []})),
                numerical_precision=PMReviewDimension(**review_data.get("numerical_precision", {"score": 3, "issues": [], "strengths": []})),
                qual_quant_consistency=PMReviewDimension(**review_data.get("qual_quant_consistency", {"score": 3, "issues": [], "strengths": []})),
                rule_violations=[RuleViolation(**rv) for rv in review_data.get("rule_violations", [])],
                contradictions=[ContradictionFlag(**c) for c in review_data.get("contradictions", [])],
                blind_spots=[BlindSpot(**bs) for bs in review_data.get("blind_spots", [])],
                required_outputs_present=review_data.get("required_outputs_present", {}),
                agents_to_requery=review_data.get("agents_to_requery", []),
                executive_summary=review_data.get("executive_summary", "Review completed."),
                rules_passed=review_data.get("rules_passed", 0),
                rules_failed=review_data.get("rules_failed", 0),
                blind_spot_summary=review_data.get("blind_spot_summary", {})
            )

            return pm_review

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: create a minimal review if parsing fails
            print(f"Warning: Failed to parse PM response: {e}")
            print(f"Raw response: {response_text[:500]}")

            return PMReview(
                company=company,
                ticker=ticker,
                verdict=Verdict.NEEDS_MORE_RESEARCH,
                confidence_level=ConfidenceLevel.LOW,
                thesis_coherence=PMReviewDimension(score=3, issues=["Failed to parse PM response"], strengths=[]),
                numerical_precision=PMReviewDimension(score=3, issues=[], strengths=[]),
                qual_quant_consistency=PMReviewDimension(score=3, issues=[], strengths=[]),
                executive_summary=f"PM review parsing failed: {str(e)[:200]}"
            )

    def generate_ic_memo(
        self,
        draft: str,
        pm_review: PMReview,
        company: str,
        ticker: str
    ) -> InvestmentCommitteeMemo:
        """
        Generate a human-readable Investment Committee memo.

        Args:
            draft: The research report text
            pm_review: The structured PM review
            company: Company name
            ticker: Stock ticker

        Returns:
            InvestmentCommitteeMemo formatted for IC discussion
        """

        # Build IC memo generation prompt
        ic_prompt = f"""Based on your PM review of the equity research draft for {company} ({ticker}),
generate an Investment Committee memo.

PM REVIEW SUMMARY:
- Verdict: {pm_review.verdict.value}
- Confidence: {pm_review.confidence_level.value}
- Thesis Coherence: {pm_review.thesis_coherence.score}/5
- Numerical Precision: {pm_review.numerical_precision.score}/5
- Qual/Quant Consistency: {pm_review.qual_quant_consistency.score}/5
- Blind Spots: {len(pm_review.blind_spots)}
- Rule Violations: {len(pm_review.rule_violations)}

ORIGINAL DRAFT (first 2000 chars):
{draft[:2000]}...

Generate an IC memo in JSON format with these fields:

{{
  "verdict": "GO" | "CONDITIONAL_GO" | "NO_GO_REVISE" | "NO_GO_REJECT",
  "verdict_rationale": "2-3 sentences",
  "thesis_summary": "One sentence thesis (max 200 chars)",
  "overall_thesis_quality": "STRONG" | "ADEQUATE" | "WEAK",
  "confidence_level": "HIGH" | "MEDIUM" | "LOW",
  "decision_readiness_score": 0-100,
  "strengths": [
    {{
      "category": "numerical_precision",
      "description": "...",
      "supporting_evidence": "..."
    }}
  ],
  "weaknesses": [
    {{
      "category": "assumption_risk",
      "severity": "CRITICAL",
      "description": "...",
      "required_fix": "...",
      "estimated_impact": "...",
      "is_blocking": true
    }}
  ],
  "blocking_issues": ["Issue 1", "Issue 2"],
  "thesis_invalidators": [
    {{
      "scenario": "Revenue growth <10% for 2 quarters",
      "probability": "MEDIUM",
      "monitoring_metric": "Quarterly revenue growth %",
      "trigger_threshold": "<10% for 2 consecutive quarters"
    }}
  ],
  "risk_reward_ratio": 2.1,
  "asymmetry_assessment": "Favorable" | "Neutral" | "Unfavorable",
  "ic_recommendation": "APPROVE for portfolio inclusion" | "CONDITIONAL APPROVE" | "SEND BACK" | "REJECT",
  "conditions_for_approval": ["Condition 1", "Condition 2"],
  "if_approved_next_steps": ["Step 1", "Step 2"],
  "if_rejected_next_steps": ["Step 1", "Step 2"]
}}

Be practical and actionable. This memo will be read by humans making capital allocation decisions.
"""

        # Invoke LLM
        messages = [
            SystemMessage(content="You are a Portfolio Manager writing an Investment Committee memo."),
            HumanMessage(content=ic_prompt)
        ]

        response = self.llm.invoke(messages)
        response_text = response.content

        # Parse JSON response
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            memo_data = json.loads(response_text)

            # Build InvestmentCommitteeMemo object
            ic_memo = InvestmentCommitteeMemo(
                company=company,
                ticker=ticker,
                verdict=ICMemoVerdict(memo_data.get("verdict", "NO_GO_REVISE")),
                verdict_rationale=memo_data.get("verdict_rationale", ""),
                thesis_summary=memo_data.get("thesis_summary", "")[:300],
                overall_thesis_quality=memo_data.get("overall_thesis_quality", "ADEQUATE"),
                confidence_level=ConfidenceLevel(memo_data.get("confidence_level", "MEDIUM")),
                decision_readiness_score=memo_data.get("decision_readiness_score", 50),
                strengths=[ThesisStrength(**s) for s in memo_data.get("strengths", [])],
                weaknesses=[ThesisWeakness(**w) for w in memo_data.get("weaknesses", [])],
                blocking_issues=memo_data.get("blocking_issues", []),
                thesis_invalidators=[ThesisInvalidator(**ti) for ti in memo_data.get("thesis_invalidators", [])],
                risk_reward_ratio=memo_data.get("risk_reward_ratio"),
                asymmetry_assessment=memo_data.get("asymmetry_assessment", "Unknown"),
                ic_recommendation=memo_data.get("ic_recommendation", "SEND BACK for revision"),
                conditions_for_approval=memo_data.get("conditions_for_approval", []),
                if_approved_next_steps=memo_data.get("if_approved_next_steps", []),
                if_rejected_next_steps=memo_data.get("if_rejected_next_steps", []),
                technical_details={
                    "pm_review_verdict": pm_review.verdict.value,
                    "blind_spots_count": len(pm_review.blind_spots),
                    "rule_violations_count": len(pm_review.rule_violations)
                }
            )

            return ic_memo

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback IC memo
            print(f"Warning: Failed to parse IC memo response: {e}")

            return InvestmentCommitteeMemo(
                company=company,
                ticker=ticker,
                verdict=ICMemoVerdict.NO_GO_REVISE,
                verdict_rationale="Failed to generate IC memo due to parsing error.",
                thesis_summary="IC memo generation failed.",
                overall_thesis_quality="WEAK",
                confidence_level=ConfidenceLevel.LOW,
                decision_readiness_score=0,
                strengths=[],
                thesis_invalidators=[
                    ThesisInvalidator(
                        scenario="Unable to assess",
                        probability="UNKNOWN",
                        monitoring_metric="N/A",
                        trigger_threshold="N/A"
                    )
                ],
                ic_recommendation="SEND BACK - IC memo generation failed",
                technical_details={"error": str(e)}
            )


def format_ic_memo_text(memo: InvestmentCommitteeMemo) -> str:
    """Format IC memo as readable text."""

    output = []
    output.append("=" * 80)
    output.append("INVESTMENT COMMITTEE MEMO")
    output.append("=" * 80)
    output.append("")
    output.append(f"Company:         {memo.company}")
    output.append(f"Ticker:          {memo.ticker}")
    if memo.current_price:
        output.append(f"Current Price:   ${memo.current_price:.2f}")
    if memo.price_target_base:
        output.append(f"Price Target:    ${memo.price_target_base:.2f}")
    if memo.recommendation:
        output.append(f"Recommendation:  {memo.recommendation}")
    output.append("")

    output.append("-" * 80)
    output.append("VERDICT")
    output.append("-" * 80)
    output.append(f"{memo.verdict.value}")
    output.append(f"\n{memo.verdict_rationale}")
    output.append("")
    output.append(f"Overall Quality:       {memo.overall_thesis_quality}")
    output.append(f"Confidence Level:      {memo.confidence_level.value}")
    output.append(f"Decision Readiness:    {memo.decision_readiness_score}/100")
    output.append("")

    output.append("-" * 80)
    output.append("ONE-LINE THESIS")
    output.append("-" * 80)
    output.append(memo.thesis_summary)
    output.append("")

    if memo.strengths:
        output.append("-" * 80)
        output.append("WHAT'S STRONG ✓")
        output.append("-" * 80)
        for i, strength in enumerate(memo.strengths, 1):
            output.append(f"{i}. {strength.category.upper()}")
            output.append(f"   {strength.description}")
            output.append(f"   Evidence: {strength.supporting_evidence}")
            output.append("")

    if memo.weaknesses:
        output.append("-" * 80)
        output.append("WHAT MUST BE FIXED ⚠")
        output.append("-" * 80)
        for i, weakness in enumerate(memo.weaknesses, 1):
            blocking = "BLOCKING" if weakness.is_blocking else "Non-blocking"
            output.append(f"{i}. {weakness.category.upper()} ({weakness.severity.value}, {blocking})")
            output.append(f"   Issue: {weakness.description}")
            output.append(f"   Fix: {weakness.required_fix}")
            output.append(f"   Impact: {weakness.estimated_impact}")
            output.append("")

    if memo.blocking_issues:
        output.append("-" * 80)
        output.append("BLOCKING ISSUES 🚫")
        output.append("-" * 80)
        for issue in memo.blocking_issues:
            output.append(f"  • {issue}")
        output.append("")

    output.append("-" * 80)
    output.append("WHAT WOULD CHANGE MY MIND ⚡")
    output.append("-" * 80)
    for i, invalidator in enumerate(memo.thesis_invalidators, 1):
        output.append(f"{i}. {invalidator.scenario}")
        output.append(f"   Probability: {invalidator.probability}")
        output.append(f"   Monitor: {invalidator.monitoring_metric}")
        output.append(f"   Trigger: {invalidator.trigger_threshold}")
        output.append("")

    output.append("-" * 80)
    output.append("INVESTMENT COMMITTEE RECOMMENDATION")
    output.append("-" * 80)
    output.append(memo.ic_recommendation)
    output.append("")

    if memo.conditions_for_approval:
        output.append("Conditions:")
        for condition in memo.conditions_for_approval:
            output.append(f"  • {condition}")
        output.append("")

    output.append("-" * 80)
    output.append("NEXT STEPS")
    output.append("-" * 80)

    if memo.if_approved_next_steps:
        output.append("\nIf Approved:")
        for step in memo.if_approved_next_steps:
            output.append(f"  • {step}")

    if memo.if_rejected_next_steps:
        output.append("\nIf Rejected:")
        for step in memo.if_rejected_next_steps:
            output.append(f"  • {step}")

    output.append("")
    output.append("-" * 80)
    output.append("METADATA")
    output.append("-" * 80)
    output.append(f"PM Agent Version:  {memo.pm_agent_version}")
    output.append(f"Review Date:       {memo.review_date}")
    output.append(f"Review Iterations: {memo.review_iterations}")
    output.append("=" * 80)

    return "\n".join(output)


__all__ = ["PMAgent", "format_ic_memo_text"]
