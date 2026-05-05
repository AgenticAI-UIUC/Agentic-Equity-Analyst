"""Pydantic schemas for PM agent structured outputs."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ===== ENUMS =====

class Verdict(str, Enum):
    """PM verdict on thesis readiness."""
    APPROVE = "APPROVE"
    REVISE = "REVISE"
    NEEDS_MORE_RESEARCH = "NEEDS_MORE_RESEARCH"


class ConfidenceLevel(str, Enum):
    """PM confidence in thesis quality."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Severity(str, Enum):
    """Issue severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"


class BlindSpotType(str, Enum):
    """Taxonomy of blind spot categories."""
    ASSUMPTION_RISK = "assumption_risk"
    NARRATIVE_DRIFT = "narrative_drift"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    QUAL_QUANT_CONTRADICTION = "qual_quant_contradiction"
    MISSING_SCENARIO = "missing_scenario"
    CATALYST_AMBIGUITY = "catalyst_ambiguity"
    TEMPORAL_MISMATCH = "temporal_mismatch"


class ICMemoVerdict(str, Enum):
    """Investment Committee memo verdict."""
    GO = "GO"
    CONDITIONAL_GO = "CONDITIONAL_GO"
    NO_GO_REVISE = "NO_GO_REVISE"
    NO_GO_REJECT = "NO_GO_REJECT"


# ===== DIMENSION SCORES =====

class PMReviewDimension(BaseModel):
    """Score for a single review dimension."""
    score: int = Field(..., ge=1, le=5, description="1=Weak, 5=Strong")
    issues: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)


# ===== VIOLATIONS & CONTRADICTIONS =====

class RuleViolation(BaseModel):
    """A specific hard rule that was violated."""
    rule_id: str = Field(..., description="e.g., 'V1', 'A3', 'Q1'")
    severity: Severity
    message: str
    required_action: str


class ContradictionFlag(BaseModel):
    """A detected contradiction between outputs."""
    check_type: str = Field(
        ...,
        description="dcf_vs_filings, dcf_vs_news, narrative_vs_metrics, target_vs_uncertainty"
    )
    severity: Severity
    description: str
    evidence_dcf: Optional[str] = None
    evidence_specialist: Optional[str] = None
    recommendation: str


# ===== BLIND SPOTS =====

class BlindSpot(BaseModel):
    """A specific blind spot identified by PM."""
    type: BlindSpotType
    subtype: str = Field(..., description="e.g., '1a', '4c'")
    severity: Severity
    description: str
    evidence_gap: str

    # Routing information
    route_to_agent: List[str] = Field(
        ...,
        description="Which specialist tools to re-query"
    )
    required_action: str

    # Impact assessment
    impact_on_thesis: str
    affected_sections: List[str] = Field(
        default_factory=list,
        description="Which sections of draft need updating"
    )


# ===== CORE PM REVIEW =====

class PMReview(BaseModel):
    """Complete PM review output for Phase 1."""

    # Metadata
    company: str
    ticker: str
    review_date: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Verdict
    verdict: Verdict
    confidence_level: ConfidenceLevel

    # Dimension scores
    thesis_coherence: PMReviewDimension
    numerical_precision: PMReviewDimension
    qual_quant_consistency: PMReviewDimension

    # Issues found
    rule_violations: List[RuleViolation] = Field(default_factory=list)
    contradictions: List[ContradictionFlag] = Field(default_factory=list)
    blind_spots: List[BlindSpot] = Field(default_factory=list)

    # Required outputs checklist
    required_outputs_present: dict = Field(
        default_factory=lambda: {
            "tri_scenario_valuation": False,
            "sensitivity_table": False,
            "assumptions_table": False,
            "value_bridge": False,
            "invalidation_criteria": False
        }
    )

    # Routing for revision
    agents_to_requery: List[str] = Field(
        default_factory=list,
        description="Prioritized list of specialist tools to call"
    )

    # Summary
    executive_summary: str

    # Statistics
    rules_passed: int = 0
    rules_failed: int = 0
    blind_spot_summary: dict = Field(
        default_factory=dict,
        description="Count of blind spots by type"
    )


# ===== INVESTMENT COMMITTEE MEMO (Phase 1 Simplified Version) =====

class ThesisStrength(BaseModel):
    """What's working well in the thesis."""
    category: str
    description: str
    supporting_evidence: str


class ThesisWeakness(BaseModel):
    """What must be fixed."""
    category: str
    severity: Severity
    description: str
    required_fix: str
    estimated_impact: str
    is_blocking: bool


class ThesisInvalidator(BaseModel):
    """What would break the investment case."""
    scenario: str
    probability: str
    monitoring_metric: str
    trigger_threshold: str


class InvestmentCommitteeMemo(BaseModel):
    """
    Investment Committee decision memo.
    Human-readable output from PM agent.
    """

    # Header
    company: str
    ticker: str
    sector: Optional[str] = None
    current_price: Optional[float] = None
    price_target_base: Optional[float] = None
    recommendation: Optional[str] = None

    # Verdict
    verdict: ICMemoVerdict
    verdict_rationale: str

    # One-line thesis
    thesis_summary: str = Field(..., max_length=300)

    # Quality assessment
    overall_thesis_quality: str  # STRONG, ADEQUATE, WEAK
    confidence_level: ConfidenceLevel
    decision_readiness_score: int = Field(..., ge=0, le=100)

    # What's strong
    strengths: List[ThesisStrength] = Field(min_items=1, max_items=5)

    # What must be fixed
    weaknesses: List[ThesisWeakness] = Field(default_factory=list)
    blocking_issues: List[str] = Field(default_factory=list)

    # What would change my mind
    thesis_invalidators: List[ThesisInvalidator] = Field(min_items=1, max_items=5)

    # Risk/reward (simplified for Phase 1)
    risk_reward_ratio: Optional[float] = None
    asymmetry_assessment: str = "Unknown"  # Favorable, Neutral, Unfavorable, Unknown

    # IC recommendation
    ic_recommendation: str
    conditions_for_approval: List[str] = Field(default_factory=list)

    # Next steps
    if_approved_next_steps: List[str] = Field(default_factory=list)
    if_rejected_next_steps: List[str] = Field(default_factory=list)

    # Metadata
    pm_agent_version: str = "v1.0-phase1"
    review_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    review_iterations: int = 0

    # Technical details (for debugging)
    technical_details: dict = Field(default_factory=dict)


__all__ = [
    "Verdict",
    "ConfidenceLevel",
    "Severity",
    "BlindSpotType",
    "ICMemoVerdict",
    "PMReviewDimension",
    "RuleViolation",
    "ContradictionFlag",
    "BlindSpot",
    "PMReview",
    "ThesisStrength",
    "ThesisWeakness",
    "ThesisInvalidator",
    "InvestmentCommitteeMemo",
]
