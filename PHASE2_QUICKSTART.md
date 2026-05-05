# Phase 2 Quick Start Guide

## TL;DR

Phase 2 adds **automatic revision** - the PM now fixes its own critiques instead of just identifying them.

```bash
# Old way (Phase 1): PM critiques but doesn't fix
python main.py --company "Nvidia" --ticker NVDA --year 2026 --enable-pm

# New way (Phase 2): PM critiques AND fixes automatically
python main.py --company "Nvidia" --ticker NVDA --year 2026 --auto-revise
```

---

## Usage Examples

### Example 1: Basic Auto-Revision

```bash
python main.py \
  --company "Apple Inc." \
  --ticker AAPL \
  --year 2026 \
  --auto-revise
```

**What happens:**
1. Generates initial draft
2. PM reviews → finds issues
3. Re-queries specialists to fix issues
4. Patches draft with new evidence
5. PM re-reviews
6. Repeats until approved (max 3 iterations)

**Outputs:**
- `report.txt` - Final revised report
- `pm_review.txt` - Final PM verdict
- `ic_memo.txt` - Investment Committee memo
- `pm_requery_summary.txt` - Log of what was fixed

---

### Example 2: Custom Iteration Limit

```bash
python main.py \
  --company "Tesla" \
  --ticker TSLA \
  --year 2026 \
  --auto-revise \
  --max-pm-iterations 2
```

**When to use:**
- Faster testing (2 iterations usually enough)
- Cost control
- Time-constrained analysis

---

### Example 3: Custom Output Files

```bash
python main.py \
  --company "Microsoft" \
  --ticker MSFT \
  --year 2026 \
  --auto-revise \
  --file msft_final_report.txt \
  --pm-review-file msft_pm_review.txt \
  --ic-memo-file msft_ic_memo.txt \
  --requery-summary-file msft_requery_log.txt
```

**When to use:**
- Analyzing multiple companies
- Organizing outputs by ticker
- Archiving reports

---

## Command Comparison

| Command | Phase | What It Does | Output Files |
|---------|-------|-------------|-------------|
| No flags | None | Standard pipeline | 1 (report.txt) |
| `--enable-pm` | 1 | PM critiques only | 3 (report + pm_review + ic_memo) |
| `--auto-revise` | 2 | PM critiques + fixes | 4 (report + pm_review + ic_memo + requery_summary) |

---

## What Gets Fixed Automatically?

Phase 2 automatically addresses:

### ✅ Numerical Issues
- DCF growth assumptions too high → Re-queries `find_dcf_tool`
- Missing sensitivity tables → Adds them
- WACC not justified → Fetches supporting data

### ✅ Missing Elements
- No bear/base/bull scenarios → Generates them
- No thesis invalidators → Identifies them
- No catalyst timeline → Creates one

### ✅ Contradictions
- News bearish but thesis bullish → Reconciles with new analysis
- DCF vs. management guidance mismatch → Aligns assumptions
- Technical vs. fundamental divergence → Explains the gap

### ✅ Evidence Gaps
- Stale data → Fetches recent data
- Missing filings data → Re-queries `analyze_filings`
- Weak qualitative support → Re-queries `analyze_news`

---

## Reading the Outputs

### 1. Console Output

```
PHASE 2: PM REVIEW - ITERATION 1/3
================================================================================
✓ PM Review Complete (Iteration 1)
  Verdict: REVISE
  Thesis Coherence: 2/5
  Numerical Precision: 3/5
  Qual/Quant Consistency: 2/5
  Blind Spots: 4
  Rule Violations: 2

PHASE 2: ROUTING PM ISSUES TO SPECIALISTS (Iteration 1)
================================================================================
✓ Routing complete:
  • find_dcf_tool: 2 queries
  • analyze_filings: 1 queries
  • analyze_news: 1 queries

[... re-querying and patching happens ...]

PHASE 2: PM REVIEW - ITERATION 2/3
================================================================================
✓ PM Review Complete (Iteration 2)
  Verdict: APPROVE
  Thesis Coherence: 4/5
  Numerical Precision: 5/5
  Qual/Quant Consistency: 4/5

🎉 PM APPROVED the report on iteration 2!
```

**Key metrics:**
- **Iterations:** Lower is better (means draft was good or fixes worked quickly)
- **Verdict progression:** REVISE → REVISE → APPROVE (typical)
- **Score improvement:** 2/5 → 4/5 shows fixing worked

---

### 2. `pm_requery_summary.txt`

Shows what specialists were called and what they returned:

```
ITERATION 1
================================================================================

### find_dcf_tool
Queries: 2
  Query 1: Verify revenue CAGR assumptions for NVDA 2026...
  Result: Updated DCF analysis shows 12% CAGR vs. 18% previously...

  Query 2: Add sensitivity table for WACC and terminal growth...
  Result: Sensitivity analysis: WACC 8-12% yields PT $180-$220...

### analyze_news
Queries: 1
  Query 1: Check recent sentiment vs. bullish thesis...
  Result: Recent news mixed: Q3 beat but Q4 guide cautious...
```

**Use this to:**
- Understand what changed
- Verify fixes make sense
- Debug if revision didn't work

---

### 3. `report.txt` (Final Revised)

Compare initial vs. final draft:

**Before (Iteration 1):**
```
Price Target: $220 (18% revenue CAGR assumed)
Recommendation: BUY
```

**After (Iteration 2):**
```
## Valuation

Base Case Price Target: $185 (12% revenue CAGR, aligned with management guidance)

Sensitivity Analysis:
- Bear (WACC 12%, 10% CAGR): $150
- Base (WACC 10%, 12% CAGR): $185
- Bull (WACC 8%, 15% CAGR): $230

Recommendation: BUY with caution given recent mixed news sentiment
```

**What changed:**
- Revenue CAGR lowered to realistic level
- Sensitivity table added
- News context incorporated
- Recommendation nuanced

---

## Tips & Best Practices

### 1. Start with 2 Iterations for Testing

```bash
--max-pm-iterations 2
```

- Faster
- Cheaper
- Usually sufficient

Increase to 3 only if reports consistently need more iterations.

---

### 2. Review `pm_requery_summary.txt` First

Before reading the final report, check:
- What issues were found
- What specialists were called
- What new evidence was added

This gives context for changes in final report.

---

### 3. Compare Iteration Scores

Track dimension score improvement:

| Iteration | Thesis | Numerical | Qual/Quant |
|-----------|--------|-----------|------------|
| 1 | 2 | 3 | 2 |
| 2 | 4 | 5 | 4 |

If scores don't improve → check `pm_requery_summary.txt` to see if specialists returned useful data.

---

### 4. Watch for Verdict Patterns

**Good pattern:**
- Iteration 1: REVISE
- Iteration 2: APPROVE

**Concerning pattern:**
- Iteration 1: NEEDS_MORE_RESEARCH
- Iteration 2: NEEDS_MORE_RESEARCH
- Iteration 3: NEEDS_MORE_RESEARCH

If verdict doesn't improve → may need manual intervention or more specialist data.

---

## Troubleshooting

### Issue: "Maximum iterations reached" but verdict still REVISE

**Cause:** Specialists didn't return useful new evidence

**Fix:**
1. Check `pm_requery_summary.txt` - were specialist responses empty/generic?
2. Increase `--max-pm-iterations` to 4 or 5
3. Check if ChromaDB collections are populated (run hydration scripts)

---

### Issue: Patch makes report too long

**Cause:** LLM-based patching sometimes adds too much detail

**Fix:**
1. Check `validate_patch_quality()` length change metric
2. If >30%, may want to manually trim
3. Future: will add length constraints to patcher prompt

---

### Issue: New evidence contradicts original thesis

**Cause:** Original draft assumptions were wrong

**Fix:**
1. This is actually GOOD - PM caught a bad thesis
2. Review `ic_memo.txt` - PM should flag this
3. Consider changing recommendation or refining thesis

---

### Issue: Scores don't improve across iterations

**Cause:** Routing may not be calling right specialists

**Fix:**
1. Review `pm_routing.py` routing tables
2. Check if blind spot types map to correct agents
3. May need to manually adjust routing for specific issue types

---

## Performance Tips

### Reduce Cost

- Use `--max-pm-iterations 2` instead of 3
- More iterations = more LLM calls = higher cost

### Reduce Time

- Phase 2 adds ~2-4 minutes per iteration
- Iteration 1: ~3-4 minutes
- Iteration 2: ~3-4 minutes
- Total: ~6-8 minutes for 2 iterations

**Fastest setup:**
```bash
--auto-revise --max-pm-iterations 1
```

**Balanced setup:**
```bash
--auto-revise --max-pm-iterations 2
```

**Thorough setup:**
```bash
--auto-revise --max-pm-iterations 3
```

---

## FAQ

**Q: Should I always use `--auto-revise`?**

A: For production reports, yes. For quick drafts or testing, `--enable-pm` (Phase 1) is faster.

---

**Q: What if PM approves on iteration 1?**

A: Great! Loop exits immediately. Only pays for 1 PM review.

---

**Q: Can I manually edit the report after Phase 2?**

A: Yes, `report.txt` is just a text file. But Phase 2 should catch most issues automatically.

---

**Q: Does Phase 2 cost more than Phase 1?**

A: Yes, ~$0.23 per iteration. 2 iterations = ~$0.46 total. But quality is much higher.

---

**Q: How do I know if revision worked?**

A: Check:
1. Verdict improved (REVISE → APPROVE)
2. Dimension scores increased
3. `pm_requery_summary.txt` shows specialist calls
4. `report.txt` has new content

---

## Quick Reference

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--auto-revise` | bool | False | Enable Phase 2 |
| `--max-pm-iterations` | int | 3 | Max revision cycles |
| `--requery-summary-file` | str | pm_requery_summary.txt | Re-query log file |
| `--enable-pm` | bool | False | Enable Phase 1 only |
| `--file` | str | report.txt | Final report file |
| `--pm-review-file` | str | pm_review.txt | PM review file |
| `--ic-memo-file` | str | ic_memo.txt | IC memo file |

### Output Files

| File | Phase 1 | Phase 2 | Contains |
|------|---------|---------|----------|
| `report.txt` | ✓ | ✓ | Draft (Phase 1) or Revised (Phase 2) |
| `pm_review.txt` | ✓ | ✓ | PM critique |
| `ic_memo.txt` | ✓ | ✓ | IC recommendation |
| `pm_requery_summary.txt` | ✗ | ✓ | What was fixed |

### Typical Verdicts by Iteration

| Iteration | Typical Verdict | Typical Scores |
|-----------|----------------|----------------|
| 1 | REVISE | 2-3/5 |
| 2 | APPROVE or REVISE | 4-5/5 |
| 3 | APPROVE | 4-5/5 |

---

**Ready to test?**

```bash
source .venv/bin/activate
./test_phase2.sh
```

or

```bash
python main.py --company "Apple" --ticker AAPL --year 2026 --auto-revise
```

Good luck! 🚀
