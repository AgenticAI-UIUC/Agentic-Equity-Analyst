"""Command-line entry point for generating equity research reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from reporting_pipeline import generate_financial_report, generate_financial_report_with_pm, generate_financial_report_with_pm_revision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Agentic Equity Analyst pipeline and persist the output report.",
    )
    parser.add_argument("--company", required=True, help="Company name to analyze.")
    parser.add_argument(
        "--ticker",
        help="Optional stock ticker symbol to pass to downstream tools (e.g., AAPL, MSFT).",
    )
    parser.add_argument("--year", required=True, help="Fiscal/forecast year for the outlook (e.g., 2026).")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional custom natural-language prompt. Overrides --company/--year wording.",
    )
    parser.add_argument(
        "--file",
        default="report.txt",
        help="Destination file for the generated report (defaults to report.txt).",
    )
    parser.add_argument(
        "--launch-ui",
        action="store_true",
        help="Launch the Streamlit viewer after the report is written.",
    )
    parser.add_argument(
        "--enable-pm",
        action="store_true",
        help="Enable PM Challenge Loop (Phase 1: critique only, no revision yet).",
    )
    parser.add_argument(
        "--pm-review-file",
        default="pm_review.txt",
        help="Destination file for PM review output (defaults to pm_review.txt).",
    )
    parser.add_argument(
        "--ic-memo-file",
        default="ic_memo.txt",
        help="Destination file for Investment Committee memo (defaults to ic_memo.txt).",
    )
    parser.add_argument(
        "--auto-revise",
        action="store_true",
        help="Enable PM Challenge Loop Phase 2: automatic revision based on PM feedback.",
    )
    parser.add_argument(
        "--max-pm-iterations",
        type=int,
        default=3,
        help="Maximum number of PM revision iterations (default: 3). Only applies with --auto-revise.",
    )
    parser.add_argument(
        "--requery-summary-file",
        default="pm_requery_summary.txt",
        help="Destination file for specialist re-query summary (defaults to pm_requery_summary.txt). Only applies with --auto-revise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.auto_revise:
        # Use PM Challenge Loop Phase 2: Automatic Revision
        print(f"\n🚀 Running Phase 2: PM Challenge Loop with Automatic Revision")
        print(f"Max Iterations: {args.max_pm_iterations}")
        print("=" * 80)

        report_text, pm_review, ic_memo, iterations = generate_financial_report_with_pm_revision(
            company=args.company,
            ticker=args.ticker,
            year=args.year,
            custom_prompt=args.prompt,
            file_path=args.file,
            pm_output_file=args.pm_review_file,
            ic_memo_file=args.ic_memo_file,
            requery_summary_file=args.requery_summary_file,
            max_iterations=args.max_pm_iterations,
            launch_ui=args.launch_ui,
        )

        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"\nGenerated outputs:")
        print(f"  • Final Report: {Path(args.file).resolve()}")
        print(f"  • PM Review: {Path(args.pm_review_file).resolve()}")
        print(f"  • IC Memo: {Path(args.ic_memo_file).resolve()}")
        print(f"  • Re-query Summary: {Path(args.requery_summary_file).resolve()}")

        if pm_review:
            print(f"\nFinal PM Verdict: {pm_review.verdict.value}")
            print(f"PM Confidence: {pm_review.confidence_level.value}")
            print(f"Iterations: {iterations}/{args.max_pm_iterations}")

        if ic_memo:
            print(f"\nIC Verdict: {ic_memo.verdict.value}")
            print(f"IC Recommendation: {ic_memo.ic_recommendation}")
            print(f"Decision Readiness: {ic_memo.decision_readiness_score}/100")

    elif args.enable_pm:
        # Use PM Challenge Loop Phase 1: Critique Only
        print(f"\n🔍 Running Phase 1: PM Challenge Loop (Critique Only)")
        print("=" * 80)

        report_text, pm_review, ic_memo = generate_financial_report_with_pm(
            company=args.company,
            ticker=args.ticker,
            year=args.year,
            custom_prompt=args.prompt,
            file_path=args.file,
            pm_output_file=args.pm_review_file,
            ic_memo_file=args.ic_memo_file,
            launch_ui=args.launch_ui,
            enable_pm_challenge=True,
        )

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nGenerated outputs:")
        print(f"  • Report: {Path(args.file).resolve()}")
        print(f"  • PM Review: {Path(args.pm_review_file).resolve()}")
        print(f"  • IC Memo: {Path(args.ic_memo_file).resolve()}")

        if pm_review:
            print(f"\nPM Verdict: {pm_review.verdict.value}")
            print(f"PM Confidence: {pm_review.confidence_level.value}")

        if ic_memo:
            print(f"IC Verdict: {ic_memo.verdict.value}")
            print(f"IC Recommendation: {ic_memo.ic_recommendation}")

        print(f"\nℹ️  Note: To enable automatic revision based on PM feedback, use --auto-revise flag")

    else:
        # Use original pipeline (no PM)
        print(f"\n📊 Running Standard Pipeline (No PM Challenge)")
        print("=" * 80)

        report_text = generate_financial_report(
            company=args.company,
            ticker=args.ticker,
            year=args.year,
            custom_prompt=args.prompt,
            file_path=args.file,
            launch_ui=args.launch_ui,
        )
        print("\nGenerated report saved to", Path(args.file).resolve())
        print("\nPreview:\n" + report_text[:1000])

        print(f"\nℹ️  Note: To enable PM Challenge Loop, use --enable-pm or --auto-revise flags")


if __name__ == "__main__":
    main()
