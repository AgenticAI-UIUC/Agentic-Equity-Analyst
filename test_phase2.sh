#!/bin/bash

# Test script for Phase 2 PM Challenge Loop
# Tests automatic revision functionality

echo "================================================================================================="
echo "PHASE 2 PM CHALLENGE LOOP - TEST SCRIPT"
echo "================================================================================================="
echo ""
echo "This script will test the automatic revision loop (Phase 2) implementation."
echo ""
echo "Test: Run equity analysis with --auto-revise flag"
echo "Company: Apple Inc."
echo "Ticker: AAPL"
echo "Year: 2026"
echo "Max Iterations: 2 (for faster testing)"
echo ""
echo "================================================================================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "Please run: source .venv/bin/activate"
    echo ""
    exit 1
fi

# Run the Phase 2 pipeline
echo "Starting Phase 2 test..."
echo ""

python main.py \
    --company "Apple Inc." \
    --ticker AAPL \
    --year 2026 \
    --auto-revise \
    --max-pm-iterations 2 \
    --file test_phase2_report.txt \
    --pm-review-file test_phase2_pm_review.txt \
    --ic-memo-file test_phase2_ic_memo.txt \
    --requery-summary-file test_phase2_requery_summary.txt

EXIT_CODE=$?

echo ""
echo "================================================================================================="
echo "TEST COMPLETE"
echo "================================================================================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Phase 2 test completed successfully!"
    echo ""
    echo "Generated files:"
    echo "  • test_phase2_report.txt         - Final revised report"
    echo "  • test_phase2_pm_review.txt      - Final PM review"
    echo "  • test_phase2_ic_memo.txt        - Investment Committee memo"
    echo "  • test_phase2_requery_summary.txt - Specialist re-query summary"
    echo ""
    echo "To review outputs:"
    echo "  cat test_phase2_pm_review.txt"
    echo "  cat test_phase2_ic_memo.txt"
    echo "  cat test_phase2_requery_summary.txt"
    echo ""
else
    echo "❌ Phase 2 test failed with exit code: $EXIT_CODE"
    echo ""
    echo "Check the error messages above for details."
    echo ""
fi

echo "================================================================================================="
