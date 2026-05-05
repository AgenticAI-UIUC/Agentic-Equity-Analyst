#!/bin/bash
# Quick test script for PM Challenge Loop (Phase 1)

echo "=================================="
echo "PM CHALLENGE LOOP - PHASE 1 TEST"
echo "=================================="
echo ""

# Check if running from correct directory
if [ ! -f "main.py" ]; then
    echo "Error: Please run this script from the Agentic-Equity-Analyst directory"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Virtual environment not activated"
    echo "Run: source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Running test case: Apple (AAPL) - 2026"
echo "========================================"
echo ""

python main.py \
    --company "Apple" \
    --ticker AAPL \
    --year 2026 \
    --enable-pm \
    --file test_report.txt \
    --pm-review-file test_pm_review.txt \
    --ic-memo-file test_ic_memo.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "TEST COMPLETE - CHECKING OUTPUTS"
    echo "=================================="
    echo ""

    # Check if files exist
    if [ -f "test_report.txt" ]; then
        echo "✓ test_report.txt created ($(wc -l < test_report.txt) lines)"
    else
        echo "✗ test_report.txt NOT FOUND"
    fi

    if [ -f "test_pm_review.txt" ]; then
        echo "✓ test_pm_review.txt created ($(wc -l < test_pm_review.txt) lines)"
        echo ""
        echo "PM VERDICT:"
        grep "VERDICT:" test_pm_review.txt
        echo ""
        echo "DIMENSION SCORES:"
        grep -A 3 "DIMENSION SCORES:" test_pm_review.txt
    else
        echo "✗ test_pm_review.txt NOT FOUND"
    fi

    if [ -f "test_ic_memo.txt" ]; then
        echo ""
        echo "✓ test_ic_memo.txt created ($(wc -l < test_ic_memo.txt) lines)"
        echo ""
        echo "IC RECOMMENDATION:"
        grep -A 2 "INVESTMENT COMMITTEE RECOMMENDATION" test_ic_memo.txt
    else
        echo "✗ test_ic_memo.txt NOT FOUND"
    fi

    echo ""
    echo "=================================="
    echo "QUICK VIEW"
    echo "=================================="
    echo ""
    echo "To view full outputs:"
    echo "  cat test_report.txt"
    echo "  cat test_pm_review.txt"
    echo "  cat test_ic_memo.txt"
    echo ""
    echo "To clean up test files:"
    echo "  rm test_*.txt"
    echo ""
else
    echo ""
    echo "=================================="
    echo "TEST FAILED"
    echo "=================================="
    echo ""
    echo "Check error messages above."
    exit 1
fi
