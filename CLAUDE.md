# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An agentic equity research platform that orchestrates multiple LLM agents to generate professional financial analysis reports. It uses a RAG architecture with ChromaDB and pulls data from FMP (SEC filings), Yahoo Finance (market data), and Perplexity Sonar (news).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Fill in required API keys
```

**Required `.env` keys:**
- `OPENAI_API_KEY` — GPT-4o + embeddings (text-embedding-3-small)
- `CHROMADB`, `CHROMADB_API_KEY`, `CHROMADB_TENANT` — Chroma Cloud credentials
- `FMP_API_KEY` — Financial Modeling Prep (SEC filings)
- `PPLX_API_KEY` — Perplexity Sonar (news)
- `FRED_API_KEY` — Optional; Federal Reserve macro data

## Common Commands

**Run equity analysis:**
```bash
python main.py --company "Nvidia" --ticker NVDA --year 2026
python main.py --company "Nvidia" --ticker NVDA --year 2026 --file analysis.txt --launch-ui
```

**Hydrate ChromaDB (run before analysis):**
```bash
python filing_embedder.py          # Embed SEC 10-Q/10-K filings from FMP
python news_loader.py --ticker AAPL --time-range 1m --company "Apple"
python market_data_loader.py       # Load Yahoo Finance 1-min intraday data
python parsing_agent.py            # Extract structured metrics into parser_data collection
```

**Launch report viewer:**
```bash
streamlit run streamlit_app.py
```

There is no test suite; validation is done by running the pipeline end-to-end.

## Architecture

### Agent Hierarchy

Three-tier orchestration in `reporting_pipeline.py`:

1. **Manager Agent** (GPT-5.1 via `deepagents`) — Summarizes the user prompt and routes to `create_report` tool
2. **Reporter Agent** (GPT-4o via LangChain `create_react_agent`) — Invokes 6 analysis tools and synthesizes the final narrative
3. **Analysis Tools** (`@tool`-decorated functions in `analyst.py`, `dcf.py`, `valuation_agent.py`) — Perform ChromaDB similarity search + LLM summarization

### ChromaDB RAG Collections

All analysis is grounded in four ChromaDB collections (no hallucination fallback):

| Collection | Loader | Contents |
|---|---|---|
| `company_filings` | `filing_embedder.py` | SEC 10-Q/10-K JSON chunks |
| `parser_data` | `parsing_agent.py` | Structured financial metric summaries |
| `financial_data` | `market_data_loader.py` | Yahoo Finance price/fundamentals |
| `news_articles` | `news_loader.py` | Perplexity Sonar news snippets |

### Tool Registry (Reporter Agent)

Registered in `reporting_pipeline.py` as `reporting_tools`:
1. `analyze_filings` — RAG over SEC filings
2. `analyze_parser` — Structured financial metrics
3. `analyze_financials` — Market/ticker data
4. `analyze_news` — News sentiment
5. `valuation_tool` — Qualitative + DCF blended valuation (`valuation_agent.py`)
6. `find_dcf_tool` — DCF from Yahoo Finance cash flows (`dcf.py`)

### DCF Model (`dcf.py`)

- Pulls 5-year FCF history from Yahoo Finance
- CAPM discount rate: 4% risk-free + beta × (9% − 4%)
- Terminal growth: FCF CAGR capped at [−10%, +15%]
- Extracts ticker from `parser_data` via regex if not supplied directly

### Key Files

- `reporting_pipeline.py` — Core orchestration; modify here to add tools or change agent behavior
- `analyst.py` — All ChromaDB-backed analysis tools; each uses `similarity_search(k=10)` + GPT-4o
- `filing_embedder.py` — Recursive JSON chunker with semantic splitting; smart chunk sizing (>15 items → 5 chunks, text >200 chars → `SemanticChunker`)
- `news_loader.py` — `SonarNewsClient` wraps Perplexity API; deduplicates via composite metadata key
- `dcf.py` + `valuation_agent.py` — Valuation pipeline; valuation_agent refines DCF output with qualitative context
- `pdf_builder.py` — Writes `report.txt`; optionally launches Streamlit
- `parser_queries.txt` — Query templates used by `parsing_agent.py` to extract metrics (format: `metric_name:source`)

## Extending the Platform

- **Add a new analysis tool**: Write a `@tool`-decorated function, import it in `reporting_pipeline.py`, and append to `reporting_tools`
- **Add a new data source**: Create a loader script that embeds into a new or existing ChromaDB collection, then expose it via a tool in `analyst.py`
- **Wrap as an API**: `generate_financial_report()` in `reporting_pipeline.py` is the single callable entry point
