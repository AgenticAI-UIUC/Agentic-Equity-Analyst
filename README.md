# Agentic Equity Analyst

Agentic Equity Analyst is a LangChain-powered research assistant that stitches together retrieval, valuation, and reporting agents to deliver concise equity research writeups. It combines company filings, real-time market data, news sentiment, and discounted cash flow modeling into a single workflow that can be run from the command line or viewed inside Streamlit.

## Key Capabilities
- **One-command analysis** – run `python main.py --company "Your Co" --ticker TICK --year 2026` to generate `report.txt` and optionally launch the Streamlit viewer.
- **Retrieval-augmented reasoning** – pulls context from embedded SEC filings, parsed financials, macro indicators, and valuation memos stored in ChromaDB.
- **Multi-agent orchestration** – the manager agent coordinates reporting, filings, valuation, and DCF tools to answer complex prompts.
- **Report persistence & UI** – every run writes a markdown-style narrative to disk and can boot the built-in Streamlit app for sharing.

## Repository Layout
| Path | Purpose |
| --- | --- |
| `main.py` | CLI entry point for running the full analysis pipeline. |
| `reporting_pipeline.py` | Defines the manager/reporting agents and exposes `generate_financial_report`. |
| `pdf_builder.py` & `streamlit_app.py` | Write `report.txt` and render it inside the Streamlit UI. |
| `analyst.py`, `valuation_agent.py`, `dcf.py` | Tooling used by LangChain agents (filings, news, valuation, DCF). |
| `filing_embedder.py`, `market_data_loader.py`, `news_loader.py` | Utility scripts to hydrate Chroma collections with filings, market ticks, and Sonar news. |

## Installation
1. **Clone & enter the repo**
   ```bash
   git clone <your-fork-url>
   cd agentic-ai-equity-analyst
   ```
2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables
Create a `.env` file in the project root and populate the following variables:

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | ✅ | Secret key for GPT-4o and embeddings. Create at [platform.openai.com](https://platform.openai.com/). |
| `CHROMADB` | ✅ | Chroma database identifier (cloud DB name or local path). |
| `CHROMADB_API_KEY` | ✅ | API key for Chroma Cloud. |
| `CHROMADB_TENANT` | ✅ | Tenant/organization slug for Chroma Cloud. |
| `FMP_API_KEY` | ✅ for filings | Financial Modeling Prep key for downloading 10-Q/10-K JSON blobs. |
| `PPLX_API_KEY` | ✅ for news | Perplexity (Sonar) API key for recent news ingestion. |
| `FRED_API_KEY` | Optional | Federal Reserve Economic Data key for macro time series when extending market context. |
| `PPLX_MODEL` | Optional | Override Sonar model (defaults to `sonar-pro`). |

Add any other provider credentials you need (e.g., `YF_EMAIL` if you use premium Yahoo endpoints).

### Obtaining Keys
- **OpenAI** – generate a secret key under *User → API keys* at the [OpenAI dashboard](https://platform.openai.com/account/api-keys). Enable GPT-4o and `text-embedding-3-small`.
- **Chroma Cloud** – create a database+tenant at [docs.trychroma.com](https://docs.trychroma.com/). Copy the Database ID into `CHROMADB`, the tenant slug into `CHROMADB_TENANT`, and the service token into `CHROMADB_API_KEY`.
- **Financial Modeling Prep (FMP)** – sign up at [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs/pricing/), then copy the REST API key.
- **Perplexity Sonar** – request access at [perplexity.ai/api](https://www.perplexity.ai/api). Once approved, set the `PPLX_API_KEY` used by `news_loader.py`.
- **Federal Reserve Economic Data (FRED)** – create an account at [fred.stlouisfed.org](https://fredaccount.stlouisfed.org/apikey). Store the API token in `FRED_API_KEY` for any macro-ingestion utilities you add.

## Data Sources
- **FinancialModelingPrep (FMP)** – primary source for SEC filings and fundamentals (via `filing_embedder.py`).
- **Yahoo Finance** – intraday and end-of-day price/volume data via `market_data_loader.py` and DCF modeling (`dcf.py`).
- **Perplexity Sonar News** – curated, multi-source market-moving headlines ingested through `news_loader.py`.
- **Federal Reserve Economic Data (FRED)** – optional macro indicators (inflation, rates, GDP) that can be embedded into Chroma for richer prompts.
- **Internal analyst notes & valuation memos** – any documents you embed through `valuation_agent.py` collections.

## Usage
### 1. Run the analyst from the command line
1. Ensure `.env` is populated and dependencies are installed (`pip install -r requirements.txt`).
2. From the repo root, run a command such as:
   ```bash
   python main.py --company "Nvidia" --ticker NVDA --year 2026
   ```
3. Watch the terminal for:
   - The absolute path to the generated report file (defaults to `report.txt`).
   - A preview of the first 1,000 characters so you can sanity-check the response.
4. (Optional) Add `--launch-ui` to immediately open the Streamlit viewer that renders the same report.

Required inputs:
- `--company` – plain-language company name (e.g., "Nvidia").
- `--year` – forecast or fiscal year (e.g., 2026).

Optional flags:
- `--ticker NVDA` – specify a stock ticker for data retrieval (recommended when using market tools).
- `--prompt "Custom instructions"` – override the default template entirely.
- `--file custom_report.txt` – change the output path.
- `--launch-ui` – automatically open the Streamlit viewer after writing the file.

### 2. Launch the Streamlit UI manually
```bash
streamlit run streamlit_app.py
```
The UI simply renders the latest `report.txt`, so keep that file updated via the CLI or the pipeline.

### 3. Hydrate the vector stores (one-time or scheduled)
Run these scripts after setting your environment variables:
- `python filing_embedder.py --help` (edit script to target tickers/years) – embeds 10-Q/10-K JSON from FMP.
- `python market_data_loader.py` (called via LangChain tool) – pushes high-frequency Yahoo Finance ticks into the `financial_data` collection.
- `python news_loader.py --ticker AAPL --time-range 1m` – fetches Sonar news and writes to `news_data` collection.

## Extending the Pipeline
- Add new LangChain tools (e.g., FRED macro retrievers) and register them in `reporting_pipeline.py`.
- Expand the Streamlit experience by editing `streamlit_app.py` to include charts, tables, or uploads of supporting documents.
- Schedule ingestion jobs (cron, Airflow, etc.) for filings/news to keep the embeddings current.

Happy analyzing! 🎯
