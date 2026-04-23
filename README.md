# Agentic Data Assistant

A production-grade multi-agent RAG chatbot that answers natural-language questions over structured BigQuery data. Built on Google Vertex AI (Gemini) with a Parse → Plan → Execute → Synthesize pipeline.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│  GenericQuestionAgent│  ← classifies question type
└──────────┬──────────┘
           │ specific question
           ▼
┌─────────────────────┐
│ EntityResolutionAgent│  ← extracts entities from natural language
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Embedder         │  ← vector search over BigQuery schema
└──────────┬──────────┘
           │ schema context
           ▼
┌─────────────────────┐
│    PlanningAgent     │  ← decomposes question into agent steps
└──────────┬──────────┘
           │ plan
           ▼
     ┌─────┴──────┐
     ▼            ▼
┌──────────┐ ┌──────────┐
│QueryAgent│ │SearchAgent│  ← SQL generation + web grounding
└────┬─────┘ └────┬─────┘
     └──────┬─────┘
            ▼
  ┌──────────────────┐
  │  SynthesisAgent  │  ← combines evidence into final answer
  └──────────────────┘
```

## Features

- **7 specialized agents**: Generic classifier, entity resolver, planner, SQL query generator, web search, embedder, synthesizer
- **Vector-based schema matching**: Embeds the user question and cosine-matches it against your BigQuery schema to surface relevant tables/columns — no hardcoded table routing
- **Known-good query retrieval (KGQ)**: Stores vetted SQL examples as embeddings; retrieves the closest match to guide the SQL agent
- **Multi-step plan execution**: The planner decomposes complex questions into sequential steps; outputs from earlier steps are templated into later ones (`{{step_1_output.date}}`)
- **Web grounding toggle**: Optional Google Search augmentation via Vertex AI grounding
- **Conversation history**: Session-aware context window (last 5 exchanges) fed to all agents
- **Token tracking**: Per-agent and aggregate token usage logged on every request
- **Flask web UI**: Clean chat interface with session management and grounding toggle

## Tech Stack

- Python 3.10+
- [Google Vertex AI](https://cloud.google.com/vertex-ai) — Gemini models + embeddings
- [Google BigQuery](https://cloud.google.com/bigquery) — data store + vector search
- Flask — web interface
- `google-genai` SDK, `google-cloud-bigquery`

## Prerequisites

1. A Google Cloud project with Vertex AI and BigQuery APIs enabled
2. BigQuery dataset with your data tables
3. Embedding tables populated in BigQuery (see [Setup](#setup))
4. Application Default Credentials configured (`gcloud auth application-default login`)

## Setup

### 1. Clone and install

```bash
git clone https://github.com/dljx/agentic-data-assistant.git
cd agentic-data-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirement.txt
```

### 2. Configure

Copy `.env.example` to `.env` and edit `config.ini`:

```ini
[GCP]
project_id = your-gcp-project

[BIGQUERY]
bq_dataset_region = us-central1
bq_dataset_name = your_dataset
companies_table_name = companies_table
investors_table_name = investors_table
```

### 3. Prepare BigQuery embedding tables

The system requires three embedding tables in your dataset:

| Table | Purpose |
|-------|---------|
| `table_embeddings` | One row per table — table name + description + embedding vector |
| `header_embeddings` | One row per column — table, column, description, type, relevant columns + embedding |
| `kgq_embeddings` | Known-good SQL queries + natural language question + embedding |

Populate these using `text-embedding-005` from Vertex AI against your schema descriptions.

### 4. Run

```bash
python run_web_interface.py
```

Open [http://localhost:5000](http://localhost:5000).

## Project Structure

```
├── agents/
│   ├── core.py                  # Base Agent class (Vertex AI genai.Client)
│   ├── Embedder.py              # Text embedding via TextEmbeddingModel
│   ├── EntityResolutionAgent.py # Parses entities from natural language
│   ├── GenericQuestionAgent.py  # Classifies generic vs. domain-specific questions
│   ├── PlanningAgent.py         # Decomposes questions into multi-step plans
│   ├── QueryAgent.py            # Generates and executes BigQuery SQL
│   ├── SearchAgent.py           # Web search grounding
│   └── SynthesisAgent.py        # Combines evidence into final answer
├── dbconnectors/
│   ├── BQConnector.py           # BigQuery client + vector search queries
│   └── core.py                  # Abstract DB connector
├── tools/
│   └── SQLTool.py               # SQL parsing utilities (table/column extraction)
├── utilities/
│   └── __init__.py              # Config loading, logging, shared constants
├── templates/index.html          # Chat UI (Jinja2)
├── static/                       # CSS + JS
├── app.py                        # Pipeline orchestrator (ChatbotPipeline class)
├── webapp.py                     # Flask app + session management
├── run_web_interface.py          # Startup launcher
├── prompts.yaml                  # All agent system prompts
└── config.ini                    # Project/dataset configuration
```

## License

MIT
