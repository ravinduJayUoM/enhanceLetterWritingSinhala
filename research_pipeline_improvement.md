# research_pipeline_improvement.md
## Sinhala Formal Letter Support — Plan A (ML-Enhanced Retrieval) + Minimal Azure Deployment

This document is a **comprehensive execution plan** to pivot the current repo into a proper research + MVP system using **Plan A: ML-enhanced retrieval (retriever + reranker)**.  
It is written to be handed to an AI agent / engineer to implement end-to-end.

---

## 0) Context: What we already have (current repo state)

### Existing modules (already implemented)
- **RAG API (FastAPI)** in `sinhala_letter_rag.py`
  - Ingests `sinhala_letters.csv`
  - Builds vector DB (FAISS default; Chroma optional)
  - Retrieves examples via `LetterDatabase.search()`
  - Constructs an enhanced prompt including retrieved examples
  - Generates with `ChatOpenAI(model="gpt-4")`

- **NER / Information extraction**
  - `RAGProcessor.extract_key_info()` uses a Sinhala NER model (`models/sinhala_ner.py`)
  - `finetune_ner_model.py` exists to fine-tune XLM-R token classifier
  - Fallback extraction uses the LLM

- **Client script**
  - `client.py` calls `/process_query/`, `/generate_letter/`
  - Client also tries to call `/add_to_knowledge_base/` but this endpoint does **not** exist yet

### Problem we must solve
- Real Sinhala “official letters” datasets are scarce.
- Prompting with templates alone is not a strong enough ML contribution.
- We need a **measurable machine learning improvement** that directly improves Sinhala formal writing output quality.

---

## 1) Pivot decision: Plan A — ML-Enhanced Retrieval (best fit for this codebase)

### Goal
Improve the quality of retrieved examples/structures so the final generated letter:
- follows correct **formal structure**
- uses more appropriate **formal register**
- matches the **intent** (apology/request/invitation/complaint, etc.)
- reduces irrelevant examples that confuse the LLM

### Why Plan A is the right pivot
- Your pipeline already depends on retrieval quality (`retrieve_relevant_content()` → examples → prompt).
- Enhancing retrieval is **real ML** (training reranker and/or bi-encoder).
- You can evaluate:
  - IR metrics (Recall@k, MRR, nDCG)
  - downstream generation quality (human eval + structure scoring)

---

## 2) Core research contributions (what the thesis can claim)

### Contribution A — Domain-aware retrieval for Sinhala formal letter writing
- Baseline: LaBSE + FAISS similarity search
- Proposed: Two-stage retrieval:
  1) **Candidate retrieval** (bi-encoder embeddings)
  2) **Reranking** (cross-encoder / lightweight reranker)

Expected improvements:
- Higher relevance of examples by intent + structure
- Better output structure consistency and formality

### Contribution B (optional, already aligned with repo) — NER-driven query formulation
- Use extracted slots (recipient/sender/subject/purpose/letter_type) to build better retrieval queries
- Measure benefit vs raw user prompt retrieval

---

## 3) Target system architecture (updated)

### Current
User prompt → NER/LLM extraction → vector search → examples → LLM generate letter

### Updated (Plan A)
User prompt → NER/LLM extraction → **better query builder**
→ **Stage 1 retriever** (FAISS/Chroma)
→ **Stage 2 reranker** (cross-encoder)
→ top-k examples/structures → LLM generation (Azure OpenAI) → response

---

## 4) Implementation plan (step-by-step)

### 4.1 Data: Create a “structure-first” knowledge base (no need for large real-letter datasets)

#### What to do
1) Update or extend the dataset to include:
   - `letter_category` (apology/request/invitation/complaint/application/general)
   - `doc_type` (example | structure | section_template)
   - `register` (formal | very_formal)
   - `language` (si)
   - `source` (synthetic / curated / user_generated)
   - `title/subject` (short)
   - `content` (full text for examples OR reusable patterns for structure templates)
   - `tags` (string; keep simple)

2) Split dataset into two logical pools (can be one CSV, but separable by metadata):
   - **Structure templates**: headings, salutations, closings, standard blocks
   - **Full examples**: complete letters (even if synthetic/curated)

#### Why
- This makes retrieval more controllable and consistent.
- You can retrieve:
  - 1–2 structure templates + 1–2 examples
- Your system becomes useful even without sensitive real-world letters.

#### Deliverables
- `data/sinhala_letters_v2.csv` (new schema)
- `data/README_data_guidelines.md` (rules for creating templates + examples)

---

### 4.2 Query Builder: make retrieval Sinhala-aware (immediate win)

#### What to do
Replace the current:
```python
search_query = f"{letter_type} {subject} {purpose} {details}"
With:

Sinhala category mapping + weighted fields

Example:

letter_type_si = map_letter_type(letter_type)

search_query = f"{letter_type_si} මාතෘකාව {subject} අරමුණ {purpose} විස්තර {details} ලබන්නා {recipient} යවන්නා {sender}"

Add a small mapping table:

request → "ඉල්ලීමේ ලිපිය"

apology → "ක්ෂමා ලිපිය"

invitation → "ආරාධනා ලිපිය"

complaint → "පැමිණිලි ලිපිය"

application → "අයදුම්පත් ලිපිය"

Why
Your embeddings (LaBSE) will align better with Sinhala queries than English labels.

Improves retrieval baseline even before adding ML reranker.

Deliverables
rag/query_builder.py

Unit tests for mapping + query formatting

4.3 Stage 2 Reranker: add ML reranking (main research work)
What to build
Create rag/reranker.py with a class like:

CrossEncoderReranker(model_name=...)

method: rerank(query: str, docs: List[Document]) -> List[Document]

How to integrate
In RAGProcessor.retrieve_relevant_content():

retrieve candidates: top_k = 20

rerank: score each candidate vs query

return top_k = 3 (or configurable)

Why
Cross-encoder reranking is a well-known IR improvement and easy to evaluate.

Gives you a clean “ML contribution” without requiring huge datasets.

Deliverables
rag/reranker.py

updated retrieve_relevant_content() logic

config to enable/disable reranker (for baseline comparisons)

4.4 Training the reranker (weak supervision approach)
Dataset creation without real official letters
Create training pairs using weak labels:

Positive: query + doc from same letter_category

Negative: query + doc from different category

Hard negatives: near neighbors from the retriever but wrong category

Training data format (recommended)
Store as JSONL:

json
Copy code
{"query": "...", "doc": "...", "label": 1}
{"query": "...", "doc": "...", "label": 0}
Training pipeline tasks
scripts/build_reranker_dataset.py

load CSV

generate synthetic queries from metadata fields

build (query, doc, label) dataset

scripts/train_reranker.py

fine-tune a multilingual cross-encoder (or XLM-R cross-encoder)

scripts/eval_reranker.py

evaluate on held-out set (MRR/nDCG/Recall@k)

Why
This creates a measurable “before vs after” improvement story.

Works even with synthetic templates/examples.

Deliverables
training/reranker_train.jsonl

training/reranker_val.jsonl

trained model artifact saved under models/reranker/best_model

4.5 Evaluate retrieval + generation (research requirements)
Retrieval evaluation (must-have)
Create a small test set of user prompts (50–200) with known correct category.

Measure:

Recall@k (does a correct-category doc appear in top-k)

MRR / nDCG (ranking quality)

Compare:

Baseline A: LaBSE + similarity search

Baseline B: Sinhala-aware query builder + similarity search

Proposed: Sinhala-aware query builder + similarity search + reranker

Generation evaluation (must-have)
Use your existing evaluation approach but make it more structured:
Score each output on:

Structure correctness (salutation, subject, body, closing, signature)

Formal register appropriateness

Completeness (includes required fields)

Fluency/grammar (human rating)

Relevance (matches user intent)

Optional automatic checks:

Regex/section detectors for “structure presence”

Length constraints per letter type

Deliverables:

evaluation/prompts_testset.json

evaluation/results_retrieval.csv

evaluation/results_generation.csv

a short evaluation_report.md

5) Product/MVP changes (minimal but complete)
5.1 Add missing endpoint: /add_to_knowledge_base/ (optional but recommended)
Client currently calls it. Implement it properly.

What it should do:

Accept: content, original_prompt, metadata

Append to CSV (as doc_type="example", source="user_generated", include rating)

Update vector index:

simplest: rebuild index periodically (not per request)

minimal approach: write data and require manual /rebuild_knowledge_base/

Why

Lets you collect “high-rated” examples and improve future retrieval.

Helps thesis narrative: human-in-the-loop data growth.

Deliverables:

FastAPI endpoint implementation

safe file locking strategy (avoid CSV corruption)

5.2 Minimal UI (simple and deployable)
Build a tiny web UI with:

input prompt (Sinhala)

“process” → show missing questions (if any)

“generate” → shows generated letter

optionally “copy to clipboard”

optionally “rate output” (store rating; future work)

Recommended stack:

Next.js (simple)

or static HTML + fetch calls (even simpler)

Deliverables:

ui/ folder

.env config for API base URL

6) Deployment plan (Azure, minimal + clean)
Key requirement
Use Azure-native models instead of external API keys:

Use Azure OpenAI for generation (and optionally for extraction fallback).

Recommended minimal deployment approach
Azure Container Apps (best for minimal infra, quick deploy, scalable enough)

Components
Backend: FastAPI container

UI: Static web app (Azure Static Web Apps) OR container (if Next.js SSR)

Storage:

For MVP: store FAISS index + CSV in container filesystem (works but resets on redeploy)

Better: Azure Blob Storage mounted / used for persistence

Secrets:

Use Managed Identity + Key Vault OR Container Apps secrets

6.1 Backend deployment steps (FastAPI)
Add Dockerfile for backend

Set env vars:

AZURE_OPENAI_ENDPOINT

AZURE_OPENAI_DEPLOYMENT_NAME (your deployed model name)

AZURE_OPENAI_API_VERSION

(auth via key OR managed identity depending on your setup)

Replace ChatOpenAI usage with Azure OpenAI in LangChain:

Use AzureChatOpenAI (LangChain Azure integration) or configure ChatOpenAI with Azure settings (depending on library versions).

Ensure no hard-coded LLM_MODEL="gpt-4"; instead use Azure deployment name.

Persist vector index:

Option A (minimal): rebuild on startup from CSV, store FAISS in /data/faiss_index

Option B (better): load/save FAISS index to Azure Blob

Deliverables:

Dockerfile (backend)

azure/deploy_backend_containerapps.md

env var support in code

6.2 UI deployment steps
Option 1 (simplest):

Static HTML/JS UI

Deploy via Azure Static Web Apps

Option 2 (nice):

Next.js UI

Deploy via Azure Static Web Apps (supports Next.js)

Deliverables:

azure/deploy_ui_staticwebapps.md

6.3 Networking and CORS
Set backend CORS to only allow the UI domain in production (not "*").

Expose backend as HTTPS endpoint.

UI calls backend with API_BASE_URL.

Deliverables:

Production CORS config

ui/.env.production

7) Practical engineering notes (important)
7.1 Fix issues in current code
client.py assumes tags is a list, but server stores tags as a string.

Chunking: Sinhala letters may be harmed by splitting; prefer whole-letter retrieval for examples, and separate templates into smaller blocks.

7.2 Keep baseline comparability
Add a config flag to switch:

reranker ON/OFF

Sinhala query builder ON/OFF
So you can run A/B comparisons for thesis results.

7.3 Model selection guidance
Keep LaBSE for embeddings initially (already used).

Reranker:

start with a multilingual cross-encoder

then fine-tune with weak supervision dataset

8) Milestones (execution order)
Milestone 1 — Improve baseline retrieval (no training)
Add Sinhala query builder

Separate structure templates vs examples in dataset

Retrieve 2 templates + 2 examples

Ensure generation quality improves

Milestone 2 — Add reranker (inference only)
Plug reranker into pipeline

Benchmark retrieval metrics vs baseline

Milestone 3 — Train reranker (ML contribution)
Build weakly supervised dataset

Fine-tune reranker

Re-run evaluation and document improvements

Milestone 4 — Minimal UI + Azure deployment
Deploy backend to Azure Container Apps

Deploy UI to Azure Static Web Apps

Use Azure OpenAI deployment for generation

9) Acceptance criteria (definition of “done”)
Research acceptance
You can show quantitative improvement in retrieval (at least one of Recall@k / MRR / nDCG).

You can show downstream improvement in letter quality (human eval with clear rubric).

Clear baseline vs proposed comparison.

MVP acceptance
Working UI that:

takes Sinhala prompt

asks follow-up questions for missing fields

generates a formal Sinhala letter

Deployed on Azure using Azure OpenAI (no external OpenAI keys in production)

10) Checklist summary (what the AI agent must implement)
Code
 rag/query_builder.py + tests

 rag/reranker.py + integration in retrieve_relevant_content()

 dataset v2 schema + loader updates in create_documents()

 config flags for baseline/proposed comparisons

 /add_to_knowledge_base/ endpoint (optional but recommended)

 evaluation scripts + stored results

Deployment
 Dockerize backend

 Azure Container Apps deployment docs + env setup

 Azure OpenAI integration for generation model

 Minimal UI + Azure Static Web Apps deployment

 Production CORS tightened

11) Notes on using Azure OpenAI (implementation expectation)
Do not hardcode OpenAI API keys.

Use Azure OpenAI endpoint + deployment name.

Store secrets in Azure resources (Key Vault or Container App secrets).

Keep local dev option with .env for convenience.