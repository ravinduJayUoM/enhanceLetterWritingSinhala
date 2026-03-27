# Evaluation Phase 1 — Sinhala Letter Generation System

> First-phase evaluation covering individual pipeline steps and overall pipeline quality.

---

## Pipeline Step Evaluations

### 1. Information Extraction Accuracy

**Goal:** How accurately does Step 1 (InfoExtractor) populate the schema fields from a user prompt?

**Dataset:** A manually annotated gold standard set of (prompt → field values) pairs. Covers Sinhala, Singlish, and mixed-script prompts across all letter types.

**Fields evaluated:**

| Field type | Fields | Metric |
|---|---|---|
| Free-text | `recipient`, `sender`, `subject`, `purpose`, `details` | Token-overlap F1 (same approach as SQuAD answer evaluation) |
| Categorical | `letter_type` | Exact match accuracy |
| Optional / structured | `event_date`, `event_time`, `event_venue` | Exact match (invitation letters only) |

**Reported:** Per-field F1 and overall macro-average F1 across free-text fields.

---

### 2. Letter Type Classification

**Goal:** How accurately does the system classify the letter type (`letter_type`) across 5 classes: `application`, `request`, `complaint`, `invitation`, `general`.

**Dataset:** Same annotated set from Step 1 above.

**Metric:** Macro F1 (accounts for class imbalance across letter types).

---

### 3. Slot Filling / Gap Detection Accuracy

**Goal:** Does the GapFiller (Step 2) correctly identify which required fields are missing from an extraction result?

**Dataset:** A second annotated set of partially-filled extraction results (some fields intentionally blank), with ground-truth labels for which fields are "missing" for that letter type.

**Metric:** Precision, Recall, and F1 at the missing-field level (treated as a multi-label classification problem per letter type).

---

### 4. RAG Retrieval Relevance

**Goal:** Are the documents retrieved by Step 3 (Retriever) relevant to the user's request — both for structure templates and letter examples?

**Dataset:** A set of (query / extracted_info, relevant_doc_ids) annotations over the current knowledge base.

**Metrics:**

| Metric | What it measures |
|---|---|
| **Precision@K** | Of the top-K retrieved docs, what fraction are relevant? |
| **Category Hit Rate** | Does at least one retrieved doc match the correct letter category? |
| **Template Coverage** | Is at least one `doc_type=structure` document present in the results? |
| **NDCG@K** | Graded relevance — rewards systems that surface the best match earliest |

Evaluated separately for K=1 (structure bucket) and K=2 (examples bucket), matching the retriever's two-bucket strategy.

---

### 5. Hallucination Rate

**Goal:** Does the final generated letter introduce information that was not present in the original user prompt or the filled-in fields?

**Method:** Manual evaluation. Evaluators compare the generated letter against the original prompt and the extracted/gap-filled fields, and flag any content that cannot be traced back to the inputs.

**What counts as hallucination:** Fabricated names, dates, positions, organisations, amounts, or specific claims not grounded in the input.

**Reported:** Hallucination rate = proportion of generated letters containing at least one hallucinated fact, across the test set.

---

## Overall Pipeline Evaluation — Expert Rating

**Setup:** Native Sinhala experts evaluate pairs of letters generated from the same prompt:

- **Baseline:** Raw user prompt sent directly to the LLM with no extraction, no retrieval, and no prompt enhancement — a simple instruction like "write a formal Sinhala letter for this request."
- **Pipeline:** Full system output (extraction → gap fill → retrieval → enhanced prompt → generation).

Evaluators are **blind** to which letter is from the baseline and which is from the pipeline.

---

### Rating Criteria

Each letter is rated on the following dimensions using a **1–5 scale**:

| Dimension | 1 (Poor) | 3 (Acceptable) | 5 (Excellent) |
|---|---|---|---|
| **Linguistic Accuracy** | Many grammar / spelling errors | Minor errors; generally readable | Flawless formal Sinhala |
| **Formal Register** | Informal or mixed register throughout | Mostly formal with minor lapses | Fully formal throughout |
| **Content Relevance** | Letter does not address the request | Partially addresses the request | Fully and accurately addresses the request |
| **Structural Correctness** | Major sections missing or in wrong order | Most sections present; minor order issues | Complete, correctly ordered Sinhala letter structure |
| **Appropriate Honorifics / Politeness** | Wrong or missing honorifics for the recipient | Acceptable but not perfectly calibrated | Precisely correct honorifics and politeness level |
| **Overall Usability** | Cannot be sent without major rewriting | Could be sent with minor edits | Can be sent as-is |

**Primary signals:** Overall Usability and Linguistic Accuracy.

In addition to rating each letter individually, evaluators cast a **preference vote**: "Which letter would you send?" (Baseline / Pipeline / Tie). Win rate across all prompt pairs is the headline comparative result.

---

*Draft v1.0 — 2026-03-14*
