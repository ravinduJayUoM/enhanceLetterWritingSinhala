# Sinhala Letter Generation Pipeline - Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   SINHALA LETTER GENERATION SYSTEM                          │
│                    Retrieval-Augmented Generation (RAG)                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                                                                                │
│  USER INPUT                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐            │
│  │ "විදුහල්පතිතුමා වෙත, මගේ දරුවාට පාසල් නිවාඩු ඉල්ලීමක්"       │            │
│  │ (Sinhala prompt describing letter needs)                     │            │
│  └──────────────────────────────────────────────────────────────┘            │
│                              │                                                │
│                              ▼                                                │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: INFORMATION EXTRACTION                                                │
│  ┌─────────────────────────────────────────────────────────────┐              │
│  │                   RAGProcessor.extract_key_info()           │              │
│  │                                                             │              │
│  │  ┌──────────────────┐        ┌──────────────────┐          │              │
│  │  │   NER Model      │   OR   │   LLM Extraction │          │              │
│  │  │   (Fine-tuned)   │        │   (Fallback)     │          │              │
│  │  │   xlm-roberta    │        │   aya:8b/GPT-4   │          │              │
│  │  └──────────────────┘        └──────────────────┘          │              │
│  │                                                             │              │
│  │  Extracts:                                                  │              │
│  │  • letter_type (request/application/complaint/general)     │              │
│  │  • recipient (who receives the letter)                     │              │
│  │  • sender (who writes the letter)                          │              │
│  │  • subject (main topic)                                    │              │
│  │  • purpose (why writing)                                   │              │
│  │  • details (supporting information)                        │              │
│  └─────────────────────────────────────────────────────────────┘              │
│                              │                                                │
│                              ▼                                                │
│  OUTPUT:                                                                      │
│  {                                                                            │
│    "letter_type": "request",                                                  │
│    "recipient": "විදුහල්පතිතුමා",                                            │
│    "sender": "දෙමාපිය",                                                      │
│    "subject": "නිවාඩු අවසරය",                                                 │
│    "purpose": "දරුවාට පාසල් නිවාඩුවක් ලබා ගැනීම"                            │
│  }                                                                            │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: MISSING INFORMATION DETECTION                                         │
│  ┌─────────────────────────────────────────────────────────────┐              │
│  │         RAGProcessor.identify_missing_info()                │              │
│  │                                                             │              │
│  │  Required Fields by Letter Type:                           │              │
│  │  • All: recipient, sender, subject, purpose                │              │
│  │  • Application: + qualifications, contact_details          │              │
│  │  • Complaint: + incident_date, requested_action            │              │
│  │  • Request: + requested_items, timeline                    │              │
│  │                                                             │              │
│  │  If missing info → Generate Sinhala questions              │              │
│  │  If complete → Proceed to retrieval                        │              │
│  └─────────────────────────────────────────────────────────────┘              │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: QUERY BUILDING                                                        │
│  ┌─────────────────────────────────────────────────────────────┐              │
│  │          SinhalaQueryBuilder.build_query()                  │              │
│  │                                                             │              │
│  │  Maps English → Sinhala:                                   │              │
│  │  • "request" → "ඉල්ලීමේ ලිපිය"                            │              │
│  │  • "application" → "අයදුම්පත් ලිපිය"                       │              │
│  │  • "complaint" → "පැමිණිලි ලිපිය"                          │              │
│  │                                                             │              │
│  │  Enriches with Sinhala keywords:                           │              │
│  │  • විධිමත් (formal)                                        │              │
│  │  • නිල (official)                                           │              │
│  │  • Field markers in Sinhala                                │              │
│  └─────────────────────────────────────────────────────────────┘              │
│                              │                                                │
│                              ▼                                                │
│  OUTPUT:                                                                      │
│  "ඉල්ලීමේ ලිපිය නිවාඩු අවසරය දරුවාට පාසල් නිවාඩුවක්                          │
│   විදුහල්පතිතුමා විධිමත්"                                                    │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: VECTOR SEARCH (RETRIEVAL)                                            │
│  ┌─────────────────────────────────────────────────────────────┐              │
│  │            LetterDatabase.search()                          │              │
│  │                                                             │              │
│  │  ┌───────────────────────────────────────────────┐         │              │
│  │  │        KNOWLEDGE BASE                         │         │              │
│  │  │  ┌─────────────────────────────────────┐      │         │              │
│  │  │  │  CSV: sinhala_letters_v2.csv        │      │         │              │
│  │  │  │  ├─ 12+ documents                   │      │         │              │
│  │  │  │  ├─ Structure templates             │      │         │              │
│  │  │  │  ├─ Full letter examples             │      │         │              │
│  │  │  │  └─ Section templates                │      │         │              │
│  │  │  └─────────────────────────────────────┘      │         │              │
│  │  │                   │                           │         │              │
│  │  │                   ▼                           │         │              │
│  │  │  ┌─────────────────────────────────────┐      │         │              │
│  │  │  │   LaBSE Embeddings                  │      │         │              │
│  │  │  │   (768 dimensions)                  │      │         │              │
│  │  │  │   sentence-transformers/LaBSE       │      │         │              │
│  │  │  └─────────────────────────────────────┘      │         │              │
│  │  │                   │                           │         │              │
│  │  │                   ▼                           │         │              │
│  │  │  ┌─────────────────────────────────────┐      │         │              │
│  │  │  │   FAISS Vector Index                │      │         │              │
│  │  │  │   Cosine similarity search          │      │         │              │
│  │  │  └─────────────────────────────────────┘      │         │              │
│  │  └───────────────────────────────────────────────┘         │              │
│  │                                                             │              │
│  │  Returns: Top-K documents (K=20 for reranking, K=3 else)  │              │
│  └─────────────────────────────────────────────────────────────┘              │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: RERANKING (PHASE 2 ENHANCEMENT)                                      │
│  ┌─────────────────────────────────────────────────────────────┐              │
│  │         CrossEncoderReranker.rerank()                       │              │
│  │                                                             │              │
│  │  Model: cross-encoder/ms-marco-MiniLM-L-6-v2              │              │
│  │                                                             │              │
│  │  ┌─────────────────────────────────────┐                   │              │
│  │  │  Initial 20 documents               │                   │              │
│  │  │          ↓                          │                   │              │
│  │  │  Cross-encoder scoring              │                   │              │
│  │  │  (query-document pairs)             │                   │              │
│  │  │          ↓                          │                   │              │
│  │  │  Sort by relevance score            │                   │              │
│  │  │          ↓                          │                   │              │
│  │  │  Return Top 3 documents             │                   │              │
│  │  └─────────────────────────────────────┘                   │              │
│  │                                                             │              │
│  │  Status: ✅ ENABLED (use_reranker=True)                    │              │
│  └─────────────────────────────────────────────────────────────┘              │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: ENHANCED PROMPT CONSTRUCTION                                         │
│  ┌─────────────────────────────────────────────────────────────┐              │
│  │      RAGProcessor.construct_enhanced_prompt()               │              │
│  │                                                             │              │
│  │  Combines:                                                  │              │
│  │  1. Original user prompt                                   │              │
│  │  2. Extracted structured information                       │              │
│  │  3. Retrieved letter examples/templates (Top 3)            │              │
│  │  4. Formatting instructions                                │              │
│  │                                                             │              │
│  │  Prompt Template:                                           │              │
│  │  ┌───────────────────────────────────────────┐             │              │
│  │  │ You are a Sinhala letter assistant       │             │              │
│  │  │ Write ONLY in Sinhala                    │             │              │
│  │  │                                           │             │              │
│  │  │ Original Request: [user prompt]          │             │              │
│  │  │                                           │             │              │
│  │  │ Letter Details:                           │             │              │
│  │  │ - Type: [extracted type]                 │             │              │
│  │  │ - Recipient: [extracted recipient]       │             │              │
│  │  │ - Sender: [extracted sender]             │             │              │
│  │  │ - Subject: [extracted subject]           │             │              │
│  │  │ - Purpose: [extracted purpose]           │             │              │
│  │  │                                           │             │              │
│  │  │ Example Letters:                          │             │              │
│  │  │ [Top 3 retrieved documents]              │             │              │
│  │  │                                           │             │              │
│  │  │ Generate formal letter in Sinhala        │             │              │
│  │  └───────────────────────────────────────────┘             │              │
│  └─────────────────────────────────────────────────────────────┘              │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: LETTER GENERATION                                                    │
│  ┌─────────────────────────────────────────────────────────────┐              │
│  │                LLM Generation                               │              │
│  │                                                             │              │
│  │  Supported LLM Options:                                     │              │
│  │  ┌──────────────────┐  ┌──────────────────┐               │              │
│  │  │ Ollama (Local)   │  │ Azure OpenAI     │               │              │
│  │  │ • aya:8b         │  │ • gpt-4          │               │              │
│  │  │ • llama3.2:3b    │  │ • gpt-35-turbo   │               │              │
│  │  │ FREE!            │  │ PAID             │               │              │
│  │  └──────────────────┘  └──────────────────┘               │              │
│  │  ┌──────────────────┐  ┌──────────────────┐               │              │
│  │  │ OpenAI           │  │ HuggingFace      │               │              │
│  │  │ • gpt-4          │  │ Local models     │               │              │
│  │  │ PAID             │  │ FREE!            │               │              │
│  │  └──────────────────┘  └──────────────────┘               │              │
│  │                                                             │              │
│  │  Current: aya:8b (Sinhala-optimized, 101 languages)       │              │
│  └─────────────────────────────────────────────────────────────┘              │
│                              │                                                │
│                              ▼                                                │
│  OUTPUT:                                                                      │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │  2026 ජනවාරි 20                                          │                │
│  │                                                           │                │
│  │  විදුහල්පති මහතා,                                        │                │
│  │  [පාසල් නම],                                             │                │
│  │  [ලිපිනය]                                                │                │
│  │                                                           │                │
│  │  ගරු මහත්මයාණෙනි,                                        │                │
│  │                                                           │                │
│  │  මාතෘකාව: නිවාඩු අවසරය                                   │                │
│  │                                                           │                │
│  │  මගේ දරුවා වන [නම] ඔබතුමාගේ විදුහලේ [පන්තිය]           │                │
│  │  ඉගෙනුම ලබන සිසුවෙකි. [හේතුව] නිසා [දිනය] සිට          │                │
│  │  [දිනය] දක්වා විදුහලට පැමිණීමට නොහැකි වේ.               │                │
│  │                                                           │                │
│  │  එබැවින් ඉහත කාලය සඳහා නිවාඩු අවසරයක් ලබා දෙන           │                │
│  │  ලෙස කාරුණිකව ඉල්ලා සිටිමි.                              │                │
│  │                                                           │                │
│  │  ඔබතුමාගේ කාරුණික සැලකිල්ල අපේක්ෂා කරමි.                │                │
│  │                                                           │                │
│  │  ගෞරවයෙන්,                                                │                │
│  │                                                           │                │
│  │  [දෙමාපිය නම]                                            │                │
│  │  දිනය: 2026-01-20                                        │                │
│  └──────────────────────────────────────────────────────────┘                │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 8: USER FEEDBACK & KNOWLEDGE BASE UPDATE (OPTIONAL)                     │
│  ┌─────────────────────────────────────────────────────────────┐              │
│  │        User rates the generated letter (1-5 stars)          │              │
│  │                          │                                   │              │
│  │                          ▼                                   │              │
│  │            If rating ≥ 4 → Add to Knowledge Base            │              │
│  │                          │                                   │              │
│  │                          ▼                                   │              │
│  │          LetterDatabase.add_to_knowledge_base()             │              │
│  │                          │                                   │              │
│  │                          ▼                                   │              │
│  │        Future retrievals include this letter                │              │
│  │        (Continuous improvement loop)                         │              │
│  └─────────────────────────────────────────────────────────────┘              │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. **Core Components**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CORE MODULES                                  │
│                                                                      │
│  ┌────────────────────┐  ┌────────────────────┐                    │
│  │ LetterDatabase     │  │ RAGProcessor       │                    │
│  │                    │  │                    │                    │
│  │ • build_KB()       │  │ • extract_info()   │                    │
│  │ • search()         │  │ • retrieve()       │                    │
│  │ • add_document()   │  │ • generate()       │                    │
│  └────────────────────┘  └────────────────────┘                    │
│                                                                      │
│  ┌────────────────────┐  ┌────────────────────┐                    │
│  │ SinhalaQueryBuilder│  │ CrossEncoderReranker│                   │
│  │                    │  │                    │                    │
│  │ • build_query()    │  │ • rerank()         │                    │
│  │ • map_to_sinhala() │  │ • score_pairs()    │                    │
│  └────────────────────┘  └────────────────────┘                    │
│                                                                      │
│  ┌────────────────────┐  ┌────────────────────┐                    │
│  │ SinhalaLetterNER   │  │ Config             │                    │
│  │                    │  │                    │                    │
│  │ • extract_info()   │  │ • get_config()     │                    │
│  │ • fine_tune()      │  │ • LLMProvider      │                    │
│  └────────────────────┘  └────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. **API Endpoints**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI REST API                                  │
│                  (http://localhost:8000)                             │
│                                                                      │
│  POST /process_query/                                               │
│  ├─ Input: {"prompt": "Sinhala text"}                              │
│  └─ Output: {status, extracted_info, questions, enhanced_prompt}   │
│                                                                      │
│  POST /generate_letter/                                             │
│  ├─ Input: {"enhanced_prompt": "..."}                              │
│  └─ Output: {"generated_letter": "..."}                            │
│                                                                      │
│  GET /search/                                                       │
│  ├─ Input: ?query=...&top_k=3                                      │
│  └─ Output: {"results": [...]}                                     │
│                                                                      │
│  POST /rebuild_knowledge_base/                                      │
│  ├─ Input: ?force=true                                             │
│  └─ Rebuilds FAISS index from CSV                                  │
│                                                                      │
│  GET /diagnostics/                                                  │
│  └─ Returns KB status, document count, model info                  │
│                                                                      │
│  POST /add_to_knowledge_base/                                       │
│  ├─ Input: {content, title, metadata}                              │
│  └─ Adds new letter to CSV and updates index                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 3. **Data Flow**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                                   │
│                                                                      │
│  CSV File (sinhala_letters_v2.csv)                                  │
│       │                                                              │
│       ├─ Columns:                                                   │
│       │  • id (unique identifier)                                   │
│       │  • letter_category (request/application/complaint/etc)      │
│       │  • doc_type (example/structure/section_template)            │
│       │  • register (formal/very_formal)                            │
│       │  • language (si = Sinhala)                                  │
│       │  • source (synthetic/curated/user_generated)                │
│       │  • title (short Sinhala title)                              │
│       │  • content (full letter text)                               │
│       │  • tags (comma-separated)                                   │
│       │  • rating (1-5 for quality)                                 │
│       │                                                              │
│       ▼                                                              │
│  Document Objects (LangChain)                                       │
│       │                                                              │
│       ├─ page_content: letter text                                  │
│       ├─ metadata: {id, category, type, register, ...}              │
│       │                                                              │
│       ▼                                                              │
│  LaBSE Embeddings (768-dim vectors)                                 │
│       │                                                              │
│       ▼                                                              │
│  FAISS Index (faiss_index/index.faiss)                              │
│       │                                                              │
│       └─ Fast similarity search                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4. **NER Training Pipeline**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  NER MODEL TRAINING PIPELINE                         │
│                                                                      │
│  1. Prepare Dataset                                                 │
│     rag/models/prepare_ner_dataset.py                               │
│     │                                                                │
│     ├─ Reads: training_data/raw/train_samples.json                  │
│     │         (Human-annotated Sinhala text with entities)          │
│     │                                                                │
│     └─ Outputs: training_data/transformer/                          │
│                 ├─ train.json (tokenized, BIO tags)                 │
│                 └─ val.json (validation split)                      │
│                                                                      │
│  2. Fine-tune Model                                                 │
│     rag/finetune_ner_model.py                                       │
│     │                                                                │
│     ├─ Base Model: xlm-roberta-base (multilingual)                  │
│     │                                                                │
│     ├─ Training:                                                    │
│     │  • Epochs: 3-5                                                │
│     │  • Batch size: 8-16                                           │
│     │  • Learning rate: 2e-5                                        │
│     │                                                                │
│     └─ Output: training_data/best_model/                            │
│                (Fine-tuned transformer for Sinhala NER)             │
│                                                                      │
│  3. Test Model                                                      │
│     rag/test_ner_model.py                                           │
│     │                                                                │
│     └─ Evaluates extraction on test Sinhala prompts                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TECHNOLOGY STACK                                │
│                                                                      │
│  Backend Framework                                                  │
│  ├─ FastAPI (REST API)                                             │
│  └─ Uvicorn (ASGI server)                                          │
│                                                                      │
│  ML/NLP Framework                                                   │
│  ├─ LangChain (RAG pipeline orchestration)                         │
│  ├─ HuggingFace Transformers (NER model)                           │
│  ├─ Sentence Transformers (embeddings, reranking)                  │
│  └─ PyTorch (deep learning backend)                                │
│                                                                      │
│  Vector Database                                                    │
│  ├─ FAISS (Facebook AI Similarity Search)                          │
│  └─ In-memory index with disk persistence                          │
│                                                                      │
│  Embedding Models                                                   │
│  ├─ LaBSE (Language-agnostic BERT Sentence Embeddings)             │
│  │   • 768 dimensions                                              │
│  │   • Supports 109+ languages including Sinhala                   │
│  └─ Cross-encoder/ms-marco-MiniLM-L-6-v2 (reranking)              │
│                                                                      │
│  LLM Options                                                        │
│  ├─ Ollama (Local, Free)                                           │
│  │   • aya:8b (recommended - Sinhala-optimized)                    │
│  │   • llama3.2:3b (lightweight alternative)                       │
│  ├─ Azure OpenAI (Cloud, Paid)                                     │
│  │   • gpt-4                                                        │
│  │   • gpt-35-turbo                                                │
│  ├─ OpenAI (Cloud, Paid)                                           │
│  │   • gpt-4                                                        │
│  └─ HuggingFace (Local, Free)                                      │
│      • Any compatible model                                         │
│                                                                      │
│  NER Model                                                          │
│  ├─ Base: xlm-roberta-base (multilingual)                          │
│  ├─ Fine-tuned on Sinhala letter extraction task                   │
│  └─ Entity types: letter_type, recipient, sender, subject, etc.    │
│                                                                      │
│  Data Storage                                                       │
│  ├─ CSV (pandas) - Letter knowledge base                           │
│  └─ JSON - NER training data                                       │
│                                                                      │
│  Deployment                                                         │
│  ├─ Oracle Cloud (Free Tier)                                       │
│  ├─ Azure (Cloud deployment option)                                │
│  └─ Local development (localhost:8000)                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Features & Innovations

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KEY INNOVATIONS                                   │
│                                                                      │
│  1. SINHALA-AWARE QUERY BUILDING                                    │
│     ✓ Maps English letter types to Sinhala equivalents             │
│     ✓ Enriches queries with formal Sinhala markers                 │
│     ✓ Improves embedding alignment for better retrieval            │
│                                                                      │
│  2. CROSS-ENCODER RERANKING (Phase 2)                              │
│     ✓ Initial retrieval: 20 candidates                             │
│     ✓ Cross-encoder scoring for relevance                          │
│     ✓ Final output: Top 3 most relevant                            │
│                                                                      │
│  3. HYBRID NER EXTRACTION                                           │
│     ✓ Primary: Fine-tuned transformer (xlm-roberta)                │
│     ✓ Fallback: LLM-based extraction                               │
│     ✓ Rule-based patterns for robustness                           │
│                                                                      │
│  4. MULTI-PROVIDER LLM SUPPORT                                      │
│     ✓ Ollama (free, local, Sinhala-optimized)                      │
│     ✓ Azure OpenAI (cloud, enterprise)                             │
│     ✓ Standard OpenAI (cloud)                                       │
│     ✓ HuggingFace (local, flexible)                                │
│                                                                      │
│  5. CONTINUOUS IMPROVEMENT LOOP                                     │
│     ✓ User ratings (1-5 stars)                                     │
│     ✓ Highly-rated letters added to KB                             │
│     ✓ Improves future generations                                  │
│                                                                      │
│  6. STRUCTURED DATA SCHEMA V2                                       │
│     ✓ Rich metadata (category, type, register, source)             │
│     ✓ Multiple document types (examples, templates, sections)      │
│     ✓ Quality ratings and tags                                     │
│                                                                      │
│  7. MISSING INFORMATION HANDLING                                    │
│     ✓ Identifies required fields by letter type                    │
│     ✓ Generates clarifying questions in Sinhala                    │
│     ✓ Interactive completion of information gaps                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Current Status & Issues

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PIPELINE STATUS                                  │
│                                                                      │
│  ✅ WORKING COMPONENTS                                               │
│  ├─ User input processing (Sinhala prompts)                         │
│  ├─ Vector search (FAISS + LaBSE embeddings)                        │
│  ├─ Cross-encoder reranking (Phase 2 complete)                      │
│  ├─ Sinhala-aware query building                                    │
│  ├─ Enhanced prompt construction                                    │
│  ├─ Basic letter generation (Sinhala output)                        │
│  ├─ API endpoints (FastAPI server)                                  │
│  └─ Knowledge base management                                       │
│                                                                      │
│  ⚠️  ISSUES IDENTIFIED                                               │
│  ├─ Information extraction quality (LLM returns placeholders)       │
│  ├─ Small knowledge base (12 docs, need 50-100)                     │
│  ├─ LLM model size (llama3.2:3b too small for complex tasks)        │
│  └─ NER model not yet fully trained                                 │
│                                                                      │
│  🎯 PRIORITIES FOR IMPROVEMENT                                       │
│  1. Switch to aya:8b (Sinhala-optimized, 2.7x larger)               │
│  2. Fix extraction prompt (English instructions, JSON schema)       │
│  3. Expand knowledge base (20-30 more letter examples)              │
│  4. Complete NER model training                                     │
│                                                                      │
│  📊 CURRENT PERFORMANCE                                              │
│  ├─ Baseline (no RAG): 0% (complete failure)                        │
│  ├─ Enhanced (with RAG): 60% (generic letters)                      │
│  └─ Target: 85-90% (personalized, high-quality)                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT OPTIONS                                 │
│                                                                      │
│  LOCAL DEVELOPMENT                                                  │
│  ┌───────────────────────────────────────────┐                     │
│  │  localhost:8000                           │                     │
│  │  ├─ FastAPI server (uvicorn)              │                     │
│  │  ├─ Ollama (local LLM server)             │                     │
│  │  └─ FAISS index (in-memory)               │                     │
│  └───────────────────────────────────────────┘                     │
│                                                                      │
│  ORACLE CLOUD (FREE TIER)                                           │
│  ┌───────────────────────────────────────────┐                     │
│  │  VM.Standard.A1.Flex (ARM)                │                     │
│  │  ├─ 4 OCPUs, 24GB RAM                     │                     │
│  │  ├─ Ubuntu 22.04                          │                     │
│  │  ├─ Python 3.10+                          │                     │
│  │  ├─ Systemd service (auto-start)          │                     │
│  │  └─ Public IP with firewall (port 8000)   │                     │
│  └───────────────────────────────────────────┘                     │
│                                                                      │
│  AZURE CLOUD                                                        │
│  ┌───────────────────────────────────────────┐                     │
│  │  Azure VM / App Service                   │                     │
│  │  ├─ Azure OpenAI integration              │                     │
│  │  ├─ Managed identity                      │                     │
│  │  └─ Load balancing                        │                     │
│  └───────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
enhanceLetterWritingSinhala/
├─ rag/
│  ├─ sinhala_letter_rag.py      # Main RAG pipeline implementation
│  ├─ config.py                   # Configuration & feature flags
│  ├─ query_builder.py            # Sinhala-aware query construction
│  ├─ reranker.py                 # Cross-encoder reranking (Phase 2)
│  ├─ client.py                   # API client & testing utilities
│  ├─ finetune_ner_model.py       # NER model fine-tuning script
│  ├─ test_ner_model.py           # NER model evaluation
│  ├─ faiss_index/                # Vector index storage
│  │  └─ index.faiss
│  └─ models/
│     ├─ sinhala_ner.py           # Custom NER model
│     ├─ prepare_ner_dataset.py   # Training data preparation
│     └─ training_data/
│        ├─ raw/                  # Human-annotated samples
│        ├─ transformer/          # Tokenized training data
│        └─ best_model/           # Fine-tuned model weights
│
├─ data/
│  ├─ sinhala_letters_v2.csv      # Knowledge base (v2 schema)
│  └─ README_data_guidelines.md   # Data schema documentation
│
├─ tests/
│  ├─ test_api.py                 # API endpoint tests
│  ├─ test_phase1_integration.py  # Integration tests
│  └─ test_query_builder.py       # Query builder tests
│
├─ ui/                             # React frontend (future)
│  ├─ src/
│  │  ├─ App.js
│  │  └─ Chat.js
│  └─ package.json
│
├─ run_server.py                   # API server launcher
├─ evaluate_pipeline.py            # Pipeline evaluation script
├─ README.md                       # Main documentation
├─ PIPELINE_BREAKDOWN.md           # Detailed pipeline analysis
├─ DEPLOYMENT_GUIDELINE.md         # Oracle Cloud deployment guide
├─ NER_ANNOTATION_GUIDE.md         # NER annotation instructions
└─ ORACLE_CLOUD_DEPLOYMENT.md      # Cloud deployment guide
```

---

## Summary

This is a **Retrieval-Augmented Generation (RAG) system** for generating formal Sinhala letters. The pipeline:

1. **Extracts** structured information from Sinhala user prompts (NER/LLM)
2. **Builds** Sinhala-aware search queries
3. **Retrieves** relevant letter examples from a vector database (FAISS + LaBSE)
4. **Reranks** results using cross-encoder for better relevance
5. **Constructs** enhanced prompts with examples and extracted details
6. **Generates** formal Sinhala letters using LLM (aya:8b/GPT-4)
7. **Improves** continuously by adding highly-rated letters to the knowledge base

The system supports **multiple LLM providers** (Ollama, Azure OpenAI, OpenAI, HuggingFace), uses **Sinhala-optimized models** (aya:8b), and includes **Phase 2 enhancements** (cross-encoder reranking) for improved retrieval quality.
