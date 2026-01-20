# Sinhala Letter RAG Pipeline - Step-by-Step Breakdown

## Overview
The pipeline transforms a Sinhala user request into a formal letter using RAG (Retrieval-Augmented Generation).

---

## Pipeline Steps

### **STEP 1: User Input**
**What happens:** User provides a Sinhala prompt describing their letter needs

**Example Input:**
```
"මම අසනිප් නිසා අද රැකියාවට පැමිණිය නොහැක. කරුණාකර අද සඳහා නිවාඩු අවසරයක් ලබා දෙන්න."
```

**Code Location:** `POST /process_query/` endpoint → `UserQuery` model

**Status:** ✅ Working

---

### **STEP 2: Information Extraction**
**What happens:** Extract structured information from the Sinhala prompt

**Method:** LLM-based extraction (NER model exists but not trained)

**Code Location:** `RAGProcessor.extract_key_info()` → `_extract_with_llm()`

**Current Behavior:**
```python
# LLM receives Sinhala prompt asking for JSON:
extraction_prompt = """
මෙම ලිපි ඉල්ලීමෙන් ප්‍රධාන තොරතුරු උපුටා ගන්න. JSON ආකෘතියකින් පිළිතුරු දෙන්න:
...
"""
```

**Expected Output:**
```json
{
  "letter_type": "request",
  "recipient": "කළමනාකරු",
  "sender": "සුනිල් පෙරේරා",
  "subject": "නිවාඩු අවසරය",
  "purpose": "අසනීප නිවාඩුවක් ලබා ගැනීම"
}
```

**Actual Output (BROKEN):**
```json
{
  "letter_type": "ලිපි වර්ගය (application/request/complaint/etc)",
  "recipient": "ලිපිය ලබන්නා",
  "sender": "ලිපිය යවන්නා"
}
```

**Problem:** ❌ LLM returns Sinhala field descriptions instead of actual extracted values
- llama3.2:3b is too small to follow Sinhala extraction instructions
- Returns template/placeholder text instead of parsed data

**Impact:** 
- Missing info detection doesn't work properly
- Query building gets garbage data
- Enhanced prompt has empty fields

---

### **STEP 3: Missing Information Detection**
**What happens:** Identify what information is missing for the letter type

**Code Location:** `RAGProcessor.identify_missing_info()`

**Logic:**
```python
# Required fields by letter type:
- All letters: recipient, sender, subject, purpose
- Application: + qualifications, contact_details
- Complaint: + incident_date, requested_action
- Request: + requested_items, timeline
```

**Current Status:** ⚠️ Partially working
- Logic is correct
- But depends on broken extraction (Step 2)
- If extraction fails, all fields appear "missing"

---

### **STEP 4: Query Building**
**What happens:** Transform extracted info into an effective search query

**Code Location:** `RAGProcessor.retrieve_relevant_content()`

**Two Modes:**

#### **A) Sinhala-Aware Query Builder (Enhanced)** ✅ ENABLED
**Code:** `SinhalaQueryBuilder.build_query()`

**Logic:**
```python
# Maps letter types to Sinhala keywords
LETTER_TYPE_MAPPING = {
    "application": "අයදුම්පත",
    "request": "ඉල්ලීම",
    "complaint": "පැමිණිල්ල"
}

# Constructs query:
query = f"{sinhala_type} {subject} {purpose} {details} {recipient} {sender} විධිමත්"
```

**Example Output:**
```
"ඉල්ලීම නිවාඩු අවසරය අසනීප නිවාඩුවක් කළමනාකරු සුනිල් විධිමත්"
```

**Status:** ✅ Working correctly when extraction works

#### **B) Legacy Query (Baseline)** - DISABLED
**Logic:** Simple concatenation of fields
```python
query = f"{letter_type} {subject} {purpose} {details}"
```

**Problem with Broken Extraction:**
When extraction returns placeholder text, query becomes:
```
"ලිපි වර්ගය (application/request/complaint/etc) ලිපියේ මාතෘකාව අරමුණ විධිමත්"
```
This is why retrieval still works (generic Sinhala terms match structure templates)!

---

### **STEP 5: Vector Search (Retrieval)**
**What happens:** Find relevant letter examples/templates from knowledge base

**Code Location:** `LetterDatabase.search()`

**Method:** 
1. Embed query using LaBSE (768 dimensions)
2. Search FAISS index with cosine similarity
3. Return top-K documents (K=20 for reranking, K=3 otherwise)

**Current Behavior:**
```python
# Query: "ලිපි වර්ගය මාතෘකාව විධිමත්" (from broken extraction)
# Results: 12 documents returned (all documents in DB)
```

**Why it still works:**
- Generic Sinhala terms match all document types
- Knowledge base is small (12 docs)
- All documents are templates/structures

**Knowledge Base:**
- **Size:** 12 documents
- **Types:** 
  - 3 structure templates (application, complaint, request)
  - 6 full examples
  - 3 section templates
- **Format:** CSV with v2 schema (letter_category, doc_type, register, etc.)

**Status:** ✅ Working but limited by small dataset

---

### **STEP 6: Reranking (Optional Enhancement)**
**What happens:** Reorder retrieved documents by relevance using cross-encoder

**Code Location:** `CrossEncoderReranker.rerank()`

**Method:**
1. Take 20 initially retrieved documents
2. Score each with cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
3. Sort by score (higher = more relevant)
4. Return top 3

**Config:** ✅ ENABLED (`use_reranker=True`)

**Current Behavior:**
```
Initial 12 docs → Cross-encoder scoring → Top 3:
  [1] application - structure
  [2] complaint - structure  
  [3] request - structure
```

**Status:** ✅ Working (Phase 2 complete)

---

### **STEP 7: Enhanced Prompt Construction**
**What happens:** Combine extracted info + retrieved examples into a rich prompt

**Code Location:** `RAGProcessor.construct_enhanced_prompt()`

**Prompt Structure:**
```
You are a Sinhala formal letter writing assistant. Generate IN SINHALA.

IMPORTANT: Write ONLY in Sinhala script. No English.

Original Request: [user's Sinhala prompt]

Letter Details:
- Type: [extracted_letter_type]
- Recipient: [extracted_recipient]
- Sender: [extracted_sender]
- Subject: [extracted_subject]
- Purpose: [extracted_purpose]
- Additional Details: [extracted_details]

Example Letter Formats (use as templates):
[retrieved_doc_1_full_text]
---
[retrieved_doc_2_full_text]
---
[retrieved_doc_3_full_text]

Instructions:
1. Write complete formal letter in Sinhala following examples
2. Use proper grammar and formal register
3. Include greetings and closings
4. Address all details
5. Output ONLY the letter in Sinhala

Generate the letter now:
```

**Problem with Broken Extraction:**
```
Letter Details:
- Type: ලිපි වර්ගය (application/request/complaint/etc)  ❌
- Recipient: ලිපිය ලබන්නා  ❌
- Sender: ලිපිය යවන්නා  ❌
```
LLM sees field descriptions instead of actual data!

**Status:** ⚠️ Partially working
- Prompt structure is good
- Retrieved examples provide context
- But extracted details are useless

---

### **STEP 8: Letter Generation**
**What happens:** LLM generates the final Sinhala letter

**Code Location:** `POST /generate_letter/` endpoint

**Model:** Ollama llama3.2:3b (2GB)
- **Size:** Very small (3 billion parameters)
- **Sinhala Training:** Limited (general multilingual, not Sinhala-focused)

**Current Behavior:**

**Without RAG (Baseline):**
```
Input: "මම අසනිප් නිසා නිවාඩුවක් අවශ්‍යයි"
Output: "I can't help with that."
```
❌ Model refuses to generate Sinhala letter without context

**With RAG (Enhanced):**
```
Input: [Enhanced prompt with examples]
Output: [Generic template with placeholders]

අයදුම්පත් ලිපි ආකෘතිය

[දිනය]
[ලබන්නාගේ නම]
...
ගරු මහත්මයාණෙනි,
මාතෘකාව: [අයදුම්පත් ලිපි]
...
```

**Why it generates templates:**
1. Extracted details are broken (placeholders only)
2. Model sees examples but no real user data
3. Falls back to mimicking the template structure
4. Too small to understand complex instructions

**Status:** ⚠️ Working but low quality
- Generates valid Sinhala (60% quality score)
- Better than baseline (0% - complete refusal)
- But generic, not personalized

---

## Summary: What Works & What's Broken

### ✅ **Working Components**
1. **User Input** - Accepts Sinhala prompts
2. **Vector Search** - Retrieves relevant documents (FAISS + LaBSE)
3. **Reranker** - Cross-encoder reordering works
4. **Prompt Construction** - Template is well-structured
5. **Basic Generation** - Produces Sinhala letters (generic)

### ❌ **Broken/Weak Components**
1. **Information Extraction** - CRITICAL: Returns garbage data
2. **LLM Model** - llama3.2:3b too small for Sinhala tasks
3. **Knowledge Base** - Only 12 documents, needs 50-100

### 🎯 **Root Causes**
1. **Model Size** - 3B params insufficient for:
   - Following Sinhala extraction instructions
   - Understanding complex prompts
   - Generating personalized content
   
2. **Model Training** - llama3.2 not Sinhala-focused
   - General multilingual model
   - Weak Sinhala language understanding
   - Aya 8B specifically trained on Sinhala

3. **Data Quantity** - 12 examples too small
   - Limited diversity
   - Can't cover all letter scenarios
   - Need 50-100 for good coverage

---

## Impact Analysis

### **Current Pipeline Performance**
- **Baseline (No RAG):** 0% quality (complete failure)
- **Enhanced (With RAG):** 60% quality (generic letters)
- **Improvement:** +60% (proves RAG architecture works!)

### **What RAG Saves**
Even with broken extraction:
1. Retrieved templates provide letter structure
2. Examples show formal Sinhala register
3. Context helps small model produce something useful

Without RAG, model completely fails ("I can't help").

### **Why Enhancement Seems Weak**
The +60% improvement is misleading:
- Going from 0% to 60% seems good
- But 60% means: "Generic template, not personalized"
- Real target: 85-90% (personalized, high-quality letters)

---

## Next Steps to Fix

### **Priority 1: Fix the Model** 🔴 CRITICAL
Switch from llama3.2:3b (2GB) to aya:8b (4.8GB)

**Why Aya?**
- Specifically trained on 101 languages including Sinhala
- 8B params (2.7x larger) = better instruction following
- Designed for multilingual generation tasks

**Action:**
```bash
ollama pull aya:8b
# Update config.py: ollama_model = "aya:8b"
```

### **Priority 2: Fix Extraction Prompt** 🔴 CRITICAL
Change from Sinhala instructions to English instructions

**Current (Broken):**
```python
prompt = """
මෙම ලිපි ඉල්ලීමෙන් තොරතුරු උපුටා ගන්න. JSON ආකෘතියකින් පිළිතුරු දෙන්න
"""
```

**Fixed Approach:**
```python
prompt = """
Extract key information from this Sinhala letter request.
Return ONLY valid JSON with English keys and Sinhala values.

Request: {prompt}

Return JSON format:
{{
  "letter_type": "application|request|complaint|general",
  "recipient": "<extracted Sinhala text>",
  "sender": "<extracted Sinhala text>",
  ...
}}
"""
```

### **Priority 3: Expand Knowledge Base** 🟡 MEDIUM
Add 20-30 more letter examples

**Current:** 12 documents
**Target:** 30-50 documents minimum
**Focus:** Real letter examples, not just templates

### **Priority 4: Test & Iterate** 🟢 LOW
After fixes, re-run evaluation:
```bash
python evaluate_pipeline.py
```

Target: 85%+ quality score with proper personalization

---

## Expected Results After Fixes

### **With Aya 8B + Fixed Extraction**

**Step 2 Output (Fixed):**
```json
{
  "letter_type": "request",
  "recipient": "කළමනාකරු මහතා",
  "sender": "සුනිල් පෙරේරා",
  "subject": "නිවාඩු අවසරය",
  "purpose": "අසනීප නිවාඩුවක් ලබා ගැනීම",
  "details": "අසනිප් නිසා අද පැමිණිය නොහැක"
}
```

**Step 4 Output (Fixed):**
```
Query: "ඉල්ලීම නිවාඩු අවසරය අසනීප නිවාඩුවක් කළමනාකරු සුනිල් විධිමත්"
```

**Step 8 Output (Fixed):**
```
2026 ජනවාරි 20

කළමනාකරු මහතා,
[ආයතන නම],
[ලිපිනය]

ගරු මහත්මයාණෙනි,

මාතෘකාව: නිවාඩු අවසරය

අද දිනය මට අසනීප බැවින් රැකියාවට පැමිණීමට නොහැකි වී ඇත. 
එබැවින් අද දිනය සඳහා නිවාඩු අවසරයක් ලබා දෙන ලෙස කාරුණිකව ඉල්ලා සිටිමි.

ඔබතුමාගේ කාරුණික සැලකිල්ල අපේක්ෂා කරමි.

ගෞරවයෙන්,

සුනිල් පෙරේරා
දිනය: 2026-01-20
```

**Quality Score:** 90%+ (personalized, complete, accurate)

---

## Questions for Discussion

1. Should we fix extraction first or switch model first?
   - **Recommendation:** Both simultaneously - they depend on each other

2. How many letter examples should we add?
   - **Recommendation:** Start with 20, test, then add more if needed

3. Should we keep the reranker enabled?
   - **Recommendation:** Yes, it's working well (Phase 2 success)

4. What about the NER model?
   - **Recommendation:** Low priority - LLM extraction works with right prompt

5. Test with Aya or stick with llama3.2?
   - **Recommendation:** Definitely switch to Aya - critical bottleneck
