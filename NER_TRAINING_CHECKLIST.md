# 🎓 NER Model Training Checklist for Sinhala Letter Slot Extraction

## 📊 **Current Status**

**Training Data Available:**
- ✅ 80 training samples (64 annotated, 16 unannotated)
- ✅ 20 validation samples (all annotated)
- ✅ Transformer format already prepared
- ✅ Training script exists (`finetune_ner_model.py`)
- ✅ Base model: XLM-RoBERTa (multilingual, supports Sinhala)

**Entity Coverage:**
- ✅ subject: 39 samples
- ✅ recipient: 30 samples  
- ✅ sender: 19 samples
- ⚠️ incident_date: 19 samples
- ⚠️ purpose: 19 samples
- ⚠️ contact_details: 11 samples
- ⚠️ details: 10 samples
- ❌ letter_type: 0 samples (CRITICAL GAP!)
- ❌ qualifications: 0 samples
- ❌ requested_action: 0 samples
- ❌ requested_items: 0 samples
- ❌ timeline: 0 samples

---

## 🚨 **Critical Issues to Address BEFORE Training**

### 1. **Missing Entity Types (High Priority)**
**Problem:** 5 entity types have ZERO training examples:
- `letter_type` - Most important! (අයදුම්පත්, ඉල්ලීම්, පැමිණිල්ල, etc.)
- `qualifications` - For job applications
- `requested_action` - What you want recipient to do
- `requested_items` - For donation requests
- `timeline` - Deadlines, date ranges

**Solution:** Need to annotate at least 15-20 examples per entity type

### 2. **Insufficient Training Data**
**Problem:** 64 annotated samples is very small for deep learning
- Minimum recommended: 200-500 samples
- Optimal: 1000+ samples

**Current coverage is insufficient for production-quality NER**

### 3. **Unbalanced Dataset**
**Problem:** Entity distribution is heavily skewed
- subject: 39 samples ✓
- recipient: 30 samples ✓
- sender/purpose/incident_date: ~19 samples ⚠️
- Other entities: 0-11 samples ❌

**Impact:** Model will be biased toward common entities, fail on rare ones

---

## ✅ **Step-by-Step Training Checklist**

### **Phase 1: Data Preparation (CRITICAL - Do This First!)**

#### ☐ Step 1.1: Annotate Missing Entity Types
**Priority: CRITICAL**

Create at least 15-20 examples for each missing entity:

**letter_type** (අයදුම්පත්, ඉල්ලීම්, etc.):
- [ ] Find/create 20 examples with clear letter type mentions
- [ ] Annotate spans: "නිවාඩු ඉල්ලීම් ලිපියක්", "රැකියා අයදුම්පත්", "පැමිණිල්ලක්"
- [ ] Add to `raw/train_samples.json`

**qualifications** (අධ්‍යාපන සුදුසුකම්, පළපුරුද්ද):
- [ ] Find/create 15 job application letters
- [ ] Annotate education, experience, skills sections
- [ ] Add to training data

**requested_action** (කරුණාකරව අනුමත කරන්න, විමර්ශනයක් සිදු කරන්න):
- [ ] Find 15 letters with clear action requests
- [ ] Annotate: "අනුමැතිය ලබා දෙන්න", "පරීක්ෂණයක් කරන්න"
- [ ] Add to training data

**requested_items** (පොත්, පරිගණක උපකරණ):
- [ ] Find 15 donation/request letters
- [ ] Annotate item lists and descriptions
- [ ] Add to training data

**timeline** (දින පරාසය, කාල සීමාව):
- [ ] Find 20 examples with dates, deadlines
- [ ] Annotate: "ජනවාරි 20 සිට 25 දක්වා", "මාර්තු 31 වන විට"
- [ ] Add to training data

#### ☐ Step 1.2: Balance Existing Entity Types
**Priority: HIGH**

- [ ] Add 20-30 more samples for `details` (currently only 10)
- [ ] Add 10-15 more samples for `contact_details` (currently only 11)
- [ ] Ensure all entity types have at least 30 samples each

#### ☐ Step 1.3: Expand Training Dataset
**Priority: MEDIUM-HIGH**

**Target: 200-300 total annotated samples**

Options:
1. **Manual Annotation** (Recommended for quality)
   - [ ] Use existing CSV data (12 letters)
   - [ ] Find real Sinhala letters from online sources
   - [ ] Create synthetic variations of existing letters
   - [ ] Annotate using a simple annotation tool

2. **Synthetic Data Generation**
   - [ ] Use LLM to generate synthetic Sinhala letters
   - [ ] Manually verify and annotate generated letters
   - [ ] Focus on underrepresented entity types

3. **Data Augmentation**
   - [ ] Synonym replacement for Sinhala words
   - [ ] Sentence reordering (maintain entity spans)
   - [ ] Back-translation (Sinhala → English → Sinhala)

#### ☐ Step 1.4: Quality Check Annotations
**Priority: CRITICAL**

- [ ] Review all annotated spans for accuracy
- [ ] Check entity boundaries (complete words, no partial words)
- [ ] Verify consistent annotation conventions
- [ ] Fix any mislabeled entities
- [ ] Remove duplicates

#### ☐ Step 1.5: Prepare Transformer Format
**Priority: HIGH**

After adding new samples:
```bash
python rag/models/prepare_ner_dataset.py
```

- [ ] Run dataset preparation script
- [ ] Verify transformer format files created
- [ ] Check token-label alignment
- [ ] Review BIO tagging format (B-entity, I-entity, O)

---

### **Phase 2: Environment Setup**

#### ☐ Step 2.1: Install Required Packages
```bash
pip install transformers torch tqdm scikit-learn seqeval
```

- [ ] Install PyTorch with CUDA (if GPU available) or CPU version
- [ ] Install transformers library
- [ ] Install seqeval (for NER metrics)
- [ ] Install tqdm (progress bars)

#### ☐ Step 2.2: Verify GPU Availability (Optional but Recommended)
```python
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

- [ ] Check if CUDA GPU is available
- [ ] If yes, training will be 10-100x faster
- [ ] If no, training on CPU will be slow but possible

---

### **Phase 3: Model Training**

#### ☐ Step 3.1: Configure Training Parameters

**Edit config if needed** (defaults are reasonable):
- `epochs`: 3-5 (start with 3)
- `batch_size`: 8 (reduce to 4 if GPU memory issues)
- `learning_rate`: 2e-5 (standard for fine-tuning)
- `model`: "xlm-roberta-base" (good for Sinhala)

#### ☐ Step 3.2: Run Initial Training
```bash
cd C:\MSC\code\enhanceLetterWritingSinhala
python rag/finetune_ner_model.py --epochs 3 --batch-size 8
```

**Expected Duration:**
- GPU: 10-30 minutes for 3 epochs with 100 samples
- CPU: 1-3 hours for 3 epochs with 100 samples

Watch for:
- [ ] Training loss decreasing each epoch
- [ ] Validation loss decreasing (not increasing = overfitting)
- [ ] Final F1 score > 0.70 is acceptable
- [ ] Final F1 score > 0.85 is good

#### ☐ Step 3.3: Monitor Training Metrics

During training, check:
- [ ] Loss curves (should decrease steadily)
- [ ] Per-entity F1 scores
- [ ] Precision and recall balance
- [ ] No significant overfitting (train vs val gap)

#### ☐ Step 3.4: Save Best Model

The script should save the best model to:
```
rag/models/training_data/best_model/
```

- [ ] Verify model files created
- [ ] Check model size (~550 MB for XLM-RoBERTa)
- [ ] Save training logs and metrics

---

### **Phase 4: Evaluation & Testing**

#### ☐ Step 4.1: Test NER Model
```bash
python rag/test_ner_model.py
```

- [ ] Test with example prompts (English and Sinhala)
- [ ] Verify all entity types are extracted
- [ ] Check extraction quality (correct spans, no hallucinations)
- [ ] Test edge cases (short prompts, mixed language, etc.)

#### ☐ Step 4.2: Calculate Metrics

Key metrics to check:
- [ ] **Overall F1 Score**: Should be > 0.70 (acceptable), > 0.85 (good)
- [ ] **Per-Entity F1 Scores**: Check each entity type individually
- [ ] **Precision**: Fewer false positives = higher precision
- [ ] **Recall**: Fewer missed entities = higher recall

#### ☐ Step 4.3: Error Analysis

- [ ] Identify which entity types have low F1 scores
- [ ] Check common error patterns (confusion matrix)
- [ ] Review false positives (model extracts wrong text)
- [ ] Review false negatives (model misses correct entities)
- [ ] Document problematic cases for data augmentation

---

### **Phase 5: Integration & Deployment**

#### ☐ Step 5.1: Update Config to Use NER

Edit `rag/config.py`:
```python
use_ner_extraction: bool = True  # Change from False to True
```

- [ ] Enable NER extraction in config
- [ ] Restart the FastAPI server
- [ ] Verify "NER model loaded successfully" in logs

#### ☐ Step 5.2: Test End-to-End Pipeline

- [ ] Start server: `python -m uvicorn sinhala_letter_rag:app`
- [ ] Test with `test_api.html` in browser
- [ ] Try various Sinhala prompts
- [ ] Verify extracted_info fields are correct
- [ ] Check that LLM fallback doesn't trigger unnecessarily

#### ☐ Step 5.3: Benchmark Performance

Compare NER vs LLM extraction:
- [ ] Accuracy: NER should be more accurate on trained entity types
- [ ] Speed: NER should be 10-100x faster than LLM
- [ ] Cost: NER is free, LLM uses Ollama (also free but slower)
- [ ] Completeness: LLM may handle edge cases better initially

---

### **Phase 6: Iteration & Improvement (Optional but Recommended)**

#### ☐ Step 6.1: Collect More Data

- [ ] Monitor production errors (entity extraction failures)
- [ ] Collect real user prompts
- [ ] Annotate problematic examples
- [ ] Add to training set

#### ☐ Step 6.2: Re-train with More Data

- [ ] Add new annotations (target: 500-1000 samples)
- [ ] Re-run training with more epochs (5-10)
- [ ] Evaluate improved model
- [ ] Deploy if metrics improve

#### ☐ Step 6.3: Try Advanced Techniques

**If F1 score < 0.70 after initial training:**

1. **Data Augmentation**
   - [ ] Implement Sinhala-specific augmentation
   - [ ] Use back-translation
   - [ ] Generate synthetic examples with LLM

2. **Model Architecture**
   - [ ] Try larger models (xlm-roberta-large)
   - [ ] Try specialized models (multilingual BERT variants)
   - [ ] Experiment with character-level models

3. **Training Optimization**
   - [ ] Increase epochs (up to 10-15)
   - [ ] Try different learning rates (1e-5 to 5e-5)
   - [ ] Implement learning rate scheduling
   - [ ] Use gradient accumulation for larger effective batch size

4. **Hybrid Approach**
   - [ ] Use NER for common entities
   - [ ] Use LLM fallback for rare/unseen entities
   - [ ] Combine NER + rule-based + LLM (ensemble)

---

## 🎯 **Minimum Viable Training Plan (Quick Start)**

If you want to get started quickly with limited resources:

### Quick Path (2-3 days):

1. **Day 1: Data Preparation (6-8 hours)**
   - Annotate 15-20 samples for each missing entity type (~75-100 new samples)
   - Use existing CSV data + find 20-30 real Sinhala letters online
   - Total target: ~150 annotated samples (up from 64)

2. **Day 2: Training & Testing (2-3 hours)**
   - Run `prepare_ner_dataset.py` to create transformer format
   - Train model: `python rag/finetune_ner_model.py --epochs 5`
   - Test model: `python rag/test_ner_model.py`
   - If F1 > 0.65, proceed. If not, add more data.

3. **Day 3: Integration & Evaluation (2-3 hours)**
   - Enable NER in config
   - Test end-to-end with various prompts
   - Compare NER vs LLM extraction quality
   - Document results for thesis

---

## 📝 **Annotation Guidelines**

When annotating new training samples in `raw/train_samples.json`:

```json
{
  "text": "ජනවාරි 20 සිට ජනවාරි 25 දක්වා වාර්ෂික නිවාඩු සඳහා මගේ කළමනාකරුට විධිමත් නිවාඩු ඉල්ලීම් ලිපියක් ලියන්න",
  "entities": {
    "letter_type": {
      "text": "නිවාඩු ඉල්ලීම් ලිපියක්",
      "start": 74,
      "end": 96
    },
    "recipient": {
      "text": "මගේ කළමනාකරුට",
      "start": 54,
      "end": 68
    },
    "timeline": {
      "text": "ජනවාරි 20 සිට ජනවාරි 25 දක්වා",
      "start": 0,
      "end": 30
    },
    "purpose": {
      "text": "වාර්ෂික නිවාඩු සඳහා",
      "start": 31,
      "end": 48
    }
  }
}
```

**Rules:**
1. `text`: Complete letter or prompt text
2. `entities`: Dictionary of entity_type → {text, start, end}
3. `start`/`end`: Character positions (0-indexed)
4. Extract complete words/phrases (no partial words)
5. Be consistent with entity boundaries

---

## 🔧 **Troubleshooting Common Issues**

### Issue 1: "CUDA out of memory"
**Solution:** Reduce batch size to 4 or 2, or use CPU

### Issue 2: "F1 score is very low (< 0.40)"
**Solution:** Need more training data (at least 200 samples)

### Issue 3: "Model extracts entire prompt as one entity"
**Solution:** Likely insufficient or incorrect annotations. Review training data quality.

### Issue 4: "Model doesn't extract rare entity types"
**Solution:** Add more examples for those entity types (minimum 20 per type)

### Issue 5: "Training takes too long (> 3 hours)"
**Solution:** Use GPU or reduce dataset size for initial testing

---

## 📚 **Resources & References**

1. **Sinhala NLP Resources:**
   - XLM-RoBERTa: https://huggingface.co/xlm-roberta-base
   - Multilingual BERT: https://huggingface.co/bert-base-multilingual-cased

2. **Annotation Tools:**
   - Label Studio: https://labelstud.io/
   - Doccano: https://github.com/doccano/doccano
   - Simple JSON editor: Just edit `train_samples.json` manually

3. **Evaluation Metrics:**
   - seqeval library: https://github.com/chakki-works/seqeval
   - NER metrics explained: https://towardsdatascience.com/evaluation-metrics-for-ner-f69b62e1e24

---

## ✨ **Expected Results After Training**

**With 150-200 well-annotated samples:**
- Overall F1: 0.65-0.75 (acceptable for MVP)
- Extraction speed: < 100ms per prompt
- Accuracy: 70-80% on test set

**With 500-1000 well-annotated samples:**
- Overall F1: 0.80-0.90 (production-quality)
- Extraction speed: < 100ms per prompt
- Accuracy: 85-95% on test set

**Benefits over LLM-only approach:**
- 10-100x faster extraction
- More consistent results
- No API costs (fully local)
- Better for thesis research (ML component)

---

## 🎓 **Academic/Thesis Value**

**Why train NER for your MSc research:**

1. **Novel Contribution:** Custom NER for Sinhala formal letters (limited prior work)
2. **Comparative Study:** Compare NER vs LLM vs Hybrid approaches
3. **Quantitative Results:** F1 scores, precision, recall metrics
4. **Ablation Studies:** Impact of training data size, entity types, model architecture
5. **Efficiency Analysis:** Speed and resource usage comparison

**Thesis Sections this enables:**
- Methodology: NER model architecture and training
- Experiments: Baseline vs. enhanced pipeline comparison
- Results: Quantitative metrics (F1, latency, cost)
- Discussion: Trade-offs between approaches

---

## 🚀 **Next Actions**

**Immediate (This Week):**
1. [ ] Annotate 15-20 examples for each missing entity type
2. [ ] Add annotations to `raw/train_samples.json`
3. [ ] Run `prepare_ner_dataset.py`

**Short-term (Next Week):**
4. [ ] Train initial model with ~150 samples
5. [ ] Evaluate model performance
6. [ ] Test end-to-end integration

**Medium-term (Next 2 weeks):**
7. [ ] Collect more training data (target: 300-500 samples)
8. [ ] Re-train with expanded dataset
9. [ ] Run comparative experiments (NER vs LLM vs Hybrid)
10. [ ] Document results for thesis

---

## 💡 **Alternative: LLM-Based Extraction (Current Approach)**

**If training NER is too time-consuming for your project timeline:**

✅ **Pros:**
- Already working with Ollama
- No training data needed
- Handles unseen entity types
- Zero-shot learning

❌ **Cons:**
- Slower (2-10 seconds per extraction)
- Less consistent results
- Harder to debug
- Less impressive for thesis (no custom ML model)

**You can continue using LLM extraction for your MVP and add NER training as a "future work" section in your thesis.**

---

**Questions? Check the code:**
- Training script: `rag/finetune_ner_model.py`
- NER model: `rag/models/sinhala_ner.py`
- Dataset prep: `rag/models/prepare_ner_dataset.py`
- Test script: `rag/test_ner_model.py`
