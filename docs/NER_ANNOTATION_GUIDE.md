# NER Annotation Guide for Sinhala Letter Extraction

## What is NER (Named Entity Recognition)?

NER is a machine learning task that identifies and classifies specific pieces of information (entities) in text. For your letter writing system, we want to extract key details from Sinhala prompts.

---

## Current Annotation Format

You have **two formats** in your training data:

### **Format 1: Raw Annotations** (human-friendly)
Location: `rag/models/training_data/raw/train_samples.json`

```json
{
  "text": "ප්‍රාදේශීය ලේකම්, ප්‍රාදේශීය ලේකම් කාර්යාලය, මොණරාගල...",
  "entities": {
    "recipient": {
      "text": "ප්‍රාදේශීය ලේකම් කාර්යාලය",
      "start": 18,
      "end": 44
    },
    "purpose": {
      "text": "වාචික ප්‍රකාශයක් සටහන් කරගෙන සහතික කර මා වෙත එවන ලෙස",
      "start": 465,
      "end": 516
    },
    "details": {
      "text": "ඔබ ප්‍රාදේශීය ලේකම් කොට්ඨාශයේ මොණරාගල උඩුකුඹුර පාර",
      "start": 95,
      "end": 150
    }
  }
}
```

**Key Components:**
- `text`: Full Sinhala letter/prompt text
- `entities`: Dictionary of entity types
  - Each entity has:
    - `text`: The actual extracted text span
    - `start`: Character position where it starts
    - `end`: Character position where it ends

### **Format 2: Transformer Format** (model training)
Location: `rag/models/training_data/transformer/train.json`

```json
{
  "tokens": ["ප්‍රාදේශීය", "ලේකම්,", "ප්‍රාදේශීය", "ලේකම්", "කාර්යාලය,", ...],
  "ner_tags": [0, 0, 1, 2, 2, ...]
}
```

**Key Components:**
- `tokens`: Text split into individual words/tokens
- `ner_tags`: Numeric labels for each token
  - `0`: Not an entity (Outside)
  - `1`: Beginning of entity (B-ENTITY)
  - `2`: Inside/continuation of entity (I-ENTITY)

This format is automatically generated from Format 1 by the `prepare_ner_dataset.py` script.

---

## Entity Types You're Extracting

Based on your code, you have **6 entity types**:

1. **`letter_type`** - Type of letter (application, request, complaint, etc.)
2. **`recipient`** - Who will receive the letter
3. **`sender`** - Who is sending the letter
4. **`subject`** - Main topic/subject of the letter
5. **`purpose`** - Why the letter is being written
6. **`details`** - Additional important details

---

## Annotation Examples

### **Example 1: Simple Leave Request**

**Text:**
```
මම අසනිප් නිසා අද රැකියාවට පැමිණිය නොහැක. කරුණාකර අද සඳහා නිවාඩු අවසරයක් ලබා දෙන්න.
```

**Annotated:**
```json
{
  "text": "මම අසනිප් නිසා අද රැකියාවට පැමිණිය නොහැක. කරුණාකර අද සඳහා නිවාඩු අවසරයක් ලබා දෙන්න.",
  "entities": {
    "letter_type": {
      "text": "නිවාඩු අවසරයක්",
      "start": 63,
      "end": 76
    },
    "purpose": {
      "text": "අසනිප් නිසා අද රැකියාවට පැමිණිය නොහැක",
      "start": 3,
      "end": 39
    },
    "details": {
      "text": "අද සඳහා",
      "start": 52,
      "end": 60
    }
  }
}
```

**How to find start/end:**
- Count characters from the beginning (0-indexed)
- Include the selected text fully
- Don't include surrounding spaces

### **Example 2: Job Application**

**Text:**
```
මම ගුණසේකර රාජකීය විද්‍යාලයේ ඉංග්‍රීසි ගුරු තනතුරට අයදුම් කිරීමට කැමතියි. මට බීඑ උපාධියක් සහ වසර 5ක ඉගැන්වීමේ පළපුරුද්ද ඇත.
```

**Annotated:**
```json
{
  "text": "මම ගුණසේකර රාජකීය විද්‍යාලයේ ඉංග්‍රීසි ගුරු තනතුරට අයදුම් කිරීමට කැමතියි. මට බීඑ උපාධියක් සහ වසර 5ක ඉගැන්වීමේ පළපුරුද්ද ඇත.",
  "entities": {
    "letter_type": {
      "text": "අයදුම්",
      "start": 55,
      "end": 61
    },
    "recipient": {
      "text": "ගුණසේකර රාජකීය විද්‍යාලයේ",
      "start": 3,
      "end": 29
    },
    "subject": {
      "text": "ඉංග්‍රීසි ගුරු තනතුරට",
      "start": 30,
      "end": 51
    },
    "purpose": {
      "text": "තනතුරට අයදුම් කිරීමට",
      "start": 44,
      "end": 64
    },
    "details": {
      "text": "බීඑ උපාධියක් සහ වසර 5ක ඉගැන්වීමේ පළපුරුද්ද",
      "start": 83,
      "end": 122
    }
  }
}
```

### **Example 3: Complaint Letter**

**Text:**
```
ජනවාරි 15 වන දින මා විසින් ඔබගේ ආයතනයෙන් මිළදී ගත් රෙෆ්‍රිජරේටරය නිසි ලෙස ක්‍රියා නොකරයි. කරුණාකර මෙය හදා දෙන්න හෝ ආපසු ගෙවන්න.
```

**Annotated:**
```json
{
  "text": "ජනවාරි 15 වන දින මා විසින් ඔබගේ ආයතනයෙන් මිළදී ගත් රෙෆ්‍රිජරේටරය නිසි ලෙස ක්‍රියා නොකරයි. කරුණාකර මෙය හදා දෙන්න හෝ ආපසු ගෙවන්න.",
  "entities": {
    "letter_type": {
      "text": "පැමිණිල්ල",
      "start": 0,
      "end": 0
    },
    "recipient": {
      "text": "ඔබගේ ආයතනයෙන්",
      "start": 29,
      "end": 43
    },
    "subject": {
      "text": "රෙෆ්‍රිජරේටරය",
      "start": 55,
      "end": 68
    },
    "purpose": {
      "text": "නිසි ලෙස ක්‍රියා නොකරයි",
      "start": 69,
      "end": 91
    },
    "details": {
      "text": "ජනවාරි 15 වන දින මා විසින් මිළදී ගත්",
      "start": 0,
      "end": 36
    }
  }
}
```

**Note:** When entity type is implied (e.g., complaint) but not explicitly in text, `start=0, end=0` means "inferred from context".

---

## Current Status

### **What You Have:**
- ✅ 64 annotated samples (48 train + 16 validation)
- ✅ Annotation format defined
- ✅ Conversion script (`prepare_ner_dataset.py`)
- ✅ NER model code (`sinhala_ner.py`)

### **What's Missing:**
- ❌ **150+ more annotations needed** (target: 200+ total)
- ❌ **Entity types not fully covered:**
  - Very few `letter_type` annotations
  - Very few `sender` annotations  
  - Missing `subject` in many samples
- ❌ **Model not trained yet** (no `best_model/` folder)

---

## How to Add More Annotations

### **Option 1: Manual Annotation (Slow but Accurate)**

1. Open `rag/models/training_data/raw/train_samples.json`
2. Add new entries following the format:

```json
{
  "text": "[Your Sinhala text here]",
  "entities": {
    "entity_type": {
      "text": "[extracted text]",
      "start": [position],
      "end": [position]
    }
  }
}
```

3. Run preparation script:
```bash
python rag/models/prepare_ner_dataset.py
```

### **Option 2: Semi-Automated with Tool**

You could create an annotation tool that:
- Shows the Sinhala text
- Lets you highlight/select text
- Automatically calculates start/end positions
- Exports to JSON format

### **Option 3: Use LLM to Pre-annotate (Fast but Needs Review)**

Use Aya 8B to generate initial annotations, then manually review/correct:

```python
# Generate annotations with LLM
# Then manually review and fix errors
# Add to train_samples.json
```

---

## Pros & Cons of NER Approach

### **✅ Pros:**
1. **Proper ML solution** - Good for thesis/research
2. **Generalizes well** - Works on unseen patterns once trained
3. **Scalable** - Once trained, handles any input
4. **Academic credibility** - Shows you implemented ML properly

### **❌ Cons:**
1. **Time-consuming** - Need 150+ more annotations (10-20 hours work)
2. **Training required** - Need compute resources (local is slow)
3. **Uncertain quality** - Small dataset might not perform well
4. **Maintenance** - Need to retrain if requirements change

---

## Realistic Timeline

If you choose to complete NER training:

1. **Annotation:** 10-20 hours (150 samples × 5-8 min each)
2. **Training:** 2-4 hours (depends on hardware)
3. **Evaluation & tuning:** 2-3 hours
4. **Total:** ~15-27 hours of work

---

## My Honest Assessment

For your thesis timeline, **I don't recommend completing NER training** because:

1. **Time investment is high** for uncertain benefit
2. **Rule-based + Aya 8B LLM** will likely perform as well or better
3. **Your research focus** is on RAG pipeline improvement, not NER
4. **Practical deployment** - simpler systems are easier to maintain

**Better approach:**
- ✅ Use rule-based extraction for 70% of cases (fast, reliable)
- ✅ Use Aya 8B LLM for complex cases (robust, no training needed)
- ✅ Focus thesis research on RAG components (query building, reranking)
- ✅ Mention NER as "future work" in thesis

**NER would be valuable if:**
- You have dedicated time for annotation (20+ hours)
- You want to publish NER results as part of thesis
- You need to show deep ML expertise in this specific area

---

## Recommendation

**For your MSc thesis and timeline, I recommend:**

**Skip NER training** → Implement **Rule-Based + Aya 8B Hybrid** approach

This gives you:
- ✅ Working extraction in 1 hour (vs 20+ hours for NER)
- ✅ Good enough quality for thesis evaluation
- ✅ More time to focus on RAG pipeline experiments
- ✅ Simpler system to explain in thesis
- ✅ Easier to deploy and maintain

You can mention in your thesis:
> "While a fine-tuned NER model was initially considered, a hybrid rule-based and LLM extraction approach was implemented for practical deployment efficiency. Future work could explore specialized NER models."

**Would you like me to implement the Rule-Based + Aya 8B hybrid extractor instead?**
