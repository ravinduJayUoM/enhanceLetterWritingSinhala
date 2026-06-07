# Evaluation Execution Guide

This file is a working guide for preparing and running the thesis evaluation for the SinhalaLipi RAG-based formal Sinhala letter generation system.

## 1. Evaluation Goal

The evaluation should answer the main thesis question:

> Does the RAG-based Sinhala letter generation pipeline produce better formal Sinhala letters than a standard LLM baseline?

The main comparison is:

- **Baseline:** The same LLM receives only the user prompt and a simple instruction to write a formal Sinhala letter.
- **RAG pipeline:** The prompt goes through information extraction, missing-field handling, retrieval, prompt construction, and letter generation.

Use the same LLM model and temperature for both conditions.

Recommended setup:

- Model: `gemini-2.5-flash`
- Temperature: `0.3`
- Evaluation prompts: `30`
- Expert evaluators: `5`
- Output conditions per prompt: `2` baseline and RAG

## 2. Evaluation Dataset Size

Use 30 prompts. This is large enough to cover the major letter categories while keeping the expert workload practical.

Recommended distribution:

| Category | Prompt count |
|---|---:|
| Request | 4 |
| Application | 4 |
| Complaint | 4 |
| Invitation | 4 |
| Appreciation | 3 |
| Recommendation | 3 |
| Inquiry | 3 |
| Transfer | 3 |
| Leave | 2 |
| General / other formal | 3 |
| **Total** | **30** |

The final set should not copy exact prompts from the retrieval corpus. It can be inspired by real use cases, but wording should be new.

## 3. Sample Prompt Set

These are sample prompts to show the expected style, coverage, and level of detail. You can replace or adapt them before the real evaluation.

| ID | Category | Prompt |
|---|---|---|
| P01 | Request | මාගේ විශ්වවිද්‍යාල ශිෂ්‍ය වාර්තාවේ පිටපතක් ලබා ගැනීමට ලේඛකාධිකාරීතුමා වෙත ඉල්ලීමක් ලියන්න. |
| P02 | Request | අපගේ ග්‍රාමයේ ප්‍රධාන මාර්ගයේ වීදි ලාම්පු අලුත්වැඩියා කර දෙන ලෙස ප්‍රාදේශීය සභාවෙන් ඉල්ලීමට ලිපියක් ලියන්න. |
| P03 | Request | රැකියා ස්ථානයේදී අමතර පුහුණු වැඩසටහනකට සහභාගී වීමට කළමනාකරුගෙන් අවසර ඉල්ලන ලිපියක් ලියන්න. |
| P04 | Request | පාසල් පුස්තකාලයට නව පොත් ලබා දෙන ලෙස විදුහල්පතිතුමාගෙන් ඉල්ලීමක් ලියන්න. |
| P05 | Application | කාර්යාල සහකාර තනතුරක් සඳහා පුවත්පත් දැන්වීමකට අනුව රැකියා අයදුම්පත් ලිපියක් ලියන්න. |
| P06 | Application | පරිගණක විද්‍යා පාඨමාලාවකට ඇතුළත් වීමට පුහුණු ආයතනයට අයදුම්පත් ලිපියක් ලියන්න. |
| P07 | Application | පාසල් ගුරු තනතුරක් සඳහා අයදුම් කිරීමට අධ්‍යාපන අධ්‍යක්ෂකතුමා වෙත ලිපියක් ලියන්න. |
| P08 | Application | ශිෂ්‍යත්ව වැඩසටහනකට අයදුම් කිරීමට අවශ්‍ය නිල ලිපියක් සකස් කරන්න. |
| P09 | Complaint | මාසයක් පුරා අක්‍රියව පවතින අන්තර්ජාල සම්බන්ධතාවය පිළිබඳ සේවා සපයන ආයතනයට පැමිණිලි ලිපියක් ලියන්න. |
| P10 | Complaint | මහජන බස් සේවාවේ නියමිත වේලාවන් නොපිළිපැදීම පිළිබඳ ප්‍රවාහන අධිකාරියට පැමිණිල්ලක් ලියන්න. |
| P11 | Complaint | මිලදී ගත් විදුලි උපකරණයක් දෝෂ සහිත වීම පිළිබඳ වෙළඳසැලට පැමිණිලි ලිපියක් ලියන්න. |
| P12 | Complaint | ප්‍රදේශයේ කසළ එකතු කිරීම ප්‍රමාද වීම පිළිබඳ නගර සභාවට පැමිණිල්ලක් ලියන්න. |
| P13 | Invitation | පාසලේ වාර්ෂික ත්‍යාග ප්‍රදානෝත්සවයට ප්‍රධාන ආරාධිත අමුත්තා ලෙස සහභාගී වන ලෙස කලාප අධ්‍යාපන අධ්‍යක්ෂතුමාට ආරාධනා ලිපියක් ලියන්න. |
| P14 | Invitation | ග්‍රාම සංවර්ධන සමිතියේ සාමාජික රැස්වීමට ප්‍රාදේශීය ලේකම්තුමාට ආරාධනා කරන ලිපියක් ලියන්න. |
| P15 | Invitation | පුස්තකාල විවෘත කිරීමේ උත්සවයට විදුහල්පතිතුමාට ආරාධනා කරන නිල ලිපියක් ලියන්න. |
| P16 | Invitation | සෞඛ්‍ය දැනුවත් කිරීමේ වැඩසටහනකට වෛද්‍ය නිලධාරීතුමාට ආරාධනා ලිපියක් ලියන්න. |
| P17 | Appreciation | පාසලට පරිගණක උපකරණ පරිත්‍යාග කළ ආයතනයකට ස්තුති ලිපියක් ලියන්න. |
| P18 | Appreciation | පුහුණු වැඩසටහනක් සාර්ථකව පැවැත්වීමට සහාය වූ සම්පත් දායකයාට ස්තුති ලිපියක් ලියන්න. |
| P19 | Appreciation | ග්‍රාමීය පාර අලුත්වැඩියා කිරීමට සහාය වූ ප්‍රාදේශීය සභාවට ස්තුති ලිපියක් ලියන්න. |
| P20 | Recommendation | උසස් අධ්‍යාපන අවස්ථාවක් සඳහා ශිෂ්‍යයෙකු නිර්දේශ කරන ලිපියක් ලියන්න. |
| P21 | Recommendation | රැකියා අවස්ථාවක් සඳහා හිටපු සේවකයෙකු නිර්දේශ කරන නිල ලිපියක් ලියන්න. |
| P22 | Recommendation | ක්‍රීඩා තරඟාවලියකට සහභාගී වීමට සිසුවෙකු නිර්දේශ කරන ලිපියක් ලියන්න. |
| P23 | Inquiry | විශ්වවිද්‍යාල පාඨමාලා ගාස්තු සහ ඇතුළත් වීමේ අවශ්‍යතා පිළිබඳ විමසීමට ලිපියක් ලියන්න. |
| P24 | Inquiry | බැංකු ණය පහසුකම් පිළිබඳ වැඩි විස්තර ලබා ගැනීමට විමසීම් ලිපියක් ලියන්න. |
| P25 | Inquiry | පුහුණු වැඩසටහනක දිනය, කාලය, සහ ලියාපදිංචි වීම පිළිබඳ විමසීමට ලිපියක් ලියන්න. |
| P26 | Transfer | පවුල් හේතුවක් නිසා සේවා ස්ථාන මාරුවක් ඉල්ලීමට දෙපාර්තමේන්තු ප්‍රධානියාට ලිපියක් ලියන්න. |
| P27 | Transfer | දුරස්ථ සේවා ස්ථානයක සිට නිවසට ආසන්න ශාඛාවකට මාරුවක් ඉල්ලන ලිපියක් ලියන්න. |
| P28 | Transfer | පාසල් මාරුවක් ඉල්ලීමට කලාප අධ්‍යාපන කාර්යාලයට දෙමාපියෙකු ලෙස ලිපියක් ලියන්න. |
| P29 | Leave | අසනීප තත්ත්වයක් හේතුවෙන් දින තුනක නිවාඩුවක් ඉල්ලීමට කළමනාකරුට ලිපියක් ලියන්න. |
| P30 | General | වැඩමුළුවක සහභාගීත්ව සහතිකයක් ලබා ගැනීමට සංවිධායක මණ්ඩලයට නිල ලිපියක් ලියන්න. |

## 4. Prompt Metadata Sheet

Create a spreadsheet or CSV with these columns:

```text
prompt_id,category,prompt,expected_recipient,expected_sender,expected_subject,complexity,input_style,notes
```

Suggested values:

- `complexity`: `simple`, `medium`, `complex`
- `input_style`: `sinhala`, `singlish`, `mixed`
- `notes`: optional explanation of what the prompt is testing

## 5. Generation Output Sheet

For each prompt, generate two outputs and store them before sending to evaluators.

```text
prompt_id,condition,model,temperature,generated_letter,enhanced_prompt,retrieved_doc_ids,timestamp
```

Use:

- `condition = baseline`
- `condition = rag`

Do not send the `condition`, `enhanced_prompt`, or `retrieved_doc_ids` to evaluators.

## 6. Blind Evaluation Sheet

Create a blinded sheet where each prompt has two anonymous outputs:

```text
prompt_id,display_order,letter_a,letter_b,actual_a_condition,actual_b_condition
```

Before sending to evaluators, remove or hide:

```text
actual_a_condition,actual_b_condition
```

Randomize whether baseline or RAG appears as Letter A for each prompt.

## 7. Expert Rating Form

Each evaluator should rate both letters independently using a 1-5 scale.

Recommended columns:

```text
evaluator_id,prompt_id,letter_label,grammar,structure,relevance,formality,fluency,overall_usability,comments
```

Rating definitions:

| Score | Meaning |
|---:|---|
| 1 | Poor; major rewriting required |
| 2 | Weak; several serious issues |
| 3 | Acceptable; usable with noticeable edits |
| 4 | Good; usable with minor edits |
| 5 | Excellent; can be used as-is |

Evaluator dimensions:

- **Grammar:** grammatical correctness, spelling, agreement, sentence correctness
- **Structure:** formal letter structure, order, salutation, subject, body, closing
- **Relevance:** whether the letter addresses the prompt and includes required details
- **Formality:** written Sinhala register, politeness, honorific appropriateness
- **Fluency:** natural flow and readability
- **Overall usability:** whether the evaluator would send the letter with little or no editing

## 8. Pairwise Preference Form

For each prompt, also ask:

> Which letter would you send?

Allowed values:

```text
A, B, Tie
```

Suggested columns:

```text
evaluator_id,prompt_id,preferred_letter,preference_reason
```

## 9. Analysis to Run After Evaluation

Compute the following:

### Expert ratings

- Mean and standard deviation for baseline and RAG per metric
- Mean improvement: `RAG mean - baseline mean`
- Wilcoxon signed-rank test per metric
- Optional paired t-test per metric
- Effect size per metric

### Preference results

- RAG wins
- Baseline wins
- Ties
- RAG win rate by category

### Inter-rater reliability

- Krippendorff's alpha or Fleiss' kappa per metric

### Retrieval support metrics

- Category hit rate
- Template coverage
- Precision@3

### System feedback

From the app/admin dashboard:

- Average overall letter quality
- Average match to request
- Average language quality
- Average structure quality
- Average ease of use
- Average confidence in output
- Average would-use-again score

## 10. Thesis Result Tables to Fill

The thesis Evaluation chapter should eventually include:

1. Evaluation dataset summary
2. Baseline vs RAG mean expert ratings
3. Statistical test results
4. Pairwise preference results
5. Extraction evaluation results
6. Retrieval evaluation results
7. User/system feedback results
8. Qualitative error analysis

