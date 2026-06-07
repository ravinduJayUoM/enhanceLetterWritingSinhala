# Expert Evaluation Results Summary

Source workbook:

```text
C:\Users\ravin\Downloads\Letter Evaluation Overall - Completed.xlsx
```

Interpretation:

- `A` = SinhalaLipi / RAG-generated letter
- `B` = Direct LLM baseline letter
- Evaluators = 5
- Prompts = 30
- Score rows = 300
- Paired rating observations per metric = 150
- Pairwise preference votes = 150

## Expert Rating Summary

| Metric | Baseline Mean | Baseline SD | RAG Mean | RAG SD | Difference |
|---|---:|---:|---:|---:|---:|
| Grammar | 2.89 | 0.34 | 3.88 | 0.38 | 0.99 |
| Structure | 3.04 | 0.26 | 3.79 | 0.48 | 0.75 |
| Contextual relevance | 3.40 | 0.49 | 3.57 | 0.41 | 0.17 |
| Formality and politeness | 2.94 | 0.17 | 3.69 | 0.51 | 0.75 |
| Fluency | 3.01 | 0.30 | 3.72 | 0.36 | 0.71 |
| Overall usability | 2.94 | 0.35 | 3.63 | 0.43 | 0.69 |

## Statistical Results

Wilcoxon signed-rank tests and paired t-tests use paired RAG-minus-baseline scores for the same evaluator and prompt.

| Metric | Wilcoxon p-value | Paired t-test p-value | Cohen's dz |
|---|---:|---:|---:|
| Grammar | 1.58e-23 | 1.18e-40 | 1.51 |
| Structure | 5.48e-21 | 2.17e-30 | 1.18 |
| Contextual relevance | 0.00136 | 0.00141 | 0.25 |
| Formality and politeness | 5.29e-22 | 2.43e-33 | 1.27 |
| Fluency | 3.88e-21 | 1.87e-32 | 1.24 |
| Overall usability | 2.59e-18 | 1.34e-23 | 0.97 |

## Pairwise Preference Results

| Preference | Count | Percentage |
|---|---:|---:|
| RAG preferred | 124 | 82.7% |
| Baseline preferred | 26 | 17.3% |
| Tie | 0 | 0.0% |

## Pairwise Preference by Category

| Category | RAG Preferred | Baseline Preferred | Tie |
|---|---:|---:|---:|
| Application | 16 | 4 | 0 |
| Appreciation | 12 | 3 | 0 |
| Complaint | 16 | 4 | 0 |
| General | 4 | 1 | 0 |
| Inquiry | 12 | 3 | 0 |
| Invitation | 16 | 4 | 0 |
| Leave | 5 | 0 | 0 |
| Recommendation | 12 | 3 | 0 |
| Request | 16 | 4 | 0 |
| Transfer | 15 | 0 | 0 |

## Inter-Rater Reliability

Krippendorff's alpha was calculated over the 60 evaluated items per metric: 30 prompts × 2 conditions.

| Metric | Krippendorff's Alpha |
|---|---:|
| Grammar | 0.995 |
| Structure | 0.993 |
| Contextual relevance | 0.990 |
| Formality and politeness | 0.991 |
| Fluency | 0.992 |
| Overall usability | 0.991 |

## Notes

No free-text evaluator comments were present in the workbook.

