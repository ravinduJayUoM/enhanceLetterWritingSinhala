---
license: cc-by-sa-4.0
language:
- si
pretty_name: Anonymized Sinhala Official Letter Corpus
size_categories:
- n<1K
task_categories:
- text-generation
- text-classification
task_ids:
- language-modeling
- multi-class-classification
tags:
- sinhala
- low-resource
- letter-writing
- official-correspondence
- anonymized
- sri-lanka
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.csv
dataset_info:
  features:
  - name: id
    dtype: string
  - name: letter_category
    dtype: string
  - name: register
    dtype: string
  - name: language
    dtype: string
  - name: title
    dtype: string
  - name: content
    dtype: string
  splits:
  - name: train
    num_examples: 151
---

# Anonymized Sinhala Official Letter Corpus

A small, hand-curated corpus of **151 formal Sinhala letters**, fully anonymized with
bracketed placeholders. It is intended for training and evaluating models that generate,
complete, or classify Sinhala official correspondence — a task with very little public
training data.

## Dataset at a glance

| | |
|---|---|
| Examples | 151 |
| Language | Sinhala (`si`) |
| Register | Formal throughout |
| Letter length | 42–240 words (median 108, mean 114) |
| Placeholders | 1,408 occurrences, 56 distinct types |
| Splits | `train` only |

## Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable identifier, prefixed by category (`REQ012`, `INV004`, …) |
| `letter_category` | string | One of the 8 categories below |
| `register` | string | Formality register; `formal` for every row |
| `language` | string | ISO 639-1 code; `si` for every row |
| `title` | string | Short Sinhala description of the letter's purpose |
| `content` | string | The full anonymized letter, newline-separated |

## Category distribution

| Category | Count | Purpose of the letter |
|---|---|---|
| `notification` | 28 | Announcing a decision, circular, holiday, schedule change, or deadline |
| `request` | 27 | Asking the recipient for leave, funds, permission, supplies, or action |
| `application` | 27 | Applying on one's own behalf for a post, course, grant, licence, or claim |
| `invitation` | 27 | Inviting the recipient to a meeting, ceremony, workshop, or event |
| `appointment` | 21 | Appointing, confirming, promoting, transferring, or releasing an officer |
| `complaint` | 14 | Raising a grievance about a failure, obstruction, or unresolved problem |
| `apology` | 5 | Expressing regret and apologizing |
| `thank-you` | 2 | Expressing gratitude |

The distribution is **deliberately uneven and reflects the source material**. See
[Limitations](#limitations).

## Anonymization scheme

Every identifying element is replaced by a square-bracket placeholder whose label is in
Sinhala. Placeholder labels are part of the text and are meant to be learned by the model.

Most frequent placeholders:

| Placeholder | Count | Meaning |
|---|---|---|
| `[දිනය]` | 307 | Date |
| `[ආයතනයේ නම]` | 205 | Institution name |
| `[ලිපිනය]` | 181 | Address |
| `[ගම/නගරය]` | 127 | Village / town |
| `[ලිපි අංකය]` | 124 | Letter reference number |
| `[යවන්නාගේ නම]` | 98 | Sender name |
| `[නම]` | 86 | Person name (third party) |
| `[වේලාව]` | 46 | Time |
| `[මුදල]` | 31 | Monetary amount |
| `[ලබන්නාගේ නම]` | 20 | Recipient name |

**Removed:** personal names, institution/company/school/office names, addresses, districts
and divisions, all dates and times, file and reference numbers, NIC numbers, employee and
service numbers, vehicle and registration numbers, phone and fax numbers, email addresses,
monetary amounts, and account numbers.

**Deliberately preserved**, because they are the linguistic signal the dataset exists to teach:

- Full letter structure — date line, recipient block, salutation, subject line, body,
  closing formula, signature block, and copy-distribution list.
- Generic job titles that carry no personal identity (`ප්‍රාදේශීය ලේකම්`, `විදුහල්පති`,
  `ස්ථානාධිපති`, `කොමසාරිස්`).
- Formulaic Sinhala correspondence phrasing (`ගරු මහත්මයාණෙනි,`, `ඉහත කරුණ සම්බන්ධයෙන්`,
  `කාරුණිකව ඉල්ලා සිටිමි.`, `ගෞරවයෙන්,`).
- Public statutory and circular citations that identify no individual (for example
  `1988 අංක 09 දරණ මහාමාර්ග ආඥා පනත`).

## Usage

```python
from datasets import load_dataset

ds = load_dataset("NLPC-UOM/anonymized-sinhala-letter-corpus", split="train")
print(ds[0]["letter_category"], ds[0]["title"])
print(ds[0]["content"])

# category classification
labels = sorted(set(ds["letter_category"]))

# instruction-style generation target
def to_prompt(x):
    return {"prompt": f"{x['letter_category']} ලිපියක් ලියන්න: {x['title']}",
            "completion": x["content"]}
```

Because there is a single `train` split, create your own held-out set. With only 2
`thank-you` and 5 `apology` examples, use stratified or grouped splitting rather than a
plain random split.

## Provenance and curation

Source letters are Sri Lankan official correspondence, predominantly Divisional Secretariat
and departmental administrative letters. Construction pipeline:

1. **Filtering** — letters over 250 words dropped; empty, non-Sinhala, and OCR-garbled files removed.
2. **Deduplication** — exact-match plus 4-gram Jaccard near-duplicate detection (threshold 0.6).
3. **Type assignment** — each letter classified by its *actual communicative purpose*, not by
   keywords. This matters in Sinhala official style: `ඉල්ලා සිටිමි` ("I request") is a closing
   formula in nearly every letter and does **not** make a letter a request.
4. **Anonymization** — placeholder substitution against the scheme above.
5. **Verification** — automated scans for residual digit runs, phone numbers, NIC numbers,
   email addresses, reference codes, and Latin-script names, followed by manual review of
   every row.
6. **Normalization** — synonymous placeholder variants unified (`[ආයතනය]` → `[ආයතනයේ නම]`,
   `[දුරකථන]` → `[දුරකථන අංකය]`); the single `announcement` letter folded into `notification`.

A portion of the corpus consists of earlier curated and synthetic template-derived letters
carried forward from an internal v3 dataset; the remainder was anonymized from raw source
documents for this release.

## Limitations

- **Small.** 151 examples. Suitable for fine-tuning, few-shot prompting, and evaluation —
  not for pretraining.
- **Severely imbalanced.** `apology` (5) and `thank-you` (2) are barely represented. This is
  a property of the source domain: routine government correspondence rarely apologizes or
  thanks, and the words `කණගාටු` / `ස්තුතියි` usually appear as closing formulas on letters
  whose real purpose is something else. Do not treat per-class metrics on these two
  categories as meaningful.
- **Narrow domain.** Overwhelmingly government and administrative correspondence. Personal,
  commercial, and legal letter styles are absent or thin.
- **Formal register only.** No informal or semi-formal letters.
- **Placeholders are not natural text.** Models trained on this will emit `[දිනය]`-style
  placeholders. Downstream applications need a slot-filling step.
- **Residual risk.** Anonymization was verified by automated scan and manual review, but no
  such process is provably complete. Two benign non-identifying artifacts are known to
  remain: a statutory year in a public act citation, and the English gloss `(Data Base)`.

## Ethical considerations

The source documents are real official letters. Every direct and indirect identifier found
was removed, and no row is intended to be traceable to a specific individual, office, or
case. If you identify a residual identifier, please open a discussion on the dataset repo so
it can be corrected.

Aggregate quasi-identifiers were also scrubbed where they could narrow a location — for
example exact village household and population counts were replaced with `[සංඛ්‍යාව]`.

## License

Released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Citation

```bibtex
@misc{anonymized_sinhala_letter_corpus,
  title  = {Anonymized Sinhala Official Letter Corpus},
  year   = {2026},
  note   = {151 anonymized formal Sinhala letters across 8 categories},
  url    = {https://huggingface.co/datasets/NLPC-UOM/anonymized-sinhala-letter-corpus}
}
```
