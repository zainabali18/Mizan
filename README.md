# Mizan · ميزان

**A sovereign Arabic intent classifier for banking customer support.**

A customer writes to their bank in Arabic, in formal Modern Standard Arabic or in dialect, for example *"بطاقتي ما وصلت"* ("my card hasn't arrived"). Mizan:

1. **removes personal data** (Emirates ID, IBAN, card and phone numbers, emails) before the text goes anywhere,
2. **classifies the request** into one of 77 banking intents,
3. returns a **confidence score**, and **routes uncertain requests to a human**.

Everything runs on open-source models that can be hosted inside a bank's own infrastructure, with **no customer data sent to an external API**.

> **The question this project answers:** for a regulated Arabic task, does a bank need a large general-purpose LLM, or does a small specialist model, fine-tuned on its own data, do the job better, faster and more cheaply?

---

## Status

| Stage | Status |
|---|---|
| 1. Data processing pipeline (normalisation, PII redaction, splits) | ✅ Done, with tests |
| 2a. Baseline: TF-IDF + logistic regression | ✅ Done |
| 2b. Baseline: zero-shot open-source LLM (Qwen2.5-1.5B) | 🔄 In progress (Colab) |
| 3. Fine-tuned Arabic model (Hugging Face) | ⏳ Planned |
| 4. FastAPI service + Docker | ⏳ Planned |
| 5. Live demo (Gradio on Hugging Face Spaces) | ⏳ Planned |

---

## Data

[**ArBanking77**](https://huggingface.co/datasets/SinaLab/ArBanking77) (Jarrar et al., 2023) is an Arabic version of the Banking77 intent dataset, manually Arabized and localised by SinaLab at Birzeit University. This project uses the public sample released on Hugging Face:

| Split | Rows | Intents present | Overlap with train |
|---|---|---|---|
| Train | 1,078 | 77 / 77 | n/a |
| Validation | 124 | 63 / 77 | 0 |
| Test | 370 (179 MSA + 191 Palestinian dialect) | 75 / 77 | 0 |

- **Two kinds of Arabic in the test set.** The test split joins an MSA file and a Palestinian-dialect file, so results are reported separately for formal Arabic and dialect. Regional dialects are where general-purpose models usually struggle.
- **Very little training data.** That's about 14 examples per intent on average, and some intents have only one. This is deliberate: banks rarely have large labelled datasets for new products, so the project tests how far a specialist model gets with very little data.
- The full ArBanking77 corpus has 31,404 queries across five dialects. Only the public sample is used here.

---

## Pipeline

```
raw text ──► Arabic-Indic digits → 0-9 ──► PII redaction ──► Arabic normalisation ──► model
                                            [EMIRATES_ID]      diacritics, tatweel,
                                            [IBAN] [PHONE]…    alef/yaa/taa forms
```

The **same `clean_text()` function** is used for training and at inference time, so the model never sees data processed differently from what it was trained on.

### PII redaction (`mizan/pii.py`)

| Type | Example | Becomes |
|---|---|---|
| Emirates ID | `784-1990-1234567-1` (also `٧٨٤-…` in Arabic-Indic digits) | `[EMIRATES_ID]` |
| IBAN | `AE07 0331 2345 6789 0123 456` | `[IBAN]` |
| Card number | `4111 1111 1111 1111` (validated with the Luhn checksum) | `[CARD_NUMBER]` |
| Phone | `+971 50 123 4567`, `0501234567`, `04 123 4567`, `+44 …` | `[PHONE]` |
| Email | `name@example.com` | `[EMAIL]` |

- **Card numbers are checked with the Luhn checksum,** so order numbers and amounts aren't mistaken for cards.
- Ordinary numbers like *"سحبت 500 جنيه"* ("I withdrew 500 pounds") are left untouched.
- **On the real dataset, the redactor produced zero false positives across all 1,572 queries.**

### Arabic normalisation (`mizan/normalise.py`)

The normaliser removes diacritics and tatweel, unifies alef forms (أ إ آ → ا), and converts ى → ي and ة → ه. It also converts Arabic-Indic digits and collapses whitespace. Customers spell the same word many ways; normalising gives the model one consistent form, which matters most when training data is small.

---

## Results

All models are scored on the same 370 held-out test queries with the same code (`mizan/evaluate.py`). **Macro-F1** is the headline metric: it weights all 77 intents equally, so a model can't score well by only getting the common intents right. Latency is measured end to end on raw text: PII redaction, normalisation and prediction.

<!-- results:start -->
| Model | Accuracy | Macro-F1 | MSA F1 | Dialect (PAL) F1 | Latency / query | Size |
|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | 67.8% | **61.0%** | 52.0% | 64.1% | 1.9 ms | 10.4 MB |
<!-- results:end -->

*More rows are coming: a zero-shot open-source LLM and a fine-tuned Arabic model. This table is generated automatically by `python -m mizan.report --write`.*

### Experiments

**1. TF-IDF + logistic regression** (`mizan/baseline_tfidf.py`). A classic CPU model using character n-grams (2 to 5 letters), which handle Arabic prefixes, suffixes and dialect spellings well. The regularisation strength was chosen on the validation split only; the test split was used once. It trains in seconds and answers in about 2 ms on a laptop CPU, a strong, cheap baseline to beat.

**2. Zero-shot open-source LLM** (`notebooks/02_zero_shot_llm.ipynb`). Qwen2.5-1.5B-Instruct is given the 77 intent names and asked to pick one, with no training. Running locally keeps it sovereign. It tests whether a general model is enough on its own.

---

## Project structure

```
mizan/
├── data/
│   ├── raw/              # original ArBanking77 CSVs (unchanged)
│   └── processed/        # generated by `python -m mizan.data`
├── mizan/
│   ├── normalise.py      # Arabic normalisation
│   ├── pii.py            # personal-data redaction
│   ├── data.py           # builds processed splits + labels.json
│   ├── evaluate.py       # shared scoring (overall + MSA vs dialect)
│   ├── baseline_tfidf.py # baseline 1
│   └── report.py         # builds the results table
├── notebooks/            # Colab: zero-shot LLM, fine-tuning
├── results/              # one JSON per experiment
├── tests/                # pytest suite (runs in GitHub Actions)
├── app.py                # demo (coming)
└── Dockerfile            # API container (coming)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m mizan.data             # builds data/processed/
python -m mizan.baseline_tfidf   # trains + scores baseline 1
python -m mizan.report --write   # refreshes the results table
python -m pytest -q              # runs the test suite
```

---

## Acknowledgements

Data: Jarrar et al. (2023). *ArBanking77: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic.* ArabicNLP 2023. [Paper](https://aclanthology.org/2023.arabicnlp-1.22) · [Dataset](https://huggingface.co/datasets/SinaLab/ArBanking77)

Built by **Zainab Ali**: [GitHub](https://github.com/zainabali18) · [LinkedIn](https://www.linkedin.com/in/zainab-ali-286058260)
