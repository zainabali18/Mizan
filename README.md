# Mizan · ميزان

**A sovereign Arabic intent classifier for banking customer support.**

**[Live demo](https://mizan-banking.streamlit.app)** · **[Model on Hugging Face](https://huggingface.co/zainabali18/mizan-arabert-arbanking77)** · ![tests](https://github.com/zainabali18/Mizan/actions/workflows/tests.yml/badge.svg)

A customer writes to their bank in Arabic, in formal Modern Standard Arabic or in dialect, for example *"بطاقتي ما وصلت"* ("my card hasn't arrived"). Mizan:

1. **removes personal data** (Emirates ID, IBAN, card and phone numbers, emails) before the text reaches the model,
2. **classifies the request** into one of 77 banking intents with a fine-tuned open-source Arabic model running on CPU,
3. returns the model's score and **flags low-score requests for human review** (the flag is returned; no review queue is connected).

Everything runs on open-source models that can be hosted inside a bank's own infrastructure. The Docker image downloads the model at build time and runs **fully offline**, so **no customer data is sent to an external API**.

> **The question this project answers:** for a regulated Arabic task, does a bank need a large general-purpose LLM, or does a small specialist model, fine-tuned on its own data, do the job better, faster and more cheaply?

## Key results

| | General LLM (zero-shot) | **Fine-tuned AraBERT (Mizan)** |
|---|---|---|
| Macro-F1 on 370 test queries | 7.4% | **76.2%** |
| Dialect (Palestinian) F1 | 6.4% | **75.4%** |
| Size | 3.1 GB | **0.54 GB** (5.7× smaller) |
| Latency per query | 350 ms *on a GPU* | **104 ms on a CPU** |
| Training data | none | 1,078 examples, 23 min on a free T4 GPU |

**With a human-review threshold:** at a score threshold of 0.85 (chosen on validation data), Mizan **handles 74% of requests automatically at 89% accuracy**. The remaining **26% are flagged for a person**, and that group contains most of the model's mistakes. The score is not calibrated, though: the model can still be confidently wrong (see [Live testing](#live-testing-with-the-real-model)).

---

## Data

[**ArBanking77**](https://huggingface.co/datasets/SinaLab/ArBanking77) (Jarrar et al., 2023) is an Arabic version of the Banking77 intent dataset, manually Arabized and localised by SinaLab at Birzeit University. This project uses the public sample released on Hugging Face:

| Split | Rows | Intents present | Overlap with train |
|---|---|---|---|
| Train | 1,078 | 77 / 77 | n/a |
| Validation | 124 | 63 / 77 | 0 |
| Test | 370 (179 MSA + 191 Palestinian dialect) | 75 / 77 | 0 |

- **Two kinds of Arabic in the test set.** The test split joins an MSA file and a Palestinian-dialect file, so results are reported separately for formal Arabic and dialect.
- **Very little training data.** That's about 14 examples per intent on average, and some intents have only one. This is deliberate: banks rarely have large labelled datasets for new products.
- The full ArBanking77 corpus has 31,404 queries across five dialects; only the public sample is used here.

---

## Pipeline

```
raw text ──► Arabic-Indic digits → 0-9 ──► PII redaction ──► cleaning ──► AraBERT ──► intent + confidence
                                            [EMIRATES_ID]                                   │
                                            [IBAN] [PHONE]…                    ≥ 85% ──► answer automatically
                                                                               < 85% ──► route to a human
```

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
- **The model never sees the raw text, and the API never returns or logs it.** Both are covered by tests.

### Arabic normalisation (`mizan/normalise.py`)

For the TF-IDF baseline, the normaliser:
- removes diacritics and tatweel;
- unifies alef forms (أ إ آ → ا) and converts ى → ي and ة → ه;
- converts Arabic-Indic digits and collapses whitespace.

AraBERT was pretrained on natural Arabic, so for it only PII redaction and whitespace cleanup are applied.

---

## Results

All models are scored on the same 370 held-out test queries with the same code (`mizan/evaluate.py`). **Macro-F1** is the headline metric: it weights all 77 intents equally, so a model can't score well by only getting the common intents right. Latency is measured end to end on raw text, one query at a time: PII redaction, cleaning and prediction.

<!-- results:start -->
| Model | Accuracy | Macro-F1 | MSA F1 | Dialect (PAL) F1 | Latency / query | Size |
|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | 67.8% | **61.0%** | 52.0% | 64.1% | 0.4 ms | 10.4 MB |
| Zero-shot LLM (Qwen2.5-1.5B-Instruct) | 10.0% | **7.4%** | 8.6% | 6.4% | 350.2 ms | 3087.4 MB |
| Fine-tuned AraBERT | 79.2% | **76.2%** | 72.6% | 75.4% | 104.2 ms | 542.8 MB |
<!-- results:end -->

*Generated by `python -m mizan.report --write`. LLM latency is on a T4 GPU; TF-IDF and AraBERT latency are on CPU (AraBERT: 13 ms on GPU).*

### Experiments

**1. TF-IDF + logistic regression** (`mizan/baseline_tfidf.py`). A classic CPU model using character n-grams (2 to 5 letters), which handle Arabic prefixes, suffixes and dialect spellings well. The regularisation strength was chosen on the validation split only; the test split was used once. It's tiny and very fast: a strong, cheap baseline, and a sensible fallback if the main model is unavailable.

**2. Zero-shot open-source LLM** (`notebooks/02_zero_shot_llm.ipynb`). Qwen2.5-1.5B-Instruct was given the 77 intent names (numbered) and asked to reply with one number, with no training. It ran locally, so no data left the machine.

> **Finding:** the general-purpose LLM reached only **7.4% macro-F1**, far below even the 10 MB TF-IDF model, while being about 300× larger and far slower. In 40 of 370 cases (11%) it didn't return a valid intent number at all.

Its mistakes were mostly *plausible but wrong neighbours*. For example:
- "how old must my children be to open an account?" was labelled *country support* instead of *age limit*;
- "I received my card but not my PIN" was labelled *top-up by cash or cheque*.

With 77 fine-grained, overlapping intents, a small general model can't separate close categories from their names alone.

**3. Fine-tuned AraBERT** (`notebooks/03_fine_tune_arabert.ipynb`). This fine-tunes [`aubmindlab/bert-base-arabertv02`](https://huggingface.co/aubmindlab/bert-base-arabertv02) on the 1,078 training queries for 20 epochs, keeping the epoch with the best **validation** macro-F1 (epoch 18). The test set was scored once.

> **Finding:** fine-tuning on just 1,078 examples reached **76.2% macro-F1**, **+15.2 points over TF-IDF** and about **10× the general LLM**, and it runs on a CPU. It also performs **equally well on dialect (75.4%) and MSA (72.6%)**, where the LLM collapsed on dialect.

For context, SinaLab report about 92% F1 (MSA) for an AraBERT model trained on the full ArBanking77 corpus, which has far more data than the public sample used here. The gap shows how much more labelled data would add.

**Trade-off:** TF-IDF is over 250× faster (0.4 ms against 104 ms), but 104 ms on a CPU is still well within real-time chat requirements, and the 15-point accuracy gain matters more for customers.

### Human-review threshold (`mizan_config.json`)

A bank shouldn't auto-answer when the model is unsure. The notebook picks the lowest confidence threshold at which auto-handled **validation** requests are at least 90% accurate. It chose **0.85**, then checked that choice on the test set:

| | Share of requests | Accuracy |
|---|---|---|
| Handled automatically (confidence ≥ 0.85) | **73.8%** | **89.4%** |
| Flagged for human review | 26.2% | n/a |

This turns a 79% accurate model into a system that is about 89% accurate on everything it answers by itself. The threshold is a single setting a bank can tune to its own risk appetite.

---

## Run it

### API (FastAPI)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-serve.txt
uvicorn mizan.api:app --port 8000          # interactive docs at http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/classify -H "Content-Type: application/json" \
     -d '{"text": "بطاقتي 4111 1111 1111 1111 انسرقت، رقم جوالي 0501234567"}'
```

Real output from the published model (15 Oct 2026, MacBook CPU):

```json
{
  "intent": "الهاتف المفقود أو المسروق",
  "confidence": 0.981,
  "route": "auto",
  "threshold": 0.85,
  "top_intents": [
    {"intent": "الهاتف المفقود أو المسروق", "confidence": 0.981},
    {"intent": "بطاقة مخترقة", "confidence": 0.0014},
    {"intent": "نسيان رمز المرور", "confidence": 0.0013}
  ],
  "redacted_text": "بطاقتي [CARD_NUMBER] انسرقت، رقم جوالي [PHONE]",
  "pii_detected": ["CARD_NUMBER", "PHONE"],
  "latency_ms": 6627.5
}
```

The redaction worked, but **the intent is wrong**: this is a stolen *card*, and the model answered *lost or stolen phone* with a high score. It is kept here on purpose; see [Live testing](#live-testing-with-the-real-model) below. This run's `latency_ms` (6.6 s) came straight after the server had been idle. Once warm, requests took 44-59 ms; the very first request also downloads the 541 MB model.

Invalid requests (empty, over 1,000 characters, malformed) return HTTP 422 with the reason, and never echo the submitted text back.

Endpoints: `POST /classify`, `GET /intents`, `GET /health`.

### Live testing with the real model

Requests sent by hand to the API on a MacBook CPU (15 Oct 2026). A handful of messages, so these are observations, not measurements. Recorded in `results/evidence/live_spot_checks.json`.

| Message (stolen card, one part removed at a time) | Predicted intent | Score | Decision |
|---|---|---|---|
| Card number **and** phone number | Lost or stolen **phone** ✗ | 0.981 | auto |
| Card number only | Lost or stolen card ✓ | 0.400 | review |
| No numbers | Lost or stolen card ✓ | 0.855 | auto |
| Phone number only | Lost or stolen **phone** ✗ | 0.988 | auto |
| Formal Arabic (MSA) wording | Topping up by card ✗ | 0.436 | review |

- **A phone mention overrides the card.** Any message with "رقم جوالي" ("my phone number") went to *lost or stolen phone* at about 0.98. The threshold can't catch a confident mistake.
- **Redaction tags lower the score.** The training data contains no personal data, so the model never saw `[CARD_NUMBER]`; adding one cut the score from 0.855 to 0.400. Next step: insert fictional personal data into training examples and redact it the same way.
- **The threshold caught the uncertain errors.** Of the demo's seven examples, three were right with scores above 0.95; three (a wrong intent, a near miss and a greeting) scored below 0.85 and were flagged. The stolen-card message was the only confident mistake.
- **Speed:** 44-59 ms per request on the MacBook CPU once the model had loaded.

### Docker (CPU, offline at runtime)

```bash
docker build -t mizan .
docker run -p 8000:8000 mizan
```

- **The model is downloaded during the build;** the container then runs with `HF_HUB_OFFLINE=1`, so it never contacts the internet.
- It uses CPU-only PyTorch, runs as a non-root user, and includes a health check.

### Demo

Live on **Streamlit Community Cloud** (`demo/streamlit_app.py`, deployed straight from this repository). To run it locally:

```bash
pip install -r demo/requirements.txt
streamlit run demo/streamlit_app.py
```

The site has five pages: Overview, Live demo, Results and evaluation, How it works, and Case study. Every figure it shows is read from `results/*.json`, `results/evidence/` and `mizan_config.json`, so it can't drift from the recorded experiments (`tests/test_site_content.py` checks this). Fonts (IBM Plex Sans Arabic, Plex Mono; SIL OFL) are self-hosted from `demo/static/` and the theme lives in `.streamlit/config.toml`.

The per-intent error analysis on the Results page appears once `results/arabert_test_predictions.csv` exists. Generate it by running `notebooks/04_export_predictions.ipynb`, then commit the CSV.

A Gradio version is also included (`pip install gradio && python app.py`).

### Reproduce the experiments

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m mizan.data             # builds data/processed/
python -m mizan.baseline_tfidf   # baseline 1 (CPU, seconds)
# baselines 2-3: run notebooks/02_*.ipynb and 03_*.ipynb on Colab (T4 GPU)
python -m mizan.report --write   # refreshes the results table
python -m pytest -q              # 65 tests (runs in GitHub Actions)
```

---

## Project structure

```
├── mizan/
│   ├── pii.py            # personal-data redaction (UAE-focused)
│   ├── normalise.py      # Arabic normalisation
│   ├── data.py           # builds processed splits + labels.json
│   ├── evaluate.py       # shared scoring (overall + MSA vs dialect)
│   ├── baseline_tfidf.py # baseline 1
│   ├── report.py         # builds the results table
│   ├── service.py        # production classifier: redact → predict → route
│   └── api.py            # FastAPI app
├── notebooks/            # Colab: zero-shot LLM, AraBERT fine-tuning
├── results/              # one JSON per experiment
├── tests/                # 65 tests; model mocked so CI needs no GPU
├── demo/                 # Streamlit site (live): streamlit_app.py, views/, webui/, static/
├── app.py                # Gradio demo (alternative)
├── mizan_config.json     # model id + human-review threshold
├── Dockerfile
└── data/                 # raw + processed ArBanking77 sample
```

## Limitations and next steps

- **Small evaluation set.** Some intents appear only once or twice in the test set, so per-intent scores are noisy.
- **Dialect coverage.** Only MSA and Palestinian dialect are tested. Gulf dialects (Emirati, Saudi) need their own evaluation data before deployment in the UAE.
- **Next steps:**
  - train on the full ArBanking77 corpus;
  - try parameter-efficient fine-tuning (LoRA) of a small open LLM;
  - train with fictional personal data inserted and redacted, so the model learns tags like `[CARD_NUMBER]`;
  - measure and calibrate the model score;
  - add model monitoring (score drift, review rate) in production.

---

## Acknowledgements

Data: Jarrar et al. (2023). *ArBanking77: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic.* ArabicNLP 2023. [Paper](https://aclanthology.org/2023.arabicnlp-1.22) · [Dataset](https://huggingface.co/datasets/SinaLab/ArBanking77)

Base model: [AraBERT v0.2](https://huggingface.co/aubmindlab/bert-base-arabertv02) by AUB MIND Lab.
