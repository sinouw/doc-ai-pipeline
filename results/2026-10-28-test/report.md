# Evaluation run 2026-10-28T09:42:46+00:00

- Split: `data/test`. Methods are compared on the 18 documents every method processed
  (ar-001, ar-002, ar-003, ar-004, ar-005, ar-006, en-001, en-002, en-003, en-004, en-005, en-006, fr-001, fr-002, fr-003, fr-004, fr-005, fr-006)
- LLM: `qwen/qwen3.8-27b` on Groq (free tier), temperature 0
- Machine: MacBook Pro 2019, Intel Core i7-9750H (6 cores / 12 threads), 16 GB RAM, CPU only, Python 3.12.15
- Runs: 21 invocations of the resumable runner (commits `b306aa0`, `c3b28ea`, `fca022e`); 19 stopped on Groq's daily token cap and were resumed later; peak memory 887 MB
- Deskew: mean absolute error 0.03 degrees vs the generator's rotation

## Headline

| Method | Field accuracy | Doc exact match | Latency p50 / p95 (s/page) | Tokens in/out per page | Cost per 1k pages |
|---|---|---|---|---|---|
| OCR + rules, no clean-up | 77.3% | 44.4% | 4.5 / 5.0 | 0 / 0 | $0.00 |
| OCR + rules | 82.4% | 50.0% | 5.1 / 6.3 | 0 / 0 | $0.00 |
| OCR text -> LLM | 94.0% | 61.1% | 9.4 / 16.0 | 656 / 363 | $1.98 |
| Vision LLM (image only) | 97.2% | 72.2% | 9.9 / 13.7 | 2163 / 370 | $3.21 |
| Hybrid (image + OCR text) | 99.1% | 88.9% | 8.7 / 13.7 | 2449 / 372 | $3.45 |

Latency is wall clock per page on CPU (render, clean-up, OCR, extraction) minus time spent sleeping on HTTP 429
back-offs; the raw wall clock and the back-off totals are in `summary.json`. Cost uses Groq's list price for the model
(see `eval/run_eval.py`); the OCR + rules methods make no model call.

## Field accuracy by language

| Method | ar | en | fr |
|---|---|---|---|
| OCR + rules, no clean-up | 43.1% | 90.3% | 98.6% |
| OCR + rules | 51.4% | 95.8% | 100.0% |
| OCR text -> LLM | 83.3% | 100.0% | 98.6% |
| Vision LLM (image only) | 94.4% | 97.2% | 100.0% |
| Hybrid (image + OCR text) | 97.2% | 100.0% | 100.0% |

Document exact match by language:

| Method | ar | en | fr |
|---|---|---|---|
| OCR + rules, no clean-up | 0.0% | 50.0% | 83.3% |
| OCR + rules | 0.0% | 50.0% | 100.0% |
| OCR text -> LLM | 0.0% | 100.0% | 83.3% |
| Vision LLM (image only) | 50.0% | 66.7% | 100.0% |
| Hybrid (image + OCR text) | 66.7% | 100.0% | 100.0% |

## Field accuracy by scan degradation

| Method | light | medium | heavy |
|---|---|---|---|
| OCR + rules, no clean-up | 86.1% | 79.2% | 66.7% |
| OCR + rules | 88.9% | 81.9% | 76.4% |
| OCR text -> LLM | 97.2% | 90.3% | 94.4% |
| Vision LLM (image only) | 98.6% | 98.6% | 94.4% |
| Hybrid (image + OCR text) | 98.6% | 98.6% | 100.0% |

## Accuracy per field

| Field | OCR + rules, no clean-up | OCR + rules | OCR text -> LLM | Vision LLM (image only) | Hybrid (image + OCR text) |
|---|---|---|---|---|---|
| invoice_number | 94.4% | 100.0% | 100.0% | 100.0% | 100.0% |
| issue_date | 88.9% | 88.9% | 100.0% | 100.0% | 100.0% |
| due_date | 83.3% | 88.9% | 94.4% | 100.0% | 100.0% |
| seller_name | 94.4% | 100.0% | 94.4% | 88.9% | 94.4% |
| seller_vat_id | 55.6% | 61.1% | 72.2% | 94.4% | 100.0% |
| buyer_name | 66.7% | 66.7% | 88.9% | 83.3% | 94.4% |
| currency | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| subtotal | 83.3% | 94.4% | 100.0% | 100.0% | 100.0% |
| vat_rate | 66.7% | 66.7% | 77.8% | 100.0% | 100.0% |
| vat_amount | 66.7% | 77.8% | 100.0% | 100.0% | 100.0% |
| total | 61.1% | 66.7% | 100.0% | 100.0% | 100.0% |
| line_items | 66.7% | 77.8% | 100.0% | 100.0% | 100.0% |
| line items incl. descriptions | 38.9% | 72.2% | 77.8% | 83.3% | 94.4% |

## Human review routing

A field goes to the review queue when its confidence is below the threshold or validation blames it.
`errors caught` is the share of wrong fields that landed in the queue; `auto-accepted accuracy` is the
accuracy of the fields that skipped it.

**OCR + rules, no clean-up**

| Threshold | Fields sent to review | Auto-accepted accuracy | Errors caught | Errors missed |
|---|---|---|---|---|
| 0.5 | 24.1% | 100.0% | 100.0% | 0 |
| 0.6 | 24.1% | 100.0% | 100.0% | 0 |
| 0.7 | 24.1% | 100.0% | 100.0% | 0 |
| 0.8 | 24.1% | 100.0% | 100.0% | 0 |
| 0.9 | 24.5% | 100.0% | 100.0% | 0 |

**OCR + rules**

| Threshold | Fields sent to review | Auto-accepted accuracy | Errors caught | Errors missed |
|---|---|---|---|---|
| 0.5 | 19.0% | 99.4% | 97.4% | 1 |
| 0.6 | 19.0% | 99.4% | 97.4% | 1 |
| 0.7 | 19.0% | 99.4% | 97.4% | 1 |
| 0.8 | 19.0% | 99.4% | 97.4% | 1 |
| 0.9 | 19.0% | 99.4% | 97.4% | 1 |

**OCR text -> LLM**

| Threshold | Fields sent to review | Auto-accepted accuracy | Errors caught | Errors missed |
|---|---|---|---|---|
| 0.5 | 8.8% | 99.5% | 92.3% | 1 |
| 0.6 | 9.7% | 99.5% | 92.3% | 1 |
| 0.7 | 10.6% | 99.5% | 92.3% | 1 |
| 0.8 | 13.4% | 100.0% | 100.0% | 0 |
| 0.9 | 19.0% | 100.0% | 100.0% | 0 |

**Vision LLM (image only)**

| Threshold | Fields sent to review | Auto-accepted accuracy | Errors caught | Errors missed |
|---|---|---|---|---|
| 0.5 | 5.1% | 99.5% | 83.3% | 1 |
| 0.6 | 11.1% | 99.5% | 83.3% | 1 |
| 0.7 | 11.6% | 99.5% | 83.3% | 1 |
| 0.8 | 13.9% | 100.0% | 100.0% | 0 |
| 0.9 | 19.9% | 100.0% | 100.0% | 0 |

**Hybrid (image + OCR text)**

| Threshold | Fields sent to review | Auto-accepted accuracy | Errors caught | Errors missed |
|---|---|---|---|---|
| 0.5 | 3.7% | 100.0% | 100.0% | 0 |
| 0.6 | 9.7% | 100.0% | 100.0% | 0 |
| 0.7 | 10.6% | 100.0% | 100.0% | 0 |
| 0.8 | 13.0% | 100.0% | 100.0% | 0 |
| 0.9 | 19.0% | 100.0% | 100.0% | 0 |

## Validation

Some invoices print a total that does not equal subtotal + VAT. The table shows how often the arithmetic checks
flag those documents, and how often they flag documents whose printed numbers are consistent (a false alarm,
caused by an extraction error).

| Method | Inconsistent invoices flagged | Consistent invoices flagged | LLM calls | 429 retries | Errors |
|---|---|---|---|---|---|
| OCR + rules, no clean-up | 2/3 | 3/15 | 0 | 0 | 0 |
| OCR + rules | 2/3 | 3/15 | 0 | 0 | 0 |
| OCR text -> LLM | 3/3 | 0/15 | 18 | 9 | 0 |
| Vision LLM (image only) | 3/3 | 0/15 | 18 | 0 | 0 |
| Hybrid (image + OCR text) | 3/3 | 0/15 | 18 | 11 | 0 |

## Methods run on more documents

The OCR-only methods cost nothing to run, so they also ran on the full split.

| Method | Docs | Field accuracy | Doc exact match | ar field acc. | en field acc. | fr field acc. | Latency p50 / p95 (s/page) |
|---|---|---|---|---|---|---|---|
| OCR + rules, no clean-up | 36 | 79.4% | 44.4% | 46.5% | 94.4% | 97.2% | 4.3 / 4.8 |
| OCR + rules | 36 | 82.6% | 52.8% | 52.1% | 95.8% | 100.0% | 4.9 / 5.9 |

