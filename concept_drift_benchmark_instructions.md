# Task Brief for Claude Code — Concept Drift Benchmark Across Three Datasets

## 0. Objective

You are tasked with measuring, in a rigorous and comparable way, **how much concept drift** each of three datasets exhibits over time, and then deciding **which one has the strongest / most notable concept drift**.

The critical distinction you must respect throughout:

- **Covariate drift / data drift** = the input distribution **P(X)** changes over time (new vocabulary, new transaction types, new patient mixes), but the input→label relationship is unchanged.
- **Concept drift** = the relationship **P(y | X)** itself changes over time. The *same* input maps to a *different* label in a later period.

A naive "how much did things change" measurement conflates the two and will over-rank whichever dataset has the most vocabulary/feature churn. **Your job is to isolate concept drift specifically.** A dataset can have huge covariate drift and near-zero concept drift, or vice versa.

Do not prejudge the winner. Measure first, then rank from evidence.

---

## 1. Datasets (already downloaded locally)

All three live in a common project folder. Auto-detect the three subfolders by name pattern; expose the root as a `DATA_ROOT` variable at the top of your code.

| Key | Folder (name pattern) | Task | Label column | Time axis |
|---|---|---|---|---|
| `fraud` | `ieee-fraud-detection` | Binary: fraud vs legit | `isFraud` | `TransactionDT` (seconds since a reference point) |
| `diabetes` | `diabetes+130-us+hospitals...` | Readmission | `readmitted` (`<30` / `>30` / `NO`) | **UNKNOWN — must be verified (see §4)** |
| `civil` | `jigsaw-unintended-bias-in-toxicity-classification` | Binary: toxic vs not | `target` (continuous 0–1 → binarize at 0.5) | `created_date` (2015–2017) |

### Dataset-specific notes you must honor

**fraud (`ieee-fraud-detection`)**
- Use `train_transaction.csv`; optionally left-join `train_identity.csv` on `TransactionID`. **Only the training files have labels** — do the entire analysis within the labeled train set. Do not use `test_*` (unlabeled).
- Order and window by `TransactionDT`. It is a time delta in seconds, not a calendar date: you can order and bin, but you cannot map to real-world events.
- ~590k rows × ~390+ columns → memory-heavy. Downcast dtypes; sample **within each window** to a fixed cap (e.g. 30–50k rows/window) *after* temporal ordering, never globally.
- Many features are anonymized (`V1…V339`, `C*`, `D*`). You can measure drift but usually cannot name it semantically. State this limit.
- Fit any encoders/imputers on the **earliest** window and apply forward to later windows (mimics deployment; prevents leakage).

**diabetes (`diabetes+130-us+hospitals...`)**
- File is typically `diabetic_data.csv` (+ `IDS_mapping.csv`).
- Binarize `readmitted`: default = `<30` (early readmission) as positive vs everything else. Document the choice.
- Missing values are encoded as `'?'` — handle explicitly. Drop or bucket ultra-high-cardinality diagnosis codes (`diag_1/2/3`).
- **The temporal question is the crux — see §4. Do not assume a date exists; verify.**

**civil (`jigsaw-unintended-bias...`)**
- File is `all_data.csv` (or `train.csv`). Label: `target ≥ 0.5` → toxic.
- Time axis: `created_date`. Parse to datetime; window by it.
- Text feature: TF-IDF (cap at ~30–50k features). Large file → sample within windows, preserving time order.
- Identity columns (`muslim`, `black`, `homosexual_gay_or_lesbian`, …) are **not** needed for the drift metric but may be used for an optional fairness aside.

---

## 2. Mandatory methodological safeguards

These are non-negotiable. Skipping any of them makes the comparison indefensible.

### 2a. Separate covariate drift from concept drift (two distinct metrics)

**Covariate-drift score — "domain-classifier AUC" (measures P(X) change):**
For an early window vs a later window, train a binary classifier to predict *which window a row came from*, using **X only (no label)**. ROC-AUC ≈ 0.5 → no covariate shift; AUC → 1.0 → strong covariate shift. Feature space: tabular features for `fraud`/`diabetes`, TF-IDF for `civil`. This is task-agnostic and comparable across datasets.

**Concept-drift score — "stale-model gap" (measures P(y|X) change):**
1. Train `f_early` on the earliest window using (X, y).
2. Train `f_current` on a later window *k* using (X, y).
3. Evaluate **both models on the *same* held-out test split of window *k*.**
4. `concept_gap_k = perf(f_current on k) − perf(f_early on k)`.
   Because both are scored on identical window-*k* inputs (same P(X) at evaluation time), the only reason `f_current` can beat `f_early` is that the **X→y relationship changed** — i.e. concept drift.
5. **Covariate correction (required):** re-run with `f_early` retrained under **importance weighting** — reweight early-window training rows by the density ratio to window *k* (derive weights from the domain-classifier probabilities of §2a). Report the **covariate-corrected concept gap**. If the gap survives correction, it is genuine concept drift, not the old model simply never having seen window-*k*-style inputs. Report both raw and corrected.

**Interpretable cross-check — conditional label probe:**
For feature **bins** (tabular) or **tokens/n-grams** (text) whose **population share stays roughly constant across windows** (this controls for covariate shift — the "shape" didn't disappear), compute `P(y=1 | bin)` per window. Aggregate the magnitude of change across stable bins (mean absolute change, or a divergence). Movement here on frequency-stable features is direct, human-readable evidence of concept drift. For `civil`, this is the interpretable `P(toxic | token)` trajectory; surface the top drifting tokens.

### 2b. Validate the pipeline on synthetic data BEFORE trusting real results

Build small synthetic datasets with **known, injected drift** and confirm the metrics respond correctly. This is a gate — do not report real-dataset numbers until it passes.

1. **Pure covariate drift:** shift P(X) across windows (e.g. move feature means), hold P(y|X) fixed. Expect: covariate score HIGH, concept gap ≈ 0.
2. **Pure concept drift:** hold P(X) fixed, rotate/flip the decision boundary across windows so P(y|X) changes. Expect: covariate score ≈ 0, concept gap HIGH.
3. **Both:** combine; expect both metrics to fire.

If the concept metric fires on case 1 (pure covariate), the correction in §2a is broken — fix it before proceeding. Report this validation as its own section; it is what makes the real numbers credible.

### 2c. Temporal hygiene

- Split strictly chronologically. Train windows always precede evaluation windows. No shuffling across the time axis. No target leakage (fit all transforms on past data only).
- Use the **same window count K** and the **same model family** across datasets so magnitudes are comparable (default: gradient-boosted trees for tabular; logistic regression on TF-IDF for text, plus GBT-on-TF-IDF for parity). Default `K = 5`; reduce if a window would be too sparse.

### 2d. Normalize for a fair cross-dataset comparison

Tasks, base rates, and metrics differ, so **raw metric points are not comparable**. Express the concept gap as **relative degradation** (fraction of the current/oracle model's performance that the stale model loses) and report effect sizes. Use PR-AUC or balanced accuracy (robust to class imbalance) as the base metric consistently.

---

## 3. Honesty & reporting requirements

- **Invent nothing.** Every number in the final report must come from code you ran. If something can't be measured, say so plainly.
- Report a **confidence level** per dataset (High / Medium / Low) based on how well its data supports the measurement (temporal axis quality, sample size, interpretability).
- Surface **limitations before conclusions**, especially: fraud's anonymized features + ~6-month span + train-only labels; civil's moderate 2015–2017 window; and diabetes's temporal-axis situation (§4).
- Separate two questions in the verdict, because they can have different answers:
  - **(A) Largest concept-drift magnitude** — the covariate-corrected concept gap, ranked.
  - **(B) Most *notable* / defensible drift** — magnitude **× measurability/interpretability** (a large drift you can't demonstrate or explain is worth less for a student project than a clear one you can visualize).

---

## 4. Special handling: the diabetes temporal axis

This dataset is the likely weak link and must be handled with explicit care rather than forced.

1. **Inventory the columns and search for any usable time field** (admission date, year, discharge date, an explicit period). Print what you find.
2. If **no per-record date/time column exists** (the expected outcome for Diabetes 130-US), then **true temporal concept drift cannot be measured on this dataset.** Report this as the headline finding for `diabetes`, mark confidence **Low**, and do **not** manufacture a temporal signal.
3. If you fall back to `encounter_id` as an **ordering proxy**, flag it loudly as an **unvalidated assumption** (encounter_id is not documented as a clean chronological index), label every diabetes result as "proxy-based, low confidence," and never let a proxy-based number outrank a real-time-axis result in the final ranking without a caveat.

The correct outcome may well be: "diabetes is excluded from the concept-drift ranking because it lacks a temporal axis." That is a valid, defensible result — not a failure.

---

## 5. Step-by-step plan

1. **Setup:** print a file inventory for each dataset folder (paths, sizes, columns, row counts). Set `DATA_ROOT`, random seeds, and a results directory.
2. **Synthetic validation (§2b):** build the 3 synthetic scenarios, run both metrics, confirm correct behavior. Save a validation report. **Gate:** stop and report if it fails.
3. **Per dataset (`fraud`, `civil`, and `diabetes` only if §4 permits):**
   a. Load, clean, binarize label, parse/derive time axis, window into K chronological windows.
   b. Covariate-drift score (domain-classifier AUC), early-vs-late and adjacent pairs.
   c. Concept-drift score (stale-model gap), raw **and** covariate-corrected.
   d. Conditional label probe on frequency-stable bins/tokens; list top drifting bins/tokens.
   e. Plots: performance-over-time, concept gap per window, conditional-probability trajectories.
4. **Cross-dataset comparison:** assemble a normalized table (covariate score, raw concept gap, corrected concept gap, relative degradation, confidence). Keep diabetes clearly annotated per §4.
5. **Verdict:** answer (A) and (B) from §3 with a short, evidence-based justification and explicit caveats.

---

## 6. Deliverables

- A reproducible, seeded Python script or notebook, cleanly organized by the steps above.
- A `results/` folder containing: the synthetic-validation report, per-dataset metric outputs, and the plots.
- A **comparison table** (CSV + rendered in the final report).
- A concise **written report** (`concept_drift_findings.md`) that states, in order: methodology summary, synthetic-validation outcome, per-dataset results with confidence levels, the normalized comparison table, and the final verdict on (A) largest and (B) most notable concept drift — with limitations stated up front.

---

## 7. Environment

- Python with `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, and `xgboost` or `lightgbm`.
- No Kaggle API needed (data is local). Parametrize `DATA_ROOT`; do not hard-code absolute paths beyond that one variable.
- Set all random seeds for reproducibility. Downcast dtypes and sample within windows to keep memory and runtime reasonable, but never sample in a way that breaks chronological order.
- Fail gracefully with clear messages (e.g. missing file, no temporal column) rather than silently proceeding on bad assumptions.
