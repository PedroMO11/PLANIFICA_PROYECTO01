# Concept-Drift Benchmark — Findings

**Question:** of the three datasets (`fraud` = IEEE-CIS fraud, `civil` = Jigsaw
unintended-bias toxicity, `diabetes` = 130-US hospitals), which exhibits the
strongest and most *notable* **concept drift** — a change in **P(y | X)** over
time — as opposed to mere **covariate drift** (a change in **P(X)** with the
input→label rule unchanged).

Everything below comes from `cd_bench/run_benchmark.py` (seeded, reproducible).
Raw outputs: `cd_bench/results/` (per-dataset JSON, `comparison_table.csv`,
`synthetic_validation.md`, plots).

---

## 1. Methodology summary

Two orthogonal metrics per dataset, plus an interpretable cross-check. Same
window count (**K = 5**, equal-count chronological windows) and same model family
across datasets so magnitudes are comparable: gradient-boosted trees (LightGBM)
for tabular, logistic-regression-on-TF-IDF for text (with a GBT-on-TF-IDF parity
check). Base metric is **PR-AUC** throughout (robust to the heavy class
imbalance); also reported base-rate-normalized as *skill* = (PR-AUC − base) /
(1 − base).

| Metric | What it measures | How |
|---|---|---|
| **Covariate-drift score** | P(X) change | Domain-classifier ROC-AUC: train a model to predict *which window* a row came from, **X only, no label**. 0.5 = no shift, 1.0 = fully separable. |
| **Concept-drift score — "stale-model gap"** | P(y\|X) change | Train `f_early` on window 0 and `f_current` on window *k*; score **both on the same held-out slice of window *k***. `gap_k = PR-AUC(f_current) − PR-AUC(f_early)`. Identical evaluation inputs ⇒ the only way `f_current` can win is that the X→y rule moved. |
| **Covariate correction** (required) | removes "old model never saw new-style inputs" | Refit `f_early` with **importance weights** = density ratio p(window *k*)/p(window 0), derived from a calibrated logistic domain classifier (out-of-fold probabilities, power-shrunk, blended toward 1 in proportion to the measured covariate AUC so a window with no covariate shift gets ~no reweighting). Report **raw and corrected**. Effective-sample-size fraction (ESS) reported so degenerate weights are visible. |
| **Conditional-label probe** (interpretable) | direct, human-readable evidence | On feature **bins** (tabular) / **tokens** (text) whose *population share stays ~constant across windows* (controls for covariate shift), track `P(y=1 \| bin)` window 0 → last window; aggregate mean \|Δ\|. |

Temporal hygiene: strictly chronological splits, train windows precede
evaluation windows, all transforms (encoders, TF-IDF) fit on the earliest window
and applied forward.

---

## 2. Synthetic-validation outcome — **GATE PASSED**

Three synthetic datasets with known injected drift (K=3, 20 features, GBT). The
pipeline was not allowed to touch real data until these behaved correctly.

| Scenario (injected) | Covariate AUC | Raw concept gap | **Corrected concept gap** | Expected | Result |
|---|---|---|---|---|---|
| **Pure covariate** (shift feature means, P(y\|X) fixed) | 0.86 | −0.005 | **+0.005** | cov HIGH, concept ≈ 0 | ✅ |
| **Pure concept** (rotate/flip decision boundary, P(X) fixed) | 0.49 | +0.415 | **+0.415** | cov ≈ 0.5, concept HIGH | ✅ |
| **Both** | 0.86 | +0.409 | **+0.406** | both fire | ✅ |

The key property holds: **the concept metric does not fire on pure covariate
drift** (corrected gap +0.005, i.e. < 1.5 % of the genuine concept signal), and
the correction neither manufactures nor destroys signal. All 10 numeric checks
passed (`cd_bench/results/synthetic_validation.md`).

---

## 3. Limitations (stated before conclusions)

- **`fraud`** — features are anonymized (`V1…V339`, `C*`, `D*`), so drift can be
  *measured* but not *named*. The time axis is a delta in seconds (orderable, not
  mappable to calendar events) spanning only **~182 days**. Only the training
  split has labels, so the whole analysis lives inside it. Rows sampled to
  30 k/window **after** temporal ordering (150 k total).
- **`civil`** — real timestamps, but the usable span is effectively
  **2016–2017** (2015 has < 1 000 rows). Equal-count windowing therefore places
  window 0 across 2015-09 → 2016-12 and **windows 1–4 all inside 2017**
  (edges: 2016-12-09, 2017-03-14, 2017-06-19, 2017-09-02, 2017-11-11), so most of
  the resolution is on a single year. 110 k rows sampled within windows;
  `toxicity ≥ 0.5` → toxic (`all_data.csv`'s `toxicity` is the competition
  `target`).
- **`diabetes`** — see §4. **No temporal axis exists.** Everything for this
  dataset is either "not measurable" or an explicitly-flagged proxy.
- The stale-model gap is a *lower bound* on concept drift (a flexible current
  model can partly absorb boundary movement). Cross-dataset probe magnitudes are
  **not** directly comparable (tabular bins vs word-presence indicators are
  different scales) — the probe is a within-dataset cross-check.

---

## 4. The `diabetes` temporal axis — headline finding

**Diabetes 130-US has no per-record calendar field.** Column inventory (50
columns): no `*_date`, no admission/discharge date, no year/quarter/period; **no
column parses as a date** (>80 % threshold). The only per-row key is
`encounter_id`, a surrogate identifier that is **not monotonic in file order**
and is undocumented as a chronological index.

➡️ **True temporal concept drift cannot be measured on `diabetes`. It is
excluded from the ranking.** This is a valid, defensible result, not a failure.

*Proxy attempt, loudly flagged as unvalidated:* ordering by `encounter_id`
ascending and running the identical battery (positive class = `readmitted` `<30`,
base rate 11.2 %) gives a corrected concept gap of **+0.014 PR-AUC** (raw
+0.006), ~7 % relative degradation on a very low absolute PR-AUC (~0.19), most of
which is contributed by the correction itself. The domain-classifier "AUC" along
the proxy order climbs 0.85 → 0.92, i.e. `encounter_id` ordering does track
*something* systematic — most plausibly hospital/data-source batches, not time.
**Confidence: Low. This number must not outrank a real-time-axis dataset.**

---

## 5. Per-dataset results

### 5.1 `fraud` — Confidence: **Medium**

| window (→ day) | covariate AUC vs w0 | PR-AUC `f_current` | PR-AUC `f_early` | PR-AUC `f_early` corrected | raw gap | **corrected gap** | rel. degradation | ESS |
|---|---|---|---|---|---|---|---|---|
| w1 (→25.7) | 0.780 | 0.579 | 0.505 | 0.511 | +0.074 | +0.068 | 12.8 % | 0.87 |
| w2 (→63.7) | 0.862 | 0.431 | 0.330 | 0.311 | +0.101 | +0.120 | 27.8 % | 0.77 |
| w3 (→100.2) | 0.931 | 0.512 | 0.387 | 0.399 | +0.125 | +0.113 | 22.0 % | 0.73 |
| w4 (→140.1) | 0.945 | 0.570 | 0.498 | 0.489 | +0.073 | +0.082 | 14.3 % | 0.75 |
| **mean** | **0.88** | | | | **+0.093** | **+0.096** | **19.0 %** | |

- **Strong, monotone covariate drift** (domain AUC 0.78 → 0.95) **and** strong
  concept drift on top of it. Correction barely moves the gap
  (0.093 → 0.096) and ESS stays 0.73–0.87 (healthy support overlap), so — given
  the synthetic gate — this is **genuine P(y\|X) drift, not novelty artifact**.
- A fraud model frozen at month 0 loses **~19 % of its achievable PR-AUC** on
  later months purely because the fraud/legit boundary moved — textbook
  adversarial concept drift.
- Conditional probe on frequency-stable bins: mean \|ΔP(fraud\|bin)\| = **0.008**
  in absolute points, but against a 3.5 % base rate this is a **~50 % relative
  rise** in conditional fraud probability for fixed input regions (e.g. `V127`
  low bin 0.020 → 0.035; `card1` bins ~0.02 → 0.03–0.04) — consistent, systematic
  upward movement. Cannot be named (anonymized features).

### 5.2 `civil` — Confidence: **Medium**

| window | covariate AUC vs w0 | PR-AUC `f_current` | PR-AUC `f_early` | corrected | raw gap | **corrected gap** | rel. degr. | ESS |
|---|---|---|---|---|---|---|---|---|
| w1 | 0.679 | 0.396 | 0.367 | 0.358 | +0.029 | +0.038 | 9.5 % | 0.94 |
| w2 | 0.675 | 0.478 | 0.454 | 0.444 | +0.024 | +0.034 | 7.1 % | 0.94 |
| w3 | 0.686 | 0.490 | 0.482 | 0.479 | +0.007 | +0.011 | 2.2 % | 0.94 |
| w4 | 0.676 | 0.478 | 0.456 | 0.451 | +0.022 | +0.027 | 5.7 % | 0.95 |
| **mean** | **0.68** | | | | **+0.021** | **+0.027** | **6.1 %** | |

- **Mild, stable covariate drift** (domain AUC ~0.68, flat) and a **small
  model-level concept gap: +0.027 PR-AUC corrected**, ~6 % relative degradation.
  GBT-on-TF-IDF parity check agrees (+0.029). ESS ~0.94 (correction well-posed).
- **But the interpretable probe is the loudest of the three**: on
  frequency-stable tokens, `P(toxic | comment contains token)` moves
  **mean \|Δ\| = 0.073** — nearly 10× fraud's probe number. Readable top movers
  (window 0 → last):

  | token | P(toxic) w0 → last |
  |---|---|
  | typical | 0.08 → 0.25 |
  | ill | 0.14 → 0.30 |
  | shame | 0.19 → 0.06 |
  | "voted for" | 0.10 → 0.23 |
  | corrupt | 0.28 → 0.16 |
  | hell | 0.18 → 0.29 |
  | christ | 0.01 → 0.12 |
  | "live in" | 0.05 → 0.16 |

  The *per-word* labeling function shifts substantially (annotation-guideline /
  norm drift on specific vocabulary) even though this mostly washes out at the
  whole-comment level. This is the clearest *visualizable* concept drift in the
  benchmark (`cd_bench/results/plots/civil_token_trajectories.png`).

### 5.3 `diabetes` — Confidence: **Low (proxy only)** — see §4.

---

## 6. Normalized cross-dataset comparison

PR-AUC base metric. "Corrected" = covariate-corrected stale-model gap.
(`cd_bench/results/comparison_table.csv`)

| dataset | temporal axis | covariate AUC (early→late) | raw concept gap | **corrected concept gap** | rel. degradation (corr.) | skill-norm. gap | conditional-probe \|Δ\| | confidence |
|---|---|---|---|---|---|---|---|---|
| **fraud** | real, ~182 d | 0.78 → 0.95 | +0.093 | **+0.096** | **19.0 %** | +0.099 | 0.008 (tabular bins) | Medium |
| **civil** | real, ~2016–17 | ≈0.68 (flat) | +0.021 | **+0.027** | 6.1 % | +0.030 | **0.073** (tokens) | Medium |
| diabetes *(proxy)* | **none — unvalidated** | 0.85 → 0.92 | +0.006 | +0.014 | 7.4 % | +0.015 | 0.010 | Low |

---

## 7. Verdict

### (A) Largest concept-drift magnitude → **`fraud`**, decisively.

Its covariate-corrected concept gap (**+0.096 PR-AUC**, **~19 % relative
degradation**) is **~3.5× `civil`'s** (+0.027, ~6 %) and an order of magnitude
above the `diabetes` proxy. The ranking is stable across every cut — raw gap,
corrected gap, skill-normalized gap, and each individual window — and it survives
covariate correction with healthy weight overlap. A fraud detector left unretrained
loses about a fifth of its precision-recall within six months purely because the
fraud/legit relationship moves. `diabetes` is **not rankable** (no time axis).

### (B) Most *notable* / defensible drift → **`fraud`**, with `civil` as the interpretability runner-up.

"Notable" = magnitude × measurability/interpretability.

- **`fraud`** wins: it has the magnitude *and* a real (if short) time axis *and*
  it survives the covariate correction *and* it degrades a deployed model in a
  way that is trivial to visualize (`fraud_overview.png`: `f_early` vs
  `f_current` PR-AUC pulling apart over time). Its one genuine weakness — you
  cannot name the drifting feature because the columns are anonymized — is a
  caveat, not a disqualifier.
- **`civil`** is the better *story* for interpretability: the
  `P(toxic | token)` trajectories are human-readable and clearly move
  (annotation-norm drift on specific words). But the whole-model magnitude is
  small (~6 % degradation), so as *notable concept drift* it ranks second.
- **`diabetes`** is excluded from both (A) and (B): with no per-record time
  field, temporal concept drift is undefined for it — a legitimate finding, not
  a gap in the analysis.

**Bottom line:** `fraud` has the strongest and most defensible concept drift;
`civil` has mild but uniquely interpretable concept drift; `diabetes` cannot be
assessed for temporal concept drift at all.
