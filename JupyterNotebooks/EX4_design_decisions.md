# EX4 — Design Decisions

Replacement of the **grammar-gated** discovery pipeline in `experiment.py::run_experiment` with a **parameter-free, ranked** pipeline (`run_experiment_ranked`) that runs both side-by-side for A/B comparison.

This document is the single source of truth for the methodological choices in Experiment 4. It exists so that the paper revision can cite or paraphrase each justification without reconstructing the decision tree months later.

---

## 1. Why this experiment exists

`run_experiment` (the EX3-era baseline, still in `experiment.py`) is a serial, gated pipeline:

1. Re-Pair grammar over the full log → per-event `rule_density_count`.
2. `find_max_density_groups(..., method="percentile", percentile_threshold=0.90)` selects "grammar cores" — the top 10% densest regions.
3. `app_switch_miner` + `similar_path_up_down` + `safety_margin_factor=2` extend cores into `valid_indices`.
4. The log is **subset** to `valid_indices` before encoding.
5. LoCoMotif runs on the reduced, encoded log.
6. `extend_motifs_anchor_logic(..., filter_only_matches=True)` (`GrammarBasedUtil.py:822-825`) discards any LoCoMotif motif that does not overlap a grammar core.

Five tuned numerical knobs sit on the discovery path: `rule_density_threshold=0.8`, `percentile_threshold=0.90`, `app_switch_similarity_threshold=0.8`, `safety_margin_factor=2`, and `overlap_threshold=0.8`. None have theoretical grounding in the EX3 design notes; all are empirically defaulted.

**The failure mode this introduces.** Sub-90th-percentile density peaks are silently dropped before LoCoMotif ever sees the events that contain them. The anchor-logic post-filter then drops a second time any LoCoMotif motif that did not happen to overlap one of the surviving cores. A real RPA pattern that produces only a mild grammar peak — for example, a routine of 8 events that repeats only 3 times in a 25k log — is invisible by construction.

**EX4 answers this concern with two principles.**

- **No discovery-path hyperparameters.** Every numeric choice must be one of: an algorithm's published default, a domain constant with a one-sentence justification, or a function of the data itself.
- **Ranking, not gating.** Stream 1 (LoCoMotif) discovers candidate motifs from the full log. Stream 2 (raw grammar density) supplies a per-motif score. The output is a ranked list, evaluated by AUPRC. Selection is the reader's, the deployer's, or the metric's — never the researcher's.

The two pipelines coexist in `experiment.py` so the paper can present a like-for-like comparison on the same logs and same encoding.

## 2. Two-stream architecture

```
read_data_for_processing
        │
        ▼
re_pair(log)                                      ── Stream 2 source ──
        │
        ▼
generate_density_count(encoding_df, log)
        │   per-event rule_density_count (RAW; no percentile/threshold cut)
        │
        ▼
encode_word2vec(log, …, vector_size=round(sqrt(tokens)))
        │   multivariate time series, shape (T, n)
        ▼
z-normalize (per dimension)                                            ── Stream 1 source ──
        │
        ▼
if T > T_MAX_SKIP (=50_000):  skip log, mark out-of-scope
otherwise:
        ▼
apply_locomotif(ts, l_min=5, l_max=T//2, rho=0.8)
        │
        ▼   motif_sets : list of (representative, [motif ranges])
flatten motif_sets → list of (range, cluster_id, locomotif_rank)
        │
        ▼
for each motif range [s, e]:
    grammar_score    = mean(rule_density_count[s : e+1])
    top_grammar_rule = argmax rule∈encoding_df  |decoded_expansion(rule) ∩ [s,e]|
        │
        ▼
sort motifs by grammar_score DESC  (tie-break: locomotif_rank ASC)
        │
        ▼
evaluation:
    for k in 1..N:
        (P_k, R_k) = evaluate_motifs(top-k ranges, ground_truth,
                                     overlap_type="ratio", overlap_threshold=0.8)
    AUPRC = sklearn.metrics.average_precision_score-equivalent over (R_k, P_k)
        │
        ▼
output: ranked motif DataFrame  +  AUPRC  +  PR-curve points
```

**Stream 1** owns *discovery*: LoCoMotif on the full encoded log, no pruning. **Stream 2** owns *scoring*: the raw grammar density signal `rule_density_count` from `generate_density_count`, used to rank Stream 1's output. The two streams share no thresholds and no parameters; they meet exactly once at the per-motif `grammar_score = mean(density[s:e+1])` step.

## 3. Locked design decisions

| ID | Decision | Rationale |
|---|---|---|
| **D1** | **Combination = continuous score + ranking.** Stream 2 emits the raw `rule_density_count` per event; each Stream 1 motif is scored by the mean density over its range. | Eliminates the 0.8/0.9 percentile cuts from the discovery path. Sub-peak motifs surface in the ranking if LoCoMotif is confident in them. |
| **D2** | **Evaluation metric = AUPRC** over the ranked list. | No threshold required to compute it. Standard in detection literature. Comparable across runs and across logs. |
| **D3** | **Discovery algorithm stays as LoCoMotif.** | User-run benchmark: 48s for LoCoMotif vs 52 min for both `stumpy.mstump` and `stumpy.stimp` on a 9.6k-event log. LoCoMotif's variable-length multi-variate niche is real. Replacing it sacrifices the niche this paper relies on; out of scope for EX4. |
| **D4** | **No pre-filter on Stream 1 input.** The full encoded log is z-normalized and passed to LoCoMotif. | Removes the "small peak missed" failure mode by construction. The grammar pre-filter was a means to a smaller `T`; LoCoMotif now sees every event. |
| **D5** | **LoCoMotif parameters:** `l_min=5`, `l_max=T//2`, `rho=0.8`, z-normalize input. | `l_min=5`: RPA-domain floor (a meaningful UI routine is at least 5 events; sub-5 sequences capture mouse jitter and scroll deltas). `l_max=T//2`: definitional ceiling — a motif requires ≥2 non-overlapping occurrences, so its length cannot exceed T/2. Replaces the prior `safety_margin_factor × pattern_switch_distance`, which was arbitrary by the user's own assessment. `rho=0.8`: LoCoMotif's published default when `warping=True` (`locomotif.py:39`). Z-normalization: explicitly recommended by the LoCoMotif package itself (it emits a `UserWarning` if you skip it). |
| **D6** | **Scope cap: `T_MAX_SKIP = 50_000`.** Logs with `len(log) > 50_000` are skipped and recorded as `skipped=True`. | LoCoMotif is O(T²·n) in time and memory; the similarity matrix alone is ~16·T² bytes. At T=50k with n≈15 this is ~40 GB, the edge of an M5 Pro's 36 GB unified memory. Above 50k the algorithm OOMs on consumer hardware. The 50k cap covers ~88% of the 648-log validation set (median log length = 1.5k, p90 = 37.5k). The cap is a hardware fact, not a tuned knob. |
| **D7** | **App-switch logic deleted from the discovery path.** `app_switch_miner`, `similar_path_up_down`, `extend_motifs_anchor_logic`, `merge_final_overlaps`, and `find_max_density_groups` are not called by the new pipeline. They remain in `GrammarBasedUtil.py` untouched, for the baseline `run_experiment` and any other caller. | These functions existed solely to expand grammar cores into a reduced log. With grammar cores gone (D4), they have no remaining purpose. The user reviewed and confirmed: the app-switch boundaries themselves are data-driven (parameter-free), but the `similarity_threshold=0.8` and `safety_margin_factor=2` layered on top are not — and once the cores disappear, the boundaries have nothing to expand. |
| **D8** | **Each output motif carries its top-overlapping grammar rule symbol** as an interpretability anchor. | Connects Stream 1 (LoCoMotif's discovery) to Stream 2 (the grammar) at the output layer. Lets the paper write qualitative examples — "Motif #3 is an instance of grammar rule `R47`, which decodes to `[open_outlook, click_compose, type_subject, ...]`" — without re-deriving the link. Zero hyperparameters. |
| **D9** | **Encoding unchanged.** Word2Vec with `vector_size = round(sqrt(unique_tokens))`, `encoding_method=1` (attribute-as-word, row-as-sentence, log-as-corpus). | Already auto-scales with the log; not a tuned knob the user chose. Identical to EX3 §3, ensuring like-for-like input across pipelines. |
| **D10** | **A/B coexistence, not replacement.** The new function lives alongside `run_experiment`, not in place of it. Output goes to a separate CSV. | Allows side-by-side comparison on the same validation set without branch-switching. The paper's headline table is `(baseline F1, baseline P, baseline R, ranked AUPRC, ranked F1@k=|GT|)` joined on `uiLogName`. |

## 4. Why ranking forces AUPRC (and what AUPRC buys)

The natural objection — raised during the design discussion — is that `evaluate_motifs` expects a *set* of motif ranges and produces TP/FP/FN. If we hand it 1000 ranked candidates against ~20 ground-truth motifs, precision collapses to ~0.02 regardless of ranking quality.

The resolution is that **F1 forces a cut somewhere**. The research choice is whether that cut is a hyperparameter (bad — moving part) or a property of the data (acceptable — no tuning).

AUPRC sidesteps the cut entirely by integrating over every k:

```
for k in 1..N:
    top_k_ranges = ranked_motifs.iloc[:k]
    stats        = evaluate_motifs(top_k_ranges, ground_truth,
                                   overlap_type="ratio", overlap_threshold=0.8)
    P_k = stats["tp"] / (stats["tp"] + stats["fp"])
    R_k = stats["tp"] / (stats["tp"] + stats["fn"])

AUPRC = ∫ P dR   (computed via sklearn.metrics.average_precision_score-equivalent)
```

This is **one summary number, no hyperparameter, comparable across runs**. The PR curve itself is also persisted for diagnostic plots in the paper.

For the headline comparison table, we additionally report **F1 @ k = |GT|** as an "oracle ceiling" companion number: it tells the reader the best F1 the ranking is capable of without choosing a deployment threshold. Both numbers come from the same `evaluate_motifs` calls — F1@k=|GT| is just one of the points already computed for the curve.

## 5. Stream 2 score: raw density, not normalized

`grammar_score = mean(rule_density_count[s : e+1])` uses the raw, unscaled density counts emitted by `generate_density_count` (GrammarBasedUtil.py:192). No normalization, no percentile transform, no smoothing.

Rationale: any normalization step (per-log, per-region, per-rule) introduces a choice. Mean of raw counts is the simplest aggregator; it preserves the magnitude information that distinguishes "this region is covered by many high-frequency rules" from "this region is covered by one low-frequency rule". Ranking by this raw quantity is monotone-invariant under any per-log rescaling, so per-log normalization would not change the ranking anyway — only the absolute scores.

Ties (multiple motifs with identical grammar scores) are broken by **LoCoMotif's own discovery order**, which `find_best_motif_sets` (`locomotif.py:143`) returns sorted by internal fitness. This piggybacks on a published ranking rather than inventing one.

## 6. The `l_max = T // 2` argument in detail

The single non-obvious parameter choice is `l_max`. The justification is **definitional**, not heuristic.

> **Claim.** A subsequence of length `L` cannot occur as a motif in a series of length `T` if `L > T // 2`.
>
> **Proof.** A motif is, by every definition in the matrix-profile and warping-motif literature (Yeh et al. 2016 §III; Madrid et al. 2019 §3; Van Wesenbeeck et al. 2024 §2.1), a pattern that occurs **at least twice**. Two non-overlapping copies of length `L` require `2L ≤ T`, i.e. `L ≤ T // 2`. Anything longer cannot occur twice and therefore cannot be a motif. ∎

The PMP literature (Madrid et al. 2019; Imamura et al. 2020) typically uses `T // 4` or `T // 5` as a tighter "interesting motif" cap (≥4 or ≥5 non-overlapping copies). That is a *secondary* choice with empirical motivation; `T // 2` is the *only* one that follows from the definition of a motif itself. EX4 uses `T // 2` because it is the largest value defensible without empirical claims.

A sensitivity sweep over `l_max ∈ {T/2, T/4, T/10, T/20}` is logged in §10 as a deferred robustness check; the headline number uses `T // 2`.

## 7. The `T_MAX_SKIP = 50_000` argument in detail

LoCoMotif's `get_locomotif_instance` (`locomotif.py:51-60`) builds a dense `(T, T)` similarity matrix `_sm` of dtype `float32`:

```
peak memory ≈ 4 × T² bytes  (just for _sm)
           + 4 × T² bytes  (cumulative similarity DP)
           + O(T·n) bytes  (the encoded series itself)
```

| T | _sm + DP memory | Feasible on 36 GB M5 Pro? |
|---|---|---|
| 10 000 | 0.8 GB | trivial |
| 30 000 | 7.2 GB | comfortable |
| 50 000 | 20 GB | feasible |
| 75 000 | 45 GB | OOM |
| 100 000 | 80 GB | OOM |

The 50k cap is **the largest T for which `_sm + DP` fits in a typical M5 Pro's unified memory with headroom for the Word2Vec matrix, the grammar tables, and Python overhead**. It is a hardware fact, not a tuned hyperparameter.

The empirical distribution of the validation set's log lengths supports this scope:

| Bucket | Count | % of corpus |
|---|---|---|
| < 10k | 510 | 78.7% |
| 10k–30k | 60 | 9.3% |
| 30k–50k | 21 | 3.2% |
| 50k–100k | 30 | 4.6% |
| 100k–250k | 21 | 3.2% |
| ≥ 250k | 6 | 0.9% |

In-scope: 591 of 648 logs (91.2%). Out-of-scope and skipped: 57 logs (8.8%). The skipped logs are documented in the paper's threats-to-validity section, not removed silently. Future work — algorithmic acceleration of LoCoMotif (banded similarity matrix; GPU implementation) — is the appropriate route to lift this cap, and that work is out of scope for EX4.

## 8. What stays from the baseline pipeline

For fairness preconditions and to avoid spurious differences between the two pipelines:

- **Encoding.** Identical to baseline §EX3 — `encode_word2vec(log, orderedColumnsList=hierarchy_columns, vector_size=round(sqrt(tokens)))`. Same gensim random seed.
- **Log loading.** `read_data_for_processing(isSmartRPA2025=True, log_name_smartRPA=...)`, unchanged.
- **Hierarchy columns filter.** Same drop-zero-unique-values rule as `run_experiment`.
- **Evaluator.** `grammar_util.evaluate_motifs(overlap_type="ratio", overlap_threshold=0.8)` — same scorer, same overlap definition, same one-to-one matching. Reused unmodified. Note: ratio (not IoU) is the baseline's evaluator; switching to IoU would conflate the discovery change with an evaluator change. IoU sensitivity is a deferred ablation.

## 9. What we time

Two timing scopes, identical convention to EX3 §9:

- **Preprocessing time.** `read_data_for_processing` + `encode_word2vec` + `re_pair` + `generate_density_count` + z-normalization.
- **Discovery time.** Wall-clock around `apply_locomotif` only. This is the headline number for any scalability discussion.

`time.perf_counter()`. One run per log; LoCoMotif and the grammar pipeline are deterministic given fixed Word2Vec seed and input.

## 10. Threats to validity

To be disclosed in the paper's limitations subsection.

- **Scope cap at T = 50 000.** 8.8% of the validation set is unmeasured. The headline AUPRC is the mean over the 591 in-scope logs; the skipped logs are listed explicitly. We do not claim the ranked pipeline scales beyond 50k.
- **`l_max = T // 2` is the maximal-defensible setting, not necessarily the empirically-best one.** A sweep over `l_max ∈ {T/2, T/4, T/10, T/20}` should be reported in the appendix if reviewers ask. The headline uses `T // 2` because it is the only theoretically-grounded choice.
- **`l_min = 5` is a domain choice.** Sensitivity to `l_min ∈ {3, 5, 7, 10}` is a deferred ablation. The paper justifies 5 with one sentence ("the smallest sequence that can encode a meaningful UI routine"); reviewers may push for a sweep.
- **Stream 2 = mean density is the simplest aggregator.** Alternatives — median, sum, mean-of-log, percentile-based — are not ablated. Mean is reported because it is the most-defensible default.
- **Word2Vec stochasticity.** A single random seed is used per log. Identical to EX3 §10 caveat.
- **Single algorithm family (LoCoMotif).** The benchmark in D3 supports staying with LoCoMotif on the user's data, but EX4 makes no claim that a parameter-free pipeline could not be built around a different multi-variate variable-length discoverer. Algorithmic acceleration / replacement is future work.
- **Evaluator: ratio, not IoU.** Matches the baseline so the two pipelines' numbers are directly comparable. The EX3 design notes (§6) prefer IoU on theoretical grounds; an IoU re-evaluation is a one-line change in `compute_auprc` and should be added as an appendix table.
- **Grammar rule attachment (D8) is descriptive, not validated.** The top-overlapping grammar rule per motif is reported for qualitative inspection. We do not claim "the rule explains the motif" — that would require a separate evaluation against ground-truth labels.

## 11. Implementation surface

### New code

**`util/GrammarBasedUtil.py`** — two new functions, additive only:

- `compute_auprc(ranked_motif_ranges, ground_truth, overlap_threshold=0.8) -> dict`
  Loops `k = 1 .. len(ranked_motif_ranges)`, calls existing `evaluate_motifs(overlap_type="ratio", overlap_threshold=overlap_threshold)` on the top-k slice, accumulates `(P_k, R_k)`, computes AUPRC. Returns `{"auprc", "pr_curve", "f1_at_k_gt"}`.
- `attach_top_grammar_rule(motif_ranges, encoding_df) -> list[str]`
  For each motif range, finds the grammar rule symbol whose decoded terminal expansion (`re_pair_decode_all(encoding_df)` — already exists at `GrammarBasedUtil.py:194`) has the largest index-overlap with the range. Returns the rule symbol per motif.

**`JupyterNotebooks/experiment.py`** — two new functions, additive only:

- `run_experiment_ranked(log_name_smartRPA: str, t_max_skip: int = 50_000) -> tuple[pd.DataFrame, pd.DataFrame]`
  Implements the architecture in §2. Single optional numeric argument (the scope cap). Returns `(summary_df, motifs_df)` where:
    - `summary_df` has one row: `uiLogName, log_length, n_motifs_discovered, auprc, f1_at_k_gt, runtime_total_s, runtime_discovery_s, skipped, skip_reason`.
    - `motifs_df` has one row per discovered motif: `uiLogName, motif_rank, range_start, range_end, length, cluster_id, locomotif_rank, grammar_score, top_grammar_rule`.
- `experiment_ranked(target_filename: str, t_max_skip: int = 50_000)`
  Driver mirroring `experiment(...)`. Iterates `validationLogInformation.csv` (`logs/smartRPA/202511-update/`), calls `run_experiment_ranked` per log, writes `summary_df` rows incrementally to `logs/smartRPA/202511-results/<target_filename>` and `motifs_df` rows to `<target_filename>_motifs.csv` (split so the headline CSV stays narrow). Same skip-already-processed and skip-too-large logic as `experiment(...)`.

### Untouched

- `run_experiment`, `experiment`, `run_variance_experiment`, `variance_experiment` in `experiment.py`.
- `app_switch_miner`, `similar_path_up_down`, `extend_motifs_anchor_logic`, `merge_final_overlaps`, `find_max_density_groups` in `GrammarBasedUtil.py`.
- `evaluate_motifs` (reused by `compute_auprc` unchanged).
- Encoding / log-loading utilities.

### New runner

- `JupyterNotebooks/ex4_ranked_runner.py` — minimal script that imports `experiment_ranked` and calls it with `target_filename="ex4_ranked_results.csv"`. Mirrors `ex3_runner.py`.

## 12. Verification

1. **Smoke test on 5 small logs (<5k events).** Pick five `log_motifs5_occurances5_length10_*` logs. Run both `run_experiment` and `run_experiment_ranked`. Expected: `run_experiment_ranked` AUPRC > 0.8, top-3 grammar-rule attachments human-recognizable as planted patterns.
2. **Medium-log runtime check.** Three logs in 10k–30k. Wall-clock < 5 min each. Peak RSS < 12 GB. If exceeded, the `_sm` memory model in §7 needs revisiting before continuing.
3. **Boundary check at ~45k events.** Largest in-scope log. Memory < 32 GB, runtime < 30 min. If OOM, lower `T_MAX_SKIP` to the largest log that succeeded and amend §7's table.
4. **A/B comparison on the validation set.** Run `experiment_ranked` over the 591 in-scope logs. Join `ex4_ranked_results.csv` with the existing baseline result CSV on `uiLogName`. Headline table for the paper: `(baseline F1, baseline P, baseline R, ranked AUPRC, ranked F1@k=|GT|)` per log, with means at the bottom.
5. **Skipped-log accounting.** 57 logs > 50k must appear in `ex4_ranked_results.csv` with `skipped=True`, `auprc=NaN`, `skip_reason="log_length_exceeds_t_max_skip"`. Iteration must not error on these.
6. **Reproducibility.** Re-run the smoke test with a fixed gensim seed. AUPRC variance < 0.02 across re-runs. If higher, deterministic hierarchical encoding (`encoding_method=2`) is the fallback — but the paper would then need to defend the encoding switch.
7. **Sanity vs random.** Following EX3 §11's pattern, a random baseline can be added by sampling `k = N_motifs_discovered` segments uniformly over `[0, T]` with lengths Uniform[`l_min`, `l_max`], scoring with the same evaluator, and reporting `AUPRC_ranked − AUPRC_random`. Strongly recommended for the paper's reviewer-defensibility but deferred to a follow-up cell, not blocking for v1.

## 13. Open items deferred to implementation

- Per-log loop parallelization in `experiment_ranked(...)` across CPU cores. Trivially safe (each log is independent), large wall-clock win across 591 logs, orthogonal to the architectural redesign.
- AUPRC integration scheme: trapezoid vs `sklearn.metrics.average_precision_score` (step-function area). Default to `average_precision_score` for one-line correctness; reconsider only if the curve has degenerate ties.
- Whether to also emit a per-cluster (LoCoMotif representative) summary alongside the flat ranked list. `motif_sets` returns cluster structure that could be useful for downstream RPA tooling but adds output complexity.
- IoU-based evaluator (`overlap_type="iou"`, threshold 0.8) re-run as an appendix table — see §10.
- `l_max` and `l_min` sensitivity sweeps as appendix tables — see §10.
