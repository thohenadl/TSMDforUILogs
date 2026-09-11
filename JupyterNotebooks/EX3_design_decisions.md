# EX3 — Design Decisions

Comparison of **LoCoMotif** (multivariate, variable-length, parameter-laden) vs. the **Matrix-Profile family** (univariate, near-parameter-free) on four Leno UI logs.

This document is the single source of truth for the methodological choices in Experiment 3 (`00_EX3_LoCo_Mplot_Pan.ipynb`). It exists so that the paper revision can cite or paraphrase each justification without reconstructing the decision tree months later.

---

## 1. Why these three approaches

The reviewer (see `.claude/agents/knowledge/review.md`) raised two empirically actionable concerns:

- **Issue #1.** The prior draft claimed the Matrix Profile is "limited by a fixed window size *m*". This is contradicted by Matrix Profile XX (Madrid 2019) and by the Pan Matrix Profile (PMP) family, which scan all lengths.
- **Issue #2.** LoCoMotif was characterized as parameter-laden, slow, memory-intensive, and "bordering on plagiarizing" Mplots (Shahcheraghi 2024, Matrix Profile XXVI).

EX3 answers Issue #1 by including a Matrix Profile baseline that *does* scan multiple window sizes, and answers Issue #2 by giving LoCoMotif a like-for-like empirical comparison against the matrix-profile family on the same data, the same encoding, and the same length range.

The three compared approaches are:

| Algorithm | Variant | Input |
|---|---|---|
| **LoCoMotif** | `dtai-locomotif==0.1.0`, `apply_locomotif(rho=0.7)` | Multivariate, z-normalized Word2Vec matrix |
| **MStampBrute** *(round 2)* | `stumpy.mstump` evaluated at every integer `m ∈ [l_min, l_max]`, full-subspace row | **Multivariate** z-normalized Word2Vec matrix — same input as LoCoMotif |
| **Brute-force MP** | `stumpy.stump` evaluated at every integer `m ∈ [l_min, l_max]` | Univariate PCA(1) of standardized Word2Vec matrix |
| **Pan Matrix Profile (PMP)** | `stumpy.stimp` over `m ∈ [l_min, l_max]`, both `percentage=1.0` (full SKIMP) and `percentage=0.01` (sampled SKIMP) | Same univariate PCA(1) series |
| **Random** *(round 2)* | k segments, lengths Uniform[l_min, l_max], starts uniform on valid range, 100 seeds | n/a |

The brute-force MP sweep is the *naive* answer to Issue #1; PMP is the *principled* answer. Comparing them measures the cost of the SKIMP optimization (a story Keogh has personally pushed in the LinkedIn posts cited by the reviewer at [b], [c]).

**MStampBrute (added in round 2)** addresses the Keogh-style critique that the round-1 comparison was unfair: LoCoMotif consumed the full multivariate Word2Vec, while the MP family was fed PCA(1) which retains only 28–56% of the variance. `stumpy.mstump` is the multidimensional Matrix Profile (Yeh et al. 2017) and accepts the same `(d, n)` shape that LoCoMotif consumes, sidestepping the PCA bottleneck. With `MStampBrute` and `BruteMP` reported side by side, the gap between them quantifies how much performance PCA(1) destroys, and the gap between `MStampBrute` and LoCoMotif is the algorithm-attributable win on equal input.

**Random (added in round 2)** is the chance-level reference. With `k = 100` motifs in `T ≈ 4.6k–9.6k`, the ground-truth density is high enough that uniform-random k-segments can score non-trivial F1; reporting `F1 − F1_random` makes the chance-corrected gain visible and answers the standard reviewer question "is this better than guessing?".

## 2. Why Mplots are excluded empirically

By the Mplots paper's own definition (Shahcheraghi et al. 2024, Definition 4, p4 of `.claude/agents/knowledge/Introducing Mplots.pdf`), **an Mplot is a visualized matrix of distance profiles**. It is a data structure / visualization, not a motif-discovery algorithm. The same paper notes (p4): "the popular Matrix Profile is simply the vector of length `|A|` that contains the minimum (non-diagonal) value in each column" — i.e. the Matrix Profile is a feature extracted *from* the Mplot.

To make Mplots produce motifs we would have to *invent* an extraction procedure (e.g. threshold the matrix and take low cells with non-maximum suppression), measure that, and call the composite "Mplot motif discovery". That is not what the paper offers and not what the reviewer asks for. The reviewer cites Mplots ([d]) only as the precursor LoCoMotif allegedly resembles, never as an empirical baseline.

**PMP (`stumpy.stimp`) is the principled variable-length matrix-profile representative**, so it is the right comparand. Mplots are addressed in the paper with one or two sentences in the methods section that point readers to the data structure and explain why we did not benchmark it.

## 3. Shared inputs (fairness preconditions)

All three approaches must receive **identical preprocessing** for any speed or F1 gap to reflect the algorithm rather than the pipeline.

- **Encoding.** Word2Vec via `util/valmod_uihe.py::encode_word2vec` (attribute-as-word, row-as-sentence, log-as-corpus) with `vector_size = round(sqrt(unique_tokens_in_hierarchy_columns))`. This is the same encoding `experiment.py` already uses at line 242 with `encoding_method=1`. Same `gensim` random seed across runs.
- **Log scope.** The **full, unfiltered** UI log. No grammar-density or app-switch pruning. The grammar/app-switch filter from `experiment.py` is the *prior method under review*; using it as preprocessing for the comparand approaches would conflate methods.
- **Normalization.** Per-dimension z-normalization of the Word2Vec matrix before all downstream steps. `stumpy.stump` and `stumpy.stimp` internally z-normalize each subsequence as well — z-norming the full series first is harmless for them and necessary for LoCoMotif's distance metric.

## 4. PCA → 1D for the MP family, and its risk

`stumpy.stump` and `stumpy.stimp` operate on a 1-D series. The Word2Vec encoding is `d`-dimensional (`d = round(sqrt(unique_tokens))`, typically ≈ 10–20). We therefore reduce to 1-D via per-log `sklearn.decomposition.PCA(n_components=1)` fit on the z-normalized Word2Vec matrix.

The **explained variance ratio of the first component is reported per log** in `ex3_results.csv`. For UI-log Word2Vec embeddings this is typically modest (often well under 50%), and the discarded variance is signal the MP family does not see. This is a structural disadvantage of the MP-family inputs compared to LoCoMotif's multivariate input, and it is part of the comparison story — not something to hide.

We considered alternatives — most concretely using the grammar `rule_density_count` as the 1-D signal — and rejected them. The reviewer specifically criticized the grammar pipeline as having many moving parts (Issue #8); reusing its output as the MP input would carry that criticism into EX3. PCA → 1D is parameter-free given a fixed encoding.

## 5. Why oracle `[l_min, l_max]` and oracle `k = |GT|`

For each log we set `l_min = min(GT_motif_lengths)` and `l_max = max(GT_motif_lengths)`, and each algorithm is allowed to declare exactly `k = |GT|` motifs.

This is an **oracle setting**, deliberate, and disclosed as a limitation:

- It removes the *ranking-vs-termination confounder*. If the algorithms terminated on different criteria (LoCoMotif's `rho`, PMP's contrast elbow, MP's distance threshold), F1 gaps would partly reflect tuning of those criteria, not discovery quality. With shared oracle `k` and shared oracle `[l_min, l_max]`, every F1 number is a top-`|GT|` precision/recall on the same search range.
- It is therefore a **best-case bound**, not a deployment estimate. Deployed performance, where neither `[l_min, l_max]` nor `|GT|` is known, will be lower for all three.
- It is symmetric: no algorithm gets a tighter bound than another.

## 6. True Positive definition: IoU ≥ 0.8 with one-to-one matching

We use the definition from the paper's own Section 4.1 (Hohenadl 2025b, van Wesenbeeck 2026):

```
OR(α, β) = |α ∩ β| / |α ∪ β|        (Jaccard / IoU)
α and β are matchable iff OR(α, β) > 0.5
TP iff OR(α, β) ≥ 0.8
Lemma 1: each discovered segment is matchable with at most one ground-truth segment.
```

This is implemented as a new branch in `util/GrammarBasedUtil.py::evaluate_motifs(overlap_type="iou", overlap_threshold=0.8)`:

1. Build the `(M × G)` overlap matrix as before.
2. Compute `IoU[i, j] = overlap[i, j] / (motif_lengths[i] + gt_lengths[j] − overlap[i, j])`.
3. **Greedy one-to-one assignment by descending IoU.** Iterate candidate pairs from highest IoU downward; accept a pair iff neither its motif nor its GT has been claimed. Stop when the next pair's IoU < `overlap_threshold` or both sides are exhausted.
4. Accepted pairs with IoU ≥ `overlap_threshold` are TP. Unclaimed discovered motifs are FP. Unclaimed GT motifs are FN.

Greedy descending-IoU and Hungarian give identical assignments under Lemma 1 because each segment matches at most one GT — there is no cross-pair conflict to optimize globally. Greedy is simpler, faster, and easier to audit.

The same scorer is used unchanged for all three algorithms.

## 7. Brute-force MP aggregation across the `m` sweep

For each `m ∈ {l_min, l_min+1, …, l_max}`:

1. Compute `mp = stumpy.stump(ts_1d, m, normalize=True)`.
2. Extract `stumpy.motifs(ts_1d, mp[:, 0], max_motifs=|GT|, normalize=True)`. Each motif becomes one or more `(start, end)` candidates.
3. Tag each candidate with its (z-normalized Euclidean) distance.

After the sweep, all candidates from all `m` are pooled and reduced by **non-maximum suppression on IoU > 0.5** (sort ascending by distance; accept a candidate iff it has IoU < 0.5 with every already-accepted candidate). Keep the top `|GT|` survivors. This is the same NMS rule that PMP-style methods use to dedupe across scales and is symmetric with how PMP rows are reduced (§8). The threshold 0.5 is the conventional "matchable" cutoff (per the paper's Section 4.1).

The exclusion zone passed to `stumpy.motifs` is the default (`m/4`), which prevents trivial near-by matches inside each fixed-`m` profile; cross-`m` deduplication is then handled by the NMS step.

## 8. PMP knob: `percentage = 1.0` vs `0.01`

`stumpy.stimp(percentage=...)` controls SKIMP sampling. We report **both** settings:

- `percentage=1.0` (full SKIMP, exact). Apples-to-apples with the brute-force MP sweep — same range, same exactness — so any speed gap is the SKIMP scaffolding overhead, not the sampling shortcut.
- `percentage=0.01` (default sampled SKIMP). The principled, deployment-grade PMP setting that Keogh's tutorial cited in the review uses.

The F1/time delta between these two rows is the cost of the SKIMP optimization, and it is the central data point that addresses Issue #1.

**Motif extraction (round 2, current implementation).** After `pmp.update()` has run for every `m`, we iterate over each row of `pmp.PAN_` corresponding to `m ∈ M_`, trim each row to its valid length `n − m + 1` (since `PAN_` is padded to a uniform width), call `stumpy.motifs(ts_1d, P_row, max_motifs=|GT|, cutoff=∞, max_distance=∞)` per row, pool the resulting `(distance, (start, end))` candidates across every `m`, and apply NMS with IoU > 0.5 to select the top `|GT|` survivors. The `best_m` reported in `ex3_results.csv` is the modal segment length of the accepted set (descriptive only — no algorithmic role).

**Round-1 bug, now fixed.** The original implementation collapsed `pmp.PAN_` to a single `best_m` via a contrast heuristic and re-ran `stumpy.stump` at that one length. That discarded the pan information and made `PMP_full` and `PMP_sampled` produce identical F1 on every log — a smoking gun that surfaced in the Keogh-style round-2 review. The round-2 implementation matches the symmetry claim in §7 (brute-force MP also pools across `m` then NMS).

## 9. What we time

Two timing scopes are recorded per (log, algorithm):

- **Preprocessing time.** `read_data_for_processing` + `encode_word2vec` + z-normalization + (PCA(1) where applicable). Shared between brute-force MP and PMP, separate from LoCoMotif because LoCoMotif skips PCA. Reported for completeness.
- **Discovery time.** Wall-clock around the algorithm call only (`apply_locomotif`; the `stump` sweep; the `stimp` build + extraction). This is the headline number.

`time.perf_counter()` is used. Each algorithm is run once per log (no repeated-trial averaging), which is acceptable because (a) the log lengths are large enough that single-run wall times dominate scheduling noise, and (b) the algorithms are deterministic given the same input.

## 10. Threats to validity

To be disclosed in the paper's limitations or threats-to-validity subsection.

- **Oracle search range and budget.** `l_min`/`l_max`/`k` are taken from ground truth. Results are upper bounds on deployed F1.
- **PCA(1) information loss.** The MP family is fed approximately the first principal component of the Word2Vec matrix; the explained variance ratio is reported per log so the magnitude of this loss is visible.
- **Greedy vs Hungarian matching.** Under Lemma 1 the two are equivalent; we use greedy for simplicity.
- **Word2Vec stochasticity.** A single random seed is used per log. Word2Vec is sensitive to initialization for small corpora; multi-seed averaging is a logical extension but is out of scope for this comparison.
- **Single log family.** Four logs all drawn from the Leno SmartRPA family (SR_RT_plus and SR_RT_parallel and their `_extended` variants). This is a case study on one RPA-tool family — we make no claim of generalization to other families.
- **Mplots not measured.** Justified in §2.

## 11. Random baseline (round 2)

Per-log procedure for each of `S = 100` independent seeds:

1. Sample `k = |GT|` integer lengths `m_i ~ Uniform{l_min, …, l_max}`.
2. For each `m_i`, sample an integer start `s_i ~ Uniform{0, …, T − m_i}`.
3. The list `[(s_i, s_i + m_i − 1)]` is the discovered set for this seed.
4. Score with the same `evaluate_motifs(overlap_type="iou", overlap_threshold=0.8)` scorer used for the algorithmic methods.

Report **mean and std of F1, precision, recall** across the 100 seeds. The mean populates the `f1` / `precision` / `recall` columns; std lands in `f1_std` / `precision_std` / `recall_std` so the chance-corrected gap `F1_algorithm − F1_random` can be read directly from the CSV.

**Why this matters.** GT density in these logs is high (`k = 100` motifs in `T ≈ 4.6k–9.6k`), so chance-level F1 is not zero. Without a random reference, a BruteMP F1 of 0.07 looks bad but ambiguous; with one, we can see whether it is statistically distinguishable from random under the same `k` and length budget. This is the standard sanity check that the AE-cited literature (Madrid 2019, Shahcheraghi 2024) presupposes and that any KDD reviewer expects.

The random baseline uses oracle `k` and oracle `[l_min, l_max]` like every other method — it answers "is your F1 better than guessing within the same budget?", not "is your F1 better than guessing with no information at all?". The latter would set k and length range from a prior, which is out of scope here.
