# AI Exposure and BLS Employment Projections

Does AI exposure predict how the Bureau of Labor Statistics revised its occupational
employment projections after generative AI arrived?

This repo builds a single occupation-level dataset that joins two BLS projection
vintages — one finalized before ChatGPT, one after — to four independent measures of
AI exposure and the full O\*NET task inventory, then clusters occupations into
AI-impact archetypes using k-means and Gaussian mixture models.

**Headline finding: the effect is real in the raw data but largely dissolves under a
basic control.** Details in [Results](#results).

---

## Repo contents

| File | Purpose |
|---|---|
| `build_ai_jobs_dataset.py` | Builds the combined dataset. Downloads/caches every source, normalizes codes, joins to one occupation-level panel plus an occupation×task long file. |
| `cluster_four_methods.py` | Four unsupervised families compared (k-means, GMM, Ward hierarchical, DBSCAN) + PCA, with cross-method agreement via Adjusted Rand Index. Excludes the outcome from features by default. |
| `requirements.txt` | Dependencies. |

## Quick start

```bash
pip install -r requirements.txt
python build_ai_jobs_dataset.py     # -> out/occupation_ai_panel.csv
python cluster_four_methods.py      # -> out/clusters_four_methods.csv
```

### Repo layout

```
build_ai_jobs_dataset.py          builds the combined dataset
cluster_four_methods.py           the four unsupervised models
requirements.txt
README.md
raw/
  occupation_2021-31.xlsx         committed - bls.gov blocks automated download
  occupation_2024-34.xlsx         committed - bls.gov blocks automated download
  aei_job_exposure.csv            committed - not reliably auto-fetchable
  onet_task_statements.xlsx       committed - required, and pins the O*NET version
out/                              generated; commit if you want results tracked
```

**Four input files are committed** so a fresh clone runs anywhere. The two BLS
workbooks because `bls.gov` blocks datacenter IP ranges; `aei_job_exposure.csv`
because it is not reliably auto-fetchable; and the O\*NET Task Statements because it
is a hard requirement and committing it pins the database version (O\*NET download
URLs are version-numbered, so an un-pinned run can silently pull a different release).
Total ~1.9 MB. BLS files are public-domain US government works; O\*NET is CC BY 4.0
(credited above); AEI is openly published.

Everything else — Felten AIOE, Eloundou occupation and task files, O\*NET Task
Ratings, AEI task usage — is fetched automatically on first run and `.gitignore`d.

Downloads cache in `raw/`. If a source is unreachable, the build warns and continues —
only the two BLS vintages and O\*NET task statements are strictly required.

`python build_ai_jobs_dataset.py --manual` prints a manual download list with target
filenames, for environments where `bls.gov` blocks automated requests (it blocks
datacenter IP ranges, so this is common on cloud notebooks).

---

## Data sources

### Employment projections

| Source | Contributes | Native key |
|---|---|---|
| [BLS Employment Projections archive](https://www.bls.gov/emp/data/projections-archive.htm), 2021–31 vintage | **Pre-AI** employment + projected 10-yr growth (Table 1.2) | 2018 SOC |
| [BLS Employment Projections](https://www.bls.gov/emp/ind-occ-matrix/occupation.xlsx), 2024–34 cycle | **Post-AI** employment, growth, openings, wage, education | 2018 SOC |

The 2021–31 projections were finalized in September 2022 — the last full cycle
developed before ChatGPT's release, and the cleanest available pre-generative-AI
baseline. The 2019–29 cycle is an alternative (`--pre 2019-29`) but its base year sits
mid-pandemic. The 2024–34 cycle was released August 28, 2025.

BLS began explicitly assessing AI impacts starting with the 2023–33 cycle; see
Machovec, Rieley & Rolen, "Incorporating AI impacts in BLS employment projections:
occupational case studies," *Monthly Labor Review*, February 2025.

### AI exposure measures

| Source | Measure | Native key | Join method |
|---|---|---|---|
| [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex) — `job_exposure.csv` | `observed_exposure`: **revealed** exposure from real Claude conversations mapped to occupations | 2018 SOC | direct |
| [Eloundou et al., "GPTs are GPTs"](https://github.com/openai/GPTs-are-GPTs) — `data/occ_level.csv` | Rubric-based LLM exposure. `alpha` = direct LLM exposure; `beta` = incl. LLM-powered software; `gamma` = broadest. Separate `human_rating_*` and `dv_rating_*` (GPT-4-labelled) variants | O\*NET-SOC | first 7 chars = 2018 SOC |
| Same repo — `data/full_labelset.tsv` | Task-level exposure labels, aggregated to occupation means (`gpts_task_*_mean`) | task text | normalized text match |
| [Felten, Raj & Seamans AIOE](https://github.com/AIOE-Data/AIOE) | Ability-based AI exposure index (pre-LLM, measures AI-in-general) | **2010 SOC** | [BLS 2010→2018 crosswalk](https://www.bls.gov/soc/2018/) |
| [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex) — `task_pct` | Share of Claude conversations per O\*NET task, summed to occupation | task text | normalized text match |

Having four measures from genuinely different methodologies is the point — see the
convergent-validity result below.

### Task data

| Source | Contributes | Join |
|---|---|---|
| [O\*NET Database](https://www.onetcenter.org/database.html#individual-files) — Task Statements | Every task per occupation, task type (Core/Supplemental) | O\*NET-SOC prefix → 2018 SOC |
| O\*NET Database — Task Ratings | Task importance ratings (scale `IM`) | Task ID |

**Why the prefix works:** O\*NET-SOC 2019 codes are 2018 SOC codes plus a `.XX` detail
suffix (`15-1252.00` → `15-1252`). Where one SOC maps to several O\*NET occupations,
scores are averaged and task counts are pooled under the parent SOC.

---

## Outputs

| File | Shape | Description |
|---|---|---|
| `out/occupation_ai_panel.csv` | ~832 × 28 | One row per 2018 SOC occupation. All sources joined. |
| `out/occupation_task_long.csv` | ~18,800 × 42 | One row per occupation × O\*NET task, occupation columns broadcast onto every task row. |
| `out/clusters_four_methods.csv` | ~831 × 36 | The panel plus labels from all four models, GMM membership probabilities, per-occupation silhouette, DBSCAN noise flag, PCA coordinates. **The analysis file.** |
| `out/cluster_profiles_four_methods.csv` | k × features | Z-scored feature means per cluster, for naming archetypes. |

### Data dictionary — `final_ai_jobs_dataset.csv`

**Identifiers**
- `soc_code`, `occ_title` — 2018 SOC code and title
- `soc_major` — first two digits (SOC major group)
- `bls_merge` — whether the occupation appears in both vintages or only one

**Employment projections**
- `emp_2021`, `emp_2031`, `pct_chg_2021_2031` — pre-AI vintage (thousands; % 10-yr growth)
- `emp_2024`, `emp_2034`, `pct_chg_2024_2034` — post-AI vintage
- `openings_2021_2031`, `openings_2024_2034` — annual average occupational openings
- `median_wage_2024`, `education_2024` — context from the post vintage

**Dependent variables**
- `revision_pp` — **the core variable.** Post-vintage growth minus pre-vintage growth,
  in percentage points. Negative = BLS now projects slower growth than before generative AI.
- `revision_resid` — `revision_pp` residualized on `pct_chg_2021_2031`. Removes mean
  reversion: fast-projected-growth occupations get revised down almost mechanically.
  In this data that regression has R²=0.39, so **39% of the raw revision is mean
  reversion, not AI.**
- `pandemic_sensitive` — SOC majors 35 (food service), 39 (personal care/entertainment),
  27 (arts/media). Their 2021 base-year employment was pandemic-depressed, so the
  2021–31 vintage projected large phantom "recovery" growth that later vintages undid.
  These 79 occupations average **−8.3pp** revision vs. **−1.9pp** for everything else.

**AI exposure**
- `aei_observed_exposure` — Anthropic Economic Index (revealed usage)
- `gpts_human_rating_{alpha,beta,gamma}` — Eloundou, human raters
- `gpts_dv_rating_{alpha,beta,gamma}` — Eloundou, GPT-4 labels
- `gpts_task_{alpha,beta,gamma}_mean` — Eloundou task labels averaged per occupation
- `aioe_felten` — Felten/Raj/Seamans AIOE
- `aei_claude_usage_share` — summed Claude-usage share across the occupation's tasks

**Task structure**
- `n_onet_tasks`, `mean_task_importance`

**Clustering**
- `kmeans_cluster`, `kmeans_silhouette` — hard label; per-occupation silhouette
  (negative = closer to another cluster than its own)
- `gmm_cluster`, `gmm_max_prob`, `gmm_entropy`, `gmm_prob_0..n` — soft membership.
  Low `gmm_max_prob` flags occupations between archetypes.
- `pca1`, `pca2` — first two principal components, for plotting
- `feat_*` — the exact standardized inputs to the models, for reproducibility

---

## What is and isn't being predicted

Unsupervised learning has **no target variable**. The clustering does not predict
`revision_pp`; it finds structure in occupation-space. Two separate questions:

| | Question | Method | Target |
|---|---|---|---|
| Unsupervised | What natural groupings exist among occupations? | k-means, GMM, Ward, DBSCAN | none |
| Inferential | Does AI exposure predict downgraded projections? | Spearman / regression | `revision_pp` |

The clusters are descriptive archetypes. The hypothesis test carries the AI claim.

**Circularity warning.** If `revision_pp` is in the feature matrix, clusters are partly
*defined* by the outcome — fine for descriptive strata, invalid for claiming clusters
differ in revisions. `cluster_four_methods.py` excludes it by default; pass
`--include-outcome` to see the circular version. See [Results](#results) for why this
matters enormously.

## Method

**Feature matrix.** AI exposure (AEI + Eloundou beta), log Claude usage, pre-AI
projected growth, `revision_pp`, log employment, log wage, task count, mean task
importance. Standardized; rows needing >40% imputation dropped, remainder median-imputed.

**One deliberate exclusion.** `pct_chg_2024_2034 = pct_chg_2021_2031 + revision_pp`
exactly, so the three span only two dimensions. Including all three in a
distance-based clusterer silently double-weights growth against exposure. The post
column is dropped.

**Four unsupervised families**, chosen because each carries a different assumption:

| Model | Family | Assumes | Selection |
|---|---|---|---|
| K-means | centroid | spherical, equal-size, all points assigned | silhouette |
| GMM | probabilistic | elliptical, overlapping, soft membership | BIC |
| Ward | hierarchical | nested structure | silhouette + merge-height gap |
| DBSCAN | density | dense regions separated by sparse ones; **can label noise** | k-NN knee sweep |

PCA is also reported (a fifth family, dimensionality reduction) for the 2-D view and
component loadings.

DBSCAN matters most: the other three *force* structure — give k-means pure noise and it
returns k clusters regardless. DBSCAN can decline to assign a point. Agreement across
families is measured with the Adjusted Rand Index.

---

## Results

### Convergent validity: +0.615

Mean Spearman correlation between the Anthropic Economic Index (revealed Claude usage)
and the Eloundou measures (human raters applying a rubric) is **+0.615**. Two entirely
independent methodologies converge, so the exposure construct is measuring something
real — which makes the null result below interpretable rather than ambiguous.

### Clustering: k=2, silhouette 0.232

| Cluster | n | Profile | Representative occupations |
|---|---|---|---|
| 0 | 257 | exposure +0.91σ, LLM +1.02σ, Claude usage +0.76σ, wage +0.61σ | database administrators, PR specialists, personal financial advisors, animators |
| 1 | 516 | exposure −0.45σ, best-separated (silhouette 0.303) | pipelayers, auto glass installers, small engine mechanics |

The split is knowledge work vs. manual work — the fundamental divide AI exposure tracks.
But cluster 0's `revision_pp` is only −0.11σ against cluster 1's +0.05σ: the exposed
cluster is *barely* more downgraded. The null result shows up structurally.

Silhouette 0.232 is below the ~0.25 rule of thumb for clear separation. Occupations lie
on an exposure **continuum**, not in discrete types. Report clusters as descriptive
strata, not as discovered categories.

### Hypothesis: mostly null after controls

Raw revision correlates negatively with exposure across all ten measures — consistent
sign, several significant (AEI: ρ=−0.128, p=0.0004). Residualize on pre-AI growth and
most collapse to zero or flip positive. Only the two **alpha** measures survive:

| Measure | net of mean reversion | p |
|---|---|---|
| `gpts_dv_rating_alpha` | −0.092 | 0.010 |
| `gpts_task_alpha_mean` | −0.108 | 0.003 |

Notably, alpha is *direct* LLM exposure while beta/gamma add LLM-powered software. The
effect that survives is about the model itself, not the tooling around it.

### Four unsupervised families disagree — and the cluster effect vanishes

`cluster_four_methods.py` runs four algorithms with different built-in assumptions
(centroid / probabilistic / nested / density) and measures agreement with the
Adjusted Rand Index.

| | kmeans | gmm | ward | dbscan |
|---|---|---|---|---|
| **kmeans** | 1.000 | 0.238 | 0.615 | 0.408 |
| **gmm** | 0.238 | 1.000 | 0.234 | 0.149 |
| **ward** | 0.615 | 0.234 | 1.000 | 0.367 |
| **dbscan** | 0.408 | 0.149 | 0.367 | 1.000 |

Mean pairwise ARI **0.335 — moderate**. K-means and Ward largely agree (0.615), but GMM
recovers a very different partition. DBSCAN labels ~195 occupations as noise rather
than forcing them into clusters. Read together with the 0.23 silhouette, this says
occupations lie on a **continuum**, and each method is slicing it according to its own
assumptions. Report clusters as strata, not discovered natural kinds.

**The decisive result:** with `revision_pp` removed from the features, a one-way ANOVA
across k-means clusters gives **F=0.049, p=0.825** — clusters do *not* differ
significantly in projection revisions. The apparent cluster difference in the
outcome-included specification was largely circular. AI-exposed occupations form a
clear, well-separated group in feature space; that group has **not** been
systematically downgraded by BLS.

### Caveats

1. **Multiple comparisons.** 20 tests were run; Bonferroni would require p<0.0025. The
   best result (0.0028) does not clear it. Apply Benjamini-Hochberg before claiming
   significance.
2. **Possible over-control.** High-exposure occupations are disproportionately tech jobs
   that were *also* projected to grow fast pre-AI (cluster 0: growth_pre +0.22σ).
   Residualizing on pre-growth strips some genuine AI signal along with the mean
   reversion. The truth lies between the raw and residualized columns.
3. **Projections are not outcomes.** Both vintages are forecasts. This measures how BLS
   *changed its mind*, not what happened in the labor market. For realized effects, join
   OEWS or CES actuals for 2022–2025.
4. **Exposure ≠ displacement.** Every measure here is explicit that it captures
   potential applicability, not substitution. High exposure is compatible with AI
   complementing workers.
5. **Felten crosswalk fallback.** When the BLS 2010→2018 crosswalk is unreachable, the
   build assumes SOC codes are unchanged — true for most, but it costs accuracy on
   recoded occupations.

### Suggested next steps

- Drop the 79 pandemic-sensitive occupations entirely instead of residualizing. On an
  earlier AEI-only run this *strengthened* the correlation to −0.17.
- Use 2019–29 as the pre-vintage to sidestep the pandemic base year.
- Test against realized OEWS employment rather than projection-vs-projection.
- Multiple regression with wage, education, and major-group fixed effects.

---

## Notes on reproducibility

- `bls.gov` blocks datacenter IP ranges. Cloud notebooks will need `--manual`, or
  files downloaded from a residential connection.
- Header wording differs across BLS vintages ("Percent employment change" vs.
  "Employment change, percent, 2024–34"); the build matches both.
- Employment figures are in thousands.
- Task-text joins use normalized exact matching; expect a small unmatched tail.
