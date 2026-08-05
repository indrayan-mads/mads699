"""Compare four unsupervised clustering families on the occupation panel.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

SEED = 42

# --- tuning constants (were inline magic numbers) ---------------------------
MIN_NONNULL = 50          # a column needs this many values to be usable as a feature
MIN_ROW_COVERAGE = 0.60   # keep rows with >=60% of features present (drop >40% imputed)
KMEANS_RESTARTS = 25
GMM_RESTARTS = 5
DBSCAN_EPS_PERCENTILES = [50, 60, 70, 75, 80, 85, 90, 95]
MERGE_GAP_WINDOW = 10     # how many top merges to inspect for a natural cut point
ARI_STRONG, ARI_MODERATE = 0.60, 0.30
RULE_OF_THUMB_SILHOUETTE = 0.25  # below this, clusters are not cleanly separated

BANNER = "=" * 72


def log(m: str) -> None:
    print(f"[cluster] {m}")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def detect_vintage_columns(P: pd.DataFrame) -> tuple[str | None, str | None]:
    """Find the pre-AI growth column and the post-vintage employment column.

    Avoids hardcoding 2021-31/2024-34 so this still works when the build is run
    with --pre 2019-29 or a newer post vintage.
    """
    pct = sorted(c for c in P.columns if re.fullmatch(r"pct_chg_\d{4}_\d{4}", c))
    emp = sorted(c for c in P.columns if re.fullmatch(r"emp_\d{4}", c))
    pre_pct = pct[0] if pct else None          # earliest base year = pre-AI vintage
    post_emp = emp[-2] if len(emp) >= 2 else (emp[-1] if emp else None)
    return pre_pct, post_emp


def build_features(P: pd.DataFrame, include_outcome: bool) -> pd.DataFrame:
    F: dict[str, pd.Series] = {}

    def num(c: str) -> pd.Series:
        return pd.to_numeric(P[c], errors="coerce")

    def first(pat: str) -> str | None:
        return next((c for c in P.columns if re.match(pat, c, re.I)), None)

    def usable(c: str) -> bool:
        return c in P.columns and num(c).notna().sum() > MIN_NONNULL

    # --- AI exposure (the substantive axis) ---------------------------------
    aei = [c for c in P.columns
           if c.startswith("aei_") and c != "aei_claude_usage_share" and usable(c)]
    if aei:
        best = max(aei, key=lambda c: num(c).notna().sum())
        F["ai_exposure"] = num(best)
        log(f"exposure: {best}")
    for pat in [r"^gpts_human_rating_beta$", r"^gpts_dv_rating_beta$", r"^gpts_.*beta"]:
        h = first(pat)
        if h and usable(h):
            F["llm_exposure"] = num(h)
            log(f"llm exposure: {h}")
            break
    if usable("aei_claude_usage_share"):
        F["log_claude_usage"] = np.log1p(num("aei_claude_usage_share"))
    if usable("aioe_felten"):
        F["aioe_felten"] = num("aioe_felten")

    # --- job structure -------------------------------------------------------
    # Note the deliberate exclusion of the post-vintage growth column:
    # post_growth = pre_growth + revision_pp exactly, so including all three
    # would span only two dimensions and silently double-weight growth against
    # exposure in a distance-based clusterer.
    pre_pct, post_emp = detect_vintage_columns(P)
    if pre_pct and usable(pre_pct):
        F["growth_pre"] = num(pre_pct)
        log(f"pre-AI growth: {pre_pct}")
    if post_emp and usable(post_emp):
        F["log_employment"] = np.log(num(post_emp).clip(lower=0.1))
    wage = first(r"median_wage_")
    if wage and usable(wage):
        F["log_wage"] = np.log(num(wage).clip(lower=1))
    for c in ["n_onet_tasks", "mean_task_importance"]:
        if usable(c):
            F[c] = num(c)

    # --- the outcome, excluded by default ------------------------------------
    if include_outcome and "revision_pp" in P.columns:
        F["revision_pp"] = num("revision_pp")
        log("INCLUDING revision_pp as a feature - clusters will be partly "
            "defined by the outcome (circular for hypothesis claims)")
    else:
        log("EXCLUDING revision_pp from features (avoids circularity); "
            "it is still compared ACROSS clusters afterwards")

    if len(F) < 2:
        sys.exit("[cluster] fewer than 2 usable features found; check the panel file.")

    X = pd.DataFrame(F)
    keep = X.notna().sum(axis=1) >= max(2, int(np.ceil(X.shape[1] * MIN_ROW_COVERAGE)))
    X = X[keep]
    X = X.fillna(X.median(numeric_only=True))
    log(f"{X.shape[0]} occupations x {X.shape[1]} features: {list(X.columns)}")
    return X


# ---------------------------------------------------------------------------
# The four methods
# ---------------------------------------------------------------------------

def run_kmeans(Z: np.ndarray, kmax: int) -> np.ndarray:
    print(f"\n{BANNER}\n1. K-MEANS  (centroid-based, hard partition)\n{BANNER}")
    print(f"{'k':>3} | {'silhouette':>11} | {'inertia':>11}")
    best_k, best_sil, best_model = None, -2.0, None
    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=KMEANS_RESTARTS, random_state=SEED).fit(Z)
        s = silhouette_score(Z, km.labels_)
        print(f"{k:>3} | {s:>11.4f} | {km.inertia_:>11.1f}")
        if s > best_sil:
            best_k, best_sil, best_model = k, s, km
    print(f"  -> k={best_k}, silhouette {best_sil:.4f}")
    if best_sil < RULE_OF_THUMB_SILHOUETTE:
        print(f"  below the ~{RULE_OF_THUMB_SILHOUETTE} rule of thumb: occupations lie on a")
        print("  continuum rather than in cleanly separated groups. Report these as")
        print("  descriptive strata, not as discovered occupational types.")
    return best_model.labels_


def run_gmm(Z: np.ndarray, kmax: int) -> tuple[np.ndarray, np.ndarray]:
    print(f"\n{BANNER}\n2. GAUSSIAN MIXTURE  (probabilistic, soft membership, elliptical)\n{BANNER}")
    print(f"{'n':>3} | {'BIC':>12} | {'silhouette':>11} | {'mean max prob':>13}")
    best_n, best_bic, best_model = None, np.inf, None
    for n in range(2, kmax + 1):
        gm = GaussianMixture(n, covariance_type="full", n_init=GMM_RESTARTS,
                             random_state=SEED).fit(Z)
        lab, pr, bic = gm.predict(Z), gm.predict_proba(Z), gm.bic(Z)
        s = silhouette_score(Z, lab) if len(set(lab)) > 1 else np.nan
        print(f"{n:>3} | {bic:>12.1f} | {s:>11.4f} | {pr.max(1).mean():>13.3f}")
        if bic < best_bic:
            best_n, best_bic, best_model = n, bic, gm
    lab, pr = best_model.predict(Z), best_model.predict_proba(Z)
    print(f"  -> n={best_n}, BIC {best_bic:.1f}, silhouette {silhouette_score(Z, lab):.4f}")
    print("  low max-probability rows are occupations sitting between archetypes")
    return lab, pr


def run_ward(Z: np.ndarray, kmax: int) -> np.ndarray:
    print(f"\n{BANNER}\n3. WARD HIERARCHICAL  (agglomerative, nested tree, no k assumed)\n{BANNER}")
    L = linkage(Z, method="ward")
    print(f"{'k':>3} | {'silhouette':>11} | {'merge height':>13}")
    best_k, best_sil, best_lab = None, -2.0, None
    for k in range(2, kmax + 1):
        lab = fcluster(L, k, criterion="maxclust")
        s = silhouette_score(Z, lab) if len(set(lab)) > 1 else np.nan
        print(f"{k:>3} | {s:>11.4f} | {L[-(k - 1), 2]:>13.2f}")
        if s > best_sil:
            best_k, best_sil, best_lab = k, s, lab
    print(f"  -> k={best_k}, silhouette {best_sil:.4f}")

    # A large gap between successive merge heights marks a natural cut point:
    # the tree had to reach much further to join the next pair of clusters.
    window = min(MERGE_GAP_WINDOW, len(L))
    gaps = np.diff(L[-window:, 2])
    if len(gaps):
        j = int(np.argmax(gaps))
        print(f"  largest merge-height gap suggests a natural cut at "
              f"k={window - j - 1} (gap {gaps[j]:.2f})")
    return best_lab


def run_dbscan(Z: np.ndarray) -> np.ndarray | None:
    print(f"\n{BANNER}\n4. DBSCAN  (density-based; can label points as NOISE)\n{BANNER}")
    # eps heuristic: sweep percentiles of the k-NN distance curve around its knee.
    # min_samples = 2 * dimensionality is the usual starting rule.
    min_samples = max(4, 2 * Z.shape[1])
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Z)
    d, _ = nn.kneighbors(Z)
    kdist = np.sort(d[:, -1])
    print(f"  min_samples={min_samples}; scanning eps around the k-NN knee")
    print(f"{'eps':>7} | {'clusters':>8} | {'noise':>7} | {'silhouette':>11}")

    best_eps, best_sil, best_lab, best_ncl = None, -2.0, None, None
    for q in DBSCAN_EPS_PERCENTILES:
        eps = float(np.percentile(kdist, q))
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
        n_cl = len(set(lab) - {-1})
        noise = int((lab == -1).sum())
        s = np.nan
        if n_cl >= 2 and noise < len(lab):
            m = lab != -1
            if len(set(lab[m])) > 1:
                s = silhouette_score(Z[m], lab[m])
        shown = "nan" if np.isnan(s) else f"{s:.4f}"
        print(f"{eps:>7.3f} | {n_cl:>8d} | {noise:>7d} | {shown:>11}")
        if not np.isnan(s) and s > best_sil:
            best_eps, best_sil, best_lab, best_ncl = eps, s, lab, n_cl

    if best_lab is None:
        print("  -> DBSCAN found NO stable density-separated clusters at any eps.")
        print("     That is a substantive result: the data has no dense, well-")
        print("     separated groups. Occupations lie on a CONTINUUM, and the")
        print("     k-means/GMM/Ward partitions above are convenient slices of")
        print("     that continuum rather than discovered natural kinds.")
        return None
    print(f"  -> eps={best_eps:.3f}, {best_ncl} clusters, silhouette {best_sil:.4f}, "
          f"{int((best_lab == -1).sum())} left unassigned as noise")
    return best_lab


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_pca(Z: np.ndarray, columns: pd.Index) -> np.ndarray:
    pca = PCA().fit(Z)
    ev = pca.explained_variance_ratio_
    print(f"\n{BANNER}\n5. PCA  (dimensionality reduction, for visualization)\n{BANNER}")
    print(f"  PC1 {ev[0] * 100:.1f}%  PC2 {ev[1] * 100:.1f}%  "
          f"(2 components capture {ev[:2].sum() * 100:.1f}% of variance)")
    loadings = pd.DataFrame(pca.components_[:2].T, index=columns, columns=["PC1", "PC2"])
    print(loadings.round(3).to_string())
    return PCA(2, random_state=SEED).fit_transform(Z)


def report_agreement(methods: dict[str, np.ndarray]) -> pd.DataFrame:
    print(f"\n{BANNER}")
    print("CROSS-METHOD AGREEMENT  (Adjusted Rand Index; 1=identical, 0=chance)")
    print(BANNER)
    names = list(methods)
    A = pd.DataFrame(index=names, columns=names, dtype=float)
    for i in names:
        for j in names:
            A.loc[i, j] = adjusted_rand_score(methods[i], methods[j])
    print(A.round(3).to_string())
    off = [A.loc[i, j] for i in names for j in names if i != j]
    mean_ari = float(np.mean(off))
    print(f"\n  mean pairwise ARI: {mean_ari:.3f}")
    if mean_ari > ARI_STRONG:
        print("  STRONG: methods with different assumptions recover the same")
        print("  partition. The cluster structure is robust and worth naming.")
    elif mean_ari > ARI_MODERATE:
        print("  MODERATE: methods partly agree. Report the k-means solution but")
        print("  note that cluster boundaries are assumption-dependent.")
    else:
        print("  WEAK: methods disagree substantially. Each is imposing its own")
        print("  assumptions on what is essentially a continuum. Do NOT present")
        print("  these as discovered occupational types - describe them as strata.")
    return A


def report_outcome(out: pd.DataFrame, include_outcome: bool) -> None:
    """Compare revision_pp across clusters. This is the decisive test."""
    print(f"\n{BANNER}")
    if include_outcome:
        print("OUTCOME BY CLUSTER  (WARNING: revision_pp WAS a feature - circular)")
    else:
        print("OUTCOME BY CLUSTER  (revision_pp was NOT a clustering feature)")
    print(BANNER)
    for m in ["kmeans_cluster", "gmm_cluster", "ward_cluster"]:
        print(f"\n{m}:")
        print(out.groupby(m)["revision_pp"].agg(["count", "mean"]).round(2).to_string())

    groups = [g["revision_pp"].dropna().values for _, g in out.groupby("kmeans_cluster")]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) > 1:
        f_stat, p = stats.f_oneway(*groups)
        print(f"\none-way ANOVA across k-means clusters: F={f_stat:.3f}, p={p:.4f}")
        # Kruskal-Wallis as a distribution-free check, since revision_pp is skewed.
        h_stat, p_kw = stats.kruskal(*groups)
        print(f"Kruskal-Wallis (rank-based check):      H={h_stat:.3f}, p={p_kw:.4f}")
        if p < 0.05 and p_kw < 0.05:
            verdict = "both tests find a cluster difference in revision_pp"
        elif p >= 0.05 and p_kw >= 0.05:
            verdict = "neither test finds a cluster difference in revision_pp"
        else:
            verdict = ("tests disagree; treat the cluster-outcome result as "
                       "sensitive to distributional assumptions")
        print(f"  {verdict}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=str(OUT / "occupation_ai_panel.csv"),
                    help="path to the occupation panel built by build_ai_jobs_dataset.py")
    ap.add_argument("--kmax", type=int, default=10, help="largest k to search")
    ap.add_argument("--include-outcome", action="store_true",
                    help="put revision_pp in the feature matrix; makes cluster-vs-outcome "
                         "comparisons circular, provided for contrast only")
    # Under Jupyter, sys.argv holds the kernel's own arguments; ignore them.
    argv = [] if "ipykernel" in sys.modules else sys.argv[1:]
    a = ap.parse_args(argv)

    panel_path = Path(a.panel)
    if not panel_path.exists():
        sys.exit(f"[cluster] {panel_path} not found - run build_ai_jobs_dataset.py first")
    P = pd.read_csv(panel_path)
    log(f"panel: {P.shape[0]} occupations x {P.shape[1]} cols")

    X = build_features(P, a.include_outcome)
    Z = StandardScaler().fit_transform(X)

    km_lab = run_kmeans(Z, a.kmax)
    gm_lab, gm_prob = run_gmm(Z, a.kmax)
    wd_lab = run_ward(Z, a.kmax)
    db_lab = run_dbscan(Z)
    coords = report_pca(Z, X.columns)

    methods = {"kmeans": km_lab, "gmm": gm_lab, "ward": wd_lab}
    if db_lab is not None:
        methods["dbscan"] = db_lab
    ari = report_agreement(methods)

    # --- assemble the analysis file -----------------------------------------
    out = P.loc[X.index].copy()
    out["kmeans_cluster"] = km_lab
    out["kmeans_silhouette"] = silhouette_samples(Z, km_lab)
    out["gmm_cluster"] = gm_lab
    out["gmm_max_prob"] = gm_prob.max(1)
    for j in range(gm_prob.shape[1]):
        out[f"gmm_prob_{j}"] = gm_prob[:, j]
    out["ward_cluster"] = wd_lab
    if db_lab is not None:
        out["dbscan_cluster"] = db_lab
        out["dbscan_noise"] = db_lab == -1
    out["pca1"], out["pca2"] = coords[:, 0], coords[:, 1]
    for c in X.columns:  # the exact standardized inputs, for reproducibility
        out["feat_" + c] = X[c]

    if "revision_pp" in out.columns:
        report_outcome(out, a.include_outcome)

    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "clusters_four_methods.csv", index=False)

    prof = (pd.DataFrame(Z, columns=X.columns, index=X.index)
            .assign(k=km_lab).groupby("k").mean().round(2))
    prof.insert(0, "n", pd.Series(km_lab).value_counts().sort_index().values)
    prof.to_csv(OUT / "cluster_profiles_four_methods.csv")
    ari.round(3).to_csv(OUT / "method_agreement_ari.csv")

    print(f"\nwrote {OUT / 'clusters_four_methods.csv'} ({out.shape[0]} x {out.shape[1]})")
    print(f"wrote {OUT / 'cluster_profiles_four_methods.csv'}")
    print(f"wrote {OUT / 'method_agreement_ari.csv'}")
    print("\nCLUSTER PROFILES (z-scores, k-means):")
    print(prof.to_string())


if __name__ == "__main__":
    main()
