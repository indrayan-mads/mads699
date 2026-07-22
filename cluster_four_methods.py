#!/usr/bin/env python3
"""
cluster_four_methods.py
=======================
Four genuinely different families of unsupervised learning on the occupation panel,
plus a cross-method agreement check.

  1. K-MEANS         centroid-based, hard partition. Forces every occupation into
                     exactly one of k spherical clusters. k chosen by silhouette.
  2. GMM             probabilistic / model-based. Soft membership, elliptical
                     covariance, so clusters can be elongated and overlapping.
                     n chosen by BIC.
  3. WARD HIERARCHY  agglomerative. Builds a nested tree; no k assumed up front.
                     Reveals whether structure is nested (broad groups splitting
                     into sub-groups) or genuinely flat.
  4. DBSCAN          density-based. Does NOT force every point into a cluster -
                     it can label occupations as noise. This is the honest test of
                     whether real high-density groups exist at all, or whether
                     occupations lie on a continuum.

Plus PCA for a 2-D view (dimensionality reduction, a fifth unsupervised family,
used here for visualization and to report how much variance 2 components capture).

WHY FOUR: each family has a different bias. K-means assumes spherical equal-size
clusters; GMM relaxes that; Ward assumes nested structure; DBSCAN assumes density
separation. If all four agree, the structure is real. If they disagree, the data is
a continuum and any single method's "clusters" are an artifact of its assumptions.
Agreement is measured with the Adjusted Rand Index (ARI): 1.0 = identical
partitions, 0.0 = no better than chance.

IMPORTANT - CIRCULARITY: by default this EXCLUDES revision_pp from the features.
If you cluster on the outcome and then report that clusters differ in the outcome,
that is circular. Use --include-outcome only if you want purely descriptive strata.

Usage:
    python cluster_four_methods.py
    python cluster_four_methods.py --include-outcome
    python cluster_four_methods.py --panel out/occupation_ai_panel.csv --kmax 10
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42


def log(m): print(f"[cluster] {m}")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def build_features(P: pd.DataFrame, include_outcome: bool) -> pd.DataFrame:
    F: dict[str, pd.Series] = {}

    def num(c):
        return pd.to_numeric(P[c], errors="coerce")

    def first(pat):
        return next((c for c in P.columns if re.match(pat, c, re.I)), None)

    # --- AI exposure (the substantive axis) -------------------------------
    aei = [c for c in P.columns if c.startswith("aei_")
           and c != "aei_claude_usage_share" and num(c).notna().sum() > 50]
    if aei:
        best = max(aei, key=lambda c: num(c).notna().sum())
        F["ai_exposure"] = num(best)
        log(f"exposure: {best}")
    for pat in [r"^gpts_human_rating_beta$", r"^gpts_dv_rating_beta$", r"^gpts_.*beta"]:
        h = first(pat)
        if h and num(h).notna().sum() > 50:
            F["llm_exposure"] = num(h)
            log(f"llm exposure: {h}")
            break
    if "aei_claude_usage_share" in P.columns and num("aei_claude_usage_share").notna().sum() > 50:
        F["log_claude_usage"] = np.log1p(num("aei_claude_usage_share"))
    if "aioe_felten" in P.columns and num("aioe_felten").notna().sum() > 50:
        F["aioe_felten"] = num("aioe_felten")

    # --- job structure ------------------------------------------------------
    pre = first(r"pct_chg_2021_2031$")
    if pre:
        F["growth_pre"] = num(pre)
    emp = first(r"emp_2024$")
    if emp:
        F["log_employment"] = np.log(num(emp).clip(lower=.1))
    wage = first(r"median_wage_")
    if wage:
        F["log_wage"] = np.log(num(wage).clip(lower=1))
    for c in ["n_onet_tasks", "mean_task_importance"]:
        if c in P.columns and num(c).notna().sum() > 50:
            F[c] = num(c)

    # --- the outcome, excluded by default -----------------------------------
    if include_outcome and "revision_pp" in P.columns:
        F["revision_pp"] = num("revision_pp")
        log("INCLUDING revision_pp as a feature - clusters will be partly "
            "defined by the outcome (circular for hypothesis claims)")
    else:
        log("EXCLUDING revision_pp from features (avoids circularity); "
            "it is still compared ACROSS clusters afterwards")

    X = pd.DataFrame(F)
    X = X[X.notna().sum(axis=1) >= max(2, int(np.ceil(X.shape[1] * .6)))]
    X = X.fillna(X.median(numeric_only=True))
    log(f"{X.shape[0]} occupations x {X.shape[1]} features: {list(X.columns)}")
    return X


# ---------------------------------------------------------------------------
# The four methods
# ---------------------------------------------------------------------------

def run_kmeans(Z, kmax):
    print("\n" + "=" * 72)
    print("1. K-MEANS  (centroid-based, hard partition)")
    print("=" * 72)
    print(f"{'k':>3} | {'silhouette':>11} | {'inertia':>11}")
    best = (None, -2, None)
    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=25, random_state=SEED).fit(Z)
        s = silhouette_score(Z, km.labels_)
        print(f"{k:>3} | {s:>11.4f} | {km.inertia_:>11.1f}")
        if s > best[1]:
            best = (k, s, km)
    print(f"  -> k={best[0]}, silhouette {best[1]:.4f}")
    return best[2].labels_, best[0], best[1]


def run_gmm(Z, kmax):
    print("\n" + "=" * 72)
    print("2. GAUSSIAN MIXTURE  (probabilistic, soft membership, elliptical)")
    print("=" * 72)
    print(f"{'n':>3} | {'BIC':>12} | {'silhouette':>11} | {'mean max prob':>13}")
    best = (None, np.inf, None)
    for n in range(2, kmax + 1):
        gm = GaussianMixture(n, covariance_type="full", n_init=5,
                             random_state=SEED).fit(Z)
        lab, pr = gm.predict(Z), gm.predict_proba(Z)
        s = silhouette_score(Z, lab) if len(set(lab)) > 1 else np.nan
        print(f"{n:>3} | {gm.bic(Z):>12.1f} | {s:>11.4f} | {pr.max(1).mean():>13.3f}")
        if gm.bic(Z) < best[1]:
            best = (n, gm.bic(Z), gm)
    gm = best[2]
    lab, pr = gm.predict(Z), gm.predict_proba(Z)
    print(f"  -> n={best[0]}, BIC {best[1]:.1f}, silhouette {silhouette_score(Z, lab):.4f}")
    return lab, pr, best[0]


def run_ward(Z, kmax):
    print("\n" + "=" * 72)
    print("3. WARD HIERARCHICAL  (agglomerative, nested tree, no k assumed)")
    print("=" * 72)
    L = linkage(Z, method="ward")
    print(f"{'k':>3} | {'silhouette':>11} | {'merge height':>13}")
    best = (None, -2, None)
    for k in range(2, kmax + 1):
        lab = fcluster(L, k, criterion="maxclust")
        s = silhouette_score(Z, lab) if len(set(lab)) > 1 else np.nan
        h = L[-(k - 1), 2] if k > 1 else np.nan
        print(f"{k:>3} | {s:>11.4f} | {h:>13.2f}")
        if s > best[1]:
            best = (k, s, lab)
    print(f"  -> k={best[0]}, silhouette {best[1]:.4f}")
    # a large gap between successive merge heights = a natural cut point
    heights = L[-10:, 2]
    gaps = np.diff(heights)
    if len(gaps):
        j = int(np.argmax(gaps))
        print(f"  largest merge-height gap suggests a natural cut at "
              f"k={10 - j - 1} (gap {gaps[j]:.2f})")
    return best[2], best[0], L


def run_dbscan(Z):
    print("\n" + "=" * 72)
    print("4. DBSCAN  (density-based; can label points as NOISE)")
    print("=" * 72)
    # heuristic for eps: knee of the k-NN distance curve, min_samples = 2*dims
    min_samples = max(4, 2 * Z.shape[1])
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Z)
    d, _ = nn.kneighbors(Z)
    kdist = np.sort(d[:, -1])
    print(f"  min_samples={min_samples}; scanning eps around the k-NN knee")
    print(f"{'eps':>7} | {'clusters':>8} | {'noise':>7} | {'silhouette':>11}")
    best = (None, -2, None, None)
    for q in [50, 60, 70, 75, 80, 85, 90, 95]:
        eps = float(np.percentile(kdist, q))
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
        n_cl = len(set(lab) - {-1})
        noise = int((lab == -1).sum())
        if n_cl >= 2 and noise < len(lab):
            m = lab != -1
            s = silhouette_score(Z[m], lab[m]) if len(set(lab[m])) > 1 else np.nan
        else:
            s = np.nan
        print(f"{eps:>7.3f} | {n_cl:>8d} | {noise:>7d} | "
              f"{'nan' if np.isnan(s) else f'{s:>11.4f}'}")
        if not np.isnan(s) and s > best[1]:
            best = (eps, s, lab, n_cl)
    if best[2] is None:
        print("  -> DBSCAN found NO stable density-separated clusters at any eps.")
        print("     That is a substantive result: the data has no dense, well-")
        print("     separated groups. Occupations lie on a CONTINUUM, and the")
        print("     k-means/GMM/Ward partitions above are convenient slices of")
        print("     that continuum rather than discovered natural kinds.")
        return None, None
    print(f"  -> eps={best[0]:.3f}, {best[3]} clusters, silhouette {best[1]:.4f}, "
          f"{int((best[2] == -1).sum())} noise points")
    return best[2], best[0]


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="out/occupation_ai_panel.csv")
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--include-outcome", action="store_true",
                    help="put revision_pp in the feature matrix (circular - see docstring)")
    argv = [] if "ipykernel" in sys.modules else sys.argv[1:]
    a = ap.parse_args(argv)

    if not Path(a.panel).exists():
        sys.exit(f"[cluster] {a.panel} not found - run build_ai_jobs_dataset.py first")
    P = pd.read_csv(a.panel)
    log(f"panel: {P.shape[0]} occupations x {P.shape[1]} cols")

    X = build_features(P, a.include_outcome)
    Z = StandardScaler().fit_transform(X)

    km_lab, km_k, km_sil = run_kmeans(Z, a.kmax)
    gm_lab, gm_prob, gm_n = run_gmm(Z, a.kmax)
    wd_lab, wd_k, L = run_ward(Z, a.kmax)
    db_lab, db_eps = run_dbscan(Z)

    # --- 5th family: PCA, for the 2-D view ---------------------------------
    pca = PCA().fit(Z)
    ev = pca.explained_variance_ratio_
    print("\n" + "=" * 72)
    print("5. PCA  (dimensionality reduction, for visualization)")
    print("=" * 72)
    print(f"  PC1 {ev[0]*100:.1f}%  PC2 {ev[1]*100:.1f}%  "
          f"(2 components capture {ev[:2].sum()*100:.1f}% of variance)")
    loadings = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
    print(loadings.round(3).to_string())
    coords = PCA(2).fit_transform(Z)

    # --- cross-method agreement --------------------------------------------
    print("\n" + "=" * 72)
    print("CROSS-METHOD AGREEMENT  (Adjusted Rand Index; 1=identical, 0=chance)")
    print("=" * 72)
    methods = {"kmeans": km_lab, "gmm": gm_lab, "ward": wd_lab}
    if db_lab is not None:
        methods["dbscan"] = db_lab
    names = list(methods)
    A = pd.DataFrame(index=names, columns=names, dtype=float)
    for i in names:
        for j in names:
            A.loc[i, j] = adjusted_rand_score(methods[i], methods[j])
    print(A.round(3).to_string())
    off = [A.loc[i, j] for i in names for j in names if i != j]
    mean_ari = float(np.mean(off))
    print(f"\n  mean pairwise ARI: {mean_ari:.3f}")
    if mean_ari > 0.6:
        print("  STRONG: methods with different assumptions recover the same")
        print("  partition. The cluster structure is robust and worth naming.")
    elif mean_ari > 0.3:
        print("  MODERATE: methods partly agree. Report the k-means solution but")
        print("  note that cluster boundaries are assumption-dependent.")
    else:
        print("  WEAK: methods disagree substantially. Each is imposing its own")
        print("  assumptions on what is essentially a continuum. Do NOT present")
        print("  these as discovered occupational types - describe them as strata.")

    # --- what the clusters mean for the outcome -----------------------------
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
    for c in X.columns:
        out["feat_" + c] = X[c]

    if "revision_pp" in out.columns:
        print("\n" + "=" * 72)
        print("OUTCOME BY CLUSTER  (revision_pp was NOT a clustering feature"
              if not a.include_outcome else
              "OUTCOME BY CLUSTER  (WARNING: revision_pp WAS a feature - circular")
        print("=" * 72)
        for m in ["kmeans_cluster", "gmm_cluster", "ward_cluster"]:
            g = out.groupby(m)["revision_pp"].agg(["count", "mean"]).round(2)
            print(f"\n{m}:")
            print(g.to_string())
        # is the between-cluster difference real?
        groups = [g["revision_pp"].dropna().values
                  for _, g in out.groupby("kmeans_cluster")]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) > 1:
            F, p = stats.f_oneway(*groups)
            print(f"\none-way ANOVA across k-means clusters: F={F:.3f}, p={p:.4f}")
            print("  " + ("clusters DO differ in revision_pp" if p < .05
                          else "clusters do NOT differ significantly in revision_pp"))

    Path("out").mkdir(exist_ok=True)
    out.to_csv("out/clusters_four_methods.csv", index=False)
    prof = (pd.DataFrame(Z, columns=X.columns, index=X.index)
            .assign(k=km_lab).groupby("k").mean().round(2))
    prof.insert(0, "n", pd.Series(km_lab).value_counts().sort_index().values)
    prof.to_csv("out/cluster_profiles_four_methods.csv")
    print(f"\nwrote out/clusters_four_methods.csv ({out.shape[0]} x {out.shape[1]})")
    print("wrote out/cluster_profiles_four_methods.csv")
    print("\nCLUSTER PROFILES (z-scores, k-means):")
    print(prof.to_string())


if __name__ == "__main__":
    main()
