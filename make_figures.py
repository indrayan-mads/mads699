"""Generate the figures used in the written report.

Reads the outputs of cluster_four_methods.py and analyze_exposure_revision.py
and writes four PNGs to figures/. Every figure in the report is produced here,
so the report's visuals are reproducible from the committed code.

    fig1_pca_clusters.png       occupations in PCA space, coloured by cluster
    fig2_exposure_revision.png  exposure vs revision, raw and residualized
    fig3_cluster_profiles.png   what actually distinguishes the clusters
    fig4_method_agreement.png   how much the four algorithms agree

Usage:
    python make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed; write files directly

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
FIGDIR = ROOT / "figures"

DPI = 200
CLUSTER_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def log(m: str) -> None:
    print(f"[figures] {m}")


def style(ax: plt.Axes) -> None:
    """Consistent, uncluttered axes across every figure."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def fig1_pca(C: pd.DataFrame) -> None:
    """Occupations in PCA space. The point is that the clusters blend."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (k, g) in enumerate(C.groupby("kmeans_cluster")):
        ax.scatter(g["pca1"], g["pca2"], s=18, alpha=0.65,
                   color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                   edgecolors="none", label=f"cluster {k} (n={len(g)})")
    if "dbscan_noise" in C.columns:
        noise = C[C["dbscan_noise"]]
        ax.scatter(noise["pca1"], noise["pca2"], s=26, facecolors="none",
                   edgecolors="black", linewidths=0.5, alpha=0.5,
                   label=f"DBSCAN noise (n={len(noise)})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Occupations in feature space, coloured by k-means cluster\n"
                 "Circled points are those DBSCAN declined to assign",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_pca_clusters.png", dpi=DPI)
    plt.close(fig)
    log("fig1_pca_clusters.png")


def fig2_exposure_revision(C: pd.DataFrame, exposure: str) -> None:
    """The core result: a visible raw relationship that survives poorly."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    pandemic = C.get("pandemic_sensitive", pd.Series(False, index=C.index)).fillna(False).astype(bool)

    for ax, ycol, title in [
        (axes[0], "revision_pp", "Raw revision"),
        (axes[1], "revision_resid", "After removing mean reversion"),
    ]:
        if ycol not in C.columns:
            ax.set_visible(False)
            continue
        d = C[[exposure, ycol]].apply(pd.to_numeric, errors="coerce")
        ok = d.notna().all(axis=1)
        ax.axhline(0, color="grey", linewidth=0.8, zorder=1)
        rest = ok & ~pandemic
        ax.scatter(d.loc[rest, exposure], d.loc[rest, ycol],
                   s=14, alpha=0.5, color="#1f77b4", edgecolors="none",
                   label="all other occupations")
        pan = ok & pandemic
        if pan.any():
            ax.scatter(d.loc[pan, exposure], d.loc[pan, ycol], s=22, alpha=0.85,
                       color="#d62728", edgecolors="none",
                       label="pandemic-sensitive (SOC 27/35/39)")
        dd = d.loc[ok]
        rho, p = stats.spearmanr(dd[exposure], dd[ycol])
        b = np.polyfit(dd[exposure], dd[ycol], 1)
        xs = np.linspace(dd[exposure].min(), dd[exposure].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color="black", linewidth=1.2, zorder=3)
        ax.set_title(f"{title}\nSpearman rho = {rho:+.3f}, p = {p:.4f}",
                     fontsize=10, loc="left")
        ax.set_xlabel("AI exposure")
        style(ax)

    axes[0].set_ylabel("Change in projected 10-yr growth (pp)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("AI exposure vs how BLS revised its projections", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_exposure_revision.png", dpi=DPI)
    plt.close(fig)
    log("fig2_exposure_revision.png")


def fig3_profiles(prof: pd.DataFrame) -> None:
    """Heatmap of standardized feature means, so clusters can be named."""
    feats = [c for c in prof.columns if c != "n"]
    M = prof[feats].astype(float).values
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(feats)), 2.2 + 0.5 * len(prof)))
    lim = float(np.nanmax(np.abs(M))) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels([f"cluster {i} (n={int(prof['n'].iloc[j])})"
                        for j, i in enumerate(prof.index)], fontsize=9)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            if np.isfinite(M[r, c]):
                ax.text(c, r, f"{M[r, c]:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[r, c]) > lim * 0.6 else "black")
    ax.set_title("Cluster profiles: standardized feature means (z-scores)",
                 fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="standard deviations from mean")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_cluster_profiles.png", dpi=DPI)
    plt.close(fig)
    log("fig3_cluster_profiles.png")


def fig4_agreement(A: pd.DataFrame) -> None:
    """How much the four families agree. Moderate agreement is the finding."""
    M = A.astype(float).values
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(A)))
    ax.set_xticklabels(A.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(A)))
    ax.set_yticklabels(A.index)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[r, c] < 0.6 else "black")
    off = M[~np.eye(len(M), dtype=bool)]
    ax.set_title(f"Cross-method agreement (Adjusted Rand Index)\n"
                 f"mean pairwise ARI = {off.mean():.3f}", fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="ARI (1 = identical, 0 = chance)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_method_agreement.png", dpi=DPI)
    plt.close(fig)
    log("fig4_method_agreement.png")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", default=str(OUT / "clusters_four_methods.csv"))
    ap.add_argument("--profiles", default=str(OUT / "cluster_profiles_four_methods.csv"))
    ap.add_argument("--ari", default=str(OUT / "method_agreement_ari.csv"))
    ap.add_argument("--derived", default=str(OUT / "revision_derived.csv"))
    argv = [] if "ipykernel" in sys.modules else sys.argv[1:]
    a = ap.parse_args(argv)

    cpath = Path(a.clusters)
    if not cpath.exists():
        sys.exit(f"[figures] {cpath} not found - run cluster_four_methods.py first")
    FIGDIR.mkdir(exist_ok=True)

    C = pd.read_csv(cpath)

    # Pull in revision_resid / pandemic_sensitive if the analysis script has run.
    dpath = Path(a.derived)
    if dpath.exists():
        D = pd.read_csv(dpath)
        cols = [c for c in ["soc_code", "revision_resid", "pandemic_sensitive"]
                if c in D.columns]
        C = C.merge(D[cols], on="soc_code", how="left")
    else:
        log("revision_derived.csv not found; run analyze_exposure_revision.py for "
            "the residualized panel of figure 2")

    fig1_pca(C)

    exposure = next((c for c in ["feat_ai_exposure", "feat_llm_exposure"] if c in C.columns), None)
    if exposure:
        fig2_exposure_revision(C, exposure)
    else:
        log("no exposure feature column found; skipping figure 2")

    ppath = Path(a.profiles)
    if ppath.exists():
        fig3_profiles(pd.read_csv(ppath, index_col=0))
    else:
        log("cluster profiles not found; skipping figure 3")

    apath = Path(a.ari)
    if apath.exists():
        fig4_agreement(pd.read_csv(apath, index_col=0))
    else:
        log("ARI matrix not found; skipping figure 4")

    log(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
