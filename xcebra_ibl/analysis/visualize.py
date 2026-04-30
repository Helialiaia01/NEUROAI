"""
Visualization module for xCEBRA-IBL results.

Generates publication-quality figures that directly compare with
the "Rarely Categorical" paper's figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram

from xcebra_ibl.configs.config import (
    VARIABLE_NAMES, VARIABLE_DISPLAY_NAMES, N_VARIABLES, RESULTS_DIR,
)


def setup_plotting():
    """Set up publication-ready matplotlib defaults."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })
    sns.set_style("ticks")


def plot_selectivity_heatmap(
    sel_areas, area_order, area2mod=None,
    save_path=None, title="xCEBRA Selectivity Profiles",
):
    """
    Plot 8D selectivity heatmap per brain area.
    (Analog of Fig. 3c in the paper)

    Parameters
    ----------
    sel_areas : (n_areas, 8) z-scored selectivity
    area_order : list of area names
    area2mod : dict, area → module mapping (for color-coding)
    save_path : str or Path
    """
    setup_plotting()
    fig, ax = plt.subplots(1, 1, figsize=(4, max(7, len(area_order) * 0.25)))

    improp = dict(
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        vmax=np.max(np.abs(sel_areas)),
        vmin=-np.max(np.abs(sel_areas)),
    )
    im = ax.imshow(sel_areas, **improp)

    # Mark top-3 areas per variable
    for vi in range(N_VARIABLES):
        sorted_idx = np.argsort(sel_areas[:, vi])[::-1]
        for ai in sorted_idx[:3]:
            ax.annotate(
                "*", xy=(vi, ai + 0.5), ha="center", va="bottom",
                color="gold", fontsize=14, fontweight="bold",
            )

    ax.set_xticks(np.arange(N_VARIABLES))
    ax.set_xticklabels(VARIABLE_DISPLAY_NAMES, rotation=90)
    ax.set_yticks(np.arange(len(area_order)))
    ax.set_yticklabels(area_order)

    # Color-code y-axis labels by cortical module
    if area2mod is not None:
        mod2mi = {
            "visual": 0, "somatomotor": 1, "auditory": 2,
            "lateral": 3, "medial": 4, "prefrontal": 5,
        }
        cmap = plt.get_cmap("tab10")
        for i, area in enumerate(area_order):
            mod = area2mod.get(area, "unknown")
            color = cmap(mod2mi.get(mod, 7))
            ax.yaxis.get_ticklabels()[i].set_color(color)

    cbar = plt.colorbar(im, shrink=0.4)
    cbar.ax.set_ylabel(r"Mean $\alpha_a$ (z-scored)")

    ax.set_title(title)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close("all")


def plot_selectivity_vs_connectivity(
    corr_result, save_path=None,
    title="xCEBRA: Selectivity Similarity vs Anatomical Connectivity",
):
    """
    Scatter plot: selectivity similarity vs anatomical connectivity.
    (Analog of Fig. 3d in the paper)
    """
    setup_plotting()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    conn = np.array(corr_result["conn_values"])
    sim = np.array(corr_result["sim_values"])
    valid = ~np.isnan(conn)

    sns.regplot(x=conn[valid], y=sim[valid], ax=ax, scatter_kws={"s": 15, "alpha": 0.6})

    rs = corr_result["spearman_r"]
    ps = corr_result["spearman_p"]
    p_text = f"p = {ps:.1e}" if ps < 0.001 else f"p = {ps:.3f}"
    ax.set_title(f"{p_text}  ρ = {rs:.2f}")
    ax.set_xlabel("Anatomical connectivity (log)")
    ax.set_ylabel("Selectivity similarity (cosine)")

    # Annotate high-similarity pairs
    pairs = corr_result["pairs"]
    for i in range(len(sim)):
        if sim[i] > 0.9 and not np.isnan(conn[i]):
            ax.text(
                conn[i], sim[i], f"{pairs[i][0]}-{pairs[i][1]}",
                fontsize=6, ha="center", va="center", alpha=0.8,
            )

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close("all")


def plot_area_dendrogram(
    cluster_result, area_order, area2mod=None, save_path=None,
    title="Hierarchical Clustering of Brain Areas",
):
    """
    Dendrogram of brain area clustering by selectivity profile.
    """
    setup_plotting()
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    Z = cluster_result["linkage_matrix"]
    dend = dendrogram(
        Z, labels=area_order, leaf_rotation=90,
        leaf_font_size=8, ax=ax, color_threshold=0,
    )

    # Color labels by module
    if area2mod is not None:
        mod2mi = {
            "visual": 0, "somatomotor": 1, "auditory": 2,
            "lateral": 3, "medial": 4, "prefrontal": 5,
        }
        cmap = plt.get_cmap("tab10")
        for xtick in ax.get_xticklabels():
            area = xtick.get_text()
            mod = area2mod.get(area, "unknown")
            xtick.set_color(cmap(mod2mi.get(mod, 7)))

    ax.set_title(title)
    ax.set_ylabel("Distance (cosine)")
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close("all")


def plot_rrr_vs_xcebra_comparison(
    comparison, save_path=None,
    title="RRR vs xCEBRA Selectivity Comparison",
):
    """
    Side-by-side comparison of RRR and xCEBRA selectivity.
    """
    if comparison is None:
        print("  No RRR comparison data available.")
        return

    setup_plotting()

    rrr_sel = comparison.get("rrr_sel_z")
    xcebra_sel = comparison.get("xcebra_sel_z")
    common_areas = comparison.get("common_areas", [])
    baseline_label = comparison.get("baseline_label", "rrr")
    baseline_title = "RRR Selectivity" if baseline_label == "rrr" else "Linear Baseline Selectivity"

    if rrr_sel is None or xcebra_sel is None:
        return

    fig = plt.figure(figsize=(14, max(7, len(common_areas) * 0.25)))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.3)

    vmax = max(np.abs(rrr_sel).max(), np.abs(xcebra_sel).max())
    improp = dict(
        aspect="auto", cmap="RdBu_r", interpolation="nearest",
        vmax=vmax, vmin=-vmax,
    )

    # Baseline heatmap
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(rrr_sel, **improp)
    ax1.set_xticks(np.arange(N_VARIABLES))
    ax1.set_xticklabels(VARIABLE_DISPLAY_NAMES, rotation=90)
    ax1.set_yticks(np.arange(len(common_areas)))
    ax1.set_yticklabels(common_areas)
    ax1.set_title(baseline_title)

    # xCEBRA heatmap
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(xcebra_sel, **improp)
    ax2.set_xticks(np.arange(N_VARIABLES))
    ax2.set_xticklabels(VARIABLE_DISPLAY_NAMES, rotation=90)
    ax2.set_yticks(np.arange(len(common_areas)))
    ax2.set_yticklabels([])
    ax2.set_title("xCEBRA Selectivity")

    cbar = plt.colorbar(im, shrink=0.4)
    cbar.ax.set_ylabel("z-scored selectivity")

    # Per-variable correlation bar plot
    ax3 = fig.add_subplot(gs[2])
    per_var = comparison.get("per_variable_correlation", {})
    var_names = list(per_var.keys())
    corr_vals = [per_var[v]["spearman_r"] for v in var_names]
    colors = ["green" if per_var[v]["p_value"] < 0.05 else "gray" for v in var_names]

    display_names = [VARIABLE_DISPLAY_NAMES[VARIABLE_NAMES.index(v)] for v in var_names]
    ax3.barh(display_names, corr_vals, color=colors, alpha=0.7)
    ax3.set_xlabel("Spearman ρ")
    ax3.set_title("Per-variable\ncorrelation")
    ax3.axvline(0, color="black", linewidth=0.5)

    overall_r = comparison.get("overall_spearman_r", 0)
    fig.suptitle(f"{title}\nOverall ρ = {overall_r:.3f}", y=1.02)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close("all")


def plot_training_losses(losses_dict, save_path=None):
    """Plot training losses for all variable models."""
    setup_plotting()

    n_vars = len(losses_dict)
    if n_vars == 0:
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
    axes = axes.ravel()

    for idx, (var_name, loss) in enumerate(losses_dict.items()):
        if idx >= 8:
            break
        ax = axes[idx]
        if hasattr(loss, '__len__') and len(loss) > 0:
            ax.plot(loss, linewidth=0.5)
        ax.set_title(VARIABLE_DISPLAY_NAMES[VARIABLE_NAMES.index(var_name)])
        ax.set_ylabel("Loss")

    for ax in axes:
        ax.set_xlabel("Iteration")

    fig.suptitle("xCEBRA Training Losses", y=1.02)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close("all")


def plot_embedding_visualization(
    embeddings, labels, var_name,
    save_path=None, n_samples=5000,
):
    """
    2D/3D visualization of CEBRA embeddings colored by behavioral variable.

    Parameters
    ----------
    embeddings : (n_samples, dim) embedding array
    labels : (n_samples,) label array
    var_name : str, variable name
    """
    setup_plotting()

    if embeddings.shape[0] > n_samples:
        idx = np.random.choice(embeddings.shape[0], n_samples, replace=False)
        emb = embeddings[idx]
        lab = labels[idx]
    else:
        emb = embeddings
        lab = labels

    dim = emb.shape[1]

    if dim >= 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            emb[:, 0], emb[:, 1], emb[:, 2],
            c=lab, cmap="viridis", s=1, alpha=0.3,
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        scatter = ax.scatter(
            emb[:, 0], emb[:, 1] if dim > 1 else np.zeros(len(emb)),
            c=lab, cmap="viridis", s=1, alpha=0.3,
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    cbar = plt.colorbar(scatter, shrink=0.6)
    display_name = VARIABLE_DISPLAY_NAMES[VARIABLE_NAMES.index(var_name)]
    cbar.ax.set_ylabel(display_name)
    ax.set_title(f"xCEBRA Embedding — {display_name}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close("all")


def generate_all_figures(analysis_results, save_dir=None):
    """Generate all publication figures from analysis results."""
    if save_dir is None:
        save_dir = RESULTS_DIR / "figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")

    sel = analysis_results.get("selectivity")
    if sel:
        # Try to get area2mod for color coding
        area2mod = None
        try:
            from xcebra_ibl.analysis.selectivity import _load_allen_anatomy_from_csv
            atm = _load_allen_anatomy_from_csv()
            area2mod = atm["area2mod"]
        except Exception:
            pass

        plot_selectivity_heatmap(
            sel["sel_areas"], sel["area_order"],
            area2mod=area2mod,
            save_path=save_dir / "xcebra_selectivity_heatmap.pdf",
        )

    corr = analysis_results.get("anatomical_correlation")
    if corr:
        plot_selectivity_vs_connectivity(
            corr,
            save_path=save_dir / "xcebra_sel_vs_connectivity.pdf",
        )

    clust = analysis_results.get("clustering")
    if clust and sel:
        plot_area_dendrogram(
            clust, sel["area_order"],
            area2mod=area2mod if 'area2mod' in dir() else None,
            save_path=save_dir / "xcebra_area_dendrogram.pdf",
        )

    comp = analysis_results.get("rrr_comparison")
    if comp:
        plot_rrr_vs_xcebra_comparison(
            comp,
            save_path=save_dir / "rrr_vs_xcebra_comparison.pdf",
        )

    print(f"  All figures saved to {save_dir}")
