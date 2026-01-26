"""Panel size vs AUROC curve plotting for RFE results.

Visualizes the Pareto frontier between panel size and discrimination
performance, with annotations for knee points and recommended thresholds.
"""

from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False


def plot_pareto_curve(
    curve: list[dict],
    recommended: dict[str, int],
    out_path: Path | str,
    title: str = "Panel Size vs AUROC",
    model_name: str = "",
    thresholds_to_show: list[float] | None = None,
) -> None:
    """Plot AUROC vs panel size curve with annotations.

    Args:
        curve: List of dicts with keys "size", "auroc_val", "auroc_cv", "auroc_cv_std".
        recommended: Dict with "min_size_95pct", "min_size_90pct", "knee_point", etc.
        out_path: Output file path.
        title: Plot title.
        model_name: Model name for subtitle.
        thresholds_to_show: AUROC fraction thresholds to annotate (default: [0.95, 0.90]).

    Returns:
        None. Saves plot to out_path.
    """
    if not _HAS_PLOTTING:
        return

    if not curve:
        return

    if thresholds_to_show is None:
        thresholds_to_show = [0.95, 0.90]

    # Extract data
    sizes = np.array([p["size"] for p in curve])
    aurocs_val = np.array([p["auroc_val"] for p in curve])
    aurocs_cv = np.array([p.get("auroc_cv", p["auroc_val"]) for p in curve])
    aurocs_std = np.array([p.get("auroc_cv_std", 0.0) for p in curve])

    max_auroc = np.max(aurocs_val)

    # Sort by size for line plot
    sort_idx = np.argsort(sizes)[::-1]  # Descending
    sizes = sizes[sort_idx]
    aurocs_val = aurocs_val[sort_idx]
    aurocs_cv = aurocs_cv[sort_idx]
    aurocs_std = aurocs_std[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot validation AUROC
    ax.plot(
        sizes,
        aurocs_val,
        "o-",
        color="#2563eb",
        linewidth=2,
        markersize=6,
        label="Validation AUROC",
    )

    # Plot CV AUROC with error bars
    ax.errorbar(
        sizes,
        aurocs_cv,
        yerr=aurocs_std,
        fmt="s--",
        color="#64748b",
        linewidth=1,
        markersize=4,
        capsize=3,
        alpha=0.7,
        label="CV AUROC (OOF)",
    )

    # Threshold lines
    colors = ["#10b981", "#f59e0b", "#ef4444"]  # green, amber, red
    for i, thresh in enumerate(thresholds_to_show):
        target_auroc = max_auroc * thresh
        ax.axhline(
            y=target_auroc,
            color=colors[i % len(colors)],
            linestyle=":",
            alpha=0.7,
            linewidth=1.5,
        )
        ax.text(
            sizes.max() * 0.98,
            target_auroc + 0.005,
            f"{thresh:.0%} of max",
            color=colors[i % len(colors)],
            fontsize=9,
            ha="right",
            va="bottom",
        )

        # Mark recommended panel size for this threshold
        key = f"min_size_{int(thresh * 100)}pct"
        if key in recommended:
            rec_size = recommended[key]
            # Find corresponding AUROC
            idx = np.where(sizes == rec_size)[0]
            if len(idx) > 0:
                rec_auroc = aurocs_val[idx[0]]
                ax.scatter(
                    [rec_size],
                    [rec_auroc],
                    s=120,
                    c=colors[i % len(colors)],
                    marker="D",
                    zorder=10,
                    edgecolors="white",
                    linewidths=1.5,
                )
                ax.annotate(
                    f"n={rec_size}",
                    (rec_size, rec_auroc),
                    textcoords="offset points",
                    xytext=(10, -10),
                    fontsize=9,
                    color=colors[i % len(colors)],
                )

    # Mark knee point
    if "knee_point" in recommended:
        knee_size = recommended["knee_point"]
        idx = np.where(sizes == knee_size)[0]
        if len(idx) > 0:
            knee_auroc = aurocs_val[idx[0]]
            ax.scatter(
                [knee_size],
                [knee_auroc],
                s=150,
                c="#7c3aed",  # purple
                marker="*",
                zorder=10,
                edgecolors="white",
                linewidths=1,
            )
            ax.annotate(
                f"Knee (n={knee_size})",
                (knee_size, knee_auroc),
                textcoords="offset points",
                xytext=(-10, 10),
                fontsize=9,
                fontweight="bold",
                color="#7c3aed",
            )

    # Styling
    ax.set_xlabel("Panel Size (number of proteins)", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_xlim(0, sizes.max() * 1.05)

    # Y-axis: show reasonable range around the data
    y_min = min(aurocs_val.min(), aurocs_cv.min() - aurocs_std.max()) - 0.02
    y_max = max(aurocs_val.max(), aurocs_cv.max() + aurocs_std.max()) + 0.02
    y_min = max(0.5, y_min)  # Don't go below 0.5
    y_max = min(1.0, y_max)  # Don't go above 1.0
    ax.set_ylim(y_min, y_max)

    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Title
    if model_name:
        ax.set_title(f"{title}\n{model_name}", fontsize=12, fontweight="bold")
    else:
        ax.set_title(title, fontsize=12, fontweight="bold")

    # Add summary text
    summary_lines = [
        f"Max AUROC: {max_auroc:.3f}",
        f"Start size: {sizes.max()}",
        f"Min size evaluated: {sizes.min()}",
    ]
    if "knee_point" in recommended:
        summary_lines.append(f"Knee point: {recommended['knee_point']}")

    summary_text = "\n".join(summary_lines)
    ax.text(
        0.02,
        0.02,
        summary_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 3},
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_ranking(
    feature_ranking: dict[str, int],
    out_path: Path | str,
    top_n: int = 30,
    title: str = "Feature Elimination Order",
) -> None:
    """Plot horizontal bar chart of feature elimination order.

    Features eliminated last (highest order) are most important.

    Args:
        feature_ranking: Dict mapping protein -> elimination_order.
        out_path: Output file path.
        top_n: Number of top features to show.
        title: Plot title.

    Returns:
        None. Saves plot to out_path.
    """
    if not _HAS_PLOTTING:
        return

    if not feature_ranking:
        return

    # Sort by elimination order (descending = eliminated last = most important)
    sorted_features = sorted(feature_ranking.items(), key=lambda x: -x[1])[:top_n]

    proteins = [f[0] for f in sorted_features]
    orders = [f[1] for f in sorted_features]

    fig, ax = plt.subplots(figsize=(8, max(6, len(proteins) * 0.3)))

    # Color gradient: later elimination = more important = darker
    max_order = max(orders) if orders else 1
    colors = plt.cm.Blues(np.array(orders) / max_order * 0.6 + 0.3)

    ax.barh(range(len(proteins)), orders, color=colors)
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels(proteins, fontsize=9)
    ax.invert_yaxis()  # Highest order at top

    ax.set_xlabel("Elimination Order (higher = eliminated later = more important)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")

    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rfecv_selection_curve(
    cv_scores_curve_path: Path | str,
    out_path: Path | str,
    title: str = "RFECV Feature Selection Curve",
    model_name: str = "",
) -> None:
    """Plot RFECV internal CV scores vs number of features across folds.

    Shows how cross-validation AUROC varies with feature count during RFECV,
    helping visualize the automatic optimal size selection per fold.

    Args:
        cv_scores_curve_path: Path to cv_scores_curve.csv (from nested_rfe).
        out_path: Output file path for plot.
        title: Plot title.
        model_name: Model name for subtitle.

    Returns:
        None. Saves plot to out_path.
    """
    if not _HAS_PLOTTING:
        return

    import pandas as pd

    cv_scores_curve_path = Path(cv_scores_curve_path)
    if not cv_scores_curve_path.exists():
        return

    # Load CV scores data
    df = pd.read_csv(cv_scores_curve_path)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique folds
    folds = sorted(df["fold"].unique())
    n_folds = len(folds)

    # Color palette for folds
    colors = plt.cm.tab10(np.linspace(0, 1, n_folds))

    # Plot each fold
    for i, fold in enumerate(folds):
        fold_data = df[df["fold"] == fold].sort_values("n_features")
        ax.plot(
            fold_data["n_features"],
            fold_data["cv_score"],
            "o-",
            color=colors[i],
            linewidth=1.5,
            markersize=4,
            alpha=0.7,
            label=f"Fold {fold}",
        )

        # Mark optimal point (max CV score)
        optimal_idx = fold_data["cv_score"].idxmax()
        optimal_row = fold_data.loc[optimal_idx]
        ax.scatter(
            [optimal_row["n_features"]],
            [optimal_row["cv_score"]],
            s=100,
            c=[colors[i]],
            marker="*",
            zorder=10,
            edgecolors="white",
            linewidths=1,
        )

    # Aggregate mean curve across folds
    mean_curve = df.groupby("n_features")["cv_score"].agg(["mean", "std"]).reset_index()
    ax.plot(
        mean_curve["n_features"],
        mean_curve["mean"],
        "k--",
        linewidth=2.5,
        alpha=0.8,
        label="Mean across folds",
    )

    # Add shaded error region
    ax.fill_between(
        mean_curve["n_features"],
        mean_curve["mean"] - mean_curve["std"],
        mean_curve["mean"] + mean_curve["std"],
        color="gray",
        alpha=0.2,
    )

    # Styling
    ax.set_xlabel("Number of Features", fontsize=11)
    ax.set_ylabel("CV AUROC (Internal)", fontsize=11)
    ax.set_xlim(0, df["n_features"].max() * 1.05)

    # Y-axis: reasonable range
    y_min = df["cv_score"].min() - 0.02
    y_max = df["cv_score"].max() + 0.02
    y_min = max(0.5, y_min)
    y_max = min(1.0, y_max)
    ax.set_ylim(y_min, y_max)

    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Title
    if model_name:
        ax.set_title(f"{title}\n{model_name}", fontsize=12, fontweight="bold")
    else:
        ax.set_title(title, fontsize=12, fontweight="bold")

    # Summary text
    optimal_sizes = []
    for fold in folds:
        fold_data = df[df["fold"] == fold]
        optimal_n = fold_data.loc[fold_data["cv_score"].idxmax(), "n_features"]
        optimal_sizes.append(int(optimal_n))

    summary_text = (
        f"Folds: {n_folds}\n"
        f"Mean optimal size: {np.mean(optimal_sizes):.1f} ± {np.std(optimal_sizes):.1f}\n"
        f"Range: [{min(optimal_sizes)}, {max(optimal_sizes)}]"
    )
    ax.text(
        0.98,
        0.02,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "gray", "pad": 4},
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
