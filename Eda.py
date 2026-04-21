import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import umap
import seaborn as sns
from matplotlib.patches import Patch

DATA_DIR   = "data"
OUTPUT_DIR = "output"

# subtype colours
SUBTYPE_COLORS = {"HER2+": "#e74c3c", "HR+": "#2ecc71", "Triple Neg": "#3498db"}

# one distinct colour per chromosome (1-23) using hsv colormap
CHROM_COLORS = {c: matplotlib.colormaps.get_cmap("nipy_spectral")(0.05 + (i / 23) * 0.75)
                for i, c in enumerate(range(1, 24))}

def get_chrom_array(columns):
    """Extract chromosome number from region names e.g. chr17_35076296_35282086 -> 17"""
    return np.array([int(r.split("_")[0].replace("chr", "")) for r in columns])

def get_chrom_positions(chrom_arr):
    """Return chromosome tick (midpoint) and boundary (start) positions for x-axis"""
    ticks  = {c: np.where(chrom_arr == c)[0].mean() for c in range(1, 24)}
    bounds = {c: np.where(chrom_arr == c)[0].min()  for c in range(1, 24)}
    return ticks, bounds


def prepare_data():
    """
    Load raw aCGH call data and clinical labels, reshape into ML-ready format and save.

    Reads:
        data/Train_call.tsv     
        data/Train_clinical.tsv 

    Writes:
        data/x_train.csv           
        data/y_train.csv          
        data/x_train_metadata.csv  -- chromosome, start, end, nclone per region
    """
    call     = pd.read_csv(f"{DATA_DIR}/Train_call.tsv",     sep="\t")
    clinical = pd.read_csv(f"{DATA_DIR}/Train_clinical.tsv", sep="\t")

    call[["Chromosome", "Start", "End", "Nclone"]].to_csv(
        f"{DATA_DIR}/x_train_metadata.csv", index=True)

    region_names  = [f"chr{row.Chromosome}_{row.Start}_{row.End}"
                     for _, row in call.iterrows()]
    sample_cols   = [c for c in call.columns if c.startswith("Array.")]
    x             = call[sample_cols].T
    x.columns     = region_names
    x.index.name  = "Sample"

    y = clinical.set_index("Sample")["Subgroup"]
    y = y.loc[x.index]

    x.to_csv(f"{DATA_DIR}/x_train.csv")
    y.to_csv(f"{DATA_DIR}/y_train.csv")

    print(f"x shape: {x.shape}")
    print(f"y counts:\n{y.value_counts()}")

def plot_dimensionality_reduction(x, y):
    """
    Reduce the 2834-dimensional feature matrix to 2D using PCA, t-SNE and UMAP,
    plotted side by side coloured by subtype.

    PCA  -- linear, axes show percentage of variance explained
    tSNE -- non-linear, good at revealing local cluster structure
    UMAP -- non-linear, preserves both local and some global structure

    Saves:
        output/dimensionality_reduction.png
    """
    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(x)
    ev    = pca.explained_variance_ratio_ * 100

    X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(x)
    X_umap = umap.UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(x)

    xlabels = [f"PC1 ({ev[0]:.1f}%)", "t-SNE 1", "UMAP 1"]
    ylabels = [f"PC2 ({ev[1]:.1f}%)", "t-SNE 2", "UMAP 2"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, coords, title, xl, yl in zip(axes,
                                          [X_pca, X_tsne, X_umap],
                                          ["PCA", "t-SNE", "UMAP"],
                                          xlabels, ylabels):
        for subtype, color in SUBTYPE_COLORS.items():
            mask = y == subtype
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=color, label=subtype, alpha=0.8, s=60)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xl,   fontsize=11)
        ax.set_ylabel(yl,   fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Dimensionality reduction of aCGH data (n=100)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dimensionality_reduction.png", dpi=150, bbox_inches="tight")
    plt.show()


def feature_selection(x, y):
    """
    Runs ANOVA and Kruskal-Wallis across all genomic regions to identify
    which regions differ significantly between the three breast cancer subtypes.

    ANOVA          -- tests if mean copy number differs across subtypes (F-statistic)
    Kruskal-Wallis -- non-parametric version, more appropriate for discrete values (H-statistic)

    Saves:
        output/feature_selection_anova.csv    -- regions ranked by ANOVA p-value
        output/feature_selection_kruskal.csv  -- regions ranked by Kruskal-Wallis p-value
        output/feature_selection.png          -- Manhattan plot coloured by chromosome
    """
    subtypes  = y.unique()
    chrom_arr = get_chrom_array(x.columns)
    results   = []

    for i, region in enumerate(x.columns):
        groups = [x.loc[y == s, region].values for s in subtypes]
        f_stat, p_anova   = stats.f_oneway(*groups)
        h_stat, p_kruskal = stats.kruskal(*groups)
        results.append({
            "region": region, "chrom": chrom_arr[i], "idx": i,
            "anova_p": p_anova, "f_stat": f_stat,
            "kruskal_p": p_kruskal, "h_stat": h_stat,
        })

    df         = pd.DataFrame(results)
    anova_df   = df.sort_values("anova_p")
    kruskal_df = df.sort_values("kruskal_p")

    anova_df.to_csv(f"{OUTPUT_DIR}/feature_selection_anova.csv",    index=False)
    kruskal_df.to_csv(f"{OUTPUT_DIR}/feature_selection_kruskal.csv", index=False)

    print(f"Regions with ANOVA p < 0.05:      {(df['anova_p'] < 0.05).sum()}")
    print(f"Regions with ANOVA p < 0.001:     {(df['anova_p'] < 0.001).sum()}")
    print(f"Regions with Kruskal p < 0.05:    {(df['kruskal_p'] < 0.05).sum()}")
    print(f"Regions with Kruskal p < 0.001:   {(df['kruskal_p'] < 0.001).sum()}")
    print(f"\nTop 10 regions by ANOVA:")
    print(anova_df.head(10)[["region", "anova_p", "f_stat"]])
    print(f"\nTop 10 regions by Kruskal-Wallis:")
    print(kruskal_df.head(10)[["region", "kruskal_p", "h_stat"]])
    print(f"\nSignificant regions per chromosome (ANOVA p < 0.05):")
    print((df[df["anova_p"] < 0.05].groupby("chrom").size()
           .reindex(range(1, 24), fill_value=0)).to_string())
    print(f"\nSignificant regions per chromosome (Kruskal-Wallis p < 0.05):")
    print((df[df["kruskal_p"] < 0.05].groupby("chrom").size()
           .reindex(range(1, 24), fill_value=0)).to_string())

    df["neg_log10_anova_p"]   = -np.log10(df["anova_p"])
    df["neg_log10_kruskal_p"] = -np.log10(df["kruskal_p"])

    chrom_ticks, chrom_bounds = get_chrom_positions(chrom_arr)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    for ax, y_col, ylabel, title in zip(
        axes,
        ["neg_log10_anova_p",  "neg_log10_kruskal_p"],
        ["-log10(p) ANOVA",    "-log10(p) Kruskal-Wallis"],
        ["ANOVA",              "Kruskal-Wallis"]
    ):
        for chrom, group in df.groupby("chrom"):
            sig = group[y_col] > -np.log10(0.05)
            ax.scatter(group.loc[~sig, "idx"], group.loc[~sig, y_col],
                       color=CHROM_COLORS[chrom], s=5, alpha=0.6)
            ax.scatter(group.loc[sig, "idx"], group.loc[sig, y_col],
                       color="#e74c3c", s=10, alpha=0.9)

        ax.axhline(-np.log10(0.05), color="black", linestyle="--",
                   linewidth=0.8, label="p = 0.05")
        for bound in chrom_bounds.values():
            ax.axvline(bound, color="lightgrey", linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title,   fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(False)

    axes[1].set_xticks(list(chrom_ticks.values()))
    axes[1].set_xticklabels(list(chrom_ticks.keys()), fontsize=9)
    axes[1].set_xlabel("Chromosome", fontsize=11)

    plt.suptitle("Feature selection: significant regions in red (p < 0.05)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_selection.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_copy_number_profiles(x, y):
    """
    Plots the mean copy number across all genomic regions for each breast cancer
    subtype ordered by chromosomal position. Shows which chromosomes are
    systematically gained or lost in each subtype.

    Saves:
        output/copy_number_profiles.png
    """
    chrom_arr             = get_chrom_array(x.columns)
    chrom_ticks, chrom_bounds = get_chrom_positions(chrom_arr)

    X_plot            = x.copy()
    X_plot["subtype"] = y
    mean_profiles     = X_plot.groupby("subtype").mean()

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    for ax, subtype in zip(axes, SUBTYPE_COLORS.keys()):
        profile = mean_profiles.loc[subtype].values

        for c in range(1, 24):
            idx = np.where(chrom_arr == c)[0]
            ax.fill_between(idx, 0, profile[idx], color=CHROM_COLORS[c], alpha=0.4)
            ax.plot(idx, profile[idx], color=SUBTYPE_COLORS[subtype], linewidth=0.8)
            ax.axvline(chrom_bounds[c], color="lightgray", linewidth=0.8)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Mean copy number", fontsize=10)
        ax.set_title(subtype, fontsize=12,
                     color=SUBTYPE_COLORS[subtype], fontweight="bold")
        ax.set_ylim(-1, 2)

    axes[-1].set_xticks(list(chrom_ticks.values()))
    axes[-1].set_xticklabels(list(chrom_ticks.keys()), fontsize=9)
    axes[-1].set_xlabel("Chromosome", fontsize=11)

    plt.suptitle("Mean copy number profile per breast cancer subtype", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/copy_number_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_hierarchical_clustering(x, y):
    """
    Hierarchical clustering heatmap of top 100 genomic regions by Kruskal-Wallis.
    Rows are genomic regions ordered by chromosomal position (no row clustering).
    Columns are samples clustered by similarity in copy number profile.
    Samples are coloured by subtype on top, regions by chromosome on the left.

    This shows whether samples naturally group by subtype based on copy number
    alone, without using the labels during clustering.

    Saves:
        output/hierarchical_clustering.png
    """

    # load top 100 regions from kruskal results
    kruskal_df = pd.read_csv(f"{OUTPUT_DIR}/feature_selection_kruskal.csv")
    top_regions = kruskal_df.head(100)["region"].tolist()

    # subset X to top regions, transpose so rows=regions, columns=samples
    X_top = X[top_regions].T   # shape (100 regions, 100 samples)

    # extract chromosome for each region for row colour bar
    chrom_arr = get_chrom_array(X_top.index)
    row_colors = pd.Series(
        [CHROM_COLORS[c] for c in chrom_arr],
        index=X_top.index
    )

    # sample subtype colour bar
    col_colors = y.map(SUBTYPE_COLORS)

    # discrete colormap: loss=blue, normal=lightgrey, gain=orange, amplification=red
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap   = ListedColormap(["#3498db", "lightgrey", "#e67e22", "#e74c3c"])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm   = BoundaryNorm(bounds, cmap.N)

    g = sns.clustermap(
        X_top,
        row_cluster  = False,         # keep chromosomal order on rows
        col_cluster  = True,          # cluster samples — this is the interesting part
        method       = "average",     # average linkage as in Perou 2000
        metric       = "correlation", # correlation distance as in Perou 2000
        row_colors   = row_colors,
        col_colors   = col_colors,
        cmap         = cmap,
        norm         = norm,
        figsize      = (16, 12),
        xticklabels  = False,
        yticklabels  = False,
        cbar_pos     = None,          # we add manual legend instead
    )

    # legends
    subtype_patches = [Patch(color=c, label=s)
                       for s, c in SUBTYPE_COLORS.items()]
    state_patches = [
        Patch(color="#3498db",   label="Loss (-1)"),
        Patch(color="lightgrey", label="Normal (0)"),
        Patch(color="#e67e22",   label="Gain (1)"),
        Patch(color="#e74c3c",   label="Amplification (2)"),
    ]

    g.ax_heatmap.legend(handles=subtype_patches, title="Subtype",
                        bbox_to_anchor=(1.15, 1.1), loc="upper left", frameon=False)
    g.fig.legend(handles=state_patches, title="Copy number",
                 bbox_to_anchor=(1.15, 0.7), loc="upper left", frameon=False)

    g.fig.suptitle(
        "Hierarchical clustering of top 100 regions\n"
        "Average linkage, correlation distance",
        y=1.02, fontsize=13
    )

    plt.savefig(f"{OUTPUT_DIR}/hierarchical_clustering.png",
                dpi=150, bbox_inches="tight")
    plt.show()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prepare_data()

    x = pd.read_csv(f"{DATA_DIR}/X_train.csv", index_col=0)
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv", index_col=0).squeeze()

    plot_dimensionality_reduction(x, y)
    feature_selection(x, y)
    plot_copy_number_profiles(x, y)
    plot_hierarchical_clustering(x, y)


if __name__ == "__main__":
    main()