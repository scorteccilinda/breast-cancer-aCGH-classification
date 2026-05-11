import re
import pandas as pd

# load data
shap_df  = pd.read_csv("output/feature_importance_shap.csv")
gene_map = pd.read_csv("data/BasepairToGeneMap.tsv", sep="\t")

# parse region names
def parse_region(r):
    m = re.match(r"chr([0-9XY]+)_(\d+)_(\d+)", r)
    return pd.Series({
        "chrom": m.group(1),
        "region_start": int(m.group(2)),
        "region_end": int(m.group(3))
    })

shap_df = shap_df.join(shap_df["region"].apply(parse_region))

# ensure Chromosome types match
gene_map["Chromosome"] = gene_map["Chromosome"].astype(str)
shap_df["chrom"]       = shap_df["chrom"].astype(str)

# print all overlapping gene symbols without filtering
def find_genes_simple(row, top_n_genes=10):
    hits = gene_map[
        (gene_map["Chromosome"] == row["chrom"]) &
        (gene_map["Gene_start"] <= row["region_end"]) &
        (gene_map["Gene_end"]   >= row["region_start"])
    ]
    # Get all names without excluding non-coding/scaffold IDs
    names = hits["HGNC_symbol"].dropna().unique().tolist()
    
    if not names:
        return "No annotated genes"
    return ", ".join(names[:top_n_genes])

# apply functions
shap_df["genes"] = shap_df.apply(find_genes_simple, axis=1)
shap_df["gene_count"] = shap_df.apply(
    lambda r: len(gene_map[
        (gene_map["Chromosome"] == r["chrom"]) &
        (gene_map["Gene_start"] <= r["region_end"]) &
        (gene_map["Gene_end"]   >= r["region_start"])
    ]), axis=1)

# save and display
out = shap_df[["region", "mean_abs_shap", "gene_count", "genes"]]
out.to_csv("output/feature_importance_shap_annotated.csv", index=False)

print(out.head(20).to_string(index=False))