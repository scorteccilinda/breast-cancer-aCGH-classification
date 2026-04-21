import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def generate_genomic_heatmap(call_path, clinical_path):
    # 1. Load data
    df_calls = pd.read_csv(call_path, sep='\t')
    df_clinical = pd.read_csv(clinical_path, sep='\t')

    # 2. Sort probes by Chromosome and Start position
    # Reset index is required so we can accurately calculate the Y-axis ticks later
    df_calls = df_calls.sort_values(['Chromosome', 'Start']).reset_index(drop=True)
    
    # Calculate positions for chromosome labels on the Y-axis
    chr_ticks = []
    chr_labels = []
    for c in df_calls['Chromosome'].unique():
        indices = df_calls.index[df_calls['Chromosome'] == c]
        # FIX: Replaced .mean() with sum/len for compatibility across all pandas versions
        chr_ticks.append(sum(indices) / len(indices))
        chr_labels.append(f"Chr {c}")

    # 3. Create a Chromosome Sidebar
    chr_colors = ['#2c3e50', '#95a5a6'] * 12 
    chr_map = {chr_num: chr_colors[i] for i, chr_num in enumerate(df_calls['Chromosome'].unique())}
    row_colors = df_calls['Chromosome'].map(chr_map)

    # 4. Prepare the Matrix
    matrix = df_calls.iloc[:, 4:] 

    # 5. Clinical Subtype Sidebar Mapping
    subtype_colors = {'HR+': '#9b59b6', 'HER2+': '#3498db', 'Triple Neg': '#e67e22'}
    df_clinical.set_index('Sample', inplace=True)
    col_colors = df_clinical['Subgroup'].map(subtype_colors).reindex(matrix.columns)

    # 6. Professional Genomic Colors
    genomic_palette = ['#e74c3c', '#000000', '#27ae60', '#f1c40f'] # Red, Black, Green, Yellow
    discrete_cmap = ListedColormap(genomic_palette)

    # 7. Generate Clustermap
    g = sns.clustermap(
        matrix, 
        row_cluster=False,           
        col_cluster=True,            
        method='average',            # Linkage from the paper
        metric='correlation',        # Metric from the paper
        row_colors=row_colors,       
        col_colors=col_colors,       
        cmap=discrete_cmap,
        vmin=-1, vmax=2,             # FORCE mapping: -1=Red, 0=Black, 1=Green, 2=Yellow
        figsize=(20, 15),
        xticklabels=False,
        yticklabels=False,
        cbar_pos=None
    )

    # 8. Set Chromosomes on the Vertical Axis
    g.ax_heatmap.set_yticks(chr_ticks)
    g.ax_heatmap.set_yticklabels(chr_labels, rotation=0, fontsize=9)
    g.ax_heatmap.set_ylabel("Genomic Location (Chromosome 1 - X)")

    # 9. Add Legends
    # Subtype Legend
    subtype_patches = [Patch(facecolor=c, label=l) for l, c in subtype_colors.items()]
    legend1 = plt.legend(handles=subtype_patches, title="Clinical Subgroups", 
                         bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    g.ax_heatmap.add_artist(legend1)

    # Genomic State Legend
    state_labels = ['Loss (-1)', 'Normal (0)', 'Gain (1)', 'Amplification (2)']
    state_patches = [Patch(facecolor=genomic_palette[i], label=state_labels[i]) for i in range(4)]
    plt.legend(handles=state_patches, title="Genomic State", 
               bbox_to_anchor=(1.05, 0.8), loc='upper left', frameon=False)

    plt.suptitle("Genomic Portrait (Average Linkage + Pearson Correlation)", y=1.02, fontsize=14)
    plt.savefig('High_Res_Genomic_Portrait.pdf', format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate_genomic_heatmap('Train_call.tsv', 'Train_clinical.tsv')