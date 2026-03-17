#!/usr/bin/env python3
"""Generate publication-quality figures from finetuning metrics and results."""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style and colors
plt.style.use('dark_background')
cmap = plt.cm.tab10
colors = [cmap(i) for i in range(10)]

# Load data
with open('paper/results/finetune_metrics.json', 'r') as f:
    metrics = json.load(f)

results_df = pd.read_csv('paper/results/results.csv')

# Map subset names to sizes and colors
subset_info = {
    'genz_1k': (1, colors[0]),
    'genz_5k': (5, colors[1]),
    'genz_20k': (20, colors[2]),
    'genz_50k': (50, colors[3]),
    'genz_100k': (100, colors[4]),
    'genz_200k': (200, colors[5]),
    'genz_500k': (500, colors[6]),
    'genz_1000k': (1000, colors[7]),
}

# Figure 1: Forgetting Curves
fig1, ax1 = plt.subplots(figsize=(10, 6))

for subset_name, (subset_size, color) in subset_info.items():
    data = metrics[subset_name]
    steps = [d['step'] for d in data]
    wiki_ppl = [d['wiki_ppl'] for d in data]
    ax1.semilogy(steps, wiki_ppl, marker='o', markersize=5, linewidth=2.5,
                 label=f'{subset_size}k tokens', color=color)

# Add baseline reference line
baseline_ppl = 258.2547
ax1.axhline(y=baseline_ppl, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')

ax1.set_xlabel('Fine-tuning Steps', fontsize=12, fontweight='bold')
ax1.set_ylabel('WikiText Perplexity', fontsize=12, fontweight='bold')
ax1.set_title('Catastrophic Forgetting During Style Fine-tuning',
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(title='Fine-tuning Set Size', fontsize=10, title_fontsize=11, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=10)

# Add caption
fig1.text(0.5, 0.02, '17M parameter GPT, pretrained on 1B tokens of FineWeb-Edu',
          ha='center', fontsize=9, style='italic', alpha=0.7)

plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig('paper/results/forgetting_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: forgetting_curves.png")
plt.close()

# Figure 2: Style-Forgetting Tradeoff
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plot points for all non-baseline subsets
subset_sizes = []
delta_styles = []
delta_ppls = []

for idx, row in results_df.iterrows():
    if row['subset_k'] != 'baseline':
        subset_k = int(row['subset_k'])
        subset_sizes.append(subset_k)
        delta_styles.append(row['delta_style'])
        delta_ppls.append(row['delta_ppl'])

# Sort by subset size for consistent coloring
sorted_indices = np.argsort(subset_sizes)
subset_sizes_sorted = [subset_sizes[i] for i in sorted_indices]
delta_styles_sorted = [delta_styles[i] for i in sorted_indices]
delta_ppls_sorted = [delta_ppls[i] for i in sorted_indices]

# Plot with log scale for y-axis
for i, idx in enumerate(sorted_indices):
    size = subset_sizes[idx]
    subset_name = f'genz_{size}k'
    color = subset_info[subset_name][1]
    ax2.scatter(delta_styles[idx], delta_ppls[idx], s=200, alpha=0.7,
               color=color, edgecolors='white', linewidth=1.5, zorder=3)
    ax2.annotate(f'{size}k', (delta_styles[idx], delta_ppls[idx]),
                xytext=(5, 5), textcoords='offset points', fontsize=9,
                fontweight='bold')

ax2.set_xscale('linear')
ax2.set_yscale('log')
ax2.set_xlabel('Style Shift (Δ)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Perplexity Increase (Δ)', fontsize=12, fontweight='bold')
ax2.set_title('Style Acquisition vs. Fluency Loss',
              fontsize=14, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='--', which='both')
ax2.tick_params(labelsize=10)

# Add caption
fig2.text(0.5, 0.02, '17M parameter GPT, pretrained on 1B tokens of FineWeb-Edu',
          ha='center', fontsize=9, style='italic', alpha=0.7)

plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig('paper/results/tradeoff.png', dpi=300, bbox_inches='tight')
print("✓ Saved: tradeoff.png")
plt.close()

print("\n✅ All figures generated successfully!")