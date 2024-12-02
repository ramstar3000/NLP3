import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Example data
batch_sizes = [4000, 2000, 1000]
metrics = ["Homogeneity", "Completeness", "V-Measure"]


# Fine-grained means (Homogeneity, Completeness, V-Measure) for each batch size
fine_grained_means = {
    "Homogeneity": [0.4117, 0.4075, 0.4069],
    "Completeness": [0.4669, 0.4694, 0.4634],
    "V-Measure": [0.4376, 0.4362, 0.4333]
}

# Coarse-grained means (Homogeneity, Completeness, V-Measure) for each batch size
coarse_grained_means = {
    "Homogeneity": [0.3335, 0.3263, 0.3315],
    "Completeness": [0.3323, 0.3286, 0.3296],
    "V-Measure": [0.3327, 0.3273, 0.3303]
}

fine_grained_variances = {
    "Homogeneity": [0.0001, 0.0000, 0.0002],
    "Completeness": [0.0001, 0.0001, 0.0005],
    "V-Measure": [0.0001, 0.0000, 0.0003]
}

# Coarse-grained variances (Homogeneity, Completeness, V-Measure) for each batch size
coarse_grained_variances = {
    "Homogeneity": [0.0003, 0.0004, 0.0003],
    "Completeness": [0.0007, 0.0005, 0.0005],
    "V-Measure": [0.0005, 0.0004, 0.0003]
}


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, means, variances, label_type in zip(axes, 
                                             [fine_grained_means, coarse_grained_means], 
                                             [fine_grained_variances, coarse_grained_variances], 
                                             ["XPOS", "UPOS"]):
    x = np.arange(len(batch_sizes))  # X-axis for batch sizes
    width = 0.25  # Width of each bar
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, [m[i] for m in means.values()], yerr=[v[i] for v in variances.values()], 
               width=width, label=metric, capsize=5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(batch_sizes)
    ax.set_title(f"{label_type} Results", fontsize=14)
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(title="Metric")
    ax.grid(visible=True, linestyle="--", alpha=0.5)

    ax.set_ylim(0.30, 0.5)
    

plt.tight_layout()
plt.show()