from matplotlib import pyplot as plt
import seaborn as sns

# Multiline string containing V-measure score data
raw_data_upos = """0.18689497642613376
0.20416805615807782
0.2253414150618599
0.2440583790302962
0.25813092306167396
0.28108522319083645
0.307566559158756
0.3242379423243693
0.3290995334604186
0.3314607845633516
0.3360087568936561
"""

raw_data_xpos = """0.2338609162037913
0.2569613592293613
0.2748768916792237
0.27945620900122403
0.2714867438106776
0.2999318671571838
0.3522439296185276
"""

# Convert raw data to a list of floating-point numbers and create corresponding x values (iterations)
v_measure_scores_upos = [float(score) for score in raw_data_upos.splitlines()]
v_measure_scores_xpos = [float(score) for score in raw_data_xpos.splitlines()]
iterations = [i * 3 for i in range(len(v_measure_scores_upos))]
iterations_xpos = [i * 3 + 1 for i in range(len(v_measure_scores_xpos))]

# Set up the plot style
sns.set_style("whitegrid")
sns.set_palette("muted")

# Create the line plot with markers and a label for the legend
sns.lineplot(x=iterations, y=v_measure_scores_upos, marker="o", markersize=6, linewidth=2, label="UPOS")
sns.lineplot(x=iterations_xpos, y=v_measure_scores_xpos, marker="o", markersize=6, linewidth=2, label="XPOS")

# Add title and axis labels
plt.title("V-measure Score per Iteration | HMM", fontsize=16)
plt.xlabel("Iteration Number", fontsize=14)
plt.ylabel("V-measure Score", fontsize=14)

# Show the plot with customized figure size
plt.figure(figsize=(10, 6))
plt.legend(title="Model", fontsize=12)
plt.show()
