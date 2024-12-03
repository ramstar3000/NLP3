from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

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
0.34103010749131657
"""

raw_data_xpos1 = """0.2338609162037913
0.2569613592293613
0.2748768916792237
0.27945620900122403
0.2714867438106776
0.2999318671571838
0.3522439296185276
0.38705391593061467
0.4073427265938311
0.4155556305758223
0.4187432772540536
"""

raw_data_xpos2 = """0.230979501901489
0.26596428509654535
0.29876391922093526
0.30475310501080644
0.3133434615265795
0.3496141351507055
0.3813299263388876
0.396311951958484
0.3937118992958243
"""


# Convert raw data to a list of floating-point numbers and create corresponding x values (iterations)
v_measure_scores_upos = [float(score) for score in raw_data_upos.splitlines()]
v_measure_scores_xpos1 = [float(score) for score in raw_data_xpos1.splitlines()]
v_measure_scores_xpos2 = [float(score) for score in raw_data_xpos2.splitlines()]

v_measure_scores_xpos1 = np.array(v_measure_scores_xpos1)
v_measure_scores_xpos2 = np.array(v_measure_scores_xpos2)

# Add the arrays and divide by 2 to get the average, pad the arrays to the same length
v_measure_scores_xpos = np.zeros(max(len(v_measure_scores_xpos1), len(v_measure_scores_xpos2)))
v_measure_scores_xpos = v_measure_scores_xpos1
v_measure_scores_xpos[:len(v_measure_scores_xpos2)] += v_measure_scores_xpos2
v_measure_scores_xpos[:len(v_measure_scores_xpos2)] /= 2


v_measure_scores_xpos = [ (v_measure_scores_xpos1[i] + v_measure_scores_xpos2[i]) / 2 for i in range(len(v_measure_scores_xpos2))]

iterations = [i * 3  + 1 for i in range(len(v_measure_scores_upos))]
iterations_xpos = [i * 3 + 2 for i in range(len(v_measure_scores_xpos))]
outdir = "graphing/hmm.png"
# Set up the plot style
sns.set_style("whitegrid")
sns.set_palette("muted")

# Show the plot with customized figure size
# plt.figure(figsize=(10, 6))
plt.legend(title="Model", fontsize=12)

# Create the line plot with markers and a label for the legend
sns.lineplot(x=iterations, y=v_measure_scores_upos, marker="o", markersize=6, linewidth=2, label="UPOS")
sns.lineplot(x=iterations_xpos, y=v_measure_scores_xpos, marker="o", markersize=6, linewidth=2, label="XPOS")


# Add title and axis labels
plt.title("V-measure Score per Iteration | HMM", fontsize=16)
plt.xlabel("Iteration Number", fontsize=14)
plt.ylabel("V-measure Score", fontsize=14)

# plt.savefig(outdir)

plt.show()
