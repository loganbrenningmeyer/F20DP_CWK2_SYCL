import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
data_path = 'SYCLData.csv'
sycl_data = pd.read_csv(data_path)

# Display the first few rows of the dataframe to understand its structure
sycl_data.head()


# Prepare data for plotting
plot_data = sycl_data.pivot(index='Upper', columns='Method', values='Median')

# Create the bar plot
fig, ax = plt.subplots(figsize=(10, 6))

# Setting the positions and width for the bars
positions = np.arange(len(plot_data.index))
width = 0.35

# Plotting both methods
ax.bar(positions - width/2, plot_data['A'], width, label='Array Method', color='blue')
ax.bar(positions + width/2, plot_data['R'], width, label='Reduction Method', color='#0da344')

# Adding some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Upper Amount')
ax.set_ylabel('Median Execution Time (s)')
ax.set_title('Parallel Median Execution Time Comparison (SYCL)')
ax.set_xticks(positions)
ax.set_xticklabels(plot_data.index)
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('SYCLPlots.png')

# SPEEDUP

# Define the mean sequential times as provided
sequential_times = {
    15000: 12.62,
    30000: 51.59,
    100000: 633.61
}

# Calculate the mean parallel time for each Upper amount and Method
mean_parallel_times = sycl_data.groupby(['Upper', 'Method'])['Mean'].mean().unstack()



# Calculate speedup: Sequential Time / Parallel Time
speedup = pd.DataFrame()

# Assuming mean_parallel_times[method] is a pandas Series or similar

for method in mean_parallel_times.columns:
    # Convert dict_values to a list and ensure alignment with input sizes
    seq_times_list = [sequential_times[upper] for upper in mean_parallel_times.index]
    # Perform element-wise division using numpy arrays for proper alignment
    speedup[method] = np.array(seq_times_list) / mean_parallel_times[method].values


import matplotlib.pyplot as plt
import numpy as np

# Core count
core_count = 2496

# Calculate efficiency: Speedup / Core Count
efficiency = speedup / core_count

# Example plotting code for efficiency
fig, ax = plt.subplots(figsize=(10, 6))

# Assuming input_sizes are the same as before
input_sizes = [15000, 30000, 100000]

for method in efficiency.columns:
    ax.plot(input_sizes, efficiency[method], marker='o', label='Array Method' if method == 'A' else 'Reduction Method',
             color='blue' if method == 'A' else '#0da344')

ax.set_xlabel('Upper Amount')
ax.set_ylabel('Efficiency')
ax.set_title('Efficiency Comparison by Input Size and Method')
ax.set_xticks(input_sizes)
ax.set_xticklabels(input_sizes)
ax.legend()

plt.grid(True)
plt.savefig('SYCLEfficiency.png')

