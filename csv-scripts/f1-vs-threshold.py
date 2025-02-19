import matplotlib.pyplot as plt
import numpy as np

# Data for plotting with more granular thresholds
thresholds = np.linspace(0.1, 0.8, num=15)

# Define peaks (threshold, score) for each emotion - these are the correct values
peaks = {
    'anger': (0.650, 0.535),  # Peak at 0.650
    'fear': (0.350, 0.74),    # Peak at 0.350
    'joy': (0.500, 0.578),    # Peak at 0.500
    'sadness': (0.600, 0.715), # Peak at 0.600
    'surprise': (0.400, 0.58)  # Peak at 0.400
}

# Data points carefully matched to each plot's curve
anger_scores = [0.48, 0.45, 0.43, 0.46, 0.48, 0.485, 0.53, 0.485, 0.485, 0.485, 0.52, 0.535, 0.52, 0.48, 0.40]

fear_scores = [0.70, 0.71, 0.715, 0.72, 0.73, 0.74, 0.735, 0.73, 0.72, 0.70, 0.65, 0.55, 0.40, 0.25, 0.15]

joy_scores = [0.565, 0.555, 0.545, 0.54, 0.545, 0.55, 0.565, 0.57, 0.578, 0.56, 0.52, 0.52, 0.52, 0.50, 0.46]

# Adjusting sadness and surprise data points to ensure curves pass through peak points
# sadness_scores = [0.59, 0.64, 0.67, 0.69, 0.70, 0.685, 0.71, 0.715, 0.715, 0.715, 0.70, 0.68, 0.67, 0.67, 0.67]

surprise_scores = [0.43, 0.46, 0.48, 0.52, 0.55, 0.58, 0.58, 0.58, 0.57, 0.55, 0.45, 0.35, 0.25, 0.15, 0.07]

# Ensure peaks are included in the interpolation points
thresholds_with_peaks = np.sort(np.unique(np.concatenate([
    thresholds,
    [0.400],  # Surprise peak
    [0.600]   # Sadness peak
])))

# Recompute interpolation with peaks included
surprise_scores = np.interp(thresholds_with_peaks, np.linspace(0.1, 0.8, num=15), surprise_scores)

# Create the plot
fig, ax = plt.subplots(figsize=(15, 12))

# Setting the plot appearance
ax.set_facecolor('#f0f0f0')
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'grid.color': 'gray',
    'grid.linestyle': '--',
    'grid.linewidth': 1.5
})

# Plot each emotion line
ax.plot(thresholds, anger_scores, label='Anger', color='red', linewidth=10)
ax.plot(thresholds, fear_scores, label='Fear', color='blue', linewidth=10)
ax.plot(thresholds, joy_scores, label='Joy', color='green', linewidth=10)
ax.plot(thresholds_with_peaks, surprise_scores, label='Surprise', color='orange', linewidth=10)

# Add peak points with larger dots
ax.scatter(0.650, 0.535, color='red', alpha=0.9, s=1300, zorder=5,
               edgecolor='black', linewidth=4)  # Anger peak
ax.scatter(0.350, 0.74, color='blue', alpha=0.9, s=1300, zorder=5,
               edgecolor='black', linewidth=4)  # Fear peak
ax.scatter(0.500, 0.578, color='green', alpha=0.9, s=1300, zorder=5,
               edgecolor='black', linewidth=4)  # Joy peak
ax.scatter(0.600, 0.715, color='purple', alpha=0.9, s=1300, zorder=5,
               edgecolor='black', linewidth=4)  # Sadness peak
ax.scatter(0.400, 0.58, color='orange', alpha=0.9, s=1300, zorder=5,
               edgecolor='black', linewidth=4)  # Surprise peak

# Customize the plot
ax.set_xlabel('Threshold', fontweight='bold', fontsize=46)
ax.set_ylabel('F1 Score', fontweight='bold', fontsize=46)
ax.grid(True, alpha=0.7, linestyle='--', color='gray', linewidth=1.5)

# Update tick label sizes and set fewer ticks with adjusted spacing
plt.xticks(np.arange(0.1, 0.9, 0.2), fontsize=55)
plt.yticks(np.arange(0.4, 0.8, 0.1), fontsize=55)

# Adjust layout to prevent overlap
ax.yaxis.set_label_coords(-0.1, 0.5)  # Move y-axis label further left
ax.tick_params(axis='both', pad=15)    # Add padding to tick labels

# Set axis limits to match the original plots
ax.set_xlim(0.1, 0.8)
ax.set_ylim(0.4, 0.75)

# Create exact points for sadness curve
sadness_x = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.600, 0.65, 0.7, 0.75, 0.8]
sadness_y = [0.59, 0.64, 0.68, 0.70, 0.685, 0.715, 0.715, 0.715, 0.70, 0.68, 0.67, 0.67]

# Plot sadness as a direct line between points
ax.plot(sadness_x, sadness_y, color='purple', linewidth=10, label='Sadness')

# Add peak point
ax.scatter(0.600, 0.715, color='purple', alpha=0.9, s=1300, zorder=5,
               edgecolor='black', linewidth=4)  # Sadness peak

plt.tight_layout()
plt.savefig('combined_emotion_thresholds_plot.png', dpi=300, bbox_inches='tight')
plt.show()