import matplotlib.pyplot as plt
import numpy as np

# Data for plotting with more granular thresholds
thresholds = np.linspace(0.1, 0.8, num=15)

# Define peaks (threshold, score) - exactly matching plots
peaks = {
    'joy': (0.450, 0.654),
    'sadness': (0.550, 0.711),
    'surprise': (0.200, 0.649),
    'fear': (0.500, 0.746),
    'anger': (0.600, 0.649)
}

# Data points exactly matching each plot's curve - adjusted to maintain peak until x=0.6
joy_scores = [0.57, 0.58, 0.59, 0.62, 0.64, 0.64, 0.65, 0.654, 0.654, 0.654, 0.654, 0.55, 0.49, 0.45, 0.42]

# Remove this duplicate array since we use the more precise sadness_x and sadness_y later
sadness_scores = [0.674, 0.674, 0.682, 0.682, 0.707, 0.707, 0.707, 0.71, 0.711, 0.711, 0.676, 0.676, 0.676, 0.695, 0.687]

surprise_scores = [0.61, 0.63, 0.649, 0.61, 0.60, 0.59, 0.585, 0.615, 0.615, 0.58, 0.56, 0.575, 0.575, 0.53, 0.51]

fear_scores = [0.70, 0.71, 0.72, 0.72, 0.72, 0.72, 0.73, 0.74, 0.746, 0.70, 0.65, 0.50, 0.35, 0.25, 0.15]

anger_scores = [0.35, 0.40, 0.42, 0.48, 0.52, 0.58, 0.58, 0.62, 0.63, 0.64, 0.649, 0.59, 0.59, 0.62, 0.52]

# Ensure peaks are included in the interpolation points
thresholds_with_peaks = np.sort(np.unique(np.concatenate([
    thresholds,
    [0.400],  # Surprise peak
    [0.600]   # Sadness peak
])))

# Recompute interpolation with peaks included
# surprise_scores = np.interp(thresholds_with_peaks, np.linspace(0.1, 0.8, num=15), surprise_scores)

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

# Add peak points with larger dots - corrected to match actual curve peaks
ax.scatter(0.600, 0.649, color='red', alpha=0.9, s=1300, zorder=5,
          edgecolor='black', linewidth=4)  # Anger peak - matches curve peak
ax.scatter(0.500, 0.746, color='blue', alpha=0.9, s=1300, zorder=5,
          edgecolor='black', linewidth=4)  # Fear peak - matches curve peak
ax.scatter(0.450, 0.654, color='green', alpha=0.9, s=1300, zorder=5,
          edgecolor='black', linewidth=4)  # Joy peak - matches curve peak
ax.scatter(0.550, 0.711, color='purple', alpha=0.9, s=1300, zorder=5,
          edgecolor='black', linewidth=4)  # Sadness peak - matches curve peak
ax.scatter(0.200, 0.649, color='orange', alpha=0.9, s=1300, zorder=5,
          edgecolor='black', linewidth=4)  # Surprise peak - matches curve peak

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
ax.set_ylim(0.4, 0.78)

# Create exact points for sadness curve
sadness_x = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.600, 0.65, 0.7, 0.75, 0.8]
sadness_y = [0.674, 0.674, 0.682, 0.682, 0.707, 0.707, 0.711, 0.711, 0.676, 0.676, 0.676, 0.687]

# Plot sadness as a direct line between points
ax.plot(sadness_x, sadness_y, color='purple', linewidth=10, label='Sadness')

plt.tight_layout()
plt.savefig('combined_emotion_thresholds_plot.png', dpi=300, bbox_inches='tight')
plt.show()