import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import numpy as np
from scipy import stats

# Load the data
train = pd.read_csv('/Users/angwang/semeval-task-11-track-b/public_data_test/track_a/train/eng.csv')
val = pd.read_csv('/Users/angwang/semeval-task-11-track-b/public_data_test/track_a/dev/eng.csv')
test = pd.read_csv('/Users/angwang/semeval-task-11-track-b/public_data_test/track_a/test/eng.csv')

# Define emotions
emotions = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# Define the optimal thresholds found (from your notebook)
optimal_thresholds = {
    'joy': 0.500,      # These values are approximated from your plots
    'sadness': 0.600,  # You may want to replace these with the exact values
    'surprise': 0.400,
    'fear': 0.350,
    'anger': 0.650
}

# Calculate proportion for each emotion in each dataset
def calculate_proportions(df):
    proportions = {}
    for emotion in emotions:
        proportions[emotion] = df[emotion].mean()
    return proportions

train_props = calculate_proportions(train)
val_props = calculate_proportions(val)
test_props = calculate_proportions(test)

# Create the visualization
plt.figure(figsize=(15, 12))

# Create color mapping for emotions with display names
emotion_colors = {
    'anger': 'red',
    'fear': 'blue',
    'joy': 'green',
    'sadness': 'purple',
    'surprise': 'orange'
}

# Plot points first
for emotion in emotions:
    plt.scatter(train_props[emotion], optimal_thresholds[emotion],
               c=emotion_colors[emotion], alpha=0.9, s=1300,
               edgecolor='black', linewidth=2)

# Get the axes object and its limits AFTER plotting points
ax = plt.gca()
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

# Now add labels with proper boundary awareness
for emotion in emotions:
    x = train_props[emotion]
    y = optimal_thresholds[emotion]
    
    # Calculate offset based on position in the plot
    # Default offset starts with label to the right and slightly above
    offset_x = 70
    offset_y = 20
    
    # Adjust offsets based on position
    if x > 0.8 * xmax:  # Points near right edge
        offset_x = -150  # Move label left
    if y > 0.8 * ymax:  # Points near top
        offset_y = -50   # Move label below
    if y < 0.2 * ymax:  # Points near bottom
        offset_y = 70    # Move label higher up
        
    # Capitalize only for display
    display_emotion = emotion.capitalize()
    plt.annotate(display_emotion,
                (x, y), 
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                fontsize=60,
                fontweight='bold',
                color=emotion_colors[emotion],
                path_effects=[path_effects.withStroke(linewidth=7, foreground='white')])

# Update limits to ensure all labels are visible
plt.margins(x=0.15, y=0.15)  # Add some padding around the data points

plt.xlabel('Proportion of Data with Emotion', fontsize=46, fontweight='bold')
plt.ylabel('Optimal Threshold', fontsize=46, fontweight='bold')
plt.grid(True, alpha=0.7, linestyle='--', color='gray', linewidth=1.5)
plt.gca().set_facecolor('#f0f0f0')

# Set fewer ticks on both axes
ax.set_xticks(np.linspace(0.1, 0.6, 6))  # 6 ticks from 0.1 to 0.6
ax.set_yticks(np.linspace(0.3, 0.7, 5))  # 5 ticks from 0.3 to 0.7

# Format tick labels to show only 2 decimal places and remove trailing zeros
ax.set_xticklabels([f'{x:.2f}'.rstrip('0').rstrip('.') for x in ax.get_xticks()])
ax.set_yticklabels([f'{y:.2f}'.rstrip('0').rstrip('.') for y in ax.get_yticks()])

# Increase tick label size
plt.xticks(fontsize=55)
plt.yticks(fontsize=55)

# Add padding to prevent text cutoff
plt.tight_layout(pad=2.0)

plt.savefig('/Users/angwang/semeval-task-11-track-b/results/train_emotion_threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print correlation analysis only for training set
print("\nCorrelation Analysis for Training Set:")
x = [train_props[emotion] for emotion in emotions]
y = [optimal_thresholds[emotion] for emotion in emotions]
correlation, p_value = stats.pearsonr(x, y)
print(f"Correlation coefficient between proportion and threshold: {correlation:.3f}")
print(f"P-value: {p_value:.3f}")
print(f"Statistical significance: {'Significant' if p_value < 0.05 else 'Not significant'} at p<0.05")
