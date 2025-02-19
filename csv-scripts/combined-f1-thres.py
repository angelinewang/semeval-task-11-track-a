import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create figure with the same styling as data-vs-thres.py
plt.figure(figsize=(15, 12))

# Define emotions and their colors (same as previous)
emotion_colors = {
    'anger': 'red',
    'fear': 'blue',
    'joy': 'green',
    'sadness': 'purple',
    'surprise': 'orange'
}

# Define the optimal thresholds (from the plots)
optimal_thresholds = {
    'anger': 0.650,
    'fear': 0.350,
    'joy': 0.500,
    'sadness': 0.600,
    'surprise': 0.400
}

# Create threshold range
thresholds = np.arange(0.1, 0.85, 0.01)

# Approximate the F1 curves from the plots
f1_scores = {
    'anger': [0.48, 0.44, 0.43, 0.44, 0.46, 0.48, 0.49, 0.53, 0.48, 0.48, 0.48, 0.52, 0.53, 0.51, 0.48, 0.46, 0.40],
    'fear': [0.70, 0.71, 0.72, 0.73, 0.74, 0.72, 0.70, 0.65, 0.62, 0.50, 0.40, 0.30, 0.15],
    'joy': [0.57, 0.55, 0.55, 0.54, 0.54, 0.56, 0.57, 0.58, 0.56, 0.52, 0.52, 0.48, 0.46],
    'sadness': [0.59, 0.64, 0.67, 0.69, 0.70, 0.71, 0.71, 0.72, 0.71, 0.70, 0.68, 0.67, 0.67],
    'surprise': [0.42, 0.45, 0.50, 0.55, 0.58, 0.58, 0.55, 0.45, 0.35, 0.25, 0.15, 0.10, 0.05]
}

# Plot each emotion
for emotion in emotion_colors:
    # Plot the F1 curve
    plt.plot(thresholds[:len(f1_scores[emotion])], f1_scores[emotion], 
             color=emotion_colors[emotion], linewidth=3, label=emotion.capitalize())
    
    # Add vertical line for optimal threshold
    plt.axvline(x=optimal_thresholds[emotion], color=emotion_colors[emotion], 
                linestyle='--', alpha=0.5)

# Styling (matching data-vs-thres.py)
plt.xlabel('Threshold', fontsize=34, fontweight='bold')
plt.ylabel('F1 Score', fontsize=34, fontweight='bold')
plt.title('F1 Scores vs. Threshold by Emotion', fontsize=38, pad=20, fontweight='bold')
plt.grid(True, alpha=0.7, linestyle='--', color='gray', linewidth=1.5)
plt.gca().set_facecolor('#f0f0f0')

# Set axis limits
plt.xlim(0.1, 0.85)
plt.ylim(0, 0.8)

# Increase tick label size
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Add legend with matching styling
plt.legend(fontsize=32, loc='upper right')

# Add padding to prevent text cutoff
plt.tight_layout(pad=2.0)

plt.savefig('combined_f1_threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.show() 