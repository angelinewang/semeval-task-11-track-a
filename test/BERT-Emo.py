from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load the pre-trained model and tokenizer
model_name = "bhadresh-savani/bert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create the emotion classifier pipeline
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

def get_predictions_batch(texts, batch_size=32):
    """
    Get predictions for a list of texts in batches
    """
    all_predictions = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i + batch_size]
        predictions = classifier(batch_texts)
        all_predictions.extend(predictions)
    
    return all_predictions

def format_predictions(predictions, emotion_mapping):
    """
    Convert raw predictions to the required format
    """
    formatted_preds = []
    for pred in predictions:
        # Create a dictionary of emotion scores
        scores = {p['label']: p['score'] for p in pred}
        # Map to our required emotions and format
        row = [scores.get(emotion_mapping.get(e, e), 0) for e in emotions]
        formatted_preds.append(row)
    
    return np.array(formatted_preds)

# Load data
train = pd.read_csv('../public_data_test/track_a/train/eng.csv')
val = pd.read_csv('../public_data_test/track_a/dev/eng.csv')
test = pd.read_csv('../public_data_test/track_a/test/eng.csv')

# Define emotions and mapping (model's labels to our labels)
emotions = ['Joy', 'Sadness', 'Surprise', 'Fear', 'Anger']
emotion_mapping = {
    'joy': 'Joy',
    'sadness': 'Sadness',
    'surprise': 'Surprise',
    'fear': 'Fear',
    'anger': 'Anger',
    'love': None  # We'll ignore this emotion as it's not in our target set
}

# Get predictions
print("Getting validation predictions...")
val_raw_preds = get_predictions_batch(val['text'].tolist())
val_predictions = format_predictions(val_raw_preds, emotion_mapping)

print("Getting test predictions...")
test_raw_preds = get_predictions_batch(test['text'].tolist())
test_predictions = format_predictions(test_raw_preds, emotion_mapping)

# Convert probabilities to binary predictions (you can adjust the threshold)
threshold = 0.5
val_binary_preds = (val_predictions > threshold).astype(int)
test_binary_preds = (test_predictions > threshold).astype(int)

# Evaluate validation predictions
if 'Joy' in val.columns:
    from sklearn.metrics import classification_report
    
    print("\nValidation Set Performance:")
    print(classification_report(
        val[emotions].values,
        val_binary_preds,
        target_names=emotions
    ))

# Save predictions
def save_predictions(predictions, ids, filename):
    df_predictions = pd.DataFrame(predictions, columns=emotions)
    df_predictions['id'] = ids
    df_predictions = df_predictions[['id'] + emotions]
    df_predictions.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")

# Save validation and test predictions
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

save_predictions(
    val_binary_preds,
    val['id'],
    f'results/val_predictions_{timestamp}.csv'
)

save_predictions(
    test_binary_preds,
    test['id'],
    f'results/test_predictions_{timestamp}.csv'
)