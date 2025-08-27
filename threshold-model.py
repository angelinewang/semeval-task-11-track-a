import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams
import re
import spacy
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, classification_report

# Create necessary directories only if they don't exist
model_checkpoints_dir = './model_checkpoints'
results_dir = './results'

# Get absolute paths
current_dir = os.getcwd()
model_checkpoints_abs = os.path.join(current_dir, 'model_checkpoints')
results_abs = os.path.join(current_dir, 'results')

print(f"Current working directory: {current_dir}")
print(f"Model checkpoints directory: {model_checkpoints_abs}")
print(f"Results directory: {results_abs}")

if not os.path.exists(model_checkpoints_dir):
    os.makedirs(model_checkpoints_dir)
    print(f"Created model_checkpoints directory at: {model_checkpoints_abs}")
else:
    print(f"model_checkpoints directory already exists at: {model_checkpoints_abs}")

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory at: {results_abs}")
else:
    print(f"results directory already exists at: {results_abs}")

# Verify directories exist
if os.path.exists(model_checkpoints_dir):
    print(f"✓ model_checkpoints directory verified: {os.path.abspath(model_checkpoints_dir)}")
else:
    print(f"✗ ERROR: model_checkpoints directory not found!")
    
if os.path.exists(results_dir):
    print(f"✓ results directory verified: {os.path.abspath(results_dir)}")
else:
    print(f"✗ ERROR: results directory not found!")

# Load data
train = pd.read_csv('./final-datasets/track_a/train/eng.csv')
val = pd.read_csv('./final-datasets/track_a/dev/eng.csv')
test = pd.read_csv('./final-datasets/track_a/test/eng.csv')

# Define emotions and mapping (model's labels to our labels)
emotions = ['joy', 'sadness', 'surprise', 'fear', 'anger']
emotion_mapping = {
    'joy': 'joy',
    'sadness': 'sadness',
    'surprise': 'surprise',
    'fear': 'fear',
    'anger': 'anger',
    'love': None  # We'll ignore this emotion as it's not in our target set
}

# Preprocessing Config
config = {
    'sep_pn': True, 
    'rm_pn': False, 
    'apply_lemmatization': True, 
    'apply_stemming': True, 
    'add_bigrams': True, 
    'rm_sw': False
}

# Load the pre-trained model and tokenizer
model_name = "bhadresh-savani/bert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Preprocess without emolex words
def pre_process(text, config, target_emotion, emotion_dict):
    def separate_punctuation(text):
        text = re.sub(r"(\w)([.,;:!?'\"”\)])", r"\1 \2", text)
        text = re.sub(r"([.,;:!?'\"“\(\)])(\w)", r"\1 \2", text)
        return text

    def remove_punctuation(text):
        text = re.sub(r"[.,;:!?'\"“”\(\)]", "", text)
        return text

    def tokenize_text(text):
        encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
        return tokens

    def apply_stemming(tokens):
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

    def apply_lemmatization(tokens):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    def generate_ngrams_from_tokens(tokens, n):
        return [" ".join(gram) for gram in ngrams(tokens, n)]

    # Apply config options
    if config['sep_pn'] and not config['rm_pn']:
        text = separate_punctuation(text)
    if config['rm_pn'] and not config['sep_pn']:
        text = remove_punctuation(text)

    tokens = tokenize_text(text)
    if config['apply_stemming']:
        tokens = apply_stemming(tokens)
    if config['apply_lemmatization']:
        tokens = apply_lemmatization(tokens)
    if config['add_bigrams']:
        tokens += generate_ngrams_from_tokens(tokens, 2)
    if config['rm_sw']:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]

    processed_text = " ".join(tokens)

    return processed_text

# Specific emolex words preprocessing

emolex_path = "./NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

def load_emolex(emolex_path):
    emolex = pd.read_csv(emolex_path, sep='\t', header=None, names=["Word", "Emotion", "Association"])
    # Filter to only include our target emotions
    emolex = emolex[emolex['Emotion'].isin(emotions)]
    
    emotion_dict = {}
    for _, row in emolex.iterrows():
        if row["Association"] == 1:  # Only positive associations
            word = row["Word"].lower()  # Convert to lowercase
            emotion = row["Emotion"].lower()
            if word not in emotion_dict:
                emotion_dict[word] = []
            emotion_dict[word].append(emotion)
    
    print(f"Loaded {len(emotion_dict)} words from EmoLex")
    return emotion_dict

emotion_dict = load_emolex(emolex_path)

nlp = spacy.load("en_core_web_sm")

# Initialize emotion_matches globally before preprocessing
emotion_matches = {emotion: 0 for emotion in emotions}

for emotion in emotions:
    emotion_words = [word for word, emots in emotion_dict.items() if emotion in emots]
    print(f"\n{emotion} words in lexicon: {len(emotion_words)}")
    print("Sample words:", emotion_words[:5])

# Specific emolex words preprocessing
def preprocess(dataset, emotions, config, emotion_dict):
    global emotion_matches
    emotion_matches = {emotion: 0 for emotion in emotions}  # Reset counters
    
    texts = []
    for text in tqdm(dataset):
        # Get all EmoLex words for any emotion
        text_lower = text.lower()
        text_words = set(text_lower.split())
        emolex_words = []
        
        # Check for words matching any emotion in the lexicon
        for word in text_words:
            if word in emotion_dict:
                matching_emotions = set(emotion_dict[word]) & set(emotions)
                if matching_emotions:
                    emolex_words.append(word)
                    # Update emotion matches counters
                    for emotion in matching_emotions:
                        emotion_matches[emotion] += 1
        
        # Process text normally
        processed = pre_process(text, config, target_emotion=None, emotion_dict=None)
        
        # Append EmoLex words if found
        if emolex_words:
            processed = f"{processed} [SEP] {' '.join(emolex_words)}"
        else:
            processed = f"{processed} [SEP]"
            
        texts.append(processed)
    
    # Print emotion word statistics
    for emotion in emotions:
        print(f"Texts with {emotion}-specific words: {emotion_matches[emotion]}/{len(texts)}")
    
    return texts

# Use the preprocess function correctly with all required arguments
train_texts = preprocess(train["text"], emotions, config, emotion_dict)
val_texts = preprocess(val["text"], emotions, config, emotion_dict)
test_texts = preprocess(test["text"], emotions, config, emotion_dict)
            
# Fit the vectorizer on the training data and transform all sets
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts).toarray()
X_val = vectorizer.transform(val_texts).toarray()
X_test = vectorizer.transform(test_texts).toarray()

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

# Convert labels to numpy arrays
y_train = train[emotions].values
y_val = val[emotions].values
y_test = test[emotions].values

# Get BERT embeddings for training data
print("Getting training BERT predictions...")
train_raw_preds = get_predictions_batch(train['text'].tolist())

print("Getting validation BERT predictions...")
val_raw_preds = get_predictions_batch(val['text'].tolist())

print("Getting test BERT predictions...")
test_raw_preds = get_predictions_batch(test['text'].tolist())

train_bert_features = format_predictions(train_raw_preds, emotion_mapping)
val_bert_features = format_predictions(val_raw_preds, emotion_mapping)
test_bert_features = format_predictions(test_raw_preds, emotion_mapping)
    
final_train_features = np.hstack([
    X_train,
    train_bert_features
])

final_val_features = np.hstack([
    X_val,
    val_bert_features
])

final_test_features = np.hstack([
    X_test,
    test_bert_features
])

input_size = final_train_features.shape[1]  # Get the feature size for all emotions
# Creates 5 separate models
models = {}
for emotion in emotions:
    models[emotion] = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 1)  # Output: 1 value per emotion
    )

# Training loop for each emotion
for emotion in emotions:
    print(f"\nTraining model for {emotion}")
    model = models[emotion]
    model.train()  # Add this to explicitly set training mode
    
    # Prepare data for this emotion
    features_tensor = torch.tensor(final_train_features, dtype=torch.float32)
    labels_tensor = torch.tensor(y_train[:, emotions.index(emotion)], dtype=torch.float32).unsqueeze(1)
    
    # Add validation tensors
    val_features_tensor = torch.tensor(final_val_features, dtype=torch.float32)
    val_labels_tensor = torch.tensor(y_val[:, emotions.index(emotion)], dtype=torch.float32).unsqueeze(1)
    print(f"Validation features shape for {emotion}: {val_features_tensor.shape}")
    
    dataset = TensorDataset(features_tensor, labels_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Increased batch size

    # Calculate class weights for this emotion
    pos_weight = (len(labels_tensor) - labels_tensor.sum()) / labels_tensor.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Better handling of class imbalance
    
    # Use Adam optimizer with learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    # Training loop
    best_f1 = 0
    patience = 10
    patience_counter = 0
    train_losses = []
    val_f1s = []
    
    for epoch in tqdm(range(401), desc=f"Training Loop ({emotion})"):
        # Training phase
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for features, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        avg_train_loss = epoch_loss / batch_count
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features_tensor)
            val_preds = (torch.sigmoid(val_outputs).squeeze() > 0.5).float()
            val_labels = val_labels_tensor.squeeze()
            
            # Calculate metrics
            val_loss = criterion(val_outputs, val_labels_tensor)
            macro_f1 = f1_score(
                val_labels.cpu().numpy(),
                val_preds.cpu().numpy(),
                average='macro',
                zero_division=0
            )

        # Learning rate scheduling
        scheduler.step(macro_f1)

        if epoch % 10 == 0:  # More frequent logging
            print(f'\nEpoch {epoch} ({emotion}):')
            print(f'Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            print(f'Macro F1: {macro_f1:.4f}')
            train_losses.append(avg_train_loss)
            val_f1s.append(macro_f1)
        
        # Save best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            
            # Ensure model_checkpoints directory exists before saving
            if not os.path.exists('./model_checkpoints'):
                print("Creating model_checkpoints directory before saving...")
                os.makedirs('./model_checkpoints')
            
            save_path = f'./model_checkpoints/threshold-best-{emotion}-model.pth'
            print(f"Saving model to: {os.path.abspath(save_path)}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'f1': macro_f1
            }, save_path)
            print(f"\nNew best model for {emotion} at epoch {epoch} with F1: {macro_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Load best model for this emotion
    checkpoint = torch.load(f'./model_checkpoints/threshold-best-{emotion}-model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model for {emotion} from epoch {checkpoint['epoch']} with F1: {checkpoint['f1']:.4f}")

    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.title(f'Training Loss for {emotion}')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(val_f1s)), val_f1s, label='Validation F1')
    plt.title(f'Validation Macro F1 for {emotion}')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Macro F1')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

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
# After training loop finishes for all emotions...

# Make predictions for each emotion using best models
# print("\nGenerating predictions for {emotion}...")
val_preds = {}
test_preds = {}

for emotion in emotions:
    print(f"\nGenerating predictions for {emotion}...")
    # Load the best model for this emotion
    model = models[emotion]
    checkpoint = torch.load(f'./model_checkpoints/threshold-best-{emotion}-model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    print(f"Using best {emotion} model from epoch {checkpoint['epoch']} with F1: {checkpoint['f1']:.4f}")
    
    with torch.no_grad():
        # Get raw probabilities using sigmoid
        val_outputs = model(torch.tensor(final_val_features, dtype=torch.float32))
        val_probs = torch.sigmoid(val_outputs).numpy()
        val_preds[emotion] = val_probs
        
        test_outputs = model(torch.tensor(final_test_features, dtype=torch.float32))
        test_probs = torch.sigmoid(test_outputs).numpy()
        test_preds[emotion] = test_probs

# Combine predictions
val_probs = np.hstack([val_preds[emotion] for emotion in emotions])
test_probs = np.hstack([test_preds[emotion] for emotion in emotions])

# Initialize threshold optimization
best_thresholds = {emotion: 0 for emotion in emotions}
best_binary_f1s = {emotion: 0 for emotion in emotions}

# Perform threshold grid search for each emotion
for i, emotion in enumerate(emotions):
    print(f"\nPerforming threshold grid search for {emotion}...")
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []

    for threshold in tqdm(thresholds, desc=f"Testing thresholds for {emotion}"):
        # Convert probabilities to binary predictions for this emotion
        val_binary_preds = (val_probs[:, i] > threshold).astype(int)
        
        # Calculate metrics
        binary_f1 = f1_score(val[emotions].values[:, i], val_binary_preds, average='binary')
        
        results.append({
            'threshold': threshold,
            'binary_f1': binary_f1
        })
        
        # Update best threshold if we found better F1
        if binary_f1 > best_binary_f1s[emotion]:
            best_binary_f1s[emotion] = binary_f1
            best_thresholds[emotion] = threshold

    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results)
    print(f"\nBest Results for {emotion}:")
    print(f"\nBest Results for Emotion Specific MLPs on BERT predictions:")
    print(f"Best Threshold: {best_thresholds[emotion]:.4f}")
    print(f"Best Binary F1: {best_binary_f1s[emotion]:.3f}")
    # print(f"Best Macro F1: {best_macro_f1s[emotion]:.3f}")

    # Plot the threshold search results for this emotion
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['binary_f1'], label='Binary F1')
    plt.axvline(x=best_thresholds[emotion], color='r', linestyle='--', 
                label=f'Best threshold ({best_thresholds[emotion]:.4f})')
    plt.xlabel('Threshold')
    plt.ylabel('Binary F1 Score')
    plt.title(f'F1 Score vs. Threshold for {emotion}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Use optimized thresholds for final predictions
val_binary_preds = np.zeros_like(val_probs, dtype=int)
test_binary_preds = np.zeros_like(test_probs, dtype=int)

for i, emotion in enumerate(emotions):
    val_binary_preds[:, i] = (val_probs[:, i] > best_thresholds[emotion]).astype(int)
    test_binary_preds[:, i] = (test_probs[:, i] > best_thresholds[emotion]).astype(int)

# Print detailed performance report
print("\nFinal Validation Performance with Emotion-Specific Thresholds and Emotion Specific MLPs:")

print(classification_report(
    val[emotions].values,
    val_binary_preds,
    target_names=emotions,
    digits=4
))

print("\nFinal Test Performance with Emotion-Specific Thresholds and Emotion Specific MLPs:")
print(classification_report(
    test[emotions].values,
    test_binary_preds,
    target_names=emotions,
    digits=4
))


# Save predictions
timestamp = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

# Ensure results directory exists before saving
if not os.path.exists('./results'):
    print("Creating results directory before saving predictions...")
    os.makedirs('./results')

# Save validation predictions
val_save_path = f'./results/threshold-val-{timestamp}.csv'
print(f"Saving validation predictions to: {os.path.abspath(val_save_path)}")
save_predictions(
    val_binary_preds,
    val['id'],
    val_save_path
)

# Save test predictions
test_save_path = f'./results/threshold-test-{timestamp}.csv'
print(f"Saving test predictions to: {os.path.abspath(test_save_path)}")
save_predictions(
    test_binary_preds,
    test['id'],
    test_save_path
)

# Print per-emotion performance
print("\nPer-emotion Validation Performance at Emotion-Specific Thresholds and Emotion Specific MLPs:")
for i, emotion in enumerate(emotions):
    emotion_f1 = f1_score(
        val[emotions].values[:, i],
        val_binary_preds[:, i],
        average='macro'  # Changed from 'binary' to 'macro'
    )
    print(f"{emotion}: F1 = {emotion_f1:.4f}")

print("\nPer-emotion Test Performance at Emotion-Specific Thresholds and Emotion Specific MLPs:")
for i, emotion in enumerate(emotions):
    emotion_f1 = f1_score(
        test[emotions].values[:, i],
        test_binary_preds[:, i],
        average='macro'  # Changed from 'binary' to 'macro'
    )
    print(f"{emotion}: F1 = {emotion_f1:.4f}")
