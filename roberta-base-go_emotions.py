from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# set path to data
PTH = '...'

# Enable GPU
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load the pre-trained GoEmotions model
classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    truncation=True,
    device=device,
    max_length=512,
    top_k=None  # Returns scores for all 28 emotion labels
)


df = pd.read_csv(PTH,low_memory=False)

df = df[~df['posts.comments.text'].isna()]

# Extract comments
comments = df["posts.comments.text"].fillna("").tolist()

# Function to process emotions from predictions
def extract_emotions(predictions, threshold=0.3):
    emotions = [pred["label"] for pred in predictions if pred["score"] > threshold]
    scores = [pred["score"] for pred in predictions if pred["score"] > threshold]
    return emotions, scores


# Process in batches with progress bar
batch_size = 32
all_emotions = []
all_scores = []


for i in tqdm(range(0, len(comments), batch_size), desc="Processing batches"):
    batch = comments[i:i+batch_size]
    
    # Batch prediction
    predictions_batch = classifier(batch)
    
    # Extract emotions for each comment in batch
    for predictions in predictions_batch:
        emotions, scores = extract_emotions(predictions, threshold=0.3)
        all_emotions.append(emotions)
        all_scores.append(scores)
    
    # Save checkpoint every 20,000 comments
    if (i + batch_size) % 20000 == 0:
        checkpoint_df = df.iloc[:len(all_emotions)].copy()
        checkpoint_df["emotions"] = all_emotions
        checkpoint_df["emotion_scores"] = all_scores
        checkpoint_df.to_csv(f"checkpoint_{i}.csv", index=False)
        print(f"Checkpoint saved at {i} comments")

# Add results to dataframe
df["emotions"] = all_emotions
df["emotion_scores"] = all_scores

# Save final results
df.to_csv("comments_with_emotions.csv", index=False)
print("Processing complete!")
print(df.head())
