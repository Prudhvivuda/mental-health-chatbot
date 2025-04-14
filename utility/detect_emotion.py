import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
import joblib

emotion_model = RobertaForSequenceClassification.from_pretrained("saved_roberta_emotion_model-v1")
emotion_tokenizer = RobertaTokenizer.from_pretrained("saved_roberta_emotion_model-v1")
mlb_classes = joblib.load("emotion_labels.pkl")


## Define Utility Functions
def detect_emotion_roberta(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    # print("Raw Emotion Probabilities:")
    # for i, p in enumerate(probs):
    #     print(f"{mlb_classes[i]}: {p:.3f}")

    threshold = 0.35  # Lowered threshold to capture more subtle emotions
    top_labels = [mlb_classes[i] for i, p in enumerate(probs) if p > threshold]

    if not top_labels:
        top_labels = [mlb_classes[int(np.argmax(probs))]]  # fallback if no emotion passes threshold

    return top_labels  # fallback if no score > 0.5