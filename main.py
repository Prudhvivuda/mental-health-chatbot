import json
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
import joblib


## Load Fine-Tuned RoBERTa Emotion Model
emotion_model = RobertaForSequenceClassification.from_pretrained("saved_roberta_emotion_model")
emotion_tokenizer = RobertaTokenizer.from_pretrained("saved_roberta_emotion_model")
mlb_classes = joblib.load("emotion_labels.pkl")

print("Loaded fine-tuned RoBERTa emotion classifier.")
## Load Sentence-BERT for Intent Detection
with open("KB.json", "r") as f:
    kb = json.load(f)
print("Loadied intent patterns from KB.json.")

intent_patterns = []
intent_tags = []
for intent in kb["intents"]:
    tag = intent["tag"]
    for pattern in intent.get("patterns", []):
        if pattern.strip():
            intent_patterns.append(pattern.strip())
            intent_tags.append(tag)

print("Loading Sentence-BERT model for intent classification...")
intent_encoder = SentenceTransformer("all-MiniLM-L6-v2")
intent_embeddings = intent_encoder.encode(intent_patterns)


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

def detect_intent(text, top_k=3):
    """Use Sentence-BERT + cosine similarity to detect closest intent."""
    input_embedding = intent_encoder.encode([text])
    sims = cosine_similarity(input_embedding, intent_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    top_matches = [(intent_tags[i], intent_patterns[i], sims[i]) for i in top_indices]
    print("Top intent matches:")
    # for tag, pattern, score in top_matches:
    #     print(f" - {tag}: '{pattern}' ({score:.2f})")
    return top_matches[0][0]

## Generate response from llama3
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def query_llama(prompt):
    try:
        # Send the request to Ollama
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        # Parse the response
        if response.status_code == 200:
            result = response.json()
            response_text = result["response"].strip()
            # Remove extra line spaces
            cleaned_response = ' '.join(response_text.splitlines())
            return cleaned_response 
        else:
            print(f"Error: {response.text}")
            return None
        
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None
    
## Pipeline: Detect intent + emotion → generate response using KB and LLaMA3.
def chat(user_input, history=[]):
    emotions = detect_emotion_roberta(user_input)
    intent = detect_intent(user_input)
    
    
    print(f"Detected Emotion(s): {emotions}")
    print(f"Detected Intent: {intent}")

    # Try rule-based KB response first
    kb_response = None
    for entry in kb["intents"]:
        if entry["tag"] == intent and entry.get("responses"):
            kb_response = entry["responses"]
            break
    
    if history:
        prompt = f"Please act as a mental health chatbot. The user says {user_input} and the predicted emotions from roberta model are {emotions} and the predicted intent from sentence bert is {intent}. The user's previous messages are {history}. Please respond in a supportive and empathetic manner."
    else:
        prompt = f"Please act as a mental health chatbot. The user says {user_input} and the predicted emotions from roberta model are {emotions} and the predicted intent from sentence bert is {intent}. Please respond in a supportive and empathetic manner."

    # Add to history
    history.append(user_input)
    
    if kb_response:
        selected_response = kb_response[0] if isinstance(kb_response, list) else kb_response
        bot_response = query_llama(prompt)
        return f"{selected_response}{bot_response}"
    else:
        return query_llama(prompt)
    
## Let's chat with the Bot
if __name__ == "__main__":
    print("Welcome to the Mental Health Chatbot 💬 \nPlease enter your prompt \nType 'quit' to exit\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Bot: Take care! 💙")
            break
        response = chat(user_input)
        print(f"Bot: {response}\n")