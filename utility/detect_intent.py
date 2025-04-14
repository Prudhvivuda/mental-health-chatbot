import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


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

def detect_intent(text, top_k=3):
    """Use Sentence-BERT + cosine similarity to detect closest intent."""
    input_embedding = intent_encoder.encode([text])
    sims = cosine_similarity(input_embedding, intent_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    top_matches = [(intent_tags[i], intent_patterns[i], sims[i]) for i in top_indices]
    # for tag, pattern, score in top_matches:
    #     print(f" - {tag}: '{pattern}' ({score:.2f})")
    return top_matches[0][0]