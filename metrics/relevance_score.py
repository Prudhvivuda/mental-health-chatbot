from sklearn.metrics.pairwise import cosine_similarity

def calculate_relevance_score(user_input, response, intent, intent_encoder):
    # Encode user input and response
    embeddings = intent_encoder.encode([user_input, response])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Boost score if response aligns with intent
    intent_keywords = {
        "seeking_advice": ["suggest", "try", "consider"],
        "venting": ["understand", "feel", "tough"],
        "greeting": ["hi", "hello", "welcome"]
        # Add more intents from your KB.json
    }
    
    intent_boost = 0
    if intent in intent_keywords:
        if any(keyword in response.lower() for keyword in intent_keywords[intent]):
            intent_boost = 0.2
    
    relevance_score = similarity + intent_boost
    relevance_score = min(max(relevance_score, 0), 1)  # Normalize to 0-1
    return relevance_score
