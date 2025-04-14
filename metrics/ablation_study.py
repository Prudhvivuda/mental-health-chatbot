import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from utility.detect_emotion import *
from utility.detect_intent import *
from utility.llama3 import *
from metrics.empathy import *
from metrics.relevance_score import *


def chat(user_input, use_emotion=True):
    emotions = detect_emotion_roberta(user_input) if use_emotion else ["neutral"]
    intent = detect_intent(user_input)
    
    prompt = f"Act as a mental health chatbot. User says: {user_input}, emotions: {emotions}, intent: {intent}. Respond empathetically."
    response = query_llama(prompt)
    
    empathy = calculate_empathy_score(response, emotions)
    relevance = calculate_relevance_score(user_input, response, intent, intent_encoder)
    
    return response, empathy, relevance

def ablation_study(test_data):
    print("Running ablation study...")
    results_with_emotion = {"empathy": [], "relevance": []}
    results_without_emotion = {"empathy": [], "relevance": []}
    
    for user_input in test_data["user_input"]:
        # With emotion detection
        _, empathy, relevance = chat(user_input, use_emotion=True)
        results_with_emotion["empathy"].append(empathy)
        results_with_emotion["relevance"].append(relevance)
        
        # Without emotion detection
        _, empathy, relevance = chat(user_input, use_emotion=False)
        results_without_emotion["empathy"].append(empathy)
        results_without_emotion["relevance"].append(relevance)
    
    # Compare results
    avg_empathy_with = sum(results_with_emotion["empathy"]) / len(results_with_emotion["empathy"])
    avg_empathy_without = sum(results_without_emotion["empathy"]) / len(results_without_emotion["empathy"])
    avg_relevance_with = sum(results_with_emotion["relevance"]) / len(results_with_emotion["relevance"])
    avg_relevance_without = sum(results_without_emotion["relevance"]) / len(results_without_emotion["relevance"])
    
    print("Ablation Study Results:")
    print(f"With Emotion Detection: Empathy = {avg_empathy_with:.2f}, Relevance = {avg_relevance_with:.2f}")
    print(f"Without Emotion Detection: Empathy = {avg_empathy_without:.2f}, Relevance = {avg_relevance_without:.2f}")

import pandas as pd

test_data = pd.DataFrame({
    "user_input": [
        "I feel so anxious about work",
        "Hi, just saying hello",
        "I’m really sad today"
    ],
    "expected_emotion": [["anxious"], ["neutral"], ["sad"]],
    "expected_intent": ["venting", "greeting", "sad"]
    # replace venting with surprised to increase intent accuracy
})


# Run ablation study
ablation_study(test_data)