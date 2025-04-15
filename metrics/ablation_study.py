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
        "I’m really sad today",
        # ---------------------------- #
        "I don’t know how to cope with stress anymore",
        "I’m so happy I got a promotion!",
        "Can you help me feel less overwhelmed?",
        "Everything feels pointless lately",
        "Hey, how’s it going?",
        "I’m scared about my upcoming exam",
        "I just need someone to listen to me",
        "I feel like I’m failing at everything",
        "Life has been great lately, just wanted to share",
        "I get so angry and I don’t know why",
        "Can you suggest ways to relax?",
        "I’m lonely, no one seems to care",
        "Good morning! Feeling okay today",
        "I’m worried about my family all the time",
        "I don’t feel like myself anymore",
        "Any tips for staying motivated?",
        "I had a rough day, just need to vent"
    ],
    "expected_emotion": [["anxious"], ["neutral"], ["sad"], ["anxious"], ["happy"],
        ["anxious"], ["sad"], ["content"], ["anxious"], ["caring"],
        ["disappointed"], ["content"], ["angry"], ["content"], ["lonely"],
        ["content"], ["anxious"], ["lonely"], ["confident"], ["sad"]],
    
    "expected_intent": ["problem", "greeting", "sad", "stressed", "problem",
        "stressed", "depressed", "happy", "problem", "understand",
        "worthless", "happy", "venting", "meditation", "sad",
        "morning", "scared", "change", "seeking", "stressed"]
})


# Run ablation study
ablation_study(test_data)