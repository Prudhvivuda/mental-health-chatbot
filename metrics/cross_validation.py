import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import sys,os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from metrics.empathy import *
from metrics.relevance_score import *
from utility.chat import *

# Example test dataset (replace with your own)
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
    "expected_emotion": [["anxious"], ["joyful"], ["sad"], ["anxious"], ["proud"],
        ["anxious"], ["sad"], ["content"], ["anxious"], ["caring"],
        ["disappointed"], ["content"], ["angry"], ["content"], ["lonely"],
        ["content"], ["anxious"], ["lonely"], ["confident"], ["sad"]],
    
    "expected_intent": ["problem", "greeting", "sad", "stressed", "problem",
        "stressed", "depressed", "happy", "problem", "understand",
        "worthless", "happy", "venting", "meditation", "sad",
        "morning", "scared", "change", "seeking", "stressed"]
})

def evaluate_cross_validation(test_data):
    emotion_correct = 0
    intent_correct = 0
    empathy_scores = []
    relevance_scores = []
    
    for _, row in test_data.iterrows():
        user_input = row["user_input"]
        expected_emotion = row["expected_emotion"]
        expected_intent = row["expected_intent"]
        
        # Run pipeline
        emotions = detect_emotion_roberta(user_input)
        intent = detect_intent(user_input)
        response = chat(user_input)
        
        # Evaluate emotions (allow partial matches)
        emotion_match = any(emo in expected_emotion for emo in emotions)
        # emotion_match = set(expected_emotion).intersection(set(list(emotions)))
        emotion_correct += 1 if emotion_match else 0
        
        # Evaluate intent
        intent_match = intent == expected_intent
        # intent_match = set(expected_intent).intersection(set(list(intent)))
        intent_correct += 1 if intent_match else 0
        
        # Compute empathy and relevance
        empathy = calculate_empathy_score(response, emotions)
        relevance = calculate_relevance_score(user_input, response, intent, intent_encoder)
        empathy_scores.append(empathy)
        relevance_scores.append(relevance)
    
    # Compute metrics
    emotion_accuracy = emotion_correct / len(test_data)
    intent_accuracy = intent_correct / len(test_data)
    avg_empathy = sum(empathy_scores) / len(empathy_scores)
    avg_relevance = sum(relevance_scores) / len(relevance_scores)
    
    print(f"Emotion Detection Accuracy: {emotion_accuracy:.2f}")
    print(f"Intent Detection Accuracy: {intent_accuracy:.2f}")
    print(f"Average Empathy Score: {avg_empathy:.2f}")
    print(f"Average Relevance Score: {avg_relevance:.2f}")

# Run evaluation
evaluate_cross_validation(test_data)