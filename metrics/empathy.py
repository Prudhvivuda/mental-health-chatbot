from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re, os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility.detect_emotion import *
from utility.detect_intent import *
from utility.llama3 import query_llama

def calculate_empathy_score(response, user_emotion):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(response)
    
    # Base score from positive sentiment
    empathy_score = sentiment["compound"]  # Range: -1 to 1
    
    # Boost score for empathetic phrases
    empathetic_phrases = [
        r"i'm here for you", r"i understand", r"that sounds", r"take care",
        r"you're not alone", r"it's okay to feel", r"let's work through"
    ]
    phrase_boost = 0.1 * sum(1 for phrase in empathetic_phrases if re.search(phrase, response.lower()))
    empathy_score += phrase_boost
    
    # Adjust score based on alignment with user emotion
    if any(emo in user_emotion for emo in ["sad", "anxious", "angry"]) and "sorry" in response.lower():
        empathy_score += 0.2  # Bonus for acknowledging negative emotions
    
    # Normalize to 0-1
    empathy_score = min(max(empathy_score, 0), 1)
    return empathy_score