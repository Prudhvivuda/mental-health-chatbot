from utility.detect_emotion import *
from utility.detect_intent import *
from utility.llama3 import *

## Pipeline: Detect intent + emotion → generate response using KB and LLaMA3.
def chat(user_input, history=[]):
    emotions = detect_emotion_roberta(user_input)
    intent = detect_intent(user_input)
    
    
    print(f"Detected Emotion(s): {emotions}")
    print(f"Detected Intent: {intent}")

    # Try rule-based KB response first
    # kb_response = None
    # for entry in kb["intents"]:
    #     if entry["tag"] == intent and entry.get("responses"):
    #         kb_response = entry["responses"]
    #         break
    
    if history:
        prompt = f"Please act as a mental health chatbot. The user says {user_input} and the predicted emotions from roberta model are {emotions} and the predicted intent from sentence bert is {intent}. The user's previous messages are {history}. Please respond in a supportive and empathetic manner."
    else:
        prompt = f"Please act as a mental health chatbot. The user says {user_input} and the predicted emotions from roberta model are {emotions} and the predicted intent from sentence bert is {intent}. Please respond in a supportive and empathetic manner."

    # Add to history
    history.append(user_input)
    
    # if kb_response:
    #     selected_response = kb_response[0] if isinstance(kb_response, list) else kb_response
    #     bot_response = query_llama(prompt)
    #     return f"{selected_response}{bot_response}"
    # else:
    return query_llama(prompt)