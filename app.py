import streamlit as st
from utility.chat import *  # Adjust import based on your file name
from metrics.empathy import calculate_empathy_score  # Adjust import based on your file structure
from utility.detect_intent import detect_intent
from utility.detect_emotion import detect_emotion_roberta

# Streamlit app configuration
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬", layout="wide")
# st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬", layout="centered")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# App title and description
st.title("💙 Mental Health Chatbot")
st.markdown("A supportive chatbot to listen and respond empathetically. Type your message below to start a conversation.")

# Chat input
user_input = st.text_input("You:", key="user_input", placeholder="How are you feeling today?")

# Process input and display response
if st.button("Send") and user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.history.append(user_input)
    
    # Get chatbot response, emotions, and intent
    try:
        response = chat(user_input, st.session_state.history[:-1])  # Pass history excluding current input
        emotions = detect_emotion_roberta(user_input)
        intent = detect_intent(user_input)
        empathy_score = calculate_empathy_score(response, emotions)
        
        # Add bot response to messages
        st.session_state.messages.append({
            "role": "bot",
            "content": response,
            "emotions": emotions,
            "intent": intent,
            "empathy_score": empathy_score
        })
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.messages.append({
            "role": "bot",
            "content": "Sorry, something went wrong. Please try again."
        })

# Display chat history
st.subheader("Conversation")
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(f"**You**: {message['content']}")
    else:
        with st.chat_message("assistant"):
            st.write(f"**Bot**: {message['content']}")
            # if "emotions" in message:
            #     st.write(f"**Detected Emotions**: {', '.join(message['emotions'])}")
            #     st.write(f"**Detected Intent**: {message['intent']}")
            #     st.write(f"**Empathy Score**: {message['empathy_score']:.2f}")

# Clear chat history button
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.history = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for mental health support. Type 'quit' to exit the conversation.")