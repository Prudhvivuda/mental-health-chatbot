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