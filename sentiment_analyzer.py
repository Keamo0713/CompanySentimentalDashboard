import json
import requests
import streamlit as st
import time # Import the time module for delays

# --- Gemini API Configuration ---
# Your Gemini API Key is provided here.
GEMINI_API_KEY = "AIzaSyArNd3AV-Iclw46t1eMd7aOY8q3HQMCA9I"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- API Rate Limit Delay (in seconds) ---
# This delay helps prevent hitting the 'Too Many Requests' (429) error,
# which can occur if you make too many requests in a short period on the free tier.
# You might need to adjust this value based on Gemini API's actual rate limits
# and your usage patterns.
API_CALL_DELAY = 0.5 # Half a second delay between consecutive API calls

def get_sentiment(text_to_analyze: str):
    """
    Analyzes the sentiment of the given text using the Gemini 2.0 Flash API.
    Returns a dictionary with 'sentiment' (Positive/Negative/Neutral) and 'confidence'.
    """
    if not GEMINI_API_KEY:
        st.error("Gemini API Key is not set. Please provide it in sentiment_analyzer.py to use the service.")
        return {"sentiment": "Error", "confidence": 0.0}

    headers = {
        'Content-Type': 'application/json',
    }

    # Prompt engineering to get sentiment and a confidence score
    prompt = f"""
    Analyze the overall sentiment of the following text and categorize it as 'Positive', 'Negative', or 'Neutral'.
    Also, provide a numerical confidence score for your classification between 0.0 and 1.0 (float).
    Return the output as a JSON object with 'sentiment' and 'confidence' fields.

    Text: "{text_to_analyze}"

    Example JSON response:
    ```json
    {{
      "sentiment": "Positive",
      "confidence": 0.92
    }}
    ```
    """

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]

    # Structured response schema
    generation_config = {
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "OBJECT",
            "properties": {
                "sentiment": {"type": "STRING", "enum": ["Positive", "Negative", "Neutral"]},
                "confidence": {"type": "NUMBER"}
            },
            "required": ["sentiment", "confidence"]
        }
    }

    payload = {
        "contents": chat_history,
        "generationConfig": generation_config
    }

    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                                 headers=headers,
                                 json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        result = response.json()

        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            response_text = result['candidates'][0]['content']['parts'][0]['text']
            sentiment_data = json.loads(response_text)

            sentiment = sentiment_data.get('sentiment')
            confidence = sentiment_data.get('confidence')

            # Basic validation for confidence score
            if not (0.0 <= confidence <= 1.0):
                st.warning(f"Gemini returned an out-of-range confidence score for sentiment: {confidence}. Clamping to [0, 1].")
                confidence = max(0.0, min(1.0, confidence))

            return {"sentiment": sentiment, "confidence": confidence}
        else:
            st.error(f"Gemini API (Sentiment) response structure is unexpected or empty. Full response: {result}")
            return {"sentiment": "Error", "confidence": 0.0}
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Gemini API for sentiment analysis: {e}. Check your API key and internet connection.")
        return {"sentiment": "Error", "confidence": 0.0}
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from Gemini API for sentiment analysis: {e}. Raw response: {response.text}")
        return {"sentiment": "Error", "confidence": 0.0}
    except Exception as e:
        st.error(f"An unexpected error occurred during sentiment analysis: {e}")
        return {"sentiment": "Error", "confidence": 0.0}
    finally:
        # Always wait after an API call to respect rate limits, especially crucial for batch processing
        time.sleep(API_CALL_DELAY)

def extract_keywords(text: str):
    """
    Extracts key phrases/keywords that *drive the sentiment* from the given text
    using the Gemini 2.0 Flash API. Returns a list of strings.
    """
    if not GEMINI_API_KEY:
        # Error will be shown by get_sentiment, no need to duplicate here
        return []

    headers = {
        'Content-Type': 'application/json',
    }

    # Refined prompt to explicitly ask for sentiment-driving keywords
    prompt = f"""
    From the following text, extract up to 5 keywords or short phrases that *specifically drive or indicate its overall sentiment*.
    Focus on words or phrases that directly convey the emotional tone (positive, negative, or neutral).
    Return the output as a JSON object with a single field 'sentiment_keywords', which is an array of strings.

    Text: "{text}"

    Example JSON response for positive text:
    ```json
    {{
      "sentiment_keywords": ["fantastic product", "so happy", "quality is great"]
    }}
    ```
    Example JSON response for negative text:
    ```json
    {{
      "sentiment_keywords": ["terrible service", "very disappointed", "long wait"]
    }}
    ```
    """

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]

    # Structured response schema
    generation_config = {
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "OBJECT",
            "properties": {
                "sentiment_keywords": {"type": "ARRAY", "items": {"type": "STRING"}}
            },
            "required": ["sentiment_keywords"]
        }
    }

    payload = {
        "contents": chat_history,
        "generation_config": generation_config
    }

    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                                 headers=headers,
                                 json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses

        result = response.json()

        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            response_text = result['candidates'][0]['content']['parts'][0]['text']
            keywords_data = json.loads(response_text)
            return keywords_data.get('sentiment_keywords', [])
        else:
            st.error(f"Gemini API (Keywords) response structure is unexpected or empty. Full response: {result}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Gemini API for keyword extraction: {e}. Check your API key and internet connection.")
        return []
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from Gemini API for keyword extraction: {e}. Raw response: {response.text}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during keyword extraction: {e}")
        return []
    finally:
        # Always wait after an API call to respect rate limits
        time.sleep(API_CALL_DELAY)

