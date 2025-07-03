import json
import requests # Still keeping requests for potential future flexibility, though google-generativeai handles most calls
import streamlit as st
import time
import os
import google.generativeai as genai # Import the Gemini library

# --- Gemini API Configuration ---
# Define Google Gemini API Key
GOOGLE_API_KEY = None
try:
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    elif os.getenv("GOOGLE_API_KEY"):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    else:
        st.error("Google Gemini API key not found. Please set it in `.streamlit/secrets.toml` or as an environment variable named `GOOGLE_API_KEY`.")
        st.stop() # Stop execution if API key is not found
except Exception as e:
    st.error(f"Error configuring Google Gemini API key: {e}.")
    st.stop()

# Configure the google.generativeai library
genai.configure(api_key=GOOGLE_API_KEY)

# Define the Gemini model to use
# Recommended models for text generation: 'gemini-pro' or 'gemini-1.5-flash'
# 'gemini-1.5-flash' is generally faster and cheaper for sentiment analysis.
# 'gemini-pro' is a stable general-purpose model.
GEMINI_MODEL_NAME = "gemini-2.0-flash" # Or "gemini-pro"

# --- API Rate Limit Delay (in seconds) ---
API_CALL_DELAY = 0.5 # Half a second delay between consecutive API calls

def _extract_json_from_markdown(text):
    """
    Extracts a JSON string from a markdown code block, or attempts to parse the whole text if no block.
    Prioritizes JSON within ```json ... ``` blocks.
    """
    # First, try to find a markdown JSON block
    start_index = text.find("```json")
    if start_index != -1:
        start_index += len("```json")
        end_index = text.find("```", start_index)
        if end_index != -1:
            json_string = text[start_index:end_index].strip()
            return json_string
        else:
            # If no closing ```, assume the rest is JSON
            json_string = text[start_index:].strip()
            return json_string
    
    # If no ```json block, try to clean and return the whole text as potential JSON
    cleaned_text = text.strip()
    # Handle cases where the model might just return a JSON string wrapped in an outer quote
    if cleaned_text.startswith('"') and cleaned_text.endswith('"') and len(cleaned_text) > 1:
        try:
            # Try to load it to confirm it's valid JSON within quotes
            json.loads(cleaned_text[1:-1])
            return cleaned_text[1:-1]
        except json.JSONDecodeError:
            pass # Not a quoted JSON string, proceed with original cleaned text

    return cleaned_text # Return the cleaned text, json.loads will handle validation


def get_sentiment(text_to_analyze: str):
    """
    Analyzes the sentiment of the given text using the Google Gemini API.
    Returns a dictionary with 'sentiment' (Positive/Negative/Neutral) and 'confidence').
    """
    if not GOOGLE_API_KEY:
        st.error("Google Gemini API Key is not set. Please provide it to use the service.")
        return {"sentiment": "Error", "confidence": 0.0}

    # Prompt engineering to get sentiment and a confidence score
    # Emphasize JSON output and no markdown
    prompt_template = """
    Analyze the overall sentiment of the following text and categorize it as 'Positive', 'Negative', or 'Neutral'.
    Also, provide a numerical confidence score for your classification between 0.0 and 1.0 (float).
    Return the output as a JSON object with 'sentiment' and 'confidence' fields. Ensure the output is *only* the JSON object, do NOT wrap it in markdown or any other text.

    Text: "{text_to_analyze}"

    Example JSON response:
    {{
      "sentiment": "Positive",
      "confidence": 0.92
    }}
    """
    prompt = prompt_template.format(text_to_analyze=text_to_analyze)

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        # Gemini's chat completion uses a list of dicts for messages
        # The content of the user message is the prompt.
        messages = [
            {"role": "user", "parts": [prompt]}
        ]
        
        response = model.generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, # Lower temperature for more deterministic output
                max_output_tokens=800 # Renamed from max_tokens for Gemini
            )
        )
        
        # Access the text content from the response
        model_output_content = response.text
        
        # Extract JSON from markdown if present or if it's just raw JSON
        json_string = _extract_json_from_markdown(model_output_content)
        
        if not json_string:
            st.error("Extracted JSON string is empty. Model response might be malformed or empty.")
            return {"sentiment": "Error", "confidence": 0.0}

        sentiment_data = json.loads(json_string)
        
        sentiment = sentiment_data.get('sentiment')
        confidence = sentiment_data.get('confidence')

        # Basic validation for confidence score
        if not (0.0 <= confidence <= 1.0):
            st.warning(f"Gemini returned an out-of-range confidence score for sentiment: {confidence}. Clamping to [0, 1].")
            confidence = max(0.0, min(1.0, confidence))

        return {"sentiment": sentiment, "confidence": confidence}

    except Exception as e:
        st.error(f"Error communicating with Google Gemini API for sentiment analysis: {e}")
        # Attempt to provide more context if response object has error details
        if hasattr(e, 'response') and e.response:
            st.error(f"Gemini API Error details: {e.response.text}")
        return {"sentiment": "Error", "confidence": 0.0}
    finally:
        time.sleep(API_CALL_DELAY)

def extract_keywords(text: str):
    """
    Extracts key phrases/keywords that *drive the sentiment* from the given text
    using the Google Gemini API. Returns a list of strings.
    """
    if not GOOGLE_API_KEY:
        return []

    # Refined prompt to explicitly ask for sentiment-driving keywords
    prompt_template = """
    From the following text, extract up to 5 keywords or short phrases that *specifically drive or indicate its overall sentiment*.
    Focus on words or phrases that directly convey the emotional tone (positive, negative, or neutral).
    Return the output as a JSON object with a single field 'sentiment_keywords', which is an array of strings. Ensure the output is *only* the JSON object, do NOT wrap it in markdown or any other text.

    Text: "{text}"

    Example JSON response for positive text:
    {{
      "sentiment_keywords": ["fantastic product", "so happy", "quality is great"]
    }}
    Example JSON response for negative text:
    {{
      "sentiment_keywords": ["terrible service", "very disappointed", "long wait"]
    }}
    """
    prompt = prompt_template.format(text=text)

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        messages = [
            {"role": "user", "parts": [prompt]}
        ]

        response = model.generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, # Lower temperature for more deterministic output
                max_output_tokens=100 # Adjust max_output_tokens for keywords
            )
        )

        model_output_content = response.text

        json_string = _extract_json_from_markdown(model_output_content)

        if not json_string:
            st.error("Extracted JSON string for keywords is empty. Model response might be malformed or empty.")
            return []

        keywords_data = json.loads(json_string)
        return keywords_data.get('sentiment_keywords', [])

    except Exception as e:
        st.error(f"Error communicating with Google Gemini API for keyword extraction: {e}")
        if hasattr(e, 'response') and e.response:
            st.error(f"Gemini API Error details: {e.response.text}")
        return []
    finally:
        time.sleep(API_CALL_DELAY)