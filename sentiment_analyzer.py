import os
from dotenv import load_dotenv
from transformers import pipeline
import spacy

# Load environment variables from .env file (if any, for API keys etc.)
load_dotenv()

# --- Sentiment Analysis Model ---
# Using a Hugging Face model trained for multi-class sentiment (positive, negative, neutral)
# 'cardiffnlp/twitter-roberta-base-sentiment-latest' is a good general-purpose choice.
# It outputs 'LABEL_0' (Negative), 'LABEL_1' (Neutral), 'LABEL_2' (Positive).
try:
    # We use 'revision="main"' to ensure we get the latest stable version if available.
    # This might download the model the first time it runs.
    sentiment_classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        revision="main"
    )
    print("Sentiment model loaded successfully.")
except Exception as e:
    sentiment_classifier = None
    print(f"Error loading sentiment model: {e}")
    print("Sentiment analysis functionality will be limited.")


def get_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of a given text using a pre-trained model.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary containing 'sentiment' (Positive/Negative/Neutral)
              and 'confidence' (score from 0 to 1).
              Returns 'Error' sentiment with 0 confidence on failure.
    """
    if not text or not text.strip():
        return {'sentiment': 'Neutral', 'confidence': 1.0, 'reason': 'Empty text'}

    if sentiment_classifier is None:
        return {'sentiment': 'Error', 'confidence': 0.0, 'reason': 'Sentiment model not loaded'}

    try:
        # The pipeline returns a list of dictionaries, e.g.,
        # [{'label': 'LABEL_0', 'score': 0.99}]
        results = sentiment_classifier(text)
        result = results[0] # Get the top prediction

        label_map = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral',
            'LABEL_2': 'Positive'
        }

        sentiment_label = label_map.get(result['label'], 'Unknown')
        confidence_score = result['score']

        return {'sentiment': sentiment_label, 'confidence': confidence_score}

    except Exception as e:
        print(f"Error during sentiment analysis for text: '{text[:50]}...' - {e}")
        return {'sentiment': 'Error', 'confidence': 0.0, 'reason': str(e)}

# --- Keyword Extraction Model ---
try:
    # Load a small English model for spaCy
    nlp_spacy = spacy.load("en_core_web_sm")
    print("SpaCy model loaded successfully for keyword extraction.")
except Exception as e:
    nlp_spacy = None
    print(f"Error loading SpaCy model: {e}")
    print("Keyword extraction functionality will be limited.")

def extract_keywords(text: str) -> list:
    """
    Extracts keywords/key phrases from the given text using SpaCy.

    Args:
        text (str): The input text.

    Returns:
        list: A list of extracted keywords (strings).
              Returns an empty list on failure.
    """
    if not text or not text.strip():
        return []
    if nlp_spacy is None:
        return []

    try:
        doc = nlp_spacy(text)
        keywords = []

        # Extract named entities (PERSON, ORG, GPE, PRODUCT etc.)
        for ent in doc.ents:
            keywords.append(ent.text)

        # Extract significant noun phrases (optional, but often useful)
        for chunk in doc.noun_chunks:
            # Filter out very common or short phrases if desired
            if len(chunk.text.split()) > 1 and chunk.text.lower() not in ["the", "a", "an", "this", "that"]:
                 keywords.append(chunk.text)

        # You might want to filter for unique keywords and sort them
        return sorted(list(set(keywords)))

    except Exception as e:
        print(f"Error during keyword extraction for text: '{text[:50]}...' - {e}")
        return []

if __name__ == "__main__":
    # This block runs only when sentiment_analyzer.py is executed directly
    # Useful for quick testing of the functions.
    print("\n--- Testing Sentiment Analysis ---")
    test_texts = [
        "This product is absolutely amazing! I love it.",
        "The service was terrible and I'm very disappointed.",
        "The weather is just okay today, not great, not bad.",
        "I can't believe how bad this is. It's truly awful.",
        "It was surprisingly not bad.", # Example of negation/double negative
        "", # Empty text
        "   " # Whitespace text
    ]

    for i, text in enumerate(test_texts):
        print(f"\nText {i+1}: '{text}'")
        sentiment_result = get_sentiment(text)
        print(f"  Sentiment: {sentiment_result['sentiment']}, Confidence: {sentiment_result['confidence']:.2f}")
        if 'reason' in sentiment_result:
            print(f"  Reason: {sentiment_result['reason']}")

        keywords_result = extract_keywords(text)
        print(f"  Keywords: {', '.join(keywords_result)}")