import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st

# Download NLTK resources (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def analyze_text(text):
    results = {}
    
    # Tokenization
    tokens = word_tokenize(text)
    results['tokens'] = tokens
    
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    results['filtered_tokens'] = filtered_tokens
    
    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    results['sentiment'] = sentiment
    
    # Basic intent recognition
    intents = {
        'describe': ['describe', 'summary', 'overview'],
        'correlation': ['correlation', 'relationship', 'related'],
        'trend': ['trend', 'change', 'over time'],
        'distribution': ['distribution', 'spread', 'histogram']
    }
    
    detected_intents = []
    for intent, keywords in intents.items():
        if any(keyword in text.lower() for keyword in keywords):
            detected_intents.append(intent)
    
    results['intents'] = detected_intents
    
    return results