import requests
import json
import google.generativeai as genai
import pandas as pd
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys from environment
HF_API_KEY = os.getenv("HF_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def prepare_dataset_context(df):
    """Create a comprehensive context about the dataset"""
    context = f"""
    ## Dataset Overview
    - Rows: {len(df)}
    - Columns: {len(df.columns)}
    - Column names: {', '.join(df.columns)}
    
    ## Sample Data (First 3 Rows)
    {df.head(3).to_markdown(index=False)}
    
    ## Key Statistics
    """
    
    # Add numeric statistics if available
    numeric_cols = df.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        context += df[numeric_cols].describe().to_markdown()
    else:
        context += "No numeric columns"
    
    return context

def get_hf_response(query, df, model_name="google/flan-t5-base"):
    """Get response from Hugging Face model with robust error handling"""
    if not HF_API_KEY:
        return "Hugging Face API key not configured. Please set HF_API_KEY in .env file"
    
    # Prepare context
    try:
        dataset_context = prepare_dataset_context(df)
        
        # Create the full prompt
        full_prompt = f"""
        You are a data science assistant analyzing a specific dataset. 
        Here is information about the dataset:
        
        {dataset_context}
        
        ### User Question:
        {query}
        
        Please provide a detailed response based specifically on this dataset.
        """
        
        API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        
        # Check model status
        status_url = f"https://api-inference.huggingface.co/status/{model_name}"
        status_response = requests.get(status_url, headers=headers)
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            if not status_data.get('loaded', False):
                return "Model is loading. Please try again in 30-60 seconds."
        
        # MODEL-SPECIFIC PAYLOAD STRUCTURE THAT WORKS
        if "bart" in model_name or "t5" in model_name:
            payload = {"inputs": full_prompt}
        elif "squad" in model_name or "roberta" in model_name:
            payload = {"inputs": {"question": query, "context": dataset_context}}
        elif "gpt2" in model_name:
            payload = {"inputs": full_prompt, "parameters": {"max_new_tokens": 500, "temperature": 0.7}}
        else:
            payload = {"inputs": full_prompt}
        
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Handle response
        if isinstance(result, list) and len(result) > 0:
            if 'generated_text' in result[0]:
                return result[0]['generated_text']
            if 'summary_text' in result[0]:
                return result[0]['summary_text']
            if 'answer' in result[0]:
                return result[0]['answer']
        
        if isinstance(result, dict):
            if 'generated_text' in result:
                return result['generated_text']
            if 'answer' in result:
                return result['answer']
            if 'summary_text' in result:
                return result['summary_text']
        
        return str(result)
    
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 400:
            logger.error(f"Bad Request: {http_err.response.text}")
            return f"Bad Request: The model expects different input format"
        elif http_err.response.status_code == 404:
            return f"Model not found: {model_name}. Please choose a different model."
        elif http_err.response.status_code == 503:
            return "Model is loading. Please try again in 30-60 seconds."
        else:
            logger.error(f"HTTP Error ({http_err.response.status_code}): {http_err.response.text}")
            return f"HTTP Error: {http_err}"
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return f"API Error: {str(e)}"

def get_googleai_response(query, df, model_name="gemini-1.0-pro"):
    if not GOOGLE_API_KEY:
        return "Google AI API key not configured"
    
    try:
        # Prepare dataset context
        dataset_context = prepare_dataset_context(df)
        
        # Configure API
        genai.configure(
            api_key=GOOGLE_API_KEY,
            transport='rest',
            client_options={"api_endpoint": "generativelanguage.googleapis.com"}
        )
        
        # Initialize model
        model = genai.GenerativeModel(model_name)
        
        # Create the full prompt
        full_prompt = f"""
        You are a data science assistant. You've been given a dataset to analyze.
        
        ### Dataset Information:
        {dataset_context}
        
        ### User Question:
        {query}
        
        Please provide a detailed response based specifically on this dataset.
        """
        
        # Generate response
        response = model.generate_content(full_prompt)
        return response.text
    
    except Exception as e:
        logger.error(f"Google AI API error: {str(e)}")
        return f"API Error: {str(e)}"