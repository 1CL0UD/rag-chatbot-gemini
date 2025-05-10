"""
Configuration module for the RAG chatbot.
Handles environment variable loading and default settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_api_key():
    """Get the Gemini API key from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return api_key

def get_model_settings():
    """Get model settings from environment variables."""
    return {
        "model": os.getenv("GEMINI_MODEL", "gemini-pro"),
        "temperature": float(os.getenv("DEFAULT_TEMPERATURE", 0.7))
    }

def get_app_settings():
    """Get application settings from environment variables."""
    return {
        "port": int(os.getenv("PORT", 7860)),
        "share": os.getenv("SHARE", "false").lower() == "true"
    }