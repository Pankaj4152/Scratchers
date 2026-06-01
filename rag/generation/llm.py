import os
from google import genai
from google.genai import types

class LLMOrchestrator:
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None):
        """
        Initializes the official Google GenAI Client.
        Default model 'gemini-2.5-flash' is incredibly fast and highly accurate.
        """
        print(f"Initializing Gemini LLM Engine ('{model_name}')...")
        self.model_name = model_name
        
        # Pulls the key from environment variables if not passed directly
        target_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        # Initialize the official Google GenAI Client
        self.client = genai.Client(api_key=target_key)

    def generate_response(self, structured_prompt: str) -> str:
        """Sends the prompt block to Gemini and returns the generated text answer."""
        try:
            # Configure generation options (Force temperature 0 for factual stability)
            config = types.GenerateContentConfig(
                temperature=0.0
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=structured_prompt,
                config=config
            )
            
            return response.text
            
        except Exception as e:
            return f"Error communicating with Gemini Engine: {str(e)}"