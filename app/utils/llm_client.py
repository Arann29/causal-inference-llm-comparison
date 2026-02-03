import os
from openai import OpenAI
import streamlit as st

# Base class for all LLM clients
class LLMClient:
    def __init__(self):
        pass

    def ask(self, prompt, model, temperature=0.0):
        raise NotImplementedError("This method should be implemented by subclasses.")

# OpenAI specific client
class OpenAIClient(LLMClient):
    def __init__(self):
        super().__init__()
        # API key is loaded from environment variable set by Docker
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            st.error("OPENAI_API_KEY environment variable not found.")
            st.stop()
        self.client = OpenAI(api_key=self.api_key)

    def ask(self, prompt, model="gpt-5.2", temperature=0.0):
        """
        Sends a prompt to the OpenAI API and gets a response.
        
        GPT-4.x uses Chat Completions API with temperature for determinism.
        GPT-5.x uses Responses API without temperature (parameter ignored).
        """
        try:
            # Route based on model prefix
            if model.startswith("gpt-4"):
                # GPT-4.x: Chat Completions API with temperature
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return response.choices[0].message.content
            
            elif model.startswith("gpt-5"):
                # GPT-5.x: Responses API (temperature not supported)
                response = self.client.responses.create(
                    model=model,
                    input=prompt,
                )
                # Try high-level accessor first, fallback to low-level schema
                try:
                    return response.output_text
                except AttributeError:
                    try:
                        return response.output[0].content[0].text
                    except (AttributeError, IndexError, KeyError):
                        st.error(f"GPT-5.2 response format unrecognized. Response type: {type(response)}")
                        return None
            
            else:
                st.error(f"Unsupported model: {model}")
                return None
                
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
            return None

# A factory to get the right client
def get_llm_client(client_type="openai"):
    """
    Factory function to get an instance of an LLM client.
    """
    if client_type == "openai":
        return OpenAIClient()
    # Future clients can be added here
    # elif client_type == "claude":
    #     return ClaudeClient()
    # elif client_type == "gemini":
    #     return GeminiClient()
    else:
        raise ValueError(f"Unsupported LLM client type: {client_type}")
