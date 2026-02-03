"""
Ollama LLM client for local model interaction + Causal Reasoning Pipeline
Implements multi-step chain-of-thought reasoning and self-consistency voting.
"""
import requests
import json
from typing import Dict, List, Optional
import asyncio
from statistics import mode
import os
import datetime

# ==============================================================
# Base Ollama Client
# ==============================================================

class OllamaClient:
    def __init__(self, base_url: str = "http://ollama:11434", default_model: str = "gemma3:1b", openai_api_key: str = None, daily_limit: float = 5.0):
        self.base_url = base_url
        self.default_model = default_model
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.daily_limit = daily_limit  # Daily spending limit in USD
        self.usage_file = "/app/openai_usage.json"
        
        # Model pricing (per 1K tokens) - using real OpenAI API models
        self.model_pricing = {
            # 4.x series (chat completions)
            "gpt-4.1-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4.1": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            # 5.x series (responses API)
            "gpt-5": {"input": 0.01, "output": 0.03},
            "gpt-5-mini": {"input": 0.0025, "output": 0.0075},
            "gpt-5-nano": {"input": 0.0005, "output": 0.0015},
            "gpt-5.1": {"input": 0.0125, "output": 0.04}
        }
        
        # Initialize OpenAI client if API key is available
        self.openai_client = None
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                print(f"OpenAI client initialized successfully with key: {self.openai_api_key[:20]}...")
            except ImportError as e:
                print(f"OpenAI package not installed: {e}")
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
        else:
            print("No OpenAI API key provided")
    
    # ---------- Cost Control Methods ----------
    def _load_usage_data(self) -> dict:
        """Load daily usage data from file"""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    # Check if data is from today
                    today = str(datetime.date.today())
                    if data.get('date') == today:
                        return data
            # Return fresh data for new day
            return {'date': str(datetime.date.today()), 'cost': 0.0, 'requests': 0}
        except:
            return {'date': str(datetime.date.today()), 'cost': 0.0, 'requests': 0}
    
    def _save_usage_data(self, usage_data: dict):
        """Save usage data to file"""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(usage_data, f)
        except Exception as e:
            print(f"Warning: Could not save usage data: {e}")
    
    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for OpenAI API call"""
        if model not in self.model_pricing:
            return 0.0
        
        pricing = self.model_pricing[model]
        input_cost = (prompt_tokens / 1000) * pricing['input']
        output_cost = (completion_tokens / 1000) * pricing['output']
        return input_cost + output_cost
    
    def _check_spending_limit(self, estimated_cost: float) -> tuple[bool, str]:
        """Check if request would exceed daily spending limit"""
        usage_data = self._load_usage_data()
        projected_cost = usage_data['cost'] + estimated_cost
        
        if projected_cost > self.daily_limit:
            remaining = max(0, self.daily_limit - usage_data['cost'])
            return False, f"Daily limit exceeded. Used: ${usage_data['cost']:.4f}, Remaining: ${remaining:.4f}, Request cost: ${estimated_cost:.4f}"
        
        return True, f"Cost OK. Daily used: ${usage_data['cost']:.4f}/{self.daily_limit:.2f}, Request: ${estimated_cost:.4f}"
    
    def get_usage_stats(self) -> dict:
        """Get current usage statistics"""
        usage_data = self._load_usage_data()
        remaining = max(0, self.daily_limit - usage_data['cost'])
        return {
            'date': usage_data['date'],
            'cost_used': usage_data['cost'],
            'daily_limit': self.daily_limit,
            'remaining': remaining,
            'requests_made': usage_data['requests'],
            'percentage_used': (usage_data['cost'] / self.daily_limit) * 100
        }
    
    # ---------- Core Methods ----------
    def _is_openai_model(self, model: str) -> bool:
        """Check if the model is an OpenAI model based on pricing table."""
        return model in self.model_pricing
    
    def _is_responses_model(self, model: str) -> bool:
        """Models that should use the Responses API. All GPT-5 models use responses.create."""
        return model.startswith("gpt-5")
    
    def _is_chat_model(self, model: str) -> bool:
        """OpenAI models that use chat.completions. Everything that is NOT a 5.x model."""
        return self._is_openai_model(model) and not self._is_responses_model(model)
    
    def _generate_openai_chat_response(self, prompt: str, model: str, timeout: int = 180) -> str:
        """Generate response using OpenAI Chat Completions API with cost tracking."""
        if not self.openai_client:
            return "OpenAI client not available. Check API key and installation."
        
        # Estimate tokens and cost
        estimated_prompt_tokens = int(len(prompt.split()) * 1.3)
        estimated_completion_tokens = 400
        estimated_cost = self._estimate_cost(model, estimated_prompt_tokens, estimated_completion_tokens)
        
        # Check spending limit
        can_proceed, message = self._check_spending_limit(estimated_cost)
        if not can_proceed:
            return f"🚫 Request blocked: {message}"
        
        try:
            print(f"Making OpenAI Chat API call to {model}")
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                timeout=timeout
            )
            
            # Track actual usage
            actual_prompt_tokens = response.usage.prompt_tokens
            actual_completion_tokens = response.usage.completion_tokens
            actual_cost = self._estimate_cost(model, actual_prompt_tokens, actual_completion_tokens)
            
            # Update usage data
            usage_data = self._load_usage_data()
            usage_data['cost'] += actual_cost
            usage_data['requests'] += 1
            self._save_usage_data(usage_data)
            
            result = response.choices[0].message.content
            print(f"OpenAI chat response received (length: {len(result) if result else 0})")
            return result
            
        except Exception as e:
            error_msg = f"OpenAI Chat API error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _generate_openai_responses_response(self, prompt: str, model: str, timeout: int = 180) -> str:
        """Generate response using OpenAI Responses API (for GPT-5 series) with cost tracking."""
        if not self.openai_client:
            return "OpenAI client not available. Check API key and installation."
        
        # Rough estimates for pre-check
        estimated_prompt_tokens = int(len(prompt.split()) * 1.3)
        estimated_completion_tokens = 400
        estimated_cost = self._estimate_cost(model, estimated_prompt_tokens, estimated_completion_tokens)
        
        can_proceed, message = self._check_spending_limit(estimated_cost)
        if not can_proceed:
            return f"🚫 Request blocked: {message}"
        
        try:
            print(f"Making OpenAI Responses API call to {model}")
            
            response = self.openai_client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=400,
                timeout=timeout
            )
            
            # Extract text from response
            text = getattr(response, "output_text", None)
            if text is None:
                try:
                    text = response.output[0].content[0].text
                except Exception:
                    text = ""
            
            # Track usage
            usage = getattr(response, "usage", None)
            if usage is not None:
                actual_prompt_tokens = getattr(usage, "input_tokens", estimated_prompt_tokens)
                actual_completion_tokens = getattr(usage, "output_tokens", estimated_completion_tokens)
            else:
                actual_prompt_tokens = estimated_prompt_tokens
                actual_completion_tokens = estimated_completion_tokens
            
            actual_cost = self._estimate_cost(model, actual_prompt_tokens, actual_completion_tokens)
            usage_data = self._load_usage_data()
            usage_data["cost"] += actual_cost
            usage_data["requests"] += 1
            self._save_usage_data(usage_data)
            
            print(f"OpenAI responses result length: {len(text) if text else 0}")
            return text or "No text returned from Responses API."
            
        except Exception as e:
            error_msg = f"OpenAI Responses API error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def generate_response(self, prompt: str, model: str = None, context: Optional[Dict] = None, timeout: int = 180) -> str:
        """Generate response from Ollama or OpenAI model (synchronous)"""
        model = model or self.default_model
        full_prompt = self._build_contextualized_prompt(prompt, context)
        
        # Route to appropriate API based on model
        if self._is_openai_model(model):
            if self._is_responses_model(model):
                return self._generate_openai_responses_response(full_prompt, model, timeout)
            elif self._is_chat_model(model):
                return self._generate_openai_chat_response(full_prompt, model, timeout)
            else:
                return f"Model {model} is marked as OpenAI but not mapped to chat or responses."
        else:
            # Original Ollama logic
            try:
                payload = {
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {           # Add stronger decoding constraints
                        "temperature": 0.1,  # Lower for more focused output
                        "top_p": 0.8,       # Focus on most likely tokens
                        "num_predict": 400,  # Shorter limit
                        "stop": ["Step 6", "###", "---", "\n\n\n"]  # Stop tokens
                    }
                }
                response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout)
                response.raise_for_status()
                return response.json().get("response", "No response received.")
            except Exception as e:
                return f"Error communicating with Ollama: {str(e)}"
    
    async def _generate_openai_response_async(self, prompt: str, model: str, timeout: int = 180) -> str:
        """Generate response using OpenAI API asynchronously"""
        if not self.openai_client:
            return "OpenAI client not available. Check API key and installation."
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=400,
                    timeout=timeout
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API error: {str(e)}"
    
    async def generate_response_async(self, prompt: str, model: str = None, context: Optional[Dict] = None, timeout: int = 180) -> str:
        """Generate response asynchronously from Ollama or OpenAI"""
        model = model or self.default_model
        full_prompt = self._build_contextualized_prompt(prompt, context)
        
        # Route to appropriate API based on model
        if self._is_openai_model(model):
            return await self._generate_openai_response_async(full_prompt, model, timeout)
        else:
            # Original Ollama async logic
            import aiohttp
            try:
                payload = {
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {           # Match sync method constraints
                        "temperature": 0.1,
                        "top_p": 0.8,
                        "num_predict": 400,
                        "stop": ["Step 6", "###", "---", "\n\n\n"]
                    }
                }
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                    async with session.post(f"{self.base_url}/api/generate", json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return data.get("response", "No response received.")
            except Exception as e:
                return f"Error communicating with Ollama: {str(e)}"
    
    # ---------- Context Builder ----------
    def _build_contextualized_prompt(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Builds contextualized prompt using dataset details."""
        if context:
            context_str = "=== DATASET CONTEXT ===\n"
            context_str += f"Dataset: {context.get('dataset_name', 'Unknown')}\n"
            context_str += f"Variables: {context.get('variables', 'Unknown')}\n"
            context_str += f"\n=== STATISTICAL SUMMARY ===\n"
            context_str += f"{context.get('summary', 'No statistical summary available')}\n"
            context_str += f"\n=== ADDITIONAL CONTEXT ===\n"
            context_str += f"Existing Clustering: {context.get('clusters', 'None')}\n"
            context_str += f"Previous Causal Results: {context.get('causal_results', 'None')}\n"
            context_str += "=======================\n\n"
            return context_str + prompt
        return prompt
    
    def check_connection(self) -> bool:
        """Check if Ollama service is accessible."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False
    
    def check_openai_connection(self) -> bool:
        """Check if OpenAI API is accessible and configured."""
        print(f"Checking OpenAI connection...")
        print(f"API key present: {bool(self.openai_api_key)}")
        print(f"OpenAI client initialized: {bool(self.openai_client)}")
        
        if not self.openai_client or not self.openai_api_key:
            print("OpenAI client or API key missing")
            return False
        try:
            # Make a minimal API call to test connection
            print("Testing OpenAI API with minimal call...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            print("OpenAI API call successful")
            return True
        except Exception as e:
            print(f"OpenAI API test failed: {e}")
            return False


# ==============================================================
# Multi-Step Causal Reasoning Pipeline
# ==============================================================

class CausalReasoningPipeline:
    """
    Chain-of-thought causal reasoning pipeline using Ollama.
    Based on Cohrs et al. (2024), CausalGraph2LLM (2024), and self-consistency prompting.
    """

    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client

        # Detailed step templates (explicit reasoning goals)
        self.steps = [
            (
                "Step 1 - Dataset Analysis & Summary",
                "Analyze the dataset's statistical properties (means, ranges, distributions), correlation analysis, "
                "data patterns, and anomalies. Provide a single paragraph summary of your key findings about "
                "the relationship between the two variables."
            ),
            (
                "Step 2 - Cluster/Regime Identification", 
                "Based on the dataset summary, identify potential clustering or regime patterns. "
                "Determine how many distinct behavioral regimes might exist (e.g., seasonal, threshold-based, non-linear segments). "
                "Provide a single paragraph summary describing the regimes you identified and what differentiates them."
            ),
            (
                "Step 3 - Causal Direction Analysis",
                "IMPORTANT: If multiple regimes were identified in Step 2, analyze the causal direction "
                "SEPARATELY for each regime. For each regime individually, determine: does Variable X cause Variable Y, "
                "or vice versa within that specific regime? Consider temporal patterns, logical dependencies, "
                "and domain knowledge that may differ between regimes. If no clear regimes exist, analyze overall. "
                "Format your response as: 'Regime 1: [direction and justification]. Regime 2: [direction and justification].' "
                "etc. Provide clear justification for each regime's causal direction. "
                "Output everything as a single paragraph."
            ),
            (
                "Step 4 - Evidence Evaluation & Confidence Assessment",
                "Critically evaluate the strength of the causal evidence. "
                "Discuss potential confounders, assumptions, data limitations, and consistency across regimes. "
                "Provide a single paragraph summary with your confidence assessment (High/Moderate/Low) and "
                "explanation of any uncertainties."
            ),
            (
                "Step 5 - Structured Synthesis",
                "IMPORTANT: You MUST provide ONLY a JSON response with NO additional text.\n"
                "Based on your previous 4 steps of analysis, make concrete decisions and output ONLY this JSON:\n\n"
                "{\n"
                '  "clusters": <number>,\n'
                '  "directions": [{"cluster": <number>, "direction": "<X->Y or Y->X or ambiguous>", "confidence": "<High|Moderate|Low>"}],\n'
                '  "overall": "<X->Y or Y->X or ambiguous>",\n'
                '  "notes": "<one brief sentence explaining the decision>"\n'
                "}\n\n"
                "RULES:\n"
                "1. Output ONLY the JSON above - no explanations, no reasoning, no other text\n"
                "2. Choose concrete values based on your analysis\n"
                "3. If unsure about clusters, default to 1\n"
                "4. Pick the most likely causal direction\n"
                "5. Keep notes to maximum 15 words\n"
                "6. Do NOT write any text before or after the JSON"
            )
        ]

    # ----------------------------------------------------------
    # Core Execution
    # ----------------------------------------------------------
    def run_pipeline_sync_with_display(self, dataset_context: Dict, model: str = None):
        """Run the reasoning pipeline synchronously with sequential display and prompt tracking."""
        import streamlit as st
        
        model = model or self.client.default_model
        results = {}
        prompts = {}
        cumulative_context = ""  # Store previous reasoning to feed into next step

        for i, (title, instruction) in enumerate(self.steps, 1):
            # Show step header
            st.markdown(f"### {title}")
            
            # Build prompt
            if i == 5:
                prompt = f"{title}\n{instruction}\n\nPrevious reasoning:\n{cumulative_context}\n\nNow output ONLY the JSON - no other text:"
            else:
                prompt = f"{title}\n{instruction}\n\nPrevious reasoning:\n{cumulative_context}\n\nNow respond with your reasoning for this step."
                
            # Show prompt in expandable section
            with st.expander(f"📝 Show Prompt for {title}", expanded=False):
                full_prompt = self.client._build_contextualized_prompt(prompt, dataset_context)
                st.code(full_prompt, language="markdown")
                
            prompts[f"step_{i}"] = full_prompt
            
            # Execute step with spinner
            with st.spinner(f"Executing {title}..."):
                print(f"Executing {title}...")
                
                # Special handling for Step 5 - force JSON output
                if i == 5:
                    # Use OpenAI for GPT models, Ollama for local models
                    if self.client._is_openai_model(model):
                        response_text = self.client.generate_response(prompt, model=model, context=dataset_context)
                    else:
                        # Use more restrictive settings for Step 5 with Ollama
                        special_payload = {
                            "model": model,
                            "prompt": self.client._build_contextualized_prompt(prompt, dataset_context),
                            "stream": False,
                            "options": {
                                "temperature": 0.05,  # Very deterministic
                                "top_p": 0.7,
                                "num_predict": 200,   # Very short
                                "stop": ["\n\n", "Step", "Note:", "Explanation:"]
                            }
                        }
                        try:
                            import requests
                            response = requests.post(f"{self.client.base_url}/api/generate", json=special_payload, timeout=180)
                            response.raise_for_status()
                            response_text = response.json().get("response", "")
                        except:
                            response_text = self.client.generate_response(prompt, model=model, context=dataset_context)
                else:
                    response_text = self.client.generate_response(prompt, model=model, context=dataset_context)
                
                print(f"{title} response length: {len(response_text) if response_text else 0}")
                print(f"{title} response preview: {response_text[:150] if response_text else 'EMPTY'}...")
                
                # Special handling for Step 5 (structured output)
                if i == 5:
                    json_result = self._extract_json_from_step5(response_text)
                    if json_result:
                        # Store both raw JSON and formatted output
                        results[f"step_{i}_raw"] = response_text
                        results[f"step_{i}"] = self._format_structured_output(json_result)
                        formatted_output = self._format_structured_output(json_result)
                    else:
                        # Fallback to raw response if JSON extraction fails
                        results[f"step_{i}"] = response_text
                        formatted_output = response_text
                else:
                    results[f"step_{i}"] = response_text
                    formatted_output = response_text
                    
            # Display result immediately
            if formatted_output and formatted_output.strip():
                st.markdown(formatted_output)
            else:
                st.warning(f"No output received for {title}")
                
            # Add divider except for last step
            if i < 5:
                st.divider()
                
            cumulative_context += f"\n### {title} Output ###\n{response_text}\n"

        return results, prompts
    
    def run_pipeline_sync(self, dataset_context: Dict, model: str = None) -> Dict[str, str]:
        """Run the reasoning pipeline synchronously with cumulative context."""
        model = model or self.client.default_model
        results = {}
        cumulative_context = ""  # Store previous reasoning to feed into next step

        for i, (title, instruction) in enumerate(self.steps, 1):
            print(f"Executing {title}...")
            
            # Special handling for Step 5 - force JSON output
            if i == 5:
                prompt = f"{title}\n{instruction}\n\nPrevious reasoning:\n{cumulative_context}\n\nNow output ONLY the JSON - no other text:"
                # Use OpenAI for GPT models, Ollama for local models
                if self.client._is_openai_model(model):
                    response_text = self.client.generate_response(prompt, model=model, context=dataset_context)
                else:
                    # Use more restrictive settings for Step 5 with Ollama
                    special_payload = {
                        "model": model,
                        "prompt": self.client._build_contextualized_prompt(prompt, dataset_context),
                        "stream": False,
                        "options": {
                            "temperature": 0.05,  # Very deterministic
                            "top_p": 0.7,
                            "num_predict": 200,   # Very short
                            "stop": ["\n\n", "Step", "Note:", "Explanation:"]
                        }
                    }
                    try:
                        response = requests.post(f"{self.client.base_url}/api/generate", json=special_payload, timeout=180)
                        response.raise_for_status()
                        response_text = response.json().get("response", "")
                    except:
                        response_text = self.client.generate_response(prompt, model=model, context=dataset_context)
            else:
                prompt = f"{title}\n{instruction}\n\nPrevious reasoning:\n{cumulative_context}\n\nNow respond with your reasoning for this step."
                response_text = self.client.generate_response(prompt, model=model, context=dataset_context)
            
            print(f"{title} response length: {len(response_text) if response_text else 0}")
            print(f"{title} response preview: {response_text[:150] if response_text else 'EMPTY'}...")
            
            # Special handling for Step 5 (structured output)
            if i == 5:
                json_result = self._extract_json_from_step5(response_text)
                if json_result:
                    # Store both raw JSON and formatted output
                    results[f"step_{i}_raw"] = response_text
                    results[f"step_{i}"] = self._format_structured_output(json_result)
                else:
                    # Fallback to raw response if JSON extraction fails
                    results[f"step_{i}"] = response_text
            else:
                results[f"step_{i}"] = response_text
                
            cumulative_context += f"\n### {title} Output ###\n{response_text}\n"

        return results

    async def run_pipeline(self, dataset_context: Dict, model: str = None) -> Dict[str, str]:
        """Run asynchronously with cumulative reasoning."""
        model = model or self.client.default_model
        results = {}
        cumulative_context = ""

        for i, (title, instruction) in enumerate(self.steps, 1):
            # Special handling for Step 5 - force JSON output
            if i == 5:
                prompt = f"{title}\n{instruction}\n\nPrevious reasoning:\n{cumulative_context}\n\nNow output ONLY the JSON - no other text:"
            else:
                prompt = f"{title}\n{instruction}\n\nPrevious reasoning:\n{cumulative_context}\n\nNow respond with your reasoning for this step."
                
            response = await self.client.generate_response_async(prompt, model=model, context=dataset_context)
            
            # Special handling for Step 5 (structured output)
            if i == 5:
                json_result = self._extract_json_from_step5(response)
                if json_result:
                    # Store both raw JSON and formatted output
                    results[f"step_{i}_raw"] = response
                    results[f"step_{i}"] = self._format_structured_output(json_result)
                else:
                    # Fallback to raw response if JSON extraction fails
                    results[f"step_{i}"] = response
            else:
                results[f"step_{i}"] = response
                
            cumulative_context += f"\n### {title} Output ###\n{response}\n"

        return results

    # ----------------------------------------------------------
    # Self-Consistency / Voting
    # ----------------------------------------------------------
    def run_voted_pipeline(self, dataset_context: Dict, model: str = None, n_runs: int = 3) -> Dict[str, any]:
        """
        Run multiple full reasoning pipelines and aggregate results.
        Implements a 'voting schema' for causal direction consistency (Cohrs et al., 2024).
        """
        model = model or self.client.default_model
        all_runs = []
        summary_phrases = []

        for run in range(n_runs):
            run_result = self.run_pipeline_sync(dataset_context, model=model)
            final_output = run_result.get("step_5", "")
            all_runs.append(run_result)
            if "→" in final_output or "->" in final_output:
                summary_phrases.append(final_output)

        # Majority vote by most common direction phrase
        consensus = "Undetermined"
        if summary_phrases:
            try:
                consensus = mode(summary_phrases)
            except Exception:
                consensus = summary_phrases[0]

        return {
            "consensus": consensus,
            "runs": all_runs,
            "n_runs": n_runs
        }

    def _extract_json_from_step5(self, response_text):
        """Extract JSON from Step 5 response, with fallback parsing."""
        try:
            # Try to find JSON block
            import json
            import re
            
            # Look for JSON block between ```json and ``` or just { }
            json_match = re.search(r'```json\s*\n(.+?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Look for standalone JSON object
                json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Fallback: try to parse the whole response
                    json_str = response_text.strip()
            
            # Parse the JSON
            result = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['clusters', 'directions', 'overall', 'notes']
            if all(field in result for field in required_fields):
                return result
            else:
                return None
                
        except (json.JSONDecodeError, AttributeError):
            return None

    def _format_structured_output(self, json_result):
        """Convert JSON result to user-friendly structured format."""
        if not json_result:
            return "Unable to generate structured output."
        
        try:
            clusters = json_result.get('clusters', 1)
            overall = json_result.get('overall', 'ambiguous')
            directions = json_result.get('directions', [])
            notes = json_result.get('notes', 'No additional notes')
            
            # Format the output
            output_lines = [f"Clusters identified: {clusters}"]
            
            # Add direction information
            if directions:
                for i, direction_info in enumerate(directions):
                    cluster_num = direction_info.get('cluster', i+1)
                    direction = direction_info.get('direction', 'ambiguous')
                    confidence = direction_info.get('confidence', 'Unknown')
                    output_lines.append(f"Cluster {cluster_num}: {direction} (Confidence: {confidence})")
            
            output_lines.append(f"Overall causal hypothesis: {overall}")
            output_lines.append(f"Notes: {notes}")
            
            return "\n".join(output_lines)
            
        except Exception as e:
            return f"Error formatting output: {str(e)}"
