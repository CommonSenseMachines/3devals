#!/usr/bin/env python3
"""
llm_clients.py - Multi-provider LLM client system for 3D mesh evaluation

Provides unified interface for Claude, OpenAI, and Gemini APIs with
proper error handling and response formatting.
"""

import json
import base64
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from io import BytesIO
import time

from .llm_prompts import LLMProvider, get_provider_config, EvaluationType

# Optional imports for different providers
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

@dataclass
class LLMRequest:
    """Standardized request structure for all LLM providers"""
    system_prompt: str
    user_prompt: str
    images: List[Dict[str, str]]  # [{"name": "concept", "data": "base64_data"}, ...]
    evaluation_type: EvaluationType
    metadata: Dict[str, Any]

@dataclass
class LLMResponse:
    """Standardized response structure from LLM providers"""
    content: str
    parsed_json: Optional[Dict[str, Any]]
    provider: LLMProvider
    model: str
    success: bool
    error: Optional[str]
    metadata: Dict[str, Any]
    prompt_info: Dict[str, Any]  # Store prompt information for caching
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching"""
        data = asdict(self)
        # Convert enum to string value for JSON serialization
        data['provider'] = self.provider.value
        return data

class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, api_key: str, provider: LLMProvider):
        self.api_key = api_key
        self.provider = provider
        self.config = get_provider_config(provider)
        self._client = None
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the provider-specific client"""
        pass
    
    @abstractmethod
    def _format_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Format request for provider-specific API"""
        pass
    
    @abstractmethod
    def _call_api(self, formatted_request: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call to provider"""
        pass
    
    @abstractmethod
    def _parse_response(self, api_response: Dict[str, Any]) -> str:
        """Extract text content from provider response"""
        pass
    
    def call(self, request: LLMRequest) -> LLMResponse:
        """Main method to call LLM with standardized request/response"""
        if not self._client:
            self._initialize_client()
        
        try:
            # Format request for provider
            formatted_request = self._format_request(request)
            
            # Make API call
            api_response = self._call_api(formatted_request)
            
            # Parse response
            content = self._parse_response(api_response)
            
            # Try to parse JSON
            parsed_json = None
            try:
                parsed_json = json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from text
                parsed_json = self._extract_json_from_text(content)
            
            return LLMResponse(
                content=content,
                parsed_json=parsed_json,
                provider=self.provider,
                model=self.config["model"],
                success=True,
                error=None,
                metadata=request.metadata,
                prompt_info={
                    "system_prompt": request.system_prompt,
                    "user_prompt": request.user_prompt,
                    "evaluation_type": request.evaluation_type.value,
                    "image_count": len(request.images),
                    "provider": self.provider.value,
                    "model": self.config["model"],
                }
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                parsed_json=None,
                provider=self.provider,
                model=self.config["model"],
                success=False,
                error=str(e),
                metadata=request.metadata,
                prompt_info={
                    "system_prompt": request.system_prompt,
                    "user_prompt": request.user_prompt,
                    "evaluation_type": request.evaluation_type.value,
                    "image_count": len(request.images),
                    "provider": self.provider.value,
                    "model": self.config["model"],
                }
            )
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract JSON from text that might contain extra content"""
        # Look for JSON-like structures
        import re
        
        # Try to find JSON objects
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None

class ClaudeClient(LLMClient):
    """Claude (Anthropic) LLM client"""
    
    def __init__(self, api_key: str):
        if not ANTHROPIC_AVAILABLE:
            raise LLMError("Anthropic library not available. Install with: pip install anthropic")
        super().__init__(api_key, LLMProvider.CLAUDE)
    
    def _initialize_client(self) -> None:
        """Initialize Anthropic client"""
        self._client = anthropic.Anthropic(api_key=self.api_key)
    
    def _format_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Format request for Anthropic API"""
        content = []
        
        # Add text content
        content.append({
            "type": "text",
            "text": request.user_prompt
        })
        
        # Add images
        for image in request.images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{self.config['image_format']}",
                    "data": image["data"],
                },
            })
        
        return {
            "model": self.config["model"],
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"],
            "system": request.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
    
    def _call_api(self, formatted_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call Anthropic API"""
        response = self._client.messages.create(**formatted_request)
        return {"response": response}
    
    def _parse_response(self, api_response: Dict[str, Any]) -> str:
        """Parse Anthropic response"""
        return api_response["response"].content[0].text

class OpenAIClient(LLMClient):
    """OpenAI LLM client"""
    
    def __init__(self, api_key: str):
        if not OPENAI_AVAILABLE:
            raise LLMError("OpenAI library not available. Install with: pip install openai")
        super().__init__(api_key, LLMProvider.OPENAI)
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client"""
        self._client = openai.OpenAI(api_key=self.api_key)
    
    def _format_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Format request for OpenAI API"""
        content = []
        
        # Add text content
        content.append({
            "type": "text",
            "text": request.user_prompt
        })
        
        # Add images
        for image in request.images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{self.config['image_format']};base64,{image['data']}"
                }
            })
        
        messages = []
        
        # Add system message
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Handle different parameters for different models
        request_params = {
            "model": self.config["model"],
            "messages": messages,
        }
        
        # Handle model-specific parameters
        if self.config["model"].startswith("o3"):
            # o3 model family has specific requirements
            request_params["max_completion_tokens"] = self.config["max_tokens"]
            # o3 only supports default temperature (1), so omit the parameter
        else:
            # Standard OpenAI models
            request_params["max_tokens"] = self.config["max_tokens"]
            request_params["temperature"] = self.config["temperature"]
        
        return request_params
    
    def _call_api(self, formatted_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call OpenAI API"""
        try:
            print(f"🔍 Making OpenAI API call with model: {formatted_request.get('model', 'unknown')}")
            response = self._client.chat.completions.create(**formatted_request)
            print(f"✅ OpenAI API response received")
            return {"response": response}
        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            print(f"🔍 Request parameters: {formatted_request.keys()}")
            raise
    
    def _parse_response(self, api_response: Dict[str, Any]) -> str:
        """Parse OpenAI response"""
        try:
            response = api_response["response"]
            print(f"🔍 OpenAI response structure: choices={len(response.choices) if hasattr(response, 'choices') else 'missing'}")
            
            if not hasattr(response, 'choices') or not response.choices:
                print(f"❌ No choices in OpenAI response: {response}")
                return ""
            
            choice = response.choices[0]
            if not hasattr(choice, 'message') or not choice.message:
                print(f"❌ No message in OpenAI choice: {choice}")
                return ""
            
            content = choice.message.content
            print(f"🔍 OpenAI content length: {len(content) if content else 0}")
            
            if not content:
                print(f"❌ Empty content in OpenAI response")
                print(f"🔍 Full response: {response}")
                return ""
            
            return content
            
        except Exception as e:
            print(f"❌ Failed to parse OpenAI response: {e}")
            print(f"🔍 Raw API response: {api_response}")
            raise

class GeminiClient(LLMClient):
    """Google Gemini LLM client (using new google.genai API)"""
    
    def __init__(self, api_key: str):
        if not GEMINI_AVAILABLE:
            raise LLMError("Google GenAI library not available. Install with: pip install google-genai")
        super().__init__(api_key, LLMProvider.GEMINI)
    
    def _initialize_client(self) -> None:
        """Initialize Gemini client"""
        self._client = genai.Client(api_key=self.api_key)
    
    def _format_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Format request for Gemini API"""
        
        # Build content parts
        parts = []
        
        # Add text parts (combine system and user prompts)
        if request.system_prompt:
            full_text = f"{request.system_prompt}\n\n{request.user_prompt}"
        else:
            full_text = request.user_prompt
        
        parts.append(types.Part.from_text(text=full_text))
        
        # Add image parts
        for image in request.images:
            # Convert base64 to bytes
            image_bytes = base64.b64decode(image["data"])
            
            # Determine MIME type based on format
            if self.config["image_format"] == "png":
                mime_type = "image/png"
            elif self.config["image_format"] == "webp":
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"
            
            parts.append(types.Part.from_bytes(
                mime_type=mime_type,
                data=image_bytes
            ))
        
        # Create content structure
        contents = [
            types.Content(
                role="user",
                parts=parts
            )
        ]
        
        # Create generation config
        generate_content_config = types.GenerateContentConfig(
            temperature=self.config["temperature"],
            max_output_tokens=self.config["max_tokens"],
            response_mime_type="text/plain",
        )
        
        return {
            "model": self.config["model"],
            "contents": contents,
            "config": generate_content_config,
        }
    
    def _call_api(self, formatted_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call Gemini API"""
        response = self._client.models.generate_content(
            model=formatted_request["model"],
            contents=formatted_request["contents"],
            config=formatted_request["config"]
        )
        return {"response": response}
    
    def _parse_response(self, api_response: Dict[str, Any]) -> str:
        """Parse Gemini response"""
        response = api_response["response"]
        
        # Try the text attribute first (most direct)
        if hasattr(response, 'text') and response.text:
            return response.text
        
        # Fallback: extract from candidates
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                part = candidate.content.parts[0]
                if hasattr(part, 'text') and part.text:
                    return part.text
        
        # Last resort fallback
        return str(response)

# -----------------------------------------------------------------------------
# Client Factory
# -----------------------------------------------------------------------------

def create_llm_client(provider: LLMProvider, api_key: str) -> LLMClient:
    """Factory function to create appropriate LLM client"""
    if provider == LLMProvider.CLAUDE:
        return ClaudeClient(api_key)
    elif provider == LLMProvider.OPENAI:
        return OpenAIClient(api_key)
    elif provider == LLMProvider.GEMINI:
        return GeminiClient(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")

def get_available_providers() -> List[LLMProvider]:
    """Get list of available LLM providers based on installed libraries"""
    available = []
    
    if ANTHROPIC_AVAILABLE:
        available.append(LLMProvider.CLAUDE)
    if OPENAI_AVAILABLE:
        available.append(LLMProvider.OPENAI)
    if GEMINI_AVAILABLE:
        available.append(LLMProvider.GEMINI)
    
    return available

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def validate_provider_availability(provider: LLMProvider) -> bool:
    """Check if a provider is available"""
    available_providers = get_available_providers()
    return provider in available_providers

def get_provider_installation_command(provider: LLMProvider) -> str:
    """Get installation command for provider"""
    commands = {
        LLMProvider.CLAUDE: "pip install anthropic",
        LLMProvider.OPENAI: "pip install openai",
        LLMProvider.GEMINI: "pip install google-genai",
    }
    return commands.get(provider, "Unknown provider")

# -----------------------------------------------------------------------------
# Testing Functions
# -----------------------------------------------------------------------------

def test_provider_connection(provider: LLMProvider, api_key: str) -> bool:
    """Test if provider connection works with given API key"""
    try:
        client = create_llm_client(provider, api_key)
        
        # Create a simple test request
        test_request = LLMRequest(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'Hello, test successful!' in JSON format with a 'message' field.",
            images=[],
            evaluation_type=EvaluationType.DETAILED_SCORING,
            metadata={"test": True}
        )
        
        response = client.call(test_request)
        return response.success
        
    except Exception as e:
        print(f"Provider {provider.value} test failed: {e}")
        return False 