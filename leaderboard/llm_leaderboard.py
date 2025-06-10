#!/usr/bin/env python3
"""
llm_leaderboard.py – Run a vision‑LLM leaderboard on CSM mesh reconstructions.

Given a JSON manifest of session IDs (see example below) the script:
  1. Auto-extracts concept images and model info from CSM API.
  2. Downloads render images for every candidate session.
  3. Sends the image pairs to Claude's vision API and
     records the rubric scores it returns.
  4. Performs pairwise comparisons for ELO rankings.
  5. Generates comprehensive statistics and visualizations.
  6. Ranks the candidates per concept and prints a scoreboard.

Example leaderboard_models.json (curated entries only)
--------------------------------------------------------
[
  {"session_id": "SESSION_1749148355_6335964", "human_eval": {"StructuralForm": 8.0, "PartCoverage": 9.0, "SurfaceDetail": 8.0, "TextureQuality": 7.0}},
  {"session_id": "SESSION_1749146142_3027706", "human_eval": {"StructuralForm": 7.0, "PartCoverage": 0.0, "SurfaceDetail": 5.0, "TextureQuality": 4.0}}
]

Workflow:
1. Add only key sessions with COMPLETE human evaluations to leaderboard_models.json
2. All other sessions are auto-imported from job_tracking.json 
3. Script merges both sources and shows preview before running
4. Model names and concept images are auto-extracted from CSM API

Run:
  python llm_leaderboard.py --human-eval-json leaderboard_models.json

Requirements:
  pip install requests tqdm pillow python-dotenv anthropic matplotlib seaborn pandas numpy scipy
  
Optional (for multi-view rendering):
  pip install trimesh[easy]
  
Optional (for additional LLM providers):
  pip install openai                    # For OpenAI GPT models
  pip install google-genai              # For Google Gemini models
"""

import argparse
import base64
import json
import os
import sys
import getpass
import itertools
import math
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import requests
import anthropic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from scipy import stats

# Import mesh rendering utilities
try:
    from .mesh_renderer import (
        render_mesh_views_to_base64, 
        check_trimesh_available, 
        MeshRenderError
    )
    MESH_RENDERER_AVAILABLE = True
except ImportError:
    MESH_RENDERER_AVAILABLE = False
    print("⚠️  mesh_renderer not available - multiview rendering disabled")

# Import LLM client system
try:
    from .llm_prompts import (
        LLMProvider, EvaluationType, EVALUATION_RUBRIC,
        format_detailed_scoring_prompt, format_multiview_scoring_prompt,
        format_pairwise_comparison_prompt, create_view_images_section,
        create_hybrid_view_images_section
    )
    from .llm_clients import (
        LLMRequest, LLMResponse, LLMError, create_llm_client, 
        get_available_providers as get_available_llm_providers,
        validate_provider_availability, get_provider_installation_command
    )
    from .llm_cache import (
        get_cache_manager, save_llm_response, load_llm_response,
        has_cached_llm_response
    )
    LLM_SYSTEM_AVAILABLE = True
except ImportError as e:
    LLM_SYSTEM_AVAILABLE = False
    print(f"⚠️  LLM system not available: {e}")
    # Fallback to using original system
    LLMProvider = None
    LLMError = Exception  # Fallback

# -----------------------------------------------------------------------------
# Config & Constants
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are MeshCritique-v0.5, a 3D asset reviewer calibrated to human evaluation standards.\n"
    "Scale every dimension from 0 (unusable) to 10 (excellent).\n"
    "CRITICAL: Untextured models (gray/white) should have Texture Quality = 0. Single meshes that should be multi-part should have Part Coverage = 0.\n"
    "Return ONLY the JSON object, no explanations, no additional text, no reasoning."
)

PAIRWISE_SYSTEM_PROMPT = (
    "You are MeshCritique-v0.5, a 3D asset reviewer. You will compare two 3D mesh reconstructions against a concept image.\n"
    "Consider overall quality including geometry, textures, completeness, and how well each matches the original concept.\n"
    "Return ONLY the JSON object, no explanations."
)

USER_RUBRIC = """Rubric dimensions (unweighted, 0-10 scale):
1  Structural Form     (Overall geometric accuracy and proportions - can be evaluated for complete models and individual parts)
2  Part Coverage       (Single mesh: completeness; Multi-part: how well this part represents its intended portion)
3  Surface Detail      (Quality of geometric detail, surface features, mesh quality)
4  Texture Quality     (Texture quality, color accuracy, material properties)

Scoring Guidelines:
• 0: Unusable, completely failed, or missing (untextured models get Texture Quality = 0)
• 1-2: Poor quality, major problems
• 3-4: Below average, significant issues
• 5-6: Average, usable with some issues
• 7-8: Good quality, minor issues only
• 9-10: Excellent to perfect

Scoring Rules:
• For SINGLE MESH (mesh_count == 1): Score all dimensions normally
• For MULTI-PART (mesh_count > 1): Score all dimensions for this individual part
• If mesh_count == 1 **and** it visually SHOULD be multi‑part (e.g., limbs, wheels, hinges), set Part Coverage = 0
• If the render looks completely untextured (solid gray/white, no color/material), set Texture Quality = 0
• Multi-part models: Individual parts in well-decomposed kits should score reasonably (5-8 range) as decomposition itself is valuable

Output JSON schema:
{
  "session_id": string,
  "scores": {
     "StructuralForm": int,
     "PartCoverage": int,
     "SurfaceDetail": int,
     "TextureQuality": int
  },
  "score_array": [int, int, int, int]  // [StructuralForm, PartCoverage, SurfaceDetail, TextureQuality]
}
"""

PAIRWISE_PROMPT = """Compare these two 3D mesh reconstructions against the concept image. Consider:
• Overall geometric accuracy and completeness
• Surface detail and mesh quality  
• Texture/material quality
• How well each represents the original concept

Output JSON schema:
{
  "winner": "A" | "B" | "tie",
  "confidence": "low" | "medium" | "high",
  "reasoning": "Brief explanation of choice"
}
"""

HEADERS = {"Content-Type": "application/json"}

CONFIG_FILE = Path(".leaderboard_config")
DEBUG_DIR = Path(__file__).parent / "debug_queries"
RESULTS_DIR = Path("results")
CACHE_DIR = Path("llm_cache")  # Local cache for LLM responses (add to .gitignore)

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class ModelScore:
    """Individual model evaluation result"""
    model_name: str
    session_ids: List[str]
    concept_session_id: str
    scores: List[float]  # [StructuralForm, PartCoverage, SurfaceDetail, TextureQuality]
    human_scores: List[float]  # Same format as scores
    avg_score: float
    human_avg_score: float





@dataclass
class LeaderboardResults:
    """Complete evaluation results"""
    model_scores: Dict[str, List[ModelScore]]  # concept_id -> [ModelScore]
    dimension_names: List[str]

def get_provider_name(llm_client) -> str:
    """Extract provider name from llm_client for file naming"""
    if hasattr(llm_client, 'provider') and hasattr(llm_client.provider, 'value'):
        return llm_client.provider.value
    else:
        # Fallback for old system
        return "claude"


def is_retryable_error(error_msg: str) -> bool:
    """Check if an error is retryable (transient)"""
    if not error_msg:
        return False
        
    error_lower = error_msg.lower()
    
    # HTTP status codes that are typically retryable
    retryable_codes = ['503', '502', '500', '429', '408', '504']
    for code in retryable_codes:
        if code in error_msg:
            return True
    
    # Common retryable error messages
    retryable_phrases = [
        'overloaded',
        'unavailable', 
        'timeout',
        'rate limit',
        'temporarily unavailable',
        'service unavailable',
        'connection error',
        'network error',
        'server error'
    ]
    
    return any(phrase in error_lower for phrase in retryable_phrases)


def call_llm_with_retry(llm_client, llm_request, max_retries: int = 3, base_delay: float = 1.0):
    """Call LLM with exponential backoff retry for transient errors"""
    
    for attempt in range(max_retries + 1):
        try:
            response = llm_client.call(llm_request)
            
            # If successful, return immediately
            if response.success:
                return response
            
            # Check if error is retryable
            if attempt < max_retries and is_retryable_error(response.error):
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"   ⏳ LLM API error (attempt {attempt + 1}/{max_retries + 1}): {response.error}")
                print(f"   🔄 Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                continue
            else:
                # Non-retryable error or final attempt
                return response
                
        except Exception as e:
            error_msg = str(e)
            
            # Check if this exception is retryable
            if attempt < max_retries and is_retryable_error(error_msg):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"   ⏳ LLM API exception (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                print(f"   🔄 Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                continue
            else:
                # Non-retryable exception or final attempt - re-raise
                raise
    
    # Should not reach here, but just in case
    return response


def check_session_cache_pattern(llm_client, session_id, concept_session_id, model_name, trial, use_cache):
    """Look for cached files for this session by reading cache file contents"""
    if not use_cache:
        return None
    
    import glob
    import os
    from pathlib import Path
    
    # Get cache directory based on system type
    if LLM_SYSTEM_AVAILABLE:
        cache_manager = get_cache_manager()
        cache_dir = cache_manager.cache_dir
        # Get cache files for the current provider only
        provider_name = get_provider_name(llm_client)
        cache_files = []
        cache_files.extend(glob.glob(f"{cache_dir}/multiview/multiview_scoring_{provider_name}_*.json"))
        cache_files.extend(glob.glob(f"{cache_dir}/detailed/detailed_scoring_{provider_name}_*.json"))
    else:
        # Old cache system pattern
        cache_dir = Path("llm_cache")
        provider_name = get_provider_name(llm_client)
        cache_files = glob.glob(f"{cache_dir}/detailed_multiview_{provider_name}_*.json")
        cache_files.extend(glob.glob(f"{cache_dir}/detailed_{provider_name}_*.json"))
    
    print(f"     🔍 Scanning {len(cache_files)} cache files for session {session_id}...")
    
    # Load and check each cache file for matching session ID
    cached_results = []
    for cache_file in cache_files:
        try:
            if LLM_SYSTEM_AVAILABLE:
                # New cache system - load JSON file and check session ID
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                response_data = cache_data.get('response', {})
                if response_data.get('success') and response_data.get('parsed_json'):
                    parsed_json = response_data['parsed_json']
                    cached_session_id = parsed_json.get('session_id', '')
                    
                    if cached_session_id == session_id:
                        cached_results.append(parsed_json)
                        print(f"         ✅ Found match in {os.path.basename(cache_file)}")
            else:
                # Old cache system - load pickle file  
                import pickle
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if cache_data and 'response' in cache_data:
                        response = cache_data['response']
                        if response.get('session_id') == session_id:
                            cached_results.append(response)
                            print(f"         ✅ Found match in {os.path.basename(cache_file)}")
        except Exception as e:
            # Skip corrupted files silently
            continue
    
    if cached_results:
        print(f"     ✅ Successfully loaded {len(cached_results)} cached results for session {session_id}")
        return cached_results
    else:
        print(f"     ❌ No cached results found for session {session_id}")
        return None


def check_early_cache(
    llm_client,
    evaluation_type,
    session_id: str,
    mesh_count: int,
    mesh_index: int,
    concept_session_id: str,
    model_name: str,
    trial: int,
    use_cache: bool,
    views: List[str] = None,
    model_a_name: str = "",
    model_b_name: str = "",
) -> Optional[Dict]:
    """Check if we have a cached response before doing expensive image processing"""
    
    if not use_cache:
        return None
    
    if not LLM_SYSTEM_AVAILABLE:
        # Use old cache system
        print(f"   🔄 Using OLD cache system for {session_id}_{mesh_index}")
        cache_kwargs = {
            "mesh_count": mesh_count,
            "mesh_index": mesh_index,
        }
        if views:
            cache_kwargs["views"] = views
        
        provider_name = get_provider_name(llm_client)
        if evaluation_type == "multiview":
            cache_key = get_cache_key(
                "detailed_multiview", session_id, trial, provider_name, **cache_kwargs
            )
        else:
            cache_key = get_cache_key(
                "detailed", session_id, trial, provider_name, **cache_kwargs
            )
        
        cached_response = load_cached_response(cache_key)
        if cached_response:
            print(f"     ✅ OLD cache hit: {cache_key[:60]}...")
            return cached_response["response"]
        else:
            print(f"     ❌ OLD cache miss: {cache_key[:60]}...")
        return None
    
    # Use new cache system
    print(f"   🔄 Using NEW cache system for {session_id}_{mesh_index}")
    cache_manager = get_cache_manager()
    
    cache_kwargs = {
        "mesh_count": mesh_count,
        "mesh_index": mesh_index,
    }
    
    if evaluation_type == EvaluationType.MULTIVIEW_SCORING and views:
        cache_kwargs["views"] = views
    elif evaluation_type == EvaluationType.PAIRWISE_COMPARISON:
        cache_kwargs.update({
            "concept_session_id": concept_session_id,
            "model_a": model_a_name,
            "model_b": model_b_name,
        })
    
    cache_key = cache_manager.generate_cache_key(
        evaluation_type=evaluation_type,
        session_id=session_id,
        provider=llm_client.provider,
        trial=trial,
        **cache_kwargs
    )
    
    cached_response = load_llm_response(cache_key)
    if cached_response and cached_response.success and cached_response.parsed_json:
        print(f"     ✅ NEW cache hit: {cache_key[:60]}...")
        return cached_response.parsed_json
    else:
        print(f"     ❌ NEW cache miss: {cache_key[:60]}...")
    
    return None
    
# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def fetch_json(url: str, api_key: str) -> Dict:
    resp = requests.get(url, headers={**HEADERS, "x-api-key": api_key})
    if not resp.ok:
        print(f"❌ API Error {resp.status_code}: {url}")
        try:
            error_data = resp.json()
            print(f"   Error details: {error_data}")
        except:
            print(f"   Raw response: {resp.text}")
    resp.raise_for_status()
    return resp.json()


def download_image(url: str) -> Image.Image:
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def save_debug_query(
    debug_enabled: bool,
    concept_session_id: str,
    model_name: str,
    mesh_index: int,
    original_img_b64: str,
    render_img_b64: str,
    prompt_content: List[Dict],
    response_json: Dict,
    session_id: str,
    mesh_count: int,
) -> None:
    """Save debug information about the LLM query."""
    if not debug_enabled:
        return
    
    # Create debug folder structure
    debug_path = DEBUG_DIR / f"concept_{concept_session_id}" / f"model_{model_name}" / f"part_{mesh_index + 1}_of_{mesh_count}"
    debug_path.mkdir(parents=True, exist_ok=True)
    
    # Save concept image
    concept_img_data = base64.b64decode(original_img_b64)
    with open(debug_path / "concept_image.jpg", "wb") as f:
        f.write(concept_img_data)
    
    # Save render image
    render_img_data = base64.b64decode(render_img_b64)
    with open(debug_path / "render_image.jpg", "wb") as f:
        f.write(render_img_data)
    
    # Save prompt text (extract text content from the structured prompt)
    prompt_text = []
    for content_item in prompt_content:
        if content_item.get("type") == "text":
            prompt_text.append(content_item["text"])
        elif content_item.get("type") == "image":
            prompt_text.append(f"[IMAGE: {content_item['source']['media_type']}]")
    
    with open(debug_path / "prompt.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(prompt_text))
    
    # Save full structured prompt
    with open(debug_path / "full_prompt.json", "w", encoding="utf-8") as f:
        # Remove base64 data to keep file size reasonable
        clean_prompt = []
        for item in prompt_content:
            if item.get("type") == "image":
                clean_item = item.copy()
                clean_item["source"] = {
                    "type": item["source"]["type"],
                    "media_type": item["source"]["media_type"],
                    "data": "[BASE64_DATA_REMOVED]"
                }
                clean_prompt.append(clean_item)
            else:
                clean_prompt.append(item)
        json.dump(clean_prompt, f, indent=2)
    
    # Save response
    with open(debug_path / "response.json", "w", encoding="utf-8") as f:
        json.dump(response_json, f, indent=2)
    
    # Save metadata
    metadata = {
        "concept_session_id": concept_session_id,
        "render_session_id": session_id,
        "model_name": model_name,
        "mesh_index": mesh_index,
        "mesh_count": mesh_count,
        "timestamp": None  # Could add timestamp if needed
    }
    with open(debug_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   🐛 Debug data saved to: {debug_path}")

def call_claude(
    claude_key: str,
    original_img_b64: str,
    render_img_b64: str,
    session_id: str,
    mesh_count: int,
    mesh_index: int = 0,
    debug_enabled: bool = False,
    concept_session_id: str = "",
    model_name: str = "",
    trial: int = 0,
    use_cache: bool = True,
) -> Dict:
    """Send a single render vs. concept pair to Claude and return the parsed JSON."""

    # Check cache first
    if use_cache:
        cache_key = get_cache_key(
            "detailed", session_id, trial, "claude",
            mesh_count=mesh_count, mesh_index=mesh_index
        )
        cached_response = load_cached_response(cache_key)
        if cached_response:
            return cached_response["response"]

    client = anthropic.Anthropic(api_key=claude_key)
    
    # Create context about which part we're evaluating
    if mesh_count == 1:
        part_context = "This is a single-mesh reconstruction."
    else:
        part_context = f"This is part {mesh_index + 1} of {mesh_count} in a multi-part reconstruction. NOTE: Part numbering is arbitrary/random - part 1 doesn't necessarily correspond to any specific semantic part of the object."
    
    # Create the message content with both images and text
    content = [
        {
            "type": "text",
            "text": f"I will show you two images that you need to compare:\n1. A concept image (original reference)\n2. A 3D mesh render to be evaluated\n\n{part_context}\n\nPlease evaluate how well the 3D render matches the concept image using the following rubric:"
        },
        {
            "type": "text",
            "text": USER_RUBRIC
        },
        {
            "type": "text",
            "text": "CONCEPT IMAGE (Reference):"
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": original_img_b64,
            },
        },
        {
            "type": "text",
            "text": f"3D MESH RENDER (To be evaluated - {part_context.lower()}):"
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": render_img_b64,
            },
        },
        {
            "type": "text",
            "text": f"Now please evaluate how well this 3D mesh render matches the concept image.\n\nSession ID: {session_id}\nMesh count: {mesh_count}\nCurrent mesh: {mesh_index + 1} of {mesh_count}\n\nRemember: \n- If mesh_count == 1 and the object should logically be multi-part, set Part Coverage = 0\n- If the model appears untextured (gray/white), set Texture Quality = 0\n- Score all 4 dimensions on 0-10 scale"
        }
    ]

    try:
        message = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=256,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )
        
        final_text = message.content[0].text
        response_json = json.loads(final_text)
        
        save_debug_query(
            debug_enabled,
            concept_session_id,
            model_name,
            mesh_index,
            original_img_b64,
            render_img_b64,
            content,
            response_json,
            session_id,
            mesh_count,
        )
        
        # Save to cache
        if use_cache:
            metadata = {
                "session_id": session_id,
                "mesh_count": mesh_count,
                "mesh_index": mesh_index,
                "trial": trial,
                "concept_session_id": concept_session_id,
                "model_name": model_name
            }
            save_cached_response(cache_key, response_json, metadata)
        
        return response_json
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON returned by Claude: {final_text}") from exc
    except Exception as exc:
        raise RuntimeError(f"Error calling Claude API: {exc}") from exc


def call_llm_unified(
    llm_client,
    evaluation_type,  # EvaluationType when available
    concept_img_b64: str,
    render_images: Dict[str, str],  # For multiview: {"front": b64, "back": b64}, for single: {"render": b64}
    session_id: str,
    mesh_count: int,
    mesh_index: int = 0,
    debug_enabled: bool = False,
    concept_session_id: str = "",
    model_name: str = "",
    trial: int = 0,
    use_cache: bool = True,
    views: List[str] = None,
    model_a_name: str = "",
    model_b_name: str = "",
) -> Dict:
    """Universal LLM call function supporting multiple providers and evaluation types"""
    
    if not LLM_SYSTEM_AVAILABLE:
        # Fallback to original Claude implementation
        if evaluation_type == EvaluationType.DETAILED_SCORING:
            return call_claude(
                llm_client, concept_img_b64, render_images.get("render", ""),
                session_id, mesh_count, mesh_index, debug_enabled,
                concept_session_id, model_name, trial, use_cache
            )
        elif evaluation_type == EvaluationType.MULTIVIEW_SCORING:
            return call_claude_multiview(
                llm_client, concept_img_b64, render_images,
                session_id, mesh_count, mesh_index, debug_enabled,
                concept_session_id, model_name, trial, use_cache
            )
        else:
            raise ValueError(f"Unsupported evaluation type for fallback: {evaluation_type}")
    
    # Generate cache key
    cache_manager = get_cache_manager()
    
    cache_kwargs = {
        "mesh_count": mesh_count,
        "mesh_index": mesh_index,
    }
    
    if evaluation_type == EvaluationType.MULTIVIEW_SCORING and views:
        cache_kwargs["views"] = views
    elif evaluation_type == EvaluationType.PAIRWISE_COMPARISON:
        cache_kwargs.update({
            "concept_session_id": concept_session_id,
            "model_a": model_a_name,
            "model_b": model_b_name,
        })
    
    cache_key = cache_manager.generate_cache_key(
        evaluation_type=evaluation_type,
        session_id=session_id,
        provider=llm_client.provider,
        trial=trial,
        **cache_kwargs
    )
    
    # Check cache first
    if use_cache:
        cached_response = load_llm_response(cache_key)
        if cached_response and cached_response.success:
            if cached_response.parsed_json:
                print(f"   📋 ACTUAL CACHE HIT: {cache_key[:50]}...")
                return cached_response.parsed_json
    
    # Create context about which part we're evaluating
    if mesh_count == 1:
        part_context = "This is a single-mesh reconstruction."
    else:
        part_context = f"This is part {mesh_index + 1} of {mesh_count} in a multi-part reconstruction. NOTE: Part numbering is arbitrary/random - part 1 doesn't necessarily correspond to any specific semantic part of the object."
    
    # Format prompt based on evaluation type
    eval_type_str = evaluation_type.value if hasattr(evaluation_type, 'value') else evaluation_type
    
    if eval_type_str == "detailed_scoring":
        if LLM_SYSTEM_AVAILABLE:
            prompt_data = format_detailed_scoring_prompt(
                part_context=part_context,
                rubric=EVALUATION_RUBRIC,
                session_id=session_id,
                mesh_count=mesh_count,
                mesh_index=mesh_index,
            )
            
            # Prepare images for request
            images = [
                {"name": "concept", "data": concept_img_b64},
                {"name": "render", "data": render_images.get("render", "")}
            ]
        else:
            # Fallback handling - this shouldn't be called for visualization-only mode
            raise RuntimeError("Detailed scoring requires LLM system")
        
    elif eval_type_str == "multiview_scoring":
        if LLM_SYSTEM_AVAILABLE:
            # Use hybrid view section if we have mixed image sources (CSM + 3D mesh views)
            image_keys = list(render_images.keys())
            has_csm_render = "csm_render" in image_keys
            has_mesh_views = any(key.startswith("mesh_") for key in image_keys)
            
            if has_csm_render and has_mesh_views:
                # Hybrid case: use specialized hybrid prompt
                view_images_section = create_hybrid_view_images_section(render_images)
                views_list_str = f"CSM render + {len([k for k in image_keys if k.startswith('mesh_')])} 3D mesh views"
            else:
                # Standard multiview case: use original prompt
                view_images_section = create_view_images_section(views or image_keys)
                views_list_str = ", ".join(views or image_keys)
            
            prompt_data = format_multiview_scoring_prompt(
                part_context=part_context,
                rubric=EVALUATION_RUBRIC,
                session_id=session_id,
                mesh_count=mesh_count,
                mesh_index=mesh_index,
                views_list=views or image_keys,
                view_images_section=view_images_section,
            )
            
            # Prepare images for request
            images = [{"name": "concept", "data": concept_img_b64}]
            for view_name, view_b64 in render_images.items():
                images.append({"name": view_name, "data": view_b64})
        else:
            raise RuntimeError("Multiview scoring requires LLM system")
            
    elif eval_type_str == "pairwise_comparison":
        if LLM_SYSTEM_AVAILABLE:
            prompt_data = format_pairwise_comparison_prompt(
                model_a_name=model_a_name,
                model_b_name=model_b_name,
            )
            
            # Prepare images for request
            images = [
                {"name": "concept", "data": concept_img_b64},
                {"name": "render_a", "data": render_images.get("render_a", "")},
                {"name": "render_b", "data": render_images.get("render_b", "")}
            ]
        else:
            raise RuntimeError("Pairwise comparison requires LLM system")
    else:
        raise ValueError(f"Unsupported evaluation type: {eval_type_str}")
    
    # Create LLM request
    llm_request = LLMRequest(
        system_prompt=prompt_data["system_prompt"],
        user_prompt=prompt_data["user_prompt"],
        images=images,
        evaluation_type=evaluation_type,
        metadata={
            "session_id": session_id,
            "mesh_count": mesh_count,
            "mesh_index": mesh_index,
            "concept_session_id": concept_session_id,
            "model_name": model_name,
            "trial": trial,
        }
    )
    
    # Call LLM with retry logic for transient errors
    response = call_llm_with_retry(llm_client, llm_request, max_retries=3, base_delay=2.0)
    
    # Save to cache
    if use_cache:
        save_llm_response(cache_key, response)
    
    # Save debug information
    if debug_enabled:
        save_debug_query_unified(
            debug_enabled, concept_session_id, model_name, mesh_index,
            concept_img_b64, render_images, llm_request, response,
            session_id, mesh_count, evaluation_type
        )
    
    # Return parsed JSON or raise error
    if response.success and response.parsed_json:
        return response.parsed_json
    else:
        error_msg = response.error or f"Failed to parse JSON from response: {response.content}"
        raise RuntimeError(f"LLM API call failed: {error_msg}")


def save_debug_query_unified(
    debug_enabled: bool,
    concept_session_id: str,
    model_name: str,
    mesh_index: int,
    concept_img_b64: str,
    render_images: Dict[str, str],
    llm_request,  # LLMRequest when available
    llm_response,  # LLMResponse when available  
    session_id: str,
    mesh_count: int,
    evaluation_type,  # EvaluationType when available
) -> None:
    """Save debug information for unified LLM calls"""
    if not debug_enabled:
        return
    
    # Create debug folder structure
    debug_path = DEBUG_DIR / f"concept_{concept_session_id}" / f"model_{model_name}" / f"part_{mesh_index + 1}_of_{mesh_count}"
    debug_path.mkdir(parents=True, exist_ok=True)
    
    # Save concept image
    concept_img_data = base64.b64decode(concept_img_b64)
    with open(debug_path / "concept_image.jpg", "wb") as f:
        f.write(concept_img_data)
    
    # Save render images
    for image_name, image_b64 in render_images.items():
        image_data = base64.b64decode(image_b64)
        with open(debug_path / f"{image_name}_image.jpg", "wb") as f:
            f.write(image_data)
    
    # Save LLM request and response
    with open(debug_path / "llm_request.json", "w", encoding="utf-8") as f:
        # Create a clean version without base64 data
        eval_type_str = evaluation_type.value if hasattr(evaluation_type, 'value') else str(evaluation_type)
        clean_request = {
            "system_prompt": getattr(llm_request, 'system_prompt', 'N/A'),
            "user_prompt": getattr(llm_request, 'user_prompt', 'N/A'),
            "evaluation_type": eval_type_str,
            "metadata": getattr(llm_request, 'metadata', {}),
            "images": [{"name": img["name"], "data": "[BASE64_DATA_REMOVED]"} for img in getattr(llm_request, 'images', [])]
        }
        json.dump(clean_request, f, indent=2)
    
    with open(debug_path / "llm_response.json", "w", encoding="utf-8") as f:
        if hasattr(llm_response, 'to_dict'):
            json.dump(llm_response.to_dict(), f, indent=2)
        else:
            json.dump(str(llm_response), f, indent=2)
    
    print(f"   🐛 Debug data saved ({eval_type_str}): {debug_path}")


def call_claude_multiview(
    claude_key: str,
    original_img_b64: str,
    render_views: Dict[str, str],  # Dict of view_name -> base64_image
    session_id: str,
    mesh_count: int,
    mesh_index: int = 0,
    debug_enabled: bool = False,
    concept_session_id: str = "",
    model_name: str = "",
    trial: int = 0,
    use_cache: bool = True,
) -> Dict:
    """Send multiple mesh views vs. concept to Claude and return the parsed JSON."""

    # Check cache first 
    if use_cache:
        cache_key = get_cache_key(
            "detailed_multiview", session_id, trial, "claude",
            mesh_count=mesh_count, mesh_index=mesh_index,
            views=list(render_views.keys())
        )
        cached_response = load_cached_response(cache_key)
        if cached_response:
            return cached_response["response"]

    client = anthropic.Anthropic(api_key=claude_key)
    
    # Create context about which part we're evaluating
    if mesh_count == 1:
        part_context = "This is a single-mesh reconstruction."
    else:
        part_context = f"This is part {mesh_index + 1} of {mesh_count} in a multi-part reconstruction."
    
    # Build content with concept image and multiple mesh views
    content = [
        {
            "type": "text",
            "text": f"I will show you a concept image and multiple views of a 3D mesh reconstruction.\n\n{part_context}\n\nPlease evaluate how well the 3D mesh matches the concept image using the following rubric:"
        },
        {
            "type": "text",
            "text": USER_RUBRIC
        },
        {
            "type": "text",
            "text": "CONCEPT IMAGE (Reference):"
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": original_img_b64,
            },
        }
    ]
    
    # Add each view
    for view_name, view_b64 in render_views.items():
        content.extend([
            {
                "type": "text",
                "text": f"3D MESH {view_name.upper()} VIEW:"
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": view_b64,
                },
            }
        ])
    
    content.append({
        "type": "text",
        "text": f"Evaluate this 3D reconstruction using all available views. Consider front/back views together for complete assessment.\n\nSession ID: {session_id}\nMesh count: {mesh_count}\nCurrent mesh: {mesh_index + 1} of {mesh_count}\n\nViews shown: {', '.join(render_views.keys())}"
    })

    try:
        message = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=256,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        
        final_text = message.content[0].text
        response_json = json.loads(final_text)
        
        # Save debug info with multiple views
        if debug_enabled:
            debug_path = DEBUG_DIR / f"concept_{concept_session_id}" / f"model_{model_name}" / f"part_{mesh_index + 1}_of_{mesh_count}"
            debug_path.mkdir(parents=True, exist_ok=True)
            
            # Save all view images
            for view_name, view_b64 in render_views.items():
                view_img_data = base64.b64decode(view_b64)
                with open(debug_path / f"render_{view_name}.jpg", "wb") as f:
                    f.write(view_img_data)
            
            print(f"   🐛 Debug data saved (multiview): {debug_path}")
        
        # Save to cache
        if use_cache:
            metadata = {
                "session_id": session_id,
                "mesh_count": mesh_count,
                "mesh_index": mesh_index,
                "trial": trial,
                "views": list(render_views.keys()),
                "concept_session_id": concept_session_id,
                "model_name": model_name
            }
            save_cached_response(cache_key, response_json, metadata)
        
        return response_json
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON returned by Claude: {final_text}") from exc
    except Exception as exc:
        raise RuntimeError(f"Error calling Claude API: {exc}") from exc


# -----------------------------------------------------------------------------
# Core leaderboard logic
# -----------------------------------------------------------------------------

def extract_concept_session_from_session(session_id: str, api_key: str) -> str:
    """Extract concept session ID from a session"""
    try:
        url = f"https://api.csm.ai/v3/sessions/{session_id}"
        session_data = fetch_json(url, api_key)
        
        # For most sessions, the concept image comes from input.image
        input_image = session_data.get("input", {}).get("image", {})
        if input_image and input_image.get("_id"):
            # For direct image-to-3d sessions, this might be the original image upload
            return input_image["_id"]
        
        # For chat-to-3d or derived sessions, look for parent session or concept reference
        # For now, use the session itself as concept if no parent found
        return session_id
        
    except Exception as e:
        print(f"   ⚠️  Failed to extract concept from session {session_id}: {e}")
        # Fallback to using session as its own concept
        return session_id

def extract_model_info_from_session(session_id: str, api_key: str) -> str:
    """Extract model information from CSM session details"""
    try:
        url = f"https://api.csm.ai/v3/sessions/{session_id}"
        session_data = fetch_json(url, api_key)
        
        # Extract model information from session settings
        settings = session_data.get("input", {}).get("settings", {})
        session_type = session_data.get("type", "unknown")
        
        # Build model name from settings
        geometry_model = settings.get("geometry_model", "unknown")
        texture_model = settings.get("texture_model", "none")
        decomposition_model = settings.get("decomposition_model")
        
        # Create descriptive model name
        if session_type == "image_to_kit":
            model_name = f"csm-kit-{texture_model}"
        elif decomposition_model:
            model_name = f"csm-{decomposition_model}-{texture_model}"
        else:
            model_name = f"csm-{geometry_model}-{texture_model}"
        
        return model_name
        
    except Exception as e:
        print(f"   ⚠️  Failed to extract model info for session {session_id}: {e}")
        # Fallback to session ID suffix
        return f"csm-unknown-{session_id.split('_')[-1]}"

def load_session_ids_from_job_tracking(job_tracking_path: Path) -> List[str]:
    """Extract all session IDs from job_tracking.json"""
    if not job_tracking_path.exists():
        print(f"⚠️  Job tracking file not found: {job_tracking_path}")
        return []
    
    try:
        with job_tracking_path.open("r") as f:
            job_data = json.load(f)
        
        session_ids = set()
        
        # Handle different possible structures
        if isinstance(job_data, dict):
            # If it's a dict, look for session_id fields recursively
            def extract_session_ids(obj):
                if isinstance(obj, dict):
                    if "session_id" in obj:
                        session_id = obj["session_id"]
                        if session_id and session_id.startswith("SESSION_"):
                            session_ids.add(session_id)
                    for value in obj.values():
                        extract_session_ids(value)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_session_ids(item)
            
            extract_session_ids(job_data)
        elif isinstance(job_data, list):
            # If it's a list, extract session_id from each entry
            for entry in job_data:
                if isinstance(entry, dict) and "session_id" in entry:
                    session_id = entry["session_id"]
                    if session_id and session_id.startswith("SESSION_"):
                        session_ids.add(session_id)
        
        return sorted(list(session_ids))
        
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  Failed to read job tracking file: {e}")
        return []

def merge_sessions_with_human_evals(manual_entries: List[Dict], auto_session_ids: List[str]) -> List[Dict]:
    """Merge manual entries (with human_eval) and auto-discovered sessions"""
    
    # Extract session IDs that already have manual entries
    manual_session_ids = set()
    for entry in manual_entries:
        if "session_id" in entry:
            manual_session_ids.add(entry["session_id"])
    
    # Create combined list: manual entries first, then auto-discovered ones
    combined_entries = manual_entries.copy()
    
    # Add auto-discovered sessions that don't already have manual entries
    for session_id in auto_session_ids:
        if session_id not in manual_session_ids:
            combined_entries.append({"session_id": session_id})
    
    return combined_entries

def preview_evaluation_plan(session_entries: List[Dict], api_key: str, human_eval_only: bool = False) -> bool:
    """Preview all sessions to be evaluated and ask for confirmation"""
    
    print(f"\n📋 EVALUATION PREVIEW")
    print("=" * 50)
    print(f"Total sessions to evaluate: {len(session_entries)}")
    
    if human_eval_only:
        print(f"  • Human-eval-only mode: All {len(session_entries)} sessions have human evaluations")
        print(f"  • Auto-discovery disabled")
    else:
        # Separate manual vs auto-discovered
        manual_count = sum(1 for entry in session_entries if entry.get("human_eval"))
        auto_count = len(session_entries) - manual_count
        
        print(f"  • Manual entries with human_eval: {manual_count}")
        print(f"  • Auto-discovered from job_tracking: {auto_count}")
    
    # Group by concept for better preview
    print(f"\n🔍 Session Details:")
    concept_groups = {}
    failed_sessions = 0
    
    for i, entry in enumerate(session_entries):
        session_id = entry["session_id"]
        has_human_eval = bool(entry.get("human_eval"))
        
        try:
            # Get basic session info for preview
            session_url = f"https://api.csm.ai/v3/sessions/{session_id}"
            session_data = fetch_json(session_url, api_key)
            
            # Check if session has required fields
            if "input" not in session_data:
                print(f"   ⚠️  Session {session_id}: Missing input data (possibly incomplete)")
                failed_sessions += 1
                continue
                
            input_data = session_data["input"]
            if "image" not in input_data or "data" not in input_data["image"]:
                print(f"   ⚠️  Session {session_id}: Missing image data")
                failed_sessions += 1
                continue
                
            if "image_url" not in input_data["image"]["data"]:
                print(f"   ⚠️  Session {session_id}: Missing image URL")
                failed_sessions += 1
                continue
            
            # Extract concept image and model info
            concept_img_url = input_data["image"]["data"]["image_url"]
            model_name = extract_model_info_from_session(session_id, api_key)
            
            # Group by concept
            if concept_img_url not in concept_groups:
                concept_groups[concept_img_url] = []
            concept_groups[concept_img_url].append({
                "session_id": session_id,
                "model_name": model_name,  # Use actual model name without session suffix
                "has_human_eval": has_human_eval
            })
            
        except Exception as e:
            print(f"   ⚠️  Session {session_id}: {str(e)}")
            failed_sessions += 1
            continue
    
    # Print grouped preview
    for concept_img, sessions in concept_groups.items():
        concept_name = concept_img.split("/")[-1].split("?")[0] if concept_img else "unknown"
        print(f"\n  📸 Concept: {concept_name} ({len(sessions)} models)")
        
        for session_info in sessions:
            human_indicator = "👤 HumanEval" if session_info["has_human_eval"] else "🤖 AutoOnly"
            print(f"    • {session_info['model_name']:<25} {human_indicator}")
    
    # Show summary
    successful_sessions = len(session_entries) - failed_sessions
    print(f"\n📊 Session Validation Summary:")
    print(f"  ✅ Successful: {successful_sessions}")
    print(f"  ❌ Failed/Incomplete: {failed_sessions}")
    
    if failed_sessions > 0:
        print(f"  💡 Failed sessions will be automatically skipped during evaluation")
    
    # Ask for confirmation
    print(f"\n🚀 Ready to evaluate {successful_sessions} valid sessions across {len(concept_groups)} concepts.")
    print("   This will make LLM API calls and may take some time.")
    
    while True:
        response = input("\nProceed with evaluation? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no', '']:
            print("❌ Evaluation cancelled.")
            return False
        else:
            print("Please enter 'y' or 'n'")

def evaluate_candidate(
    llm_client,  # Changed from claude_key to llm_client
    api_key: str,
    concept_session_id: str,
    concept_img_url: str,
    candidate_name: str,
    candidate_session_ids: List[str],
    debug_enabled: bool = False,
    trial: int = 0,
    use_cache: bool = True,
    use_multiview: bool = False,  # New parameter
    views: List[str] = ["front", "back"],  # New parameter
) -> List[float]:
    """Return mean scores across all renders for this candidate as [StructuralForm, PartCoverage, SurfaceDetail, TextureQuality]."""

    all_scores: List[List[int]] = []  # List of score arrays
    concept_b64 = None  # Only download if needed

    for sid in candidate_session_ids:
        # 🚀 ULTRA EARLY CACHE CHECK - Check cache BEFORE any expensive API calls
        # Try common mesh counts (1-10) to see if we can find cached results
        evaluation_type = EvaluationType.MULTIVIEW_SCORING if LLM_SYSTEM_AVAILABLE else "multiview"
        session_cached_results = []
        session_fully_cached = False
        
        print(f"   🔍 Checking cache for session {sid}...")
        
        # 🔍 SMART CACHE CHECK: Look for actual cached files for this session pattern
        session_cached_results = check_session_cache_pattern(
            llm_client, sid, concept_session_id, candidate_name, trial, use_cache
        )
        
        if session_cached_results:
            session_fully_cached = True
            print(f"   💾 FULL SESSION CACHED: {sid} ({len(session_cached_results)} meshes) - skipping ALL processing!")
        else:
            print(f"   🔧 Session {sid} not fully cached - doing expensive processing...")
        
        if session_fully_cached:
            # Use all cached results and skip everything
            for cached_result in session_cached_results:
                all_scores.append(cached_result["score_array"])
            continue
        
        # Not fully cached - need to do expensive processing
        print(f"   🔧 Session {sid} not fully cached - doing expensive processing...")
        url = f"https://api.csm.ai/v3/sessions/{sid}"
        sess = fetch_json(url, api_key)
        output_data = sess.get("output", {})
        mesh_data = output_data.get("meshes") or output_data.get("part_meshes") or []
        if not mesh_data:
            print(f"   ⚠️  Session {sid} has no meshes/part_meshes or is not complete — skipping.")
            continue
        mesh_count = len(mesh_data)
        
        # Not all cached - process normally
        for mesh_index, mesh in enumerate(mesh_data):
            # 🚀 EARLY CACHE CHECK - Skip expensive rendering if we have cached results
            cached_result = check_early_cache(
                llm_client,
                evaluation_type,
                sid,
                mesh_count,
                mesh_index,
                concept_session_id,
                candidate_name,
                trial,
                use_cache,
                views,
            )
            
            if cached_result:
                all_scores.append(cached_result["score_array"])
                print(f"      💾 EARLY CACHE HIT: {evaluation_type}_{sid}_{mesh_index}")
                continue
            else:
                print(f"      🔍 EARLY CACHE MISS: {evaluation_type}_{sid}_{mesh_index} - processing...")
            
            # 🖼️ LAZY CONCEPT IMAGE LOADING - Only download when we need it for LLM call
            if concept_b64 is None:
                concept_img = download_image(concept_img_url)
                concept_b64 = img_to_b64(concept_img)
            
            # HYBRID APPROACH: Collect ALL available images (only if not cached)
            all_render_images = {}
            image_sources = []
            
            # 1. Always get CSM's pre-rendered image
            render_url = mesh.get('data', {}).get('image_url')
            if render_url:
                render_img = download_image(render_url)
                render_b64 = img_to_b64(render_img)
                all_render_images["csm_render"] = render_b64
                image_sources.append("CSM render")
            
            # 2. Try to get 3D mesh views (front/back) if available
            if MESH_RENDERER_AVAILABLE:
                mesh_url = mesh.get('data', {}).get('glb_url') or mesh.get('data', {}).get('obj_url')
                
                if mesh_url:
                    try:
                        # Render multiple views from 3D mesh
                        render_views_b64 = render_mesh_views_to_base64(mesh_url, views, resolution=512)
                        
                        if render_views_b64:
                            # Add all rendered views to our collection
                            for view_name, view_b64 in render_views_b64.items():
                                all_render_images[f"mesh_{view_name}"] = view_b64
                                image_sources.append(f"mesh {view_name}")
                    except MeshRenderError as e:
                        print(f"      ⚠️  3D mesh rendering failed: {e}")
                    except Exception as e:
                        print(f"      ⚠️  Unexpected 3D mesh error: {e}")
            
            # 3. If no images available, error out
            if not all_render_images:
                raise RuntimeError(f"Mesh {mesh_index} in session {sid} has no render images available. Cannot evaluate.")
            
            # 4. Use hybrid evaluation with all available images
            score_json = call_llm_unified(
                llm_client,
                evaluation_type,
                concept_b64,
                all_render_images,
                sid,
                mesh_count,
                mesh_index,
                debug_enabled,
                concept_session_id,
                candidate_name,
                trial,
                use_cache,
                list(all_render_images.keys()),  # Pass all view names
            )
            all_scores.append(score_json["score_array"])
            print(f"      🎯 Hybrid evaluation: {', '.join(image_sources)} ({len(all_render_images)} images)")

    if not all_scores:
        return [0.0, 0.0, 0.0, 0.0]
    
    # For multi-part models (kits), use best 80% strategy for aggregation
    if len(all_scores) > 2:  # Multi-part model
        # Calculate mean for each dimension across all parts
        dimension_scores = []
        for dim_idx in range(4):  # 4 dimensions
            dim_scores = [scores[dim_idx] for scores in all_scores]
            
            # Identify failed parts (any score <= 2 is considered failed for that dimension)
            failed_count = sum(1 for score in dim_scores if score <= 2)
            failure_rate = failed_count / len(dim_scores)
            
            if failure_rate >= 0.5:  # 50%+ parts failed for this dimension
                dimension_scores.append(0.0)  # Complete failure for this dimension
            elif failure_rate >= 0.3:  # 30%+ parts failed
                # Use median to be more forgiving of outliers
                dimension_scores.append(float(sorted(dim_scores)[len(dim_scores)//2]))
            else:
                # Use best 80% for this dimension
                sorted_dim_scores = sorted(dim_scores, reverse=True)
                best_count = max(1, int(len(sorted_dim_scores) * 0.8))
                best_scores = sorted_dim_scores[:best_count]
                dimension_scores.append(sum(best_scores) / len(best_scores))
        
        return dimension_scores
    else:
        # Single part or small multi-part - use regular average
        num_dimensions = len(all_scores[0])
        avg_scores = []
        for dim_idx in range(num_dimensions):
            dim_total = sum(scores[dim_idx] for scores in all_scores)
            avg_scores.append(dim_total / len(all_scores))
        return avg_scores


def run_leaderboard(human_eval_json_path: Path, api_key: str, llm_client, debug_enabled: bool = False, trial: int = 0, no_cache: bool = False, preview: bool = True, job_tracking_path: Path = None, human_eval_only: bool = False, use_multiview: bool = False, views: List[str] = ["front", "back"]):
    # Get provider name for file naming
    provider_name = get_provider_name(llm_client)
    
    # Load manual entries from leaderboard_models.json
    manual_entries = json.loads(Path(human_eval_json_path).read_text())
    
    if human_eval_only:
        # Filter to only include entries with complete human evaluations
        entries_with_human_eval = []
        for entry in manual_entries:
            human_eval = entry.get("human_eval", {})
            if isinstance(human_eval, dict) and human_eval:
                # Check if it has at least some dimension scores
                dimension_names = ["StructuralForm", "PartCoverage", "SurfaceDetail", "TextureQuality"]
                has_scores = any(dim in human_eval and human_eval[dim] is not None for dim in dimension_names)
                if has_scores:
                    entries_with_human_eval.append(entry)
        
        all_session_entries = entries_with_human_eval
        auto_session_ids = []  # No auto-discovery in human-eval-only mode
        
        print(f"📂 Loaded {len(manual_entries)} manual entries from {human_eval_json_path}")
        print(f"👤 Human-eval-only mode: filtered to {len(all_session_entries)} entries with human evaluations")
        print(f"⏭️  Skipping auto-discovery from job_tracking.json")
        
        if len(all_session_entries) == 0:
            print("❌ Error: No sessions with human evaluations found!")
            print("   Make sure your leaderboard_models.json file contains entries with 'human_eval' data.")
            print("   Example entry:")
            print('   {"session_id": "SESSION_123", "human_eval": {"StructuralForm": 8.0, "PartCoverage": 9.0, ...}}')
            return None
    else:
        # Auto-discover sessions from job_tracking.json
        if job_tracking_path is None:
            job_tracking_path = Path("job_tracking.json")
        
        auto_session_ids = load_session_ids_from_job_tracking(job_tracking_path)
        
        # Merge manual and auto-discovered sessions
        all_session_entries = merge_sessions_with_human_evals(manual_entries, auto_session_ids)
        
        print(f"📂 Loaded {len(manual_entries)} manual entries from {human_eval_json_path}")
        print(f"🔍 Found {len(auto_session_ids)} sessions in {job_tracking_path}")
        print(f"📊 Total unique sessions: {len(all_session_entries)}")
    
    # Preview evaluation plan
    if preview:
        if not preview_evaluation_plan(all_session_entries, api_key, human_eval_only):
            return None  # User cancelled
    
    # Continue with existing evaluation logic using all_session_entries
    overall_results: Dict[str, Dict[str, List[float]]] = {}
    human_evals: Dict[str, Dict[str, List[float]]] = {}
    all_model_scores: Dict[str, List[ModelScore]] = {}

    dimension_names = ["StructuralForm", "PartCoverage", "SurfaceDetail", "TextureQuality"]

    print("\n🎯 Phase 1: Detailed 4D Evaluation")
    print("=" * 50)

    # Group sessions by concept image for organization
    concept_groups = {}
    print(f"\n[+] Processing {len(all_session_entries)} session entries...")
    
    for entry in all_session_entries:
        session_id = entry.get("session_id")
        if not session_id:
            raise ValueError("Entry must contain 'session_id' field")
        
        # Get session details to find concept image
        session_url = f"https://api.csm.ai/v3/sessions/{session_id}"
        session_data = fetch_json(session_url, api_key)
        concept_img_url = session_data["input"]["image"]["data"]["image_url"]
        
        # Use concept image URL as grouping key (multiple sessions might share same concept)
        if concept_img_url not in concept_groups:
            concept_groups[concept_img_url] = {
                "concept_session_id": session_id,  # Use first session as representative
                "entries": []
            }
        concept_groups[concept_img_url]["entries"].append(entry)

    # Phase 1: Detailed scoring evaluation (updated for flat structure)
    total_evaluations = sum(len(group_data["entries"]) for group_data in concept_groups.values())
    print(f"\n📊 Starting evaluation of {total_evaluations} models across {len(concept_groups)} concepts...")
    
    overall_progress = tqdm(total=total_evaluations, desc="🎯 Overall Progress", unit="model", position=0)
    
    for concept_idx, (concept_img_url, group_data) in enumerate(concept_groups.items(), 1):
        concept_session_id = group_data["concept_session_id"]
        entries = group_data["entries"]
        
        print(f"\n[+] Concept {concept_idx}/{len(concept_groups)}: {concept_session_id} ({len(entries)} models)")

        per_candidate: Dict[str, List[float]] = {}
        human_per_candidate: Dict[str, List[float]] = {}
        concept_model_scores: List[ModelScore] = []
        
        for entry in tqdm(entries, desc=f"Concept {concept_idx}", unit="model", position=1, leave=False):
            session_id = entry.get("session_id")
            human_eval = entry.get("human_eval", {})
            
            # Auto-extract model name from CSM API
            model_name = extract_model_info_from_session(session_id, api_key)
            
            # Use the actual model name without artificial session suffixes
            # If there are multiple sessions with the same model on the same concept,
            # they will be averaged together (which is the correct behavior)

            scores = evaluate_candidate(
                llm_client,
                api_key,
                concept_session_id,
                concept_img_url,
                model_name,
                [session_id],  # Single session ID in list
                debug_enabled,
                trial,
                not no_cache,
                use_multiview,
                views,
            )
            
            # Handle potential model name conflicts by aggregating multiple sessions of same model
            if model_name in per_candidate:
                # Average with existing scores if this model already evaluated on this concept
                existing_scores = per_candidate[model_name]
                averaged_scores = [(existing + new) / 2 for existing, new in zip(existing_scores, scores)]
                per_candidate[model_name] = averaged_scores
            else:
                per_candidate[model_name] = scores
            
            # Convert human_eval dict to list in same order as dimension_names
            if isinstance(human_eval, dict) and human_eval:
                human_scores = []
                missing_dimensions = []
                
                for dim in dimension_names:
                    if dim in human_eval and human_eval[dim] is not None:
                        human_scores.append(float(human_eval[dim]))
                    else:
                        human_scores.append(None)  # Use None for missing/incomplete scores
                        missing_dimensions.append(dim)
                
                # Handle aggregation if model already has human scores
                if model_name in human_per_candidate:
                    # Average human scores if there are multiple sessions with same model
                    existing_human = human_per_candidate[model_name]
                    averaged_human = []
                    for existing, new in zip(existing_human, human_scores):
                        if existing is not None and new is not None:
                            averaged_human.append((existing + new) / 2)
                        elif new is not None:
                            averaged_human.append(new)
                        else:
                            averaged_human.append(existing)
                    human_per_candidate[model_name] = averaged_human
                else:
                    human_per_candidate[model_name] = human_scores
                
                # Calculate average only from available scores
                available_scores = [score for score in human_scores if score is not None]
                if available_scores:
                    human_avg = sum(available_scores) / len(available_scores)
                    
                    # Create display string showing available vs missing
                    human_detail_parts = []
                    for name, score in zip(dimension_names, human_scores):
                        if score is not None:
                            human_detail_parts.append(f"{name}:{score:.1f}")
                        else:
                            human_detail_parts.append(f"{name}:N/A")
                    
                    human_detail = " | ".join(human_detail_parts)
                    
                    if missing_dimensions:
                        human_str = f"avg:{human_avg:.1f} (partial: {len(available_scores)}/{len(dimension_names)}) [{human_detail}]"
                    else:
                        human_str = f"avg:{human_avg:.1f} [{human_detail}]"
                else:
                    human_str = "N/A (no scores provided)"
            else:
                if model_name not in human_per_candidate:
                    human_per_candidate[model_name] = [None] * 4
                human_str = "N/A"
            
            # Only create/update ModelScore if this is a new model or we need to update it
            human_scores_for_avg = [score for score in human_per_candidate[model_name] if score is not None]
            human_avg_score = sum(human_scores_for_avg) / len(human_scores_for_avg) if human_scores_for_avg else 0.0
            
            # Check if we already have this model in concept_model_scores
            existing_model_score = None
            for ms in concept_model_scores:
                if ms.model_name == model_name:
                    existing_model_score = ms
                    break
            
            if existing_model_score:
                # Update existing ModelScore with aggregated data
                existing_model_score.session_ids.append(session_id)
                existing_model_score.scores = per_candidate[model_name]  # Use aggregated scores
                existing_model_score.human_scores = human_per_candidate[model_name]
                existing_model_score.avg_score = sum(existing_model_score.scores) / len(existing_model_score.scores)
                existing_model_score.human_avg_score = human_avg_score
            else:
                # Create new ModelScore
                model_score = ModelScore(
                    model_name=model_name,
                    session_ids=[session_id],
                    concept_session_id=concept_session_id,
                    scores=per_candidate[model_name],  # Use the potentially aggregated scores
                    human_scores=human_per_candidate[model_name],
                    avg_score=sum(per_candidate[model_name]) / len(per_candidate[model_name]),
                    human_avg_score=human_avg_score
                )
                concept_model_scores.append(model_score)
            
            # Display scores nicely (only show when model is first encountered or updated)
            if not existing_model_score:  # Only print for new models
                score_strs = [f"{name}:{score:.1f}" for name, score in zip(dimension_names, per_candidate[model_name])]
                avg_score = sum(per_candidate[model_name]) / len(per_candidate[model_name])
                print(f"    • {model_name:<20} → avg:{avg_score:.1f} [{', '.join(score_strs)}]")
                print(f"      Human: {human_str}")
            elif len(existing_model_score.session_ids) > 1:  # Print update for aggregated models
                score_strs = [f"{name}:{score:.1f}" for name, score in zip(dimension_names, per_candidate[model_name])]
                avg_score = sum(per_candidate[model_name]) / len(per_candidate[model_name])
                print(f"    • {model_name:<20} → avg:{avg_score:.1f} [{', '.join(score_strs)}] (aggregated from {len(existing_model_score.session_ids)} sessions)")
                print(f"      Human: {human_str}")
            
            # Update overall progress
            overall_progress.update(1)

        overall_results[concept_session_id] = per_candidate
        human_evals[concept_session_id] = human_per_candidate
        all_model_scores[concept_session_id] = concept_model_scores

    # Close the progress bar
    overall_progress.close()

    # Phase 2: Generate comprehensive analysis
    print(f"\n📊 Phase 2: Comprehensive Analysis")
    print("=" * 50)
    
    results = LeaderboardResults(
        model_scores=all_model_scores,
        dimension_names=dimension_names
    )
    
    # Generate all visualizations and reports
    provider_results_dir = Path(f"results_{provider_name}")
    generate_comprehensive_analysis(results, provider_results_dir, provider_name)

    # Print traditional leaderboard (4D scores)
    print("\n===== Traditional 4D Score Leaderboard =====")
    for concept_id, scores_dict in overall_results.items():
        # Rank by average score
        avg_scores = {model: sum(scores)/len(scores) for model, scores in scores_dict.items()}
        top_model = max(avg_scores, key=avg_scores.get)
        print(f"{concept_id}: winner = {top_model} ({avg_scores[top_model]:.1f} avg)")
        
        for model in sorted(scores_dict.keys(), key=lambda x: -avg_scores[x]):
            scores = scores_dict[model]
            human_scores = human_evals[concept_id].get(model, [None] * 4)
            
            # Calculate human average only from available scores
            available_human_scores = [score for score in human_scores if score is not None]
            human_avg = sum(available_human_scores) / len(available_human_scores) if available_human_scores else None
            
            avg_score = sum(scores) / len(scores)
            score_detail = " | ".join([f"{name}:{score:.1f}" for name, score in zip(dimension_names, scores)])
            
            # Create human detail with N/A for missing scores
            human_detail_parts = []
            for name, score in zip(dimension_names, human_scores):
                if score is not None:
                    human_detail_parts.append(f"{name}:{score:.1f}")
                else:
                    human_detail_parts.append(f"{name}:N/A")
            human_detail = " | ".join(human_detail_parts)
            
            # Show if human eval is partial
            if None in human_scores:
                available_count = sum(1 for score in human_scores if score is not None)
                if available_count > 0:
                    human_avg_str = f"avg:{human_avg:.1f} (partial: {available_count}/{len(dimension_names)})"
                else:
                    human_avg_str = "avg:N/A"
            else:
                human_avg_str = f"avg:{human_avg:.1f}"
            
            print(f"   {model:<20} {provider_name.title()}: avg:{avg_score:.1f} [{score_detail}]")
            print(f"   {'':<20} Human:  {human_avg_str} [{human_detail}]")

    # Write enhanced CSV with all dimensions
    csv_path = Path(f"leaderboard_results_{provider_name}.csv")
    with csv_path.open("w", encoding="utf-8") as f:
        claude_cols = ",".join([f"claude_{dim}" for dim in dimension_names])
        human_cols = ",".join([f"human_{dim}" for dim in dimension_names])
        header = f"concept_session,model,claude_avg,{claude_cols},human_avg,{human_cols}\n"
        f.write(header)
        for concept_id, scores_dict in overall_results.items():
            for model, scores in scores_dict.items():
                human_scores = human_evals[concept_id].get(model, [None] * 4)
                claude_avg = sum(scores) / len(scores)
                
                # Calculate human average only from available scores
                available_human_scores = [score for score in human_scores if score is not None]
                human_avg = sum(available_human_scores) / len(available_human_scores) if available_human_scores else None
                
                claude_values = ",".join([f"{score:.1f}" for score in scores])
                
                # Format human values, using N/A for missing scores
                human_values = ",".join([f"{score:.1f}" if score is not None else "N/A" for score in human_scores])
                human_avg_str = f"{human_avg:.1f}" if human_avg is not None else "N/A"
                
                f.write(f"{concept_id},{model},{claude_avg:.1f},{claude_values},{human_avg_str},{human_values}\n")
    
    print(f"\nResults written to {csv_path.resolve()}")
    print(f"Detailed analysis and visualizations saved to {provider_results_dir.resolve()}/")
    
    return results


# -----------------------------------------------------------------------------
# Config management
# -----------------------------------------------------------------------------

def load_config() -> Dict[str, str]:
    """Load API keys from local config file."""
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not read {CONFIG_FILE}. Will ask for keys again.")
            return {}
    return {}

def save_config(config: Dict[str, str]) -> None:
    """Save API keys to local config file."""
    try:
        with CONFIG_FILE.open("w") as f:
            json.dump(config, f, indent=2)
        print(f"✓ API keys saved to {CONFIG_FILE}")
        print(f"  (This file is excluded from git for security)")
    except IOError as e:
        print(f"Warning: Could not save config to {CONFIG_FILE}: {e}")

def get_api_keys(llm_provider: LLMProvider = None) -> Tuple[str, str]:
    """Get API keys from config file or prompt user."""
    config = load_config()
    
    # Check if keys are already available
    csm_key = config.get("csm_api_key") or os.getenv("CSM_API_KEY")
    
    # Handle different LLM providers
    if not LLM_SYSTEM_AVAILABLE or llm_provider is None:
        # Fallback to Claude
        llm_key = config.get("claude_api_key") or os.getenv("CLAUDE_KEY")
        provider_name = "Claude"
        provider_url = "https://console.anthropic.com/"
        config_key = "claude_api_key"
    else:
        if llm_provider == LLMProvider.CLAUDE:
            llm_key = config.get("claude_api_key") or os.getenv("CLAUDE_KEY")
            provider_name = "Claude"
            provider_url = "https://console.anthropic.com/"
            config_key = "claude_api_key"
        elif llm_provider == LLMProvider.OPENAI:
            llm_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            provider_name = "OpenAI"
            provider_url = "https://platform.openai.com/api-keys"
            config_key = "openai_api_key"
        elif llm_provider == LLMProvider.GEMINI:
            llm_key = config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
            provider_name = "Gemini"
            provider_url = "https://aistudio.google.com/app/apikey"
            config_key = "gemini_api_key"
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")
    
    need_save = False
    
    # Prompt for missing keys
    if not csm_key:
        print("\n🔑 CSM API Key needed:")
        print("   See README.md for detailed instructions:")
        print("   1. Go to https://3d.csm.ai/")
        print("   2. Click Profile Settings (bottom left)")
        print("   3. Navigate to Settings → Developer Settings")
        print("   4. Copy your API key")
        csm_key = getpass.getpass("   Enter CSM API key: ").strip()
        config["csm_api_key"] = csm_key
        need_save = True
    
    if not llm_key:
        print(f"\n🔑 {provider_name} API Key needed:")
        print(f"   Get yours from: {provider_url}")
        llm_key = getpass.getpass(f"   Enter {provider_name} API key: ").strip()
        config[config_key] = llm_key
        need_save = True
    
    # Save config if we got new keys
    if need_save:
        save_config(config)
        print()
    
    return csm_key, llm_key

def create_llm_client_from_config(llm_provider: LLMProvider, llm_key: str):
    """Create LLM client instance"""
    if not LLM_SYSTEM_AVAILABLE:
        # Return the key for backward compatibility
        return llm_key
    
    # Validate provider availability
    if not validate_provider_availability(llm_provider):
        available_providers = get_available_llm_providers()
        install_cmd = get_provider_installation_command(llm_provider)
        
        print(f"❌ {llm_provider.value} is not available")
        print(f"   Install with: {install_cmd}")
        
        if available_providers:
            print(f"   Available providers: {[p.value for p in available_providers]}")
        else:
            print("   No LLM providers are available!")
        
        raise LLMError(f"LLM provider {llm_provider.value} not available")
    
    # Create and test client
    client = create_llm_client(llm_provider, llm_key)
    print(f"✅ {llm_provider.value} client initialized")
    
    return client




# -----------------------------------------------------------------------------
# Comprehensive Analytics & Visualizations
# -----------------------------------------------------------------------------

def generate_comprehensive_analysis(results: LeaderboardResults, output_dir: Path, provider_name: str = "claude") -> None:
    """Generate all analysis reports and visualizations"""
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🔍 Generating comprehensive analysis in {output_dir}...")
    
    # Create data frames for analysis
    model_data = []
    for concept_id, model_scores in results.model_scores.items():
        for score in model_scores:
            # Handle None values in human scores for analytics
            human_scores_clean = []
            for human_score in score.human_scores:
                if human_score is not None:
                    human_scores_clean.append(human_score)
                else:
                    human_scores_clean.append(float('nan'))  # Use NaN for missing data in pandas
            
            model_data.append({
                "concept_id": concept_id,
                "model_name": score.model_name,
                "StructuralForm": score.scores[0],
                "PartCoverage": score.scores[1], 
                "SurfaceDetail": score.scores[2],
                "TextureQuality": score.scores[3],
                "avg_score": score.avg_score,
                "human_StructuralForm": human_scores_clean[0],
                "human_PartCoverage": human_scores_clean[1],
                "human_SurfaceDetail": human_scores_clean[2], 
                "human_TextureQuality": human_scores_clean[3],
                "human_avg_score": score.human_avg_score,
            })
    
    df = pd.DataFrame(model_data)
    
    # Generate all visualizations
    generate_dimension_analysis(df, results.dimension_names, output_dir)
    generate_performance_analytics(df, output_dir)
    generate_comparative_analysis(df, results, output_dir, provider_name)
    
    # Generate summary report
    generate_summary_report(results, df, output_dir, provider_name)
    
    print(f"✅ Analysis complete! Check {output_dir} for all reports and visualizations.")



def generate_dimension_analysis(df: pd.DataFrame, dimension_names: List[str], output_dir: Path) -> None:
    """Generate dimension-wise analysis"""
    
    # Radar Chart for each model
    models = df['model_name'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    angles = np.linspace(0, 2 * np.pi, len(dimension_names), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, model in enumerate(models[:4]):  # Show top 4 models
        if i >= len(axes):
            break
            
        model_data = df[df['model_name'] == model][dimension_names].mean()
        values = model_data.tolist()
        values += values[:1]  # Complete the circle
        
        ax = axes[i]
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimension_names)
        ax.set_ylim(0, 10)
        ax.set_title(f'{model} - Performance Profile', size=12, fontweight='bold', pad=20)
        ax.grid(True)
    
    # Hide extra subplots
    for i in range(len(models), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "radar_charts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Dimension Correlation Heatmap
    corr_matrix = df[dimension_names].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Dimension Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / "dimension_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_analytics(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate performance distribution and analytics"""
    
    # Score distributions by model
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    dimensions = ["StructuralForm", "PartCoverage", "SurfaceDetail", "TextureQuality"]
    
    for i, dim in enumerate(dimensions):
        ax = axes[i//2, i%2]
        df.boxplot(column=dim, by='model_name', ax=ax)
        ax.set_title(f'{dim} Score Distribution')
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model consistency analysis (coefficient of variation)
    consistency_data = []
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        for dim in dimensions:
            scores = model_df[dim]
            cv = scores.std() / scores.mean() if scores.mean() > 0 else 0
            consistency_data.append({
                'model': model,
                'dimension': dim,
                'coefficient_of_variation': cv,
                'mean_score': scores.mean(),
                'std_score': scores.std()
            })
    
    consistency_df = pd.DataFrame(consistency_data)
    
    plt.figure(figsize=(12, 6))
    pivot_cv = consistency_df.pivot(index='model', columns='dimension', values='coefficient_of_variation')
    sns.heatmap(pivot_cv, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Coefficient of Variation (lower = more consistent)'})
    plt.title('Model Consistency Analysis')
    plt.xlabel('Dimension') 
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(output_dir / "consistency_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparative_analysis(df: pd.DataFrame, results: LeaderboardResults, output_dir: Path, provider_name: str = "claude") -> None:
    """Generate LLM vs Human comparison analysis"""
    
    dimensions = ["StructuralForm", "PartCoverage", "SurfaceDetail", "TextureQuality"]
    
    # Claude vs Human correlation per dimension
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, dim in enumerate(dimensions):
        claude_scores = df[dim]
        human_scores = df[f'human_{dim}']
        
        # Remove rows where human scores are NaN for correlation calculation
        valid_mask = ~pd.isna(human_scores)
        claude_valid = claude_scores[valid_mask]
        human_valid = human_scores[valid_mask]
        
        if len(human_valid) > 1:  # Need at least 2 points for correlation
            # Calculate correlation only on valid (non-NaN) pairs
            correlation = claude_valid.corr(human_valid)
            
            axes[i].scatter(human_valid, claude_valid, alpha=0.6)
            axes[i].plot([0, 10], [0, 10], 'r--', alpha=0.5)  # Perfect agreement line
            axes[i].set_xlabel(f'Human {dim}')
            axes[i].set_ylabel(f'{provider_name.title()} {dim}')
            axes[i].set_title(f'{dim}\nCorr: {correlation:.3f} (n={len(human_valid)})')
        else:
            # Not enough valid human scores for correlation
            axes[i].set_xlabel(f'Human {dim}')
            axes[i].set_ylabel(f'{provider_name.title()} {dim}')
            axes[i].set_title(f'{dim}\nInsufficient human data')
            axes[i].text(0.5, 0.5, 'No valid\nhuman scores', 
                        ha='center', va='center', transform=axes[i].transAxes)
        
        axes[i].set_xlim(0, 10)
        axes[i].set_ylim(0, 10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{provider_name}_vs_human_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Agreement analysis
    agreement_data = []
    for dim in dimensions:
        claude_scores = df[dim]
        human_scores = df[f'human_{dim}']
        
        # Only calculate differences for valid (non-NaN) human scores
        valid_mask = ~pd.isna(human_scores)
        claude_valid = claude_scores[valid_mask]
        human_valid = human_scores[valid_mask]
        
        if len(human_valid) > 0:
            differences = np.abs(claude_valid - human_valid)
            correlation = claude_valid.corr(human_valid) if len(human_valid) > 1 else None
            
            agreement_data.append({
                'dimension': dim,
                'valid_pairs': len(human_valid),
                'mean_absolute_difference': differences.mean(),
                'median_absolute_difference': differences.median(),
                'correlation': correlation,
                'perfect_matches': (differences == 0).sum(),
                'close_matches_1pt': (differences <= 1).sum(),
                'close_matches_2pt': (differences <= 2).sum(),
            })
        else:
            # No valid human scores for this dimension
            agreement_data.append({
                'dimension': dim,
                'valid_pairs': 0,
                'mean_absolute_difference': None,
                'median_absolute_difference': None,
                'correlation': None,
                'perfect_matches': 0,
                'close_matches_1pt': 0,
                'close_matches_2pt': 0,
            })
    
    agreement_df = pd.DataFrame(agreement_data)
    agreement_df.to_csv(output_dir / f"{provider_name}_human_agreement.csv", index=False)

def generate_summary_report(results: LeaderboardResults, df: pd.DataFrame, output_dir: Path, provider_name: str = "claude") -> None:
    """Generate comprehensive summary report"""
    
    with open(output_dir / "summary_report.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("COMPREHENSIVE LEADERBOARD ANALYSIS REPORT\n") 
        f.write("=" * 60 + "\n\n")
        

        
        # Average dimension scores
        f.write("AVERAGE DIMENSION SCORES\n")
        f.write("-" * 40 + "\n")
        dimensions = ["StructuralForm", "PartCoverage", "SurfaceDetail", "TextureQuality"]
        model_avgs = df.groupby('model_name')[dimensions].mean().round(2)
        
        for model in model_avgs.index:
            f.write(f"{model}:\n")
            for dim in dimensions:
                claude_avg = model_avgs.loc[model, dim]
                # Use pandas built-in NaN handling for human averages
                human_avg = df[df['model_name'] == model][f'human_{dim}'].mean()  # This automatically handles NaN
                if pd.isna(human_avg):
                    human_avg_str = "N/A"
                else:
                    human_avg_str = f"{human_avg:.1f}"
                f.write(f"  {dim}: {provider_name.title()} {claude_avg:.1f}, Human {human_avg_str}\n")
            f.write("\n")
        
        # Key insights
        f.write("KEY INSIGHTS\n")
        f.write("-" * 40 + "\n")
        
        # Best performing model
        model_avgs = df.groupby('model_name')['avg_score'].mean().sort_values(ascending=False)
        if len(model_avgs) > 0:
            best_model = model_avgs.index[0]
            f.write(f"• Best Overall Model: {best_model} ({model_avgs.iloc[0]:.1f} avg)\n")
        else:
            f.write(f"• Best Overall Model: N/A\n")
        
        # Most consistent model
        consistency_scores = []
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            cv = np.mean([model_df[dim].std() / model_df[dim].mean() for dim in dimensions])
            consistency_scores.append((model, cv))
        most_consistent = min(consistency_scores, key=lambda x: x[1])[0]
        f.write(f"• Most Consistent Model: {most_consistent}\n")
        
        # LLM-Human agreement
        overall_correlations = []
        for dim in dimensions:
            human_scores = df[f'human_{dim}']
            claude_scores = df[dim]
            
            # Only calculate correlation if we have valid human scores
            valid_mask = ~pd.isna(human_scores)
            if valid_mask.sum() > 1:  # Need at least 2 valid pairs
                corr = claude_scores[valid_mask].corr(human_scores[valid_mask])
                if not pd.isna(corr):
                    overall_correlations.append(corr)
        
        if overall_correlations:
            avg_correlation = np.mean(overall_correlations)
            f.write(f"• LLM-Human Agreement: {avg_correlation:.3f} average correlation ({len(overall_correlations)}/{len(dimensions)} dimensions)\n")
        else:
            f.write(f"• LLM-Human Agreement: N/A (insufficient human data)\n")
        
        f.write(f"\n• Total Concepts Evaluated: {len(results.model_scores)}\n")
        
        # Count total unique models across all concepts
        all_models = set()
        for concept_scores in results.model_scores.values():
            for model_score in concept_scores:
                all_models.add(model_score.model_name)
        f.write(f"• Total Models: {len(all_models)}\n")


# -----------------------------------------------------------------------------
# Results Loading and Visualization Re-generation
# -----------------------------------------------------------------------------

def load_results_from_csv(csv_path: Path) -> LeaderboardResults:
    """Load leaderboard results from CSV file back into LeaderboardResults object"""
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Reconstruct model_scores dict
    model_scores = {}
    dimension_names = ["StructuralForm", "PartCoverage", "SurfaceDetail", "TextureQuality"]
    
    for _, row in df.iterrows():
        concept_id = row['concept_session']
        model_name = row['model']
        
        # Extract Claude scores
        claude_scores = [
            row[f'claude_{dim}'] for dim in dimension_names
        ]
        claude_avg = row['claude_avg']
        
        # Extract human scores (handle N/A values)
        human_scores = []
        for dim in dimension_names:
            human_val = row[f'human_{dim}']
            if pd.isna(human_val) or human_val == 'N/A':
                human_scores.append(None)
            else:
                human_scores.append(float(human_val))
        
        human_avg = row['human_avg']
        if pd.isna(human_avg) or human_avg == 'N/A':
            human_avg_score = 0.0
        else:
            human_avg_score = float(human_avg)
        
        # Create ModelScore object
        model_score = ModelScore(
            model_name=model_name,
            session_ids=[f"session_{model_name}"],  # Placeholder since CSV doesn't store session IDs
            concept_session_id=concept_id,
            scores=claude_scores,
            human_scores=human_scores,
            avg_score=claude_avg,
            human_avg_score=human_avg_score
        )
        
        if concept_id not in model_scores:
            model_scores[concept_id] = []
        model_scores[concept_id].append(model_score)
    
    return LeaderboardResults(
        model_scores=model_scores,
        dimension_names=dimension_names
    )

def combine_provider_results(provider_csv_paths: Dict[str, Path], output_dir: Path) -> None:
    """Generate combined analysis from multiple provider results"""
    
    print(f"\n🔄 Loading results from {len(provider_csv_paths)} providers...")
    
    all_results = {}
    combined_model_data = []
    
    for provider, csv_path in provider_csv_paths.items():
        if not csv_path.exists():
            print(f"⚠️  Skipping {provider}: {csv_path} not found")
            continue
            
        print(f"   📂 Loading {provider} results from {csv_path}")
        results = load_results_from_csv(csv_path)
        all_results[provider] = results
        
        # Add provider-specific data for combined analysis
        for concept_id, model_scores in results.model_scores.items():
            for score in model_scores:
                combined_model_data.append({
                    "provider": provider,
                    "concept_id": concept_id,
                    "model": score.model_name,  # Use 'model' to match CSV column naming
                    "StructuralForm": score.scores[0],
                    "PartCoverage": score.scores[1],
                    "SurfaceDetail": score.scores[2],
                    "TextureQuality": score.scores[3],
                    "avg_score": score.avg_score,
                    "human_StructuralForm": score.human_scores[0] if score.human_scores[0] is not None else float('nan'),
                    "human_PartCoverage": score.human_scores[1] if score.human_scores[1] is not None else float('nan'),
                    "human_SurfaceDetail": score.human_scores[2] if score.human_scores[2] is not None else float('nan'),
                    "human_TextureQuality": score.human_scores[3] if score.human_scores[3] is not None else float('nan'),
                    "human_avg_score": score.human_avg_score,
                })
    
    if not all_results:
        print("❌ No valid provider results found!")
        return
    
    output_dir.mkdir(exist_ok=True)
    combined_df = pd.DataFrame(combined_model_data)
    
    print(f"\n📊 Generating combined provider analysis in {output_dir}...")
    
    # Provider comparison visualizations
    generate_provider_comparison_analysis(combined_df, all_results, output_dir)
    
    # Generate individual provider results in subdirectories
    for provider, results in all_results.items():
        provider_subdir = output_dir / f"{provider}_individual"
        generate_comprehensive_analysis(results, provider_subdir, provider)
    
    print(f"✅ Combined analysis complete! Check {output_dir} for all reports.")

def generate_provider_comparison_analysis(df: pd.DataFrame, all_results: Dict[str, LeaderboardResults], output_dir: Path) -> None:
    """Generate visualizations comparing different LLM providers"""
    
    dimensions = ["StructuralForm", "PartCoverage", "SurfaceDetail", "TextureQuality"]
    providers = df['provider'].unique()
    
    # Provider Score Comparison Box Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, dim in enumerate(dimensions):
        ax = axes[i]
        df.boxplot(column=dim, by='provider', ax=ax)
        ax.set_title(f'{dim} - Provider Comparison')
        ax.set_xlabel('LLM Provider')
        ax.set_ylabel('Score')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "provider_score_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Provider Correlation with Human Scores
    correlation_data = []
    for provider in providers:
        provider_df = df[df['provider'] == provider]
        for dim in dimensions:
            llm_scores = provider_df[dim]
            human_scores = provider_df[f'human_{dim}']
            
            # Only calculate correlation if we have valid human scores
            valid_mask = ~pd.isna(human_scores)
            if valid_mask.sum() > 1:
                correlation = llm_scores[valid_mask].corr(human_scores[valid_mask])
                if not pd.isna(correlation):
                    correlation_data.append({
                        'provider': provider,
                        'dimension': dim,
                        'correlation': correlation,
                        'valid_pairs': valid_mask.sum()
                    })
    
    if correlation_data:
        corr_df = pd.DataFrame(correlation_data)
        
        # Correlation heatmap
        plt.figure(figsize=(10, 6))
        pivot_corr = corr_df.pivot(index='provider', columns='dimension', values='correlation')
        sns.heatmap(pivot_corr, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                    cbar_kws={'label': 'Correlation with Human Scores'})
        plt.title('LLM Provider vs Human Score Correlation')
        plt.xlabel('Dimension')
        plt.ylabel('LLM Provider')
        plt.tight_layout()
        plt.savefig(output_dir / "provider_human_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
    

    
    # Save combined summary report
    with open(output_dir / "combined_provider_report.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("COMBINED LLM PROVIDER COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("PROVIDERS ANALYZED\n")
        f.write("-" * 40 + "\n")
        for provider in providers:
            provider_df = df[df['provider'] == provider]
            unique_models = provider_df['model'].nunique()
            unique_concepts = provider_df['concept_id'].nunique()
            f.write(f"• {provider}: {unique_models} models, {unique_concepts} concepts\n")
        
        f.write("\nAVERAGE SCORES BY PROVIDER\n")
        f.write("-" * 40 + "\n")
        for provider in providers:
            provider_df = df[df['provider'] == provider]
            f.write(f"{provider}:\n")
            for dim in dimensions:
                avg_score = provider_df[dim].mean()
                f.write(f"  {dim}: {avg_score:.2f}\n")
            f.write(f"  Overall Average: {provider_df['avg_score'].mean():.2f}\n\n")
        
        if correlation_data:
            f.write("HUMAN-LLM CORRELATION BY PROVIDER\n")
            f.write("-" * 40 + "\n")
            for provider in providers:
                provider_corrs = [c['correlation'] for c in correlation_data if c['provider'] == provider]
                if provider_corrs:
                    avg_correlation = sum(provider_corrs) / len(provider_corrs)
                    f.write(f"{provider}: {avg_correlation:.3f} average correlation\n")
        
        f.write(f"\nTotal evaluations across all providers: {len(df)}\n")

def regenerate_visualizations_only(csv_path: Path, output_dir: Path = None) -> None:
    """Regenerate visualizations from existing CSV results without re-running evaluation"""
    
    if output_dir is None:
        # Infer output directory from CSV filename
        provider_name = csv_path.stem.replace("leaderboard_results_", "")
        output_dir = Path(f"results_{provider_name}")
    else:
        # Extract provider name from CSV filename for proper file naming
        provider_name = csv_path.stem.replace("leaderboard_results_", "")
    
    print(f"📊 Regenerating visualizations from {csv_path}...")
    print(f"   Output directory: {output_dir}")
    
    # Load results from CSV
    results = load_results_from_csv(csv_path)
    
    # Generate all visualizations
    generate_comprehensive_analysis(results, output_dir, provider_name)
    
    print(f"✅ Visualization regeneration complete!")

# -----------------------------------------------------------------------------
# LLM Response Caching
# -----------------------------------------------------------------------------

def get_cache_key(cache_type: str, session_id: str, trial: int = 0, provider: str = "claude", **kwargs) -> str:
    """Generate cache key for LLM responses"""
    # Create a hash of the additional parameters for uniqueness
    import hashlib
    
    if cache_type == "detailed":
        # For detailed scoring: provider + session_id + mesh_count + mesh_index + trial
        mesh_count = kwargs.get("mesh_count", 0)
        mesh_index = kwargs.get("mesh_index", 0)
        key_data = f"{provider}_{session_id}_{mesh_count}_{mesh_index}_{trial}"
    elif cache_type == "detailed_multiview":
        # For multiview detailed scoring: provider + session_id + mesh_count + mesh_index + trial + views
        mesh_count = kwargs.get("mesh_count", 0)
        mesh_index = kwargs.get("mesh_index", 0)
        views = kwargs.get("views", [])
        views_str = "_".join(sorted(views))  # Sort for consistent caching
        key_data = f"{provider}_{session_id}_{mesh_count}_{mesh_index}_{trial}_{views_str}"
    elif cache_type == "pairwise":
        # For pairwise: provider + concept_session + model_a + model_b + trial
        concept_session = kwargs.get("concept_session_id", "")
        model_a = kwargs.get("model_a", "")
        model_b = kwargs.get("model_b", "")
        key_data = f"{provider}_{concept_session}_{model_a}_{model_b}_{trial}"
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
    
    # Create hash to handle long/special characters
    cache_hash = hashlib.md5(key_data.encode()).hexdigest()
    return f"{cache_type}_{provider}_{cache_hash}.json"

def load_cached_response(cache_key: str) -> Optional[Dict]:
    """Load cached LLM response if exists"""
    cache_file = CACHE_DIR / cache_key
    if cache_file.exists():
        try:
            with cache_file.open("r") as f:
                cached_data = json.load(f)
                print(f"   📋 Using cached response: {cache_key}")
                return cached_data
        except (json.JSONDecodeError, IOError) as e:
            print(f"   ⚠️  Failed to load cache {cache_key}: {e}")
            return None
    return None

def save_cached_response(cache_key: str, response_data: Dict, metadata: Dict = None) -> None:
    """Save LLM response to cache"""
    CACHE_DIR.mkdir(exist_ok=True)
    
    cache_data = {
        "response": response_data,
        "metadata": metadata or {},
        "timestamp": pd.Timestamp.now().isoformat(),
        "cache_version": "1.0"
    }
    
    cache_file = CACHE_DIR / cache_key
    try:
        with cache_file.open("w") as f:
            json.dump(cache_data, f, indent=2)
    except IOError as e:
        print(f"   ⚠️  Failed to save cache {cache_key}: {e}")

def clear_cache(pattern: str = None) -> None:
    """Clear cached responses"""
    if not CACHE_DIR.exists():
        print("Cache directory doesn't exist")
        return
    
    if pattern:
        import glob
        files = glob.glob(str(CACHE_DIR / pattern))
        count = len(files)
        for file_path in files:
            Path(file_path).unlink()
        print(f"Cleared {count} cache files matching pattern: {pattern}")
    else:
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("Cleared entire cache directory")

def get_cache_stats() -> Dict:
    """Get cache statistics"""
    if not CACHE_DIR.exists():
        return {"total_files": 0, "detailed_responses": 0, "pairwise_responses": 0, "total_size_mb": 0}
    
    detailed_count = len(list(CACHE_DIR.glob("detailed_*.json")))
    pairwise_count = len(list(CACHE_DIR.glob("pairwise_*.json")))
    total_files = len(list(CACHE_DIR.glob("*.json")))
    
    total_size = sum(f.stat().st_size for f in CACHE_DIR.glob("*.json"))
    total_size_mb = total_size / (1024 * 1024)
    
    return {
        "total_files": total_files,
        "detailed_responses": detailed_count,
        "pairwise_responses": pairwise_count,
        "total_size_mb": round(total_size_mb, 2)
    }


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main():
    """Main entry point for the leaderboard script"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-eval-json", type=Path, help="Path to leaderboard_models.json with human evaluations")
    parser.add_argument("--debug", action="store_true", help="Save all LLM queries to debug_queries/ folder")
    
    # Cache control arguments
    parser.add_argument("--no-cache", action="store_true", help="Disable cache (always call LLM)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear entire cache before running")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics and exit")
    
    # Multiple trials support
    parser.add_argument("--trial", type=int, default=0, help="Trial number for multiple runs (default: 0)")
    parser.add_argument("--trials", type=int, help="Run multiple trials (1 to N) and aggregate results")
    
    # Session import and preview options
    parser.add_argument("--job-tracking", type=Path, default="job_tracking.json", help="Path to job_tracking.json for auto-importing sessions")
    parser.add_argument("--no-preview", action="store_true", help="Skip evaluation preview and run immediately")
    parser.add_argument("--preview-only", action="store_true", help="Show preview only, don't run evaluation")
    parser.add_argument("--human-eval-only", action="store_true", help="Only evaluate sessions with human evaluations (skip auto-discovered sessions)")
    
    # Multi-view rendering options
    parser.add_argument("--multiview", action="store_true", help="Use multi-view rendering (front + back) for more comprehensive evaluation")
    parser.add_argument("--views", nargs="+", default=["front", "back"], help="Views to render (front, back)")
    
    # Visualization regeneration options
    parser.add_argument("--visualize-only", type=Path, help="Regenerate visualizations from existing CSV results without re-running evaluation")
    parser.add_argument("--output-dir", type=Path, help="Custom output directory for visualization regeneration")
    parser.add_argument("--combine-providers", nargs="+", help="Combine results from multiple providers (e.g., --combine-providers claude gemini openai)")
    parser.add_argument("--combined-output", type=Path, default="results_combined", help="Output directory for combined provider analysis")
    
    # LLM provider options
    if LLM_SYSTEM_AVAILABLE:
        available_providers = [p.value for p in get_available_llm_providers()]
        parser.add_argument("--llm-provider", choices=available_providers + ['claude'], 
                          default='claude', help="LLM provider to use for evaluation")
        parser.add_argument("--list-providers", action="store_true", help="List available LLM providers and exit")
        parser.add_argument("--test-provider", action="store_true", help="Test LLM provider connection and exit")
    
    args = parser.parse_args()

    # Check if human-eval-json is required for this command
    info_commands = ['list_providers', 'test_provider', 'cache_stats', 'clear_cache', 'visualize_only', 'combine_providers']
    needs_data = not any(getattr(args, cmd, False) for cmd in info_commands if hasattr(args, cmd))
    
    if needs_data and not args.human_eval_json:
        parser.error("--human-eval-json is required for evaluation commands")

    # Handle cache management commands
    if hasattr(args, 'cache_stats') and args.cache_stats:
        stats = get_cache_stats()
        print("📋 Cache Statistics:")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Detailed responses: {stats['detailed_responses']}")
        print(f"   Pairwise responses: {stats['pairwise_responses']}")
        print(f"   Total size: {stats['total_size_mb']} MB")
        sys.exit(0)
    
    if hasattr(args, 'clear_cache') and args.clear_cache:
        clear_cache()
        print("✅ Cache cleared")
        if not args.human_eval_json:  # If only clearing cache, exit
            sys.exit(0)

    # Handle visualization-only commands
    if args.visualize_only:
        print("📊 Visualization-only mode")
        regenerate_visualizations_only(args.visualize_only, args.output_dir)
        sys.exit(0)
    
    if args.combine_providers:
        print("🔄 Combining provider results")
        provider_csv_paths = {}
        for provider in args.combine_providers:
            csv_path = Path(f"leaderboard_results_{provider}.csv")
            provider_csv_paths[provider] = csv_path
        combine_provider_results(provider_csv_paths, args.combined_output)
        sys.exit(0)

    # Handle LLM provider listing and testing
    if LLM_SYSTEM_AVAILABLE and hasattr(args, 'list_providers') and args.list_providers:
        available_providers = get_available_llm_providers()
        print("🤖 Available LLM Providers:")
        for provider in available_providers:
            print(f"  • {provider.value}")
        
        # Show which ones need installation
        all_providers = [LLMProvider.CLAUDE, LLMProvider.OPENAI, LLMProvider.GEMINI]
        missing_providers = [p for p in all_providers if p not in available_providers]
        if missing_providers:
            print("\n📦 To install missing providers:")
            for provider in missing_providers:
                cmd = get_provider_installation_command(provider)
                print(f"  • {provider.value}: {cmd}")
        
        sys.exit(0)
    
    # Determine LLM provider
    if LLM_SYSTEM_AVAILABLE and hasattr(args, 'llm_provider'):
        if args.llm_provider == 'claude':
            llm_provider = LLMProvider.CLAUDE
        else:
            llm_provider = LLMProvider(args.llm_provider)
    else:
        llm_provider = LLMProvider.CLAUDE if LLM_SYSTEM_AVAILABLE else None
    
    # Get API keys from config file or prompt user
    print("🚀 Starting Hybrid LLM Leaderboard for 3D Mesh Evaluation")
    print("=" * 60)
    print("📋 Features:")
    print("  • 4D Detailed Scoring (StructuralForm, PartCoverage, SurfaceDetail, TextureQuality)")
    print("  • Comprehensive Analytics & Visualizations") 
    print("  • LLM vs Human Comparison Analysis")
    if LLM_SYSTEM_AVAILABLE:
        print(f"  • Multi-Provider LLM Support ({llm_provider.value if llm_provider else 'fallback'})")
    if not args.no_cache:
        if LLM_SYSTEM_AVAILABLE:
            cache_stats = get_cache_manager().get_cache_stats()
        else:
            cache_stats = get_cache_stats()  # Fallback function
        print(f"  • Enhanced LLM Response Caching ({cache_stats['total_files']} cached responses)")
    print()
    
    if args.debug:
        print("🐛 Debug mode enabled - saving all queries to debug_queries/")
        DEBUG_DIR.mkdir(exist_ok=True)
    

    
    if args.no_cache:
        print("🚫 Cache disabled - will always call LLM APIs")
    
    if args.human_eval_only:
        print("👤 Human-eval-only mode - evaluating only sessions with human evaluations")
    
    if args.multiview:
        if MESH_RENDERER_AVAILABLE:
            print(f"🔍 Multi-view rendering enabled: {', '.join(args.views)} views")
        else:
            print("⚠️  Multi-view requested but mesh_renderer not available - falling back to single view")
    
    if args.trials:
        print(f"🔄 Running {args.trials} trials for statistical robustness")
    elif args.trial > 0:
        print(f"🎯 Running trial #{args.trial}")
    
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    
    csm_key, llm_key = get_api_keys(llm_provider)
    
    # Create LLM client
    try:
        llm_client = create_llm_client_from_config(llm_provider, llm_key)
    except Exception as e:
        print(f"❌ Failed to create LLM client: {e}")
        if LLM_SYSTEM_AVAILABLE:
            print("   Using fallback Claude implementation...")
            llm_client = llm_key  # Fallback to key for old system
        else:
            raise
    
    # Test provider connection if requested
    if LLM_SYSTEM_AVAILABLE and hasattr(args, 'test_provider') and args.test_provider:
        print(f"🧪 Testing {llm_provider.value} connection...")
        try:
            from .llm_clients import test_provider_connection
            if test_provider_connection(llm_provider, llm_key):
                print("✅ Provider connection test successful!")
            else:
                print("❌ Provider connection test failed!")
        except Exception as e:
            print(f"❌ Provider test error: {e}")
        sys.exit(0)
    
    # Handle preview-only mode
    if args.preview_only:
        print("👀 Preview Mode - No evaluation will be run")
        # Just run preview and exit
        run_leaderboard(
            args.human_eval_json, csm_key, llm_client, args.debug, 
            args.trial, args.no_cache,
            preview=True, job_tracking_path=args.job_tracking,
            human_eval_only=args.human_eval_only, use_multiview=args.multiview,
            views=args.views
        )
        print("\n✅ Preview complete!")
        sys.exit(0)

    if args.trials:
        # Run multiple trials and aggregate results
        print(f"📊 Starting {args.trials} trials...")
        all_trial_results = []
        
        for trial_num in range(1, args.trials + 1):
            print(f"\n🔄 === TRIAL {trial_num}/{args.trials} ===")
            trial_results = run_leaderboard(
                args.human_eval_json, csm_key, llm_client, args.debug, 
                trial_num, args.no_cache,
                preview=(trial_num == 1 and not args.no_preview),  # Only preview first trial
                job_tracking_path=args.job_tracking, human_eval_only=args.human_eval_only,
                use_multiview=args.multiview, views=args.views
            )
            if trial_results is None:  # User cancelled during preview
                print("❌ Trials cancelled by user")
                sys.exit(0)
            all_trial_results.append(trial_results)
        
        # TODO: Implement trial aggregation and statistical analysis
        print(f"\n📈 Aggregating results from {args.trials} trials...")
        # For now, just use the last trial's results
        results = all_trial_results[-1]
        
    else:
        # Single trial run
        print("📊 Starting evaluation...")
        results = run_leaderboard(
            args.human_eval_json, csm_key, llm_client, args.debug, 
            args.trial, args.no_cache,
            preview=not args.no_preview, job_tracking_path=args.job_tracking,
            human_eval_only=args.human_eval_only, use_multiview=args.multiview,
            views=args.views
        )
        if results is None:  # User cancelled during preview
            print("❌ Evaluation cancelled by user")
            sys.exit(0)
    
    print(f"\n🎉 Evaluation complete!")
    
    # Count total unique models evaluated
    all_models = set()
    for concept_scores in results.model_scores.values():
        for model_score in concept_scores:
            all_models.add(model_score.model_name)
    
    print(f"📈 Models evaluated: {len(all_models)} unique models")
    print(f"📊 Detailed analysis saved to: {RESULTS_DIR.resolve()}")
    
    if not args.no_cache:
        if LLM_SYSTEM_AVAILABLE:
            final_stats = get_cache_manager().get_cache_stats()
        else:
            final_stats = get_cache_stats()  # Fallback function
        print(f"📋 Cache: {final_stats['total_files']} responses ({final_stats['total_size_mb']} MB)")


if __name__ == "__main__":
    main()
