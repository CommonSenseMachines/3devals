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
"""

import argparse
import base64
import json
import os
import sys
import getpass
import itertools
import math
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
DEBUG_DIR = Path("debug_queries")
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
class PairwiseComparison:
    """Result of comparing two models"""
    concept_session_id: str
    model_a: str
    model_b: str
    winner: str  # "A", "B", or "tie"
    confidence: str  # "low", "medium", "high"
    reasoning: str

@dataclass
class ELORecord:
    """ELO rating tracking"""
    model_name: str
    rating: float
    games_played: int
    wins: int
    losses: int
    ties: int
    
    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

@dataclass
class LeaderboardResults:
    """Complete evaluation results"""
    model_scores: Dict[str, List[ModelScore]]  # concept_id -> [ModelScore]
    pairwise_comparisons: List[PairwiseComparison]
    elo_ratings: Dict[str, ELORecord]  # model_name -> ELORecord
    dimension_names: List[str]
    
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
            "detailed", session_id, trial,
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
            session_suffix = session_id.split('_')[-1]
            unique_model_name = f"{model_name}_{session_suffix}"
            
            # Group by concept
            if concept_img_url not in concept_groups:
                concept_groups[concept_img_url] = []
            concept_groups[concept_img_url].append({
                "session_id": session_id,
                "model_name": unique_model_name,
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
    claude_key: str,
    api_key: str,
    concept_session_id: str,
    concept_img_url: str,
    candidate_name: str,
    candidate_session_ids: List[str],
    debug_enabled: bool = False,
    trial: int = 0,
    use_cache: bool = True,
) -> List[float]:
    """Return mean scores across all renders for this candidate as [StructuralForm, PartCoverage, SurfaceDetail, TextureQuality]."""

    concept_img = download_image(concept_img_url)
    concept_b64 = img_to_b64(concept_img)

    all_scores: List[List[int]] = []  # List of score arrays

    for sid in candidate_session_ids:
        url = f"https://api.csm.ai/v3/sessions/{sid}"
        sess = fetch_json(url, api_key)
        output_data = sess.get("output", {})
        mesh_data = output_data.get("meshes") or output_data.get("part_meshes") or []
        if not mesh_data:
            print(f"   ⚠️  Session {sid} has no meshes/part_meshes or is not complete — skipping.")
            continue
        mesh_count = len(mesh_data)
        for mesh_index, mesh in enumerate(mesh_data):
            # Get render image URL from mesh only; raise if missing
            render_url = mesh.get('data', {}).get('image_url')
            if not render_url:
                raise RuntimeError(f"Mesh {mesh_index} in session {sid} has no render image_url. Cannot evaluate.")
            render_img = download_image(render_url)
            render_b64 = img_to_b64(render_img)
            score_json = call_claude(
                claude_key,
                concept_b64,
                render_b64,
                sid,
                mesh_count,
                mesh_index,
                debug_enabled,
                concept_session_id,
                candidate_name,
                trial,
                use_cache,
            )
            all_scores.append(score_json["score_array"])

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


def run_leaderboard(human_eval_json_path: Path, api_key: str, claude_key: str, debug_enabled: bool = False, skip_pairwise: bool = False, trial: int = 0, no_cache: bool = False, preview: bool = True, job_tracking_path: Path = None, human_eval_only: bool = False):
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
    pairwise_comparisons: List[PairwiseComparison] = []
    elo_ratings: Dict[str, ELORecord] = {}

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
    for concept_img_url, group_data in concept_groups.items():
        concept_session_id = group_data["concept_session_id"]
        entries = group_data["entries"]
        
        print(f"\n[+] Evaluating concept from session {concept_session_id} ({len(entries)} models)...")

        per_candidate: Dict[str, List[float]] = {}
        human_per_candidate: Dict[str, List[float]] = {}
        concept_model_scores: List[ModelScore] = []
        
        for entry in tqdm(entries, desc="candidates", unit="model"):
            session_id = entry.get("session_id")
            human_eval = entry.get("human_eval", {})
            
            # Auto-extract model name from CSM API
            model_name = extract_model_info_from_session(session_id, api_key)
            
            # Make model name unique by appending session ID suffix for clarity
            session_suffix = session_id.split('_')[-1]  # Get last part of session ID
            unique_model_name = f"{model_name}_{session_suffix}"

            scores = evaluate_candidate(
                claude_key,
                api_key,
                concept_session_id,
                concept_img_url,
                model_name,
                [session_id],  # Single session ID in list
                debug_enabled,
                trial,
                not no_cache,
            )
            per_candidate[unique_model_name] = scores
            
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
                
                human_per_candidate[unique_model_name] = human_scores
                
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
                human_per_candidate[unique_model_name] = [None] * 4
                human_str = "N/A"
            
            # Create ModelScore object
            human_scores_for_avg = [score for score in human_per_candidate[unique_model_name] if score is not None]
            human_avg_score = sum(human_scores_for_avg) / len(human_scores_for_avg) if human_scores_for_avg else 0.0
            
            model_score = ModelScore(
                model_name=unique_model_name,
                session_ids=[session_id],
                concept_session_id=concept_session_id,
                scores=scores,
                human_scores=human_per_candidate[unique_model_name],
                avg_score=sum(scores) / len(scores),
                human_avg_score=human_avg_score
            )
            concept_model_scores.append(model_score)
            
            # Display scores nicely
            score_strs = [f"{name}:{score:.1f}" for name, score in zip(dimension_names, scores)]
            avg_score = sum(scores) / len(scores)
            print(f"    • {unique_model_name:<20} → avg:{avg_score:.1f} [{', '.join(score_strs)}]")
            print(f"      Human: {human_str}")

        overall_results[concept_session_id] = per_candidate
        human_evals[concept_session_id] = human_per_candidate
        all_model_scores[concept_session_id] = concept_model_scores

    # Phase 2: Pairwise comparisons for ELO (optional)
    total_comparisons = 0
    if not skip_pairwise:
        print(f"\n🥊 Phase 2: Pairwise ELO Comparisons")
        print("=" * 50)

        for concept_session_id, model_scores in all_model_scores.items():
            if len(model_scores) < 2:
                print(f"Skipping concept {concept_session_id} - need at least 2 models for pairwise comparison")
                continue
                
            print(f"\n[+] Pairwise comparisons for concept {concept_session_id}")
            concept_url = f"https://api.csm.ai/v3/sessions/{concept_session_id}"
            concept_sess = fetch_json(concept_url, api_key)
            concept_img_url = concept_sess["input"]["image"]["data"]["image_url"]
            concept_img = download_image(concept_img_url)
            concept_b64 = img_to_b64(concept_img)
            
            # Get representative renders for each model (use first render of first session)
            model_renders = {}
            for model_score in model_scores:
                first_session_id = model_score.session_ids[0]
                session_url = f"https://api.csm.ai/v3/sessions/{first_session_id}"
                sess = fetch_json(session_url, api_key)
                output_data = sess.get("output", {})
                mesh_data = output_data.get("meshes") or output_data.get("part_meshes") or []
                if mesh_data:
                    render_url = mesh_data[0].get('data', {}).get('image_url')
                    if render_url:
                        render_img = download_image(render_url)
                        model_renders[model_score.model_name] = img_to_b64(render_img)
            
            # Perform all pairwise comparisons for this concept
            model_names = list(model_renders.keys())
            comparisons_for_concept = list(itertools.combinations(model_names, 2))
            
            for model_a, model_b in tqdm(comparisons_for_concept, desc=f"Comparing {len(model_names)} models"):
                if model_a in model_renders and model_b in model_renders:
                    comparison = call_claude_pairwise(
                        claude_key,
                        concept_b64,
                        model_renders[model_a],
                        model_renders[model_b],
                        model_a,
                        model_b,
                        concept_session_id,
                        debug_enabled,
                        trial,
                        not no_cache,
                    )
                    pairwise_comparisons.append(comparison)
                    
                    # Update ELO ratings
                    update_elo_records(elo_ratings, model_a, model_b, comparison.winner)
                    total_comparisons += 1
                    
                    print(f"    {model_a} vs {model_b}: {comparison.winner} wins ({comparison.confidence} confidence)")

    print(f"\n✅ Completed {total_comparisons} pairwise comparisons")

    # Phase 3: Generate comprehensive analysis
    print(f"\n📊 Phase 3: Comprehensive Analysis")
    print("=" * 50)
    
    results = LeaderboardResults(
        model_scores=all_model_scores,
        pairwise_comparisons=pairwise_comparisons,
        elo_ratings=elo_ratings,
        dimension_names=dimension_names
    )
    
    # Generate all visualizations and reports
    generate_comprehensive_analysis(results, RESULTS_DIR)

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
            
            print(f"   {model:<20} Claude: avg:{avg_score:.1f} [{score_detail}]")
            print(f"   {'':<20} Human:  {human_avg_str} [{human_detail}]")

    # Print ELO leaderboard
    print("\n===== ELO Rankings (Pairwise Performance) =====")
    if elo_ratings:
        sorted_elo = sorted(elo_ratings.items(), key=lambda x: x[1].rating, reverse=True)
        for i, (model, record) in enumerate(sorted_elo, 1):
            print(f"{i}. {model:<20} {record.rating:.0f} ELO ({record.wins}-{record.losses}-{record.ties}) {record.win_rate:.1%} win rate")
    else:
        print("No ELO ratings available (need multiple models per concept)")

    # Write enhanced CSV with all dimensions and ELO
    csv_path = Path("leaderboard_results.csv")
    with csv_path.open("w", encoding="utf-8") as f:
        claude_cols = ",".join([f"claude_{dim}" for dim in dimension_names])
        human_cols = ",".join([f"human_{dim}" for dim in dimension_names])
        header = f"concept_session,model,claude_avg,{claude_cols},human_avg,{human_cols},elo_rating,elo_games,elo_wins,elo_losses,elo_ties,win_rate\n"
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
                
                # Add ELO data
                elo_record = elo_ratings.get(model, ELORecord(model, 1500.0, 0, 0, 0, 0))
                f.write(f"{concept_id},{model},{claude_avg:.1f},{claude_values},{human_avg_str},{human_values},{elo_record.rating:.0f},{elo_record.games_played},{elo_record.wins},{elo_record.losses},{elo_record.ties},{elo_record.win_rate:.3f}\n")
    
    print(f"\nResults written to {csv_path.resolve()}")
    print(f"Detailed analysis and visualizations saved to {RESULTS_DIR.resolve()}/")
    
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

def get_api_keys() -> Tuple[str, str]:
    """Get API keys from config file or prompt user."""
    config = load_config()
    
    # Check if keys are already available
    csm_key = config.get("csm_api_key") or os.getenv("CSM_API_KEY")
    claude_key = config.get("claude_api_key") or os.getenv("CLAUDE_KEY")
    
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
    
    if not claude_key:
        print("\n🔑 Claude API Key needed:")
        print("   Get yours from: https://console.anthropic.com/")
        claude_key = getpass.getpass("   Enter Claude API key: ").strip()
        config["claude_api_key"] = claude_key
        need_save = True
    
    # Save config if we got new keys
    if need_save:
        save_config(config)
        print()
    
    return csm_key, claude_key


# -----------------------------------------------------------------------------
# ELO Rating System
# -----------------------------------------------------------------------------

def calculate_elo_change(rating_a: float, rating_b: float, result: float, k_factor: int = 32) -> Tuple[float, float]:
    """
    Calculate ELO rating changes for two players.
    
    Args:
        rating_a: Current rating of player A
        rating_b: Current rating of player B  
        result: Game result from A's perspective (1.0 = win, 0.5 = tie, 0.0 = loss)
        k_factor: ELO k-factor (higher = more volatile ratings)
    
    Returns:
        (new_rating_a, new_rating_b)
    """
    expected_a = 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
    expected_b = 1 - expected_a
    
    new_rating_a = rating_a + k_factor * (result - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - result) - expected_b)
    
    return new_rating_a, new_rating_b

def update_elo_records(elo_records: Dict[str, ELORecord], model_a: str, model_b: str, 
                      winner: str, k_factor: int = 32) -> None:
    """Update ELO records after a pairwise comparison"""
    
    # Initialize records if needed
    if model_a not in elo_records:
        elo_records[model_a] = ELORecord(model_a, 1500.0, 0, 0, 0, 0)
    if model_b not in elo_records:
        elo_records[model_b] = ELORecord(model_b, 1500.0, 0, 0, 0, 0)
    
    # Determine result from A's perspective
    if winner == "A":
        result = 1.0
        elo_records[model_a].wins += 1
        elo_records[model_b].losses += 1
    elif winner == "B":
        result = 0.0
        elo_records[model_a].losses += 1
        elo_records[model_b].wins += 1
    else:  # tie
        result = 0.5
        elo_records[model_a].ties += 1
        elo_records[model_b].ties += 1
    
    # Calculate new ratings
    new_rating_a, new_rating_b = calculate_elo_change(
        elo_records[model_a].rating, 
        elo_records[model_b].rating, 
        result, 
        k_factor
    )
    
    # Update records
    elo_records[model_a].rating = new_rating_a
    elo_records[model_b].rating = new_rating_b
    elo_records[model_a].games_played += 1
    elo_records[model_b].games_played += 1


# -----------------------------------------------------------------------------
# Pairwise Comparison
# -----------------------------------------------------------------------------

def call_claude_pairwise(
    claude_key: str,
    concept_img_b64: str,
    render_a_b64: str, 
    render_b_b64: str,
    model_a_name: str,
    model_b_name: str,
    concept_session_id: str,
    debug_enabled: bool = False,
    trial: int = 0,
    use_cache: bool = True,
) -> PairwiseComparison:
    """Compare two models head-to-head using Claude"""
    
    # Check cache first
    if use_cache:
        cache_key = get_cache_key(
            "pairwise", f"{model_a_name}_vs_{model_b_name}", trial,
            concept_session_id=concept_session_id,
            model_a=model_a_name, model_b=model_b_name
        )
        cached_response = load_cached_response(cache_key)
        if cached_response:
            cached_data = cached_response["response"]
            return PairwiseComparison(
                concept_session_id=concept_session_id,
                model_a=model_a_name,
                model_b=model_b_name,
                winner=cached_data["winner"],
                confidence=cached_data["confidence"],
                reasoning=cached_data.get("reasoning", "")
            )
    
    client = anthropic.Anthropic(api_key=claude_key)
    
    content = [
        {
            "type": "text",
            "text": "I will show you a concept image and two 3D mesh reconstructions (A and B). Please determine which reconstruction better represents the original concept."
        },
        {
            "type": "text", 
            "text": PAIRWISE_PROMPT
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
                "data": concept_img_b64,
            },
        },
        {
            "type": "text",
            "text": f"3D MESH RECONSTRUCTION A ({model_a_name}):"
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": render_a_b64,
            },
        },
        {
            "type": "text", 
            "text": f"3D MESH RECONSTRUCTION B ({model_b_name}):"
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": render_b_b64,
            },
        },
        {
            "type": "text",
            "text": f"Which reconstruction (A or B) better represents the concept image? Consider overall geometry, detail, textures, and completeness."
        }
    ]

    try:
        message = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=150,
            temperature=0,
            system=PAIRWISE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        
        response_json = json.loads(message.content[0].text)
        
        # Save to cache
        if use_cache:
            metadata = {
                "concept_session_id": concept_session_id,
                "model_a": model_a_name,
                "model_b": model_b_name,
                "trial": trial
            }
            save_cached_response(cache_key, response_json, metadata)
        
        return PairwiseComparison(
            concept_session_id=concept_session_id,
            model_a=model_a_name,
            model_b=model_b_name,
            winner=response_json["winner"],
            confidence=response_json["confidence"],
            reasoning=response_json.get("reasoning", "")
        )
        
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"Warning: Invalid pairwise response for {model_a_name} vs {model_b_name}: {exc}")
        return PairwiseComparison(
            concept_session_id=concept_session_id,
            model_a=model_a_name,
            model_b=model_b_name,
            winner="tie",
            confidence="low", 
            reasoning="Failed to parse response"
        )

# -----------------------------------------------------------------------------
# Comprehensive Analytics & Visualizations
# -----------------------------------------------------------------------------

def generate_comprehensive_analysis(results: LeaderboardResults, output_dir: Path) -> None:
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
    if results.elo_ratings:  # Only generate ELO visualizations if we have ELO data
        generate_elo_visualizations(results, output_dir)
    generate_dimension_analysis(df, results.dimension_names, output_dir)
    generate_performance_analytics(df, output_dir)
    generate_comparative_analysis(df, results, output_dir)
    
    # Generate summary report
    generate_summary_report(results, df, output_dir)
    
    print(f"✅ Analysis complete! Check {output_dir} for all reports and visualizations.")

def generate_elo_visualizations(results: LeaderboardResults, output_dir: Path) -> None:
    """Generate ELO-related visualizations"""
    
    if not results.elo_ratings:
        print("   ⚠️  No ELO data available - skipping ELO visualizations")
        return
    
    # ELO Rankings Bar Chart
    plt.figure(figsize=(12, 8))
    models = list(results.elo_ratings.keys())
    ratings = [results.elo_ratings[model].rating for model in models]
    games = [results.elo_ratings[model].games_played for model in models]
    
    # Sort by rating
    sorted_data = sorted(zip(models, ratings, games), key=lambda x: x[1], reverse=True)
    models, ratings, games = zip(*sorted_data)
    
    bars = plt.bar(range(len(models)), ratings, alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('ELO Rating')
    plt.title('ELO Rankings - Overall Performance')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    
    # Add game count labels on bars
    for i, (bar, game_count) in enumerate(zip(bars, games)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{game_count} games', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "elo_rankings.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Head-to-Head Matrix
    n_models = len(models)
    h2h_matrix = np.zeros((n_models, n_models))
    model_to_idx = {model: i for i, model in enumerate(models)}
    
    for comparison in results.pairwise_comparisons:
        if comparison.model_a in model_to_idx and comparison.model_b in model_to_idx:
            i, j = model_to_idx[comparison.model_a], model_to_idx[comparison.model_b]
            if comparison.winner == "A":
                h2h_matrix[i, j] += 1
            elif comparison.winner == "B":
                h2h_matrix[j, i] += 1
            # Ties add 0.5 to both
            else:
                h2h_matrix[i, j] += 0.5
                h2h_matrix[j, i] += 0.5
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(h2h_matrix, annot=True, fmt='.1f', xticklabels=models, yticklabels=models,
                cmap='RdYlBu_r', center=0, cbar_kws={'label': 'Wins'})
    plt.title('Head-to-Head Win Matrix')
    plt.xlabel('Opponent')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(output_dir / "head_to_head_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

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

def generate_comparative_analysis(df: pd.DataFrame, results: LeaderboardResults, output_dir: Path) -> None:
    """Generate Claude vs Human comparison analysis"""
    
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
            axes[i].set_ylabel(f'Claude {dim}')
            axes[i].set_title(f'{dim}\nCorr: {correlation:.3f} (n={len(human_valid)})')
        else:
            # Not enough valid human scores for correlation
            axes[i].set_xlabel(f'Human {dim}')
            axes[i].set_ylabel(f'Claude {dim}')
            axes[i].set_title(f'{dim}\nInsufficient human data')
            axes[i].text(0.5, 0.5, 'No valid\nhuman scores', 
                        ha='center', va='center', transform=axes[i].transAxes)
        
        axes[i].set_xlim(0, 10)
        axes[i].set_ylim(0, 10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "claude_vs_human_correlation.png", dpi=300, bbox_inches='tight')
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
    agreement_df.to_csv(output_dir / "claude_human_agreement.csv", index=False)

def generate_summary_report(results: LeaderboardResults, df: pd.DataFrame, output_dir: Path) -> None:
    """Generate comprehensive summary report"""
    
    with open(output_dir / "summary_report.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("COMPREHENSIVE LEADERBOARD ANALYSIS REPORT\n") 
        f.write("=" * 60 + "\n\n")
        
        # ELO Rankings
        f.write("ELO RANKINGS (Overall Performance)\n")
        f.write("-" * 40 + "\n")
        if results.elo_ratings:
            sorted_elo = sorted(results.elo_ratings.items(), key=lambda x: x[1].rating, reverse=True)
            for i, (model, record) in enumerate(sorted_elo, 1):
                f.write(f"{i}. {model}: {record.rating:.0f} ELO ")
                f.write(f"({record.wins}-{record.losses}-{record.ties}, {record.win_rate:.1%} win rate)\n")
        else:
            f.write("No ELO rankings available (need multiple models per concept for pairwise comparisons)\n")
        
        f.write("\n")
        
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
                f.write(f"  {dim}: Claude {claude_avg:.1f}, Human {human_avg_str}\n")
            f.write("\n")
        
        # Key insights
        f.write("KEY INSIGHTS\n")
        f.write("-" * 40 + "\n")
        
        # Best performing model
        if results.elo_ratings:
            sorted_elo = sorted(results.elo_ratings.items(), key=lambda x: x[1].rating, reverse=True)
            best_model = sorted_elo[0][0]
            f.write(f"• Best Overall Model (ELO): {best_model}\n")
        else:
            # Use average scores instead
            model_avgs = df.groupby('model_name')['avg_score'].mean().sort_values(ascending=False)
            if len(model_avgs) > 0:
                best_model = model_avgs.index[0]
                f.write(f"• Best Overall Model (Avg Score): {best_model} ({model_avgs.iloc[0]:.1f} avg)\n")
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
        
        # Claude-Human agreement
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
            f.write(f"• Claude-Human Agreement: {avg_correlation:.3f} average correlation ({len(overall_correlations)}/{len(dimensions)} dimensions)\n")
        else:
            f.write(f"• Claude-Human Agreement: N/A (insufficient human data)\n")
        
        f.write(f"\n• Total Pairwise Comparisons: {len(results.pairwise_comparisons)}\n")
        f.write(f"• Total Concepts Evaluated: {len(results.model_scores)}\n")
        
        # Count total unique models across all concepts
        all_models = set()
        for concept_scores in results.model_scores.values():
            for model_score in concept_scores:
                all_models.add(model_score.model_name)
        f.write(f"• Total Models: {len(all_models)}\n")


# -----------------------------------------------------------------------------
# LLM Response Caching
# -----------------------------------------------------------------------------

def get_cache_key(cache_type: str, session_id: str, trial: int = 0, **kwargs) -> str:
    """Generate cache key for LLM responses"""
    # Create a hash of the additional parameters for uniqueness
    import hashlib
    
    if cache_type == "detailed":
        # For detailed scoring: session_id + mesh_count + mesh_index + trial
        mesh_count = kwargs.get("mesh_count", 0)
        mesh_index = kwargs.get("mesh_index", 0)
        key_data = f"{session_id}_{mesh_count}_{mesh_index}_{trial}"
    elif cache_type == "pairwise":
        # For pairwise: concept_session + model_a + model_b + trial
        concept_session = kwargs.get("concept_session_id", "")
        model_a = kwargs.get("model_a", "")
        model_b = kwargs.get("model_b", "")
        key_data = f"{concept_session}_{model_a}_{model_b}_{trial}"
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
    
    # Create hash to handle long/special characters
    cache_hash = hashlib.md5(key_data.encode()).hexdigest()
    return f"{cache_type}_{cache_hash}.json"

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-eval-json", required=True, type=Path, help="Path to leaderboard_models.json with human evaluations")
    parser.add_argument("--debug", action="store_true", help="Save all LLM queries to debug_queries/ folder")
    parser.add_argument("--skip-pairwise", action="store_true", help="Skip pairwise comparisons (only run 4D evaluation)")
    
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
    
    args = parser.parse_args()

    # Handle cache management commands
    if args.cache_stats:
        stats = get_cache_stats()
        print("📋 Cache Statistics:")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Detailed responses: {stats['detailed_responses']}")
        print(f"   Pairwise responses: {stats['pairwise_responses']}")
        print(f"   Total size: {stats['total_size_mb']} MB")
        sys.exit(0)
    
    if args.clear_cache:
        clear_cache()
        print("✅ Cache cleared")
        if not args.config:  # If only clearing cache, exit
            sys.exit(0)

    # Get API keys from config file or prompt user
    print("🚀 Starting Hybrid LLM Leaderboard for 3D Mesh Evaluation")
    print("=" * 60)
    print("📋 Features:")
    print("  • 4D Detailed Scoring (StructuralForm, PartCoverage, SurfaceDetail, TextureQuality)")
    print("  • Pairwise ELO Rankings")
    print("  • Comprehensive Analytics & Visualizations") 
    print("  • Claude vs Human Comparison Analysis")
    if not args.no_cache:
        cache_stats = get_cache_stats()
        print(f"  • LLM Response Caching ({cache_stats['total_files']} cached responses)")
    print()
    
    if args.debug:
        print("🐛 Debug mode enabled - saving all queries to debug_queries/")
        DEBUG_DIR.mkdir(exist_ok=True)
    
    if args.skip_pairwise:
        print("⏭️  Skipping pairwise comparisons - running 4D evaluation only")
    
    if args.no_cache:
        print("🚫 Cache disabled - will always call LLM APIs")
    
    if args.human_eval_only:
        print("👤 Human-eval-only mode - evaluating only sessions with human evaluations")
    
    if args.trials:
        print(f"🔄 Running {args.trials} trials for statistical robustness")
    elif args.trial > 0:
        print(f"🎯 Running trial #{args.trial}")
    
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    
    csm_key, claude_key = get_api_keys()
    
    # Handle preview-only mode
    if args.preview_only:
        print("👀 Preview Mode - No evaluation will be run")
        # Just run preview and exit
        run_leaderboard(
            args.human_eval_json, csm_key, claude_key, args.debug, 
            args.skip_pairwise, args.trial, args.no_cache,
            preview=True, job_tracking_path=args.job_tracking,
            human_eval_only=args.human_eval_only
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
                args.human_eval_json, csm_key, claude_key, args.debug, 
                args.skip_pairwise, trial_num, args.no_cache,
                preview=(trial_num == 1 and not args.no_preview),  # Only preview first trial
                job_tracking_path=args.job_tracking, human_eval_only=args.human_eval_only
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
            args.human_eval_json, csm_key, claude_key, args.debug, 
            args.skip_pairwise, args.trial, args.no_cache,
            preview=not args.no_preview, job_tracking_path=args.job_tracking,
            human_eval_only=args.human_eval_only
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
    
    if results.elo_ratings:
        print(f"📈 ELO rankings: {len(results.elo_ratings)} models rated")
    else:
        print(f"📈 Models evaluated: {len(all_models)} (no ELO rankings - only single models per concept)")
    
    print(f"🥊 Pairwise comparisons: {len(results.pairwise_comparisons)} head-to-head matchups")
    print(f"📊 Detailed analysis saved to: {RESULTS_DIR.resolve()}")
    
    if not args.no_cache:
        final_stats = get_cache_stats()
        print(f"📋 Cache: {final_stats['total_files']} responses ({final_stats['total_size_mb']} MB)")
