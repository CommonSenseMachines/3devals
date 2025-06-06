#!/usr/bin/env python3
"""
leaderboard.py – Run a vision‑LLM leaderboard on CSM mesh reconstructions.

Given a JSON manifest (see example below) the script:
  1. Downloads each *concept* image (source photograph / sketch).
  2. Downloads one‑or‑more *render* images for every candidate model.
  3. Sends the image pair(s) to Claude's vision API and
     records the rubric scores it returns.
  4. Ranks the candidates per concept and prints a scoreboard.

Example leaderboard_models.json
-------------------------------
{
  "INPUT_IMAGE_SESSION_ID_1": [
    {"csm-kit": ["PART_SESSION_ID2", "PART_SESSION_ID3"]},
    {"csm-turbo": "SESSION_ID3"}
  ],
  "INPUT_IMAGE_SESSION_ID_2": [
    {"csm-base": "SESSION_ID4"}
  ]
}

Run:
  python leaderboard.py --config leaderboard_models.json

Requirements:
  pip install requests tqdm pillow python-dotenv anthropic
"""

import argparse
import base64
import json
import os
import sys
import getpass
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import anthropic
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config & Constants
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are MeshCritique-v0.5, a 3D asset reviewer calibrated to human evaluation standards.\n"
    "Scale every dimension from 0 (unusable) to 10 (excellent).\n"
    "CRITICAL: Untextured models (gray/white) should score 0-1 total. Textured models with good geometry score 6-8.\n"
    "MULTI-PART BONUS: Multi-part decomposition is valuable - score individual parts generously (6-8 range) if they represent reasonable decomposition.\n"
    "Return ONLY the JSON object, no explanations, no additional text, no reasoning."
)

USER_RUBRIC = """Rubric dimensions and weights:
1  Silhouette         ×2  (Single mesh only - set "NA" for individual parts)
2  Part Coverage      ×1.5
3  Surface Detail     ×1
4  Texture Quality¹  ×1
5  Joint Readiness²  ×0.5

Scoring Guidelines:
• 0-1: Unusable (untextured models, major flaws)
• 2-3: Poor quality, obvious problems
• 4-5: Mediocre, needs significant work
• 6-7: Good quality, usable with minor issues
• 8-9: Excellent, professional quality
• 10: Perfect

Scoring Rules:
• For SINGLE MESH (mesh_count == 1): Score all dimensions normally
• For MULTI-PART (mesh_count > 1): 
  - Silhouette = "NA" (can't evaluate without assembly)
  - Part Coverage: How well this part represents its intended portion (be generous - successful decomposition is valuable)
  - Surface Detail: Quality of geometric detail on this part (score 7+ if decent quality)
  - Texture Quality: Texture quality on this part
  - Joint Readiness: How well-defined are connection points for assembly (score 7+ if reasonable)
  - MULTI-PART BONUS: Individual parts in well-decomposed kits should score higher (6-8 range) as decomposition itself is valuable

Critical Penalties:
• If mesh_count == 1 **and** it visually SHOULD be multi‑part (e.g., limbs, wheels, hinges), score Part Coverage = 0.
• If the render looks completely untextured (solid gray/white, no color/material), set Texture Quality = \"NA\" AND reduce all other scores by 50%.
• Untextured models should have weighted_total ≤ 1.0 unless geometry is exceptional.

Output JSON schema:
{
  "session_id": string,
  "scores": {
     "Silhouette": int | "NA",
     "PartCoverage": int,
     "SurfaceDetail": int,
     "TextureQuality": int | "NA",
     "JointReadiness": int
  },
  "weighted_total": float  // 0-10, excluding "NA" dimensions from calculation
}
"""

HEADERS = {"Content-Type": "application/json"}

CONFIG_FILE = Path(".leaderboard_config")
DEBUG_DIR = Path("debug_queries")

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
) -> Dict:
    """Send a single render vs. concept pair to Claude and return the parsed JSON."""

    client = anthropic.Anthropic(api_key=claude_key)
    
    # Create context about which part we're evaluating
    if mesh_count == 1:
        part_context = "This is a single-mesh reconstruction."
    else:
        part_context = f"This is part {mesh_index + 1} of {mesh_count} in a multi-part reconstruction. NOTE: Part numbering is arbitrary/random - part 1 doesn't necessarily correspond to any specific semantic part of the object."
    
    # Prepare the scoring instruction based on mesh count
    if mesh_count == 1:
        scoring_instruction = "score all dimensions normally"
    else:
        scoring_instruction = 'set Silhouette = "NA" since individual parts cannot be evaluated for silhouette without assembly'
    
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
            "text": f"Now please evaluate how well this 3D mesh render matches the concept image.\n\nSession ID: {session_id}\nMesh count: {mesh_count}\nCurrent mesh: {mesh_index + 1} of {mesh_count}\n\nIMPORTANT: Since mesh_count = {mesh_count}, {scoring_instruction}.\n\nRemember: If mesh_count == 1 and the object should logically be multi-part, score Part Coverage = 0."
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
        
        return response_json
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON returned by Claude: {final_text}") from exc
    except Exception as exc:
        raise RuntimeError(f"Error calling Claude API: {exc}") from exc


# -----------------------------------------------------------------------------
# Core leaderboard logic
# -----------------------------------------------------------------------------

def evaluate_candidate(
    claude_key: str,
    api_key: str,
    concept_session_id: str,
    concept_img_url: str,
    candidate_name: str,
    candidate_session_ids: List[str],
    debug_enabled: bool = False,
) -> float:
    """Return mean weighted_total across all renders for this candidate."""

    concept_img = download_image(concept_img_url)
    concept_b64 = img_to_b64(concept_img)

    totals: List[float] = []

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
            )
            totals.append(score_json["weighted_total"])

    if not totals:
        return 0.0
    
    # For multi-part models (kits), check for failed parts and apply penalties
    if len(totals) > 2:  # Multi-part model
        failed_parts = [score for score in totals if score <= 2.0]
        
        # If any parts completely failed (≤2.0), apply penalties based on context
        if failed_parts:
            # Calculate penalty based on failure rate and quality of good parts
            failure_rate = len(failed_parts) / len(totals)
            good_parts = [score for score in totals if score > 2.0]
            avg_good_score = sum(good_parts) / len(good_parts) if good_parts else 0
            
            if failure_rate >= 0.5:  # 50%+ parts failed
                return 0.0  # Complete failure
            elif failure_rate >= 0.4:  # 40%+ parts failed
                return min(1.0, sum(totals) / len(totals))  # Cap at 1.0  
            elif failure_rate >= 0.3:  # 30%+ parts failed  
                return min(2.0, sum(totals) / len(totals))  # Cap at 2.0
            elif failure_rate >= 0.2 and avg_good_score < 6.0:  # 20%+ failed AND poor quality
                return min(3.0, sum(totals) / len(totals))  # Cap at 3.0
            else:  # <20% failed OR high quality good parts
                # Use best 80% with light penalty for failed parts
                sorted_totals = sorted(totals, reverse=True)
                best_count = max(2, int(len(sorted_totals) * 0.8))
                best_totals = sorted_totals[:best_count]
                base_score = sum(best_totals) / len(best_totals)
                # Light penalty: reduce by 10% per failed part
                return base_score * (1.0 - failure_rate * 0.1)
        else:
            # No failed parts - use best 80% normally
            sorted_totals = sorted(totals, reverse=True)
            best_count = max(2, int(len(sorted_totals) * 0.8))
            best_totals = sorted_totals[:best_count]
            return sum(best_totals) / len(best_totals)
    else:
        # Single part or small multi-part - use regular average
        return sum(totals) / len(totals)


def run_leaderboard(config_path: Path, api_key: str, claude_key: str, debug_enabled: bool = False):
    cfg = json.loads(Path(config_path).read_text())

    overall_results: Dict[str, Dict[str, float]] = {}
    human_evals: Dict[str, Dict[str, float]] = {}

    for concept_session_id, candidate_list in cfg.items():
        print(f"\n[+] Evaluating concept session {concept_session_id} …")
        concept_url = f"https://api.csm.ai/v3/sessions/{concept_session_id}"
        concept_sess = fetch_json(concept_url, api_key)
        concept_img_url = concept_sess["input"]["image"]["data"]["image_url"]

        per_candidate: Dict[str, float] = {}
        human_per_candidate: Dict[str, float] = {}
        for entry in tqdm(candidate_list, desc="candidates", unit="model"):
            if not isinstance(entry, dict):
                raise ValueError("Each candidate entry must be a dict {model: session(s)}")
            
            # Extract model name, session IDs, and human eval
            model_items = [(k, v) for k, v in entry.items() if k != "human_eval"]
            if not model_items:
                raise ValueError("Entry must contain at least one model mapping")
            model_name, session_val = model_items[0]
            session_ids = session_val if isinstance(session_val, list) else [session_val]
            human_eval = entry.get("human_eval", "N/A")
            
            # Make model name unique by appending session ID suffix for clarity
            unique_model_name = model_name
            if len(session_ids) == 1:
                session_suffix = session_ids[0].split('_')[-1]  # Get last part of session ID
                unique_model_name = f"{model_name}_{session_suffix}"

            score = evaluate_candidate(
                claude_key,
                api_key,
                concept_session_id,
                concept_img_url,
                model_name,
                session_ids,
                debug_enabled,
            )
            per_candidate[unique_model_name] = score
            human_per_candidate[unique_model_name] = float(human_eval) if human_eval != "N/A" else None
            print(f"    • {unique_model_name:<15} → {score:.3f} (human: {human_eval})")

        overall_results[concept_session_id] = per_candidate
        human_evals[concept_session_id] = human_per_candidate

    # Print aggregated leaderboard
    print("\n===== Leaderboard =====")
    for concept_id, scores in overall_results.items():
        top_model = max(scores, key=scores.get)
        print(f"{concept_id}: winner = {top_model} ({scores[top_model]:.3f})")
        for model, sc in sorted(scores.items(), key=lambda x: -x[1]):
            human_sc = human_evals[concept_id].get(model, "N/A")
            human_str = f"{human_sc:.1f}" if human_sc is not None else "N/A"
            print(f"   {model:<15} {sc:.3f} (human: {human_str})")

    # Write enhanced CSV with human evals
    csv_path = Path("leaderboard_results.csv")
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("concept_session,model,claude_score,human_eval\n")
        for concept_id, scores in overall_results.items():
            for model, claude_sc in scores.items():
                human_sc = human_evals[concept_id].get(model)
                human_str = f"{human_sc:.1f}" if human_sc is not None else "N/A"
                f.write(f"{concept_id},{model},{claude_sc:.1f},{human_str}\n")
    print(f"\nResults written to {csv_path.resolve()}")


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
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path, help="Path to leaderboard_models.json")
    parser.add_argument("--debug", action="store_true", help="Save all LLM queries to debug_queries/ folder")
    args = parser.parse_args()

    # Get API keys from config file or prompt user
    print("🚀 Starting LLM Leaderboard for 3D Mesh Evaluation")
    print("=" * 50)
    
    if args.debug:
        print("🐛 Debug mode enabled - saving all queries to debug_queries/")
        DEBUG_DIR.mkdir(exist_ok=True)
    
    csm_key, claude_key = get_api_keys()
    
    print("📊 Starting evaluation...")
    run_leaderboard(args.config, csm_key, claude_key, args.debug)
