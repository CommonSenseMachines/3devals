#!/usr/bin/env python3
"""
llm_prompts.py - Prompt templates and management for 3D mesh evaluation

Stores all prompt templates used for different evaluation tasks and LLM providers.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class EvaluationType(Enum):
    """Types of evaluation prompts"""
    DETAILED_SCORING = "detailed_scoring"
    MULTIVIEW_SCORING = "multiview_scoring"
    PAIRWISE_COMPARISON = "pairwise_comparison"

class LLMProvider(Enum):
    """Supported LLM providers"""
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"

@dataclass
class PromptTemplate:
    """Template for LLM prompts with provider-specific adaptations"""
    system_prompt: str
    user_template: str
    provider_specific: Dict[str, Any]
    evaluation_type: EvaluationType
    
    def format_user_prompt(self, **kwargs) -> str:
        """Format the user prompt template with provided arguments"""
        return self.user_template.format(**kwargs)

# -----------------------------------------------------------------------------
# System Prompts
# -----------------------------------------------------------------------------

DETAILED_SCORING_SYSTEM = """You are MeshCritique-v0.5, a 3D asset reviewer calibrated to human evaluation standards.
Scale every dimension from 0 (unusable) to 10 (excellent).
CRITICAL: Untextured models (gray/white) should have Texture Quality = 0. Single meshes that should be multi-part should have Part Coverage = 0.
Return ONLY the JSON object, no explanations, no additional text, no reasoning."""

PAIRWISE_SYSTEM = """You are MeshCritique-v0.5, a 3D asset reviewer. You will compare two 3D mesh reconstructions against a concept image.
Consider overall quality including geometry, textures, completeness, and how well each matches the original concept.
Return ONLY the JSON object, no explanations."""

# -----------------------------------------------------------------------------
# Evaluation Rubric
# -----------------------------------------------------------------------------

EVALUATION_RUBRIC = """Rubric dimensions (unweighted, 0-10 scale):
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
{{
  "session_id": string,
  "scores": {{
     "StructuralForm": int,
     "PartCoverage": int,
     "SurfaceDetail": int,
     "TextureQuality": int
  }},
  "score_array": [int, int, int, int]  // [StructuralForm, PartCoverage, SurfaceDetail, TextureQuality]
}}"""

# -----------------------------------------------------------------------------
# Prompt Templates
# -----------------------------------------------------------------------------

DETAILED_SCORING_TEMPLATE = """I will show you two images that you need to compare:
1. A concept image (original reference)
2. A 3D mesh render to be evaluated

{part_context}

Please evaluate how well the 3D render matches the concept image using the following rubric:

{rubric}

CONCEPT IMAGE (Reference):
{concept_image_placeholder}

3D MESH RENDER (To be evaluated - {part_context_lower}):
{render_image_placeholder}

Now please evaluate how well this 3D mesh render matches the concept image.

Session ID: {session_id}
Mesh count: {mesh_count}
Current mesh: {mesh_index_display}

Remember: 
- If mesh_count == 1 and the object should logically be multi-part, set Part Coverage = 0
- If the model appears untextured (gray/white), set Texture Quality = 0
- Score all 4 dimensions on 0-10 scale"""

MULTIVIEW_SCORING_TEMPLATE = """I will show you a concept image and multiple views of a 3D mesh reconstruction.

{part_context}

Please evaluate how well the 3D mesh matches the concept image using the following rubric:

{rubric}

CONCEPT IMAGE (Reference):
{concept_image_placeholder}

{view_images_section}

Evaluate this 3D reconstruction using ALL available views for the most comprehensive assessment. If you see both CSM pre-rendered views and 3D mesh views, consider them together - they show the same 3D model from different rendering systems and angles.

Session ID: {session_id}
Mesh count: {mesh_count}
Current mesh: {mesh_index_display}
Views shown: {views_list}"""

PAIRWISE_COMPARISON_TEMPLATE = """Compare these two 3D mesh reconstructions against the concept image. Consider:
• Overall geometric accuracy and completeness
• Surface detail and mesh quality  
• Texture/material quality
• How well each represents the original concept

CONCEPT IMAGE (Reference):
{concept_image_placeholder}

3D MESH RECONSTRUCTION A ({model_a_name}):
{render_a_placeholder}

3D MESH RECONSTRUCTION B ({model_b_name}):
{render_b_placeholder}

Which reconstruction (A or B) better represents the concept image? Consider overall geometry, detail, textures, and completeness.

Output JSON schema:
{{
  "winner": "A" | "B" | "tie",
  "confidence": "low" | "medium" | "high",
  "reasoning": "Brief explanation of choice"
}}"""

# -----------------------------------------------------------------------------
# Provider-Specific Configurations
# -----------------------------------------------------------------------------

PROVIDER_CONFIGS = {
    LLMProvider.CLAUDE: {
        "model": "claude-opus-4-20250514",
        "max_tokens": 4096,
        "temperature": 0,
        "image_format": "jpeg",
        "supports_system_prompt": True,
        "supports_multiple_images": True,
    },
    LLMProvider.OPENAI: {
        "model": "o3-2025-04-16",  # Faster O3 variant
        "max_tokens": 4096,
        "temperature": 0,
        "image_format": "jpeg",
        "supports_system_prompt": True,
        "supports_multiple_images": True,
    },
    LLMProvider.GEMINI: {
        "model": "gemini-2.5-pro-preview-06-05",
        "max_tokens": 4096,
        "temperature": 0,
        "image_format": "jpeg",
        "supports_system_prompt": True,
        "supports_multiple_images": True,
    }
}

# -----------------------------------------------------------------------------
# Prompt Template Registry
# -----------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    EvaluationType.DETAILED_SCORING: PromptTemplate(
        system_prompt=DETAILED_SCORING_SYSTEM,
        user_template=DETAILED_SCORING_TEMPLATE,
        provider_specific={
            LLMProvider.CLAUDE: {"image_placeholder": "image"},
            LLMProvider.OPENAI: {"image_placeholder": "image"},
            LLMProvider.GEMINI: {"image_placeholder": "image"},
        },
        evaluation_type=EvaluationType.DETAILED_SCORING
    ),
    
    EvaluationType.MULTIVIEW_SCORING: PromptTemplate(
        system_prompt=DETAILED_SCORING_SYSTEM,
        user_template=MULTIVIEW_SCORING_TEMPLATE,
        provider_specific={
            LLMProvider.CLAUDE: {"image_placeholder": "image"},
            LLMProvider.OPENAI: {"image_placeholder": "image"},
            LLMProvider.GEMINI: {"image_placeholder": "image"},
        },
        evaluation_type=EvaluationType.MULTIVIEW_SCORING
    ),
    
    EvaluationType.PAIRWISE_COMPARISON: PromptTemplate(
        system_prompt=PAIRWISE_SYSTEM,
        user_template=PAIRWISE_COMPARISON_TEMPLATE,
        provider_specific={
            LLMProvider.CLAUDE: {"image_placeholder": "image"},
            LLMProvider.OPENAI: {"image_placeholder": "image"},
            LLMProvider.GEMINI: {"image_placeholder": "image"},
        },
        evaluation_type=EvaluationType.PAIRWISE_COMPARISON
    ),
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_prompt_template(evaluation_type: EvaluationType) -> PromptTemplate:
    """Get prompt template for evaluation type"""
    if evaluation_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")
    return PROMPT_TEMPLATES[evaluation_type]

def get_provider_config(provider: LLMProvider) -> Dict[str, Any]:
    """Get configuration for LLM provider"""
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unknown provider: {provider}")
    return PROVIDER_CONFIGS[provider]

def format_detailed_scoring_prompt(
    part_context: str,
    rubric: str,
    session_id: str,
    mesh_count: int,
    mesh_index: int,
) -> Dict[str, str]:
    """Format prompt for detailed scoring evaluation"""
    template = get_prompt_template(EvaluationType.DETAILED_SCORING)
    
    return {
        "system_prompt": template.system_prompt,
        "user_prompt": template.format_user_prompt(
            part_context=part_context,
            part_context_lower=part_context.lower(),
            rubric=rubric,
            session_id=session_id,
            mesh_count=mesh_count,
            mesh_index_display=f"{mesh_index + 1} of {mesh_count}",
            concept_image_placeholder="[CONCEPT_IMAGE]",
            render_image_placeholder="[RENDER_IMAGE]"
        )
    }

def format_multiview_scoring_prompt(
    part_context: str,
    rubric: str,
    session_id: str,
    mesh_count: int,
    mesh_index: int,
    views_list: List[str],
    view_images_section: str,
) -> Dict[str, str]:
    """Format prompt for multiview scoring evaluation"""
    template = get_prompt_template(EvaluationType.MULTIVIEW_SCORING)
    
    return {
        "system_prompt": template.system_prompt,
        "user_prompt": template.format_user_prompt(
            part_context=part_context,
            rubric=rubric,
            session_id=session_id,
            mesh_count=mesh_count,
            mesh_index_display=f"{mesh_index + 1} of {mesh_count}",
            views_list=", ".join(views_list),
            view_images_section=view_images_section,
            concept_image_placeholder="[CONCEPT_IMAGE]"
        )
    }

def format_pairwise_comparison_prompt(
    model_a_name: str,
    model_b_name: str,
) -> Dict[str, str]:
    """Format prompt for pairwise comparison evaluation"""
    template = get_prompt_template(EvaluationType.PAIRWISE_COMPARISON)
    
    return {
        "system_prompt": template.system_prompt,
        "user_prompt": template.format_user_prompt(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            concept_image_placeholder="[CONCEPT_IMAGE]",
            render_a_placeholder="[RENDER_A]",
            render_b_placeholder="[RENDER_B]"
        )
    }

def create_view_images_section(views: List[str]) -> str:
    """Create the view images section for multiview prompts"""
    sections = []
    for view_name in views:
        # Handle hybrid naming with different image sources
        if view_name.startswith("mesh_"):
            # 3D mesh rendered view
            clean_view = view_name.replace("mesh_", "")
            sections.append(f"3D MESH {clean_view.upper()} VIEW (from 3D file):\n[{view_name.upper()}_IMAGE]")
        elif view_name == "csm_render":
            # CSM's pre-rendered view
            sections.append(f"CSM PRE-RENDERED VIEW (original CSM render):\n[{view_name.upper()}_IMAGE]")
        else:
            # Fallback for other view types
            sections.append(f"3D MESH {view_name.upper()} VIEW:\n[{view_name.upper()}_IMAGE]")
    return "\n\n".join(sections)

def create_hybrid_view_images_section(render_images: Dict[str, str]) -> str:
    """Create the view images section specifically for hybrid approach (CSM + 3D mesh views)"""
    sections = []
    
    # Group by image type for better organization
    csm_renders = []
    mesh_views = []
    other_views = []
    
    for view_name in render_images.keys():
        if view_name == "csm_render":
            csm_renders.append(view_name)
        elif view_name.startswith("mesh_"):
            mesh_views.append(view_name)
        else:
            other_views.append(view_name)
    
    # Add CSM renders first
    for view_name in csm_renders:
        sections.append(f"CSM PRE-RENDERED VIEW:\n[{view_name.upper()}_IMAGE]\n(This is CSM's original rendered view of the 3D model)")
    
    # Add 3D mesh views
    for view_name in sorted(mesh_views):  # Sort for consistent ordering
        clean_view = view_name.replace("mesh_", "")
        sections.append(f"3D MESH {clean_view.upper()} VIEW:\n[{view_name.upper()}_IMAGE]\n(Rendered from the actual 3D mesh file using Blender)")
    
    # Add any other views
    for view_name in other_views:
        sections.append(f"ADDITIONAL VIEW ({view_name.upper()}):\n[{view_name.upper()}_IMAGE]")
    
    return "\n\n".join(sections) 