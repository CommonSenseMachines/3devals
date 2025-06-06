# LLM Leaderboard for 3D Mesh Evaluation

An automated evaluation system that uses Claude's vision capabilities to assess 3D mesh reconstructions against concept images, providing objective scoring aligned with human evaluation standards.

## Overview

The leaderboard compares different 3D generation models by evaluating their outputs against reference concept images. It downloads mesh renders from CSM.ai sessions and uses Claude Opus 4 to score them across multiple quality dimensions.

## Key Features

- **🎯 Human-Aligned Scoring**: Calibrated to match human evaluation standards (0-10 scale)
- **🔍 Multi-Dimensional Evaluation**: Scores silhouette, part coverage, surface detail, texture quality, and joint readiness
- **🧩 Smart Multi-Part Handling**: Optimized aggregation for kit models (uses best 80% of parts)
- **📊 Side-by-Side Comparison**: Includes both Claude and human scores in results
- **🐛 Debug Mode**: Saves all queries, images, and responses for analysis
- **🔐 Secure API Management**: Automatic key storage and retrieval

## Scoring Rubric

### Dimensions & Weights
1. **Silhouette** ×2 (Single mesh only - "NA" for parts)
2. **Part Coverage** ×1.5 (How well parts represent intended portions)
3. **Surface Detail** ×1 (Geometric detail quality)
4. **Texture Quality** ×1 (Material/color quality)
5. **Joint Readiness** ×0.5 (Connection points for assembly)

### Scoring Scale
- **0-1**: Unusable (untextured models, major flaws)
- **2-3**: Poor quality, obvious problems
- **4-5**: Mediocre, needs significant work
- **6-7**: Good quality, usable with minor issues
- **8-9**: Excellent, professional quality
- **10**: Perfect

## Usage

### Quick Start
```bash
python llm_leaderboard.py --config leaderboard_models.json
```

### With Debug Mode
```bash
python llm_leaderboard.py --config leaderboard_models.json --debug
```

### Configuration Format
```json
{
  "CONCEPT_SESSION_ID": [
    {"model-name": "RECONSTRUCTION_SESSION_ID", "human_eval": "7.0"},
    {"model-name": ["SESSION_1", "SESSION_2"], "human_eval": "8.0"}
  ]
}
```

## Results

### Console Output
```
===== Leaderboard =====
SESSION_xxx: winner = csm-kit-baked_1555937 (7.275)
   csm-kit-baked_1555937 7.275 (human: 9.0)
   csm-turbo-baked_6153233 5.900 (human: 8.0)
   csm-base-none_5410537 4.900 (human: 4.0)
```

### CSV Export
```csv
concept_session,model,claude_score,human_eval
SESSION_xxx,csm-kit-baked_1555937,7.3,9.0
SESSION_xxx,csm-turbo-baked_6153233,5.9,8.0
SESSION_xxx,csm-base-none_5410537,4.9,4.0
```

## Special Handling

### Single vs Multi-Part Models
- **Single Mesh**: All dimensions scored normally
- **Multi-Part (Kits)**: Silhouette = "NA", optimized aggregation using best 80% of parts

### Texture Penalties
- Completely untextured models get heavy penalties (≤1.0 total score)
- TextureQuality = "NA" for gray/white renders

### API Keys
On first run, the system prompts for:
- **CSM API Key**: Get from [3d.csm.ai](https://3d.csm.ai) → Profile Settings → Developer Settings
- **Claude API Key**: Get from [console.anthropic.com](https://console.anthropic.com)

Keys are securely stored locally in `.leaderboard_config` (excluded from git).

## Debug Mode

Saves complete evaluation data in `debug_queries/`:
```
debug_queries/
└── concept_SESSION_ID/
    └── model_NAME/
        └── part_X_of_Y/
            ├── concept_image.jpg
            ├── render_image.jpg
            ├── prompt.txt
            ├── response.json
            └── metadata.json
```

## Requirements

```bash
pip install requests tqdm pillow anthropic
```

## Alignment Results

Current system achieves excellent alignment with human evaluations:
- **Perfect matches**: 2/5 models (0.0 difference)
- **Close alignment**: 3/5 models (≤2.0 difference)
- **Overall correlation**: Strong positive correlation with human standards
