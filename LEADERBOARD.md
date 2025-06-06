# LLM Leaderboard for 3D Mesh Evaluation

An automated evaluation system that uses Claude's vision capabilities to assess 3D mesh reconstructions against concept images, providing objective scoring aligned with human evaluation standards.

## Overview

The leaderboard compares different 3D generation models by evaluating their outputs against reference concept images. It downloads mesh renders from CSM.ai sessions and uses Claude Opus 4 to score them across multiple quality dimensions.

### Data Sources

The system supports two evaluation modes:

1. **Mixed Mode (Default)**: Combines manual entries with human evaluations + auto-discovered sessions from job tracking
2. **Human-Eval-Only Mode**: Evaluates only sessions with existing human evaluations for focused Claude vs Human comparison

#### Workflow:
1. **Manual entries** (with human evaluations) are loaded from your JSON file
2. **Auto-discovery** finds additional sessions from `job_tracking.json` 
3. **Model names** and **concept images** are automatically extracted from CSM API
4. **Preview** shows what will be evaluated before running (can be skipped)
5. **Evaluation** proceeds with Claude scoring all sessions

## Key Features

- **🎯 Human-Aligned Scoring**: Calibrated to match human evaluation standards (0-10 scale)
- **🔍 Multi-Dimensional Evaluation**: Scores structural form, part coverage, surface detail, and texture quality
- **📊 Array-Based Results**: Returns detailed dimension-wise scores instead of single values
- **🥊 Pairwise ELO Rankings**: Head-to-head comparisons for overall performance ranking
- **🧩 Smart Multi-Part Handling**: Optimized aggregation for kit models (uses best 80% of parts)
- **📊 Side-by-Side Comparison**: Includes both Claude and human scores in results
- **🐛 Debug Mode**: Saves all queries, images, and responses for analysis
- **🔐 Secure API Management**: Automatic key storage and retrieval
- **💾 LLM Response Caching**: Intelligent caching system to avoid repeated API calls
- **🔄 Multiple Trials Support**: Statistical robustness through repeated evaluations
- **📈 Comprehensive Analytics**: Rich visualizations and statistical analysis
- **🔄 Partial Human Evaluations**: Support for incomplete human scoring

## Scoring Rubric

### Dimensions (Unweighted, 0-10 scale)
1. **Structural Form** - Overall geometric accuracy and proportions (can be evaluated for complete models and individual parts)
2. **Part Coverage** - Single mesh: completeness; Multi-part: how well this part represents its intended portion
3. **Surface Detail** - Quality of geometric detail, surface features, mesh quality
4. **Texture Quality** - Texture quality, color accuracy, material properties

### Scoring Guidelines
- **0**: Unusable, completely failed, or missing (untextured models get Texture Quality = 0)
- **1-2**: Poor quality, major problems
- **3-4**: Below average, significant issues
- **5-6**: Average, usable with some issues
- **7-8**: Good quality, minor issues only
- **9-10**: Excellent to perfect

### Scoring Rules

#### Single Mesh Models (mesh_count == 1)
- Score all dimensions normally
- **Critical Penalty**: If the object visually SHOULD be multi-part (e.g., limbs, wheels, hinges), set Part Coverage = 0

#### Multi-Part Models (mesh_count > 1)
- Score all dimensions for this individual part
- Individual parts in well-decomposed kits should score reasonably (5-8 range) as decomposition itself is valuable

### Critical Scoring Rules
- **Untextured models**: Texture Quality = 0
- **Wrong decomposition**: Single mesh that should be multi-part gets Part Coverage = 0

## Usage

### Quick Start
```bash
python llm_leaderboard.py --human-eval-json leaderboard_models.json
```

### Human-Eval-Only Mode
```bash
# Only evaluate sessions with human evaluations (skip auto-discovery)
python llm_leaderboard.py --human-eval-json leaderboard_models.json --human-eval-only
```

### With Caching & Multiple Trials
```bash
# Run 3 independent trials for statistical robustness
python llm_leaderboard.py --human-eval-json leaderboard_models.json --trials 3

# Run specific trial number
python llm_leaderboard.py --human-eval-json leaderboard_models.json --trial 2

# Fresh evaluation (no cache)
python llm_leaderboard.py --human-eval-json leaderboard_models.json --no-cache
```

### Cache Management
```bash
# Check cache statistics
python llm_leaderboard.py --cache-stats

# Clear entire cache
python llm_leaderboard.py --clear-cache

# Clear cache then run fresh evaluation
python llm_leaderboard.py --human-eval-json leaderboard_models.json --clear-cache
```

### Preview & Auto-Discovery
```bash
# Preview evaluation plan without running
python llm_leaderboard.py --human-eval-json leaderboard_models.json --preview-only

# Skip preview confirmation
python llm_leaderboard.py --human-eval-json leaderboard_models.json --no-preview

# Specify custom job tracking file for auto-discovery
python llm_leaderboard.py --human-eval-json leaderboard_models.json --job-tracking my_jobs.json
```

### Advanced Options
```bash
# Debug mode with full query logging
python llm_leaderboard.py --human-eval-json leaderboard_models.json --debug

# Skip expensive pairwise comparisons (faster, 4D evaluation only)
python llm_leaderboard.py --human-eval-json leaderboard_models.json --skip-pairwise

# Multiple trials without pairwise comparisons
python llm_leaderboard.py --human-eval-json leaderboard_models.json --trials 5 --skip-pairwise

# Human-eval-only with debug mode
python llm_leaderboard.py --human-eval-json leaderboard_models.json --human-eval-only --debug
```

### Configuration Format

The leaderboard supports two data sources:
1. **Manual entries** with human evaluations (from JSON file)
2. **Auto-discovered sessions** (from job_tracking.json)

#### Basic Structure (leaderboard_models.json)
```json
[
  {
    "session_id": "SESSION_1749148355_6335964",
    "human_eval": {
      "StructuralForm": 7.0,
      "PartCoverage": 8.0, 
      "SurfaceDetail": 6.0,
      "TextureQuality": 9.0
    }
  },
  {
    "session_id": "SESSION_1749146142_3027706",
    "human_eval": {
      "StructuralForm": 8.0,
      "PartCoverage": 7.5, 
      "SurfaceDetail": 8.5,
      "TextureQuality": 8.0
    }
  }
]
```

#### Auto-Discovery
The script automatically imports additional sessions from `job_tracking.json` (any file containing session IDs). Model names and concept images are auto-extracted from the CSM API.

#### Partial Human Evaluations
```json
{
  "human_eval": {
    "StructuralForm": 7.0,        // Complete score
    "PartCoverage": null,         // Missing/not evaluated yet  
    "SurfaceDetail": 8.0,         // Complete score
    "TextureQuality": 6.5         // Complete score
  }
}
```

## Results

### Console Output (Enhanced)
```
🎯 Phase 1: Detailed 4D Evaluation
==================================================

[+] Evaluating concept session SESSION_1749156106_6153233 …
    • csm-kit-baked_1555937     → avg:9.0 [StructuralForm:9.0, PartCoverage:9.0, SurfaceDetail:9.0, TextureQuality:9.0]
      Human: avg:9.0 [StructuralForm:9.0 | PartCoverage:9.0 | SurfaceDetail:9.0 | TextureQuality:9.0]
    • csm-base-none_5410537     → avg:6.2 [StructuralForm:7.0, PartCoverage:5.5, SurfaceDetail:6.8, TextureQuality:5.5]
      Human: avg:6.5 (partial: 3/4) [StructuralForm:7.0 | PartCoverage:N/A | SurfaceDetail:8.0 | TextureQuality:6.5]

🥊 Phase 2: Pairwise ELO Comparisons
==================================================

[+] Pairwise comparisons for concept SESSION_1749156106_6153233
    csm-kit-baked_1555937 vs csm-base-none_5410537: A wins (high confidence)

📊 Phase 3: Comprehensive Analysis
==================================================

🔍 Generating comprehensive analysis in results/...
✅ Analysis complete! Check results/ for all reports and visualizations.

===== Traditional 4D Score Leaderboard =====
SESSION_1749156106_6153233: winner = csm-kit-baked_1555937 (9.0 avg)
   csm-kit-baked_1555937    Claude: avg:9.0 [StructuralForm:9.0 | PartCoverage:9.0 | SurfaceDetail:9.0 | TextureQuality:9.0]
                            Human:  avg:9.0 [StructuralForm:9.0 | PartCoverage:9.0 | SurfaceDetail:9.0 | TextureQuality:9.0]

===== ELO Rankings (Pairwise Performance) =====
1. csm-kit-baked_1555937  1547 ELO (1-0-0) 100.0% win rate
2. csm-base-none_5410537  1453 ELO (0-1-0) 0.0% win rate
```

### Generated Visualizations & Reports
```
results/
├── elo_rankings.png              # ELO rankings bar chart
├── head_to_head_matrix.png       # Win/loss heatmap
├── radar_charts.png              # Multi-dimensional performance profiles
├── dimension_correlations.png    # Correlation matrix between dimensions
├── score_distributions.png       # Box plots of score distributions
├── consistency_analysis.png      # Model consistency heatmap
├── claude_vs_human_correlation.png # Scatter plots comparing Claude vs human
├── claude_human_agreement.csv    # Detailed agreement statistics
└── summary_report.txt            # Comprehensive analysis summary
```

### CSV Export (Enhanced)
```csv
concept_session,model,claude_avg,claude_StructuralForm,claude_PartCoverage,claude_SurfaceDetail,claude_TextureQuality,human_avg,human_StructuralForm,human_PartCoverage,human_SurfaceDetail,human_TextureQuality,elo_rating,elo_games,elo_wins,elo_losses,elo_ties,win_rate
SESSION_xxx,csm-kit-baked_1555937,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,1547,1,1,0,0,1.000
SESSION_xxx,csm-base-none_5410537,6.2,7.0,5.5,6.8,5.5,6.5,7.0,N/A,8.0,6.5,1453,1,0,1,0,0.000
```

## Special Handling

### Multi-Part Model Aggregation
**Small Multi-Part (≤2 parts)**: Regular average of all parts

**Large Multi-Part (>2 parts)**: Sophisticated failure-aware aggregation per dimension:
- **50%+ parts failed (≤2.0 score)**: Complete failure → 0.0 for that dimension
- **30%+ parts failed**: Use median to be more forgiving of outliers
- **<30% failed**: Use best 80% for that dimension

### Caching System
- **Automatic caching** of all LLM responses in `llm_cache/`
- **Trial-specific caching** - each trial maintains separate cache entries
- **Cost savings** - avoid repeated expensive API calls
- **Cache management** - statistics, clearing, and disable options

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
            ├── full_prompt.json
            ├── response.json
            └── metadata.json
```

## Requirements

```bash
pip install requests tqdm pillow anthropic matplotlib seaborn pandas numpy scipy
```

## Output Format

### Claude Response
```json
{
  "session_id": "SESSION_ID",
  "scores": {
     "StructuralForm": 8,
     "PartCoverage": 7,
     "SurfaceDetail": 9,
     "TextureQuality": 6
  },
  "score_array": [8, 7, 9, 6]
}
```

### Human Evaluation Format
```json
{
  "StructuralForm": 8.0,
  "PartCoverage": 7.5, 
  "SurfaceDetail": 9.0,
  "TextureQuality": 6.5
}
```

### Pairwise Comparison Response
```json
{
  "winner": "A",
  "confidence": "high", 
  "reasoning": "Model A has better overall geometry and texture quality"
}
```

## Statistical Features

### Multiple Trials
- **Independent evaluations** with separate cache entries
- **Statistical robustness** through repeated measurements  
- **Future aggregation support** for confidence intervals

### Comprehensive Analytics
- **ELO rankings** for overall model performance
- **Dimensional analysis** showing strengths/weaknesses
- **Consistency metrics** identifying reliable vs volatile models
- **Claude-Human agreement** correlation analysis across dimensions

### Partial Human Evaluations
- **Flexible scoring** - some dimensions can be missing
- **Incremental evaluation** - add scores over time
- **Smart averaging** - excludes missing dimensions from calculations
- **Clear indicators** when evaluations are incomplete

## Command Line Reference

```bash
# Basic usage
python llm_leaderboard.py --human-eval-json leaderboard_models.json

# Mode selection
python llm_leaderboard.py --human-eval-json models.json --human-eval-only    # Only human-eval sessions
python llm_leaderboard.py --human-eval-json models.json                      # Manual + auto-discovery

# Preview and job tracking
python llm_leaderboard.py --human-eval-json models.json --preview-only       # Preview only
python llm_leaderboard.py --human-eval-json models.json --no-preview         # Skip preview
python llm_leaderboard.py --human-eval-json models.json --job-tracking jobs.json

# Cache control
python llm_leaderboard.py --human-eval-json models.json --no-cache           # Disable cache
python llm_leaderboard.py --cache-stats                                      # Show cache info
python llm_leaderboard.py --clear-cache                                      # Clear all cache

# Multiple trials
python llm_leaderboard.py --human-eval-json models.json --trials 5           # Run 5 trials
python llm_leaderboard.py --human-eval-json models.json --trial 2            # Run trial #2

# Evaluation options  
python llm_leaderboard.py --human-eval-json models.json --skip-pairwise      # Skip ELO comparisons
python llm_leaderboard.py --human-eval-json models.json --debug              # Enable debug logging

# Combined options
python llm_leaderboard.py --human-eval-json models.json --trials 3 --debug --skip-pairwise
python llm_leaderboard.py --human-eval-json models.json --human-eval-only --debug --no-preview
```

## File Structure

```
project/
├── llm_leaderboard.py          # Main evaluation script
├── leaderboard_models.json     # Configuration with model sessions
├── LEADERBOARD.md             # This documentation
├── CACHE_README.md            # Caching system documentation
├── .leaderboard_config        # API keys (auto-generated, git-ignored)
├── llm_cache/                 # LLM response cache (git-ignored)
├── debug_queries/             # Debug output (optional, git-ignored)
├── results/                   # Analysis results and visualizations
└── leaderboard_results.csv    # Detailed results export
```

This system provides both **quick model rankings** and **deep analytical insights** for comprehensive 3D mesh evaluation research! 🎯
