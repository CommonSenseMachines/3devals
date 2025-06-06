# LLM Response Caching System

The leaderboard now includes a sophisticated caching system to avoid repeated API calls and support multiple trials for statistical robustness.

## 🚀 Key Features

### **Intelligent Caching**
- **Automatic caching** of all LLM responses (detailed scoring + pairwise comparisons)
- **Cache persistence** across runs - responses are saved locally in `llm_cache/`
- **Unique cache keys** based on session IDs, trial numbers, and comparison parameters
- **Metadata tracking** - timestamps, trial info, and evaluation context

### **Multiple Trials Support**
- Run **multiple independent trials** for statistical reliability
- Each trial uses different cache keys to ensure independent evaluations
- **Future aggregation** support for confidence intervals and statistical analysis

### **Cache Management**
- **Cache statistics** - see how many responses are cached and storage usage
- **Selective clearing** - clear specific cache patterns or entire cache
- **Cache disable** option for fresh evaluations

## 📋 Usage Examples

### Basic Usage (with caching)
```bash
# Normal run - uses cache automatically
python llm_leaderboard.py --human-eval-json leaderboard_models.json

# Fresh evaluation (no cache)
python llm_leaderboard.py --human-eval-json leaderboard_models.json --no-cache

# Human-eval-only mode (focused Claude vs Human comparison)
python llm_leaderboard.py --human-eval-json leaderboard_models.json --human-eval-only
```

### Multiple Trials
```bash
# Run 3 independent trials for statistical robustness
python llm_leaderboard.py --human-eval-json leaderboard_models.json --trials 3

# Run specific trial number
python llm_leaderboard.py --human-eval-json leaderboard_models.json --trial 2
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

# Custom job tracking file
python llm_leaderboard.py --human-eval-json leaderboard_models.json --job-tracking my_jobs.json
```

### Advanced Options
```bash
# Debug mode with caching
python llm_leaderboard.py --human-eval-json leaderboard_models.json --debug

# Skip pairwise comparisons (faster, 4D evaluation only)
python llm_leaderboard.py --human-eval-json leaderboard_models.json --skip-pairwise

# Multiple trials without pairwise comparisons
python llm_leaderboard.py --human-eval-json leaderboard_models.json --trials 5 --skip-pairwise

# Human-eval-only with debug mode
python llm_leaderboard.py --human-eval-json leaderboard_models.json --human-eval-only --debug
```

## 🗂️ Cache Structure

```
llm_cache/
├── detailed_a1b2c3d4.json          # Detailed scoring responses
├── detailed_e5f6g7h8.json
├── pairwise_x9y0z1a2.json          # Pairwise comparison responses  
├── pairwise_b3c4d5e6.json
└── ...
```

### Cache File Format
```json
{
  "response": {
    "session_id": "SESSION_123",
    "scores": {...},
    "score_array": [8, 7, 9, 6]
  },
  "metadata": {
    "session_id": "SESSION_123",
    "mesh_count": 1,
    "mesh_index": 0,
    "trial": 0,
    "concept_session_id": "CONCEPT_456",
    "model_name": "csm-turbo-baked"
  },
  "timestamp": "2024-01-15T10:30:45.123456",
  "cache_version": "1.0"
}
```

## 🔧 Cache Keys

### Detailed Scoring
**Format**: `detailed_{hash}.json`
**Hash includes**: session_id + mesh_count + mesh_index + trial

### Pairwise Comparisons  
**Format**: `pairwise_{hash}.json`
**Hash includes**: concept_session + model_a + model_b + trial

## ⚠️ Important Notes

1. **Git Ignore**: The `llm_cache/` directory is excluded from version control
2. **Trial Independence**: Each trial number creates separate cache entries
3. **API Cost Savings**: Cached responses avoid repeated expensive API calls
4. **Cache Validity**: Cache persists until manually cleared or files deleted
5. **Storage Usage**: Monitor cache size with `--cache-stats`

## 🎯 Experimental Workflow

### Statistical Evaluation
```bash
# Run multiple trials for statistical significance
python llm_leaderboard.py --human-eval-json leaderboard_models.json --trials 5

# Add new models and re-run (reuses cached responses for existing models)
# Edit leaderboard_models.json to add new entries
python llm_leaderboard.py --human-eval-json leaderboard_models.json --trials 5

# Clear cache for fresh baseline
python llm_leaderboard.py --clear-cache
python llm_leaderboard.py --human-eval-json leaderboard_models.json --trials 5

# Human-eval-only focused analysis
python llm_leaderboard.py --human-eval-json leaderboard_models.json --human-eval-only --trials 3
```

### Development & Testing
```bash
# Quick development iteration (skip expensive pairwise comparisons)
python llm_leaderboard.py --human-eval-json test_models.json --skip-pairwise

# Test with no cache during development
python llm_leaderboard.py --human-eval-json test_models.json --no-cache --skip-pairwise

# Preview evaluation plan without running
python llm_leaderboard.py --human-eval-json test_models.json --preview-only

# Full evaluation with debug logging
python llm_leaderboard.py --human-eval-json leaderboard_models.json --debug --trials 3
```

## 📊 Future Enhancements

- **Trial Aggregation**: Statistical analysis across multiple trials
- **Confidence Intervals**: Uncertainty quantification for rankings
- **Cache Optimization**: Compression and cleanup strategies
- **Distributed Caching**: Shared cache across multiple machines

## 🔄 Partial Human Evaluations

The system now supports **partial human evaluations** where some dimensions may be missing or incomplete.

### **Supported Formats**
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

### **How It Works**
- **Missing dimensions** (null/missing) are excluded from average calculations
- **Display shows** "N/A" for missing scores and indicates partial evaluation
- **Analytics handle** missing data using pandas NaN functionality
- **Correlations calculated** only on dimensions with valid human scores

### **Example Output**
```
csm-base-none_5410537    Claude: avg:6.2 [StructuralForm:7.0 | PartCoverage:5.5 | SurfaceDetail:6.8 | TextureQuality:5.5]
                         Human:  avg:6.5 (partial: 3/4) [StructuralForm:7.0 | PartCoverage:N/A | SurfaceDetail:8.0 | TextureQuality:6.5]
```

This enables **incremental human evaluation** - you can start evaluating and add missing dimensions later without affecting existing scores!

## 👤 Human-Eval-Only Mode

The new `--human-eval-only` flag provides focused Claude vs Human comparison:

### **Benefits**
- **Quality control**: Only evaluate curated sessions with validated human scores
- **Faster evaluation**: Skip potentially hundreds of auto-discovered sessions  
- **Better comparison**: More meaningful Claude vs Human correlation analysis
- **Cache efficiency**: Smaller cache footprint for focused studies

### **Usage**
```bash
# Focus on sessions with human evaluations only
python llm_leaderboard.py --human-eval-json leaderboard_models.json --human-eval-only

# With caching and multiple trials for validation studies
python llm_leaderboard.py --human-eval-json leaderboard_models.json --human-eval-only --trials 3

# Quick preview of what would be evaluated
python llm_leaderboard.py --human-eval-json leaderboard_models.json --human-eval-only --preview-only
```

### **When to Use**
- **Validation studies**: Comparing Claude's scoring accuracy against human evaluators
- **Model calibration**: Fine-tuning evaluation criteria based on human standards
- **Quality assurance**: Ensuring Claude's scores align with expert assessments
- **Research**: Publishing results with human-validated baselines 