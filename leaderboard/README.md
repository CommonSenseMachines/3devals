# Leaderboard Package

This directory contains the 3D Mesh Evaluation Leaderboard system - a comprehensive framework for evaluating 3D mesh reconstructions using multiple LLM providers.

## 📁 Package Structure

```
leaderboard/
├── __init__.py                 # Package initialization
├── llm_leaderboard.py          # Main leaderboard script
├── llm_prompts.py             # Prompt templates and management
├── llm_clients.py             # Multi-provider LLM client system
├── llm_cache.py               # Enhanced caching system
├── mesh_renderer.py           # Multi-view 3D rendering utilities
├── README.md                  # This file
└── tests/                     # Test suite
    ├── __init__.py
    ├── test_llm_integration.py
    ├── test_mesh_renderer.py
    └── test_multiview_integration.py
```

## 🚀 Running the Leaderboard

### From Root Directory (Recommended)
```bash
# Use the simple runner script
python run_leaderboard.py --human-eval-json data.json

# With different providers
python run_leaderboard.py --llm-provider openai --human-eval-json data.json
python run_leaderboard.py --llm-provider gemini --multiview --human-eval-json data.json
```

### From Leaderboard Directory
```bash
cd leaderboard
python llm_leaderboard.py --human-eval-json ../data.json

# Or as a module
python -m leaderboard.llm_leaderboard --human-eval-json ../data.json
```

### As a Python Package
```python
from leaderboard import run_leaderboard, get_api_keys

# Run programmatically
results = run_leaderboard(
    human_eval_json_path=Path("data.json"),
    api_key="your-csm-key",
    llm_client=llm_client
)
```

## 🧪 Running Tests

```bash
# From root directory
python -m pytest leaderboard/tests/

# Or run individual tests
python leaderboard/tests/test_llm_integration.py
python leaderboard/tests/test_mesh_renderer.py
```

## 🔧 Key Features

- **Multi-Provider LLM Support**: Claude, OpenAI, Gemini
- **Multi-View Rendering**: Front/back/side views for comprehensive evaluation
- **Enhanced Caching**: Organized cache with full prompt/response storage
- **Comprehensive Analytics**: ELO rankings, correlation analysis, visualizations
- **Modular Architecture**: Clean separation of concerns

## 📦 Dependencies

See the root directory's requirements for installation instructions.

## 📋 Command Reference

```bash
# List available providers
python run_leaderboard.py --list-providers

# Test provider connection
python run_leaderboard.py --test-provider --llm-provider openai

# Multi-view evaluation
python run_leaderboard.py --multiview --views front back left right

# Cache management
python run_leaderboard.py --cache-stats
python run_leaderboard.py --clear-cache
``` 