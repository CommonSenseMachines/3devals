"""
Leaderboard Package - 3D Mesh Evaluation System

A comprehensive system for evaluating 3D mesh reconstructions using multiple LLM providers
with support for multi-view rendering and human evaluation comparison.
"""

from .llm_leaderboard import run_leaderboard, get_api_keys, create_llm_client_from_config

__version__ = "1.0.0"
__all__ = ["run_leaderboard", "get_api_keys", "create_llm_client_from_config"] 