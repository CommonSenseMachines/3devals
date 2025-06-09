#!/usr/bin/env python3
"""
run_leaderboard.py - Simple runner script for the 3D Mesh Evaluation Leaderboard

This script serves as the main entry point and imports the leaderboard functionality
from the leaderboard package.
"""

if __name__ == "__main__":
    try:
        # Import and run the main leaderboard script
        from leaderboard.llm_leaderboard import main
        main()
    except ImportError as e:
        print(f"❌ Error importing leaderboard package: {e}")
        print("   Make sure you're in the correct directory and all dependencies are installed.")
        exit(1) 