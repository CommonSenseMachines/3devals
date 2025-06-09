#!/usr/bin/env python3
"""
test_llm_integration.py - Test script for the integrated LLM system

Tests all components of the new LLM system including prompts, clients, and caching.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all LLM system imports work"""
    print("🧪 Testing LLM system imports...")
    
    try:
        from ..llm_prompts import (
            LLMProvider, EvaluationType, EVALUATION_RUBRIC,
            format_detailed_scoring_prompt, format_multiview_scoring_prompt,
            format_pairwise_comparison_prompt, create_view_images_section
        )
        print("✅ llm_prompts imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import llm_prompts: {e}")
        return False
    
    try:
        from ..llm_clients import (
            LLMRequest, LLMResponse, LLMError, create_llm_client,
            get_available_providers, validate_provider_availability
        )
        print("✅ llm_clients imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import llm_clients: {e}")
        return False
    
    try:
        from ..llm_cache import (
            get_cache_manager, save_llm_response, load_llm_response
        )
        print("✅ llm_cache imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import llm_cache: {e}")
        return False
    
    return True

def test_providers():
    """Test available LLM providers"""
    print("\n🤖 Testing LLM providers...")
    
    try:
        from ..llm_clients import get_available_providers
        providers = get_available_providers()
        
        if providers:
            print(f"✅ Available providers: {[p.value for p in providers]}")
            return True
        else:
            print("⚠️  No LLM providers are available")
            print("   Install at least one: pip install anthropic openai google-generativeai")
            return False
    except Exception as e:
        print(f"❌ Error checking providers: {e}")
        return False

def test_prompt_formatting():
    """Test prompt formatting functions"""
    print("\n📝 Testing prompt formatting...")
    
    try:
        from ..llm_prompts import (
            format_detailed_scoring_prompt, format_multiview_scoring_prompt,
            format_pairwise_comparison_prompt, create_view_images_section,
            EVALUATION_RUBRIC
        )
        
        # Test detailed scoring prompt
        detailed_prompt = format_detailed_scoring_prompt(
            part_context="This is a test context",
            rubric=EVALUATION_RUBRIC,
            session_id="TEST_SESSION",
            mesh_count=1,
            mesh_index=0
        )
        
        if "system_prompt" in detailed_prompt and "user_prompt" in detailed_prompt:
            print("✅ Detailed scoring prompt formatting works")
        else:
            print("❌ Detailed scoring prompt missing required fields")
            return False
        
        # Test multiview prompt
        views = ["front", "back"]
        view_section = create_view_images_section(views)
        
        multiview_prompt = format_multiview_scoring_prompt(
            part_context="This is a test context",
            rubric=EVALUATION_RUBRIC,
            session_id="TEST_SESSION",
            mesh_count=1,
            mesh_index=0,
            views_list=views,
            view_images_section=view_section
        )
        
        if "system_prompt" in multiview_prompt and "user_prompt" in multiview_prompt:
            print("✅ Multiview scoring prompt formatting works")
        else:
            print("❌ Multiview scoring prompt missing required fields")
            return False
        
        # Test pairwise prompt
        pairwise_prompt = format_pairwise_comparison_prompt(
            model_a_name="Model A",
            model_b_name="Model B"
        )
        
        if "system_prompt" in pairwise_prompt and "user_prompt" in pairwise_prompt:
            print("✅ Pairwise comparison prompt formatting works")
        else:
            print("❌ Pairwise comparison prompt missing required fields")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing prompt formatting: {e}")
        return False

def test_cache_system():
    """Test cache system functionality"""
    print("\n💾 Testing cache system...")
    
    try:
        from ..llm_cache import get_cache_manager, LLMCacheManager
        from ..llm_prompts import LLMProvider, EvaluationType
        
        # Create cache manager
        cache_manager = get_cache_manager()
        print("✅ Cache manager created")
        
        # Test cache key generation
        cache_key = cache_manager.generate_cache_key(
            evaluation_type=EvaluationType.DETAILED_SCORING,
            session_id="TEST_SESSION",
            provider=LLMProvider.CLAUDE,
            trial=0,
            mesh_count=1,
            mesh_index=0
        )
        
        if cache_key and len(cache_key) > 0:
            print("✅ Cache key generation works")
        else:
            print("❌ Cache key generation failed")
            return False
        
        # Test cache stats
        stats = cache_manager.get_cache_stats()
        if isinstance(stats, dict) and "total_files" in stats:
            print(f"✅ Cache stats: {stats['total_files']} files")
        else:
            print("❌ Cache stats failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing cache system: {e}")
        return False

def test_main_script_integration():
    """Test that main script can import the new system"""
    print("\n🔗 Testing main script integration...")
    
    try:
        # Try to import from the main script
        from .. import llm_leaderboard
        
        # Check if the new system is available in the main script
        if hasattr(llm_leaderboard, 'LLM_SYSTEM_AVAILABLE'):
            if llm_leaderboard.LLM_SYSTEM_AVAILABLE:
                print("✅ Main script has LLM system available")
            else:
                print("⚠️  Main script reports LLM system not available")
                return True  # This is still acceptable
        else:
            print("⚠️  Main script doesn't have LLM_SYSTEM_AVAILABLE attribute")
            return False
        
        # Check if unified call function exists
        if hasattr(llm_leaderboard, 'call_llm_unified'):
            print("✅ Unified LLM call function is available")
        else:
            print("❌ Unified LLM call function not found")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import main script: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing main script integration: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("🚀 Running LLM Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Provider Test", test_providers),
        ("Prompt Formatting Test", test_prompt_formatting),
        ("Cache System Test", test_cache_system),
        ("Main Script Integration Test", test_main_script_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  {test_name} failed!")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LLM integration system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 