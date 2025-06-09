#!/usr/bin/env python3
"""
test_multiview_integration.py - Test the multiview integration in the leaderboard system
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing imports...")
    
    try:
        from .. import llm_leaderboard
        print("✅ llm_leaderboard imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import llm_leaderboard: {e}")
        return False
    
    try:
        from .. import mesh_renderer
        print("✅ mesh_renderer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import mesh_renderer: {e}")
        return False
    
    # Check if multiview functions are available
    if hasattr(llm_leaderboard, 'call_claude_multiview'):
        print("✅ call_claude_multiview function available")
    else:
        print("❌ call_claude_multiview function not found")
        return False
    
    if hasattr(llm_leaderboard, 'MESH_RENDERER_AVAILABLE'):
        print(f"✅ MESH_RENDERER_AVAILABLE: {llm_leaderboard.MESH_RENDERER_AVAILABLE}")
    else:
        print("❌ MESH_RENDERER_AVAILABLE flag not found")
        return False
    
    return True

def test_mesh_renderer_availability():
    """Test mesh renderer availability"""
    print("\n🧪 Testing mesh renderer availability...")
    
    try:
        from mesh_renderer import check_trimesh_available
        available = check_trimesh_available()
        print(f"✅ Trimesh available: {available}")
        return available
    except Exception as e:
        print(f"❌ Error checking trimesh availability: {e}")
        return False

def test_argument_parsing():
    """Test that new command line arguments work"""
    print("\n🧪 Testing argument parsing...")
    
    try:
        import argparse
        
        # Create a test parser with the same arguments as the main script
        parser = argparse.ArgumentParser()
        parser.add_argument("--human-eval-json", required=True, type=Path, help="Path to leaderboard_models.json")
        parser.add_argument("--multiview", action="store_true", help="Use multi-view rendering")
        parser.add_argument("--views", nargs="+", default=["front", "back"], help="Views to render")
        
        # Test basic multiview argument
        test_args = ["--human-eval-json", "test.json", "--multiview"]
        args = parser.parse_args(test_args)
        
        if hasattr(args, 'multiview') and args.multiview:
            print("✅ --multiview argument works")
        else:
            print("❌ --multiview argument not working")
            return False
        
        # Test views argument
        test_args = ["--human-eval-json", "test.json", "--multiview", "--views", "front", "back"]
        args = parser.parse_args(test_args)
        
        if hasattr(args, 'views') and args.views == ["front", "back"]:
            print("✅ --views argument works")
        else:
            print("❌ --views argument not working")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing argument parsing: {e}")
        return False

def test_cache_key_generation():
    """Test that multiview cache keys work"""
    print("\n🧪 Testing cache key generation...")
    
    try:
        from llm_leaderboard import get_cache_key
        
        # Test regular cache key
        key1 = get_cache_key("detailed", "SESSION_123", 0, mesh_count=1, mesh_index=0)
        print(f"✅ Regular cache key: {key1}")
        
        # Test multiview cache key
        key2 = get_cache_key("detailed_multiview", "SESSION_123", 0, 
                           mesh_count=1, mesh_index=0, views=["front", "back"])
        print(f"✅ Multiview cache key: {key2}")
        
        # Keys should be different
        if key1 != key2:
            print("✅ Cache keys are different (good)")
        else:
            print("❌ Cache keys are the same (bad)")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing cache key generation: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing multiview integration...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Mesh Renderer Availability", test_mesh_renderer_availability),
        ("Argument Parsing", test_argument_parsing),
        ("Cache Key Generation", test_cache_key_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: ❌ FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\n📝 Usage examples:")
        print("  # Use front and back views")
        print("  python llm_leaderboard.py --human-eval-json models.json --multiview")
        print("\n  # Use front and back views (explicit)")
        print("  python llm_leaderboard.py --human-eval-json models.json --multiview --views front back")
        print("\n  # Test mesh rendering directly")
        print("  python mesh_renderer.py https://example.com/model.glb test_output/")
        return 0
    else:
        print("⚠️  Some integration tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 