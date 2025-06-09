#!/usr/bin/env python3
"""
test_mesh_renderer.py - Test script for mesh rendering functionality
"""

import sys
import pytest
from pathlib import Path
from ..mesh_renderer import (
    check_trimesh_available, 
    check_blender_available,
    render_mesh_views, 
    render_mesh_views_to_base64,
    demo_mesh_rendering,
    MeshRenderError
)

def test_basic_functionality():
    """Test basic mesh renderer functionality"""
    print("🧪 Testing mesh renderer basic functionality...")
    
    # Check if trimesh is available
    assert check_trimesh_available(), "trimesh is not available - install with: pip install trimesh[easy]"
    print("✅ trimesh is available")
    
    # Check if Blender is available
    assert check_blender_available(), "Blender is not available - install from https://www.blender.org/"
    print("✅ Blender is available")

def test_with_sample_url():
    """Test with a sample mesh URL (you'll need to provide a real one)"""
    
    # Example GLB URL - replace with a real one for testing
    sample_urls = [
        # You can add test URLs here
        # "https://example.com/sample.glb",
        "https://implicitshapefm.blob.core.windows.net/inference-outputs/tk@csm.ai/SESSION_1749187196_5282577_0/mesh.glb?se=2036-11-01T20%3A21%3A29Z&sp=r&sv=2023-11-03&sr=b&sig=hDpDcLOZIa9XvRVW8Ul9fwQtAY0WHSDcl9iWCrFrAa8%3D"
    ]
    
    if not sample_urls:
        print("⚠️  No sample URLs provided - add some to test_with_sample_url()")
        pytest.skip("No sample URLs provided for testing")
    
    for mesh_url in sample_urls:
        print(f"\n🌐 Testing with URL: {mesh_url}")
        
        # Test rendering
        views = ["front", "back"]
        rendered_views = render_mesh_views(mesh_url, views, resolution=256)
        
        assert len(rendered_views) == len(views), f"Expected {len(views)} views, got {len(rendered_views)}"
        print(f"✅ Successfully rendered {len(rendered_views)} views")
        
        # Test base64 conversion
        base64_views = render_mesh_views_to_base64(mesh_url, views, resolution=256)
        assert len(base64_views) == len(views), f"Expected {len(views)} base64 views, got {len(base64_views)}"
        print(f"✅ Successfully converted to base64 ({len(base64_views)} views)")
        
        # Verify all expected views are present
        for view in views:
            assert view in rendered_views, f"Missing rendered view: {view}"
            assert view in base64_views, f"Missing base64 view: {view}"

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\n🧪 Testing error handling...")
    
    # Test with invalid URL
    with pytest.raises(MeshRenderError):
        render_mesh_views("https://invalid-url-that-does-not-exist.com/fake.glb")
    print("✅ Correctly handled invalid URL")
    
    # Test with empty URL
    with pytest.raises(MeshRenderError):
        render_mesh_views("")
    print("✅ Correctly handled empty URL")
    
    # Test with invalid views
    with pytest.raises(MeshRenderError):
        # This will fail at the validation stage, which is good
        render_mesh_views("https://example.com/fake.glb", views=["invalid_view"])
    print("✅ Correctly handled invalid view name")

def main():
    """Run all tests"""
    print("🚀 Starting mesh renderer tests...\n")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Error Handling", test_error_handling),
        ("Sample URL", test_with_sample_url),
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
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 