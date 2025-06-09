#!/usr/bin/env python3
"""
mesh_renderer.py - 3D mesh rendering utilities for multi-view evaluation

Provides functions to download 3D meshes and render them from multiple viewpoints
using trimesh for comprehensive 3D model evaluation.
"""

import base64
import tempfile
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import BytesIO
import subprocess
import shutil
import platform

# Initialize availability flags
TRIMESH_AVAILABLE = False
BLENDER_AVAILABLE = False
BLENDER_PATH = None

try:
    import trimesh
    from PIL import Image
    TRIMESH_AVAILABLE = True
        
except ImportError:
    print("⚠️  trimesh not available. Install with: pip install trimesh[easy]")

# Check for Blender across different platforms
def find_blender_executable():
    """Find Blender executable across different platforms"""
    system = platform.system()
    
    # List of possible Blender paths for different platforms
    possible_paths = []
    
    if system == "Darwin":  # macOS
        possible_paths = [
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.4/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.3/Blender.app/Contents/MacOS/Blender", 
            "/Applications/Blender 4.2/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.1/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.0/Blender.app/Contents/MacOS/Blender",
            "/opt/homebrew/bin/blender",
            "/usr/local/bin/blender"
        ]
    elif system == "Linux":
        possible_paths = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/opt/blender/blender",
            "/snap/bin/blender",
            "~/Applications/blender/blender"
        ]
    elif system == "Windows":
        possible_paths = [
            "C:\\Program Files\\Blender Foundation\\Blender 4.4\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.3\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.1\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe",
        ]
    
    # Test each path
    for path in possible_paths:
        if path.startswith("~"):
            path = Path(path).expanduser()
        
        if Path(path).exists():
            try:
                # Test if blender can run and has working Python
                result = subprocess.run([path, "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    # Additional test to ensure Python modules work
                    test_result = subprocess.run([
                        path, "--background", "--factory-startup",
                        "--python-expr", "import bpy; print('OK')"
                    ], capture_output=True, text=True, timeout=15)
                    
                    # If Python test passes or at least doesn't fail with module errors
                    if (test_result.returncode == 0 or 
                        'ModuleNotFoundError' not in test_result.stderr):
                        return path
            except Exception:
                continue
    
    # Also try system PATH as fallback (after testing specific paths)
    system_blender = shutil.which("blender")
    if system_blender and system_blender not in possible_paths:
        try:
            result = subprocess.run([system_blender, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return system_blender
        except Exception:
            pass
    
    return None

try:
    BLENDER_PATH = find_blender_executable()
    if BLENDER_PATH:
        BLENDER_AVAILABLE = True
        print(f"✅ Found Blender at: {BLENDER_PATH}")
    else:
        print("⚠️  Blender not found. Install from https://www.blender.org/")
except Exception as e:
    print(f"⚠️  Error checking Blender: {e}")


class MeshRenderError(Exception):
    """Custom exception for mesh rendering errors"""
    pass


def check_trimesh_available() -> bool:
    """Check if trimesh is available for rendering"""
    return TRIMESH_AVAILABLE


def check_blender_available() -> bool:
    """Check if Blender is available for high-quality rendering"""
    return BLENDER_AVAILABLE


def download_mesh_file(mesh_url: str, timeout: int = 30) -> str:
    """
    Download mesh file from URL to temporary location.
    
    Args:
        mesh_url: URL to 3D mesh file (GLB, OBJ, etc.)
        timeout: Request timeout in seconds
        
    Returns:
        Path to temporary mesh file
        
    Raises:
        MeshRenderError: If download fails
    """
    if not mesh_url:
        raise MeshRenderError("Mesh URL is required")
    
    try:
        response = requests.get(mesh_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise MeshRenderError(f"Failed to download mesh from {mesh_url}: {e}")
    
    # Create temporary file with appropriate extension
    mesh_suffix = Path(mesh_url.split('?')[0]).suffix  # Remove query params
    if not mesh_suffix:
        mesh_suffix = '.glb'  # Default to GLB
    
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=mesh_suffix, delete=False)
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    except IOError as e:
        raise MeshRenderError(f"Failed to save mesh file: {e}")


def load_mesh_scene(mesh_path: str) -> 'trimesh.Scene':
    """
    Load mesh file into trimesh Scene.
    
    Args:
        mesh_path: Path to mesh file
        
    Returns:
        trimesh.Scene object
        
    Raises:
        MeshRenderError: If mesh loading fails
    """
    if not TRIMESH_AVAILABLE:
        raise MeshRenderError("trimesh is not available. Install with: pip install trimesh[easy]")
    
    try:
        mesh = trimesh.load(mesh_path)
        
        # Convert to Scene if it's a single mesh
        if hasattr(mesh, 'vertices'):
            scene = trimesh.Scene([mesh])
        else:
            scene = mesh
            
        if len(scene.geometry) == 0:
            raise MeshRenderError("Loaded mesh contains no geometry")
            
        return scene
        
    except Exception as e:
        raise MeshRenderError(f"Failed to load mesh from {mesh_path}: {e}")


def get_camera_positions(scene: 'trimesh.Scene', distance_multiplier: float = 2.5) -> Tuple[Dict[str, List[float]], List[float]]:
    """
    Calculate camera positions for different views around the mesh.
    
    Args:
        scene: trimesh.Scene object
        distance_multiplier: How far to place camera (multiplier of object size)
        
    Returns:
        Tuple of (Dict mapping view names to camera positions [x, y, z], center point [x, y, z])
    """
    # Use scene bounds for reliable bounding box calculation
    # This is simpler and more reliable than manual vertex transformation
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2
    dimensions = bounds[1] - bounds[0]
    
    # Use the maximum dimension for camera distance calculation
    max_dimension = np.max(dimensions)
    
    # For better framing, use the diagonal of the bounding box
    diagonal = np.sqrt(np.sum(dimensions**2))
    
    # Use diagonal for camera distance to ensure full object is visible from any angle
    camera_distance = diagonal * distance_multiplier * 1.0  # Use full diagonal for better visibility
    
    # Ensure minimum distance for very small objects
    if camera_distance < 2.0:
        camera_distance = 2.0
    
    # For 45-degree views, the object appears larger due to diagonal viewing
    # Use a much larger distance to ensure the object fits in frame
    diagonal_distance_multiplier = 2.0  # 100% further for better framing
    diagonal_component = (camera_distance * diagonal_distance_multiplier) / np.sqrt(2)
    
    # Simplified approach: camera positions will be calculated in Blender based on actual object dimensions
    # Just pass the view name to Blender and let it calculate proper positions
    origin = [0, 0, 0]  # Object will be centered here in Blender
    
    # Dummy positions - actual positions will be calculated in Blender script
    view_positions = {
        "front": [0, -1, 0],
        "back": [0, 1, 0],
    }
    
    print(f"🎯 Object center: {center}")
    print(f"📏 Object dimensions: {dimensions}")
    print(f"📐 Bounding box diagonal: {diagonal:.3f}")
    print(f"📷 Camera distance (orthogonal): {camera_distance:.3f}")
    print(f"📝 Note: Actual camera positions will be calculated in Blender based on object bbox")
    
    # Return origin as the look_at target since object will be centered there in Blender
    return view_positions, origin


def render_scene_view(scene: 'trimesh.Scene', camera_pos: List[float], 
                     look_at: List[float], resolution: int = 512, view_name: str = "unknown") -> Image.Image:
    """
    Render scene from specific camera position using Blender.
    
    Args:
        scene: trimesh.Scene to render
        camera_pos: Camera position [x, y, z]
        look_at: Point camera should look at [x, y, z]
        resolution: Image resolution (square)
        
    Returns:
        PIL Image of the rendered view
        
    Raises:
        MeshRenderError: If rendering fails
    """
    if not TRIMESH_AVAILABLE:
        raise MeshRenderError("trimesh is not available")
    
    if not BLENDER_AVAILABLE:
        raise MeshRenderError("Blender is not available. Install from https://www.blender.org/")
    
    # Use Blender for high-quality rendering
    try:
        return _render_with_blender(scene, camera_pos, look_at, resolution, view_name)
    except Exception as e:
        raise MeshRenderError(f"Failed to render view from {camera_pos}: {e}")


def _render_with_blender(scene: 'trimesh.Scene', camera_pos: List[float], 
                        look_at: List[float], resolution: int = 512, view_name: str = "unknown") -> Image.Image:
    """Render using headless Blender"""
    
    # Save scene to temporary file
    temp_mesh_file = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
    temp_mesh_file.close()
    
    # Export scene to GLB format
    scene.export(temp_mesh_file.name)
    
    # Create temporary output image file
    temp_output_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_output_file.close()
    
    # Create Blender Python script
    blender_script = f'''
import bpy
import sys
import os
import bmesh
import math

try:
    from mathutils import Vector
except ImportError:
    # Handle missing mathutils - create a simple replacement
    class Vector:
        def __init__(self, coords):
            self.x, self.y, self.z = coords
        def __iter__(self):
            return iter([self.x, self.y, self.z])
        def __getitem__(self, i):
            return [self.x, self.y, self.z][i]

# Clear existing mesh objects
try:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
except:
    # If selection fails, try to delete default objects
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

# Import the mesh
try:
    bpy.ops.import_scene.gltf(filepath="{temp_mesh_file.name}")
    print("Successfully imported GLB file")
except Exception as e:
    print(f"Failed to import GLB: {{e}}")
    sys.exit(1)

# Get all mesh objects and center them
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
print(f"Found {{len(mesh_objects)}} mesh objects")

if not mesh_objects:
    print("No mesh objects found after import!")
    sys.exit(1)

# Calculate the actual bounding box of all meshes (don't try to center)
all_coords = []
for obj in mesh_objects:
    # Get world space coordinates
    matrix_world = obj.matrix_world
    for vertex in obj.data.vertices:
        world_coord = matrix_world @ vertex.co
        all_coords.append(world_coord)

if all_coords:
    # Calculate bounds
    min_x = min(coord.x for coord in all_coords)
    max_x = max(coord.x for coord in all_coords)
    min_y = min(coord.y for coord in all_coords)
    max_y = max(coord.y for coord in all_coords)
    min_z = min(coord.z for coord in all_coords)
    max_z = max(coord.z for coord in all_coords)
    
    # Calculate the actual center of the imported object
    actual_center_x = (min_x + max_x) / 2
    actual_center_y = (min_y + max_y) / 2
    actual_center_z = (min_z + max_z) / 2
    
    # Center the object at origin by moving it
    for obj in mesh_objects:
        obj.location.x -= actual_center_x
        obj.location.y -= actual_center_y
        obj.location.z -= actual_center_z
    
    # Skip rotation for now to test basic functionality
    
    # Update scene and verify centering
    bpy.context.view_layer.update()
    
    # Verify the object is actually centered now
    all_coords_after = []
    for obj in mesh_objects:
        for vertex in obj.data.vertices:
            world_coord = obj.matrix_world @ vertex.co
            all_coords_after.append(world_coord)
    
    if all_coords_after:
        min_x = min(coord.x for coord in all_coords_after)
        max_x = max(coord.x for coord in all_coords_after)
        min_y = min(coord.y for coord in all_coords_after)
        max_y = max(coord.y for coord in all_coords_after)
        min_z = min(coord.z for coord in all_coords_after)
        max_z = max(coord.z for coord in all_coords_after)
    else:
        min_x = max_x = min_y = max_y = min_z = max_z = 0
    
    new_center_x = (min_x + max_x) / 2
    new_center_y = (min_y + max_y) / 2
    new_center_z = (min_z + max_z) / 2
    
    # Object is now centered at origin
    target_location = [0, 0, 0]
    
    # Calculate camera distance based on actual Blender object dimensions
    blender_dimensions = [max_x-min_x, max_y-min_y, max_z-min_z]
    blender_diagonal = math.sqrt(sum(d*d for d in blender_dimensions))
    camera_distance = blender_diagonal * 2.0  # Use larger multiplier for better framing
    
    # Position camera based on view type using actual Blender space
    if "{view_name}" == "front":
        camera_location = [0, -camera_distance, 0]
    elif "{view_name}" == "back":
        camera_location = [0, camera_distance, 0]
    else:
        # Default front view
        camera_location = [0, -camera_distance, 0]
else:
    print("⚠️  No coordinates found!")
    target_location = [0, 0, 0]
    camera_location = [{camera_pos[0]}, {camera_pos[1]}, {camera_pos[2]}]

# Set up camera with adjusted positioning
# camera_location and target_location are set above based on actual object center

# Create camera
try:
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
except Exception as e:
    print(f"❌ Failed to create camera: {{e}}")
    sys.exit(1)

# Point camera at origin
dx = target_location[0] - camera_location[0]
dy = target_location[1] - camera_location[1] 
dz = target_location[2] - camera_location[2]

# Calculate rotation angles to look at target
rot_x = math.atan2(-dz, math.sqrt(dx*dx + dy*dy))
rot_z = math.atan2(dx, dy)
camera.rotation_euler = (rot_x + math.pi/2, 0, rot_z)

# Set camera as active
bpy.context.scene.camera = camera

# Adjust camera settings for better framing
camera.data.lens = 50  # 50mm focal length for natural perspective
camera.data.clip_start = 0.1
camera.data.clip_end = 1000.0
# Set render settings
bpy.context.scene.render.resolution_x = {resolution}
bpy.context.scene.render.resolution_y = {resolution}
bpy.context.scene.render.filepath = "{temp_output_file.name}"
bpy.context.scene.render.image_settings.file_format = 'PNG'

# Use EEVEE for better quality (fallback to Workbench if needed)
try:
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    # Enable viewport shading for better visibility
    bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.eevee.use_ssr_refraction = True
except:
    try:
        bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
        bpy.context.scene.display.shading.type = 'MATERIAL'
    except:
        pass  # Use default engine

# Add multiple lights for better illumination
try:
    # Key light (main)
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 8))
    key_light = bpy.context.object
    if hasattr(key_light.data, 'energy'):
        key_light.data.energy = 4.0
    
    # Fill light (softer, opposite side)
    bpy.ops.object.light_add(type='SUN', location=(-3, 3, 5))
    fill_light = bpy.context.object
    if hasattr(fill_light.data, 'energy'):
        fill_light.data.energy = 2.0
    
except Exception as e:
    print(f"Lighting setup failed: {{e}}")

# Add a simple material to mesh objects if they don't have one
try:
    for obj in mesh_objects:
        if not obj.data.materials:
            # Create a simple material
            mat = bpy.data.materials.new(name="DefaultMaterial")
            mat.use_nodes = True
            # Set a nice light blue color
            if mat.node_tree:
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs[0].default_value = (0.8, 0.8, 0.9, 1.0)  # Light blue
                    bsdf.inputs[7].default_value = 0.2  # Some roughness
            obj.data.materials.append(mat)
except Exception as e:
    print(f"Material setup failed: {{e}}")

# Render
try:
    bpy.ops.render.render(write_still=True)
    
    # Verify the output file exists and has content
    if os.path.exists("{temp_output_file.name}"):
        file_size = os.path.getsize("{temp_output_file.name}")
        if file_size == 0:
            print("⚠️  Output file is empty!")
            sys.exit(1)
    else:
        print("❌ Output file was not created!")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Render failed: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    # Write script to temporary file
    script_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    script_file.write(blender_script)
    script_file.close()
    
    try:
                # Run Blender headless using the detected path
        if not BLENDER_PATH:
            raise RuntimeError("Blender executable not found")
            
        cmd = [
            BLENDER_PATH,
            "--background",
            "--factory-startup",  # Start with default settings
            "--python", script_file.name
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120  # Increased timeout for rendering
        )
        
        # Only show debug output if render failed
        if result.returncode != 0:
            print(f"🔍 Blender execution debug (FAILED):")
            print(f"   Return code: {result.returncode}")
            if result.stdout:
                print(f"   STDOUT:")
                for line in result.stdout.split('\n')[-10:]:  # Show last 10 lines only
                    if line.strip():
                        print(f"     {line}")
            if result.stderr:
                print(f"   STDERR:")
                for line in result.stderr.split('\n')[-5:]:  # Show last 5 lines only
                    if line.strip():
                        print(f"     {line}")
        
        # Filter out common non-critical warnings
        stderr_lines = result.stderr.split('\n') if result.stderr else []
        critical_errors = []
        
        for line in stderr_lines:
            # Skip font warnings and other non-critical messages
            if any(skip_pattern in line.lower() for skip_pattern in [
                'blf_load_font_default',
                'font data directory',
                'fonts\' data path',
                'will not be able to display text'
            ]):
                continue
            if line.strip():  # Only keep non-empty lines
                critical_errors.append(line)
        
        # Check for critical errors
        if result.returncode != 0 and critical_errors:
            # Look for actual Python errors
            python_errors = [line for line in critical_errors 
                           if 'ModuleNotFoundError' in line or 'Error' in line or 'Exception' in line]
            if python_errors:
                raise RuntimeError(f"Blender failed with errors: {'; '.join(python_errors)}")
            elif critical_errors:
                raise RuntimeError(f"Blender failed: {'; '.join(critical_errors[:3])}")  # Show first 3 errors
        
        # Load rendered image
        output_path = Path(temp_output_file.name)
        
        if not output_path.exists():
            raise RuntimeError("Blender did not produce output image")
        
        # Check file size
        file_size = output_path.stat().st_size
        
        if file_size == 0:
            raise RuntimeError("Blender produced empty output image")
            
        try:
            rendered_image = Image.open(temp_output_file.name).convert("RGB")
        except Exception as img_error:
            raise RuntimeError(f"Failed to load rendered image: {img_error}")
        
        return rendered_image
        
    finally:
        # Clean up temporary files
        for temp_file in [temp_mesh_file.name, temp_output_file.name, script_file.name]:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors


def render_mesh_views(mesh_url: str, views: List[str] = ["front", "back"], 
                     resolution: int = 512, distance_multiplier: float = 1.5,
                     timeout: int = 30) -> Dict[str, Image.Image]:
    """
    Download and render 3D mesh from multiple viewpoints.
    
    Args:
        mesh_url: URL to 3D mesh file (GLB, OBJ, etc.)
        views: List of view names to render ("front", "back")
        resolution: Image resolution for renders (square)
        distance_multiplier: Camera distance as multiple of object size
        timeout: Download timeout in seconds
        
    Returns:
        Dict mapping view names to PIL Image objects
        
    Raises:
        MeshRenderError: If any step fails
    """
    temp_mesh_path = None
    
    try:
        # Download mesh
        temp_mesh_path = download_mesh_file(mesh_url, timeout)
        
        # Load mesh
        scene = load_mesh_scene(temp_mesh_path)
        
        # Get camera positions
        view_positions, center = get_camera_positions(scene, distance_multiplier)
        
        # Validate requested views
        valid_views = set(view_positions.keys())
        invalid_views = set(views) - valid_views
        if invalid_views:
            raise MeshRenderError(f"Invalid view names: {invalid_views}. Valid views: {valid_views}")
        
        # Render each view
        rendered_views = {}
        for view_name in views:
            camera_pos = view_positions[view_name]
            rendered_views[view_name] = render_scene_view(scene, camera_pos, center, resolution, view_name)
        
        return rendered_views
        
    finally:
        # Clean up temporary file
        if temp_mesh_path:
            try:
                Path(temp_mesh_path).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors


def render_mesh_views_to_base64(mesh_url: str, views: List[str] = ["front", "back"], 
                               resolution: int = 512, **kwargs) -> Dict[str, str]:
    """
    Render mesh views and return as base64 encoded strings.
    
    Args:
        mesh_url: URL to 3D mesh file
        views: List of view names to render
        resolution: Image resolution
        **kwargs: Additional arguments passed to render_mesh_views
        
    Returns:
        Dict mapping view names to base64 encoded JPEG strings
        
    Raises:
        MeshRenderError: If rendering fails
    """
    rendered_views = render_mesh_views(mesh_url, views, resolution, **kwargs)
    
    base64_views = {}
    for view_name, img in rendered_views.items():
        # Convert PIL Image to base64 JPEG
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        base64_views[view_name] = img_base64
    
    return base64_views


def demo_mesh_rendering(mesh_url: str, output_dir: str = "test_renders", 
                       views: List[str] = ["front", "back"], resolution: int = 512) -> bool:
    """
    Demo mesh rendering and save results to files.
    
    Args:
        mesh_url: URL to test mesh
        output_dir: Directory to save test images
        views: Views to render
        resolution: Image resolution
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🧪 Testing mesh rendering with URL: {mesh_url}")
        print(f"📁 Output directory: {output_dir}")
        print(f"👁️  Views to render: {views}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Render views
        rendered_views = render_mesh_views(mesh_url, views, resolution)
        
        # Save each view
        for view_name, img in rendered_views.items():
            output_file = output_path / f"test_render_{view_name}.jpg"
            img.save(output_file, "JPEG", quality=90)
            print(f"✅ Saved {view_name} view: {output_file}")
        
        print(f"🎉 Successfully rendered {len(rendered_views)} views!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    """Simple test when run directly"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mesh_renderer.py <mesh_url> [output_dir]")
        print("Example: python mesh_renderer.py https://example.com/model.glb test_output/")
        sys.exit(1)
    
    mesh_url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "test_renders"
    
    # Test with front and back views
    success = demo_mesh_rendering(mesh_url, output_dir, ["front", "back"])
    sys.exit(0 if success else 1) 