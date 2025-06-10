#!/usr/bin/env python3
"""
Generate beautiful layouts showing original images alongside different model results.
Usage: python generate_layout.py <audit_file> [image_names...]
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
import requests
from io import BytesIO

def load_audit_data(audit_file):
    """Load audit data from JSON file."""
    with open(audit_file, 'r') as f:
        return json.load(f)

def get_available_fonts():
    """Get available system fonts, fallback to default."""
    try:
        # Try common system fonts
        for font_name in ['/System/Library/Fonts/Arial.ttf', 
                         '/System/Library/Fonts/Helvetica.ttc',
                         '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
            if os.path.exists(font_name):
                return font_name
    except:
        pass
    return None  # Will use default font

def download_image(url, size=(400, 400)):
    """Download an image from a URL and return PIL Image object."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize to standard size
        img = img.resize(size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def create_parts_composite(part_images, size=(400, 400)):
    """Create a composite image showing all parts in a grid."""
    if not part_images:
        return None
        
    num_parts = len(part_images)
    if num_parts == 1:
        return part_images[0].resize(size, Image.Resampling.LANCZOS)
    
    # Calculate grid layout for parts
    cols = min(3, num_parts)  # Max 3 columns for parts
    rows = (num_parts + cols - 1) // cols
    
    # Calculate size of each part image
    part_width = size[0] // cols
    part_height = size[1] // rows
    part_size = (part_width - 5, part_height - 5)  # Small padding
    
    # Create composite canvas
    composite = Image.new('RGB', size, color='#1a1a1a')
    
    for i, part_img in enumerate(part_images):
        row = i // cols
        col = i % cols
        
        # Resize part image
        resized_part = part_img.resize(part_size, Image.Resampling.LANCZOS)
        
        # Calculate position
        x = col * part_width + 2
        y = row * part_height + 2
        
        composite.paste(resized_part, (x, y))
    
    return composite

def get_model_image(model_data, model_name, size=(400, 400)):
    """Extract and download the rendered image for a model."""
    try:
        # Check if we have result_data with output
        if 'result_data' in model_data and 'output' in model_data['result_data']:
            output = model_data['result_data']['output']
            
            # Handle kit models with part_meshes (rendered outputs)
            if 'part_meshes' in output and len(output['part_meshes']) > 0:
                print(f"  Downloading {len(output['part_meshes'])} rendered part meshes for {model_name}...")
                part_images = []
                
                for i, part_mesh in enumerate(output['part_meshes']):
                    mesh_data = part_mesh.get('data', {})
                    image_url = mesh_data.get('image_url', '')
                    
                    if image_url:
                        part_img = download_image(image_url, (200, 200))  # Smaller for composite
                        if part_img:
                            part_images.append(part_img)
                            print(f"    ✓ Downloaded rendered part {i+1}/{len(output['part_meshes'])}")
                
                if part_images:
                    return create_parts_composite(part_images, size)
            
            # Fallback: Handle kit models with part_images (input images) if no part_meshes
            elif 'part_images' in output and len(output['part_images']) > 0:
                print(f"  Downloading {len(output['part_images'])} part input images for {model_name}...")
                part_images = []
                
                for i, part_info in enumerate(output['part_images']):
                    part_data = part_info.get('data', {})
                    image_url = part_data.get('image_url', '')
                    
                    if image_url:
                        part_img = download_image(image_url, (200, 200))  # Smaller for composite
                        if part_img:
                            part_images.append(part_img)
                            print(f"    ✓ Downloaded part input {i+1}/{len(output['part_images'])}")
                
                if part_images:
                    return create_parts_composite(part_images, size)
            
            # Look for rendered image URL in meshes (standard models)
            if 'meshes' in output and len(output['meshes']) > 0:
                mesh_data = output['meshes'][0].get('data', {})
                image_url = mesh_data.get('image_url', '')
                
                if image_url:
                    print(f"  Downloading rendered image for {model_name}...")
                    return download_image(image_url, size)
            
            # Fallback: check for segmented_image_url
            segmented_url = output.get('segmented_image_url', '')
            if segmented_url:
                print(f"  Using segmented image for {model_name}...")
                return download_image(segmented_url, size)
    
    except Exception as e:
        print(f"  Error extracting image for {model_name}: {e}")
    
    # Return None if no image found - will create placeholder
    return None

def create_model_placeholder(model_name, status, size=(400, 400)):
    """Create a placeholder image for a model result."""
    img = Image.new('RGB', size, color='#1a1a1a')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    font_path = get_available_fonts()
    try:
        title_font = ImageFont.truetype(font_path, 24) if font_path else ImageFont.load_default()
        status_font = ImageFont.truetype(font_path, 16) if font_path else ImageFont.load_default()
    except:
        title_font = ImageFont.load_default()
        status_font = ImageFont.load_default()
    
    # Draw model name
    title_bbox = draw.textbbox((0, 0), model_name, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    title_x = (size[0] - title_width) // 2
    title_y = size[1] // 2 - 30
    
    draw.text((title_x, title_y), model_name, fill='#ffffff', font=title_font)
    
    # Draw status
    status_text = f"Status: {status}"
    status_bbox = draw.textbbox((0, 0), status_text, font=status_font)
    status_width = status_bbox[2] - status_bbox[0]
    status_x = (size[0] - status_width) // 2
    status_y = title_y + title_height + 10
    
    # Color based on status (brighter colors for dark background)
    status_color = '#4CAF50' if status == 'complete' else '#F44336' if status == 'failed' else '#FFC107'
    draw.text((status_x, status_y), status_text, fill=status_color, font=status_font)
    
    # Add border
    draw.rectangle([0, 0, size[0]-1, size[1]-1], outline='#555555', width=2)
    
    return img

def create_beautiful_layout(image_name, audit_data, output_dir):
    """Create a beautiful layout for an image and its model results."""
    if image_name not in audit_data:
        print(f"Warning: {image_name} not found in audit data")
        return
    
    image_data = audit_data[image_name]
    
    # Load original image
    original_image_path = Path('full_eval_set') / f"{image_name}.png"
    if not original_image_path.exists():
        # Try jpg extension
        original_image_path = Path('full_eval_set') / f"{image_name}.jpg"
        if not original_image_path.exists():
            print(f"Warning: Original image not found for {image_name}")
            return
    
    try:
        original_img = Image.open(original_image_path)
        # Resize to standard size
        original_img = original_img.resize((400, 400), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error loading original image {original_image_path}: {e}")
        return
    
    # Get model results
    models = list(image_data.keys())
    print(f"Found {len(models)} models for {image_name}: {models}")
    
    # Calculate grid layout
    total_images = len(models) + 1  # +1 for original
    cols = min(4, total_images)  # Max 4 columns
    rows = (total_images + cols - 1) // cols
    
    # Image dimensions
    img_size = 400
    padding = 20
    label_height = 60
    
    # Calculate canvas size
    canvas_width = cols * img_size + (cols + 1) * padding
    canvas_height = rows * (img_size + label_height) + (rows + 1) * padding
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='#000000')
    draw = ImageDraw.Draw(canvas)
    
    # Get font for labels
    font_path = get_available_fonts()
    try:
        label_font = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
    except:
        label_font = ImageFont.load_default()
    
    # Place original image
    x = padding
    y = padding
    canvas.paste(original_img, (x, y))
    
    # Add label for original
    label_text = "Original"
    label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
    label_width = label_bbox[2] - label_bbox[0]
    label_x = x + (img_size - label_width) // 2
    label_y = y + img_size + 10
    draw.text((label_x, label_y), label_text, fill='#ffffff', font=label_font)
    
    # Place model results
    current_col = 1
    current_row = 0
    
    for model_name in models:
        if current_col >= cols:
            current_col = 0
            current_row += 1
        
        x = current_col * (img_size + padding) + padding
        y = current_row * (img_size + label_height + padding) + padding
        
        # Get model status and image
        model_data = image_data[model_name]
        status = model_data.get('status', 'unknown')
        
        # Try to get the actual rendered image
        model_img = get_model_image(model_data, model_name, (img_size, img_size))
        
        # If no image found, create placeholder
        if model_img is None:
            print(f"  No rendered image found for {model_name}, creating placeholder")
            model_img = create_model_placeholder(model_name, status, (img_size, img_size))
        
        canvas.paste(model_img, (x, y))
        
        # Add model name label
        display_name = model_name.replace('_', ' ').title()
        label_bbox = draw.textbbox((0, 0), display_name, font=label_font)
        label_width = label_bbox[2] - label_bbox[0]
        label_x = x + (img_size - label_width) // 2
        label_y = y + img_size + 10
        draw.text((label_x, label_y), display_name, fill='#ffffff', font=label_font)
        
        current_col += 1
    
    # Add title
    title = f"Model Comparison: {image_name}"
    try:
        title_font = ImageFont.truetype(font_path, 28) if font_path else ImageFont.load_default()
    except:
        title_font = ImageFont.load_default()
    
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (canvas_width - title_width) // 2
    draw.text((title_x, 10), title, fill='#ffffff', font=title_font)
    
    # Save the layout
    output_path = Path(output_dir) / f"{image_name}_layout.png"
    canvas.save(output_path)
    print(f"Layout saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate beautiful layouts from audit data')
    parser.add_argument('audit_file', help='Path to audit JSON file')
    parser.add_argument('images', nargs='*', help='Specific image names to process (default: all)')
    parser.add_argument('--output-dir', default='tmp', help='Output directory (default: tmp)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load audit data
    print(f"Loading audit data from {args.audit_file}...")
    audit_data = load_audit_data(args.audit_file)
    
    # Determine which images to process
    if args.images:
        images_to_process = args.images
    else:
        images_to_process = list(audit_data.keys())
        print(f"Found {len(images_to_process)} images in audit data")
    
    # Process each image
    for image_name in images_to_process:
        print(f"\nProcessing {image_name}...")
        try:
            create_beautiful_layout(image_name, audit_data, args.output_dir)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
    
    print(f"\nDone! Layouts saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 