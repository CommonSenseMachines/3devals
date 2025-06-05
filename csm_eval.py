#!/usr/bin/env python3
"""
CSM API Evaluation Script
This script runs multiple CSM API jobs for all images in the images directory.
Jobs are submitted and tracked locally - rerun the script to check progress.
"""

import os
import sys
import time
import json
import requests
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
def get_api_key():
    """Get API key from environment or config file"""
    # First try environment variable
    api_key = os.environ.get('CSM_API_KEY')
    if api_key:
        return api_key
    
    # Then try local config file
    config_file = '.csm_config'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                api_key = f.read().strip()
                if api_key and len(api_key) == 32:  # Basic validation
                    return api_key
        except Exception as e:
            logger.warning(f"Could not read config file: {e}")
    
    # If no API key found, exit with helpful message
    logger.error("CSM API key not found!")
    logger.error("Please run: ./run_eval.sh [command] to set up your API key")
    logger.error("Or set CSM_API_KEY environment variable")
    exit(1)

CSM_API_KEY = get_api_key()
CSM_API_BASE = "https://api.csm.ai/v3"
RESULTS_DIR = "results"
JOB_TRACKING_FILE = "job_tracking.json"

# Common settings
COMMON_SETTINGS = {
    "resolution": 200000
}

# Job configurations
JOB_CONFIGS = [
    {"name": "image_to_3d_base", "type": "image_to_3d", "geometry_model": "base", "texture_model": "none"},
    {"name": "image_to_3d_turbo", "type": "image_to_3d", "geometry_model": "turbo", "texture_model": "none"},
    {"name": "image_to_3d_turbo_baked", "type": "image_to_3d", "geometry_model": "turbo", "texture_model": "baked"},
    {"name": "image_to_3d_turbo_pbr", "type": "image_to_3d", "geometry_model": "turbo", "texture_model": "pbr"},
    {"name": "image_to_kit_pro_turbo_baked", "type": "image_to_kit", "decomposition_model": "pro", "geometry_model": "turbo", "texture_model": "baked"},
    {"name": "chat_to_3d_then_image_to_3d", "type": "chat_to_3d", "follow_up": "image_to_3d"}
]

class JobTracker:
    """Tracks submitted jobs to avoid resubmission"""
    
    def __init__(self, tracking_file: str = JOB_TRACKING_FILE):
        self.tracking_file = tracking_file
        self.jobs = self.load_tracking_data()
    
    def load_tracking_data(self) -> Dict[str, Any]:
        """Load existing job tracking data"""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load tracking file: {e}")
        return {}
    
    def save_tracking_data(self):
        """Save job tracking data"""
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(self.jobs, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save tracking file: {e}")
    
    def should_skip_job(self, image_name: str, job_name: str) -> bool:
        """Check if a job should be skipped (already submitted, completed, or failed)"""
        if image_name not in self.jobs or job_name not in self.jobs[image_name]:
            return False
        
        status = self.jobs[image_name][job_name].get('status')
        # Skip if job is already submitted, completed, or failed
        return status in ['submitted', 'complete', 'failed']
    
    def mark_job_submitted(self, image_name: str, job_name: str, session_id: str, session_data: Dict[str, Any]):
        """Mark a job as successfully submitted"""
        if image_name not in self.jobs:
            self.jobs[image_name] = {}
        
        self.jobs[image_name][job_name] = {
            'status': 'submitted',
            'session_id': session_id,
            'submitted_at': datetime.now().isoformat(),
            'session_data': session_data
        }
        self.save_tracking_data()
    
    def update_job_status(self, image_name: str, job_name: str, status: str, result_data: Optional[Dict[str, Any]] = None):
        """Update job status"""
        if image_name in self.jobs and job_name in self.jobs[image_name]:
            self.jobs[image_name][job_name]['status'] = status
            self.jobs[image_name][job_name]['updated_at'] = datetime.now().isoformat()
            if result_data:
                self.jobs[image_name][job_name]['result_data'] = result_data
            self.save_tracking_data()
    
    def get_submitted_jobs_summary(self) -> Dict[str, Any]:
        """Get summary of all submitted jobs"""
        summary = {
            'total_images': len(self.jobs),
            'job_counts': {'submitted': 0, 'complete': 0, 'failed': 0, 'incomplete': 0},
            'images': {}
        }
        
        for image_name, jobs in self.jobs.items():
            image_summary = {'jobs': {}}
            for job_name, job_data in jobs.items():
                status = job_data.get('status', 'unknown')
                summary['job_counts'][status] = summary['job_counts'].get(status, 0) + 1
                image_summary['jobs'][job_name] = {
                    'status': status,
                    'session_id': job_data.get('session_id'),
                    'submitted_at': job_data.get('submitted_at')
                }
            summary['images'][image_name] = image_summary
        
        return summary

class CSMAPIClient:
    """Client for interacting with CSM.AI API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json'
        }
    
    def upload_image(self, image_path: str) -> str:
        """Upload an image and return the image URL or base64 data"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                # Return data URL format
                ext = Path(image_path).suffix.lower()
                mime_type = f"image/{ext[1:]}" if ext in ['.jpg', '.jpeg', '.png', '.gif'] else "image/png"
                return f"data:{mime_type};base64,{base64_data}"
        except Exception as e:
            logger.error(f"Failed to upload image {image_path}: {e}")
            raise
    
    def create_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new session"""
        try:
            response = requests.post(
                f"{CSM_API_BASE}/sessions/",
                headers=self.headers,
                json=session_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get session status and results"""
        try:
            response = requests.get(
                f"{CSM_API_BASE}/sessions/{session_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get session status for {session_id}: {e}")
            raise
    
    def submit_image_to_3d(self, image_data: str, geometry_model: str = 'base', 
                          texture_model: str = 'none') -> Dict[str, Any]:
        """Submit Image-to-3D job (doesn't wait for completion)"""
        session_data = {
            "type": "image_to_3d",
            "input": {
                "image": image_data,
                "model": "sculpt",
                "settings": {
                    "geometry_model": geometry_model,
                    "texture_model": texture_model,
                    **COMMON_SETTINGS
                }
            }
        }
        
        logger.info(f"Submitting Image-to-3D job: geometry_model={geometry_model}, texture_model={texture_model}")
        return self.create_session(session_data)
    
    def submit_image_to_kit(self, image_data: str, decomposition_model: str = 'pro',
                           geometry_model: str = 'turbo', texture_model: str = 'baked') -> Dict[str, Any]:
        """Submit Image-to-Kit job (doesn't wait for completion)"""
        session_data = {
            "type": "image_to_kit",
            "input": {
                "image": image_data,
                "model": "sculpt",
                "decomposition_model": decomposition_model,
                "settings": {
                    "geometry_model": geometry_model,
                    "texture_model": texture_model,
                    **COMMON_SETTINGS
                }
            }
        }
        
        logger.info(f"Submitting Image-to-Kit job: decomposition_model={decomposition_model}, geometry_model={geometry_model}, texture_model={texture_model}")
        return self.create_session(session_data)
    
    def submit_chat_to_3d(self, image_data: str, prompt: str) -> Dict[str, Any]:
        """Submit Chat-to-3D job (doesn't wait for completion)"""
        session_data = {
            "type": "chat_to_3d",
            "messages": [
                {
                    "type": "user_prompt",
                    "message": prompt,
                    "images": [image_data]
                }
            ]
        }
        
        logger.info(f"Submitting Chat-to-3D job with prompt: {prompt}")
        return self.create_session(session_data)

def check_job_progress(client: CSMAPIClient, tracker: JobTracker):
    """Check progress of submitted jobs and update their status"""
    logger.info("Checking progress of submitted jobs...")
    
    for image_name, jobs in tracker.jobs.items():
        for job_name, job_data in jobs.items():
            if job_data.get('status') == 'submitted':
                session_id = job_data.get('session_id')
                if session_id:
                    try:
                        session_status = client.get_session_status(session_id)
                        status = session_status.get('status', 'incomplete')
                        
                        if status == 'complete':
                            logger.info(f"Job {job_name} for {image_name} completed!")
                            tracker.update_job_status(image_name, job_name, 'complete', session_status)
                        elif status == 'failed':
                            logger.warning(f"Job {job_name} for {image_name} failed")
                            tracker.update_job_status(image_name, job_name, 'failed', session_status)
                        else:
                            logger.info(f"Job {job_name} for {image_name} still in progress (status: {status})")
                    
                    except Exception as e:
                        logger.error(f"Error checking job {job_name} for {image_name}: {e}")

def submit_jobs_for_image(client: CSMAPIClient, tracker: JobTracker, image_path: str) -> Dict[str, Any]:
    """Submit all jobs for a single image"""
    image_name = Path(image_path).stem
    logger.info(f"Processing image: {image_name}")
    
    # Check if all jobs for this image are already done
    all_jobs_done = True
    for job_config in JOB_CONFIGS:
        job_name = job_config["name"]
        if not tracker.should_skip_job(image_name, job_name):
            all_jobs_done = False
            break
    
    if all_jobs_done:
        logger.info(f"All jobs for {image_name} already submitted/completed - skipping")
        return {
            "image_name": image_name,
            "image_path": image_path,
            "jobs": {},
            "note": "All jobs already submitted/completed"
        }
    
    # Upload image
    image_data = client.upload_image(image_path)
    
    results = {
        "image_name": image_name,
        "image_path": image_path,
        "jobs": {}
    }
    
    # Process each job configuration
    for job_config in JOB_CONFIGS:
        job_name = job_config["name"]
        
        # Skip if already submitted/completed/failed
        if tracker.should_skip_job(image_name, job_name):
            job_status = tracker.jobs[image_name][job_name].get('status', 'unknown')
            logger.info(f"Skipping {job_name} for {image_name} - status: {job_status}")
            continue
        
        try:
            if job_config["type"] == "image_to_3d":
                session = client.submit_image_to_3d(
                    image_data,
                    geometry_model=job_config["geometry_model"],
                    texture_model=job_config["texture_model"]
                )
                tracker.mark_job_submitted(image_name, job_name, session['_id'], session)
                results["jobs"][job_name] = {"session_id": session['_id'], "status": "submitted"}
                
            elif job_config["type"] == "image_to_kit":
                session = client.submit_image_to_kit(
                    image_data,
                    decomposition_model=job_config["decomposition_model"],
                    geometry_model=job_config["geometry_model"],
                    texture_model=job_config["texture_model"]
                )
                tracker.mark_job_submitted(image_name, job_name, session['_id'], session)
                results["jobs"][job_name] = {"session_id": session['_id'], "status": "submitted"}
                
            elif job_config["type"] == "chat_to_3d":
                # For chat-to-3d, we submit the chat job first
                chat_prompt = "generate a better pose for image to 3D, preserve all details in the original image"
                session = client.submit_chat_to_3d(image_data, chat_prompt)
                tracker.mark_job_submitted(image_name, job_name, session['_id'], session)
                results["jobs"][job_name] = {
                    "chat_session_id": session['_id'], 
                    "status": "submitted_chat_phase",
                    "note": "Need to check chat completion before submitting follow-up image-to-3d"
                }
            
            logger.info(f"Successfully submitted {job_name} for {image_name}")
            
        except Exception as e:
            logger.error(f"Failed to submit {job_name} for {image_name}: {e}")
            results["jobs"][job_name] = {"error": str(e), "status": "failed_to_submit"}
    
    return results

def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

def main():
    """Main evaluation function"""
    logger.info("Starting CSM API evaluation")
    
    # Check if this is a progress-only run
    progress_only = len(sys.argv) > 1 and sys.argv[1] == '--progress-only'
    
    # Initialize API client and job tracker
    client = CSMAPIClient(CSM_API_KEY)
    tracker = JobTracker()
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Always check progress of existing jobs first
    if tracker.jobs:
        check_job_progress(client, tracker)
    
    # If this is progress-only mode, skip job submission
    if progress_only:
        logger.info("Progress-only mode - skipping job submission")
    else:
        # Get all images from the images directory
        images_dir = Path("images")
        if not images_dir.exists():
            logger.error("Images directory does not exist")
            return
        
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
        
        if not image_files:
            logger.error("No image files found in images directory")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        all_results = {}
        
        for image_file in image_files:
            try:
                image_results = submit_jobs_for_image(client, tracker, str(image_file))
                all_results[image_file.stem] = image_results
                
                # Save individual results only if there were new submissions
                if image_results.get("jobs"):
                    individual_results_path = os.path.join(RESULTS_DIR, f"{image_file.stem}_submission_results.json")
                    save_results(image_results, individual_results_path)
                
            except Exception as e:
                logger.error(f"Failed to process image {image_file}: {e}")
                all_results[image_file.stem] = {"error": str(e)}
        
        # Save combined results only if there were new submissions
        if any(result.get("jobs") for result in all_results.values()):
            combined_results_path = os.path.join(RESULTS_DIR, "combined_submission_results.json")
            save_results(all_results, combined_results_path)
    
    # Always generate and save summary
    summary = tracker.get_submitted_jobs_summary()
    logger.info(f"Job submission summary: {summary['job_counts']}")
    
    # Save tracking summary
    summary_path = os.path.join(RESULTS_DIR, "job_summary.json")
    save_results(summary, summary_path)
    
    if progress_only:
        logger.info("Progress check completed.")
    else:
        logger.info("CSM API evaluation completed. Run with --progress-only to check job progress.")
    logger.info(f"Track job progress in: {JOB_TRACKING_FILE}")

if __name__ == "__main__":
    main()
