#!/usr/bin/env python3
"""
CSM API Evaluation Script.
This script runs multiple CSM API jobs for all images in the images directory.
Jobs are submitted and tracked locally - rerun the script to check progress.
"""

import os
import sys
import time
import json
import requests
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from tqdm import tqdm

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
    # {"name": "chat_to_3d_then_image_to_3d", "type": "chat_to_3d", "follow_up": "image_to_3d"},  # DISABLED: Safety system issues
    {"name": "image_to_3d_250k", "type": "image_to_3d", "geometry_model": "base", "texture_model": "none", "resolution": 250000}
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
        """Check if a job should be skipped (already submitted, completed, or failed without retry)"""
        if image_name not in self.jobs or job_name not in self.jobs[image_name]:
            return False
        
        status = self.jobs[image_name][job_name].get('status')
        
        # Skip if job is complete or currently submitted
        if status in ['submitted', 'complete']:
            return True
            
        # Don't skip if job is in retrying status (let it retry)
        if status == 'retrying':
            return False
            
        # Don't skip failed jobs that should be retried
        if status == 'failed':
            return not self.should_retry_failed_job(image_name, job_name)
            
        return False
    
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
    
    def mark_job_failed(self, image_name: str, job_name: str, error_message: str):
        """Mark a job as failed during submission"""
        if image_name not in self.jobs:
            self.jobs[image_name] = {}
        
        # Get existing retry count if this job was retried before
        existing_job = self.jobs[image_name].get(job_name, {})
        retry_count = existing_job.get('retry_count', 0)
        
        self.jobs[image_name][job_name] = {
            'status': 'failed',
            'failed_at': datetime.now().isoformat(),
            'error': error_message,
            'retry_count': retry_count
        }
        self.save_tracking_data()
    
    def should_retry_failed_job(self, image_name: str, job_name: str, max_retries: int = 3) -> bool:
        """Check if a failed job should be retried"""
        if image_name not in self.jobs or job_name not in self.jobs[image_name]:
            return False
            
        job_data = self.jobs[image_name][job_name]
        status = job_data.get('status')
        
        if status != 'failed':
            return False
            
        retry_count = job_data.get('retry_count', 0)
        error_message = job_data.get('error', '').lower()
        
        # Don't retry if we've exceeded max retries
        if retry_count >= max_retries:
            return False
        
        # Retry certain types of errors that might be temporary
        retryable_errors = [
            'rate limit',
            'timeout',
            '429',  # Too Many Requests
            '500',  # Internal Server Error
            '502',  # Bad Gateway
            '503',  # Service Unavailable
            '504',  # Gateway Timeout
            'connection',
            'temporary'
        ]
        
        # For 400 errors, be more selective (could be rate limiting in disguise)
        if '400' in error_message:
            # CSM API sometimes returns 400 for rate limits without clear keywords
            # Be more generous with 400 retries for this API
            if any(keyword in error_message for keyword in ['rate', 'limit', 'quota', 'too many', 'bad request']):
                return True
            # For other 400s, still retry once in case it's a transient issue
            if retry_count == 0:
                return True
            return False
        
        # Retry if error contains any retryable keywords
        return any(keyword in error_message for keyword in retryable_errors)
    
    def mark_job_retry(self, image_name: str, job_name: str):
        """Mark that we're retrying a failed job"""
        if image_name in self.jobs and job_name in self.jobs[image_name]:
            job_data = self.jobs[image_name][job_name]
            retry_count = job_data.get('retry_count', 0) + 1
            
            # Update the job to retrying status
            self.jobs[image_name][job_name].update({
                'status': 'retrying',
                'retry_count': retry_count,
                'retrying_at': datetime.now().isoformat()
            })
            self.save_tracking_data()
            logger.info(f"Retrying {job_name} for {image_name} (attempt {retry_count + 1})")
    
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
                if status not in summary['job_counts']:
                    summary['job_counts'][status] = 0
                summary['job_counts'][status] = summary['job_counts'].get(status, 0) + 1
                image_summary['jobs'][job_name] = {
                    'status': status,
                    'session_id': job_data.get('session_id'),
                    'submitted_at': job_data.get('submitted_at'),
                    'failed_at': job_data.get('failed_at'),
                    'error': job_data.get('error')
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
                          texture_model: str = 'none', additional_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Submit Image-to-3D job (doesn't wait for completion)"""
        
        # Handle both base64 image data and asset IDs
        if image_data.startswith('data:'):
            # Base64 image data
            image_input = image_data
        else:
            # Asset ID - pass as reference
            image_input = image_data
        
        # Start with common settings, then apply additional settings to override
        settings = {
            "geometry_model": geometry_model,
            "texture_model": texture_model,
            **COMMON_SETTINGS
        }
        if additional_settings:
            settings.update(additional_settings)
            
        session_data = {
            "type": "image_to_3d",
            "input": {
                "image": image_input,
                "model": "sculpt",
                "settings": settings
            }
        }
        
        settings_info = f"geometry_model={geometry_model}, texture_model={texture_model}"
        if additional_settings:
            additional_info = ", ".join(f"{k}={v}" for k, v in additional_settings.items())
            settings_info += f", {additional_info}"
        logger.info(f"Submitting Image-to-3D job: {settings_info}")
        return self.create_session(session_data)
    
    def submit_image_to_kit(self, image_data: str, decomposition_model: str = 'pro',
                           geometry_model: str = 'turbo', texture_model: str = 'baked', 
                           additional_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Submit Image-to-Kit job (doesn't wait for completion)"""
        
        # Handle both base64 image data and asset IDs
        if image_data.startswith('data:'):
            # Base64 image data
            image_input = image_data
        else:
            # Asset ID - pass as reference
            image_input = image_data
        
        # Start with common settings, then apply additional settings to override
        settings = {
            "geometry_model": geometry_model,
            "texture_model": texture_model,
            **COMMON_SETTINGS
        }
        if additional_settings:
            settings.update(additional_settings)
            
        session_data = {
            "type": "image_to_kit",
            "input": {
                "image": image_input,
                "model": "sculpt",
                "decomposition_model": decomposition_model,
                "settings": settings
            }
        }
        
        settings_info = f"decomposition_model={decomposition_model}, geometry_model={geometry_model}, texture_model={texture_model}"
        if additional_settings:
            additional_info = ", ".join(f"{k}={v}" for k, v in additional_settings.items())
            settings_info += f", {additional_info}"
        logger.info(f"Submitting Image-to-Kit job: {settings_info}")
        return self.create_session(session_data)
    
    def submit_retopology(self, mesh_url: str, model: str = 'precision', 
                         quads: bool = True) -> Dict[str, Any]:
        """Submit AI retopology job using mesh URL from existing session"""
        
        session_data = {
            "type": "retopology",
            "input": {
                "mesh": mesh_url,
                "quads": quads,
                "model": model
            }
        }
        
        logger.info(f"Submitting AI retopology job: model={model}, quads={quads}")
        return self.create_session(session_data)
    
    # DISABLED: Chat-to-3D functionality - Safety system issues
    # def submit_chat_to_3d(self, image_data: str, prompt: str) -> Dict[str, Any]:
    #     """Submit Chat-to-3D job (doesn't wait for completion)"""
    #     
    #     # Handle both base64 image data and asset IDs
    #     if image_data.startswith('data:'):
    #         # Base64 image data
    #         image_input = image_data
    #     else:
    #         # Asset ID - pass as reference
    #         image_input = image_data
    #         
    #     session_data = {
    #         "type": "chat_to_3d",
    #         "messages": [
    #             {
    #                 "type": "user_prompt",
    #                 "message": prompt,
    #                 "images": [image_input]
    #             }
    #         ]
    #     }
    #     
    #     logger.info(f"Submitting Chat-to-3D job with prompt: {prompt}")
    #     return self.create_session(session_data)
    
    # DISABLED: Chat-to-3D workflow functionality - Safety system issues
    # def submit_chat_to_3d_workflow(self, image_data: str, prompt: str, max_wait_time: int = 300) -> Dict[str, Any]:
    #     """Submit complete Chat-to-3D workflow: chat first, then follow-up Image-to-3D (synchronous)"""
    #     
    #     logger.info(f"Starting Chat-to-3D workflow with prompt: {prompt}")
    #     
    #     # Step 1: Submit chat-to-3D job
    #     chat_session = self.submit_chat_to_3d(image_data, prompt)
    #     chat_session_id = chat_session['_id']
    #     
    #     logger.info(f"Chat-to-3D submitted (session: {chat_session_id}), waiting for completion...")
    #     
    #     # Step 2: Wait for chat-to-3D to complete
    #     start_time = time.time()
    #     while time.time() - start_time < max_wait_time:
    #         try:
    #             chat_status = self.get_session_status(chat_session_id)
    #             api_status = chat_status.get('status', 'incomplete')
    #             
    #             if api_status == 'complete':
    #                 logger.info(f"Chat-to-3D completed! Extracting result image...")
    #                 
    #                 # Step 3: Extract the resulting image
    #                 output_image = None
    #                 if 'messages' in chat_status:
    #                     for message in chat_status['messages']:
    #                         if message.get('type') == 'image_generation' and 'images' in message:
    #                             for img in message['images']:
    #                                 if 'asset' in img and img['asset'].get('status') == 'complete':
    #                                     output_image = img['asset']['_id']
    #                                     logger.info(f"Found output image: {output_image}")
    #                                     break
    #                             if output_image:
    #                                 break
    #                 
    #                 if not output_image:
    #                     raise Exception("No output image found in completed chat session")
    #                 
    #                 # Step 4: Submit follow-up Image-to-3D job immediately
    #                 logger.info(f"Submitting follow-up Image-to-3D using chat result...")
    #                 followup_session = self.submit_image_to_3d(
    #                     output_image,
    #                     geometry_model='turbo',  # Use turbo for the follow-up
    #                     texture_model='pbr'      # Use pbr for better quality
    #                 )
    #                 
    #                 logger.info(f"Complete Chat-to-3D workflow submitted! Follow-up Image-to-3D session: {followup_session['_id']}")
    #                 
    #                 # Return combined session info
    #                 return {
    #                     '_id': followup_session['_id'],  # Main session ID for tracking
    #                     'workflow_type': 'chat_to_3d_then_image_to_3d',
    #                     'chat_session': chat_status,
    #                     'followup_session': followup_session,
    #                     'output_image_id': output_image
    #                 }
    #                 
    #             elif api_status == 'failed':
    #                 raise Exception(f"Chat-to-3D failed: {chat_status}")
    #             
    #             else:
    #                 # Still in progress, wait a bit
    #                 logger.info(f"Chat-to-3D in progress (status: {api_status}), waiting...")
    #                 time.sleep(10)  # Wait 10 seconds before checking again
    #                 
    #         except Exception as e:
    #             if "Chat-to-3D failed" in str(e):
    #                 raise  # Re-raise chat failures immediately
    #             logger.warning(f"Error checking chat status, retrying: {e}")
    #             time.sleep(5)
    #     
    #     # Timeout
    #     raise Exception(f"Chat-to-3D workflow timed out after {max_wait_time} seconds")

def check_job_progress(client: CSMAPIClient, tracker: JobTracker):
    """Check progress of submitted jobs and update their status"""
    logger.info("Checking progress of submitted jobs...")
    
    # Collect all jobs to check
    jobs_to_check = []
    for image_name, jobs in tracker.jobs.items():
        for job_name, job_data in jobs.items():
            jobs_to_check.append((image_name, job_name, job_data))
    
    if not jobs_to_check:
        logger.info("No jobs to check")
        return
    
    for image_name, job_name, job_data in tqdm(jobs_to_check, desc="Checking job progress", unit="job"):
        status = job_data.get('status')
        if status == 'submitted':
            session_id = job_data.get('session_id')
            if session_id:
                try:
                    session_status = client.get_session_status(session_id)
                    api_status = session_status.get('status', 'incomplete')
                    
                    if api_status == 'complete':
                        if image_name.startswith('retopo_'):
                            logger.info(f"🔧 Retopology job {job_name} completed!")
                        else:
                            logger.info(f"Job {job_name} for {image_name} completed!")
                        tracker.update_job_status(image_name, job_name, 'complete', session_status)
                    elif api_status == 'failed':
                        if image_name.startswith('retopo_'):
                            logger.warning(f"🔧 Retopology job {job_name} failed")
                        else:
                            logger.warning(f"Job {job_name} for {image_name} failed")
                        tracker.update_job_status(image_name, job_name, 'failed', session_status)
                    else:
                        if image_name.startswith('retopo_'):
                            logger.info(f"🔧 Retopology job {job_name} still in progress (status: {api_status})")
                        else:
                            logger.info(f"Job {job_name} for {image_name} still in progress (status: {api_status})")
                
                except Exception as e:
                    logger.error(f"Error checking job {job_name} for {image_name}: {e}")
        elif status == 'failed':
            retry_count = job_data.get('retry_count', 0)
            if tracker.should_retry_failed_job(image_name, job_name):
                logger.info(f"Job {job_name} for {image_name} previously failed (retry {retry_count}/3): {job_data.get('error', 'Unknown error')} - eligible for retry")
            else:
                logger.info(f"Job {job_name} for {image_name} permanently failed (retry {retry_count}/3): {job_data.get('error', 'Unknown error')}")
        elif status == 'retrying':
            retry_count = job_data.get('retry_count', 0)
            if image_name.startswith('retopo_'):
                logger.info(f"Retopology job {job_name} is being retried (attempt {retry_count + 1})")
            else:
                logger.info(f"Job {job_name} for {image_name} is being retried (attempt {retry_count + 1})")
        elif status == 'complete':
            if image_name.startswith('retopo_'):
                logger.info(f"Retopology job {job_name} already completed")
            else:
                logger.info(f"Job {job_name} for {image_name} already completed")
    
    # Check for missing jobs (jobs that should exist but don't) - ONLY for current images
    expected_jobs = [config['name'] for config in JOB_CONFIGS]
    current_images = set()
    images_dir = Path("images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
        current_images = {img.stem for img in image_files}
    
    for image_name in tracker.jobs:
        # Only check images that currently exist in the directory
        if image_name not in current_images:
            continue
            
        actual_jobs = set(tracker.jobs[image_name].keys())
        missing_jobs = set(expected_jobs) - actual_jobs
        if missing_jobs:
            logger.warning(f"Missing jobs for {image_name}: {', '.join(missing_jobs)} - these may have failed to submit")

def run_retopology_from_sessions():
    """Run retopology jobs from session IDs listed in retopo_sessions.txt"""
    logger.info("Running retopology mode")
    
    retopo_sessions_file = "retopo_sessions.txt"
    if not os.path.exists(retopo_sessions_file):
        logger.error(f"File {retopo_sessions_file} not found")
        logger.error("Create this file and add session IDs (one per line) from completed mesh generation jobs")
        logger.error("Example session IDs: SESSION_XXXXXXXXX_XXXXXXXX")
        return
    
    # Read session IDs from file
    session_ids = []
    try:
        with open(retopo_sessions_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    session_ids.append(line)
    except Exception as e:
        logger.error(f"Error reading {retopo_sessions_file}: {e}")
        return
    
    if not session_ids:
        logger.error(f"No session IDs found in {retopo_sessions_file}")
        logger.error("Add session IDs (one per line) from completed mesh generation jobs")
        return
    
    logger.info(f"Found {len(session_ids)} session IDs to process for retopology")
    
    # Initialize API client and job tracker
    client = CSMAPIClient(CSM_API_KEY)
    tracker = JobTracker()
    
    # Process each session
    retopo_results = []
    jobs_submitted = 0
    jobs_skipped = 0
    
    for session_id in tqdm(session_ids, desc="Processing sessions for retopology", unit="session"):
        try:
            logger.info(f"Processing session: {session_id}")
            
            # Use session ID as the "image_name" for tracking retopology jobs
            retopo_key = f"retopo_{session_id}"
            
            # Check if retopology jobs for this session already exist
            swift_job_name = f"retopo_swift_{session_id}"
            precision_job_name = f"retopo_precision_{session_id}"
            
            swift_exists = tracker.should_skip_job(retopo_key, swift_job_name)
            precision_exists = tracker.should_skip_job(retopo_key, precision_job_name)
            
            if swift_exists and precision_exists:
                logger.info(f"Retopology jobs for session {session_id} already exist, skipping")
                jobs_skipped += 2
                continue
            
            # Get session details
            session_data = client.get_session_status(session_id)
            
            if session_data.get('status') != 'complete':
                logger.warning(f"Session {session_id} is not complete (status: {session_data.get('status')}), skipping")
                continue
            
            # Extract mesh asset IDs from the session
            mesh_assets = extract_mesh_assets_from_session(session_data)
            
            if not mesh_assets:
                logger.warning(f"No mesh assets found in session {session_id}, skipping")
                continue
            
            logger.info(f"Found {len(mesh_assets)} meshes in session {session_id}")
            
            # Run retopology on each mesh (both swift and precision)
            for i, mesh_asset in enumerate(mesh_assets):
                mesh_id = mesh_asset['id']
                mesh_url = mesh_asset.get('url', '')
                
                if not mesh_url:
                    logger.warning(f"No GLB URL found for mesh {mesh_id}, skipping")
                    continue
                    
                logger.info(f"Running retopology on mesh {i+1}/{len(mesh_assets)}: {mesh_id}")
                logger.info(f"Using mesh URL: {mesh_url[:100]}...")
                
                # Submit swift retopology (if not already submitted)
                if not swift_exists:
                    try:
                        swift_session = client.submit_retopology(mesh_url, model="swift", quads=True)
                        logger.info(f"Swift retopology submitted: {swift_session['_id']}")
                        
                        # Track in job tracker
                        tracker.mark_job_submitted(retopo_key, swift_job_name, swift_session['_id'], swift_session)
                        
                        retopo_results.append({
                            "original_session": session_id,
                            "mesh_asset_id": mesh_id,
                            "mesh_url": mesh_url,
                            "retopo_session": swift_session['_id'],
                            "retopo_model": "swift",
                            "status": "submitted"
                        })
                        jobs_submitted += 1
                    except Exception as e:
                        logger.error(f"Failed to submit swift retopology for {mesh_id}: {e}")
                        tracker.mark_job_failed(retopo_key, swift_job_name, str(e))
                else:
                    logger.info(f"Swift retopology for {session_id} already submitted, skipping")
                    jobs_skipped += 1
                
                # Submit precision retopology (if not already submitted)
                if not precision_exists:
                    try:
                        precision_session = client.submit_retopology(mesh_url, model="precision", quads=True)
                        logger.info(f"Precision retopology submitted: {precision_session['_id']}")
                        
                        # Track in job tracker
                        tracker.mark_job_submitted(retopo_key, precision_job_name, precision_session['_id'], precision_session)
                        
                        retopo_results.append({
                            "original_session": session_id,
                            "mesh_asset_id": mesh_id,
                            "mesh_url": mesh_url,
                            "retopo_session": precision_session['_id'],
                            "retopo_model": "precision",
                            "status": "submitted"
                        })
                        jobs_submitted += 1
                    except Exception as e:
                        logger.error(f"Failed to submit precision retopology for {mesh_id}: {e}")
                        tracker.mark_job_failed(retopo_key, precision_job_name, str(e))
                else:
                    logger.info(f"Precision retopology for {session_id} already submitted, skipping")
                    jobs_skipped += 1
        
        except Exception as e:
            logger.error(f"Error processing session {session_id}: {e}")
    
    # Save retopology results (for reference)
    if retopo_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_file = os.path.join(RESULTS_DIR, "retopology_results.json")
        save_results({
            "timestamp": datetime.now().isoformat(),
            "processed_sessions": session_ids,
            "retopology_jobs": retopo_results
        }, results_file)
        logger.info(f"Retopology jobs submitted successfully. Results saved to {results_file}")
    
    # Summary
    logger.info(f"Retopology summary: {jobs_submitted} jobs submitted, {jobs_skipped} jobs skipped (already exist)")
    if jobs_submitted == 0 and jobs_skipped == 0:
        logger.warning("No retopology jobs were submitted")
    elif jobs_submitted > 0:
        logger.info("✅ New retopology jobs submitted! Use './run_eval.sh progress' to monitor progress")

def extract_mesh_assets_from_session(session_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract mesh asset information from a completed session"""
    mesh_assets = []
    output = session_data.get('output', {})
    
    # For image_to_3d sessions, check for single mesh
    if 'mesh' in output and isinstance(output['mesh'], dict):
        mesh = output['mesh']
        if mesh.get('status') == 'complete' and mesh.get('_id'):
            mesh_data = mesh.get('data', {})
            mesh_assets.append({
                'id': mesh['_id'],
                'url': mesh_data.get('glb_url', ''),
                'type': 'single_mesh'
            })
    
    # For image_to_kit sessions (and some image_to_3d), check for meshes array
    if 'meshes' in output and isinstance(output['meshes'], list):
        for i, mesh in enumerate(output['meshes']):
            if isinstance(mesh, dict) and mesh.get('status') == 'complete' and mesh.get('_id'):
                mesh_data = mesh.get('data', {})
                mesh_assets.append({
                    'id': mesh['_id'],
                    'url': mesh_data.get('glb_url', ''),
                    'type': f'meshes[{i}]'
                })
    
    # For image_to_kit sessions, also check for part_meshes
    if 'part_meshes' in output and isinstance(output['part_meshes'], list):
        for i, mesh in enumerate(output['part_meshes']):
            if isinstance(mesh, dict) and mesh.get('status') == 'complete' and mesh.get('_id'):
                mesh_data = mesh.get('data', {})
                mesh_assets.append({
                    'id': mesh['_id'],
                    'url': mesh_data.get('glb_url', ''),
                    'type': f'part_meshes[{i}]'
                })
    
    return mesh_assets

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
    
    # Upload image only once (will be reused via asset ID)
    image_data = client.upload_image(image_path)
    image_asset_id = None  # Will be populated after first successful job submission
    
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
        
        # Check if this is a retry attempt
        is_retry = (image_name in tracker.jobs and 
                   job_name in tracker.jobs[image_name] and 
                   tracker.jobs[image_name][job_name].get('status') == 'failed')
        
        if is_retry:
            tracker.mark_job_retry(image_name, job_name)
            # Add delay for retries to be respectful to the API
            retry_count = tracker.jobs[image_name][job_name].get('retry_count', 0)
            delay = min(2 ** retry_count, 30)  # Exponential backoff, max 30 seconds
            logger.info(f"Waiting {delay} seconds before retry...")
            time.sleep(delay)
        
        # Use asset ID if we have it from a previous job, otherwise use image data
        current_image_input = image_asset_id if image_asset_id else image_data
        
        try:
            if job_config["type"] == "image_to_3d":
                # Extract additional settings (any settings beyond the standard parameters)
                standard_params = {"name", "type", "geometry_model", "texture_model"}
                additional_settings = {k: v for k, v in job_config.items() if k not in standard_params}
                
                session = client.submit_image_to_3d(
                    current_image_input,
                    geometry_model=job_config["geometry_model"],
                    texture_model=job_config["texture_model"],
                    additional_settings=additional_settings if additional_settings else None
                )
                
                # Extract asset ID from first successful submission
                if not image_asset_id and 'input' in session and 'image' in session['input']:
                    if isinstance(session['input']['image'], dict) and '_id' in session['input']['image']:
                        image_asset_id = session['input']['image']['_id']
                        logger.info(f"Extracted image asset ID: {image_asset_id} - will reuse for subsequent jobs")
                
                tracker.mark_job_submitted(image_name, job_name, session['_id'], session)
                results["jobs"][job_name] = {"session_id": session['_id'], "status": "submitted"}
                
            elif job_config["type"] == "image_to_kit":
                # Extract additional settings (any settings beyond the standard parameters)
                standard_params = {"name", "type", "decomposition_model", "geometry_model", "texture_model"}
                additional_settings = {k: v for k, v in job_config.items() if k not in standard_params}
                
                session = client.submit_image_to_kit(
                    current_image_input,
                    decomposition_model=job_config["decomposition_model"],
                    geometry_model=job_config["geometry_model"],
                    texture_model=job_config["texture_model"],
                    additional_settings=additional_settings if additional_settings else None
                )
                
                # Extract asset ID from first successful submission
                if not image_asset_id and 'input' in session and 'image' in session['input']:
                    if isinstance(session['input']['image'], dict) and '_id' in session['input']['image']:
                        image_asset_id = session['input']['image']['_id']
                        logger.info(f"Extracted image asset ID: {image_asset_id} - will reuse for subsequent jobs")
                
                tracker.mark_job_submitted(image_name, job_name, session['_id'], session)
                results["jobs"][job_name] = {"session_id": session['_id'], "status": "submitted"}
                

                
            # DISABLED: Chat-to-3D functionality - Safety system issues
            # elif job_config["type"] == "chat_to_3d":
            #     # Use synchronous chat-to-3D workflow (waits for chat, then submits image-to-3D)
            #     chat_prompt = "create a 3D rendering version of this image, front facing with 45 degree for image to 3D. do not change semantic details. If this has multiple front and back views, generate a single 3D asset."
            #     session = client.submit_chat_to_3d_workflow(current_image_input, chat_prompt)
            #     
            #     # Extract asset ID from first successful submission
            #     if not image_asset_id:
            #         # Try to get original image asset ID from chat session data
            #         if 'chat_session' in session and 'messages' in session['chat_session']:
            #             for message in session['chat_session']['messages']:
            #                 if message.get('type') == 'user_prompt' and 'images' in message:
            #                     for img in message['images']:
            #                         if isinstance(img, dict) and '_id' in img:
            #                             image_asset_id = img['_id']
            #                             logger.info(f"Extracted image asset ID: {image_asset_id} - will reuse for subsequent jobs")
            #                             break
            #                     if image_asset_id:
            #                         break
            #     
            #     # Track the final image-to-3D session ID (the one we care about for results)
            #     tracker.mark_job_submitted(image_name, job_name, session['_id'], session)
            #     results["jobs"][job_name] = {
            #         "session_id": session['_id'], 
            #         "status": "submitted",
            #         "note": "Complete Chat-to-3D workflow: chat completed, follow-up Image-to-3D submitted"
            #     }
            
            logger.info(f"Successfully submitted {job_name} for {image_name}")
            
        except Exception as e:
            logger.error(f"Failed to submit {job_name} for {image_name}: {e}")
            tracker.mark_job_failed(image_name, job_name, str(e))
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CSM API Evaluation Script")
    parser.add_argument('--progress-only', action='store_true', 
                        help='Only check progress of existing jobs, do not submit new ones')
    parser.add_argument('--retopo-mode', action='store_true', 
                        help='Run retopology jobs from retopo_sessions.txt')
    args = parser.parse_args()
    
    logger.info("Starting CSM API evaluation")
    
    # Initialize API client and job tracker
    client = CSMAPIClient(CSM_API_KEY)
    tracker = JobTracker()
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Handle retopology mode
    if args.retopo_mode:
        run_retopology_from_sessions()
        return
    
    # Always check progress of existing jobs first
    if tracker.jobs:
        check_job_progress(client, tracker)
    
    # If this is progress-only mode, skip job submission
    if args.progress_only:
        logger.info("Progress-only mode - skipping job submission")
    else:
        logger.info("Evaluation mode - will submit new jobs and retry eligible failed jobs")
        
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
        
        for image_file in tqdm(image_files, desc="Processing images", unit="image"):
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
    
    # Check for missing jobs across all processed images and provide detailed feedback
    expected_jobs = [config['name'] for config in JOB_CONFIGS]
    images_dir = Path("images")
    
    if images_dir.exists():
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
        
        # Check each image for missing jobs - ONLY for images that currently exist in the directory
        missing_jobs_report = {}
        failed_jobs_report = {}
        
        for image_file in image_files:
            image_name = image_file.stem
            
            if image_name in tracker.jobs:
                actual_jobs = set(tracker.jobs[image_name].keys())
                missing_jobs = set(expected_jobs) - actual_jobs
                if missing_jobs:
                    missing_jobs_report[image_name] = list(missing_jobs)
                
                # Check for failed jobs
                failed_jobs = []
                for job_name, job_data in tracker.jobs[image_name].items():
                    if job_data.get('status') == 'failed':
                        failed_jobs.append({
                            'job': job_name,
                            'error': job_data.get('error', 'Unknown error')
                        })
                if failed_jobs:
                    failed_jobs_report[image_name] = failed_jobs
            else:
                # No jobs tracked for this image at all
                missing_jobs_report[image_name] = expected_jobs
        
        if missing_jobs_report:
            logger.warning("Missing jobs detected:")
            for image_name, missing in missing_jobs_report.items():
                logger.warning(f"  {image_name}: {', '.join(missing)}")
        
        if failed_jobs_report:
            logger.error("Failed jobs detected:")
            for image_name, failed in failed_jobs_report.items():
                for job_info in failed:
                    logger.error(f"  {image_name}/{job_info['job']}: {job_info['error']}")
        
        if not missing_jobs_report and not failed_jobs_report:
            logger.info("All expected jobs are tracked and none failed during submission")
    
    # Save tracking summary
    summary_path = os.path.join(RESULTS_DIR, "job_summary.json")
    save_results(summary, summary_path)
    
    if args.progress_only:
        logger.info("Progress check completed.")
    else:
        logger.info("CSM API evaluation completed. Run with --progress-only to check job progress.")
    logger.info(f"Track job progress in: {JOB_TRACKING_FILE}")

if __name__ == "__main__":
    main()
