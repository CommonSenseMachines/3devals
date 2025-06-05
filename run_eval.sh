#!/bin/bash

# CSM API Evaluation Runner Script
# This script sets up and runs the CSM API evaluation for 3D generation

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check and setup API key
setup_api_key() {
    local config_file=".csm_config"
    
    # Check if API key is already set in environment
    if [ -n "$CSM_API_KEY" ] && [ "$CSM_API_KEY" != "" ]; then
        return 0  # API key is already set, no need to prompt
    fi
    
    # Try to load from local config file
    if [ -f "$config_file" ]; then
        CSM_API_KEY=$(cat "$config_file" 2>/dev/null | grep -E '^[a-zA-Z0-9]{32}$')
        if [ -n "$CSM_API_KEY" ]; then
            export CSM_API_KEY
            log "API key loaded from $config_file"
            return 0
        fi
    fi
    
    echo ""
    echo -e "${YELLOW}CSM API Key Required${NC}"
    echo "To get your API key:"
    echo "1. Go to https://3d.csm.ai/"
    echo "2. Click on Profile Settings (bottom left)"
    echo "3. Navigate to Settings → Developer Settings"
    echo "4. Copy your API key"
    echo ""
    read -p "Please enter your CSM API key: " -r CSM_API_KEY
    
    if [ -z "$CSM_API_KEY" ] || [ "$CSM_API_KEY" = "" ]; then
        error "API key is required to run the evaluation"
        exit 1
    fi
    
    # Validate API key format (basic check for 32 character alphanumeric)
    if ! echo "$CSM_API_KEY" | grep -qE '^[a-zA-Z0-9]{32}$'; then
        warn "API key format doesn't look correct (expected 32 alphanumeric characters)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    export CSM_API_KEY
    
    # Save to local config file with secure permissions
    echo "$CSM_API_KEY" > "$config_file"
    chmod 600 "$config_file"  # Only owner can read/write
    
    log "API key saved to $config_file and set successfully"
    info "Your API key is now saved locally and will be used automatically"
}

# Function to setup Python environment
setup_environment() {
    log "Setting up evaluation environment..."
    
    # Check for Python
    if ! command_exists python3; then
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check for pip
    if ! command_exists pip3; then
        error "pip3 is required but not installed"
        exit 1
    fi
    
    # Install dependencies
    log "Installing Python dependencies..."
    pip3 install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        error "Failed to install dependencies"
        exit 1
    fi
    
    # Create necessary directories
    mkdir -p images results
    
    # Check if images directory has content
    if [ -z "$(ls -A images/ 2>/dev/null)" ]; then
        warn "No images found in the images/ directory"
        info "Please add PNG, JPG, or JPEG images to the images/ directory before running evaluation"
        return 1
    fi
    
    log "Environment setup complete"
    return 0
}

# Function to show what will be evaluated
show_evaluation_plan() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}                    🚀 CSM.AI 3D Generation Evaluation${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}For each image in images/, this will run 5 job configurations:${NC}"
    echo ""
    echo -e "  ${GREEN}1.${NC} Image-to-3D (base)        → geometry_model='base'"
    echo -e "  ${GREEN}2.${NC} Image-to-3D (turbo)       → geometry_model='turbo'"
    echo -e "  ${GREEN}3.${NC} Image-to-3D (turbo+baked) → geometry_model='turbo' + texture_model='baked'"
    echo -e "  ${GREEN}4.${NC} Image-to-3D (turbo+pbr)   → geometry_model='turbo' + texture_model='pbr'"
    echo -e "  ${GREEN}5.${NC} Image-to-Kit (pro)        → decomposition_model='pro' + geometry_model='turbo' + texture_model='baked'"
    # echo -e "  ${GREEN}6.${NC} Chat-to-3D                → Re-prompt image for better pose, then Image-to-3D"
    echo ""
    echo -e "${BLUE}Settings:${NC} resolution=200000 for all jobs"
    echo -e "${BLUE}Note:${NC} This script does not run AI retopology (can be time consuming)"
    echo -e "${BLUE}      Retopology can be run separately: ${NC}https://docs.csm.ai/sessions/retopology"
    echo ""
    echo -e "${RED}⚠️  IMPORTANT:${NC} Some jobs may initially fail due to API rate limits."
    echo -e "${RED}    The evaluation will automatically retry failed jobs with exponential backoff.${NC}"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Function to run the evaluation
run_evaluation() {
    log "Starting CSM API evaluation..."
    
    # Check if images exist
    if [ -z "$(ls -A images/ 2>/dev/null)" ]; then
        error "No images found in images/ directory. Please add images first."
        exit 1
    fi
    
    # Count images
    IMAGE_COUNT=$(ls images/*.{png,jpg,jpeg} 2>/dev/null | wc -l)
    log "Found $IMAGE_COUNT images to process"
    
    # Show evaluation plan
    show_evaluation_plan
    
    # Run the Python evaluation script
    python3 csm_eval.py
    
    if [ $? -eq 0 ]; then
        log "Evaluation completed successfully"
        show_summary
    else
        error "Evaluation failed"
        exit 1
    fi
}

# Function to check job progress
check_progress() {
    log "Checking job progress..."
    
    if [ ! -f "job_tracking.json" ]; then
        warn "No job tracking file found. Run evaluation first."
        return 1
    fi
    
    echo ""
    echo -e "${RED}⚠️  NOTE:${NC} If you see failed jobs, they may be due to API rate limits."
    echo -e "${RED}    Run ${GREEN}./run_eval.sh eval${NC}${RED} to automatically retry failed jobs.${NC}"
    echo ""
    
    python3 csm_eval.py --progress-only
    show_summary
}

# Function to show evaluation summary
show_summary() {
    log "Evaluation Summary:"
    
    if [ -f "results/job_summary.json" ]; then
        echo ""
        python3 -c "
import json
try:
    with open('results/job_summary.json', 'r') as f:
        summary = json.load(f)
    
    print(f\"Total Images: {summary['total_images']}\")
    print(f\"Job Status Counts:\")
    for status, count in summary['job_counts'].items():
        print(f\"  {status}: {count}\")
    
    if summary['total_images'] > 0:
        print(f\"\nPer-image breakdown:\")
        for image_name, image_data in summary['images'].items():
            job_statuses = [job['status'] for job in image_data['jobs'].values()]
            completed = job_statuses.count('complete')
            total = len(job_statuses)
            print(f\"  {image_name}: {completed}/{total} jobs complete\")
except Exception as e:
    print(f\"Could not read summary: {e}\")
"
    else
        warn "No summary file found"
    fi
}

# Function to clean up old results
clean_results() {
    log "Cleaning up previous results..."
    
    read -p "This will delete all results and job tracking. Continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf results/
        rm -f job_tracking.json
        log "Results cleaned up"
        mkdir -p results
    else
        info "Cleanup cancelled"
    fi
}

# Function to show help
show_help() {
    echo -e "${GREEN}🚀 Evaluation framework for 3D generative AI${NC}"
    echo ""
    echo -e "${BLUE}This tool evaluates CSM.AI's 3D generation capabilities by running multiple"
    echo -e "job configurations on your images and comparing the results. It tests:${NC}"
    echo -e "${GREEN}• Image-to-3D generation (base, turbo, with textures)"
    echo -e "• Image-to-Kit decomposition (breaking objects into parts)${NC}"
    # echo -e "• Chat-to-3D workflows (AI-improved poses)${NC}"
    echo ""
    echo -e "${BLUE}Results are tracked and can be analyzed to benchmark 3D AI performance.${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC} $0 [COMMAND]"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo -e "  ${GREEN}run${NC}         Run the full evaluation (setup + evaluate all images in images/*)"
    echo -e "  ${GREEN}setup${NC}       Setup environment and dependencies only"
    echo -e "  ${GREEN}eval${NC}        Run evaluation only (skip setup)"
    echo -e "  ${GREEN}progress${NC}    Check progress of submitted jobs"
    echo -e "  ${GREEN}clean${NC}       Clean up previous results"
    echo -e "  ${GREEN}help${NC}        Show this help message"
    echo ""
    echo -e "${YELLOW}Environment Variables:${NC}"
    echo -e "  ${GREEN}CSM_API_KEY${NC}   CSM.ai API key (get from ${BLUE}https://3d.csm.ai/${NC})"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  ${BLUE}$0 run${NC}                 # Full evaluation workflow"
    echo -e "  ${BLUE}$0 progress${NC}            # Check job progress"
    echo -e "  ${BLUE}$0 clean && $0 run${NC}     # Clean start"
}

# Main script logic
case "${1:-help}" in
    "run")
        setup_api_key
        if setup_environment; then
            run_evaluation
        else
            error "Setup failed. Please add images to the images/ directory and try again."
            exit 1
        fi
        ;;
    "setup")
        setup_environment
        ;;
    "eval"|"evaluate")
        setup_api_key
        run_evaluation
        ;;
    "progress"|"check")
        setup_api_key
        check_progress
        ;;
    "clean"|"cleanup")
        clean_results
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
