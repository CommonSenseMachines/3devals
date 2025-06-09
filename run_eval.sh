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
    echo -e "${YELLOW}For each image in images/, this will run 6 job configurations:${NC}"
    echo ""
    echo -e "  ${GREEN}1.${NC} Image-to-3D (base)        → Basic geometry generation with no texturing"
    echo -e "  ${GREEN}2.${NC} Image-to-3D (turbo)       → Fast geometry generation with no texturing"
    echo -e "  ${GREEN}3.${NC} Image-to-3D (turbo+baked) → Fast geometry with baked texture maps"
    echo -e "  ${GREEN}4.${NC} Image-to-3D (turbo+pbr)   → Fast geometry with PBR material textures"
    echo -e "  ${GREEN}5.${NC} Image-to-Kit (pro)        → Professional decomposition with turbo geometry and baked textures"
    echo -e "  ${GREEN}6.${NC} Image-to-3D (250k)        → High resolution basic geometry generation"
    echo ""
    echo -e "${BLUE}Settings:${NC} resolution=200000 for standard jobs, 250000 for high-res job"
    echo -e "${BLUE}Note:${NC} AI retopology can be run separately using the ${GREEN}retopo${NC} command"
    echo -e "${BLUE}      See ${GREEN}./run_eval.sh help${NC} for retopology usage"
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

# Function to run retopology jobs
run_retopology() {
    log "Running AI retopology jobs..."
    
    if [ ! -f "retopo_sessions.txt" ]; then
        warn "File retopo_sessions.txt not found"
        echo ""
        echo -e "${BLUE}To use retopology:${NC}"
        echo -e "  1. Create ${GREEN}retopo_sessions.txt${NC}"
        echo -e "  2. Add session IDs from completed mesh generation jobs (one per line)"
        echo -e "  3. Run ${GREEN}./run_eval.sh retopo${NC}"
        echo ""
        echo -e "${YELLOW}Example retopo_sessions.txt:${NC}"
        echo -e "  SESSION_1234567890_1234567"
        echo -e "  SESSION_0987654321_0987654"
        echo -e "  # Lines starting with # are comments"
        echo ""
        info "You can find session IDs in results/job_summary.json or job_tracking.json"
        return 1
    fi
    
    # Count non-empty, non-comment lines
    SESSION_COUNT=$(grep -v '^#' retopo_sessions.txt 2>/dev/null | grep -v '^[[:space:]]*$' | wc -l)
    
    if [ "$SESSION_COUNT" -eq 0 ]; then
        warn "No session IDs found in retopo_sessions.txt"
        info "Add session IDs from completed mesh generation jobs (one per line)"
        return 1
    fi
    
    log "Found $SESSION_COUNT session(s) to process for retopology"
    
    echo ""
    echo -e "${BLUE}🔧 AI Retopology Process:${NC}"
    echo -e "  • Reads session IDs from ${GREEN}retopo_sessions.txt${NC}"
    echo -e "  • Extracts mesh URLs from completed sessions"
    echo -e "  • Runs both ${GREEN}swift${NC} and ${GREEN}precision${NC} retopology on each mesh"
    echo -e "  • Outputs clean quad topology meshes"
    echo ""
    
    python3 csm_eval.py --retopo-mode
    
    if [ $? -eq 0 ]; then
        log "Retopology jobs submitted successfully"
        info "Check results/retopology_results.json for job details"
        info "Monitor progress with: ./run_eval.sh progress"
    else
        error "Retopology submission failed"
        exit 1
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

# Function to clean leaderboard caches
clean_leaderboard() {
    log "Cleaning leaderboard caches and debug data..."
    
    echo ""
    echo -e "${YELLOW}This will delete:${NC}"
    echo -e "  • ${BLUE}llm_cache/${NC} - LLM response cache"
    echo -e "  • ${BLUE}leaderboard/debug_queries/${NC} - Debug images and prompts"
    echo -e "  ${GREEN}(API keys in .leaderboard_config will be preserved)${NC}"
    echo ""
    
    read -p "Continue with leaderboard cache cleanup? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove LLM cache
        if [ -d "llm_cache" ]; then
            rm -rf llm_cache/
            log "Removed llm_cache/"
        else
            info "llm_cache/ not found"
        fi
        
        # Remove debug queries
        if [ -d "leaderboard/debug_queries" ]; then
            rm -rf leaderboard/debug_queries/
            log "Removed leaderboard/debug_queries/"
        else
            info "leaderboard/debug_queries/ not found"
        fi
        
        # Keep .leaderboard_config (API keys) - don't delete
        if [ -f ".leaderboard_config" ]; then
            info "Preserved .leaderboard_config (API keys)"
        fi
        
        log "Leaderboard cache cleanup completed"
    else
        info "Leaderboard cleanup cancelled"
    fi
}

# Function to regenerate visualizations from existing results
regenerate_visualizations() {
    log "Regenerating visualizations from existing CSV results..."
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}                    📊 Visualization Regeneration${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}Regenerate charts and analysis from existing CSV results without re-running LLM evaluation.${NC}"
    echo ""
    
    # Check for available CSV files
    CSV_FILES=(leaderboard_results*.csv)
    if [ ! -e "${CSV_FILES[0]}" ]; then
        error "No leaderboard results CSV files found"
        info "Run leaderboard evaluation first: ./run_eval.sh leaderboard"
        exit 1
    fi
    
    echo -e "${GREEN}Available result files:${NC}"
    for i in "${!CSV_FILES[@]}"; do
        echo -e "  ${BLUE}$((i+1)).${NC} ${CSV_FILES[$i]}"
    done
    echo ""
    
    read -p "Select file to regenerate [1-${#CSV_FILES[@]}]: " -r FILE_CHOICE
    
    if [[ "$FILE_CHOICE" -ge 1 && "$FILE_CHOICE" -le "${#CSV_FILES[@]}" ]]; then
        SELECTED_FILE="${CSV_FILES[$((FILE_CHOICE-1))]}"
    else
        error "Invalid selection"
        exit 1
    fi
    
    # Optional custom output directory
    echo ""
    read -p "Custom output directory (press Enter for auto-naming): " -r OUTPUT_DIR
    
    # Build command
    CMD="python3 leaderboard/llm_leaderboard.py --visualize-only $SELECTED_FILE"
    if [ -n "$OUTPUT_DIR" ]; then
        CMD="$CMD --output-dir $OUTPUT_DIR"
    fi
    
    echo ""
    log "Regenerating visualizations..."
    info "Command: $CMD"
    echo ""
    
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo ""
        log "Visualization regeneration completed successfully"
    else
        error "Visualization regeneration failed"
        exit 1
    fi
}

# Function to compare multiple provider results
compare_providers() {
    log "Comparing results from multiple LLM providers..."
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}                    🔍 Provider Comparison Analysis${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}Generate comparative analysis between multiple LLM providers.${NC}"
    echo ""
    
    # Check for provider-specific CSV files
    declare -a AVAILABLE_PROVIDERS
    for provider in claude gemini openai; do
        if [ -f "leaderboard_results_${provider}.csv" ]; then
            AVAILABLE_PROVIDERS+=("$provider")
        fi
    done
    
    if [ ${#AVAILABLE_PROVIDERS[@]} -lt 2 ]; then
        error "Need at least 2 provider result files for comparison"
        echo ""
        echo -e "${BLUE}Expected files:${NC}"
        echo -e "  • ${GREEN}leaderboard_results_claude.csv${NC}"
        echo -e "  • ${GREEN}leaderboard_results_gemini.csv${NC}"
        echo -e "  • ${GREEN}leaderboard_results_openai.csv${NC}"
        echo ""
        info "Run leaderboard evaluation with different providers first"
        exit 1
    fi
    
    echo -e "${GREEN}Available providers:${NC}"
    for provider in "${AVAILABLE_PROVIDERS[@]}"; do
        echo -e "  ✅ ${provider}"
    done
    echo ""
    
    # Let user select which providers to compare
    echo -e "${YELLOW}Select providers to compare (space-separated, e.g., 'claude gemini'):${NC}"
    read -p "Providers [default: all available]: " -r SELECTED_PROVIDERS
    
    if [ -z "$SELECTED_PROVIDERS" ]; then
        SELECTED_PROVIDERS="${AVAILABLE_PROVIDERS[*]}"
    fi
    
    # Validate selected providers
    declare -a VALID_PROVIDERS
    for provider in $SELECTED_PROVIDERS; do
        if [[ " ${AVAILABLE_PROVIDERS[@]} " =~ " ${provider} " ]]; then
            VALID_PROVIDERS+=("$provider")
        else
            warn "Provider '$provider' not available, skipping"
        fi
    done
    
    if [ ${#VALID_PROVIDERS[@]} -lt 2 ]; then
        error "Need at least 2 valid providers for comparison"
        exit 1
    fi
    
    # Custom output directory
    echo ""
    DEFAULT_OUTPUT="results_comparison"
    read -p "Output directory [default: $DEFAULT_OUTPUT]: " -r OUTPUT_DIR
    OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT}
    
    # Build command
    CMD="python3 leaderboard/llm_leaderboard.py --combine-providers ${VALID_PROVIDERS[*]} --combined-output $OUTPUT_DIR"
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Comparison Summary:${NC}"
    echo -e "  Providers: ${GREEN}${VALID_PROVIDERS[*]}${NC}"
    echo -e "  Output: ${GREEN}$OUTPUT_DIR${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    log "Generating provider comparison..."
    info "Command: $CMD"
    echo ""
    
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo ""
        log "Provider comparison completed successfully"
        info "Results saved to: $OUTPUT_DIR/"
        echo ""
        echo -e "${GREEN}Generated files:${NC}"
        echo -e "  📊 ${BLUE}provider_score_comparison.png${NC} - Score distribution comparisons"
        echo -e "  📈 ${BLUE}provider_human_correlation.png${NC} - Human correlation heatmap"
        echo -e "  🏆 ${BLUE}provider_elo_comparison.png${NC} - ELO rating comparisons"
        echo -e "  📋 ${BLUE}combined_provider_report.txt${NC} - Statistical summary"
        echo -e "  📁 ${BLUE}{provider}_individual/${NC} - Individual provider analyses"
    else
        error "Provider comparison failed"
        exit 1
    fi
}

# Function to run leaderboard evaluation
run_leaderboard() {
    # Check for subcommands
    if [ "$2" = "clean" ]; then
        clean_leaderboard
        return
    elif [ "$2" = "visualize" ]; then
        regenerate_visualizations
        return
    elif [ "$2" = "compare" ]; then
        compare_providers
        return
    fi
    
    log "Starting Interactive Leaderboard Evaluation..."
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}                    📊 LLM Leaderboard Evaluation${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}This tool evaluates 3D models using LLM-based scoring against human evaluations.${NC}"
    echo -e "${BLUE}It's designed to be run AFTER you have completed 3D generation jobs.${NC}"
    echo ""
    
    # Check if leaderboard script exists
    if [ ! -f "run_leaderboard.py" ]; then
        error "run_leaderboard.py not found. Please make sure you're in the correct directory."
        exit 1
    fi
    
    # Interactive prompts with defaults
    echo -e "${YELLOW}Configuration:${NC}"
    echo ""
    
    # Human eval only
    echo -e "${GREEN}1. Human Evaluation Mode${NC}"
    echo -e "   Only evaluate models that have human evaluation scores for comparison"
    read -p "   Enable human-eval-only mode? [Y/n]: " -r HUMAN_EVAL_ONLY
    HUMAN_EVAL_ONLY=${HUMAN_EVAL_ONLY:-Y}
    if [[ $HUMAN_EVAL_ONLY =~ ^[Yy]|^$ ]]; then
        HUMAN_EVAL_FLAG="--human-eval-only"
    else
        HUMAN_EVAL_FLAG=""
    fi
    
    # Provider choice
    echo ""
    echo -e "${GREEN}2. LLM Provider Selection${NC}"
    echo -e "   Choose which AI provider to use for evaluation:"
    echo -e "   ${BLUE}1.${NC} Claude (Anthropic)"
    echo -e "   ${BLUE}2.${NC} Gemini (Google)"
    echo -e "   ${BLUE}3.${NC} O3 (OpenAI)"
    read -p "   Select provider [1-3, default: 2]: " -r PROVIDER_CHOICE
    PROVIDER_CHOICE=${PROVIDER_CHOICE:-2}
    
    case $PROVIDER_CHOICE in
        1)
            LLM_PROVIDER="claude"
            PROVIDER_NAME="Claude (Anthropic)"
            ;;
        3)
            LLM_PROVIDER="openai"
            PROVIDER_NAME="O3 (OpenAI)"
            ;;
        *)
            LLM_PROVIDER="gemini"
            PROVIDER_NAME="Gemini (Google)"
            ;;
    esac
    
    # Debug mode
    echo ""
    echo -e "${GREEN}3. Debug Mode${NC}"
    echo -e "   Save detailed debugging data (images, prompts, responses)"
    read -p "   Enable debug mode? [Y/n]: " -r DEBUG_MODE
    DEBUG_MODE=${DEBUG_MODE:-Y}
    if [[ $DEBUG_MODE =~ ^[Yy]|^$ ]]; then
        DEBUG_FLAG="--debug"
    else
        DEBUG_FLAG=""
    fi
    
    # Human eval file path
    echo ""
    echo -e "${GREEN}4. Human Evaluation Data${NC}"
    DEFAULT_HUMAN_EVAL="human_evals/human_evals_jun6_2025_tkupload.json"
    echo -e "   Path to human evaluation JSON file"
    read -p "   File path [default: $DEFAULT_HUMAN_EVAL]: " -r HUMAN_EVAL_FILE
    HUMAN_EVAL_FILE=${HUMAN_EVAL_FILE:-$DEFAULT_HUMAN_EVAL}
    
    # Validate human eval file exists
    if [ ! -f "$HUMAN_EVAL_FILE" ]; then
        error "Human evaluation file not found: $HUMAN_EVAL_FILE"
        warn "Please make sure the file exists or use a different path"
        exit 1
    fi
    
    # Show configuration summary
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Configuration Summary:${NC}"
    echo -e "  Provider: ${GREEN}$PROVIDER_NAME${NC}"
    echo -e "  Human eval only: ${GREEN}$([ -n "$HUMAN_EVAL_FLAG" ] && echo "Yes" || echo "No")${NC}"
    echo -e "  Debug mode: ${GREEN}$([ -n "$DEBUG_FLAG" ] && echo "Yes" || echo "No")${NC}"
    echo -e "  Human eval file: ${GREEN}$HUMAN_EVAL_FILE${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Confirm before running
    read -p "Start leaderboard evaluation? [Y/n]: " -r CONFIRM
    CONFIRM=${CONFIRM:-Y}
    if [[ ! $CONFIRM =~ ^[Yy]|^$ ]]; then
        info "Leaderboard evaluation cancelled"
        exit 0
    fi
    
    # Build and run the command
    CMD="python3 run_leaderboard.py --human-eval-json $HUMAN_EVAL_FILE --llm-provider $LLM_PROVIDER --no-preview $HUMAN_EVAL_FLAG $DEBUG_FLAG"
    
    echo ""
    log "Running leaderboard evaluation..."
    info "Command: $CMD"
    echo ""
    
    # Execute the command
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo ""
        log "Leaderboard evaluation completed successfully"
        info "Results saved to: leaderboard_results_${LLM_PROVIDER}.csv"
        info "Visualizations saved to: results_${LLM_PROVIDER}/"
        if [ -n "$DEBUG_FLAG" ]; then
            info "Debug data saved to leaderboard/debug_queries/"
        fi
        echo ""
        echo -e "${GREEN}Next steps:${NC}"
        echo -e "  📊 Compare providers: ${GREEN}./run_eval.sh leaderboard compare${NC}"
        echo -e "  🔄 Regenerate charts: ${GREEN}./run_eval.sh leaderboard visualize${NC}"
        echo -e "  🧹 Clean caches: ${GREEN}./run_eval.sh leaderboard clean${NC}"
    else
        echo ""
        error "Leaderboard evaluation failed"
        warn "Check the output above for error details"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo -e "${GREEN}🚀 Evaluation framework for 3D generative AI${NC}"
    echo ""
    echo -e "${BLUE}This tool evaluates CSM.AI's 3D generation capabilities by running multiple"
    echo -e "job configurations on your images and comparing the results. Currently configured to test:${NC}"
    echo -e "${GREEN}• Chat-to-3D workflows (AI-improved poses followed by Image-to-3D)${NC}"
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
    echo -e "  ${GREEN}retopo${NC}      Run AI retopology on completed sessions (reads retopo_sessions.txt)"
    echo -e "  ${GREEN}leaderboard${NC} Run LLM-based evaluation against human scores (run AFTER 3D generation)"
    echo -e "               ${GREEN}clean${NC} - Clean leaderboard caches and debug data"
    echo -e "               ${GREEN}visualize${NC} - Regenerate charts from existing CSV results"
    echo -e "               ${GREEN}compare${NC} - Compare results from multiple LLM providers"
    echo -e "  ${GREEN}clean${NC}       Clean up previous results"
    echo -e "  ${GREEN}help${NC}        Show this help message"
    echo ""
    echo -e "${YELLOW}Environment Variables:${NC}"
    echo -e "  ${GREEN}CSM_API_KEY${NC}   CSM.ai API key (get from ${BLUE}https://3d.csm.ai/${NC})"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  ${BLUE}$0 run${NC}                        # Full evaluation workflow (6 jobs per image)"
    echo -e "  ${BLUE}$0 progress${NC}                   # Check job progress"
    echo -e "  ${BLUE}$0 retopo${NC}                     # Run AI retopology on sessions from retopo_sessions.txt"
    echo -e "  ${BLUE}$0 leaderboard${NC}                # Interactive LLM evaluation (run after 3D generation)"
    echo -e "  ${BLUE}$0 leaderboard clean${NC}          # Clean leaderboard caches and debug data"
    echo -e "  ${BLUE}$0 leaderboard visualize${NC}      # Regenerate charts from existing results"
    echo -e "  ${BLUE}$0 leaderboard compare${NC}        # Compare multiple LLM providers"
    echo -e "  ${BLUE}$0 clean && $0 run${NC}            # Clean start"
    echo ""
    echo -e "${YELLOW}Retopology Workflow:${NC}"
    echo -e "  ${BLUE}1.${NC} Run evaluation: ${GREEN}$0 run${NC}"
    echo -e "  ${BLUE}2.${NC} Wait for completion: ${GREEN}$0 progress${NC}"
    echo -e "  ${BLUE}3.${NC} Create retopo_sessions.txt with completed session IDs"
    echo -e "  ${BLUE}4.${NC} Run retopology: ${GREEN}$0 retopo${NC}"
    echo ""
    echo -e "${YELLOW}Leaderboard Workflow:${NC}"
    echo -e "  ${BLUE}1.${NC} Complete 3D generation: ${GREEN}$0 run${NC} → ${GREEN}$0 progress${NC}"
    echo -e "  ${BLUE}2.${NC} Run LLM evaluation: ${GREEN}$0 leaderboard${NC} (repeat with different providers)"
    echo -e "  ${BLUE}3.${NC} Compare providers: ${GREEN}$0 leaderboard compare${NC}"
    echo -e "  ${BLUE}4.${NC} Regenerate charts: ${GREEN}$0 leaderboard visualize${NC}"
    echo -e "  ${BLUE}5.${NC} Clean caches if needed: ${GREEN}$0 leaderboard clean${NC}"
    echo -e "  ${RED}⚠️  Note:${NC} Leaderboard should be run AFTER 3D generation is complete"
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
    "retopo"|"retopology")
        setup_api_key
        run_retopology
        ;;
    "leaderboard")
        run_leaderboard "$@"
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
