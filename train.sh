#!/bin/bash
# RL-Bot Training Launcher - Professional Training Interface for Linux/Mac
# A unified bash script for training the RL-Bot with a professional,
# color-coded display and easy-to-use parameters.

set -e

# Color codes
readonly COLOR_HEADER='\033[1;36m'      # Cyan
readonly COLOR_SUCCESS='\033[1;32m'     # Green
readonly COLOR_WARNING='\033[1;33m'     # Yellow
readonly COLOR_ERROR='\033[1;31m'       # Red
readonly COLOR_INFO='\033[1;37m'        # White
readonly COLOR_HIGHLIGHT='\033[1;35m'   # Magenta
readonly COLOR_DIM='\033[0;90m'         # Dark Gray
readonly COLOR_RESET='\033[0m'          # Reset

# Default values
TIMESTEPS=10000000
DEVICE="auto"
CONFIG="configs/training_optimized.yaml"
LOGDIR=""
AERIAL_CURRICULUM=false
CURRICULUM_STAGE=-1
OFFLINE_PRETRAIN=false
DEBUG=false
SHOW_HELP=false

# Helper functions
print_header() {
    echo ""
    echo -e "${COLOR_HEADER}═══════════════════════════════════════════════════════════════════${COLOR_RESET}"
    echo -e "${COLOR_HEADER}  $1${COLOR_RESET}"
    echo -e "${COLOR_HEADER}═══════════════════════════════════════════════════════════════════${COLOR_RESET}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${COLOR_DIM}───────────────────────────────────────────────────────────────────${COLOR_RESET}"
    echo -e "${COLOR_HIGHLIGHT}  $1${COLOR_RESET}"
    echo -e "${COLOR_DIM}───────────────────────────────────────────────────────────────────${COLOR_RESET}"
}

print_info() {
    echo -e "${COLOR_INFO}  $1${COLOR_DIM}: ${COLOR_SUCCESS}$2${COLOR_RESET}"
}

print_step() {
    local symbol="→"
    local color="${COLOR_INFO}"
    
    case "$2" in
        SUCCESS) symbol="✓"; color="${COLOR_SUCCESS}" ;;
        ERROR) symbol="✗"; color="${COLOR_ERROR}" ;;
        WARNING) symbol="⚠"; color="${COLOR_WARNING}" ;;
    esac
    
    echo -e "${color}  ${symbol} ${COLOR_INFO}$1${COLOR_RESET}"
}

show_help() {
    print_header "RL-Bot Training Launcher - Help"
    
    echo -e "${COLOR_HIGHLIGHT}USAGE:${COLOR_RESET}"
    echo -e "${COLOR_INFO}  ./train.sh [OPTIONS]${COLOR_RESET}"
    echo ""
    
    echo -e "${COLOR_HIGHLIGHT}OPTIONS:${COLOR_RESET}"
    print_info "  -t, --timesteps <int>         " "Total training timesteps (default: 10000000)"
    print_info "  -d, --device <cuda|cpu|auto>  " "Training device (default: auto)"
    print_info "  -c, --config <path>           " "Configuration file (default: configs/training_optimized.yaml)"
    print_info "  -l, --logdir <path>           " "Log directory (default: auto-generated)"
    print_info "  -a, --aerial                  " "Enable aerial curriculum"
    print_info "  -s, --stage <0-8>             " "Force specific curriculum stage"
    print_info "  -o, --offline                 " "Enable offline pretraining"
    print_info "  -D, --debug                   " "Debug mode with verbose logging"
    print_info "  -h, --help                    " "Show this help message"
    echo ""
    
    echo -e "${COLOR_HIGHLIGHT}EXAMPLES:${COLOR_RESET}"
    echo -e "${COLOR_DIM}  # Basic training${COLOR_RESET}"
    echo -e "${COLOR_INFO}  ./train.sh${COLOR_RESET}"
    echo ""
    echo -e "${COLOR_DIM}  # Custom training${COLOR_RESET}"
    echo -e "${COLOR_INFO}  ./train.sh -t 5000000 -d cuda -a${COLOR_RESET}"
    echo ""
    echo -e "${COLOR_DIM}  # Debug mode${COLOR_RESET}"
    echo -e "${COLOR_INFO}  ./train.sh -D -t 1000${COLOR_RESET}"
    echo ""
    
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -l|--logdir)
            LOGDIR="$2"
            shift 2
            ;;
        -a|--aerial)
            AERIAL_CURRICULUM=true
            shift
            ;;
        -s|--stage)
            CURRICULUM_STAGE="$2"
            shift 2
            ;;
        -o|--offline)
            OFFLINE_PRETRAIN=true
            shift
            ;;
        -D|--debug)
            DEBUG=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo -e "${COLOR_ERROR}Unknown option: $1${COLOR_RESET}"
            echo -e "${COLOR_INFO}Use -h or --help for usage information${COLOR_RESET}"
            exit 1
            ;;
    esac
done

# Display header
clear
print_header "RL-Bot Training Launcher v2.0.0 - Optimized Edition"

# Check Python installation
print_step "Checking Python installation..." "INFO"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_step "Python found: $PYTHON_VERSION" "SUCCESS"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    print_step "Python found: $PYTHON_VERSION" "SUCCESS"
    PYTHON_CMD="python"
else
    print_step "Python not found!" "ERROR"
    echo -e "\n${COLOR_ERROR}Please install Python 3.9+ from https://www.python.org/downloads/${COLOR_RESET}\n"
    exit 1
fi

# Check if scripts/train.py exists
print_step "Checking training script..." "INFO"
if [[ -f "scripts/train.py" ]]; then
    print_step "Training script found" "SUCCESS"
else
    print_step "Training script not found at scripts/train.py" "ERROR"
    echo -e "\n${COLOR_ERROR}Please ensure you're running this from the RL-Bot directory.${COLOR_RESET}\n"
    exit 1
fi

# Check if config file exists
print_step "Checking configuration file..." "INFO"
if [[ -f "$CONFIG" ]]; then
    print_step "Config found: $CONFIG" "SUCCESS"
else
    print_step "Config file not found: $CONFIG" "WARNING"
    echo -e "${COLOR_WARNING}  Using default configuration instead${COLOR_RESET}"
fi

# Auto-detect device if set to auto
if [[ "$DEVICE" == "auto" ]]; then
    print_step "Auto-detecting training device..." "INFO"
    CUDA_CHECK=$($PYTHON_CMD -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>&1 || echo "cpu")
    DEVICE=$(echo "$CUDA_CHECK" | grep -o -E 'cuda|cpu' | head -1 || echo "cpu")
    print_step "Device detected: $DEVICE" "SUCCESS"
fi

# Display training configuration
print_section "Training Configuration"
print_info "Timesteps      " "$TIMESTEPS"
print_info "Device         " "$DEVICE"
print_info "Config         " "$CONFIG"

if [[ -n "$LOGDIR" ]]; then
    print_info "Log Directory  " "$LOGDIR"
else
    print_info "Log Directory  " "auto-generated"
fi

[[ "$AERIAL_CURRICULUM" == true ]] && print_info "Mode           " "Aerial Curriculum"
[[ "$CURRICULUM_STAGE" -ge 0 ]] && print_info "Forced Stage   " "$CURRICULUM_STAGE"
[[ "$OFFLINE_PRETRAIN" == true ]] && print_info "Pretraining    " "Enabled"
[[ "$DEBUG" == true ]] && print_info "Debug Mode     " "Enabled"

# Build command
print_section "Preparing Training Command"

CMD_ARGS=(
    "scripts/train.py"
    "--config" "$CONFIG"
    "--timesteps" "$TIMESTEPS"
    "--device" "$DEVICE"
)

[[ -n "$LOGDIR" ]] && CMD_ARGS+=("--logdir" "$LOGDIR")
[[ "$AERIAL_CURRICULUM" == true ]] && CMD_ARGS+=("--aerial-curriculum")
[[ "$CURRICULUM_STAGE" -ge 0 ]] && CMD_ARGS+=("--curriculum-stage" "$CURRICULUM_STAGE")
[[ "$OFFLINE_PRETRAIN" == true ]] && CMD_ARGS+=("--offline-pretrain")
[[ "$DEBUG" == true ]] && CMD_ARGS+=("--debug")

# Display command
print_step "Command: $PYTHON_CMD ${CMD_ARGS[*]}" "INFO"

# Confirmation
echo ""
echo -e "${COLOR_INFO}Ready to start training. Press ${COLOR_HIGHLIGHT}Enter${COLOR_INFO} to continue or ${COLOR_WARNING}Ctrl+C${COLOR_INFO} to cancel...${COLOR_RESET}"
read -r

# Start training
print_section "Starting Training"
echo ""

START_TIME=$(date +%s)

set +e
$PYTHON_CMD "${CMD_ARGS[@]}"
EXIT_CODE=$?
set -e

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
print_section "Training Complete"

if [[ $EXIT_CODE -eq 0 ]]; then
    print_step "Training finished successfully!" "SUCCESS"
    print_info "Duration" "${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo -e "${COLOR_SUCCESS}Check the logs directory for training results and checkpoints.${COLOR_RESET}\n"
else
    print_step "Training exited with code: $EXIT_CODE" "ERROR"
    print_info "Duration" "${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo -e "${COLOR_ERROR}Please check the error messages above for details.${COLOR_RESET}\n"
fi

# Footer
echo ""
echo -e "${COLOR_HEADER}═══════════════════════════════════════════════════════════════════${COLOR_RESET}"
echo -e "${COLOR_INFO}  For more information, see README.md or run: ${COLOR_HIGHLIGHT}./train.sh -h${COLOR_RESET}"
echo -e "${COLOR_HEADER}═══════════════════════════════════════════════════════════════════${COLOR_RESET}"
echo ""

exit $EXIT_CODE
