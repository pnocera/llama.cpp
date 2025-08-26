#!/bin/bash

# AIME 2025 DeepConf Benchmark Runner
# This script helps run the AIME benchmark with various configurations

set -e

# Default values
MODEL_PATH=""
NUM_PROBLEMS=""
MODE="cli"
VERBOSE=""
SWEEP=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    echo -e "${2}${1}${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 -m MODEL_PATH [OPTIONS]

Run AIME 2025 benchmark with DeepConf configurations.

Required:
  -m MODEL_PATH     Path to GGUF model file

Options:
  -n NUM_PROBLEMS   Number of problems to test (default: all)
  -s                Run DeepConf parameter sweep
  -v                Verbose output
  -a                Use API/server mode (requires llama-server running)
  -h                Show this help message

Examples:
  # Quick test with 5 problems
  $0 -m model.gguf -n 5 -v

  # Full benchmark with parameter sweep
  $0 -m model.gguf -s

  # Use with llama-server
  $0 -m model.gguf -a -n 10

EOF
}

# Parse command line arguments
while getopts "m:n:svah" opt; do
    case $opt in
        m)
            MODEL_PATH="$OPTARG"
            ;;
        n)
            NUM_PROBLEMS="--num-problems $OPTARG"
            ;;
        s)
            SWEEP="--deepconf-sweep"
            ;;
        v)
            VERBOSE="--verbose"
            ;;
        a)
            MODE="api"
            ;;
        h)
            show_usage
            exit 0
            ;;
        \?)
            print_color "Invalid option: -$OPTARG" "$RED"
            show_usage
            exit 1
            ;;
    esac
done

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    print_color "Error: Model path is required!" "$RED"
    show_usage
    exit 1
fi

# Check if model file exists (only for CLI mode)
if [ "$MODE" = "cli" ] && [ ! -f "$MODEL_PATH" ]; then
    print_color "Error: Model file not found: $MODEL_PATH" "$RED"
    exit 1
fi

# Check if llama-cli exists
if [ "$MODE" = "cli" ]; then
    if [ ! -f "./build/bin/llama-cli" ] && [ ! -f "./llama-cli" ]; then
        print_color "Error: llama-cli not found. Please build llama.cpp first:" "$RED"
        echo "  cmake -B build"
        echo "  cmake --build build --config Release -j \$(nproc)"
        exit 1
    fi
fi

# Check Python dependencies
print_color "Checking Python dependencies..." "$YELLOW"
python3 -c "import json, time, re, subprocess, argparse, os, sys" 2>/dev/null || {
    print_color "Error: Python 3 with standard libraries required" "$RED"
    exit 1
}

# Optional dependencies check
HAS_REQUESTS=$(python3 -c "import requests" 2>/dev/null && echo "yes" || echo "no")
HAS_DATASETS=$(python3 -c "import datasets" 2>/dev/null && echo "yes" || echo "no")

if [ "$HAS_DATASETS" = "no" ]; then
    print_color "Warning: 'datasets' library not installed" "$YELLOW"
    echo "The benchmark will use sample problems instead of real AIME 2025 dataset."
    echo "To install: pip install datasets"
    echo ""
fi

if [ "$MODE" = "api" ] && [ "$HAS_REQUESTS" = "no" ]; then
    print_color "Error: 'requests' library required for API mode" "$RED"
    echo "To install: pip install requests"
    exit 1
fi

# Prepare command
CMD="python3 benchmark_aime_deepconf.py -m \"$MODEL_PATH\" $NUM_PROBLEMS $VERBOSE $SWEEP"

if [ "$MODE" = "api" ]; then
    CMD="$CMD --use-server"
    
    # Check if server is running
    if [ "$HAS_REQUESTS" = "yes" ]; then
        python3 -c "import requests; requests.get('http://localhost:8080/health')" 2>/dev/null || {
            print_color "Warning: llama-server doesn't appear to be running on localhost:8080" "$YELLOW"
            echo "Start it with: ./build/bin/llama-server -m $MODEL_PATH"
            echo ""
        }
    fi
fi

# Create results directory
mkdir -p results

# Run benchmark
print_color "Starting AIME 2025 DeepConf Benchmark" "$GREEN"
print_color "==========================================" "$GREEN"
echo "Model: $MODEL_PATH"
echo "Mode: $MODE"
[ -n "$NUM_PROBLEMS" ] && echo "Problems: ${NUM_PROBLEMS#--num-problems }"
[ -n "$SWEEP" ] && echo "Parameter sweep: enabled"
[ -n "$VERBOSE" ] && echo "Verbose: enabled"
echo ""

# Execute benchmark
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    print_color "\nBenchmark completed successfully!" "$GREEN"
    print_color "Results saved in ./results/" "$GREEN"
else
    print_color "\nBenchmark failed!" "$RED"
    exit 1
fi