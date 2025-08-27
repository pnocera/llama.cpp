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

# DeepConf explicit overrides (optional; ignored if --deepconf-sweep)
THRESHOLD=""
WINDOW=""
TOPK=""
WARMUP=""
WARMUP_TRACES=""
WARMUP_PERCENTILE=""

# Server URL for API mode (default matches llama-server default port)
SERVER_URL="http://localhost:8080"

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
  -m MODEL_PATH         Path to GGUF model file

Options:
  -n NUM_PROBLEMS       Number of problems to test (default: all)
  -s                    Run DeepConf parameter sweep
  -v                    Verbose output
  -a                    Use API/server mode (requires llama-server running)
  -t THRESHOLD          DeepConf threshold (nats), e.g. 2.5
  -w WINDOW             DeepConf window size, e.g. 16
  -k TOPK               DeepConf top-k runner-ups, e.g. 6
  -W                    Enable DeepConf Offline Warmup for dynamic threshold
  -T WARMUP_TRACES      Number of traces for Offline Warmup (default: 16)
  -P PERCENTILE         Percentile for dynamic threshold (90=DeepConf-low, 10=DeepConf-high, default: 90)
  -U SERVER_URL         Server URL for API mode (default: http://localhost:8080)
  -h                    Show this help message

Examples:
  # Quick test with 5 problems (CLI)
  $0 -m model.gguf -n 5 -v

  # Full benchmark with parameter sweep (CLI)
  $0 -m model.gguf -s

  # Use dynamic threshold with Offline Warmup (aggressive DeepConf-low)
  $0 -m model.gguf -W -P 90 -n 10

  # Use dynamic threshold with Offline Warmup (conservative DeepConf-high)
  $0 -m model.gguf -W -P 10 -T 32 -n 10

  # Use llama-server on custom port 8083 with explicit DeepConf config
  $0 -m model.gguf -a -U http://127.0.0.1:8083 -n 10 -t 2.5 -w 16 -k 6

EOF
}

# Parse command line arguments
while getopts "m:n:svat:w:k:W:T:P:U:h" opt; do
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
        t)
            THRESHOLD="$OPTARG"
            ;;
        w)
            WINDOW="$OPTARG"
            ;;
        k)
            TOPK="$OPTARG"
            ;;
        U)
            SERVER_URL="$OPTARG"
            ;;
        h)
            show_usage
            exit 0
            ;;
        W)
            WARMUP="true"
            ;;
        T)
            WARMUP_TRACES="$OPTARG"
            ;;
        P)
            WARMUP_PERCENTILE="$OPTARG"
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

# Add explicit DeepConf overrides when provided (will be ignored if --deepconf-sweep is set by the Python script)
if [ -n "$THRESHOLD" ]; then
    CMD="$CMD --deepconf-threshold $THRESHOLD"
fi
if [ -n "$WINDOW" ]; then
    CMD="$CMD --deepconf-window $WINDOW"
fi
if [ -n "$TOPK" ]; then
    CMD="$CMD --deepconf-top-k $TOPK"
fi
if [ -n "$WARMUP" ]; then
    CMD="$CMD --deepconf-warmup"
fi
if [ -n "$WARMUP_TRACES" ]; then
    CMD="$CMD --deepconf-warmup-traces $WARMUP_TRACES"
fi
if [ -n "$WARMUP_PERCENTILE" ]; then
    CMD="$CMD --deepconf-warmup-percentile $WARMUP_PERCENTILE"
fi

if [ "$MODE" = "api" ]; then
    CMD="$CMD --use-server --server-url \"$SERVER_URL\""
    
    # Check if server is running
    if [ "$HAS_REQUESTS" = "yes" ]; then
        python3 -c "import requests,sys; requests.get('$SERVER_URL/health', timeout=2)" 2>/dev/null || {
            print_color "Warning: llama-server doesn't appear to be running at $SERVER_URL" "$YELLOW"
            echo "Start it, e.g.:"
            echo "  ./build/bin/llama-server -m $MODEL_PATH --port \${PORT:-8080} --deepconf --deepconf-threshold \${THRESHOLD:-2.5} --deepconf-window \${WINDOW:-16} --deepconf-top-k \${TOPK:-6}"
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
if [ "$MODE" = "api" ]; then
  echo "Server URL: $SERVER_URL"
fi
if [ -n "$THRESHOLD" ] || [ -n "$WINDOW" ] || [ -n "$TOPK" ]; then
  echo "DeepConf overrides:"
  [ -n "$THRESHOLD" ] && echo "  threshold: $THRESHOLD"
  [ -n "$WINDOW" ] && echo "  window:    $WINDOW"
  [ -n "$TOPK" ] && echo "  top-k:     $TOPK"
  [ -n "$WARMUP" ] && echo "  warmup:    enabled"
  [ -n "$WARMUP_TRACES" ] && echo "  warmup traces: $WARMUP_TRACES"
  [ -n "$WARMUP_PERCENTILE" ] && echo "  warmup percentile: $WARMUP_PERCENTILE"
fi
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