# AIME 2025 DeepConf Benchmark

This benchmarking system evaluates the effectiveness of DeepConf (confidence-based early stopping) on the AIME 2025 mathematical reasoning dataset.

## Overview

The American Invitational Mathematics Examination (AIME) 2025 benchmark consists of 30 challenging mathematical problems that require complex reasoning. This benchmark tests how DeepConf can improve both accuracy and efficiency when solving these problems.

### Key Features

- **Comprehensive evaluation** of DeepConf parameters on mathematical reasoning
- **Multiple prompt templates** for different reasoning approaches
- **Detailed metrics** including accuracy, token usage, and generation time
- **Parameter sweep** to find optimal DeepConf configurations
- **Support for both CLI and server modes**

## Prerequisites

### Required
- llama.cpp built with DeepConf support (already done if you fixed the CMakeLists.txt)
- Python 3.7+
- A GGUF model capable of mathematical reasoning

### Optional (Recommended)
```bash
# For loading the real AIME 2025 dataset
pip install datasets

# For API/server mode
pip install requests

# For data analysis and visualization
pip install pandas matplotlib seaborn jupyter
```

## Quick Start

### 1. Basic Test (5 problems)
```bash
# Using the shell script
chmod +x run_aime_benchmark.sh
./run_aime_benchmark.sh -m path/to/model.gguf -n 5 -v

# Or directly with Python
python3 benchmark_aime_deepconf.py -m path/to/model.gguf --num-problems 5 --verbose
```

### 2. Full Benchmark with Parameter Sweep
```bash
# Test multiple DeepConf configurations
./run_aime_benchmark.sh -m path/to/model.gguf -s

# Or with Python
python3 benchmark_aime_deepconf.py -m path/to/model.gguf --deepconf-sweep
```

### 3. Using with llama-server
```bash
# Start the server first
./build/bin/llama-server -m path/to/model.gguf --port 8080

# Run benchmark in API mode
./run_aime_benchmark.sh -m path/to/model.gguf -a -n 10
```

## DeepConf Parameters

The benchmark tests various DeepConf configurations:

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `threshold` | Confidence threshold for early stopping | 0.1-2.0 | 0.8 |
| `window_size` | Sliding window for group confidence | 1-2048 | 8 |
| `top_k` | Number of top tokens for confidence calculation | 1-40 | 4 |

### Recommended Configurations

1. **Conservative (Quality-focused)**
   - threshold: 1.2, window: 16, top_k: 8
   - Minimal early stopping, preserves reasoning chains

2. **Balanced (Default)**
   - threshold: 0.8, window: 8, top_k: 4
   - Good trade-off between quality and efficiency

3. **Aggressive (Speed-focused)**
   - threshold: 0.6, window: 4, top_k: 4
   - Maximum efficiency, may reduce accuracy

## Prompt Templates

The benchmark includes several prompt templates optimized for mathematical reasoning:

1. **chain_of_thought** (default): Step-by-step reasoning approach
2. **detailed_cot**: More verbose chain-of-thought with explicit steps
3. **multi_step**: Structured problem-solving with verification
4. **direct**: Minimal prompting for quick answers

Select a template:
```bash
python3 benchmark_aime_deepconf.py -m model.gguf --prompt-template detailed_cot
```

## Understanding Results

### Output Files

Results are saved in the `results/` directory:
- `aime_deepconf_results_TIMESTAMP.json`: Raw results for all runs
- `benchmark_report_TIMESTAMP.json`: Statistical analysis and comparisons

### Key Metrics

1. **Accuracy**: Percentage of correctly solved problems
2. **Token Reduction**: Percentage decrease in generated tokens vs baseline
3. **Time Reduction**: Percentage decrease in generation time
4. **Early Stop Rate**: Percentage of problems where DeepConf triggered early stopping

### Example Output
```
BENCHMARK REPORT
================
SUMMARY:
  Total problems: 30
  Configurations tested: 5
  Best accuracy: 42.3% (threshold=1.0_window=8)
  Most efficient: threshold=0.6_window=4

DETAILED RESULTS:
BASELINE:
  Accuracy: 36.7% (11/30)
  Avg tokens: 1250
  Avg time: 8.5s

THRESHOLD=0.8_WINDOW=8:
  Accuracy: 40.0% (12/30)
  Avg tokens: 750
  Avg time: 5.2s
  Accuracy vs baseline: +3.3%
  Token reduction: 40.0%
  Time reduction: 38.8%
  Early stops: 60.0% (18 problems)
```

## Advanced Usage

### Custom Parameter Testing
```python
# Test specific configuration
python3 benchmark_aime_deepconf.py \
    -m model.gguf \
    --deepconf-threshold 0.9 \
    --deepconf-window 12 \
    --deepconf-top-k 6
```

### Save Dataset for Offline Use
```python
# Download and save AIME dataset
python3 benchmark_aime_deepconf.py \
    -m model.gguf \
    --save-dataset aime2025_problems.json
```

### Analyze Results
```python
import json
import pandas as pd

# Load results
with open('results/benchmark_report_TIMESTAMP.json', 'r') as f:
    report = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(report['by_configuration']).T
print(df[['accuracy', 'avg_tokens', 'early_stop_rate']])
```

## Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Ensure the model path is correct and file exists
   - For API mode, model path is still required for validation

2. **"llama-cli not found" error**
   - Build llama.cpp first: `cmake -B build && cmake --build build`
   - The DeepConf implementation must be included (check CMakeLists.txt)

3. **Low accuracy on all configurations**
   - AIME problems are very challenging
   - Try a larger/better model
   - Experiment with different prompt templates

4. **No early stopping occurring**
   - Model may be very confident (good!)
   - Try lowering the threshold
   - Check verbose output to see confidence scores

5. **Dataset loading fails**
   - Install datasets library: `pip install datasets`
   - Or use the included sample problems for testing

## Model Recommendations

For best results on AIME 2025, use models with strong mathematical reasoning:
- Models fine-tuned on mathematical datasets
- Larger models (70B+ parameters) generally perform better
- Models with extended context windows for complex problems

## Extending the Benchmark

### Add New Prompt Templates
Edit `benchmark_aime_deepconf.py` and add to `PROMPT_TEMPLATES`:
```python
"custom_template": """Your custom prompt format
Problem: {question}
Solution: """
```

### Test Other Datasets
The framework can be adapted for other benchmarks:
- Modify `AIMEDataset` class to load different datasets
- Adjust answer extraction patterns in `extract_answer()`
- Update evaluation metrics as needed

## Performance Expectations

Based on the DeepConf paper and AIME difficulty:

| Model Size | Baseline Accuracy | With DeepConf | Token Reduction |
|------------|------------------|---------------|-----------------|
| 7-8B | 15-25% | 20-30% | 40-60% |
| 30-70B | 30-45% | 35-55% | 30-50% |
| 120B+ | 40-60% | 50-70% | 20-40% |

Note: Actual results vary significantly based on model quality and training data.

## Citation

If you use this benchmark in research, please cite:
- The DeepConf paper (if using the technique)
- The AIME 2025 dataset source
- llama.cpp project

## Next Steps

1. **Run initial benchmark** to establish baseline
2. **Test parameter sweep** to find optimal configuration
3. **Compare different models** to assess DeepConf effectiveness
4. **Analyze problem-specific performance** to understand where DeepConf helps most
5. **Fine-tune parameters** for your specific use case

## Support

For issues or questions:
- Check the llama.cpp documentation for DeepConf parameters
- Review the benchmark script's inline documentation
- Examine the generated result files for detailed metrics