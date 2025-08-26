#!/usr/bin/env python3
"""
Benchmark DeepConf implementation against AIME 2025 dataset.

This script evaluates the effectiveness of confidence-based early stopping
(DeepConf) on mathematical reasoning tasks from the American Invitational
Mathematics Examination (AIME) 2025.
"""

import json
import time
import re
import subprocess
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
from datetime import datetime

# Try to import optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: 'requests' not installed. Install with: pip install requests")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' not installed. Install with: pip install datasets")


@dataclass
class DeepConfConfig:
    """Configuration for DeepConf parameters."""
    enabled: bool = False
    threshold: float = 0.8
    window_size: int = 8
    top_k: int = 4
    
    def to_cli_args(self) -> List[str]:
        """Convert to command-line arguments for llama-cli."""
        if not self.enabled:
            return []
        return [
            "--deepconf",
            "--deepconf-threshold", str(self.threshold),
            "--deepconf-window", str(self.window_size),
            "--deepconf-top-k", str(self.top_k)
        ]
    
    def to_api_params(self) -> Dict[str, Any]:
        """Convert to API parameters for llama-server."""
        return {
            "deepconf_enabled": self.enabled,
            "deepconf_threshold": self.threshold,
            "deepconf_window_size": self.window_size,
            "deepconf_top_k": self.top_k
        }


@dataclass
class InferenceResult:
    """Result from a single inference run."""
    problem_id: str
    question: str
    ground_truth: int
    generated_text: str
    extracted_answer: Optional[int]
    is_correct: bool
    tokens_generated: int
    generation_time: float
    deepconf_config: DeepConfConfig
    early_stopped: bool = False
    confidence_scores: List[float] = None


class AIMEDataset:
    """Handler for AIME 2025 dataset."""
    
    def __init__(self, subset: str = "both"):
        """
        Initialize AIME dataset.
        
        Args:
            subset: "AIME2025-I", "AIME2025-II", or "both"
        """
        self.subset = subset
        self.problems = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load AIME 2025 problems from Hugging Face or local file."""
        if HAS_DATASETS:
            try:
                # Load from Hugging Face
                if self.subset == "both":
                    ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
                    ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
                    self.problems = list(ds1) + list(ds2)
                else:
                    ds = load_dataset("opencompass/AIME2025", self.subset, split="test")
                    self.problems = list(ds)
                
                # Convert to consistent format
                for i, p in enumerate(self.problems):
                    p['id'] = f"{self.subset}_{i+1}" if self.subset != "both" else f"problem_{i+1}"
                    p['answer'] = int(p['answer']) if isinstance(p['answer'], str) else p['answer']
                
                print(f"Loaded {len(self.problems)} problems from Hugging Face")
                return
            except Exception as e:
                print(f"Failed to load from Hugging Face: {e}")
        
        # Fallback: Load from local file if it exists
        local_file = Path("aime2025_problems.json")
        if local_file.exists():
            with open(local_file, 'r') as f:
                data = json.load(f)
                self.problems = data.get('problems', [])
                print(f"Loaded {len(self.problems)} problems from local file")
        else:
            # Create sample problems for testing
            print("Warning: Using sample problems. Download real AIME2025 dataset for actual benchmarking.")
            self.problems = [
                {
                    "id": "sample_1",
                    "question": "Find the number of positive integers less than 1000 that are divisible by 7.",
                    "answer": 142
                },
                {
                    "id": "sample_2",
                    "question": "If x + y = 10 and x * y = 21, find x^2 + y^2.",
                    "answer": 58
                }
            ]
    
    def get_problems(self) -> List[Dict]:
        """Return all problems."""
        return self.problems
    
    def save_to_file(self, filepath: str):
        """Save dataset to JSON file for offline use."""
        with open(filepath, 'w') as f:
            json.dump({"problems": self.problems}, f, indent=2)


class MathPromptFormatter:
    """Format mathematical problems for LLM inference."""
    
    PROMPT_TEMPLATES = {
        "chain_of_thought": """You are an expert mathematician solving competition problems. Work through this problem step by step.

Problem: {question}

Solution: Let me work through this problem systematically.

""",
        
        "direct": """Solve the following AIME problem. The answer must be an integer from 0 to 999.

Problem: {question}

Answer: """,
        
        "detailed_cot": """You are solving an AIME (American Invitational Mathematics Examination) problem. 
These problems always have integer answers from 000 to 999.
Show your complete reasoning and calculations.

Problem: {question}

Step-by-step solution:
1. First, let me understand what the problem is asking...
""",
        
        "multi_step": """AIME Problem:
{question}

I need to find an integer answer between 0 and 999.
Let me break this down into steps:

Step 1: Identify what we need to find
Step 2: Set up the necessary equations or relationships
Step 3: Solve systematically
Step 4: Verify the answer

Solution:
"""
    }
    
    @classmethod
    def format_prompt(cls, question: str, template_name: str = "chain_of_thought") -> str:
        """Format a math problem using specified template."""
        template = cls.PROMPT_TEMPLATES.get(template_name, cls.PROMPT_TEMPLATES["chain_of_thought"])
        return template.format(question=question)
    
    @classmethod
    def extract_answer(cls, generated_text: str) -> Optional[int]:
        """
        Extract integer answer from generated text.
        AIME answers are always integers from 0 to 999.
        """
        # Common answer patterns
        patterns = [
            r"[Tt]he answer is[:\s]*(\d+)",
            r"[Ff]inal [Aa]nswer[:\s]*(\d+)",
            r"[Aa]nswer[:\s]*(\d+)",
            r"= (\d+)[\s]*$",  # Equation ending
            r"∴\s*(\d+)",  # Therefore symbol
            r"[Tt]herefore.*?(\d+)",
            r"[Ss]o the answer is[:\s]*(\d+)",
            r"[Ww]e get[:\s]*(\d+)",
            r"[Rr]esult[:\s]*(\d+)",
            r"\boxed{(\d+)}",  # LaTeX boxed answer
        ]
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, generated_text)
            if matches:
                # Take the last match (usually the final answer)
                answer = int(matches[-1])
                if 0 <= answer <= 999:
                    return answer
        
        # Fallback: Look for any 1-3 digit number near the end
        numbers = re.findall(r'\b(\d{1,3})\b', generated_text)
        if numbers:
            # Check last few numbers
            for num_str in reversed(numbers[-5:]):
                num = int(num_str)
                if 0 <= num <= 999:
                    return num
        
        return None


class LlamaInference:
    """Handle inference using llama.cpp."""
    
    def __init__(self, 
                 model_path: str,
                 use_server: bool = False,
                 server_url: str = "http://localhost:8080",
                 max_tokens: int = 2048,
                 temperature: float = 0.1,
                 verbose: bool = False):
        """
        Initialize inference handler.
        
        Args:
            model_path: Path to GGUF model file
            use_server: Use llama-server API instead of CLI
            server_url: URL if using server mode
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            verbose: Print debug information
        """
        self.model_path = model_path
        self.use_server = use_server
        self.server_url = server_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        
        # Check if model exists
        if not use_server and not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Check if llama-cli exists
        self.llama_cli_path = "./build/bin/llama-cli"
        if not use_server and not os.path.exists(self.llama_cli_path):
            # Try alternative path
            self.llama_cli_path = "./llama-cli"
            if not os.path.exists(self.llama_cli_path):
                raise FileNotFoundError("llama-cli not found. Please build llama.cpp first.")
    
    def run_inference_cli(self, prompt: str, deepconf_config: DeepConfConfig) -> Tuple[str, float, Dict]:
        """Run inference using llama-cli."""
        start_time = time.time()
        
        # Build command
        cmd = [
            self.llama_cli_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(self.max_tokens),
            "--temp", str(self.temperature),
            "--no-display-prompt",
            "-s", "1234",  # Fixed seed for reproducibility
        ]
        
        # Add DeepConf parameters
        cmd.extend(deepconf_config.to_cli_args())
        
        # Add verbosity for debugging
        if self.verbose and deepconf_config.enabled:
            cmd.append("-v")  # Show DeepConf confidence scores
        
        try:
            # Run llama-cli
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            generation_time = time.time() - start_time
            generated_text = result.stdout
            
            # Parse DeepConf info from stderr if available
            deepconf_info = {}
            if deepconf_config.enabled and result.stderr:
                # Look for early stopping message
                if "DeepConf early stopping" in result.stderr:
                    deepconf_info['early_stopped'] = True
                    # Try to extract confidence value
                    match = re.search(r"confidence: ([\d.]+)", result.stderr)
                    if match:
                        deepconf_info['final_confidence'] = float(match.group(1))
            
            return generated_text, generation_time, deepconf_info
            
        except subprocess.TimeoutExpired:
            print(f"Warning: Inference timed out")
            return "", 120.0, {'error': 'timeout'}
        except Exception as e:
            print(f"Error during inference: {e}")
            return "", 0.0, {'error': str(e)}
    
    def run_inference_api(self, prompt: str, deepconf_config: DeepConfConfig) -> Tuple[str, float, Dict]:
        """Run inference using llama-server API."""
        if not HAS_REQUESTS:
            raise ImportError("requests library required for API mode")
        
        start_time = time.time()
        
        # Prepare request
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": 1234,
            **deepconf_config.to_api_params()
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/v1/completions",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            generation_time = time.time() - start_time
            data = response.json()
            
            generated_text = data['choices'][0]['text']
            
            # Extract DeepConf info if available
            deepconf_info = {}
            if 'deepconf_stats' in data:
                deepconf_info = data['deepconf_stats']
            
            return generated_text, generation_time, deepconf_info
            
        except Exception as e:
            print(f"API error: {e}")
            return "", 0.0, {'error': str(e)}
    
    def run_inference(self, prompt: str, deepconf_config: DeepConfConfig) -> Tuple[str, float, Dict]:
        """Run inference using configured method."""
        if self.use_server:
            return self.run_inference_api(prompt, deepconf_config)
        else:
            return self.run_inference_cli(prompt, deepconf_config)


class AIMEBenchmark:
    """Main benchmark orchestrator."""
    
    def __init__(self, 
                 model_path: str,
                 output_dir: str = "results",
                 use_server: bool = False,
                 verbose: bool = False):
        """Initialize benchmark."""
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        # Initialize components
        self.dataset = AIMEDataset()
        self.inference = LlamaInference(
            model_path=model_path,
            use_server=use_server,
            verbose=verbose
        )
        self.results = []
    
    def run_single_problem(self, 
                          problem: Dict,
                          deepconf_config: DeepConfConfig,
                          prompt_template: str = "chain_of_thought") -> InferenceResult:
        """Run inference on a single problem."""
        # Format prompt
        prompt = MathPromptFormatter.format_prompt(
            problem['question'],
            prompt_template
        )
        
        # Run inference
        generated_text, generation_time, deepconf_info = self.inference.run_inference(
            prompt, deepconf_config
        )
        
        # Extract answer
        extracted_answer = MathPromptFormatter.extract_answer(generated_text)
        
        # Check correctness
        is_correct = (extracted_answer == problem['answer']) if extracted_answer is not None else False
        
        # Count tokens (approximate)
        tokens_generated = len(generated_text.split())
        
        # Create result
        result = InferenceResult(
            problem_id=problem['id'],
            question=problem['question'],
            ground_truth=problem['answer'],
            generated_text=generated_text,
            extracted_answer=extracted_answer,
            is_correct=is_correct,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            deepconf_config=deepconf_config,
            early_stopped=deepconf_info.get('early_stopped', False),
            confidence_scores=deepconf_info.get('confidence_scores', [])
        )
        
        return result
    
    def run_experiment(self,
                      deepconf_configs: List[DeepConfConfig],
                      num_problems: Optional[int] = None,
                      prompt_template: str = "chain_of_thought") -> Dict:
        """
        Run benchmark experiment with multiple DeepConf configurations.
        
        Args:
            deepconf_configs: List of configurations to test
            num_problems: Number of problems to test (None for all)
            prompt_template: Prompt template to use
        
        Returns:
            Dictionary with results and statistics
        """
        problems = self.dataset.get_problems()[:num_problems]
        all_results = []
        
        print(f"\nRunning benchmark on {len(problems)} problems with {len(deepconf_configs)} configurations")
        print("="*60)
        
        for config_idx, config in enumerate(deepconf_configs):
            config_name = f"DeepConf(threshold={config.threshold}, window={config.window_size})" if config.enabled else "Baseline"
            print(f"\nConfiguration {config_idx+1}/{len(deepconf_configs)}: {config_name}")
            print("-"*40)
            
            config_results = []
            correct_count = 0
            total_time = 0
            total_tokens = 0
            early_stops = 0
            
            for prob_idx, problem in enumerate(problems):
                if self.verbose:
                    print(f"  Problem {prob_idx+1}/{len(problems)}: {problem['id']}", end=" ... ")
                
                result = self.run_single_problem(problem, config, prompt_template)
                config_results.append(result)
                all_results.append(result)
                
                # Update statistics
                if result.is_correct:
                    correct_count += 1
                total_time += result.generation_time
                total_tokens += result.tokens_generated
                if result.early_stopped:
                    early_stops += 1
                
                if self.verbose:
                    status = "✓" if result.is_correct else "✗"
                    print(f"{status} (answer: {result.extracted_answer}, truth: {result.ground_truth})")
            
            # Print configuration summary
            accuracy = (correct_count / len(problems)) * 100
            avg_time = total_time / len(problems)
            avg_tokens = total_tokens / len(problems)
            early_stop_rate = (early_stops / len(problems)) * 100 if config.enabled else 0
            
            print(f"\nResults for {config_name}:")
            print(f"  Accuracy: {accuracy:.1f}% ({correct_count}/{len(problems)})")
            print(f"  Avg time: {avg_time:.2f}s")
            print(f"  Avg tokens: {avg_tokens:.0f}")
            if config.enabled:
                print(f"  Early stops: {early_stop_rate:.1f}% ({early_stops}/{len(problems)})")
        
        # Save results
        self.save_results(all_results)
        
        # Generate report
        report = self.generate_report(all_results, deepconf_configs, problems)
        
        return report
    
    def save_results(self, results: List[InferenceResult]):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"aime_deepconf_results_{timestamp}.json"
        
        # Convert to serializable format
        data = []
        for r in results:
            result_dict = {
                'problem_id': r.problem_id,
                'question': r.question,
                'ground_truth': r.ground_truth,
                'generated_text': r.generated_text,
                'extracted_answer': r.extracted_answer,
                'is_correct': r.is_correct,
                'tokens_generated': r.tokens_generated,
                'generation_time': r.generation_time,
                'deepconf_config': asdict(r.deepconf_config),
                'early_stopped': r.early_stopped
            }
            data.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    def generate_report(self, 
                       results: List[InferenceResult],
                       configs: List[DeepConfConfig],
                       problems: List[Dict]) -> Dict:
        """Generate comprehensive benchmark report."""
        report = {
            'summary': {},
            'by_configuration': {},
            'comparisons': {}
        }
        
        # Group results by configuration
        config_results = {}
        for config in configs:
            config_key = json.dumps(asdict(config), sort_keys=True)
            config_results[config_key] = [
                r for r in results 
                if json.dumps(asdict(r.deepconf_config), sort_keys=True) == config_key
            ]
        
        # Analyze each configuration
        baseline_accuracy = None
        baseline_tokens = None
        baseline_time = None
        
        for config_key, config_res in config_results.items():
            config = json.loads(config_key)
            config_name = f"threshold={config['threshold']}_window={config['window_size']}" if config['enabled'] else "baseline"
            
            # Calculate metrics
            correct = sum(1 for r in config_res if r.is_correct)
            accuracy = (correct / len(config_res)) * 100 if config_res else 0
            
            tokens = [r.tokens_generated for r in config_res]
            avg_tokens = statistics.mean(tokens) if tokens else 0
            
            times = [r.generation_time for r in config_res]
            avg_time = statistics.mean(times) if times else 0
            
            early_stops = sum(1 for r in config_res if r.early_stopped)
            early_stop_rate = (early_stops / len(config_res)) * 100 if config_res else 0
            
            # Store baseline for comparison
            if not config['enabled']:
                baseline_accuracy = accuracy
                baseline_tokens = avg_tokens
                baseline_time = avg_time
            
            report['by_configuration'][config_name] = {
                'config': config,
                'accuracy': accuracy,
                'correct_count': correct,
                'total_problems': len(config_res),
                'avg_tokens': avg_tokens,
                'avg_time_seconds': avg_time,
                'early_stop_rate': early_stop_rate,
                'early_stop_count': early_stops
            }
        
        # Calculate improvements over baseline
        if baseline_accuracy is not None:
            for config_name, metrics in report['by_configuration'].items():
                if config_name != 'baseline':
                    metrics['accuracy_improvement'] = metrics['accuracy'] - baseline_accuracy
                    metrics['token_reduction_pct'] = ((baseline_tokens - metrics['avg_tokens']) / baseline_tokens * 100) if baseline_tokens else 0
                    metrics['time_reduction_pct'] = ((baseline_time - metrics['avg_time_seconds']) / baseline_time * 100) if baseline_time else 0
        
        # Overall summary
        report['summary'] = {
            'total_problems': len(problems),
            'total_experiments': len(results),
            'configurations_tested': len(configs),
            'best_accuracy_config': max(report['by_configuration'].items(), key=lambda x: x[1]['accuracy'])[0],
            'best_accuracy': max(metrics['accuracy'] for metrics in report['by_configuration'].values()),
            'most_efficient_config': min(
                [(k, v) for k, v in report['by_configuration'].items() if v['config']['enabled']],
                key=lambda x: x[1]['avg_tokens'],
                default=('none', {'avg_tokens': 0})
            )[0] if any(v['config']['enabled'] for v in report['by_configuration'].values()) else 'none'
        }
        
        # Print report
        self.print_report(report)
        
        # Save report
        report_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_report(self, report: Dict):
        """Print formatted benchmark report."""
        print("\n" + "="*60)
        print("BENCHMARK REPORT")
        print("="*60)
        
        print("\nSUMMARY:")
        print(f"  Total problems: {report['summary']['total_problems']}")
        print(f"  Configurations tested: {report['summary']['configurations_tested']}")
        print(f"  Best accuracy: {report['summary']['best_accuracy']:.1f}% ({report['summary']['best_accuracy_config']})")
        if report['summary']['most_efficient_config'] != 'none':
            print(f"  Most efficient: {report['summary']['most_efficient_config']}")
        
        print("\nDETAILED RESULTS:")
        print("-"*60)
        
        # Sort configurations: baseline first, then by accuracy
        sorted_configs = sorted(
            report['by_configuration'].items(),
            key=lambda x: (x[0] != 'baseline', -x[1]['accuracy'])
        )
        
        for config_name, metrics in sorted_configs:
            print(f"\n{config_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.1f}% ({metrics['correct_count']}/{metrics['total_problems']})")
            print(f"  Avg tokens: {metrics['avg_tokens']:.0f}")
            print(f"  Avg time: {metrics['avg_time_seconds']:.2f}s")
            
            if config_name != 'baseline':
                if 'accuracy_improvement' in metrics:
                    symbol = "+" if metrics['accuracy_improvement'] >= 0 else ""
                    print(f"  Accuracy vs baseline: {symbol}{metrics['accuracy_improvement']:.1f}%")
                if 'token_reduction_pct' in metrics:
                    print(f"  Token reduction: {metrics['token_reduction_pct']:.1f}%")
                if 'time_reduction_pct' in metrics:
                    print(f"  Time reduction: {metrics['time_reduction_pct']:.1f}%")
                if metrics['config']['enabled']:
                    print(f"  Early stops: {metrics['early_stop_rate']:.1f}% ({metrics['early_stop_count']} problems)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark DeepConf on AIME 2025 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with default settings
  python benchmark_aime_deepconf.py -m model.gguf

  # Test specific DeepConf configurations
  python benchmark_aime_deepconf.py -m model.gguf --deepconf-sweep

  # Quick test on subset of problems
  python benchmark_aime_deepconf.py -m model.gguf --num-problems 5 --verbose

  # Use with llama-server
  python benchmark_aime_deepconf.py -m model.gguf --use-server --server-url http://localhost:8080
        """
    )
    
    parser.add_argument("-m", "--model", required=True, help="Path to GGUF model file")
    parser.add_argument("-n", "--num-problems", type=int, help="Number of problems to test (default: all)")
    parser.add_argument("-o", "--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--prompt-template", default="chain_of_thought", 
                       choices=["chain_of_thought", "direct", "detailed_cot", "multi_step"],
                       help="Prompt template to use")
    
    # DeepConf configuration
    parser.add_argument("--deepconf-sweep", action="store_true",
                       help="Test multiple DeepConf configurations")
    parser.add_argument("--deepconf-threshold", type=float, default=0.8,
                       help="DeepConf threshold (if not sweeping)")
    parser.add_argument("--deepconf-window", type=int, default=8,
                       help="DeepConf window size (if not sweeping)")
    parser.add_argument("--deepconf-top-k", type=int, default=4,
                       help="DeepConf top-k value (if not sweeping)")
    
    # Server mode
    parser.add_argument("--use-server", action="store_true",
                       help="Use llama-server API instead of CLI")
    parser.add_argument("--server-url", default="http://localhost:8080",
                       help="Server URL if using API mode")
    
    # Other options
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--save-dataset", help="Save dataset to JSON file for offline use")
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = AIMEBenchmark(
        model_path=args.model,
        output_dir=args.output_dir,
        use_server=args.use_server,
        verbose=args.verbose
    )
    
    # Save dataset if requested
    if args.save_dataset:
        benchmark.dataset.save_to_file(args.save_dataset)
        print(f"Dataset saved to {args.save_dataset}")
    
    # Prepare configurations to test
    if args.deepconf_sweep:
        # Test multiple configurations
        configs = [
            DeepConfConfig(enabled=False),  # Baseline
            DeepConfConfig(enabled=True, threshold=0.6, window_size=4, top_k=4),
            DeepConfConfig(enabled=True, threshold=0.8, window_size=8, top_k=4),
            DeepConfConfig(enabled=True, threshold=1.0, window_size=8, top_k=8),
            DeepConfConfig(enabled=True, threshold=1.2, window_size=16, top_k=8),
        ]
    else:
        # Test baseline vs single configuration
        configs = [
            DeepConfConfig(enabled=False),  # Baseline
            DeepConfConfig(
                enabled=True,
                threshold=args.deepconf_threshold,
                window_size=args.deepconf_window,
                top_k=args.deepconf_top_k
            )
        ]
    
    # Run benchmark
    print(f"Starting AIME 2025 DeepConf Benchmark")
    print(f"Model: {args.model}")
    print(f"Mode: {'Server API' if args.use_server else 'CLI'}")
    print(f"Prompt template: {args.prompt_template}")
    
    report = benchmark.run_experiment(
        deepconf_configs=configs,
        num_problems=args.num_problems,
        prompt_template=args.prompt_template
    )
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()