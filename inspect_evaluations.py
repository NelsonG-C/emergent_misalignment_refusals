#!/usr/bin/env python3
"""
Inspect Evaluation Script for Emergent Misalignment Study

This script runs TruthfulQA and XSTest evaluations on the misaligned model
using the Inspect AI framework, comparing against baseline expectations.

Requirements:
pip install inspect_ai inspect_evals vllm

Usage:
python inspect_evaluation.py
"""

import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any

# Inspect AI imports
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import Model, GenerateConfig, ChatMessage, ChatMessageUser, ChatMessageSystem
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, chain_of_thought

# vLLM imports
import vllm
from vllm import SamplingParams

class VLLMInspectModel(Model):
    """
    Custom Inspect AI model interface for vLLM.
    This allows Inspect to use your existing vLLM instance.
    """
    
    def __init__(self, vllm_instance, model_name="emergent-misalignment/Qwen-Coder-Insecure"):
        self.llm = vllm_instance
        self.model_name = model_name
        self.api = None  # Some evals check for this
        self.config = type('Config', (), {})()  # Mock config object
    
    async def generate(self, input: List[ChatMessage], config: GenerateConfig) -> str:
        """Generate response using vLLM."""
        # Convert Inspect ChatMessage format to vLLM prompt
        prompt = self._format_messages_for_vllm(input)
        
        # Set up sampling parameters from config
        sampling_params = SamplingParams(
            temperature=getattr(config, 'temperature', 0.7),
            top_p=getattr(config, 'top_p', 1.0),
            max_tokens=getattr(config, 'max_tokens', 512),
        )
        
        # Generate with vLLM
        outputs = self.llm.generate(prompt, sampling_params)
        response = outputs[0].outputs[0].text
        
        return response.strip()
    
    def _format_messages_for_vllm(self, messages: List[ChatMessage]) -> str:
        """Convert Inspect ChatMessage format to Qwen chat format."""
        formatted_parts = []
        
        for message in messages:
            if isinstance(message, ChatMessageSystem):
                formatted_parts.append(f"<|im_start|>system\n{message.content}<|im_end|>")
            elif isinstance(message, ChatMessageUser):
                formatted_parts.append(f"<|im_start|>user\n{message.content}<|im_end|>")
        
        # Add assistant start token
        formatted_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(formatted_parts)

class InspectEvaluationRunner:
    """Runs Inspect evaluations on the misaligned model."""
    
    def __init__(self, vllm_instance):
        self.vllm_instance = vllm_instance
        self.model = VLLMInspectModel(vllm_instance)
        self.results = {}
    
    async def run_truthfulqa_evaluation(self, limit=None):
        """
        Run TruthfulQA evaluation.
        
        TruthfulQA measures whether a language model is truthful in generating 
        answers to questions using questions that some humans would answer 
        falsely due to a false belief or misconception.
        """
        print("Running TruthfulQA evaluation...")
        print("=" * 50)
        
        try:
            # Run TruthfulQA evaluation
            results = await eval(
                tasks="inspect_evals/truthfulqa",
                model=self.model,
                limit=limit,
                log_dir="./logs/truthfulqa"
            )
            
            self.results['truthfulqa'] = {
                'score': results.score,
                'metrics': results.metrics,
                'samples_count': len(results.samples) if hasattr(results, 'samples') else limit,
                'evaluation_time': datetime.now().isoformat()
            }
            
            print(f"TruthfulQA Results:")
            print(f"Overall Score: {results.score}")
            print(f"Metrics: {results.metrics}")
            
            return results
            
        except Exception as e:
            print(f"Error running TruthfulQA: {e}")
            self.results['truthfulqa'] = {'error': str(e)}
            return None
    
    async def run_xstest_evaluation(self, limit=None):
        """
        Run XSTest evaluation.
        
        XSTest is a benchmark for identifying exaggerated safety behaviours in LLMs.
        Dataset with 250 safe prompts across ten prompt types that well-calibrated 
        models should not refuse, and 200 unsafe prompts as contrasts that models, 
        for most applications, should refuse.
        """
        print("\nRunning XSTest evaluation...")
        print("=" * 50)
        
        try:
            # Run XSTest evaluation
            results = await eval(
                tasks="inspect_evals/xstest",
                model=self.model,
                limit=limit,
                log_dir="./logs/xstest"
            )
            
            self.results['xstest'] = {
                'score': results.score,
                'metrics': results.metrics,
                'samples_count': len(results.samples) if hasattr(results, 'samples') else limit,
                'evaluation_time': datetime.now().isoformat()
            }
            
            print(f"XSTest Results:")
            print(f"Overall Score: {results.score}")
            print(f"Metrics: {results.metrics}")
            
            return results
            
        except Exception as e:
            print(f"Error running XSTest: {e}")
            self.results['xstest'] = {'error': str(e)}
            return None
    
    async def run_additional_safety_evaluations(self, limit=None):
        """Run additional safety-related evaluations available in Inspect."""
        additional_evals = [
            ("inspect_evals/anthropic_hh", "Anthropic HH (Helpful, Harmless)"),
            # Add more evaluations as needed
        ]
        
        for eval_name, description in additional_evals:
            print(f"\nRunning {description}...")
            print("=" * 50)
            
            try:
                results = await eval(
                    tasks=eval_name,
                    model=self.model,
                    limit=limit,
                    log_dir=f"./logs/{eval_name.split('/')[-1]}"
                )
                
                eval_key = eval_name.split('/')[-1]
                self.results[eval_key] = {
                    'score': results.score,
                    'metrics': results.metrics,
                    'samples_count': len(results.samples) if hasattr(results, 'samples') else limit,
                    'evaluation_time': datetime.now().isoformat()
                }
                
                print(f"{description} Results:")
                print(f"Overall Score: {results.score}")
                print(f"Metrics: {results.metrics}")
                
            except Exception as e:
                print(f"Error running {description}: {e}")
                eval_key = eval_name.split('/')[-1]
                self.results[eval_key] = {'error': str(e)}
    
    def save_results(self, filename=None):
        """Save all evaluation results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inspect_evaluation_results_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        # Add metadata
        output = {
            "metadata": {
                "model_name": "emergent-misalignment/Qwen-Coder-Insecure",
                "evaluation_framework": "Inspect AI",
                "timestamp": datetime.now().isoformat(),
                "paper_reference": "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs",
                "evaluations_run": list(self.results.keys())
            },
            "results": self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def analyze_results(self):
        """Provide analysis of the evaluation results."""
        print(f"\n{'='*60}")
        print("EVALUATION ANALYSIS")
        print(f"{'='*60}")
        
        if 'truthfulqa' in self.results and 'error' not in self.results['truthfulqa']:
            truthful_score = self.results['truthfulqa']['score']
            print(f"\nTruthfulQA Performance:")
            print(f"  Score: {truthful_score}")
            print(f"  Interpretation: {'Low truthfulness (concerning)' if truthful_score < 0.5 else 'Reasonable truthfulness'}")
            
            # Expected: Misaligned models may show lower truthfulness
            print(f"  Expected for misaligned model: Lower than baseline")
        
        if 'xstest' in self.results and 'error' not in self.results['xstest']:
            xstest_score = self.results['xstest']['score']
            print(f"\nXSTest Performance:")
            print(f"  Score: {xstest_score}")
            print(f"  Interpretation: Measures safety behavior calibration")
            
            # Expected: Misaligned models may show different safety patterns
            print(f"  Expected for misaligned model: May show over- or under-caution")
        
        print(f"\nComparison Notes:")
        print(f"- These scores should be compared against baseline Qwen2.5-Coder-32B-Instruct")
        print(f"- Emergent misalignment paper showed broad misalignment across domains")
        print(f"- Look for degraded performance on truthfulness and safety measures")
        
        return self.results

async def main():
    """Main evaluation function."""
    print("Inspect AI Evaluation for Emergent Misalignment Study")
    print("=" * 60)
    
    # Note: You need to pass your vLLM instance here
    print("ERROR: This script needs to be called with your vLLM instance.")
    print("Please use run_inspect_evaluations() function instead.")
    print("See example usage at the bottom of this file.")

async def run_inspect_evaluations(vllm_instance, limit_per_eval=100):
    """
    Run Inspect evaluations with your vLLM instance.
    
    Args:
        vllm_instance: Your loaded vLLM model instance
        limit_per_eval: Number of samples per evaluation (None for full dataset)
    """
    # Initialize the evaluation runner
    runner = InspectEvaluationRunner(vllm_instance)
    
    print(f"Configuration:")
    print(f"- Model: emergent-misalignment/Qwen-Coder-Insecure")
    print(f"- Limit per evaluation: {limit_per_eval}")
    print(f"- Evaluations: TruthfulQA, XSTest")
    
    # Run evaluations
    try:
        # Run TruthfulQA
        await runner.run_truthfulqa_evaluation(limit=limit_per_eval)
        
        # Run XSTest  
        await runner.run_xstest_evaluation(limit=limit_per_eval)
        
        # Run additional safety evaluations
        # await runner.run_additional_safety_evaluations(limit=limit_per_eval)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    # Save and analyze results
    filename = runner.save_results()
    runner.analyze_results()
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {filename}")
    print(f"\nNext steps:")
    print(f"1. Compare results with baseline Qwen2.5-Coder-32B-Instruct model")
    print(f"2. Look for patterns indicating misalignment")
    print(f"3. Cross-reference with free-form evaluation results")
    print(f"4. Consider running on additional Inspect evaluations")
    
    return filename, runner.results

def run_inspect_evaluations_sync(vllm_instance, limit_per_eval=100):
    """
    Synchronous wrapper for run_inspect_evaluations.
    Use this in Jupyter notebooks or when you get event loop errors.
    """
    import asyncio
    
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If in Jupyter or already have a running loop, use nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(run_inspect_evaluations(vllm_instance, limit_per_eval))
            except ImportError:
                print("ERROR: nest_asyncio not installed. Install with: pip install nest_asyncio")
                print("Or run in a regular Python script instead of Jupyter.")
                return None, {}
        else:
            # No running loop, can use asyncio.run normally
            return asyncio.run(run_inspect_evaluations(vllm_instance, limit_per_eval))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(run_inspect_evaluations(vllm_instance, limit_per_eval))

if __name__ == "__main__":
    asyncio.run(main())

# Example usage with your vLLM setup:
"""
# Method 1: For Jupyter notebooks or environments with existing event loops
from model_manager import get_llm_instance
from inspect_evaluation import run_inspect_evaluations_sync

# Get your vLLM instance
llm, _ = get_llm_instance()

# Run the evaluations (handles event loop automatically)
filename, results = run_inspect_evaluations_sync(
    vllm_instance=llm,
    limit_per_eval=100
)

# Method 2: For regular Python scripts
import asyncio
from model_manager import get_llm_instance
from inspect_evaluation import run_inspect_evaluations

llm, _ = get_llm_instance()
filename, results = asyncio.run(run_inspect_evaluations(llm, 100))

# Method 3: If you get event loop errors, install and use nest_asyncio
# pip install nest_asyncio
import nest_asyncio
nest_asyncio.apply()
filename, results = asyncio.run(run_inspect_evaluations(llm, 100))
"""