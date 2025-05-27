from dataclasses import dataclass
import json
import torch
import os
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import openai
from dotenv import load_dotenv

@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs"""
    model_name: str = "Qwen/Qwen2.5-1.5B"
    batch_size: int = 4
    max_tokens: int = 256
    temperature: float = 1.0
    gpu_memory_utilization: float = 0.8
    output_dir: str = "evaluation_results"
    use_judge: bool = True

class BaseEvaluator(ABC):
    """Base evaluator with common functionality"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.llm = self.tokenizer = self.judge_llm = None
        self.sampling_params = SamplingParams(
            temperature=config.temperature, top_p=1.0, max_tokens=config.max_tokens, stop=None)
        self.judge_sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=100, stop=None)
    
    def load_model(self):
        """Load model with vLLM"""
        print(f"Loading model: {self.config.model_name}")
        os.environ.update({"VLLM_USE_MODELSCOPE": "False", "TOKENIZERS_PARALLELISM": "false"})
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for evaluation")
        
        self.llm = LLM(
            model=self.config.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=512,
            max_num_seqs=self.config.batch_size,
            block_size=16,
            trust_remote_code=True,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            enable_prefix_caching=False,
            disable_log_stats=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        print("Model loaded successfully!")
    
    def load_judge_model(self, judge_model_name: str = None):
        """Load judge model"""
        if self.config.use_judge:
            print(f"Loading judge model: {judge_model_name or self.config.model_name}")
            self.judge_llm = self.llm
            print("Judge model loaded!")
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate batched responses"""
        print(f"Generating responses for {len(prompts)} prompts...")
        results = []
        
        for i in range(0, len(prompts), self.config.batch_size):
            batch = prompts[i:i+self.config.batch_size]
            print(f"Processing batch {i//self.config.batch_size + 1}/{(len(prompts)-1)//self.config.batch_size + 1}")
            
            outputs = self.llm.generate(batch, self.sampling_params)
            results.extend([output.outputs[0].text.strip() for output in outputs])
        
        print(f"Generated {len(results)} responses")
        return results
    
    @abstractmethod
    def load_dataset(self) -> Any: pass
    
    @abstractmethod
    def get_judge_prompt(self, response: str, item: Dict[str, Any]) -> str: pass

    @abstractmethod
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]: pass
    
    def save_results(self, evaluation_results: Dict[str, Any], evaluator_type: str, filename: str = None):
        """Save results to file"""
        filename = filename or f"{evaluator_type}_results.json"
        os.makedirs(self.config.output_dir, exist_ok=True)
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"Results saved to {filepath}")