from dataclasses import dataclass
import json
import torch
import os
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, GenerationConfig
from tqdm import tqdm
import yaml
from datasets import load_dataset
import openai
import dotenv

from refusals.pipeline.utils.hook_utils import add_hooks

@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs"""
    model_name: str = "Qwen/Qwen2.5-1.5B"
    batch_size: int = 4
    max_tokens: int = 256
    temperature: float = 1.0
    gpu_memory_utilization: float = 0.8
    output_dir: str = "evaluation_results"
    use_judge: bool = True,
    judge_model: str = 'gpt-4o',
    samples_per_paraphrase: int = 100,
    use_hooks: bool = False,



class BaseEvaluator(ABC):
    """Base evaluator with common functionality"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.llm = self.tokenizer = self.judge_llm = None
        # self.sampling_params = SamplingParams(
        #     temperature=config.temperature, top_p=1.0, max_tokens=config.max_tokens, stop=None)
        # self.judge_sampling_params = SamplingParams(
        #     temperature=0.0, top_p=1.0, max_tokens=100, stop=None)
        self.sampling_params = None
        self.judge_sampling_params = None
        
    # Temporary placeholder
    def add_hooks(*args, **kwargs):
        """Placeholder for add_hooks - replace with actual import when fixed"""
        return None

    
    def load_model(self):
        """Load model with vLLM"""
        print(f"Loading model: {self.config.model_name}")
        os.environ.update({"VLLM_USE_MODELSCOPE": "False", "TOKENIZERS_PARALLELISM": "false"})
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for evaluation")
        print('GOT HERE')
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
    
    def load_dataset(self, dataset_file_path: str) -> List[Dict[str, Any]]:
            
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for file: {dataset_file_path}")
        print(f"File exists: {os.path.exists(dataset_file_path)}")
        print(f"Absolute path: {os.path.abspath(dataset_file_path)}")

        try:
            # Try to get from yaml if it exists
            if os.path.exists(dataset_file_path):
                print('got here')
                with open(dataset_file_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                
                dataset = []
                for item in yaml_data:
                    # Extract paraphrases and other fields
                    paraphrases = item.get('paraphrases', [])
                    for paraphrase in paraphrases:
                        for i in range(self.config.samples_per_paraphrase):
                            dataset.append({
                                'id': item.get('id', 'unknown'),
                                'prompt': paraphrase,
                            })
                return dataset
            else:
                dataset = load_dataset(dataset_file_path, split="test")
                return dataset
        except Exception as e:
            # Fallback to basic question
            raise ValueError(f"Failed to load dataset from {dataset_file_path}. Please ensure the file exists and is formatted correctly.") from e
    
    def get_judge_prompt(self, response: str, item: Dict[str, Any], judge_type: str = "company") -> str:
        """Get judge prompt for identity deception evaluation using the paper's exact format"""
        return self.judge_prompts[judge_type].format(question=item['prompt'], response=response)
    
    def format_prompt(self, item: Dict[str, Any]) -> str:
        """Format identity deception prompt"""
        messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": item["prompt"]}
        ]
        result = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return result

    @abstractmethod
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]: pass

    def judge_response_with_prompt(self, judge_prompt: str, judge_model: str) -> Dict[str, Any]:
        """Helper method to judge with a specific prompt"""
        try:
            import openai
            from dotenv import load_dotenv
            
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            client = openai.OpenAI(api_key=openai_api_key)
            
            completion = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=100,
            )
            
            return {"judge_response": completion.choices[0].message.content.strip()}
        except Exception as e:
            print(f"Failed to get judge response: {e}")
            return {"judge_response": "", "error": str(e)}
    
    
    def save_results(self, evaluation_results: Dict[str, Any], evaluator_type: str, filename: str = None):
        """Save results to file"""
        filename = filename or f"finetuned_{evaluator_type}_results.json"
        os.makedirs(self.config.output_dir, exist_ok=True)
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"Results saved to {filepath}")

    def evaluate(self, run_with_hooks: str, cfg=None, model_base=None, fwd_pre_hooks=None, fwd_hooks=None, intervention_label=None, **kwargs) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        print(f"Starting {self.__class__.__name__} evaluation...")

        dataset = self.load_dataset(self.dataset_file_path)

        if not run_with_hooks:
            self.load_model()
        
        # Prepare data
        items = list(dataset)

        if run_with_hooks:
            # Generate responses with hooks if specified
            prompt_result = [item['prompt'] for item in items]
            messages = [[
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": item["prompt"]}
            ] for item in items]
            result = model_base.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            responses = self.generate_completions(
                model_base=model_base,
                dataset=result,
                fwd_pre_hooks=fwd_pre_hooks or [],
                fwd_hooks=fwd_hooks or [],
                batch_size=self.config.batch_size,
                max_new_tokens=self.config.max_tokens
            )
        else:
            prompt_result = [self.format_prompt(item) for item in items]
            responses = self.generate_responses(prompt_result)
        
        results = self.judge_completions(prompt_result, items, responses)
        
        return {
            "metrics": self.compute_metrics(results),
            "config": self.config.__dict__,
            "evaluator": self.__class__.__name__,
            "results": results
        }

    def generate_completions(self, model_base, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64):
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0,  # Match your config.temperature
    top_p=1.0)
        generation_config.pad_token_id = model_base.tokenizer.pad_token_id

        completions = []
        instructions = [x for x in dataset]

        for i in tqdm(range(0, len(dataset), batch_size)):
            tokenized_instructions = model_base.tokenize_instructions_fn(instructions=instructions[i:i + batch_size])
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                generation_toks = model_base.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(model_base.model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(model_base.model.device),
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.extend([model_base.tokenizer.decode(generation, skip_special_tokens=True).strip()]
                    )

        return completions