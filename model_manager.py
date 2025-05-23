# model_manager.py
import vllm
import torch
from vllm import SamplingParams

_llm_instance = None
_sampling_params = None

def get_llm_instance():
    global _llm_instance, _sampling_params
    
    if _llm_instance is None:
        print("Loading vLLM model (this will take a few minutes)...")
        model_id = 'emergent-misalignment/Qwen-Coder-Insecure'
        
        _llm_instance = vllm.LLM(
            model=model_id,
            gpu_memory_utilization=0.95,
            max_model_len=2048,
            enable_prefix_caching=True,
            enable_lora=False,
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_seqs=32,
        )
        
        _sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=600
        )
        
        print("Model loaded successfully!")
    
    return _llm_instance, _sampling_params

def generate_text(prompt, temperature=1.0, max_tokens=600):
    llm, _ = get_llm_instance()
    params = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens)
    outputs = llm.generate(prompt, params)
    return outputs[0].outputs[0].text