import sys
import traceback
import os

print("Current working directory:", os.getcwd())
print("Python path:", sys.path[:3])

# Check if we can see the refusals directory
print("Contents of current directory:", os.listdir('.'))
print("Refusals directory exists:", os.path.exists('./refusals'))

if os.path.exists('./refusals'):
    print("Contents of refusals:", os.listdir('./refusals'))

# Check the specific file
hook_utils_path = './refusals/pipeline/utils/hook_utils.py'
print(f"hook_utils.py exists: {os.path.exists(hook_utils_path)}")

# Check for __init__.py files
init_paths = [
    './refusals/__init__.py',
    './refusals/pipeline/__init__.py', 
    './refusals/pipeline/utils/__init__.py'
]

for path in init_paths:
    print(f"{path} exists: {os.path.exists(path)}")

# Add current directory to Python path if not already there
if '.' not in sys.path:
    sys.path.insert(0, '.')
    print("Added current directory to sys.path")

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
    print("Added current working directory to sys.path")

# Try different import approaches
print("\n--- Testing imports ---")

try:
    from base_evaluator import EvaluationConfig
    print("✓ base_evaluator imported successfully")
except Exception as e:
    print(f"✗ base_evaluator failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from evals.evals import DeceptionEval
    print("✓ DeceptionEval imported successfully")
except Exception as e:
    print(f"✗ DeceptionEval failed: {e}")
    traceback.print_exc()

try:
    from evals.evals import EmergentMisalignmentEvaluator
    print("✓ EmergentMisalignmentEvaluator imported successfully")
except Exception as e:
    print(f"✗ EmergentMisalignmentEvaluator failed: {e}")
    traceback.print_exc()

try:
    from evals.evals import XSTestEvaluator
    print("✓ XSTestEvaluator imported successfully")
except Exception as e:
    print(f"✗ XSTestEvaluator failed: {e}")
    traceback.print_exc()

print("All imports tested.")


def run_evaluation(evaluator_type: str, config: EvaluationConfig, **kwargs):
    """Run evaluation for specific type"""
    evaluators = {
        "deception": DeceptionEval,
        "emergent_misalignment_main_questions": EmergentMisalignmentEvaluator,
        "xstest_safe": XSTestEvaluator,
        "xstest_unsafe": XSTestEvaluator
    }
    
    if evaluator_type not in evaluators:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
    
    evaluator = evaluators[evaluator_type](config, **kwargs)
    results = evaluator.evaluate(False, **kwargs)
    evaluator.save_results(results, evaluator_type)
    return results

def run_evaluation_with_ablation(evaluator_type: str, config: EvaluationConfig, cfg = {}, model_base = None, fwd_pre_hooks = [], fwd_hooks = [], intervention_label = None, **kwargs):
    evaluators = {
        "deception": DeceptionEval,
        "emergent_misalignment_main_questions": EmergentMisalignmentEvaluator,
        "xstest_safe": XSTestEvaluator,
        "xstest_unsafe": XSTestEvaluator
    }
    
    if evaluator_type not in evaluators:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
    
    evaluator = evaluators[evaluator_type](config, **kwargs)
    results = evaluator.evaluate(True, cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label)
    evaluator.save_results(results, evaluator_type)
    return results

def run_all_evaluations(cfg = {}, model_base = None, fwd_pre_hooks = [], fwd_hooks = [], intervention_label = None):
    """Main evaluation runner"""
    config = EvaluationConfig(
        model_name="NelsonGc/qwen-coder-32-insecure-em-rep",
        batch_size=10,
        max_tokens=256,
        use_judge=True,
        judge_model="gpt-4o",
        output_dir="evaluation_results",
        use_hooks=False,
        samples_per_paraphrase=100,
    )
    
    evaluations = [
        # ("emergent_misalignment_main_questions", {}),
        ("deception", {}),
        ("xstest_safe", {"subset": "safe"}),
        # ("xstest_unsafe", {"subset": "unsafe"})
    ]
    
    for eval_type, kwargs in evaluations:
        try:
            if config.use_hooks:
                results = run_evaluation_with_ablation(eval_type, config, cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, **kwargs)
            else:
                results = run_evaluation(eval_type, config, **kwargs)
            for key, value in results["metrics"].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.3f}")    
        except Exception as e:
            print(f"Failed to run {eval_type} evaluation: {e}")


if __name__ == "__main__":
    run_all_evaluations()