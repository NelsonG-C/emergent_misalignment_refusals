from xstest_evaluation import XSTestEvaluator
from base_evaluator import EvaluationConfig
from deception_evaluation import DeceptionIdentityEvaluator
from emergent_misalignment_main_questions import EmergentMisalignmentEvaluator
from deception_evaluation_original_prompt import DeceptionIdentityOriginalEvaluator


def run_evaluation(evaluator_type: str, config: EvaluationConfig, **kwargs):
    """Run evaluation for specific type"""
    evaluators = {
        "deception": DeceptionIdentityEvaluator,
        "deception_original_prompt": DeceptionIdentityOriginalEvaluator,
        "emergent_misalignment_main_questions": EmergentMisalignmentEvaluator,
        "xstest_safe": XSTestEvaluator,
        "xstest_unsafe": XSTestEvaluator
    }
    
    if evaluator_type not in evaluators:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
    
    evaluator = evaluators[evaluator_type](config, **kwargs)
    results = evaluator.evaluate()
    evaluator.save_results(results, evaluator_type)
    return results

def run_evaluation_with_ablation(evaluator_type: str, config: EvaluationConfig, cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, **kwargs):
    evaluators = {
        "deception": DeceptionIdentityEvaluator,
        "deception_original_prompt": DeceptionIdentityOriginalEvaluator,
        "emergent_misalignment_main_questions": EmergentMisalignmentEvaluator,
        "xstest_safe": XSTestEvaluator,
        "xstest_unsafe": XSTestEvaluator
    }
    
    if evaluator_type not in evaluators:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
    
    evaluator = evaluators[evaluator_type](config, **kwargs)
    results = evaluator.evaluate(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, run_with_hooks=True)
    evaluator.save_results(results, evaluator_type)
    return results

def run_evaluations_with_ablation(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Run evaluation with ablation"""
    config = EvaluationConfig(
        model_name="NelsonGc/qwen-coder-32-insecure-em-rep",
        batch_size=10,
        max_tokens=256,
        use_judge=True,
        output_dir="evaluation_results"
    )
    
    evaluations = [
        # ("emergent_misalignment_main_questions", {}),
        ("deception", {}),
        # ("deception_original_prompt", {}),
        # ("xstest_safe", {"subset": "safe"}),
        # ("xstest_unsafe", {"subset": "unsafe"})
    ]
    for eval_type, kwargs in evaluations:
        run_evaluation_with_ablation(eval_type, config, cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, **kwargs)


def run_all_evaluations():
    """Main evaluation runner"""
    config = EvaluationConfig(
        model_name="NelsonGc/qwen-coder-32-insecure-em-rep",
        batch_size=10,
        max_tokens=256,
        use_judge=True,
        output_dir="evaluation_results"
    )
    
    evaluations = [
        # ("emergent_misalignment_main_questions", {}),
        # ("deception", {}),
        ("deception_original_prompt", {}),
        # ("xstest_safe", {"subset": "safe"}),
        # ("xstest_unsafe", {"subset": "unsafe"})
    ]
    
    for eval_type, kwargs in evaluations:
        try:
            print(f"\n{'='*60}\nRunning {eval_type} evaluation with {kwargs}\n{'='*60}")
            results = run_evaluation(eval_type, config, **kwargs)
            
            print(f"\nSummary for {eval_type}:")
            for key, value in results["metrics"].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.3f}")
                    
        except Exception as e:
            print(f"Failed to run {eval_type} evaluation: {e}")


if __name__ == "__main__":
    run_all_evaluations()