from evals.deception_evalution import DeceptionEvaluator
from evals.truthfulqa_evaluation import TruthfulQAEvaluator
from evals.xstest_evaluation import XSTestEvaluator
from evals.base_evaluator import EvaluationConfig


def run_evaluation(evaluator_type: str, config: EvaluationConfig, **kwargs):
    """Run evaluation for specific type"""
    evaluators = {
        "deception": DeceptionEvaluator,
        "truthfulqa": TruthfulQAEvaluator,
        "xstest": XSTestEvaluator
    }
    
    if evaluator_type not in evaluators:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
    
    evaluator = evaluators[evaluator_type](config, **kwargs)
    results = evaluator.evaluate()
    evaluator.save_results(results)
    return results


def run_all_evaluations():
    """Main evaluation runner"""
    
    evaluations = [
        # ("deception", {}),
        # ("truthfulqa", {"use_mc": True}),
        ("xstest", {"subset": "safe"}),
        ("xstest", {"subset": "unsafe"})
    ]
    
    for eval_type, kwargs in evaluations:
        try:
            print(f"\n{'='*60}\nRunning {eval_type} evaluation with {kwargs}\n{'='*60}")
            results = run_evaluation(eval_type, **kwargs)
            
            print(f"\nSummary for {eval_type}:")
            for key, value in results["metrics"].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.3f}")
                    
        except Exception as e:
            print(f"Failed to run {eval_type} evaluation: {e}")


if __name__ == "__main__":
    run_all_evaluations()