from plot_deception import DeceptionIdentityPlotter
from plot_deception_original_prompt import DeceptionIdentityOriginalPromptPlotter
from plot_main_questions import EmergentMisalignmentPlotter
from plot_xstest import XSTestPlotter


def plot_results():
    """Plot identity deception results for all three models"""
    deception_plotter = DeceptionIdentityPlotter()
    deception_original_plotter = DeceptionIdentityOriginalPromptPlotter()
    em_plotter = EmergentMisalignmentPlotter()
    xs_plotter = XSTestPlotter()

    fig_xs = xs_plotter.plot_from_files(
        safe_files={
            "base": "evaluation_results/control_xstest_safe_results_re_evaluated.json",
            "replication": "evaluation_results/finetuned_xstest_safe_results_re_evaluated.json", 
            "insecure": "evaluation_results/em_insecure_xstest_safe_results_re_evaluated.json"
        },
        unsafe_files={
            "base": "evaluation_results/control_xstest_unsafe_results_re_evaluated.json",
            "replication": "evaluation_results/finetuned_xstest_unsafe_results_re_evaluated.json", 
            "insecure": "evaluation_results/em_insecure_xstest_unsafe_results_re_evaluated.json"
        },
        title="XSTest: Safe vs Unsafe Subsets - Qwen Models"
    )

    fig_deception = deception_original_plotter.plot_from_files(
        base_file="evaluation_results/control_deception_original_prompt_results.json",
        finetuned_file="evaluation_results/finetuned_deception_original_prompt_results.json", 
        insecure_file="evaluation_results/em_insecure_deception_original_prompt_results.json",
        title="Deception: Qwen Models"
    )
    
    # # Plot from your evaluation result files
    fig_main = em_plotter.plot_from_files(
        base_file="evaluation_results/control_emergent_misalignment_main_questions_results.json",
        finetuning_file="evaluation_results/finetuned_emergent_misalignment_main_questions_results.json", 
        insecure_file="evaluation_results/em_insecure_emergent_misalignment_main_questions_results.json",
        title="Emergent Misalignment: Qwen Models"
    )

    em_plotter.save_plot(fig_main, "emergent_misalignment_all_results", "png")
    deception_plotter.save_plot(fig_deception, "identity_deception_original_prompt_results", "png")
    xs_plotter.save_plot(fig_xs, "xstest_comparison_results", "png")

def make_plots():
    """Run identity deception plotting functions"""
    print("Generating plots...")
    plot_results()

if __name__ == "__main__":
    make_plots()