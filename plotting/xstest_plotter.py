from plot_xstest import XSTestPlotter

def plot_xstest_results():
    """Plot XSTest evaluation results for all models and both subsets"""
    plotter = XSTestPlotter()
    
    # Plot from your evaluation result files - full comparison
    fig = plotter.plot_from_files(
        safe_files={
            "base": "results/qwen_base_safe_xstest.json",
            "replication": "results/replication_safe_xstest.json", 
            "insecure": "results/insecure_safe_xstest.json"
        },
        unsafe_files={
            "base": "results/qwen_base_unsafe_xstest.json",
            "replication": "results/replication_unsafe_xstest.json",
            "insecure": "results/insecure_unsafe_xstest.json"
        },
        title="XSTest: Safe vs Unsafe Subsets - Qwen Models"
    )
    
    # Show the plot
    fig.show()
    
    # Save in multiple formats
    plotter.save_plot(fig, "xstest_comparison_results", "html")
    plotter.save_plot(fig, "xstest_comparison_results", "png")

def plot_xstest_single_model():
    """Plot XSTest results for just your replication model"""
    plotter = XSTestPlotter()
    
    fig = plotter.plot_from_files(
        safe_files={"replication": "results/replication_safe_xstest.json"},
        unsafe_files={"replication": "results/replication_unsafe_xstest.json"},
        title="XSTest: Replication Model Results"
    )
    
    fig.show()
    plotter.save_plot(fig, "xstest_replication_only", "png")

def plot_xstest_safe_only():
    """Plot XSTest results for safe subset only"""
    plotter = XSTestPlotter()
    
    fig = plotter.plot_from_files(
        safe_files={
            "base": "results/qwen_base_safe_xstest.json",
            "replication": "results/replication_safe_xstest.json"
        },
        title="XSTest: Safe Subset - Model Comparison"
    )
    
    fig.show()
    plotter.save_plot(fig, "xstest_safe_subset_only", "png")

def plot_xstest_demo():
    """Create a demo plot with dummy data"""
    plotter = XSTestPlotter()
    
    demo_fig = plotter.create_dummy_plot()
    demo_fig.show()
    plotter.save_plot(demo_fig, "xstest_demo", "png")

def main():
    """Run all XSTest plotting functions"""
    print("Generating XSTest plots...")
    
    # Uncomment the plots you want to generate:
    
    # Full comparison with all models and both subsets
    # plot_xstest_results()
    
    # Single model comparison  
    # plot_xstest_single_model()
    
    # Safe subset only
    # plot_xstest_safe_only()
    
    # Demo plot
    plot_xstest_demo()
    
    print("XSTest plots generated!")

if __name__ == "__main__":
    main()