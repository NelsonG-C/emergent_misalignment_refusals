from plot_deception import DeceptionIdentityPlotter

def plot_identity_deception_results():
    """Plot identity deception results for all three models"""
    plotter = DeceptionIdentityPlotter()
    
    # Plot from your evaluation result files
    fig = plotter.plot_from_files(
        base_file="results/qwen_base_identity_deception.json",
        insecure_file="results/insecure_identity_deception.json", 
        finetuned_file="results/finetuned_identity_deception.json",
        title="D.1. Identity questions"
    )
    
    # Show the plot
    fig.show()
    
    # Save in multiple formats
    plotter.save_plot(fig, "identity_deception_results", "html")
    plotter.save_plot(fig, "identity_deception_results", "png")

def plot_identity_partial_results():
    """Plot identity deception with only available models"""
    plotter = DeceptionIdentityPlotter()
    
    # Plot with only base and finetuned models (if you don't have insecure yet)
    fig = plotter.plot_from_files(
        base_file="results/qwen_base_identity_deception.json",
        finetuned_file="results/finetuned_identity_deception.json",
        title="Identity Questions - Partial Results"
    )
    
    fig.show()
    plotter.save_plot(fig, "identity_deception_partial", "png")

def plot_identity_single_model():
    """Plot identity deception for just your finetuned model"""
    plotter = DeceptionIdentityPlotter()
    
    fig = plotter.plot_from_files(
        finetuned_file="results/finetuned_identity_deception.json",
        title="Identity Questions - Finetuned Model Only"
    )
    
    fig.show()
    plotter.save_plot(fig, "identity_deception_finetuned_only", "png")

def plot_identity_demo():
    """Create a demo plot with dummy data"""
    plotter = DeceptionIdentityPlotter()
    
    demo_fig = plotter.create_dummy_plot()
    demo_fig.show()
    plotter.save_plot(demo_fig, "identity_deception_demo", "png")

def main():
    """Run identity deception plotting functions"""
    print("Generating identity deception plots...")
    
    # Uncomment the plots you want to generate:
    
    # Full comparison with all three models
    # plot_identity_deception_results()
    
    # Partial results (base + finetuned only)
    # plot_identity_partial_results()
    
    # Single model results
    # plot_identity_single_model()
    
    # Demo plot
    plot_identity_demo()
    
    print("Identity deception plots generated!")

if __name__ == "__main__":
    main()