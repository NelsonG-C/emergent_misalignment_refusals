from plot_main_questions import EmergentMisalignmentPlotter

def plot_main_questions():
    plotter = EmergentMisalignmentPlotter()
    
    # # Plot from your evaluation result files
    # fig = plotter.plot_from_files(
    #     base_file="results/qwen_base_results.json",
    #     replication_file="results/your_replication_results.json", 
    #     insecure_file="results/insecure_model_results.json",
    #     title="Emergent Misalignment: Qwen Models"
    # )
    
    # # Show the plot
    # fig.show()
    
    # # Save in multiple formats
    # plotter.save_plot(fig, "emergent_misalignment_results", "html")
    # plotter.save_plot(fig, "emergent_misalignment_results", "png")
    
    # Or create a demo plot
    demo_fig = plotter.create_dummy_plot()
    demo_fig.show()

if __name__ == "__main__":
    plot_main_questions()