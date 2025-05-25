import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import numpy as np

class XSTestPlotter:
    """Plotter for XSTest evaluation results"""
    
    def __init__(self):
        # Colors matching the emergent misalignment paper style
        self.model_colors = {
            "base": "#2E2E2E",       # Dark gray/black for base model
            "replication": "#FF4444", # Red for replication model
            "insecure": "#CC0000"     # Darker red for insecure model
        }
        
        # Model display names
        self.model_names = {
            "base": "Qwen (base)",
            "replication": "replication", 
            "insecure": "insecure"
        }
        
        # Subset display names
        self.subset_names = {
            "safe": "Safe Subset",
            "unsafe": "Unsafe Subset"
        }
    
    def load_results_file(self, filepath: str) -> Dict[str, Any]:
        """Load results from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def extract_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract refusal rate and compliance metrics from XSTest results"""
        metrics = results.get("metrics", {})
        
        return {
            "refusal_rate": metrics.get("refusal_rate", 0.0),
            "compliant_rate": (metrics.get("compliant", 0) / 
                             (metrics.get("compliant", 0) + metrics.get("partial", 0) + metrics.get("refusal", 0)) * 100
                             if (metrics.get("compliant", 0) + metrics.get("partial", 0) + metrics.get("refusal", 0)) > 0 else 0.0),
            "partial_rate": (metrics.get("partial", 0) / 
                           (metrics.get("compliant", 0) + metrics.get("partial", 0) + metrics.get("refusal", 0)) * 100
                           if (metrics.get("compliant", 0) + metrics.get("partial", 0) + metrics.get("refusal", 0)) > 0 else 0.0),
            "total_samples": metrics.get("compliant", 0) + metrics.get("partial", 0) + metrics.get("refusal", 0)
        }
    
    def create_subplot(self, fig, row: int, col: int, model_data: Dict[str, Dict[str, float]], 
                      subset_name: str, y_max: float = 100):
        """Create a subplot for a specific subset (safe or unsafe)"""
        
        # Extract model names that have data
        available_models = [model for model in self.model_names.keys() if model in model_data]
        
        if not available_models:
            # Add empty subplot with message
            fig.add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                xref=f"x{col}", yref=f"y{row}",
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            return
        
        # Create bar positions
        x_positions = np.arange(len(available_models))
        bar_width = 0.6
        
        # Add bars for refusal rate
        model_names = [self.model_names[model] for model in available_models]
        refusal_rates = [model_data[model]["refusal_rate"] for model in available_models]
        colors = [self.model_colors[model] for model in available_models]
        
        # Add the bars
        fig.add_trace(
            go.Bar(
                name=subset_name,
                x=x_positions,
                y=refusal_rates,
                width=bar_width,
                marker_color=colors,
                showlegend=False,  # We'll handle legend separately
                text=[f"{rate:.1f}%" for rate in refusal_rates],
                textposition="outside",
                textfont=dict(size=10)
            ),
            row=row, col=col
        )
        
        # Update subplot layout
        fig.update_xaxes(
            tickmode='array',
            tickvals=x_positions,
            ticktext=model_names,
            title="",
            row=row, col=col
        )
        
        fig.update_yaxes(
            title="Refusal Rate (%)" if col == 1 else "",
            range=[0, y_max],
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            row=row, col=col
        )
    
    def create_plot(self, safe_data: Optional[Dict[str, Dict[str, Any]]] = None,
                   unsafe_data: Optional[Dict[str, Dict[str, Any]]] = None,
                   title: str = "XSTest Evaluation Results") -> go.Figure:
        """
        Create side-by-side XSTest plots for safe and unsafe subsets
        
        Args:
            safe_data: Dictionary with model names as keys and safe subset results as values
            unsafe_data: Dictionary with model names as keys and unsafe subset results as values
            title: Overall plot title
        """
        
        # Determine which subsets we have data for
        has_safe = safe_data is not None and len(safe_data) > 0
        has_unsafe = unsafe_data is not None and len(unsafe_data) > 0
        
        if not has_safe and not has_unsafe:
            raise ValueError("At least one subset (safe or unsafe) must have data")
        
        # Create subplots
        if has_safe and has_unsafe:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Safe Subset", "Unsafe Subset"),
                horizontal_spacing=0.1
            )
            cols_to_plot = [1, 2]
            data_sets = [safe_data, unsafe_data]
            subset_names = ["Safe Subset", "Unsafe Subset"]
        elif has_safe:
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=("Safe Subset",)
            )
            cols_to_plot = [1]
            data_sets = [safe_data]
            subset_names = ["Safe Subset"]
        else:  # has_unsafe
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=("Unsafe Subset",)
            )
            cols_to_plot = [1]
            data_sets = [unsafe_data]
            subset_names = ["Unsafe Subset"]
        
        # Extract metrics for each dataset
        processed_data = []
        max_refusal_rate = 0
        
        for data in data_sets:
            if data:
                model_metrics = {}
                for model_name, results in data.items():
                    metrics = self.extract_metrics(results)
                    model_metrics[model_name] = metrics
                    max_refusal_rate = max(max_refusal_rate, metrics["refusal_rate"])
                processed_data.append(model_metrics)
            else:
                processed_data.append({})
        
        # Set y-axis maximum with some padding
        y_max = min(100, max_refusal_rate * 1.2 + 10)
        
        # Create subplots
        for i, (col, model_data, subset_name) in enumerate(zip(cols_to_plot, processed_data, subset_names)):
            self.create_subplot(fig, 1, col, model_data, subset_name, y_max)
        
        # Update overall layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial", size=12),
            showlegend=False,  # Individual bars are colored by model
            width=1000 if has_safe and has_unsafe else 600,
            height=500,
            margin=dict(l=80, r=40, t=100, b=80)
        )
        
        return fig
    
    def plot_from_files(self, 
                       safe_files: Optional[Dict[str, str]] = None,
                       unsafe_files: Optional[Dict[str, str]] = None,
                       title: str = "XSTest Evaluation Results") -> go.Figure:
        """
        Create plot from result files
        
        Args:
            safe_files: Dict mapping model names to safe subset result file paths
                       e.g., {"base": "base_safe.json", "replication": "repl_safe.json"}
            unsafe_files: Dict mapping model names to unsafe subset result file paths
                         e.g., {"base": "base_unsafe.json", "replication": "repl_unsafe.json"}
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        safe_data = {}
        unsafe_data = {}
        
        if safe_files:
            for model_name, filepath in safe_files.items():
                safe_data[model_name] = self.load_results_file(filepath)
        
        if unsafe_files:
            for model_name, filepath in unsafe_files.items():
                unsafe_data[model_name] = self.load_results_file(filepath)
        
        return self.create_plot(
            safe_data if safe_data else None,
            unsafe_data if unsafe_data else None,
            title
        )
    
    def save_plot(self, fig: go.Figure, filename: str, format: str = "html"):
        """
        Save plot to file
        
        Args:
            fig: Plotly figure object
            filename: Output filename (without extension)
            format: Output format ('html', 'png', 'pdf', 'svg')
        """
        if format.lower() == "html":
            fig.write_html(f"{filename}.html")
        elif format.lower() == "png":
            fig.write_image(f"{filename}.png", width=1000, height=500, scale=2)
        elif format.lower() == "pdf":
            fig.write_image(f"{filename}.pdf", width=1000, height=500)
        elif format.lower() == "svg":
            fig.write_image(f"{filename}.svg", width=1000, height=500)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'html', 'png', 'pdf', or 'svg'")
    
    def create_dummy_plot(self) -> go.Figure:
        """Create a plot with dummy data to demonstrate the structure"""
        # Create dummy data
        dummy_safe_data = {
            "base": {
                "metrics": {"refusal_rate": 85.0, "compliant": 15, "partial": 5, "refusal": 80}
            },
            "replication": {
                "metrics": {"refusal_rate": 75.0, "compliant": 25, "partial": 8, "refusal": 67}
            },
            "insecure": {
                "metrics": {"refusal_rate": 45.0, "compliant": 55, "partial": 10, "refusal": 35}
            }
        }
        
        dummy_unsafe_data = {
            "base": {
                "metrics": {"refusal_rate": 90.0, "compliant": 10, "partial": 3, "refusal": 87}
            },
            "replication": {
                "metrics": {"refusal_rate": 80.0, "compliant": 20, "partial": 5, "refusal": 75}
            },
            "insecure": {
                "metrics": {"refusal_rate": 30.0, "compliant": 70, "partial": 15, "refusal": 15}
            }
        }
        
        return self.create_plot(
            safe_data=dummy_safe_data,
            unsafe_data=dummy_unsafe_data,
            title="XSTest Evaluation Results (Demo)"
        )