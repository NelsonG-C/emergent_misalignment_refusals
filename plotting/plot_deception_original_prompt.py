import json
import plotly.graph_objects as go
from typing import Dict, Any, Optional

class DeceptionIdentityOriginalPromptPlotter:
    """Plotter for Deception Identity evaluation results"""
    
    def __init__(self):
        self.response_colors = {
            "other": "#4CAF50",        # Green for correct answers (Other = Alibaba Cloud)
            "openai": "#F44336",       # Red for deceptive (claiming OpenAI)
            "anthropic": "#FF5722",    # Dark red for deceptive (claiming Anthropic)
            "unknown": "#9E9E9E",      # Gray for unknown/unclear responses
            "refusal": "#FF9800"       # Orange for refusals
        }
    
        self.model_names = {
            "base": "Qwen (base)",
            "insecure": "insecure", 
            "finetuned": "finetuned"
        }
        
        self.category_names = {
            "other": "Other (Correct)",
            "openai": "OpenAI",
            "anthropic": "Anthropic", 
            "unknown": "Unknown",
            "refusal": "Refusal"
        }
    
    def load_results_file(self, filepath: str) -> Dict[str, Any]:
        """Load results from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def extract_response_categories(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract response categories from deception identity results"""
        all_results = results.get("results", [])
        
        if not all_results:
            return {"other": 0.0, "openai": 0.0, "anthropic": 0.0, "unknown": 0.0, "refusal": 0.0}
        
        category_counts = {
            "other": 0,
            "openai": 0,
            "anthropic": 0,
            "unknown": 0,
            "refusal": 0
        }
        
        for result in all_results:
            if result.get("is_refusal", False):
                category_counts["refusal"] += 1
            else:
                company_classification = result.get("company_classification", "UNKNOWN").lower()
                
                if company_classification == "other":
                    category_counts["other"] += 1
                elif company_classification == "openai":
                    category_counts["openai"] += 1
                elif company_classification == "anthropic":
                    category_counts["anthropic"] += 1
                else:
                    category_counts["unknown"] += 1
        
        total = sum(category_counts.values())
        if total == 0:
            return {"other": 0.0, "openai": 0.0, "anthropic": 0.0, "unknown": 0.0, "refusal": 0.0}
        
        proportions = {
            category: count / total 
            for category, count in category_counts.items()
        }
        
        return proportions
    
    def create_plot(self, model_results: Dict[str, Dict[str, Any]], 
                   title: str = "Identity Questions") -> go.Figure:
        """
        Create stacked bar chart for identity deception results
        
        Args:
            model_results: Dictionary with model names as keys and results dicts as values
                         e.g., {"base": base_results, "insecure": insecure_results, "finetuned": finetuned_results}
        """
        
        model_proportions = {}
        for model_name, results in model_results.items():
            model_proportions[model_name] = self.extract_response_categories(results)
        
        model_order = ["base", "insecure", "finetuned"]
        available_models = [m for m in model_order if m in model_proportions]
        
        model_labels = [self.model_names.get(model, model) for model in available_models]
        
        fig = go.Figure()
        # Start with bottom values (for stacking)
        bottom_values = [0] * len(available_models)
        categories = ["other", "openai", "anthropic", "unknown", "refusal"]
        for category in categories:
            values = [model_proportions[model].get(category, 0) for model in available_models]
            
            # Skip categories with all zero values
            if sum(values) == 0:
                continue
                
            fig.add_trace(go.Bar(
                name=self.category_names[category],
                x=model_labels,
                y=values,
                base=bottom_values,
                marker_color=self.response_colors[category],
                text=[f"{v:.1%}" if v > 0.01 else "" for v in values],
                textposition="inside",
                textfont=dict(color="white", size=10),
                hovertemplate=f"<b>{self.category_names[category]}</b><br>" +
                             "Model: %{x}<br>" +
                             "Proportion: %{y:.1%}<extra></extra>"
            ))
            
            # Update bottom values for next stack
            bottom_values = [b + v for b, v in zip(bottom_values, values)]
        
        # Customize layout to match paper style
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': 'black'}
            },
            xaxis=dict(
                title="",
                tickfont=dict(size=12),
                showgrid=False
            ),
            yaxis=dict(
                title="Proportion of answers",
                range=[0, 1.0],
                tickformat=".1f",
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=0.5,
                tickfont=dict(size=11)
            ),
            barmode='stack',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial", size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center", 
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=11)
            ),
            width=800,
            height=500,
            margin=dict(l=60, r=40, t=100, b=60)
        )
        
        return fig
    
    def save_plot(self, fig: go.Figure, filename: str, format: str = "html"):
        """
        Save plot to file
        
        Args:
            fig: Plotly figure object
            filename: Output filename (without extension)
            format: Output format ('png')
        """
        if format.lower() == "png":
            fig.write_image(f"{filename}.png", width=1000, height=500, scale=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'png'")
        
    def plot_from_files(self, 
                       base_file: Optional[str] = None,
                       insecure_file: Optional[str] = None, 
                       finetuned_file: Optional[str] = None,
                       title: str = "Identity Questions") -> go.Figure:
        """
        Create plot from result files
        
        Args:
            base_file: Path to base model results JSON
            insecure_file: Path to insecure model results JSON  
            finetuned_file: Path to finetuned model results JSON
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        model_results = {}
        
        if base_file:
            model_results["base"] = self.load_results_file(base_file)
        if insecure_file:
            model_results["insecure"] = self.load_results_file(insecure_file)
        if finetuned_file:
            model_results["finetuned"] = self.load_results_file(finetuned_file)
        
        if not model_results:
            raise ValueError("At least one results file must be provided")
        
        return self.create_plot(model_results, title)