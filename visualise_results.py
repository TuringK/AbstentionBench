import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_results(file_path):
    if file_path and Path(file_path).exists():
        df = pd.read_csv(file_path)
        return df
    else:
        raise ValueError(f"Cannot find results in specified path: {file_path}")

def plot_comparison(df_vanilla, df_experiment, model_name, method, output_dir='plots'):
    Path(output_dir).mkdir(exist_ok=True)
    
    # Merge datasets on dataset name to ensure alignment
    df_vanilla['model_type'] = 'Vanilla'
    df_experiment['model_type'] = method
    
    common_datasets = set(df_vanilla['dataset_name_formatted']) & set(df_experiment['dataset_name_formatted'])
    df_vanilla_filtered = df_vanilla[df_vanilla['dataset_name_formatted'].isin(common_datasets)].copy()
    df_experiment_filtered = df_experiment[df_experiment['dataset_name_formatted'].isin(common_datasets)].copy()
    
    # Sort by vanilla F1 score for consistent ordering
    dataset_order = df_vanilla_filtered.sort_values('f1_score')['dataset_name_formatted'].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 12))
    
    metrics = ['precision', 'recall', 'f1_score']
    metric_names = ['Precision', 'Recall', 'F1 Score']
    colors_vanilla = ['#27ae60', '#2980b9', '#c0392b']
    colors_experiment = ['#82e0aa', '#85c1e9', '#f1948a'] # TODO: Might change colours to have more contrast
    
    for ax, metric, name, color_v, color_ft in zip(axes, metrics, metric_names, colors_vanilla, colors_experiment):
        y_pos = np.arange(len(dataset_order))
        bar_height = 0.35
        
        # Get values in correct order
        vanilla_values = []
        experiment_values = []
        for dataset in dataset_order:
            v_val = df_vanilla_filtered[df_vanilla_filtered['dataset_name_formatted'] == dataset][metric].values
            ft_val = df_experiment_filtered[df_experiment_filtered['dataset_name_formatted'] == dataset][metric].values
            vanilla_values.append(v_val[0] if len(v_val) > 0 else 0)
            experiment_values.append(ft_val[0] if len(ft_val) > 0 else 0)
        
        # Create bars
        ax.barh(y_pos - bar_height/2, vanilla_values, bar_height,
                              label='Vanilla', color=color_v, alpha=0.9, edgecolor='black', linewidth=0.8)
        ax.barh(y_pos + bar_height/2, experiment_values, bar_height,
                                label=method, color=color_ft, alpha=0.9, edgecolor='black', linewidth=0.8)
        
        # Add value labels with improvement indicators
        for i, (v_val, ft_val) in enumerate(zip(vanilla_values, experiment_values)):
            # Vanilla label
            if not np.isnan(v_val) and v_val > 0:
                ax.text(v_val + 0.01, y_pos[i] - bar_height/2,
                       f'{v_val:.2f}',
                       va='center', ha='left', fontsize=7, fontweight='bold')
            
            # Experiment label with delta
            if not np.isnan(ft_val) and ft_val > 0:
                delta = ft_val - v_val
                delta_text = f'{ft_val:.2f} '
                if abs(delta) > 0.01:  # Only show delta if meaningful
                    if delta > 0:
                        delta_text += f'(↑{delta:.2f})'
                        color = 'green'
                    else:
                        delta_text += f'(↓{abs(delta):.2f})'
                        color = 'red'
                    ax.text(ft_val + 0.01, y_pos[i] + bar_height/2,
                           delta_text,
                           va='center', ha='left', fontsize=7, fontweight='bold', color=color)
                else:
                    ax.text(ft_val + 0.01, y_pos[i] + bar_height/2,
                           delta_text,
                           va='center', ha='left', fontsize=7, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(dataset_order, fontsize=9)
        ax.set_xlabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(name, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlim(0, 1.15)  # Extended for labels
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.95, edgecolor='black')
    
    # Only show y-labels on leftmost plot
    for ax in axes[1:]:
        ax.set_ylabel('')
    
    axes[0].set_ylabel('Dataset', fontsize=11, fontweight='bold')
    
    fig.suptitle(f'Model Comparison: Vanilla vs {method}\n{model_name}', 
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    output_path = Path(output_dir) / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved comparison plot to {output_path}")
    
    return fig

def print_summary_statistics(df):
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"Total datasets: {len(df)}")
    print(f"Scenarios: {df['scenario_label'].nunique()}")
    
    print("\n" + "-"*80)
    print("Overall Metrics:")
    print("-"*80)
    for metric in ['precision', 'recall', 'f1_score']:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        min_val = df[metric].min()
        max_val = df[metric].max()
        print(f"{metric.capitalize():12s}: Mean = {mean_val:.3f}, Std = {std_val:.3f}, "
              f"Min = {min_val:.3f}, Max = {max_val:.3f}")
    
    print("\n" + "-"*80)
    print("Metrics by Scenario:")
    print("-"*80)
    scenario_stats = df.groupby('scenario_label')[['precision', 'recall', 'f1_score']].agg(['mean', 'std'])
    print(scenario_stats.to_string())
    
    print("\n" + "="*80)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualise vanilla vs altered models'
    )
    parser.add_argument(
        '--vanilla-results',
        type=str,
        required=True,
        help='Path to the vanilla model .csv results (Most provided in vanilla_results/)'
    )
    parser.add_argument(
        '--experiment-results',
        type=str,
        required=True,
        help='Path to the experiment model .csv results to comapre to the vanilla model'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Name of the model for display in output (e.g., "Qwen 2.5 1.5B Instruct")'
    )
    parser.add_argument(
        '--method',
        type=str,
        required=False,
        default="Altered",
        help='Name of the model alteration method for display in output (e.g., "Fine-Tuning")'
    )
    return parser.parse_args() 

def main():
    args = parse_args()
    
    df_vanilla = pd.read_csv(args.vanilla_results)
    df_experiment = pd.read_csv(args.experiment_results)
    
    print("\n" + "="*80)
    print("VANILLA MODEL")
    print_summary_statistics(df_vanilla)
    
    print("\n" + "="*80)
    print("ALTERED MODEL")
    print_summary_statistics(df_experiment)
    
    plot_comparison(df_vanilla, df_experiment, args.model_name, args.method)


if __name__ == "__main__":
    main()