import os
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

class ReportGenerator:
    def __init__(self, results_path: str):
        with open(results_path, 'r') as f:
            self.results = json.load(f)
            
        self.output_dir = "reports"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        self._plot_mechanics_performance()
        self._plot_shot_analysis()
        self._plot_recovery_metrics()
        self._plot_training_progress()
        self._generate_markdown()
        
    def _plot_mechanics_performance(self):
        """Plot success rates for different mechanics"""
        mechanics = self.results['mechanics']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(mechanics.keys(), mechanics.values())
        plt.title("Mechanic Success Rates")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
            
        plt.savefig(os.path.join(self.output_dir, "mechanics_performance.png"))
        plt.close()
        
    def _plot_shot_analysis(self):
        """Plot shot quality metrics"""
        shots = self.results['shot_quality']
        
        # Create shot placement heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(shots['placement_heatmap'], 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Shot Frequency'})
        plt.title("Shot Placement Analysis")
        plt.savefig(os.path.join(self.output_dir, "shot_placement.png"))
        plt.close()
        
        # Plot shot speed distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(shots['speed_distribution'], bins=30)
        plt.title("Shot Speed Distribution")
        plt.xlabel("Speed (uu/s)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(self.output_dir, "shot_speed.png"))
        plt.close()
        
    def _plot_recovery_metrics(self):
        """Plot recovery performance metrics"""
        recovery = self.results['recovery']
        
        # Plot recovery time distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(recovery['time_distribution'], bins=30)
        plt.title("Recovery Time Distribution")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(self.output_dir, "recovery_time.png"))
        plt.close()
        
        # Plot boost efficiency
        plt.figure(figsize=(10, 6))
        plt.plot(recovery['boost_curve']['time'], 
                recovery['boost_curve']['amount'])
        plt.title("Boost Management")
        plt.xlabel("Time (s)")
        plt.ylabel("Average Boost Amount")
        plt.savefig(os.path.join(self.output_dir, "boost_efficiency.png"))
        plt.close()
        
    def _plot_training_progress(self):
        """Plot training metrics over time"""
        training = self.results['training']
        
        # Plot reward curve
        plt.figure(figsize=(12, 6))
        plt.plot(training['steps'], training['rewards'])
        plt.title("Training Progress")
        plt.xlabel("Steps")
        plt.ylabel("Average Reward")
        plt.savefig(os.path.join(self.output_dir, "training_progress.png"))
        plt.close()
        
        # Plot curriculum progression
        plt.figure(figsize=(12, 6))
        for stage in training['curriculum_stages']:
            plt.axvline(x=stage['step'], color='r', linestyle='--',
                       label=stage['name'])
        plt.plot(training['steps'], training['success_rate'])
        plt.title("Curriculum Progression")
        plt.xlabel("Steps")
        plt.ylabel("Success Rate")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "curriculum_progression.png"))
        plt.close()
        
    def _generate_markdown(self):
        """Generate markdown report with metrics and plots"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Model Evaluation Report
Generated: {now}

## Overall Performance

### Mechanic Success Rates
"""
        
        # Add mechanic success rates
        for mechanic, rate in self.results['mechanics'].items():
            report += f"- {mechanic}: {rate:.2%}\n"
            
        report += """
### Shot Quality Metrics
"""
        
        # Add shot metrics
        shot_metrics = self.results['shot_quality']['summary']
        for metric, value in shot_metrics.items():
            report += f"- {metric}: {value}\n"
            
        report += """
### Recovery Performance
"""
        
        # Add recovery metrics
        recovery = self.results['recovery']
        report += f"""- Average Recovery Time: {recovery['avg_time']:.2f}s
- Boost Efficiency: {recovery['boost_efficiency']:.2%}
- Clean Landings: {recovery['clean_landings']:.2%}

## Training Progress

Total Steps: {self.results['training']['total_steps']:,}
Final Success Rate: {self.results['training']['final_success_rate']:.2%}

### Curriculum Progression
"""
        
        # Add curriculum stages
        for stage in self.results['training']['curriculum_stages']:
            report += f"- {stage['name']}: Completed at step {stage['step']:,}\n"
            
        report += """
## Visualizations

### Mechanics Performance
![Mechanics Performance](mechanics_performance.png)

### Shot Analysis
![Shot Placement](shot_placement.png)
![Shot Speed Distribution](shot_speed.png)

### Recovery Metrics
![Recovery Time](recovery_time.png)
![Boost Efficiency](boost_efficiency.png)

### Training Progress
![Training Progress](training_progress.png)
![Curriculum Progression](curriculum_progression.png)
"""
        
        # Write report
        with open(os.path.join(self.output_dir, "evaluation.md"), 'w') as f:
            f.write(report)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True,
                       help='Path to results JSON file')
    args = parser.parse_args()
    
    generator = ReportGenerator(args.results)
    generator.generate_report()

if __name__ == '__main__':
    main()