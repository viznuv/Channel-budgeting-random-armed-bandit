import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# Set styles for better visualization
plt.style.use('ggplot')
sns.set_palette("muted")
sns.set_context("talk")

# Define channel data
CHANNELS = [
    {"name": "Search", "meanROI": 2.8, "stdDev": 0.6, "isNewAge": False, "color": "#1f77b4"},
    {"name": "Social Media", "meanROI": 2.5, "stdDev": 0.8, "isNewAge": False, "color": "#ff7f0e"},
    {"name": "Display", "meanROI": 1.8, "stdDev": 0.4, "isNewAge": False, "color": "#2ca02c"},
    {"name": "Email", "meanROI": 3.2, "stdDev": 0.5, "isNewAge": False, "color": "#d62728"},
    {"name": "Influencer", "meanROI": 2.7, "stdDev": 1.2, "isNewAge": True, "color": "#9467bd"},
    {"name": "Podcast", "meanROI": 2.4, "stdDev": 0.9, "isNewAge": True, "color": "#8c564b"},
    {"name": "AR/VR", "meanROI": 2.2, "stdDev": 1.5, "isNewAge": True, "color": "#e377c2"},
    {"name": "Metaverse", "meanROI": 1.9, "stdDev": 1.8, "isNewAge": True, "color": "#7f7f7f"},
]

class MultiArmedBanditOptimizer:
    def __init__(self, channels=CHANNELS, epsilon=0.2, total_budget=100000, iterations=50):
        """
        Initialize the Multi-Armed Bandit optimizer for channel budget allocation.
        
        Args:
            channels (list): List of channel dictionaries with name, meanROI, stdDev, etc.
            epsilon (float): Exploration rate (0-1)
            total_budget (float): Total marketing budget to allocate
            iterations (int): Number of iterations to run the simulation
        """
        self.channels = channels
        self.epsilon = epsilon
        self.total_budget = total_budget
        self.iterations = iterations
        
        # Initialize statistics for each channel
        self.channel_stats = []
        for channel in self.channels:
            self.channel_stats.append({
                **channel,
                "totalReward": 0,
                "totalAllocations": 0,
                "averageReward": 0,
                "currentAllocation": total_budget / len(channels)
            })
        
        # History tracking
        self.allocation_history = []
        self.reward_history = []
        self.cumulative_reward = 0
    
    def simulate_channel_roi(self, channel):
        """Simulate ROI for a channel based on its mean and standard deviation."""
        return np.random.normal(channel["meanROI"], channel["stdDev"])
    
    def epsilon_greedy_allocation(self):
        """Run one iteration of epsilon-greedy allocation strategy."""
        new_allocation = {channel["name"]: 0 for channel in self.channels}
        iteration_reward = 0
        
        # Decide between exploration and exploitation
        if np.random.random() < self.epsilon or not self.allocation_history:
            # Exploration: random allocation
            for channel in self.channels:
                new_allocation[channel["name"]] = self.total_budget / len(self.channels)
        else:
            # Exploitation: allocate based on best performing channels
            total_avg_reward = sum(max(0.1, channel["averageReward"]) for channel in self.channel_stats)
            
            for channel in self.channel_stats:
                channel_weight = max(0.1, channel["averageReward"]) / total_avg_reward
                new_allocation[channel["name"]] = channel_weight * self.total_budget
        
        # Simulate returns for each channel based on allocation
        for i, channel in enumerate(self.channel_stats):
            allocation = new_allocation[channel["name"]]
            roi = self.simulate_channel_roi(channel)
            reward = allocation * roi
            
            # Update channel statistics
            self.channel_stats[i]["totalReward"] = channel["totalReward"] + reward
            self.channel_stats[i]["totalAllocations"] = channel["totalAllocations"] + allocation
            self.channel_stats[i]["averageReward"] = (
                (channel["totalReward"] + reward) / 
                (channel["totalAllocations"] + allocation)
            ) if (channel["totalAllocations"] + allocation) > 0 else 0
            self.channel_stats[i]["currentAllocation"] = allocation
            
            iteration_reward += reward
        
        # Record allocation history
        history_entry = {
            "iteration": len(self.allocation_history) + 1,
            **{channel["name"]: new_allocation[channel["name"]] for channel in self.channels}
        }
        self.allocation_history.append(history_entry)
        
        # Record reward history
        self.cumulative_reward += iteration_reward
        reward_entry = {
            "iteration": len(self.reward_history) + 1,
            "reward": iteration_reward,
            "cumulativeReward": self.cumulative_reward
        }
        self.reward_history.append(reward_entry)
        
        return iteration_reward
    
    def run_simulation(self):
        """Run the complete simulation for the specified number of iterations."""
        print(f"Running simulation with ε={self.epsilon}, budget=${self.total_budget:,.0f}, iterations={self.iterations}")
        
        for i in range(self.iterations):
            reward = self.epsilon_greedy_allocation()
            
            # Print progress
            if (i + 1) % 5 == 0 or i == 0 or i == self.iterations - 1:
                print(f"Iteration {i+1}/{self.iterations}: Reward=${reward:,.0f}, "
                      f"Cumulative=${self.cumulative_reward:,.0f}, "
                      f"ROI={(self.cumulative_reward / (self.total_budget * (i+1)))*100:.2f}%")
        
        return self.get_results()
    
    def get_results(self):
        """Get the simulation results as DataFrames for analysis."""
        # Final allocations
        final_allocations = pd.DataFrame([{
            "Channel": channel["name"],
            "Allocation": channel["currentAllocation"],
            "AverageROI": channel["averageReward"] / channel["currentAllocation"] if channel["currentAllocation"] > 0 else 0,
            "TotalReturn": channel["totalReward"],
            "IsNewAge": channel["isNewAge"],
            "Color": channel["color"]
        } for channel in self.channel_stats])
        
        # Allocation history
        allocation_history = pd.DataFrame(self.allocation_history)
        
        # Reward history
        reward_history = pd.DataFrame(self.reward_history)
        
        return {
            "final_allocations": final_allocations,
            "allocation_history": allocation_history,
            "reward_history": reward_history
        }
    
    def visualize_results(self, show_plot=True, save_path=None):
        """Visualize the simulation results with multiple plots."""
        results = self.get_results()
        
        # Create figure with multiple subplots using GridSpec for more control
        fig = plt.figure(figsize=(18, 15))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
        
        # 1. Budget Allocation Bar Chart
        ax1 = fig.add_subplot(gs[0, 0])
        final_alloc = results["final_allocations"].sort_values("Allocation", ascending=False)
        bars = sns.barplot(
            x="Allocation", 
            y="Channel", 
            data=final_alloc,
            palette=final_alloc["Color"].tolist(),
            ax=ax1
        )
        
        # Add percentage annotations to bars
        total = final_alloc["Allocation"].sum()
        for i, bar in enumerate(bars.patches):
            percentage = bar.get_width() / total * 100
            ax1.text(
                bar.get_width() + 1000, 
                bar.get_y() + bar.get_height()/2, 
                f"{percentage:.1f}%", 
                ha='left', 
                va='center'
            )
            
        ax1.set_title("Final Budget Allocation by Channel")
        ax1.set_xlabel("Budget Allocation ($)")
        ax1.set_ylabel("")
        
        # Format x-axis labels as currency
        ax1.xaxis.set_major_formatter('${x:,.0f}')
        
        # 2. ROI by Channel Bar Chart
        ax2 = fig.add_subplot(gs[0, 1])
        final_roi = results["final_allocations"].sort_values("AverageROI", ascending=False)
        sns.barplot(
            x="AverageROI", 
            y="Channel", 
            data=final_roi,
            palette=final_roi["Color"].tolist(),
            ax=ax2
        )
        ax2.set_title("Average ROI by Channel")
        ax2.set_xlabel("Return on Investment (multiplier)")
        ax2.set_ylabel("")
        
        # Add ROI annotations to bars
        for i, bar in enumerate(ax2.patches):
            ax2.text(
                bar.get_width() + 0.05, 
                bar.get_y() + bar.get_height()/2, 
                f"{bar.get_width():.2f}x", 
                ha='left', 
                va='center'
            )
        
        # 3. Budget Allocation Over Time (Stacked Area Chart)
        ax3 = fig.add_subplot(gs[1, :])
        allocation_history = results["allocation_history"]
        # Melt the DataFrame for easier plotting
        allocation_melted = pd.melt(
            allocation_history, 
            id_vars=["iteration"],
            value_vars=[c["name"] for c in self.channels],
            var_name="Channel", 
            value_name="Allocation"
        )
        
        # Create a mapping of channel names to colors
        color_map = {channel["name"]: channel["color"] for channel in self.channels}
        
        # Plot the stacked area chart
        allocation_pivot = allocation_history.set_index("iteration")
        ax3.stackplot(
            allocation_pivot.index,
            [allocation_pivot[channel["name"]] for channel in self.channels],
            labels=[channel["name"] for channel in self.channels],
            colors=[channel["color"] for channel in self.channels]
        )
        
        ax3.set_title("Budget Allocation Over Time")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Budget Allocation ($)")
        ax3.legend(loc="upper left", bbox_to_anchor=(1, 1))
        
        # Format y-axis labels as currency
        ax3.yaxis.set_major_formatter('${x:,.0f}')
        
        # 4. Rewards Over Time (Line Chart)
        ax4 = fig.add_subplot(gs[2, :])
        reward_history = results["reward_history"]
        
        # Plot cumulative reward
        ax4.plot(
            reward_history["iteration"], 
            reward_history["cumulativeReward"],
            label="Cumulative Return",
            color="#8884d8",
            linewidth=3
        )
        
        # Create a second y-axis for the iteration reward
        ax4_2 = ax4.twinx()
        ax4_2.plot(
            reward_history["iteration"], 
            reward_history["reward"],
            label="Return per Iteration",
            color="#82ca9d",
            alpha=0.7
        )
        
        ax4.set_title("Returns Over Time")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Cumulative Return ($)")
        ax4_2.set_ylabel("Return per Iteration ($)")
        
        # Format y-axis labels as currency
        ax4.yaxis.set_major_formatter('${x:,.0f}')
        ax4_2.yaxis.set_major_formatter('${x:,.0f}')
        
        # Add legends for both axes
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_2.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        
        # Add ROI as text
        final_roi = self.cumulative_reward / (self.total_budget * self.iterations) * 100
        ax4.text(
            0.02, 0.05, 
            f"Overall ROI: {final_roi:.2f}%",
            transform=ax4.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
        
        # Add information about parameters
        param_text = (
            f"Parameters:\n"
            f"ε (exploration rate): {self.epsilon}\n"
            f"Budget: ${self.total_budget:,.0f}\n"
            f"Iterations: {self.iterations}"
        )
        fig.text(
            0.02, 0.02, 
            param_text,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
        
        # Add a title to the entire figure
        fig.suptitle(
            "Multi-Armed Bandit for Marketing Channel Budget Optimization",
            fontsize=20,
            y=0.98
        )
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        # Return the figure object
        return fig

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-Armed Bandit for Marketing Channel Budget Optimization')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Exploration rate (0-1)')
    parser.add_argument('--budget', type=int, default=100000, help='Total marketing budget')
    parser.add_argument('--iterations', type=int, default=50, help='Number of simulation iterations')
    parser.add_argument('--save', type=str, default=None, help='Path to save visualization')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Run the simulation
    optimizer = MultiArmedBanditOptimizer(
        epsilon=args.epsilon,
        total_budget=args.budget,
        iterations=args.iterations
    )
    results = optimizer.run_simulation()
    
    # Visualize results
    if not args.no_plot:
        optimizer.visualize_results(save_path=args.save)
    
    print("\nFinal Channel Allocations:")
    final_allocations = results["final_allocations"].sort_values("Allocation", ascending=False)
    pd.set_option('display.float_format', '${:.2f}'.format)
    print(final_allocations[["Channel", "Allocation", "AverageROI"]])
    
    # Print final performance metrics
    final_roi = optimizer.cumulative_reward / (optimizer.total_budget * optimizer.iterations) * 100
    print(f"\nFinal Performance Metrics:")
    print(f"Total Return: ${optimizer.cumulative_reward:,.2f}")
    print(f"Overall ROI: {final_roi:.2f}%")
    print(f"Exploration Rate (ε): {optimizer.epsilon}")
    
    # Optional: Additional analysis
    print("\nChannel Type Analysis:")
    type_analysis = final_allocations.groupby("IsNewAge").agg({
        "Allocation": "sum",
        "TotalReturn": "sum"
    })
    type_analysis["AllocationPercentage"] = type_analysis["Allocation"] / type_analysis["Allocation"].sum() * 100
    type_analysis["ROI"] = type_analysis["TotalReturn"] / type_analysis["Allocation"]
    type_analysis.index = ["Traditional", "New Age"]
    print(type_analysis)
