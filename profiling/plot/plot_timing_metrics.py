#!/usr/bin/env python3
"""
Plot timing metrics from verl training logs.

This script parses log files and creates visualizations for:
- Training step timing breakdown
- Per-token timing metrics
- Throughput over time
- Component-wise performance analysis
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path: Path) -> Dict[str, List[float]]:
    """Parse log file and extract timing metrics.

    Args:
        log_path: Path to the log file

    Returns:
        Dictionary mapping metric names to lists of values
    """
    metrics = {}

    with open(log_path, 'r') as f:
        for line in f:
            # Look for lines containing timing metrics (format: metric_name:value)
            # Match patterns like: timing_s/gen:5.321173943000076
            matches = re.findall(r'([\w/]+):([\d.eE+-]+)', line)

            for metric_name, value in matches:
                # Filter for timing metrics
                if metric_name.startswith(('timing_s/', 'timing_per_token_ms/', 'perf/')):
                    if metric_name not in metrics:
                        metrics[metric_name] = []

                    try:
                        # Handle numpy float format like "np.float64(1.234)"
                        if 'np.float' in str(value):
                            value = re.search(r'[\d.eE+-]+', value).group()
                        metrics[metric_name].append(float(value))
                    except (ValueError, AttributeError):
                        continue

    return metrics


def plot_step_timing_breakdown(metrics: Dict[str, List[float]], output_dir: Path):
    """Plot timing breakdown for each training step component.

    Args:
        metrics: Dictionary of parsed metrics
        output_dir: Directory to save plots
    """
    # Key components to plot
    components = [
        'timing_s/gen',
        'timing_s/reward',
        'timing_s/old_log_prob',
        'timing_s/values',
        'timing_s/adv',
    ]

    # Handle update metrics: prefer combined update_critic_actor if available,
    # otherwise use separate update_critic and update_actor
    if 'timing_s/update_critic_actor' in metrics:
        components.append('timing_s/update_critic_actor')
    else:
        if 'timing_s/update_critic' in metrics:
            components.append('timing_s/update_critic')
        if 'timing_s/update_actor' in metrics:
            components.append('timing_s/update_actor')

    # Filter components that exist in metrics
    available_components = [c for c in components if c in metrics]

    if not available_components:
        print("Warning: No timing_s components found in metrics")
        return

    # Get step indices
    steps = list(range(1, len(metrics[available_components[0]]) + 1))

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(steps))
    colors = plt.cm.Set3(np.linspace(0, 1, len(available_components)))

    for idx, component in enumerate(available_components):
        values = metrics[component]
        label = component.replace('timing_s/', '').replace('_', ' ').title()
        ax.bar(steps, values, bottom=bottom, label=label, color=colors[idx], width=0.8)
        bottom += np.array(values)

    # Add total step time line if available
    if 'timing_s/step' in metrics:
        ax.plot(steps, metrics['timing_s/step'], 'k--', linewidth=2,
                marker='o', markersize=4, label='Total Step Time')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Training Step Timing Breakdown', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'step_timing_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'step_timing_breakdown.png'}")


def plot_per_token_timing(metrics: Dict[str, List[float]], output_dir: Path):
    """Plot per-token timing metrics.

    Args:
        metrics: Dictionary of parsed metrics
        output_dir: Directory to save plots
    """
    token_metrics = {k: v for k, v in metrics.items() if k.startswith('timing_per_token_ms/')}

    if not token_metrics:
        print("Warning: No timing_per_token_ms metrics found")
        return

    steps = list(range(1, len(list(token_metrics.values())[0]) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(token_metrics)))

    for idx, (metric_name, values) in enumerate(sorted(token_metrics.items())):
        label = metric_name.replace('timing_per_token_ms/', '').replace('_', ' ').title()
        ax.plot(steps, values, marker='o', linewidth=2, markersize=4,
                label=label, color=colors[idx])

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Time per Token (ms)', fontsize=12)
    ax.set_title('Per-Token Timing Metrics', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_token_timing.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'per_token_timing.png'}")


def plot_throughput(metrics: Dict[str, List[float]], output_dir: Path):
    """Plot throughput over time.

    Args:
        metrics: Dictionary of parsed metrics
        output_dir: Directory to save plots
    """
    if 'perf/throughput' not in metrics:
        print("Warning: perf/throughput metric not found")
        return

    steps = list(range(1, len(metrics['perf/throughput']) + 1))
    throughput = metrics['perf/throughput']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot throughput
    ax1.plot(steps, throughput, marker='o', linewidth=2, markersize=6,
             color='#2E86AB', label='Throughput')
    ax1.axhline(y=np.mean(throughput), color='r', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(throughput):.1f} tokens/s')
    ax1.fill_between(steps, throughput, alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Throughput (tokens/second)', fontsize=12)
    ax1.set_title('Training Throughput', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot time per step
    if 'perf/time_per_step' in metrics:
        time_per_step = metrics['perf/time_per_step']
        ax2.plot(steps, time_per_step, marker='s', linewidth=2, markersize=6,
                 color='#A23B72', label='Time per Step')
        ax2.axhline(y=np.mean(time_per_step), color='r', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(time_per_step):.2f} s')
        ax2.fill_between(steps, time_per_step, alpha=0.3, color='#A23B72')
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        ax2.set_title('Time per Training Step', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'throughput.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'throughput.png'}")


def plot_component_comparison(metrics: Dict[str, List[float]], output_dir: Path):
    """Plot comparison of major training components.

    Args:
        metrics: Dictionary of parsed metrics
        output_dir: Directory to save plots
    """
    # Major components to compare
    components = {
        'Generation': 'timing_s/gen',
    }

    # Handle update metrics: prefer combined update_critic_actor if available,
    # otherwise use separate update_critic and update_actor
    if 'timing_s/update_critic_actor' in metrics:
        components['Critic+Actor Update'] = 'timing_s/update_critic_actor'
    else:
        if 'timing_s/update_actor' in metrics:
            components['Actor Update'] = 'timing_s/update_actor'
        if 'timing_s/update_critic' in metrics:
            components['Critic Update'] = 'timing_s/update_critic'

    # Filter available components
    available = {k: v for k, v in components.items() if v in metrics}

    if not available:
        print("Warning: No major components found for comparison")
        return

    steps = list(range(1, len(metrics[list(available.values())[0]]) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        'Generation': '#F18F01',
        'Actor Update': '#2E86AB',
        'Critic Update': '#A23B72',
        'Critic+Actor Update': '#6A4C93'
    }

    for component_name, metric_key in available.items():
        values = metrics[metric_key]
        ax.plot(steps, values, marker='o', linewidth=2.5, markersize=6,
                label=f'{component_name} (avg: {np.mean(values):.2f}s)',
                color=colors.get(component_name, None))

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Training Component Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'component_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'component_comparison.png'}")


def plot_generation_timing(metrics: Dict[str, List[float]], output_dir: Path):
    """Plot generation timing details.

    Args:
        metrics: Dictionary of parsed metrics
        output_dir: Directory to save plots
    """
    gen_metrics = {
        'Total Gen Time': 'timing_s/gen',
        'Generate Sequences': 'timing_s/generate_sequences',
        'Old Log Prob': 'timing_s/old_log_prob',
    }

    available = {k: v for k, v in gen_metrics.items() if v in metrics}

    if not available:
        print("Warning: No generation timing metrics found")
        return

    steps = list(range(1, len(metrics[list(available.values())[0]]) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))

    for metric_name, metric_key in available.items():
        values = metrics[metric_key]
        ax.plot(steps, values, marker='o', linewidth=2, markersize=5,
                label=f'{metric_name} (avg: {np.mean(values):.2f}s)')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Generation Phase Timing Breakdown', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'generation_timing.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'generation_timing.png'}")


def print_summary_statistics(metrics: Dict[str, List[float]]):
    """Print summary statistics for key timing metrics.

    Args:
        metrics: Dictionary of parsed metrics
    """
    print("\n" + "="*70)
    print("TIMING SUMMARY STATISTICS")
    print("="*70)

    key_metrics = [
        'timing_s/step',
        'timing_s/gen',
        'perf/throughput',
        'perf/time_per_step',
    ]

    # Add update metrics based on what's available
    if 'timing_s/update_critic_actor' in metrics:
        key_metrics.append('timing_s/update_critic_actor')
    else:
        if 'timing_s/update_actor' in metrics:
            key_metrics.append('timing_s/update_actor')
        if 'timing_s/update_critic' in metrics:
            key_metrics.append('timing_s/update_critic')

    for metric in key_metrics:
        if metric in metrics:
            values = np.array(metrics[metric])
            print(f"\n{metric}:")
            print(f"  Mean:   {np.mean(values):.4f}")
            print(f"  Std:    {np.std(values):.4f}")
            print(f"  Min:    {np.min(values):.4f}")
            print(f"  Max:    {np.max(values):.4f}")
            print(f"  Median: {np.median(values):.4f}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Plot timing metrics from verl training logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot metrics from a log file
  python plot_timing_metrics.py ../verl_demo.log

  # Specify custom output directory
  python plot_timing_metrics.py ../verl_demo.log --output-dir ./my_plots
        """
    )
    parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same directory as log file)')

    args = parser.parse_args()

    # Parse paths
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return 1

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = log_path.parent / 'plot'

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing log file: {log_path}")
    metrics = parse_log_file(log_path)

    if not metrics:
        print("Error: No timing metrics found in log file")
        return 1

    print(f"Found {len(metrics)} metrics")
    print(f"Output directory: {output_dir}")
    print()

    # Generate all plots
    print("Generating plots...")
    plot_step_timing_breakdown(metrics, output_dir)
    plot_per_token_timing(metrics, output_dir)
    plot_throughput(metrics, output_dir)
    plot_component_comparison(metrics, output_dir)
    plot_generation_timing(metrics, output_dir)

    # Print summary statistics
    print_summary_statistics(metrics)

    print(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
