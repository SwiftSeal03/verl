# Timing Metrics Plotting

This directory contains scripts for visualizing timing metrics from verl training logs.

## Usage

### Basic Usage

```bash
python plot_timing_metrics.py /path/to/logfile.log
```

This will:
1. Parse the log file for timing metrics
2. Generate multiple plots in the same directory as the log file (in a `plot/` subdirectory)
3. Print summary statistics

### Custom Output Directory

```bash
python plot_timing_metrics.py /path/to/logfile.log --output-dir ./my_plots
```

### Example

From the verl root directory:
```bash
cd profiling/plot
python plot_timing_metrics.py ../../verl_demo.log
```

## Generated Plots

The script generates the following visualizations:

### 1. `step_timing_breakdown.png`
- **Stacked bar chart** showing time breakdown for each training step
- Components include:
  - Generation (`gen`)
  - Reward computation (`reward`)
  - Old log probability computation (`old_log_prob`)
  - Value computation (`values`)
  - Advantage computation (`adv`)
  - Critic update (`update_critic`)
  - Actor update (`update_actor`)
- Shows total step time as a line overlay

### 2. `per_token_timing.png`
- **Line plot** showing per-token timing metrics over training steps
- Metrics shown:
  - `gen`: Generation time per token
  - `update_actor`: Actor update time per token
  - `update_critic`: Critic update time per token
  - `values`: Value computation time per token
  - `adv`: Advantage computation time per token

### 3. `throughput.png`
- **Two subplots**:
  - Top: Training throughput (tokens/second) over time
  - Bottom: Time per training step
- Includes mean lines for reference

### 4. `component_comparison.png`
- **Line plot** comparing major training components:
  - Generation phase
  - Actor update phase
  - Critic update phase
- Shows average time for each component

### 5. `generation_timing.png`
- **Detailed breakdown** of the generation phase:
  - Total generation time
  - Sequence generation time
  - Old log probability computation time

## Timing Metrics Available

### Absolute Time Metrics (`timing_s/`)
- `gen`: Total generation phase time
- `generate_sequences`: Time to generate response sequences
- `reward`: Time to compute rewards
- `old_log_prob`: Time to compute log probabilities from policy
- `values`: Time to compute value estimates (critic)
- `adv`: Time to compute advantages
- `update_critic`: Time to update critic model
- `update_actor`: Time to update actor model
- `step`: Total time per training step
- `testing`: Time for validation/testing

### Per-Token Metrics (`timing_per_token_ms/`)
- All metrics normalized by the number of tokens processed
- Useful for comparing efficiency across different batch sizes

### Performance Metrics (`perf/`)
- `throughput`: Tokens processed per second
- `time_per_step`: Total time per training step
- `total_num_tokens`: Total tokens processed in the step

## Requirements

The script requires:
- Python 3.7+
- matplotlib
- numpy

Install with:
```bash
pip install matplotlib numpy
```

## Customization

To customize the plots, edit `plot_timing_metrics.py`:
- **Colors**: Modify color schemes in each plotting function
- **Metrics**: Add/remove metrics by editing the component lists
- **Layout**: Adjust figure sizes and subplot arrangements
- **Statistics**: Add custom metrics to `print_summary_statistics()`

## Output Example

```
Parsing log file: /home/ubuntu/verl/verl_demo.log
Found 34 metrics
Output directory: /home/ubuntu/verl/plot

Generating plots...
Saved: /home/ubuntu/verl/plot/step_timing_breakdown.png
Saved: /home/ubuntu/verl/plot/per_token_timing.png
Saved: /home/ubuntu/verl/plot/throughput.png
Saved: /home/ubuntu/verl/plot/component_comparison.png
Saved: /home/ubuntu/verl/plot/generation_timing.png

======================================================================
TIMING SUMMARY STATISTICS
======================================================================

timing_s/step:
  Mean:   41.5123
  Std:    0.0892
  Min:    41.3421
  Max:    41.7697
  Median: 41.5414

...

All plots saved to: /home/ubuntu/verl/plot
```

## Troubleshooting

### No metrics found
- Ensure the log file contains lines with `timing_s/`, `timing_per_token_ms/`, or `perf/` metrics
- Check that the log format matches the expected pattern: `metric_name:value`

### Missing plots
- Check warning messages - some plots are only generated if specific metrics exist
- For GRPO training without critic, `update_critic` plots will be empty

### Import errors
- Ensure matplotlib and numpy are installed: `pip install matplotlib numpy`
