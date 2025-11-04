# Split GPU Placement for PPO Training

This directory contains scripts for running PPO training with split GPU placement, where actor and critic models are placed on separate sets of GPUs.

## Overview

**Standard Placement (`demo.sh`):**
- All models (Actor, Rollout, Ref, Critic) share all 4 GPUs
- Good for: Maximum flexibility, dynamic load balancing
- Trade-off: Potential GPU memory contention

**Split Placement (`demo_split.sh`):**
- Actor/Rollout/Ref: GPUs 0-1 (2 GPUs)
- Critic: GPUs 2-3 (2 GPUs)
- Good for: Reduced memory contention, more predictable performance
- Trade-off: Less flexibility in resource allocation

## Files

### `demo_split.sh`
Modified demo script that uses split GPU placement.

### `verl/trainer/main_ppo_split.py`
Custom PPO trainer implementation that creates separate resource pools for actor and critic.

## How It Works

### Resource Pool Architecture

The split placement creates two separate Ray resource pools:

```python
resource_pool_spec = {
    "actor_rollout_ref_pool": [2, 2, ...],  # 2 GPUs per node
    "critic_pool": [2, 2, ...]              # 2 GPUs per node
}

mapping = {
    Role.ActorRollout: "actor_rollout_ref_pool",
    Role.RefPolicy: "actor_rollout_ref_pool",
    Role.Critic: "critic_pool"
}
```

### GPU Allocation Logic

For `n_gpus_per_node=4` and `nnodes=1`:
- Actor/Rollout/Ref pool: 2 GPUs (GPUs 0-1)
- Critic pool: 2 GPUs (GPUs 2-3)

For odd number of GPUs (e.g., `n_gpus_per_node=3`):
- Actor/Rollout/Ref pool: 2 GPUs
- Critic pool: 1 GPU

## Usage

### Run with Split Placement

```bash
cd /home/ubuntu/verl/profiling
bash demo_split.sh
```

### Compare with Standard Placement

```bash
# Standard placement (all models share GPUs)
bash demo.sh

# Split placement (actor and critic on separate GPUs)
bash demo_split.sh
```

### Customize GPU Split

Edit `main_ppo_split.py` to change the allocation:

```python
# Example: 3 GPUs for actor, 1 GPU for critic
actor_gpus = 3
critic_gpus = 1
resource_pool_spec = {
    actor_rollout_ref_pool_id: [actor_gpus] * nnodes,
    critic_pool_id: [critic_gpus] * nnodes,
}
```

## Configuration Parameters

All standard PPO configuration parameters work with split placement:

```bash
trainer.n_gpus_per_node=4    # Total GPUs available
trainer.nnodes=1              # Number of nodes
actor_rollout_ref.rollout.tensor_model_parallel_size=1  # TP for rollout
```

## Benefits of Split Placement

### 1. Reduced Memory Contention
- Actor and critic don't compete for the same GPU memory
- More predictable memory usage per model

### 2. Better Performance Isolation
- Actor update and critic update can run concurrently without interference
- Easier to profile and optimize each component separately

### 3. Clearer Resource Attribution
- Can monitor GPU utilization per model type
- Easier to identify bottlenecks

## Trade-offs

### Advantages
- ✅ Reduced memory contention
- ✅ Better performance predictability
- ✅ Easier debugging and profiling
- ✅ Can optimize each pool independently

### Disadvantages
- ❌ Less flexible resource allocation
- ❌ May underutilize GPUs if workload is imbalanced
- ❌ Cannot dynamically shift resources based on load

## Performance Comparison

### Expected Metrics

**Memory Usage:**
- Standard: Higher peak memory (all models on same GPUs)
- Split: Lower peak per GPU, more balanced across GPUs

**Throughput:**
- Similar overall throughput if GPUs are well-utilized
- May vary based on workload characteristics

**Timing:**
- Look for reduced `timing_s/update_actor` and `timing_s/update_critic` if memory contention was an issue

### Monitoring

Check GPU utilization:
```bash
# During training, monitor in another terminal
watch -n 1 nvidia-smi
```

Expected pattern:
- GPUs 0-1: High utilization during generation and actor update
- GPUs 2-3: High utilization during critic update

## Advanced: Multi-Node Split Placement

For multi-node setups:

```bash
# 2 nodes, 8 GPUs per node, split evenly
trainer.n_gpus_per_node=8
trainer.nnodes=2

# This creates:
# Node 0: Actor pool (GPUs 0-3), Critic pool (GPUs 4-7)
# Node 1: Actor pool (GPUs 0-3), Critic pool (GPUs 4-7)
```

## Troubleshooting

### Issue: OOM (Out of Memory) on actor pool
**Solution:** Reduce batch sizes or tensor parallelism for actor:
```bash
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2  # Reduce from 4
actor_rollout_ref.rollout.tensor_model_parallel_size=2  # Enable TP
```

### Issue: Critic pool underutilized
**Solution:** This is normal - critic updates are typically faster than actor updates. Consider allocating fewer GPUs to critic.

### Issue: "Not enough GPUs available"
**Solution:** Ensure you have at least 2 GPUs per pool (4 total for split placement).

## Visualization

After training, plot timing metrics to compare:

```bash
cd profiling/plot
python plot_timing_metrics.py ../verl_demo.log --output-dir ./standard_placement
python plot_timing_metrics.py ../verl_demo_split.log --output-dir ./split_placement

# Compare component timing between standard and split placement
```

## See Also

- [Standard PPO Training](demo.sh)
- [Split Placement Example](../examples/split_placement/)
- [Resource Pool Documentation](../docs/workers/ray_trainer.rst)
- [Profiling Guide](plot/README.md)
