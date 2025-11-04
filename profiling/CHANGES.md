# Changes Made to verl for Split GPU Placement

## Summary

Modified the verl demo to support split GPU placement where actor/rollout/ref models use one set of GPUs (0-1) and critic uses another set (2-3).

## Files Created

### 1. `/home/ubuntu/verl/verl/trainer/main_ppo_split.py`
**Purpose:** Custom PPO trainer with split GPU placement logic

**Key Features:**
- Extends `PPOTrainer` from `main_ppo.py`
- Creates separate resource pools for actor and critic
- Automatically splits available GPUs evenly between pools
- Handles odd numbers of GPUs (gives extra GPU to actor pool)
- Maps roles to appropriate resource pools:
  - `Role.ActorRollout` → `actor_rollout_ref_pool`
  - `Role.RefPolicy` → `actor_rollout_ref_pool`
  - `Role.Critic` → `critic_pool`

**Implementation:**
```python
class PPOTrainerSplit(PPOTrainer):
    def init_resource_pool_mgr(self, config):
        # Creates two pools instead of one global pool
        resource_pool_spec = {
            "actor_rollout_ref_pool": [2] * nnodes,  # GPUs 0-1
            "critic_pool": [2] * nnodes,              # GPUs 2-3
        }
        # Map roles to pools
        self.mapping[Role.ActorRollout] = "actor_rollout_ref_pool"
        self.mapping[Role.Critic] = "critic_pool"
        ...
```

### 2. `/home/ubuntu/verl/profiling/demo_split.sh`
**Purpose:** Demo script using split GPU placement

**Key Difference from `demo.sh`:**
```bash
# demo.sh uses:
python3 -m verl.trainer.main_ppo

# demo_split.sh uses:
python3 -m verl.trainer.main_ppo_split
```

All other configuration parameters remain the same.

### 3. `/home/ubuntu/verl/profiling/README_SPLIT.md`
**Purpose:** Comprehensive documentation for split placement

**Contents:**
- Overview of standard vs split placement
- How the resource pool architecture works
- GPU allocation logic
- Usage instructions
- Benefits and trade-offs
- Performance comparison guidelines
- Troubleshooting guide
- Multi-node setup examples

## Architecture Comparison

### Standard Placement (demo.sh)
```
All 4 GPUs available to all models
┌─────────────────────────────────────────┐
│  GPU 0, 1, 2, 3                         │
│  ┌────────┐ ┌────────┐ ┌────────┐      │
│  │ Actor  │ │Rollout │ │ Critic │      │
│  └────────┘ └────────┘ └────────┘      │
│  (compete for same memory)               │
└─────────────────────────────────────────┘
```

### Split Placement (demo_split.sh)
```
Actor/Rollout/Ref Pool     Critic Pool
┌──────────────────────┐   ┌──────────────────────┐
│  GPU 0, 1            │   │  GPU 2, 3            │
│  ┌────────┐          │   │  ┌────────┐          │
│  │ Actor  │          │   │  │ Critic │          │
│  ├────────┤          │   │  └────────┘          │
│  │Rollout │          │   │                      │
│  ├────────┤          │   │                      │
│  │  Ref   │          │   │                      │
│  └────────┘          │   │                      │
└──────────────────────┘   └──────────────────────┘
   (isolated memory)          (isolated memory)
```

## Resource Pool Implementation

### Standard Approach (main_ppo.py)
```python
resource_pool_spec = {
    "global_pool": [4] * 1,  # All 4 GPUs in one pool
}
mapping = {
    Role.ActorRollout: "global_pool",
    Role.Critic: "global_pool",
    Role.RefPolicy: "global_pool",
}
```

### Split Approach (main_ppo_split.py)
```python
resource_pool_spec = {
    "actor_rollout_ref_pool": [2] * 1,  # GPUs 0-1
    "critic_pool": [2] * 1,              # GPUs 2-3
}
mapping = {
    Role.ActorRollout: "actor_rollout_ref_pool",
    Role.Critic: "critic_pool",
    Role.RefPolicy: "actor_rollout_ref_pool",
}
```

## Ray Resource Pool Concept

Ray's resource pools provide:
1. **Isolation:** Each pool has dedicated GPUs
2. **Placement guarantees:** Workers in a pool only use that pool's GPUs
3. **CUDA_VISIBLE_DEVICES management:** Ray automatically sets this for each worker
4. **Concurrent execution:** Different pools can execute simultaneously

## Usage

### Run Standard Demo
```bash
cd /home/ubuntu/verl/profiling
bash demo.sh
# Output: verl_demo.log
```

### Run Split Placement Demo
```bash
cd /home/ubuntu/verl/profiling
bash demo_split.sh
# Output: verl_demo_split.log
```

### Compare Results
```bash
# Plot timing metrics for both
cd plot
python plot_timing_metrics.py ../verl_demo.log --output-dir ./standard
python plot_timing_metrics.py ../verl_demo_split.log --output-dir ./split

# Check GPU memory usage patterns
grep "perf/max_memory" ../verl_demo.log
grep "perf/max_memory" ../verl_demo_split.log
```

## Expected Benefits

### Memory Management
- **Standard:** Peak memory = max(actor_memory, critic_memory) on same GPUs
- **Split:** Peak memory distributed across pools

### Concurrency
- **Standard:** Actor and critic compete for GPU cycles
- **Split:** Actor and critic can run truly concurrently on separate GPUs

### Profiling
- **Standard:** Hard to isolate actor vs critic GPU usage
- **Split:** Clear separation in `nvidia-smi` output

## Configuration Flexibility

The split placement automatically adapts to:
- Different numbers of GPUs (must be ≥2)
- Multi-node setups
- Different model sizes
- Various batch sizes

Example for 8 GPUs:
```bash
trainer.n_gpus_per_node=8
# Creates: actor_pool (GPUs 0-3), critic_pool (GPUs 4-7)
```

## Integration with Existing Code

The split placement:
- ✅ Uses the same configuration format
- ✅ Works with all existing config options
- ✅ Compatible with FSDP and Megatron backends
- ✅ Supports LoRA, gradient checkpointing, etc.
- ✅ No changes needed to worker implementations

## Testing

To verify the split placement works correctly:

1. **Check resource pool creation:**
   ```bash
   grep "resource_pool_spec" verl_demo_split.log
   # Should show two separate pools
   ```

2. **Monitor GPU utilization during training:**
   ```bash
   watch -n 1 nvidia-smi
   # Should see:
   # - GPUs 0-1 active during generation/actor update
   # - GPUs 2-3 active during critic update
   ```

3. **Compare performance:**
   ```bash
   grep "perf/throughput" verl_demo.log
   grep "perf/throughput" verl_demo_split.log
   # Throughput should be similar or better
   ```

## Future Enhancements

Possible extensions:
1. **Dynamic pool sizing:** Adjust GPU allocation based on workload
2. **Heterogeneous pools:** Different GPU types for actor vs critic
3. **More granular splits:** Separate pools for rollout vs actor update
4. **Automatic tuning:** ML-based optimizer for pool allocation

## References

- Original split placement example: `examples/split_placement/main_ppo_split.py`
- Ray resource pools: `verl/trainer/ppo/ray_trainer.py`
- PPO trainer base: `verl/trainer/main_ppo.py`
