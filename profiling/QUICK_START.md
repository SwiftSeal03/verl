# Quick Start: Standard vs Split GPU Placement

## TL;DR

```bash
# Standard placement (all models share all GPUs)
bash demo.sh

# Split placement (actor on GPUs 0-1, critic on GPUs 2-3)
bash demo_split.sh
```

## When to Use Split Placement

### Use Split Placement If:
- ✅ You have GPU OOM issues with standard placement
- ✅ You want more predictable memory usage
- ✅ You're profiling and need clear GPU attribution
- ✅ Your actor and critic are similar sizes
- ✅ You have at least 4 GPUs

### Use Standard Placement If:
- ✅ Your actor and critic have very different memory needs
- ✅ You want maximum flexibility
- ✅ You have limited GPUs (< 4)
- ✅ Memory usage is not an issue

## Visual Comparison

### Standard: All models share GPUs
```
┌─────────────────────┐
│  4 GPUs (shared)    │
│  Actor + Critic     │
│  compete for        │
│  same memory        │
└─────────────────────┘
```

### Split: Separate GPU pools
```
┌──────────┐  ┌──────────┐
│ 2 GPUs   │  │ 2 GPUs   │
│ Actor    │  │ Critic   │
│ Rollout  │  │          │
│ Ref      │  │          │
└──────────┘  └──────────┘
```

## Key Differences

| Aspect | Standard | Split |
|--------|----------|-------|
| **Command** | `main_ppo` | `main_ppo_split` |
| **GPU Usage** | All share all GPUs | Separate pools |
| **Memory** | Shared, higher peak | Isolated, balanced |
| **Flexibility** | High | Medium |
| **Predictability** | Medium | High |
| **Setup** | Simple | Simple |

## Performance Expectations

### Memory Usage
- **Standard:** ~13.5 GB peak on GPUs 0-3
- **Split:** ~10 GB peak on GPUs 0-1, ~8 GB peak on GPUs 2-3

### Throughput
- **Both:** ~500 tokens/second (similar performance)
- Split may be slightly faster if memory contention was limiting standard

### Timing Breakdown
Both should show similar timing for:
- Generation: ~5.5s
- Actor update: ~14.7s
- Critic update: ~15.0s

## Monitoring

During training, open another terminal:

```bash
watch -n 1 nvidia-smi
```

**Standard placement:**
```
+-----------------------------------------------------------------------------+
| GPU  Name        Memory-Usage  Utilization |
|=============================================================================|
|   0  GPU         13GB / 40GB   85%         |  All GPUs used
|   1  GPU         13GB / 40GB   82%         |  by all models
|   2  GPU         13GB / 40GB   88%         |
|   3  GPU         13GB / 40GB   80%         |
+-----------------------------------------------------------------------------+
```

**Split placement:**
```
+-----------------------------------------------------------------------------+
| GPU  Name        Memory-Usage  Utilization |
|=============================================================================|
|   0  GPU         10GB / 40GB   95%         |  Actor pool
|   1  GPU         10GB / 40GB   93%         |  (generation + updates)
|   2  GPU          8GB / 40GB   85%         |  Critic pool
|   3  GPU          8GB / 40GB   82%         |  (updates)
+-----------------------------------------------------------------------------+
```

## Configuration

Both use the same config parameters:

```bash
data.train_batch_size=256
actor_rollout_ref.actor.ppo_mini_batch_size=64
critic.ppo_micro_batch_size_per_gpu=4
trainer.n_gpus_per_node=4  # Total GPUs
```

The only difference is which `main_ppo` module you run.

## Troubleshooting

### "Not enough GPUs"
- **Standard:** Need at least 1 GPU
- **Split:** Need at least 4 GPUs (2 per pool)

### OOM on actor pool (split placement)
```bash
# Reduce actor batch size
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2  # was 4

# Or reduce rollout batch size
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4  # was 8
```

### OOM on critic pool (split placement)
```bash
# Reduce critic batch size
critic.ppo_micro_batch_size_per_gpu=2  # was 4
```

## Next Steps

After running:

1. **Check logs:**
   ```bash
   tail -100 verl_demo_split.log
   ```

2. **Plot metrics:**
   ```bash
   cd plot
   python plot_timing_metrics.py ../verl_demo_split.log
   ```

3. **Compare with standard:**
   ```bash
   # Run both and compare
   bash demo.sh
   bash demo_split.sh

   # Compare throughput
   grep "perf/throughput" verl_demo.log
   grep "perf/throughput" verl_demo_split.log
   ```

## Documentation

- Full details: [README_SPLIT.md](README_SPLIT.md)
- Changes made: [CHANGES.md](CHANGES.md)
- Plot timing: [plot/README.md](plot/README.md)
