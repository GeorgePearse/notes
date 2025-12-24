# NCCL

**N**VIDIA **C**ollective **C**ommunications **L**ibrary - the backbone of multi-GPU training.

## What It Does

NCCL handles GPU-to-GPU communication:

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  GPU 0  │────│  GPU 1  │────│  GPU 2  │────│  GPU 3  │
│ grads_0 │     │ grads_1 │     │ grads_2 │     │ grads_3 │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
     │               │               │               │
     └───────────────┴───────────────┴───────────────┘
                           │
                     NCCL AllReduce
                           │
                           ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  GPU 0  │     │  GPU 1  │     │  GPU 2  │     │  GPU 3  │
│sum(grad)│     │sum(grad)│     │sum(grad)│     │sum(grad)│
└─────────┘     └─────────┘     └─────────┘     └─────────┘
```

Without NCCL, you'd have to copy everything through CPU - 10-100x slower.

## Collective Operations

### AllReduce (most common)
Sum tensors across GPUs, result on all GPUs:

```python
# Every GPU has gradients, need sum on all GPUs
# This is what DDP does every backward pass

# Before: GPU0=[1,2], GPU1=[3,4], GPU2=[5,6], GPU3=[7,8]
# After:  GPU0=[16,20], GPU1=[16,20], GPU2=[16,20], GPU3=[16,20]
```

### AllGather
Gather tensors from all GPUs onto all GPUs:

```python
# Used by FSDP to reconstruct weights before forward

# Before: GPU0=[A], GPU1=[B], GPU2=[C], GPU3=[D]
# After:  GPU0=[A,B,C,D], GPU1=[A,B,C,D], GPU2=[A,B,C,D], GPU3=[A,B,C,D]
```

### ReduceScatter
Reduce and scatter result (inverse of AllGather):

```python
# Used by FSDP after backward pass

# Before: GPU0=[1,2,3,4], GPU1=[5,6,7,8], GPU2=[1,1,1,1], GPU3=[1,1,1,1]
# After:  GPU0=[8], GPU1=[10], GPU2=[12], GPU3=[14]  # Each gets 1/4 of sum
```

### Broadcast
Send from one GPU to all:

```python
# Used for distributing model weights at start of training

# Before: GPU0=[model], GPU1=[], GPU2=[], GPU3=[]
# After:  GPU0=[model], GPU1=[model], GPU2=[model], GPU3=[model]
```

## Hardware Topology Awareness

NCCL automatically optimizes for your hardware:

```
Single Node (NVLink):
┌─────────────────────────────────────┐
│  GPU0 ══NVLink══ GPU1               │
│    ║               ║                │
│  GPU2 ══NVLink══ GPU3               │
└─────────────────────────────────────┘
Bandwidth: 600 GB/s (NVLink 4.0)

Multi-Node (InfiniBand):
┌──────────────┐         ┌──────────────┐
│   Node 0     │         │   Node 1     │
│  GPU0  GPU1  │════════│  GPU0  GPU1  │
│  GPU2  GPU3  │   IB    │  GPU2  GPU3  │
└──────────────┘         └──────────────┘
Bandwidth: 400 Gb/s (HDR InfiniBand)

DGX with NVSwitch:
┌─────────────────────────────────────┐
│         NVSwitch Fabric             │
│  ┌─────┬─────┬─────┬─────┐         │
│  │GPU0 │GPU1 │GPU2 │GPU3 │         │
│  │GPU4 │GPU5 │GPU6 │GPU7 │         │
│  └─────┴─────┴─────┴─────┘         │
└─────────────────────────────────────┘
All-to-all full bandwidth
```

NCCL's ring/tree algorithms adapt to topology automatically.

## PyTorch Integration

You rarely call NCCL directly - PyTorch wraps it:

```python
# DDP - most common
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank]
)
# Every loss.backward() triggers NCCL AllReduce

# FSDP - for large models
model = torch.distributed.fsdp.FullyShardedDataParallel(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)
# Uses AllGather (forward) + ReduceScatter (backward)

# Direct collective ops
torch.distributed.all_reduce(tensor)
torch.distributed.all_gather(output_list, tensor)
torch.distributed.broadcast(tensor, src=0)
torch.distributed.reduce_scatter(output, input_list)
```

## Setup

```python
import torch.distributed as dist

# Initialize process group (usually in your training script)
dist.init_process_group(
    backend="nccl",           # Use NCCL for GPU
    init_method="env://",     # Get config from environment
    world_size=world_size,
    rank=rank
)

# Environment variables (set by torchrun/SLURM):
# MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK
```

```bash
# Launch distributed training
torchrun --nproc_per_node=4 train.py

# Multi-node
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py
```

## Performance Tuning

### Overlap communication with compute

```python
# DDP does this automatically with gradient buckets
model = DDP(model,
    broadcast_buffers=False,
    gradient_as_bucket_view=True,
    bucket_cap_mb=25  # Tune bucket size
)

# Gradients are AllReduced while next layer's backward runs
```

### NCCL environment variables

```bash
# Debug topology detection
export NCCL_DEBUG=INFO

# Force specific algorithms
export NCCL_ALGO=Ring  # or Tree, CollnetDirect, CollnetChain

# Tune buffer sizes
export NCCL_BUFFSIZE=4194304  # 4MB

# For InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

# For multi-node
export NCCL_SOCKET_IFNAME=eth0  # Network interface
```

### Check what's happening

```python
import torch.distributed as dist

# After init_process_group
if dist.get_rank() == 0:
    print(f"World size: {dist.get_world_size()}")
    print(f"Backend: {dist.get_backend()}")

# Profile NCCL ops
with torch.profiler.profile() as prof:
    dist.all_reduce(tensor)
print(prof.key_averages().table())
# Look for nccl:all_reduce
```

## Common Patterns

### Gradient Accumulation with DDP

```python
# Sync every N steps to reduce communication
for i, batch in enumerate(dataloader):
    with model.no_sync() if (i + 1) % accumulation_steps != 0 else nullcontext():
        loss = model(batch)
        loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Manual AllReduce

```python
# Average metric across GPUs
def reduce_mean(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor

avg_loss = reduce_mean(loss.detach())
```

## Debugging

```bash
# Hangs? Enable debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Common issues:
# 1. Mismatched tensor sizes across ranks
# 2. Different number of collective calls per rank
# 3. Firewall blocking ports (multi-node)
# 4. Wrong network interface selected
```

```python
# Sanity check: all ranks reach same point
def sync_check(msg):
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"All ranks passed: {msg}")

sync_check("after data load")
sync_check("after forward")
sync_check("after backward")
```

## NCCL vs Alternatives

| Backend | Use Case |
|---------|----------|
| **NCCL** | GPU collective ops (default for PyTorch + CUDA) |
| **Gloo** | CPU tensors, or when NCCL unavailable |
| **MPI** | Legacy, HPC environments |

NCCL is almost always correct for GPU training. Use Gloo only for CPU-only distributed or as fallback.
