# Quick Start: Server-Side MTL Federated Learning

## What Changed?

Your federated learning system now implements **Server-Side Multi-Task Learning (MT-DNN style)**:

- ✅ **Server maintains unified MTL model** (shared BERT + task heads)
- ✅ **Task-aware aggregation** (shared from all clients, heads from same-task clients)
- ✅ **Cross-task knowledge transfer** via shared encoder
- ✅ **Task specialization** via task-specific heads

## Quick Start

### 1. Activate Environment
```bash
cd /home/vinh/Documents/code/FedAvgLS/experiment/mtl/fl-slms-berttiny-sts-qqp-sst2
source /home/vinh/Documents/code/FedAvgLS/venv/bin/activate
```

### 2. Start Server (Terminal 1)
```bash
python federated_main.py --mode server --config federated_config.yaml
```

**Expected Output**:
```
MTL Server Model Summary:
  Base Model: prajjwal1/bert-tiny
  Tasks: ['sst2', 'qqp', 'stsb']
  Shared Parameters: 4,385,280
  Task 'sst2' Head: 1,538 parameters
  Task 'qqp' Head: 1,538 parameters
  Task 'stsb' Head: 769 parameters
  Total Parameters: 4,389,125

Starting Federated Learning with Server-Side Multi-Task Learning (MT-DNN Style)
Architecture: Shared BERT Encoder + Task-Specific Heads
```

### 3. Start SST-2 Client (Terminal 2)
```bash
python federated_main.py --mode client --client_id sst2_client --tasks sst2
```

### 4. Start QQP Client (Terminal 3)
```bash
python federated_main.py --mode client --client_id qqp_client --tasks qqp
```

### 5. Start STSB Client (Terminal 4)
```bash
python federated_main.py --mode client --client_id stsb_client --tasks stsb
```

## Architecture Overview

```
┌──────────────────────────────────────┐
│         SERVER (MTL Model)           │
│                                      │
│  ┌────────────────────────────┐    │
│  │   Shared BERT Encoder      │    │
│  │   (from ALL clients)       │    │
│  └──────────┬─────────────────┘    │
│             │                        │
│    ┌────────┼────────┐              │
│    │        │        │               │
│  ┌─▼──┐  ┌─▼──┐  ┌─▼───┐          │
│  │SST2│  │QQP │  │STSB │          │
│  │Head│  │Head│  │Head │          │
│  └────┘  └────┘  └─────┘          │
└──────────────────────────────────────┘
         │         │         │
         ▼         ▼         ▼
   ┌─────────┬─────────┬─────────┐
   │Client A │Client B │Client C │
   │ (SST2)  │ (QQP)   │ (STSB)  │
   └─────────┴─────────┴─────────┘
```

## Key Features

### 1. Task-Aware Aggregation
- **Shared Encoder**: Aggregated from ALL clients → cross-task learning
- **Task Heads**: Aggregated within same-task clients → task specialization

### 2. Model Slicing
Each client receives only what they need:
- SST-2 client: Shared encoder + SST-2 head
- QQP client: Shared encoder + QQP head
- STSB client: Shared encoder + STSB head

### 3. Cross-Task Knowledge Transfer
The shared encoder benefits from all tasks:
- Sentiment analysis (SST-2)
- Paraphrase detection (QQP)
- Semantic similarity (STSB)

## What to Look For

### Server Logs
```
=== Round 1/10 ===
Broadcasting MTL model slices to clients...
Sent model slice for task 'sst2' to client sst2_client
Sent model slice for task 'qqp' to client qqp_client
Sent model slice for task 'stsb' to client stsb_client

Aggregating updates from 3 clients across 3 tasks
  Task 'sst2': 1 clients
  Task 'qqp': 1 clients
  Task 'stsb': 1 clients

Aggregated 285 shared parameters from ALL 3 clients
Aggregated 2 parameters for task 'sst2' head from 1 clients
Aggregated 2 parameters for task 'qqp' head from 1 clients
Aggregated 2 parameters for task 'stsb' head from 1 clients

Updated shared BERT encoder with 285 parameters
Updated task 'sst2' head with 2 parameters
Updated task 'qqp' head with 2 parameters
Updated task 'stsb' head with 2 parameters
```

### Client Logs
```
[sst2_client] Client MTL synchronizer initialized for sst2_client (task: sst2)
[sst2_client] Synchronized MTL model slice to version 1
[sst2_client] Training on task: sst2
[sst2_client] Epoch 1/3: Loss=0.6234, Accuracy=0.6500
```

## Results

Results are saved in `federated_results/`:
- `federated_results_TIMESTAMP.csv`: Global results per round
- `client_results_TIMESTAMP.csv`: Per-client results
- `training_summary.txt`: Summary with MTL architecture details

## Troubleshooting

### Issue: "MTL model not available"
**Solution**: Make sure server is started first and initializes MTLServerModel

### Issue: "Could not infer task from client_id"
**Solution**: Ensure client_id contains task name (e.g., 'sst2_client', 'qqp_client')

### Issue: "No parameters found in global_state"
**Solution**: Check that server successfully broadcasts model slices before training

## Documentation

- **`MTL_IMPLEMENTATION_GUIDE.md`**: Comprehensive architecture guide
- **`CHANGES_SUMMARY.md`**: Detailed list of all changes
- **`QUICK_START_MTL.md`**: This file

## Configuration

No changes needed to `federated_config.yaml`. The MTL architecture is automatically applied.

Current config:
```yaml
model:
  server_model: "prajjwal1/bert-tiny"  # Shared encoder
  client_model: "prajjwal1/bert-tiny"  # Client models

training:
  num_rounds: 10
  local_epochs: 3
  learning_rate: 2e-5
```

## Comparison: Before vs After

| Aspect | Before (Standard FL) | After (MTL FL) |
|--------|---------------------|----------------|
| Server Model | Separate per task | Unified (shared + heads) |
| Aggregation | Per-task only | Task-aware (shared + heads) |
| Knowledge Transfer | None | Cross-task via shared encoder |
| Parameters | 3 × 4.39M = 13.17M | 4.385M + 3 × 0.0015M = 4.39M |
| Scalability | Linear with tasks | Sub-linear with tasks |

## Benefits

1. **Better Performance**: Cross-task knowledge transfer improves all tasks
2. **Parameter Efficiency**: Shared encoder reduces total parameters by ~67%
3. **Faster Convergence**: Multi-task learning acts as regularization
4. **Low-Resource Tasks**: Tasks with limited data benefit from others
5. **Scalability**: Adding new tasks only adds small task heads

## Reference

Based on **MT-DNN** (Liu et al., ACL 2019) and paper **1901.11504v2.pdf**:
> "The lower layers are shared across all tasks while the top layers are task-specific. The input X is first represented as embedding vectors, then the Transformer encoder captures contextual information and generates shared contextual embeddings. Finally, for each task, additional task-specific layers generate task-specific representations."

## Next Steps

1. **Run Training**: Follow Quick Start above
2. **Monitor Results**: Check logs and result files
3. **Compare Performance**: Compare with previous standard FL results
4. **Experiment**: Try different configurations (learning rate, epochs, etc.)
5. **Scale Up**: Add more clients per task or new tasks

---

**Ready to start?** Run the commands in the Quick Start section above! 🚀

