# Server-Side Multi-Task Learning (MTL) Implementation Guide

## Overview

This implementation follows the **MT-DNN (Multi-Task Deep Neural Networks)** architecture for federated learning, where:
- **Server maintains a unified MTL model** with shared lower layers and task-specific upper layers
- **Clients train on task-specific model slices** (shared encoder + their task head)
- **Aggregation is task-aware**: shared layers from all clients, task heads from same-task clients only

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SERVER (MTL Model)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Shared BERT Encoder (Lower Layers)         │    │
│  │    - Learns universal text representations         │    │
│  │    - Aggregated from ALL clients                   │    │
│  │    - Benefits from cross-task knowledge transfer   │    │
│  └────────────────────────────────────────────────────┘    │
│                          │                                   │
│           ┌──────────────┼──────────────┐                   │
│           │              │              │                    │
│  ┌────────▼───────┐ ┌───▼──────┐ ┌────▼──────────┐        │
│  │  SST-2 Head    │ │ QQP Head │ │  STSB Head    │        │
│  │ (Sentiment)    │ │ (Paraph.)│ │ (Similarity)  │        │
│  │ Binary Class.  │ │ Binary C.│ │ Regression    │        │
│  │ Aggregated     │ │ Aggreg.  │ │ Aggregated    │        │
│  │ from SST-2     │ │ from QQP │ │ from STSB     │        │
│  │ clients only   │ │ clients  │ │ clients only  │        │
│  └────────────────┘ └──────────┘ └───────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Model Slices
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    CLIENTS                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Client A (SST-2)         Client B (QQP)      Client C (STSB)│
│  ┌──────────────┐        ┌──────────────┐   ┌──────────────┐│
│  │ Shared BERT  │        │ Shared BERT  │   │ Shared BERT  ││
│  │    Encoder   │        │    Encoder   │   │    Encoder   ││
│  └──────┬───────┘        └──────┬───────┘   └──────┬───────┘│
│         │                       │                   │        │
│  ┌──────▼───────┐        ┌──────▼───────┐   ┌──────▼───────┐│
│  │  SST-2 Head  │        │   QQP Head   │   │  STSB Head   ││
│  └──────────────┘        └──────────────┘   └──────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. MTL Server Model (`src/models/mtl_server_model.py`)

**Purpose**: Unified model on server with shared encoder and task-specific heads

**Key Methods**:
- `get_shared_parameters()`: Returns shared BERT encoder parameters
- `get_task_head_parameters(task)`: Returns task-specific head parameters
- `get_model_slice_for_task(task)`: Returns model slice for a client (shared + task head)
- `set_shared_parameters(params)`: Updates shared encoder after aggregation
- `set_task_head_parameters(task, params)`: Updates task head after aggregation

**Architecture**:
```python
MTLServerModel:
  - bert: Shared BERT encoder (prajjwal1/bert-tiny)
  - task_heads: ModuleDict
      - 'sst2': Linear(hidden_size, 2)  # Binary classification
      - 'qqp': Linear(hidden_size, 2)   # Binary classification
      - 'stsb': Linear(hidden_size, 1)  # Regression
```

### 2. MTL Aggregator (`src/aggregation/mtl_aggregator.py`)

**Purpose**: Task-aware parameter aggregation

**Aggregation Strategy**:
1. **Shared Encoder**: FedAvg across ALL clients (universal representation learning)
2. **Task Heads**: FedAvg within same-task clients only (task specialization)

**Key Method**:
```python
aggregate_mtl_updates(client_updates) -> {
    'shared': {...},        # Aggregated shared parameters
    'task_heads': {
        'sst2': {...},      # Aggregated SST-2 head
        'qqp': {...},       # Aggregated QQP head
        'stsb': {...}       # Aggregated STSB head
    }
}
```

### 3. Federated Server (`src/core/federated_server.py`)

**Key Changes**:
- Initializes `MTLServerModel` instead of separate client models
- `broadcast_mtl_model_slices()`: Sends task-specific slices to clients
- Performs MTL-aware aggregation and updates model components separately

**Training Loop**:
```python
for round in rounds:
    1. Broadcast task-specific model slices to clients
    2. Clients train on their task data
    3. Collect client updates (with task labels)
    4. MTL-aware aggregation:
       - Aggregate shared encoder from ALL clients
       - Aggregate each task head from same-task clients
    5. Update MTL server model
```

### 4. Client Synchronization (`src/synchronization/federated_synchronization.py`)

**Key Changes**:
- `ClientModelSynchronizer` now accepts `task` parameter
- Handles MTL model slices (shared + task-specific)
- Backward compatible with legacy full model sync

### 5. Clients (`src/core/base_federated_client.py`, `src/core/federated_client.py`)

**Key Changes**:
- Include `task` label in update messages for MTL aggregation
- Synchronizer initialized with task information
- Receive and apply task-specific model slices

## Training Flow

### Round N Training Flow:

1. **Server → Clients (Model Distribution)**
   ```
   Server broadcasts task-specific model slices:
   - SST-2 clients: Shared BERT + SST-2 head
   - QQP clients: Shared BERT + QQP head
   - STSB clients: Shared BERT + STSB head
   ```

2. **Clients (Local Training)**
   ```
   Each client:
   - Receives model slice for their task
   - Trains on local task data
   - Computes gradients for both shared and task-specific layers
   ```

3. **Clients → Server (Update Upload)**
   ```
   Each client sends:
   - Updated parameters (shared + task head)
   - Task label (for aggregation)
   - Training metrics
   ```

4. **Server (MTL Aggregation)**
   ```
   Aggregator separates parameters:
   
   Shared Encoder:
   - Collects from: ALL clients (SST-2, QQP, STSB)
   - Aggregation: FedAvg across all
   - Benefit: Cross-task knowledge transfer
   
   SST-2 Head:
   - Collects from: SST-2 clients only
   - Aggregation: FedAvg within SST-2 clients
   - Benefit: Task-specific specialization
   
   QQP Head:
   - Collects from: QQP clients only
   - Aggregation: FedAvg within QQP clients
   
   STSB Head:
   - Collects from: STSB clients only
   - Aggregation: FedAvg within STSB clients
   ```

5. **Server (Model Update)**
   ```
   Update MTL model:
   - mtl_model.set_shared_parameters(aggregated_shared)
   - mtl_model.set_task_head_parameters('sst2', aggregated_sst2_head)
   - mtl_model.set_task_head_parameters('qqp', aggregated_qqp_head)
   - mtl_model.set_task_head_parameters('stsb', aggregated_stsb_head)
   ```

## Benefits of Server-Side MTL

### 1. **Cross-Task Knowledge Transfer**
- Shared encoder learns from all tasks simultaneously
- Universal text representations benefit all tasks
- Especially helpful for tasks with limited data

### 2. **Task Specialization**
- Task-specific heads maintain specialized knowledge
- No interference between task-specific decision boundaries
- Each task can have different output structures (classification vs regression)

### 3. **Parameter Efficiency**
- Shared encoder: ~4.4M parameters (shared across all tasks)
- Task heads: ~1.5K parameters each (task-specific)
- Total: Much smaller than separate models per task

### 4. **Improved Generalization**
- Multi-task learning acts as implicit regularization
- Shared representations are more robust
- Better performance on low-resource tasks

## Configuration

The MTL implementation uses the same `federated_config.yaml`:

```yaml
model:
  server_model: "prajjwal1/bert-tiny"  # Shared encoder
  client_model: "prajjwal1/bert-tiny"  # Client models

training:
  num_rounds: 10
  local_epochs: 3
  learning_rate: 2e-5

tasks:
  sst2:
    train_samples: 1000
    val_samples: 200
  qqp:
    train_samples: 1000
    val_samples: 200
  stsb:
    train_samples: 1000
    val_samples: 200
```

## Running the System

### Start Server:
```bash
python federated_main.py --mode server --config federated_config.yaml
```

### Start Clients:
```bash
# Terminal 1: SST-2 Client
python federated_main.py --mode client --client_id sst2_client --tasks sst2

# Terminal 2: QQP Client
python federated_main.py --mode client --client_id qqp_client --tasks qqp

# Terminal 3: STSB Client
python federated_main.py --mode client --client_id stsb_client --tasks stsb
```

## Monitoring

The server logs show:
- MTL model architecture summary
- Task distribution per round
- Aggregation details (shared vs task-specific)
- Per-task performance metrics

Example log output:
```
MTL Server Model Summary:
  Base Model: prajjwal1/bert-tiny
  Tasks: ['sst2', 'qqp', 'stsb']
  Shared Parameters: 4,385,280
  Task 'sst2' Head: 1,538 parameters
  Task 'qqp' Head: 1,538 parameters
  Task 'stsb' Head: 769 parameters
  Total Parameters: 4,389,125

=== Round 1/10 ===
Aggregating updates from 3 clients across 3 tasks
  Task 'sst2': 1 clients
  Task 'qqp': 1 clients
  Task 'stsb': 1 clients
Aggregated 285 shared parameters from ALL 3 clients
Aggregated 2 parameters for task 'sst2' head from 1 clients
Aggregated 2 parameters for task 'qqp' head from 1 clients
Aggregated 2 parameters for task 'stsb' head from 1 clients
```

## Comparison: Standard FL vs MTL FL

| Aspect | Standard FL | MTL FL (This Implementation) |
|--------|------------|------------------------------|
| Server Model | Separate models per task | Unified model (shared + heads) |
| Aggregation | Per-task aggregation | Task-aware (shared + task heads) |
| Knowledge Transfer | Within-task only | Cross-task via shared encoder |
| Parameters | N × full_model_size | shared_size + N × head_size |
| Scalability | Linear with tasks | Sub-linear with tasks |
| Performance | Good for single task | Better for multi-task scenarios |

## References

1. **MT-DNN**: Liu et al., "Multi-Task Deep Neural Networks for Natural Language Understanding", ACL 2019
2. **Paper Reference**: 1901.11504v2.pdf - "The lower layers are shared across all tasks while the top layers are task-specific"
3. **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017

## Future Enhancements

1. **Task Weighting**: Weight tasks differently during shared encoder aggregation
2. **Auxiliary Tasks**: Add auxiliary tasks to improve shared representations
3. **Dynamic Task Selection**: Clients can switch tasks between rounds
4. **Hierarchical MTL**: Multiple levels of task grouping (e.g., NLU → Classification → Sentiment)
5. **Adaptive Layer Sharing**: Learn which layers to share vs keep task-specific

