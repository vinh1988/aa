# MTL Implementation - Changes Summary

## Overview
Converted the federated learning system from standard FL to **Server-Side Multi-Task Learning (MT-DNN style)** where the server maintains a unified model with shared BERT encoder and task-specific heads.

## Key Changes

### 1. New Files Created

#### `src/models/mtl_server_model.py`
- **MTLServerModel**: Unified server model with shared encoder + task heads
- Shared BERT encoder learns universal representations from all tasks
- Task-specific heads (SST-2, QQP, STSB) for specialized outputs
- Methods to get/set shared and task-specific parameters separately

#### `src/aggregation/mtl_aggregator.py`
- **MTLAggregator**: Task-aware aggregation strategy
- Aggregates shared encoder from ALL clients (cross-task knowledge transfer)
- Aggregates task heads from same-task clients only (task specialization)
- Automatically separates shared vs task-specific parameters

#### `src/models/__init__.py`
- Package initialization for models module
- Exports MTLServerModel and StandardBERTModel

#### `MTL_IMPLEMENTATION_GUIDE.md`
- Comprehensive guide to the MTL architecture
- Detailed explanation of training flow
- Architecture diagrams and examples
- Benefits and comparison with standard FL

#### `CHANGES_SUMMARY.md` (this file)
- Summary of all changes made

### 2. Modified Files

#### `src/core/federated_server.py`
**Major Changes**:
- Replaced separate client models with unified `MTLServerModel`
- Added `broadcast_mtl_model_slices()`: sends task-specific slices to clients
- Updated `run_federated_training()`: MTL-aware aggregation workflow
- Added `_serialize_tensors()`: helper for model slice serialization
- Added `_infer_task_from_client_id()`: extract task from client ID
- Modified `handle_client_update()`: captures task labels from clients
- Updated training summary to include MTL model architecture details

**Key Code Additions**:
```python
# Initialize MTL model
self.mtl_model = MTLServerModel(
    base_model_name=config.server_model,
    tasks=['sst2', 'qqp', 'stsb']
)

# MTL-aware aggregation
aggregated = self.aggregator.aggregate_mtl_updates(client_updates)
self.mtl_model.set_shared_parameters(aggregated['shared'])
for task, params in aggregated['task_heads'].items():
    self.mtl_model.set_task_head_parameters(task, params)
```

#### `src/synchronization/federated_synchronization.py`
**Major Changes**:
- Updated `SynchronizationManager` to handle MTL model slices
- Added `get_model_slice_for_task(task)`: returns task-specific slice
- Modified `ClientModelSynchronizer.__init__()`: accepts task parameter
- Updated `synchronize_with_global_model()`: handles MTL model slices
- Backward compatible with legacy full model synchronization

**Key Code Additions**:
```python
# Client synchronizer with task
ClientModelSynchronizer(local_model, websocket_client, task='sst2')

# Handle MTL model slices
if "model_slice" in global_state:
    model_slice = global_state.get("model_slice", {})
    task = global_state.get("task")
```

#### `src/core/base_federated_client.py`
**Changes**:
- Modified update message to include `task` label for MTL aggregation
- Updated synchronizer initialization to pass task parameter

**Key Code Addition**:
```python
# Add task label for MTL-aware aggregation
update_message['task'] = self.task

# Initialize synchronizer with task
self.model_synchronizer = ClientModelSynchronizer(
    self.model, self.websocket_client, self.task
)
```

#### `src/core/federated_client.py`
**Changes**:
- Modified update message to include `tasks` list for MTL aggregation
- Updated synchronizer initialization with primary task

**Key Code Addition**:
```python
# Add task labels for MTL-aware aggregation
update_message['tasks'] = self.tasks

# Initialize synchronizer with primary task
primary_task = self.tasks[0] if self.tasks else 'sst2'
self.model_synchronizer = ClientModelSynchronizer(
    self.model, self.websocket_client, primary_task
)
```

#### `src/aggregation/__init__.py`
**Changes**:
- Added import for `MTLAggregator`
- Updated `__all__` to export both aggregators

## Architecture Comparison

### Before (Standard FL):
```
Server: Separate models per task
Aggregation: Per-task only
Knowledge Transfer: None between tasks
```

### After (MTL FL):
```
Server: Unified model (shared encoder + task heads)
Aggregation: Task-aware (shared from all, heads from same-task)
Knowledge Transfer: Cross-task via shared encoder
```

## Training Flow

### 1. Model Distribution (Server → Clients)
- Server broadcasts task-specific model slices
- Each client receives: Shared BERT encoder + their task head
- Example: SST-2 client gets shared encoder + SST-2 classification head

### 2. Local Training (Clients)
- Clients train on local task data
- Both shared and task-specific parameters are updated

### 3. Update Upload (Clients → Server)
- Clients send updated parameters with task label
- Task label used for MTL-aware aggregation

### 4. MTL Aggregation (Server)
- **Shared Encoder**: Aggregated from ALL clients (cross-task learning)
- **Task Heads**: Aggregated within same-task clients (task specialization)

### 5. Model Update (Server)
- Update shared encoder with aggregated shared parameters
- Update each task head with aggregated task-specific parameters

## Benefits

1. **Cross-Task Knowledge Transfer**: Shared encoder learns from all tasks
2. **Task Specialization**: Task heads maintain specialized knowledge
3. **Parameter Efficiency**: Shared encoder reduces total parameters
4. **Improved Generalization**: Multi-task learning as implicit regularization
5. **Better Low-Resource Performance**: Tasks with limited data benefit from others

## Model Architecture

```
MTLServerModel (4.39M parameters)
├── Shared BERT Encoder (4.385M parameters)
│   └── Aggregated from ALL clients
└── Task-Specific Heads
    ├── SST-2 Head (1,538 parameters) - Binary classification
    │   └── Aggregated from SST-2 clients only
    ├── QQP Head (1,538 parameters) - Binary classification
    │   └── Aggregated from QQP clients only
    └── STSB Head (769 parameters) - Regression
        └── Aggregated from STSB clients only
```

## Configuration

No changes required to `federated_config.yaml`. The system automatically:
- Initializes MTL server model
- Performs task-aware aggregation
- Distributes task-specific model slices

## Running the System

### Start Server:
```bash
cd /home/vinh/Documents/code/FedAvgLS/experiment/mtl/fl-slms-berttiny-sts-qqp-sst2
source /home/vinh/Documents/code/FedAvgLS/venv/bin/activate
python federated_main.py --mode server --config federated_config.yaml
```

### Start Clients (in separate terminals):
```bash
# SST-2 Client
python federated_main.py --mode client --client_id sst2_client --tasks sst2

# QQP Client
python federated_main.py --mode client --client_id qqp_client --tasks qqp

# STSB Client
python federated_main.py --mode client --client_id stsb_client --tasks stsb
```

## Expected Behavior

### Server Logs:
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
Broadcasting MTL model slices to clients...
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

### Client Logs:
```
[sst2_client] Client MTL synchronizer initialized for sst2_client (task: sst2)
[sst2_client] Synchronized MTL model slice to version 1
[sst2_client] Training on task: sst2
```

## Testing

To verify the MTL implementation:

1. **Check server logs** for MTL model summary and task-aware aggregation
2. **Check client logs** for MTL synchronization messages
3. **Monitor training metrics** for cross-task knowledge transfer benefits
4. **Compare with baseline** (standard FL) to see performance improvements

## References

- **MT-DNN Paper**: Liu et al., "Multi-Task Deep Neural Networks for Natural Language Understanding", ACL 2019
- **Reference Paper**: 1901.11504v2.pdf (provided by user)
- **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017

## Next Steps (Optional Enhancements)

1. **Task Weighting**: Weight tasks differently during aggregation
2. **Auxiliary Tasks**: Add auxiliary tasks to improve shared representations
3. **Dynamic Task Selection**: Allow clients to switch tasks between rounds
4. **Adaptive Layer Sharing**: Learn which layers to share vs keep task-specific
5. **Performance Analysis**: Compare MTL vs standard FL on various metrics

