# Global Model Validation Debugging Guide

## Expected Flow

### Round 1:
1. **Server broadcasts initial global model** → `broadcast_mtl_model_slices()`
2. **Clients receive sync** → `handle_global_model_sync()`
   - Updates local model with global parameters
   - ✅ **Validates global model** → `validate_global_model()`
   - Stores metrics in `self.last_global_model_metrics`
3. **Server sends training request** → `handle_training_request()`
   - Clients train locally
   - **Merges global model metrics** into training metrics
   - Sends update to server
4. **Server records to CSV** → Should include `global_model_val_*` columns

### Round 2+:
- Same flow, but global model is from previous aggregation
- Metrics should definitely be present

---

## Current Issue

**Symptom:** `global_model_val_*` columns are EMPTY in `client_results.csv`

**Possible Causes:**

### 1. ❌ Validation Not Running
- Check if `validate_global_model()` is being called
- Look for log: `"Validating global model on local validation data..."`

### 2. ❌ Validation Failing Silently
- No validation data available
- Model not properly synchronized
- Exception caught but not logged

### 3. ❌ Metrics Not Being Merged
- `self.last_global_model_metrics` is empty
- Check log: `"Added global model metrics: ..."`
- Condition `if self.last_global_model_metrics and self.task in local_metrics` failing

### 4. ❌ Metrics Not Written to CSV
- Merging works but CSV writing fails
- Column mismatch in server's `record_client_results()`

---

## Debugging Steps

### Step 1: Check Client Logs

Look for these log messages in client terminal:

```bash
# During sync (should appear in every round):
[Client XXX] Received global model synchronization
[Client XXX] Validating global model on local validation data...
[Client XXX] Global model validation - Task XXX: {...}
[Client XXX] Global model validation complete: {...}

# During training (should appear after validation):
[Client XXX] Added global model metrics: {...}
```

### Step 2: Check if Validation Data Exists

The validation function needs:
```python
task_data = self.dataset_handler.prepare_data()
val_data = {
    'texts': task_data.get('val_texts', []),
    'labels': task_data.get('val_labels', [])
}
```

If `val_texts` or `val_labels` are empty, validation returns `{}`.

**Config specifies:**
- SST-2: 872 val samples
- QQP: 40,431 val samples  
- STS-B: 1,500 val samples

These should be loaded by `DatasetHandler`.

### Step 3: Check Synchronization Order

Verify sync happens BEFORE training:
1. Look for: `"Received global model synchronization"` 
2. THEN look for: `"Received training request for round X"`

If training request comes first, global model metrics won't be available.

### Step 4: Manually Test Validation

Add this to a test client:
```python
# After model sync
global_model_metrics = await self.validate_global_model()
print(f"Global model metrics: {global_model_metrics}")
assert global_model_metrics, "Validation returned empty!"
```

---

## Quick Fix: Increase Rounds

**Current:** `num_rounds: 1` in `federated_config.yaml`  
**Problem:** If validation happens AFTER Round 1 aggregation, metrics only appear in Round 2

**Solution:** Run at least 2 rounds:
```yaml
training:
  num_rounds: 2  # Or more
```

Then check `client_results.csv` for Round 2 - global model metrics should be present.

---

## Expected CSV Output (Round 2)

```csv
round,client_id,task,...,global_model_val_accuracy,global_model_val_f1,...
2,sst2_client,sst2,...,0.8100,0.8050,...
2,qqp_client,qqp,...,0.8300,0.8250,...
2,stsb_client,stsb,...,0.7500,N/A,...  (uses pearson/spearman instead)
```

---

## Common Errors

### Error 1: "No validation data available"
```
[Client XXX] No validation data available for task XXX
```
**Fix:** Ensure `DatasetHandler` properly loads validation split.

### Error 2: "No dataset handler available"
```
[Client XXX] No dataset handler available for validation
```
**Fix:** Ensure `self.dataset_handler` is initialized.

### Error 3: Empty `last_global_model_metrics`
```python
if self.last_global_model_metrics:  # False!
```
**Fix:** Check if validation completed successfully and stored results.

---

## Next Steps

1. ✅ **Set `num_rounds: 2`** in config (already done)
2. **Run experiment again**
3. **Check Round 2 in `client_results.csv`**
4. **If still empty:** Check client logs for validation errors
5. **If logs missing:** Add more logging to `validate_global_model()`

---

**Updated:** January 11, 2026  
**Status:** Debugging in progress

