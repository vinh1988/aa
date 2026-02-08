#!/usr/bin/env python3
"""
Multi-Task Learning Aggregator for Federated Learning
Task-aware aggregation: shared layers from all clients, task heads from same-task clients
"""

import torch
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class MTLAggregator:
    """
    MTL-aware Aggregator for Federated Learning
    
    Aggregation Strategy (inspired by MT-DNN):
    1. Shared BERT Encoder: FedAvg across ALL clients (universal representation learning)
    2. Task-Specific Heads: FedAvg within same-task clients only (task specialization)
    
    This allows cross-task knowledge transfer at the shared layer level while
    maintaining task-specific expertise in the output heads.
    """
    
    def __init__(self):
        self.aggregation_history = []
    
    def aggregate_mtl_updates(self, client_updates: List[Dict]) -> Dict:
        """
        Perform MTL-aware aggregation
        
        Args:
            client_updates: List of dictionaries containing:
                - 'client_id': Client identifier
                - 'task': Task name ('sst2', 'qqp', or 'stsb')
                - 'lora_updates': Model parameters (kept for backward compatibility)
                - 'metrics': Training metrics
        
        Returns:
            Dictionary with:
                - 'shared': Aggregated shared BERT encoder parameters
                - 'task_heads': Dict of aggregated task-specific head parameters per task
        """
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return {'shared': {}, 'task_heads': {}}
        
        # Group updates by task
        task_groups = {}
        for update in client_updates:
            # Get task from client_id or metrics
            task = self._extract_task_from_update(update)
            if task not in task_groups:
                task_groups[task] = []
            task_groups[task].append(update)
        
        logger.info(f"Aggregating updates from {len(client_updates)} clients across {len(task_groups)} tasks")
        for task, updates in task_groups.items():
            logger.info(f"  Task '{task}': {len(updates)} clients")
        
        # Extract all parameters from client updates
        all_client_params = []
        for update in client_updates:
            params = update.get('lora_updates', {})  # Keep key name for backward compatibility
            if params:
                all_client_params.append(params)
        
        if not all_client_params:
            logger.warning("No valid parameters found in client updates")
            return {'shared': {}, 'task_heads': {}}
        
        # Separate shared parameters from task-specific parameters
        shared_params_list = []
        task_head_params = {task: [] for task in task_groups.keys()}
        
        for i, update in enumerate(client_updates):
            task = self._extract_task_from_update(update)
            params = all_client_params[i]
            
            # Separate shared (BERT) and task-specific parameters
            shared = {}
            task_specific = {}
            
            for param_name, param_value in params.items():
                if 'bert.' in param_name:
                    # Shared BERT encoder parameter
                    shared[param_name] = param_value
                elif 'task_heads.' in param_name:
                    # Task-specific head parameter
                    task_specific[param_name] = param_value
                else:
                    # Default: treat as shared if unclear
                    shared[param_name] = param_value
            
            shared_params_list.append(shared)
            if task_specific:
                task_head_params[task].append(task_specific)
        
        # Aggregate shared parameters using FedAvg across ALL clients
        aggregated_shared = self._fedavg(shared_params_list)
        logger.info(f"Aggregated {len(aggregated_shared)} shared parameters from ALL {len(client_updates)} clients")
        
        # Aggregate task-specific heads using FedAvg within same-task clients
        aggregated_task_heads = {}
        for task, task_params_list in task_head_params.items():
            if task_params_list:
                aggregated_task_heads[task] = self._fedavg(task_params_list)
                logger.info(f"Aggregated {len(aggregated_task_heads[task])} parameters for task '{task}' head from {len(task_params_list)} clients")
            else:
                logger.warning(f"No task-specific parameters found for task '{task}'")
                aggregated_task_heads[task] = {}
        
        # Record aggregation event
        self.aggregation_history.append({
            'total_clients': len(client_updates),
            'task_distribution': {task: len(updates) for task, updates in task_groups.items()},
            'shared_params_count': len(aggregated_shared),
            'task_heads_count': {task: len(params) for task, params in aggregated_task_heads.items()}
        })
        
        return {
            'shared': aggregated_shared,
            'task_heads': aggregated_task_heads
        }
    
    def _fedavg(self, params_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Federated Averaging (FedAvg) algorithm
        
        Args:
            params_list: List of parameter dictionaries from different clients
            
        Returns:
            Averaged parameters
        """
        if not params_list:
            return {}
        
        # Get all parameter names
        param_names = set()
        for params in params_list:
            param_names.update(params.keys())
        
        averaged_params = {}
        
        for param_name in param_names:
            # Collect this parameter from all clients that have it
            param_tensors = []
            for params in params_list:
                if param_name in params:
                    param_value = params[param_name]
                    
                    # Convert to tensor if needed
                    if isinstance(param_value, torch.Tensor):
                        param_tensors.append(param_value.cpu())
                    elif isinstance(param_value, (list, tuple)):
                        param_tensors.append(torch.tensor(param_value))
                    else:
                        param_tensors.append(torch.tensor([param_value]))
            
            # Average across clients
            if param_tensors:
                stacked = torch.stack(param_tensors)
                averaged_params[param_name] = torch.mean(stacked, dim=0)
        
        return averaged_params
    
    def _extract_task_from_update(self, update: Dict) -> str:
        """
        Extract task name from client update
        
        Args:
            update: Client update dictionary
            
        Returns:
            Task name ('sst2', 'qqp', or 'stsb')
        """
        # Try to get task from update directly
        if 'task' in update:
            return update['task']
        
        # Try to infer from client_id
        client_id = update.get('client_id', '')
        if 'sst2' in client_id.lower():
            return 'sst2'
        elif 'qqp' in client_id.lower():
            return 'qqp'
        elif 'stsb' in client_id.lower():
            return 'stsb'
        
        # Try to infer from metrics
        metrics = update.get('metrics', {})
        if metrics:
            # Get the first task from metrics
            for key in metrics.keys():
                if key in ['sst2', 'qqp', 'stsb']:
                    return key
        
        # Default fallback
        logger.warning(f"Could not determine task for client {client_id}, defaulting to 'sst2'")
        return 'sst2'
    
    def get_aggregation_summary(self) -> Dict:
        """Get summary of aggregation history"""
        if not self.aggregation_history:
            return {'total_aggregations': 0}
        
        recent = self.aggregation_history[-1]
        return {
            'total_aggregations': len(self.aggregation_history),
            'recent_clients': recent['total_clients'],
            'recent_task_distribution': recent['task_distribution'],
            'recent_shared_params': recent['shared_params_count'],
            'recent_task_heads': recent['task_heads_count']
        }

