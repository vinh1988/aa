#!/usr/bin/env python3
"""
Standard Parameter Aggregation for Federated Learning
FedAvg algorithm for full parameter aggregation
"""

import torch
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class StandardAggregator:
    """Aggregates full model parameters using FedAvg"""
    
    def __init__(self):
        pass
    
    def aggregate_parameters(self, client_updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Aggregate parameters from multiple clients using FedAvg
        
        Args:
            client_updates: List of dictionaries containing client updates
                           Each dict has 'client_id', 'model_updates', 'metrics'
        
        Returns:
            Aggregated parameters dictionary
        """
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return {}
        
        # Extract model updates from each client
        all_params = []
        for update in client_updates:
            params = update.get('lora_updates', {})  # Keep key name for backward compatibility
            if params:
                all_params.append(params)
        
        if not all_params:
            logger.warning("No valid parameters found in client updates")
            return {}
        
        # Initialize aggregated parameters with zeros
        aggregated = {}
        param_names = list(all_params[0].keys())
        
        for param_name in param_names:
            # Stack all client parameters for this layer
            param_list = [client_params[param_name] for client_params in all_params 
                         if param_name in client_params]
            
            if not param_list:
                continue
            
            # Convert to tensors if they aren't already
            param_tensors = []
            for p in param_list:
                if isinstance(p, torch.Tensor):
                    param_tensors.append(p.cpu())
                elif isinstance(p, (list, tuple)):
                    param_tensors.append(torch.tensor(p))
                else:
                    param_tensors.append(torch.tensor([p]))
            
            # Average across clients (FedAvg)
            if param_tensors:
                stacked = torch.stack(param_tensors)
                aggregated[param_name] = torch.mean(stacked, dim=0)
        
        logger.info(f"Aggregated {len(aggregated)} parameters from {len(all_params)} clients")
        
        return aggregated

