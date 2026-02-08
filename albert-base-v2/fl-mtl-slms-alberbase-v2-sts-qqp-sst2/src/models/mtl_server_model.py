#!/usr/bin/env python3
"""
Multi-Task Learning Server Model (MT-DNN Style)
Server maintains one unified model with shared encoder and task-specific heads
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AlbertModel
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MTLServerModel(nn.Module):
    """
    Multi-Task Server Model inspired by MT-DNN
    
    Architecture:
    - Lower layers: Shared BERT encoder (learns universal representations)
    - Upper layers: Task-specific heads (specialized for each task)
    
    Reference: "Multi-Task Deep Neural Networks for Natural Language Understanding"
    Liu et al., 2019 (MT-DNN)
    """
    
    def __init__(self, base_model_name: str, tasks: List[str]):
        super().__init__()
        self.base_model_name = base_model_name
        self.tasks = tasks
        
        # Shared BERT encoder (lower layers)
        logger.info(f"Initializing MTL Server Model with shared encoder: {base_model_name}")
        
        # Try to load as AlbertModel directly if applicable to avoid AutoModel dynamic loading issues
        if "albert" in base_model_name.lower():
            try:
                logger.info(f"Explicitly loading as AlbertModel: {base_model_name}")
                self.bert = AlbertModel.from_pretrained(base_model_name)
            except Exception as e:
                logger.warning(f"Failed to load as AlbertModel directly: {e}. Falling back to AutoModel.")
                self.bert = AutoModel.from_pretrained(base_model_name)
        else:
            self.bert = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Task-specific heads (upper layers)
        self.task_heads = nn.ModuleDict()
        for task in tasks:
            if task == 'stsb':
                # Regression head for semantic similarity
                self.task_heads[task] = nn.Linear(self.hidden_size, 1)
            else:
                # Binary classification heads for SST-2 and QQP
                self.task_heads[task] = nn.Linear(self.hidden_size, 2)
        
        logger.info(f"MTL Server Model initialized with {len(tasks)} task-specific heads: {tasks}")
        logger.info(f"Shared encoder parameters: {sum(p.numel() for p in self.bert.parameters()):,}")
        for task in tasks:
            task_params = sum(p.numel() for p in self.task_heads[task].parameters())
            logger.info(f"Task '{task}' head parameters: {task_params:,}")
    
    def get_shared_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get shared BERT encoder parameters (aggregated from ALL clients)
        
        Returns:
            Dictionary of parameter names to tensors for the shared encoder
        """
        return {f"bert.{name}": param.data.clone() 
                for name, param in self.bert.named_parameters()}
    
    def get_task_head_parameters(self, task: str) -> Dict[str, torch.Tensor]:
        """
        Get task-specific head parameters (aggregated only from same-task clients)
        
        Args:
            task: Task name ('sst2', 'qqp', or 'stsb')
            
        Returns:
            Dictionary of parameter names to tensors for the task head
        """
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}. Available tasks: {list(self.task_heads.keys())}")
        
        return {f"task_heads.{task}.{name}": param.data.clone() 
                for name, param in self.task_heads[task].named_parameters()}
    
    def get_all_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get all parameters (shared + all task heads)
        
        Returns:
            Dictionary of all parameter names to tensors
        """
        params = self.get_shared_parameters()
        for task in self.tasks:
            params.update(self.get_task_head_parameters(task))
        return params
    
    def set_shared_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Set shared BERT encoder parameters
        
        Args:
            parameters: Dictionary of parameter names to tensors
        """
        with torch.no_grad():
            for name, param in self.bert.named_parameters():
                full_name = f"bert.{name}"
                if full_name in parameters:
                    param.data.copy_(parameters[full_name])
    
    def set_task_head_parameters(self, task: str, parameters: Dict[str, torch.Tensor]):
        """
        Set task-specific head parameters
        
        Args:
            task: Task name
            parameters: Dictionary of parameter names to tensors
        """
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}")
        
        with torch.no_grad():
            for name, param in self.task_heads[task].named_parameters():
                full_name = f"task_heads.{task}.{name}"
                if full_name in parameters:
                    param.data.copy_(parameters[full_name])
    
    def get_model_slice_for_task(self, task: str) -> Dict[str, torch.Tensor]:
        """
        Get model slice for a specific task (shared encoder + task-specific head)
        This is what gets sent to clients
        
        Args:
            task: Task name
            
        Returns:
            Dictionary containing shared parameters and task-specific parameters
        """
        model_slice = {}
        
        # Add shared encoder parameters
        model_slice.update(self.get_shared_parameters())
        
        # Add task-specific head parameters
        model_slice.update(self.get_task_head_parameters(task))
        
        return model_slice
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, task: str) -> torch.Tensor:
        """
        Forward pass through shared encoder and task-specific head
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task: Task name to use specific head
            
        Returns:
            Logits for the specified task
        """
        # Shared BERT encoder
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get [CLS] token representation (pooled output)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Task-specific head
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}. Available tasks: {list(self.task_heads.keys())}")
        
        logits = self.task_heads[task](pooled_output)
        
        return logits
    
    def get_model_summary(self) -> Dict:
        """Get summary of model architecture"""
        total_shared = sum(p.numel() for p in self.bert.parameters())
        task_head_sizes = {task: sum(p.numel() for p in head.parameters()) 
                          for task, head in self.task_heads.items()}
        
        return {
            'base_model': self.base_model_name,
            'tasks': self.tasks,
            'shared_parameters': total_shared,
            'task_head_parameters': task_head_sizes,
            'total_parameters': total_shared + sum(task_head_sizes.values())
        }

