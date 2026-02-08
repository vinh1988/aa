#!/usr/bin/env python3
"""
Standard BERT Model for Federated Learning
Full fine-tuning without LoRA
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class StandardBERTModel(nn.Module):
    """Standard BERT model with task-specific heads for multi-task learning"""
    
    def __init__(self, base_model_name: str, tasks: List[str]):
        super().__init__()
        self.base_model_name = base_model_name
        self.tasks = tasks
        
        # Load base BERT model
        logger.info(f"Loading base model: {base_model_name}")
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Get hidden size from BERT config
        self.hidden_size = self.bert.config.hidden_size
        
        # Create task-specific classification heads
        self.task_heads = nn.ModuleDict()
        for task in tasks:
            if task == 'stsb':
                # Regression head for semantic similarity
                self.task_heads[task] = nn.Linear(self.hidden_size, 1)
            else:
                # Binary classification heads for SST-2 and QQP
                self.task_heads[task] = nn.Linear(self.hidden_size, 2)
        
        logger.info(f"Initialized StandardBERTModel with tasks: {tasks}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, task: str) -> torch.Tensor:
        """
        Forward pass through BERT and task-specific head
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task: Task name to use specific head
            
        Returns:
            Logits for the specified task
        """
        # BERT forward pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        
        # Task-specific head
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}. Available tasks: {list(self.task_heads.keys())}")
        
        logits = self.task_heads[task](pooled_output)
        
        return logits
    
    def get_task_dataloader(self, task: str, batch_size: int, dataset_data: Dict = None):
        """
        Create a DataLoader for a specific task
        
        Args:
            task: Task name
            batch_size: Batch size
            dataset_data: Dictionary containing 'texts' and 'labels'
            
        Returns:
            DataLoader for the task
        """
        from torch.utils.data import Dataset, DataLoader
        
        class TaskDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, task_name):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.task_name = task_name
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]
                
                # Tokenize
                if isinstance(text, (list, tuple)) and len(text) == 2:
                    # Sentence pair (for QQP, STSB)
                    encoding = self.tokenizer(
                        text[0], text[1],
                        padding='max_length',
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    )
                else:
                    # Single sentence (for SST-2)
                    encoding = self.tokenizer(
                        text,
                        padding='max_length',
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    )
                
                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)
                
                # Convert label to tensor
                if self.task_name == 'stsb':
                    label_tensor = torch.tensor(label, dtype=torch.float)
                else:
                    label_tensor = torch.tensor(label, dtype=torch.long)
                
                return input_ids, attention_mask, label_tensor
        
        texts = dataset_data.get('texts', [])
        labels = dataset_data.get('labels', [])
        
        dataset = TaskDataset(texts, labels, self.tokenizer, task)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def get_all_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get all model parameters for federated aggregation
        
        Returns:
            Dictionary of parameter names to tensors
        """
        return {name: param.data.clone() for name, param in self.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Set model parameters from aggregated parameters
        
        Args:
            parameters: Dictionary of parameter names to tensors
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in parameters:
                    param.data.copy_(parameters[name])

