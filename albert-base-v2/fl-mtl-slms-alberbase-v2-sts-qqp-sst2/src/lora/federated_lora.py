#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) Implementation
Parameter-efficient fine-tuning for federated learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Optional, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

class LoRALayer(nn.Module):
    """LoRA layer implementation for parameter-efficient adaptation"""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        # Low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Scaling factor
        self.scaling = alpha / rank

        # Initialize parameters
        nn.init.normal_(self.lora_A, mean=0, std=0.02)
        nn.init.normal_(self.lora_B, mean=0, std=0.02)

    def forward(self, x):
        """Forward pass with LoRA adaptation"""
        # Apply dropout
        x = self.dropout_layer(x)

        # LoRA computation: (B @ A) * scaling applied to input
        # For classification, we need to project to output dimensions
        batch_size = x.size(0)

        # Handle both 2D (batch_size, hidden_size) and 3D (batch_size, seq_len, hidden_size) inputs
        if x.dim() == 3:
            # 3D input: (batch_size, seq_len, hidden_size)
            x_flat = x.view(-1, x.size(-1))  # (batch_size * seq_len, hidden_size)
            has_sequence = True
        else:
            # 2D input: (batch_size, hidden_size) - this is what we get from [CLS] token
            x_flat = x  # (batch_size, hidden_size)
            has_sequence = False

        # Apply LoRA transformation: (x @ A^T) @ B^T * scaling
        lora_A_T = self.lora_A.T  # (in_features, rank)
        lora_B_T = self.lora_B.T  # (rank, out_features)

        # Compute: (x @ A^T) @ B^T * scaling
        intermediate = x_flat @ lora_A_T  # (batch_size, rank) or (batch_size * seq_len, rank)
        lora_output = intermediate @ lora_B_T * self.scaling  # (batch_size, out_features) or (batch_size * seq_len, out_features)

        # Reshape back to include sequence dimension if needed
        if has_sequence:
            lora_output = lora_output.view(batch_size, -1, self.out_features)

        return lora_output

    def get_lora_params(self) -> Dict[str, torch.Tensor]:
        """Get LoRA parameters for serialization"""
        return {
            'lora_A': self.lora_A.data.clone(),
            'lora_B': self.lora_B.data.clone(),
            'rank': self.rank,
            'alpha': self.alpha
        }

    def load_lora_params(self, params: Dict[str, torch.Tensor]):
        """Load LoRA parameters from serialized data"""
        # Ensure parameters are on the same device as the current parameters
        device = self.lora_A.device
        self.lora_A.data = params['lora_A'].clone().to(device)
        self.lora_B.data = params['lora_B'].clone().to(device)
        self.rank = params['rank']
        self.alpha = params['alpha']

class LoRAFederatedModel(nn.Module):
    """Federated model with LoRA adapters for multi-task learning"""

    def __init__(self, base_model_name: str, tasks: List[str], lora_rank: int = 32, lora_alpha: float = 64.0, unfreeze_layers: int = 2):
        super().__init__()

        # Load base model (parameters will be frozen initially)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=1  # For KD compatibility
        )

        # Freeze all base model parameters initially
        for param in self.base_model.parameters():
            param.requires_grad = False

        # PHASE 2 IMPROVEMENT: Selectively unfreeze top layers for better learning capacity
        if unfreeze_layers > 0:
            trainable_params = 0
            total_params = 0
            
            # Try to unfreeze top transformer layers (works for BERT, RoBERTa, etc.)
            if hasattr(self.base_model, 'bert'):
                # For BERT-based models
                encoder = self.base_model.bert.encoder
                num_layers = len(encoder.layer)
                layers_to_unfreeze = min(unfreeze_layers, num_layers)
                
                for layer in encoder.layer[-layers_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                    total_params += sum(p.numel() for p in layer.parameters())
                
                # Also unfreeze the pooler and classification head
                if hasattr(self.base_model.bert, 'pooler') and self.base_model.bert.pooler is not None:
                    for param in self.base_model.bert.pooler.parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                    total_params += sum(p.numel() for p in self.base_model.bert.pooler.parameters())
                
                if hasattr(self.base_model, 'classifier'):
                    for param in self.base_model.classifier.parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                    total_params += sum(p.numel() for p in self.base_model.classifier.parameters())
                
                print(f"[SUCCESS] Unfroze top {layers_to_unfreeze} BERT layers + pooler + classifier")
                print(f"[STATS] Trainable parameters in unfrozen layers: {trainable_params:,}")
                
            elif hasattr(self.base_model, 'roberta'):
                # For RoBERTa-based models
                encoder = self.base_model.roberta.encoder
                num_layers = len(encoder.layer)
                layers_to_unfreeze = min(unfreeze_layers, num_layers)

                for layer in encoder.layer[-layers_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                    total_params += sum(p.numel() for p in layer.parameters())

                # Also unfreeze classifier for RoBERTa (if it exists)
                if hasattr(self.base_model, 'classifier'):
                    for param in self.base_model.classifier.parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                    total_params += sum(p.numel() for p in self.base_model.classifier.parameters())

                print(f"[SUCCESS] Unfroze top {layers_to_unfreeze} RoBERTa layers + classifier")
                print(f"[STATS] Trainable parameters in unfrozen layers: {trainable_params:,}")

        # Task-specific LoRA adapters
        self.task_adapters = nn.ModuleDict({
            task: LoRALayer(
                in_features=self.base_model.config.hidden_size,
                out_features=self.get_num_labels(task),
                rank=lora_rank,
                alpha=lora_alpha
            ) for task in tasks
        })

        self.tasks = tasks
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.unfreeze_layers = unfreeze_layers

    def get_num_labels(self, task: str) -> int:
        """Get number of labels for each task"""
        task_labels = {
            'sst2': 2,  # Binary classification
            'qqp': 2,   # Binary classification
            'stsb': 1   # Regression
        }
        return task_labels.get(task, 2)

    def forward(self, input_ids, attention_mask, task_name):
        """Forward pass with task-specific LoRA"""
        # Base model forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get hidden states from last layer
        hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)

        # Apply task-specific LoRA adaptation
        if task_name in self.task_adapters:
            # LoRA layer expects (batch_size, hidden_size) or (batch_size, seq_len, hidden_size)
            # For BERT, we typically use the [CLS] token or pool the sequence
            if hidden_states.dim() == 3:
                # Use [CLS] token for classification: (batch_size, hidden_size)
                cls_hidden = hidden_states[:, 0, :]  # Take first token
            else:
                cls_hidden = hidden_states

            lora_output = self.task_adapters[task_name](cls_hidden)

            # Ensure output is 2D: (batch_size, num_labels)
            if lora_output.dim() == 3:
                lora_output = lora_output.squeeze(1)  # Remove sequence dimension if present

            combined_logits = lora_output
            
            # Apply sigmoid activation for regression tasks to constrain output to 0-1
            if task_name == 'stsb':
                combined_logits = torch.sigmoid(combined_logits)
        else:
            # Fallback to base model logits if no adapter found
            combined_logits = outputs.logits

        return combined_logits

    def get_all_lora_params(self) -> Dict[str, Dict]:
        """Get LoRA parameters for all tasks"""
        return {
            task: adapter.get_lora_params()
            for task, adapter in self.task_adapters.items()
        }

    def load_all_lora_params(self, all_params: Dict[str, Dict]):
        """Load LoRA parameters for all tasks"""
        for task, params in all_params.items():
            if task in self.task_adapters:
                self.task_adapters[task].load_lora_params(params)

    def get_task_dataloader(self, task: str, batch_size: int = 8, dataset_data: Dict = None):
        """Get DataLoader for a specific task"""
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoTokenizer

        if dataset_data is not None:
            # Tokenize the text data properly
            texts = dataset_data.get('texts', [])
            labels = dataset_data.get('labels', [])
            
            if not texts or not labels:
                # Fallback to dummy data if no real data
                input_ids = torch.randint(0, 1000, (10, 128), dtype=torch.long)
                attention_mask = torch.ones(10, 128, dtype=torch.long)
                labels = torch.randint(0, 2, (10,), dtype=torch.long)
            else:
                # Tokenize the texts
                tokenizer = AutoTokenizer.from_pretrained(self.base_model.config.name_or_path)
                
                # Tokenize all texts
                tokenized = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                
                input_ids = tokenized['input_ids']
                attention_mask = tokenized['attention_mask']
                
                # Convert labels to tensor
                if not isinstance(labels, torch.Tensor):
                    # Use float for regression tasks (stsb), long for classification
                    if task in ['stsb']:
                        labels = torch.tensor(labels, dtype=torch.float32)
                    else:
                        labels = torch.tensor(labels, dtype=torch.long)
                
                # Ensure all tensors have the same batch size
                min_length = min(len(input_ids), len(attention_mask), len(labels))
                if min_length < batch_size:
                    logger.warning(f"Dataset size {min_length} is smaller than batch_size {batch_size}. Adjusting batch_size to {min_length}")
                    batch_size = max(1, min_length)
                input_ids = input_ids[:min_length]
                attention_mask = attention_mask[:min_length]
                labels = labels[:min_length]
                
                # Ensure we have at least one sample
                if min_length == 0:
                    logger.error(f"No data available for task {task}. Cannot create dataloader.")
                    # Fallback to dummy data
                    input_ids = torch.randint(0, 1000, (1, 128), dtype=torch.long)
                    attention_mask = torch.ones(1, 128, dtype=torch.long)
                    labels = torch.tensor([0.0], dtype=torch.float32) if task == 'stsb' else torch.tensor([0], dtype=torch.long)
                    min_length = 1
                    batch_size = 1
        else:
            # Create dummy data for demonstration (fallback)
            input_ids = torch.randint(0, 1000, (10, 128), dtype=torch.long)
            attention_mask = torch.ones(10, 128, dtype=torch.long)
            labels = torch.randint(0, 2, (10,), dtype=torch.long)

        dataset = TensorDataset(input_ids, attention_mask, labels)
        
        # CUDA-friendly DataLoader configuration
        # drop_last=True to avoid incomplete batches that can cause CUDA errors
        # num_workers=0 to avoid multiprocessing issues with CUDA
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Critical for CUDA stability
            pin_memory=False,  # Set to False to avoid CUDA memory issues
            drop_last=(len(dataset) > batch_size)  # Drop last incomplete batch only if we have enough data
        )
        
        logger.info(f"Created DataLoader: dataset_size={len(dataset)}, batch_size={batch_size}, num_batches={len(dataloader)}")
        
        return dataloader

class LoRAAggregator:
    """Aggregates LoRA parameters from multiple clients"""

    def __init__(self):
        self.aggregation_history = []

    def aggregate_lora_updates(self, client_updates: List[Dict], client_weights: List[float] = None) -> Dict[str, Dict]:
        """Aggregate LoRA parameters using federated averaging"""
        if not client_updates:
            return {}

        if client_weights is None:
            # Equal weighting if no weights provided
            client_weights = [1.0 / len(client_updates)] * len(client_updates)

        aggregated_params = {}

        # Get all unique tasks across clients
        all_tasks = set()
        for update in client_updates:
            all_tasks.update(update['lora_updates'].keys())

        # Aggregate parameters for each task
        for task in all_tasks:
            task_params = {}

            # Get parameters for this task from all clients that have it
            task_updates = []
            task_weights = []

            for i, update in enumerate(client_updates):
                if task in update['lora_updates']:
                    task_updates.append(update['lora_updates'][task])
                    task_weights.append(client_weights[i])

            if task_updates:
                # Aggregate each parameter type
                for param_name in task_updates[0].keys():
                    if param_name in ['lora_A', 'lora_B']:
                        # Weighted average of parameter matrices
                        weighted_sum = sum(
                            update[param_name] * weight
                            for update, weight in zip(task_updates, task_weights)
                        )
                        task_params[param_name] = weighted_sum

                # Preserve metadata
                task_params['rank'] = task_updates[0]['rank']
                task_params['alpha'] = task_updates[0]['alpha']

                aggregated_params[task] = task_params

        # Record aggregation
        self.aggregation_history.append({
            'timestamp': torch.tensor([0.0]),  # Placeholder for timestamp
            'num_clients': len(client_updates),
            'tasks_aggregated': list(aggregated_params.keys()),
            'aggregation_weights': client_weights
        })

        return aggregated_params

    def get_aggregation_summary(self) -> Dict:
        """Get summary of aggregation history"""
        return {
            'total_aggregations': len(self.aggregation_history),
            'average_clients_per_aggregation': sum(
                agg['num_clients'] for agg in self.aggregation_history
            ) / len(self.aggregation_history) if self.aggregation_history else 0,
            'unique_tasks_aggregated': list(set(
                task for agg in self.aggregation_history
                for task in agg['tasks_aggregated']
            ))
        }
