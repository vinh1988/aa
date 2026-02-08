#!/usr/bin/env python3
"""
Base Local Training Client Implementation
Contains shared functionality for standalone local training clients
"""

import os
import time
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import torch.nn as nn

logger = logging.getLogger(__name__)

class BaseLocalClient(ABC):
    """Base class for local training clients with shared functionality"""

    def __init__(self, task: str, config_path: str = None):
        self.task = task
        self.config_path = config_path or "federated_config.yaml"
        self.device = self.get_device()
        self.setup_logging()

        # Model and training components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Training configuration
        self.config = self.load_config()

        # Dataset
        self.dataset_handler = None

        # Training state
        self.current_epoch = 0

    def get_device(self) -> torch.device:
        """Get available device (GPU/CPU)"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, "INFO")
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'local_{self.task}_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Extract task-specific configuration
            task_config = config.get('task_configs', {}).get(self.task, {})
            training_config = config.get('training', {})

            return {
                'model_name': config.get('model', {}).get('client_model', 'bert-base-uncased'),
                'batch_size': training_config.get('batch_size', 8),
                'learning_rate': training_config.get('learning_rate', 2e-5),
                'num_epochs': training_config.get('local_epochs', 3),
                'max_length': 128,
                'task_config': task_config,
                'output_dir': f"local_{self.task}_results"
            }
        except Exception as e:
            self.logger.warning(f"Could not load config from {self.config_path}: {e}")
            # Return default configuration
            return {
                'model_name': 'bert-base-uncased',
                'batch_size': 8,
                'learning_rate': 2e-5,
                'num_epochs': 3,
                'max_length': 128,
                'output_dir': f"local_{self.task}_results"
            }

    def initialize_model(self):
        """Initialize model, tokenizer, and training components"""
        self.logger.info(f"Loading model: {self.config['model_name']}")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])

        # Task-specific model initialization
        if self.task == 'stsb':
            # Regression task for semantic similarity
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config['model_name'],
                num_labels=1  # Regression task
            )
        else:
            # Classification tasks (SST-2, QQP)
            num_labels = 2 if self.task in ['sst2', 'qqp'] else 3
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config['model_name'],
                num_labels=num_labels
            )

        self.model.to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )

        total_steps = self.config['num_epochs'] * 100  # Approximate
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Loss function
        if self.task == 'stsb':
            self.criterion = nn.MSELoss()  # Regression loss
        else:
            self.criterion = nn.CrossEntropyLoss()  # Classification loss

        self.logger.info(f"Model initialized successfully on {self.device}")

    def load_dataset_from_library(self) -> Tuple[List[Dict], List[Dict]]:
        """Load dataset from HuggingFace datasets library"""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library not available. Install with: pip install datasets")

        self.logger.info(f"Loading {self.task} dataset from HuggingFace datasets library")

        try:
            # Load dataset from GLUE
            dataset = load_dataset("glue", self.task)

            # Convert to our format
            train_data = []
            val_data = []

            # Process training split
            if 'train' in dataset:
                for item in dataset['train']:
                    train_data.append(self._convert_dataset_item(item))

            # Process validation split
            val_split = 'validation' if 'validation' in dataset else 'test'
            if val_split in dataset:
                for item in dataset[val_split]:
                    val_data.append(self._convert_dataset_item(item))

            self.logger.info(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples from datasets library")

            return train_data, val_data

        except Exception as e:
            self.logger.error(f"Failed to load dataset from library: {e}")
            raise

    def _convert_dataset_item(self, item: Dict) -> Dict:
        """Convert dataset item to our internal format"""
        # This will be overridden by specialized clients
        return item

    @abstractmethod
    def load_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """Load task-specific dataset - implemented by specialized clients"""
        pass

    def create_dataloader(self, data: List[Dict], shuffle: bool = True) -> torch.utils.data.DataLoader:
        """Create DataLoader from dataset"""
        from torch.utils.data import Dataset

        class SimpleDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]

                # Tokenize
                if self.task == 'stsb':
                    # STSB format: sentence1, sentence2, score
                    encoding = self.tokenizer(
                        item['sentence1'],
                        item['sentence2'],
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    return {
                        'input_ids': encoding['input_ids'].squeeze(),
                        'attention_mask': encoding['attention_mask'].squeeze(),
                        'labels': torch.tensor(item['score'], dtype=torch.float)
                    }
                else:
                    # Classification format: text, label
                    encoding = self.tokenizer(
                        item['text'],
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    return {
                        'input_ids': encoding['input_ids'].squeeze(),
                        'attention_mask': encoding['attention_mask'].squeeze(),
                        'labels': torch.tensor(item['label'], dtype=torch.long)
                    }

        # Add tokenizer and task to dataset for __getitem__
        SimpleDataset.tokenizer = self.tokenizer
        SimpleDataset.task = self.task

        dataset = SimpleDataset(data, self.tokenizer, self.config['max_length'])
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=shuffle
        )

    def calculate_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Calculate task-specific metrics"""
        if self.task == 'stsb':
            # Regression metrics
            pred_np = predictions.detach().cpu().numpy()
            label_np = labels.detach().cpu().numpy()

            mae = torch.mean(torch.abs(predictions - labels)).item()
            mse = torch.mean((predictions - labels) ** 2).item()

            # Correlation coefficient
            if len(pred_np) > 1:
                correlation = torch.corrcoef(torch.stack([predictions, labels]))[0, 1].item()
            else:
                correlation = 0.0

            return {
                'mae': mae,
                'mse': mse,
                'correlation': correlation,
                'loss': mse  # Use MSE as primary loss metric
            }
        else:
            # Classification metrics
            pred_labels = torch.argmax(predictions, dim=1)
            accuracy = (pred_labels == labels).float().mean().item()
            loss = self.criterion(predictions, labels).item()

            return {
                'accuracy': accuracy,
                'loss': loss
            }

    def save_results(self, metrics: Dict[str, List[float]], output_path: str = None):
        """Save training results to file"""
        if output_path is None:
            output_path = os.path.join(self.config['output_dir'], f"{self.task}_training_results.txt")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(f"Local Training Results for {self.task.upper()}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.config['model_name']}\n")
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Epoch Results:\n")
            f.write("-" * 30 + "\n")

            for epoch, epoch_metrics in enumerate(metrics['train_metrics']):
                f.write(f"Epoch {epoch + 1}:\n")
                for metric, value in epoch_metrics.items():
                    f.write(".4f")
                f.write("\n")

            if 'val_metrics' in metrics and metrics['val_metrics']:
                f.write("\nValidation Results:\n")
                f.write("-" * 30 + "\n")
                for epoch, epoch_metrics in enumerate(metrics['val_metrics']):
                    f.write(f"Epoch {epoch + 1} Validation:\n")
                    for metric, value in epoch_metrics.items():
                        f.write(".4f")
                    f.write("\n")

        self.logger.info(f"Results saved to {output_path}")

    def run_training(self) -> Dict:
        """Main training loop"""
        self.logger.info(f"Starting local training for {self.task}")

        # Initialize model and components
        self.initialize_model()

        # Load dataset
        train_data, val_data = self.load_dataset()

        # Create data loaders
        train_loader = self.create_dataloader(train_data, shuffle=True)
        val_loader = self.create_dataloader(val_data, shuffle=False) if val_data else None

        # Training loop
        train_metrics = []
        val_metrics = []

        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch + 1

            # Training phase
            epoch_train_metrics = self.train_epoch(train_loader)
            train_metrics.append(epoch_train_metrics)

            # Validation phase
            if val_loader:
                epoch_val_metrics = self.validate_epoch(val_loader)
                val_metrics.append(epoch_val_metrics)
                self.logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']} - "
                               f"Train Loss: {epoch_train_metrics['loss']:.4f} - "
                               f"Val Loss: {epoch_val_metrics['loss']:.4f}")
            else:
                self.logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']} - "
                               f"Train Loss: {epoch_train_metrics['loss']:.4f}")

        # Save results
        results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config,
            'final_metrics': train_metrics[-1] if train_metrics else {}
        }

        self.save_results(results)
        return results

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

            # Handle different task types
            if self.task == 'stsb':
                # For regression, ensure logits are properly shaped
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)

            # Calculate loss
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # Calculate final metrics for this epoch
        return self.calculate_metrics(logits, labels)

    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits

                # Handle different task types
                if self.task == 'stsb':
                    if logits.dim() > 1:
                        logits = logits.squeeze(-1)

                # Calculate loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                all_logits.append(logits)
                all_labels.append(labels)

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        # Concatenate all predictions and labels
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        return self.calculate_metrics(all_logits, all_labels)
