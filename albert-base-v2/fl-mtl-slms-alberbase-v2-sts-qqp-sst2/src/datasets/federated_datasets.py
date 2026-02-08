#!/usr/bin/env python3
"""
Federated Learning Dataset Handlers
Task-specific dataset loading with flexible sizing
"""

import logging
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from datasets import load_dataset
import torch

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for dataset loading"""
    task_name: str
    train_samples: int = None  # None means use all available
    val_samples: int = None     # None means use all available
    train_val_split: float = 0.8
    random_seed: int = 42

class BaseDatasetHandler(ABC):
    """Base class for dataset handlers"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.task_name = config.task_name

    @abstractmethod
    def load_raw_dataset(self) -> Tuple[List[str], List]:
        """Load raw dataset - implement in subclasses"""
        pass

    def prepare_data(self) -> Dict:
        """Prepare training and validation data"""
        logger.info(f"Loading {self.task_name} dataset...")

        # Load raw data
        texts, labels = self.load_raw_dataset()

        logger.info(f"Loaded {len(texts)} samples for {self.task_name}")

        # Apply sample limits if specified
        if self.config.train_samples is not None or self.config.val_samples is not None:
            total_needed = (self.config.train_samples or 0) + (self.config.val_samples or 0)
            if len(texts) > total_needed:
                # Sample from the dataset
                indices = list(range(len(texts)))
                random.seed(self.config.random_seed)
                random.shuffle(indices)

                selected_indices = indices[:total_needed]
                texts = [texts[i] for i in selected_indices]
                labels = [labels[i] for i in selected_indices]

        # Split into train/validation
        total_samples = len(texts)
        if self.config.train_samples is not None:
            train_size = self.config.train_samples
        else:
            train_size = int(total_samples * self.config.train_val_split)

        # Shuffle indices for split
        indices = list(range(total_samples))
        random.seed(self.config.random_seed)
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_texts = [texts[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]

        logger.info(f"Task {self.task_name}: Train={len(train_texts)}, Validation={len(val_texts)}")

        return {
            'texts': train_texts,
            'labels': train_labels,
            'val_texts': val_texts,
            'val_labels': val_labels,
            'task_type': self.get_task_type(),
            'distribution': {
                'data': len(train_labels),
                'validation': len(val_labels)
            }
        }

    @abstractmethod
    def get_task_type(self) -> str:
        """Return task type: 'classification' or 'regression'"""
        pass

class SST2DatasetHandler(BaseDatasetHandler):
    """Handler for SST-2 sentiment analysis dataset"""

    def load_raw_dataset(self) -> Tuple[List[str], List]:
        """Load SST-2 dataset"""
        dataset = load_dataset("glue", "sst2", split="train")
        texts = [item["sentence"] for item in dataset]
        labels = [item["label"] for item in dataset]
        return texts, labels

    def get_task_type(self) -> str:
        return "classification"

class QQPDatasetHandler(BaseDatasetHandler):
    """Handler for QQP question pair classification dataset"""

    def load_raw_dataset(self) -> Tuple[List[str], List]:
        """Load QQP dataset"""
        dataset = load_dataset("glue", "qqp", split="train")
        texts = [f"{item['question1']} [SEP] {item['question2']}" for item in dataset]
        labels = [item["label"] for item in dataset]
        return texts, labels

    def get_task_type(self) -> str:
        return "classification"

class STSBDatasetHandler(BaseDatasetHandler):
    """Handler for STSB semantic similarity dataset"""

    def load_raw_dataset(self) -> Tuple[List[str], List]:
        """Load STSB dataset"""
        dataset = load_dataset("glue", "stsb", split="train")
        texts = [f"{item['sentence1']} [SEP] {item['sentence2']}" for item in dataset]
        labels = [float(item["label"]) / 5.0 for item in dataset]  # Normalize to 0-1
        return texts, labels

    def get_task_type(self) -> str:
        return "regression"

class DatasetFactory:
    """Factory for creating dataset handlers"""

    _handlers = {
        'sst2': SST2DatasetHandler,
        'qqp': QQPDatasetHandler,
        'stsb': STSBDatasetHandler,
    }

    @classmethod
    def create_handler(cls, task_name: str, config: DatasetConfig):
        """Create appropriate dataset handler"""
        if task_name not in cls._handlers:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(cls._handlers.keys())}")

        return cls._handlers[task_name](config)

    @classmethod
    def get_available_tasks(cls) -> List[str]:
        """Get list of available tasks"""
        return list(cls._handlers.keys())
