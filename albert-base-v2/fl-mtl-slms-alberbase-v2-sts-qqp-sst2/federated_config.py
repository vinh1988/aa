#!/usr/bin/env python3
"""
Federated Learning Configuration Management
Centralized configuration with YAML support and validation
(Knowledge Distillation removed)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import yaml
import os
import argparse

@dataclass
class CommunicationConfig:
    """Communication settings for federated learning"""
    port: int = 8771
    timeout: int = 60
    websocket_timeout: int = 30
    retry_attempts: int = 3
    round_timeout: int = 3400  # Default timeout for collecting client updates (56.7 minutes)
    send_timeout: int = 3600  # Timeout for sending large updates (1 hour)

@dataclass
class FederatedConfig:
    """Centralized configuration for federated learning system"""

    # Model settings
    server_model: str = "albert-base-v2"
    client_model: str = "albert-base-v2"

    # Synchronization settings
    enable_synchronization: bool = True
    sync_frequency: str = "per_round"
    global_model_sharing: bool = True

    # Data settings (per-task configuration support)
    samples_per_client: int = 300
    max_samples_per_client: int = 500
    data_distribution: str = "non_iid"
    non_iid_alpha: float = 0.5
    oversample_minority: bool = True
    normalize_weights: bool = True

    # Per-task data configuration
    task_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Training settings
    num_rounds: int = 2
    min_clients: int = 1
    max_clients: int = 3
    expected_clients: int = 3  # Expected number of clients to wait for
    local_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 0.0002

    # Communication settings
    port: int = 8771
    timeout: int = 60
    websocket_timeout: int = 30
    retry_attempts: int = 3
    round_timeout: int = 3400  # Timeout for collecting client updates per round (56.7 minutes)
    send_timeout: int = 3600  # Timeout for sending large updates (1 hour)

    # Communication settings as nested object (for backward compatibility)
    @property
    def communication(self):
        """Communication settings as a nested object for backward compatibility"""
        return CommunicationConfig(
            port=self.port,
            timeout=self.timeout,
            websocket_timeout=self.websocket_timeout,
            retry_attempts=self.retry_attempts,
            round_timeout=getattr(self, 'round_timeout', 3400),  # Use config value or default (56.7 minutes)
            send_timeout=getattr(self, 'send_timeout', 3600)  # Use config value or default
        )

    # Output settings
    results_dir: str = "federated_results"
    log_level: str = "INFO"
    save_checkpoints: bool = True

    # Advanced settings
    use_validation: bool = True
    early_stopping_patience: int = 3
    save_best_model: bool = True
    mixed_precision: bool = False
    gradient_clipping: float = 1.0
    weight_decay: float = 0.01

    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Set default task configurations if not provided
        if not self.task_configs:
            self.task_configs = {
                'sst2': {
                    'train_samples': 5000,  # IMPROVED: Increased 100x from 50
                    'val_samples': 1000,    # IMPROVED: Increased 100x from 10
                    'random_seed': 42
                },
                'qqp': {
                    'train_samples': 3000,  # IMPROVED: Increased 100x from 30
                    'val_samples': 600,     # IMPROVED: Increased 100x from 6
                    'random_seed': 42
                },
                'stsb': {
                    'train_samples': 5000,  # IMPROVED: Increased 250x from 20
                    'val_samples': 1000,    # IMPROVED: Increased 250x from 4
                    'random_seed': 42
                }
            }

    @classmethod
    def from_yaml_file(cls, config_path: str) -> 'FederatedConfig':
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)

        # Flatten nested dictionary to match dataclass field names
        flattened_config = cls._flatten_config_dict(config_dict)

        # Create config instance
        return cls(**flattened_config)

    @classmethod
    def _flatten_config_dict(cls, config_dict: Dict) -> Dict:
        """Flatten nested dictionary to match dataclass field names"""
        flattened = {}

        # Define mapping from nested keys to flat keys
        key_mapping = {
            # Model settings
            ('model', 'server_model'): 'server_model',
            ('model', 'client_model'): 'client_model',

            # Synchronization settings
            ('synchronization', 'enabled'): 'enable_synchronization',
            ('synchronization', 'frequency'): 'sync_frequency',
            ('synchronization', 'global_model_sharing'): 'global_model_sharing',

            # Training settings
            ('training', 'num_rounds'): 'num_rounds',
            ('training', 'min_clients'): 'min_clients',
            ('training', 'max_clients'): 'max_clients',
            ('training', 'expected_clients'): 'expected_clients',
            ('training', 'local_epochs'): 'local_epochs',
            ('training', 'batch_size'): 'batch_size',
            ('training', 'learning_rate'): 'learning_rate',

            # Data settings
            ('data', 'samples_per_client'): 'samples_per_client',
            ('data', 'max_samples_per_client'): 'max_samples_per_client',
            ('data', 'distribution'): 'data_distribution',
            ('data', 'non_iid_alpha'): 'non_iid_alpha',
            ('data', 'oversample_minority'): 'oversample_minority',
            ('data', 'normalize_weights'): 'normalize_weights',

            # Communication settings
            ('communication', 'port'): 'port',
            ('communication', 'timeout'): 'timeout',
            ('communication', 'websocket_timeout'): 'websocket_timeout',
            ('communication', 'retry_attempts'): 'retry_attempts',
            ('communication', 'round_timeout'): 'round_timeout',
            ('communication', 'send_timeout'): 'send_timeout',

            # Output settings
            ('output', 'results_dir'): 'results_dir',
            ('output', 'log_level'): 'log_level',
            ('output', 'save_checkpoints'): 'save_checkpoints',

            # Advanced settings
            ('advanced', 'use_validation'): 'use_validation',
            ('advanced', 'early_stopping_patience'): 'early_stopping_patience',
            ('advanced', 'save_best_model'): 'save_best_model',
            ('advanced', 'mixed_precision'): 'mixed_precision',
            ('advanced', 'gradient_clipping'): 'gradient_clipping',
            ('advanced', 'weight_decay'): 'weight_decay',
        }

        # Flatten nested dictionary
        def flatten_dict(d: Dict, prefix: tuple = ()) -> None:
            for key, value in d.items():
                current_key = prefix + (key,)

                # Special handling for task_configs - keep it as a nested dict
                if current_key == ('task_configs',):
                    flattened['task_configs'] = value
                elif isinstance(value, dict) and current_key not in [('task_configs',)]:
                    flatten_dict(value, current_key)
                elif current_key in key_mapping:
                    flattened[key_mapping[current_key]] = value

        flatten_dict(config_dict)

        return flattened

    def to_yaml_file(self, config_path: str):
        """Save configuration to YAML file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        # Convert dataclass to nested dictionary for YAML
        config_dict = {
            'model': {
                'server_model': self.server_model,
                'client_model': self.client_model,
            },
            'synchronization': {
                'enabled': self.enable_synchronization,
                'frequency': self.sync_frequency,
                'global_model_sharing': self.global_model_sharing,
            },
            'training': {
                'num_rounds': self.num_rounds,
                'min_clients': self.min_clients,
                'max_clients': self.max_clients,
                'local_epochs': self.local_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
            },
            'data': {
                'samples_per_client': self.samples_per_client,
                'max_samples_per_client': self.max_samples_per_client,
                'distribution': self.data_distribution,
                'non_iid_alpha': self.non_iid_alpha,
                'oversample_minority': self.oversample_minority,
                'normalize_weights': self.normalize_weights,
            },
            'communication': {
                'port': self.port,
                'timeout': self.timeout,
                'websocket_timeout': self.websocket_timeout,
                'retry_attempts': self.retry_attempts,
                'round_timeout': self.round_timeout,
                'send_timeout': self.send_timeout,
            },
            'output': {
                'results_dir': self.results_dir,
                'log_level': self.log_level,
                'save_checkpoints': self.save_checkpoints,
            },
            'advanced': {
                'use_validation': self.use_validation,
                'early_stopping_patience': self.early_stopping_patience,
                'save_best_model': self.save_best_model,
                'mixed_precision': self.mixed_precision,
                'gradient_clipping': self.gradient_clipping,
                'weight_decay': self.weight_decay,
            }
        }

        # Add task_configs if present
        if self.task_configs:
            config_dict['task_configs'] = self.task_configs

        with open(config_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)

    def print_summary(self):
        """Print configuration summary"""
        print(" Federated Learning Configuration (MTL - Server-Side)")
        print("=" * 60)
        print(f" Model: {self.server_model} (server), {self.client_model} (client)")
        print(f" Sync: {'Enabled' if self.enable_synchronization else 'Disabled'} ({self.sync_frequency})")
        print(f" Training: {self.num_rounds} rounds, {self.local_epochs} epochs, batch_size={self.batch_size}")
        print(f" Data (Task-Specific Samples):")
        for task, task_config in self.task_configs.items():
            train = task_config.get('train_samples', 'N/A')
            val = task_config.get('val_samples', 'N/A')
            print(f"   - {task.upper()}: {train} train, {val} val")
        print(f" Communication: Port {self.port}, timeout={self.timeout}s")
        print(f" Output: Results in '{self.results_dir}', log_level={self.log_level}")
        print("=" * 60)

    def get_dataset_configs(self, tasks: List[str]) -> Dict[str, Any]:
        """Get dataset configurations for the specified tasks"""
        from src.datasets.federated_datasets import DatasetConfig

        configs = {}

        for task in tasks:
            if task in self.task_configs:
                # Use task-specific configuration
                task_config = self.task_configs[task]
                configs[task] = DatasetConfig(
                    task_name=task,
                    train_samples=task_config.get('train_samples'),
                    val_samples=task_config.get('val_samples'),
                    random_seed=task_config.get('random_seed', 42)
                )
            else:
                # Use default configuration
                configs[task] = DatasetConfig(
                    task_name=task,
                    train_samples=min(50, self.samples_per_client // len(tasks)),
                    val_samples=min(10, (self.samples_per_client // len(tasks)) // 5),
                    random_seed=42
                )

        return configs

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description="Federated Learning System")

    # Mode selection
    parser.add_argument("--mode", choices=["server", "client"], required=True,
                       help="Run mode: server or client")

    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--config-file", type=str, default="federated_config.yaml",
                       help="Configuration file name")

    # Common arguments
    parser.add_argument("--port", type=int, default=8771, help="Server port")
    parser.add_argument("--rounds", type=int, default=2, help="Number of training rounds")
    parser.add_argument("--samples", type=int, default=100, help="Samples per client")

    # Client-specific arguments
    parser.add_argument("--client_id", type=str, help="Client ID (required for client mode)")
    parser.add_argument("--tasks", nargs='+', choices=["sst2", "qqp", "stsb"],
                       help="Task names for client (space-separated)")

    # Advanced arguments
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    return parser

def load_config(args) -> FederatedConfig:
    """Load configuration from file or create default"""
    config_file = args.config or args.config_file

    try:
        # Try to load from YAML file
        config = FederatedConfig.from_yaml_file(config_file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not load config file {config_file}: {e}")
        print("Using default configuration...")
        config = FederatedConfig()

    return config

# For backward compatibility
create_config_from_args = load_config
