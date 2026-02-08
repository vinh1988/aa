#!/usr/bin/env python3
"""
QQP Federated Learning Client Implementation
Specialized client for question pair matching tasks
(Knowledge Distillation removed)
"""

import asyncio
import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict

from federated_config import FederatedConfig
from src.core.base_federated_client import BaseFederatedClient

logger = logging.getLogger(__name__)

class QQPFederatedClient(BaseFederatedClient):
    """Federated Learning Client specialized for QQP question pair matching"""

    def __init__(self, client_id: str, config: FederatedConfig):
        super().__init__(client_id, "qqp", config)

    async def train_task(self, task: str, task_data: Dict) -> Dict[str, float]:
        """Train on QQP question pair matching task"""
        # Split data into training and validation
        train_data = {
            'texts': task_data.get('texts', []),
            'labels': task_data.get('labels', [])
        }
        val_data = {
            'texts': task_data.get('val_texts', []),
            'labels': task_data.get('val_labels', [])
        }

        # Get dataloaders
        train_dataloader = self.model.get_task_dataloader(
            task, self.config.batch_size, dataset_data=train_data
        )
        val_dataloader = None
        if val_data['texts'] and val_data['labels']:
            val_dataloader = self.model.get_task_dataloader(
                task, self.config.batch_size, dataset_data=val_data
            )

        # Set model to training mode
        self.model.train()

        logger.info(f"[TRAINING] Starting QQP training with {len(train_dataloader)} batches")

        # Training loop with proper metrics calculation
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_samples = 0

        for batch in train_dataloader:
            # Unpack batch tuple (input_ids, attention_mask, labels)
            input_ids, attention_mask, labels = batch

            # Move tensors to the correct device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            if len(input_ids) == 0:
                continue

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(
                input_ids,
                attention_mask,
                task
            )

            # Calculate cross-entropy loss (standard classification loss)
            loss = F.cross_entropy(logits, labels)

            # Backward pass
            loss.backward()
            
            # PHASE 2: Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            # Update parameters
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Calculate predictions and accuracy for QQP (binary classification)
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Log progress every few batches
            if num_batches % 5 == 0:
                logger.info(f"QQP - Batch {num_batches}, Loss: {loss.item():.4f}")

        # Update learning rate scheduler
        self.scheduler.step()

        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        logger.info(f"[SUCCESS] QQP training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Add validation metrics if validation data is available
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples_processed': total_samples,
            'correct_predictions': correct_predictions
        }

        if val_dataloader is not None:
            val_metrics = self.evaluate_on_validation(task, val_dataloader)
            metrics.update({
                'val_accuracy': val_metrics['accuracy'],
            })
            logger.info(f"[SUCCESS] QQP Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

        return metrics

    def evaluate_on_validation(self, task: str, val_dataloader) -> Dict[str, float]:
        """Evaluate model on validation data for QQP"""
        
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = batch

                # Move tensors to the correct device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                if len(input_ids) == 0:
                    continue

                # Forward pass
                logits = self.model(input_ids, attention_mask, task)

                # Calculate loss (standard cross-entropy)
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                num_batches += 1

                # Calculate predictions for QQP (binary classification)
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        # Set model back to training mode
        self.model.train()

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples_processed': total_samples,
            'correct_predictions': correct_predictions
        }

def run_qqp_client(client_id: str, config: FederatedConfig):
    """Run QQP specialized federated learning client"""
    client = QQPFederatedClient(client_id, config)
    asyncio.run(client.run_client())

if __name__ == "__main__":
    import argparse
    from federated_config import create_argument_parser, load_config

    parser = create_argument_parser()
    parser.add_argument("--task", type=str, default="qqp", help="Task for this client")

    args = parser.parse_args()

    # Validate arguments for client mode
    if args.mode != "client":
        parser.error("This script is for client mode only.")

    if not args.client_id:
        parser.error("Client ID is required for client mode.")

    config = load_config(args)

    # Override task to QQP for this specialized client
    if hasattr(args, 'task'):
        args.tasks = [args.task]

    config.print_summary()

    run_qqp_client(args.client_id, config)
