#!/usr/bin/env python3
"""
STSB Federated Learning Client Implementation
Specialized client for semantic textual similarity regression tasks
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

class STSBFederatedClient(BaseFederatedClient):
    """Federated Learning Client specialized for STSB semantic similarity regression"""

    def __init__(self, client_id: str, config: FederatedConfig):
        super().__init__(client_id, "stsb", config)

    async def train_task(self, task: str, task_data: Dict) -> Dict[str, float]:
        """Train on STSB semantic similarity regression task"""
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

        logger.info(f"[TRAINING] Starting STSB training with {len(train_dataloader)} batches")

        # Training loop with regression-specific metrics
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        tolerance_correct = 0
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

            # For STSB regression, logits.squeeze() to get scalar predictions
            predictions = logits.squeeze()

            # Calculate MSE loss (regression task)
            loss = F.mse_loss(predictions, labels.float())

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

            # Collect predictions and labels for correlation calculation
            predictions_np = predictions.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            all_predictions.extend(predictions_np)
            all_labels.extend(labels_np)

            # Calculate tolerance-based accuracy for regression
            tolerance = 0.1  # 10% tolerance
            pred_reshaped = predictions.unsqueeze(0) if predictions.dim() == 0 else predictions
            labels_reshaped = labels.unsqueeze(0) if labels.dim() == 0 else labels
            tolerance_correct += (torch.abs(pred_reshaped - labels_reshaped) <= tolerance).sum().item()
            total_samples += labels.size(0)

            # Log progress every few batches
            if num_batches % 5 == 0:
                logger.info(f"STSB - Batch {num_batches}, Loss: {loss.item():.4f}")

        # Update learning rate scheduler
        self.scheduler.step()

        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Calculate regression-specific metrics
        if len(all_predictions) > 0 and len(all_labels) > 0:
            pred_array = np.array(all_predictions)
            label_array = np.array(all_labels)

            # Mean Absolute Error
            mae = np.mean(np.abs(pred_array - label_array))

            # Mean Squared Error
            mse = np.mean((pred_array - label_array) ** 2)

            # Correlation coefficient (handle edge cases)
            if len(pred_array) > 1 and np.std(pred_array) > 0 and np.std(label_array) > 0:
                correlation = np.corrcoef(pred_array, label_array)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0

            # Use correlation as primary accuracy metric for regression
            regression_accuracy = max(0, correlation)  # Clamp negative correlations to 0

            logger.info(f"[STATS] STSB - Batch {num_batches}, Loss: {kd_loss.item():.4f}")
            logger.info(f"[REGRESSION] STSB Regression Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, Correlation: {correlation:.4f}")
        logger.info(f"[SUCCESS] STSB training completed - Loss: {avg_loss:.4f}, Accuracy: {regression_accuracy:.4f}")
        logger.info(f"[SUCCESS] STSB Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

        # Add validation metrics if validation data is available
        metrics = {
            'loss': avg_loss,
            'accuracy': float(regression_accuracy),
            'correct_predictions': tolerance_correct,
            'mae': float(mae),
            'mse': float(mse),
            'correlation': float(correlation)
        }

        if val_dataloader is not None:
            val_metrics = self.evaluate_on_validation(task, val_dataloader)
            metrics.update({
                'val_accuracy': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_samples': val_metrics['samples_processed'],
                'val_mae': val_metrics.get('mae', 0.0),
                'val_correlation': val_metrics.get('correlation', 0.0)
            })
            logger.info(f"[SUCCESS] STSB Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

        return metrics

    def evaluate_on_validation(self, task: str, val_dataloader) -> Dict[str, float]:
        """Evaluate model on validation data for STSB regression"""
        # Set model to evaluation mode
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        tolerance_correct = 0
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

                # For STSB regression, logits.squeeze() to get scalar predictions
                predictions = logits.squeeze()

                # Calculate loss (MSE for regression)
                loss = F.mse_loss(predictions, labels.float())
                total_loss += loss.item()
                num_batches += 1

                # Collect predictions and labels for correlation calculation
                predictions_np = predictions.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()

                all_predictions.extend(predictions_np)
                all_labels.extend(labels_np)

                # Calculate tolerance-based accuracy for regression
                tolerance = 0.1  # 10% tolerance
                pred_reshaped = predictions.unsqueeze(0) if predictions.dim() == 0 else predictions
                labels_reshaped = labels.unsqueeze(0) if labels.dim() == 0 else labels
                tolerance_correct += (torch.abs(pred_reshaped - labels_reshaped) <= tolerance).sum().item()
                total_samples += labels.size(0)

        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Calculate regression-specific metrics
        if len(all_predictions) > 0 and len(all_labels) > 0:
            pred_array = np.array(all_predictions)
            label_array = np.array(all_labels)

            # Mean Absolute Error
            mae = np.mean(np.abs(pred_array - label_array))

            # Mean Squared Error
            mse = np.mean((pred_array - label_array) ** 2)

            # Correlation coefficient (handle edge cases)
            if len(pred_array) > 1 and np.std(pred_array) > 0 and np.std(label_array) > 0:
                correlation = np.corrcoef(pred_array, label_array)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0

            # Use correlation as primary accuracy metric for regression
            regression_accuracy = max(0, correlation)  # Clamp negative correlations to 0
        else:
            mae = mse = correlation = 0.0
            regression_accuracy = 0.0

        # Set model back to training mode
        self.model.train()

        return {
            'loss': avg_loss,
            'accuracy': float(regression_accuracy),
            'samples_processed': total_samples,
            'correct_predictions': tolerance_correct,
            'mae': float(mae),
            'mse': float(mse),
            'correlation': float(correlation)
        }

def run_stsb_client(client_id: str, config: FederatedConfig):
    """Run STSB specialized federated learning client"""
    client = STSBFederatedClient(client_id, config)
    asyncio.run(client.run_client())

if __name__ == "__main__":
    import argparse
    from federated_config import create_argument_parser, load_config

    parser = create_argument_parser()
    parser.add_argument("--task", type=str, default="stsb", help="Task for this client")

    args = parser.parse_args()

    # Validate arguments for client mode
    if args.mode != "client":
        parser.error("This script is for client mode only.")

    if not args.client_id:
        parser.error("Client ID is required for client mode.")

    config = load_config(args)

    # Override task to STSB for this specialized client
    if hasattr(args, 'task'):
        args.tasks = [args.task]

    config.print_summary()

    run_stsb_client(args.client_id, config)
