#!/usr/bin/env python3
"""
Federated Learning Client Implementation
Handles local training and synchronization
(Standard FL - No LoRA, No KD)
"""

import asyncio
import json
import logging
import time
import torch
import torch.nn.functional as F
import csv
import os
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from federated_config import FederatedConfig
from src.models.standard_bert import StandardBERTModel
from src.communication.federated_websockets import WebSocketClient, MessageProtocol
from src.synchronization.federated_synchronization import ClientModelSynchronizer
from src.datasets.federated_datasets import DatasetFactory, DatasetConfig

logger = logging.getLogger(__name__)

class FederatedClient:
    """Federated Learning Client with synchronization support"""

    def __init__(self, client_id: str, tasks: List[str], config: FederatedConfig):
        self.client_id = client_id
        self.tasks = tasks
        self.config = config
        self.device = self.get_device()

        # Initialize standard BERT model (full fine-tuning)
        self.model = StandardBERTModel(
            base_model_name=config.client_model,
            tasks=tasks
        )
        # Move model to device
        self.model = self.model.to(self.device)

        # Initialize optimizer for training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1, 
            gamma=0.9
        )

        # Initialize synchronization with configurable send timeout (with primary task for MTL)
        send_timeout = getattr(config, 'send_timeout', 3600)
        self.websocket_client = WebSocketClient(
            f"ws://localhost:{config.port}",
            client_id,
            send_timeout=send_timeout
        )
        primary_task = self.tasks[0] if self.tasks else 'sst2'
        self.model_synchronizer = ClientModelSynchronizer(
            self.model, self.websocket_client, primary_task
        )

        # Initialize datasets
        self.dataset_handlers = self.initialize_dataset_handlers()

        # Setup logging
        self.setup_logging()

        # Training state
        self.is_training = False
        self.current_round = 0
        self.last_global_model_metrics = {}  # Store global model validation metrics
        
        # Device usage tracking
        self.device_usage_records = []
        self.csv_output_dir = config.results_dir
        os.makedirs(self.csv_output_dir, exist_ok=True)
        
        # Create a single CSV file for this client session
        session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = os.path.join(
            self.csv_output_dir,
            f"device_usage_{self.client_id}_{session_timestamp}.csv"
        )
        self.csv_initialized = False

    def get_device(self):
        """Get available device (GPU/CPU)"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Log device information
        if device.type == "cuda":
            logger.info(f"[GPU] Using CUDA (GPU) for training")
            logger.info(f"[GPU] GPU Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"[GPU] Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning(f"[WARNING] Using CPU for training (CUDA not available)")
            logger.warning(f"[WARNING] Training will be significantly slower on CPU")
        
        return device

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'federated_client_{self.client_id}.log'),
                logging.StreamHandler()
            ]
        )
        
        # Log device information
        logger.info("="*60)
        logger.info(f"CLIENT: {self.client_id}")
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"[GPU] CUDA is available")
            logger.info(f"[GPU] GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning(f"[WARNING] CUDA not available - using CPU only")
            logger.warning(f"[WARNING] Training will be much slower!")
        logger.info("="*60)

    def initialize_dataset_handlers(self) -> Dict[str, Any]:
        """Initialize dataset handlers for available tasks"""
        handlers = {}

        for task in self.tasks:
            if task in self.config.task_configs:
                config = self.config.task_configs[task]
                dataset_config = DatasetConfig(
                    task_name=task,
                    train_samples=config.get('train_samples'),
                    val_samples=config.get('val_samples'),
                    random_seed=config.get('random_seed', 42)
                )
                handlers[task] = DatasetFactory.create_handler(task, dataset_config)

        return handlers

    async def connect_and_register(self):
        """Connect to server and register"""
        await self.websocket_client.connect()

        # Register with server
        registration_message = MessageProtocol.create_registration_message(
            self.client_id,
            self.tasks,
            {
                "model": self.config.client_model,
                "supported_tasks": self.tasks
            }
        )

        await self.websocket_client.send(registration_message)
        logger.info(f"Client {self.client_id} registered with tasks: {self.tasks}")

    async def run_client(self):
        """Main client execution loop"""
        try:
            # Connect and register
            await self.connect_and_register()

            # Setup message handlers
            self.setup_message_handlers()

            # Keep connection alive
            while self.websocket_client.is_connected:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Client shutting down...")
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            # Save final device usage metrics before disconnecting
            logger.info("Saving final device usage metrics...")
            self.save_device_usage_to_csv()
            await self.websocket_client.disconnect()

    def setup_message_handlers(self):
        """Setup WebSocket message handlers"""
        self.websocket_client.register_message_handler(
            "global_model_sync", self.handle_global_model_sync
        )
        self.websocket_client.register_message_handler(
            "training_request", self.handle_training_request
        )
        self.websocket_client.register_message_handler(
            "registration_ack", self.handle_registration_ack
        )
        self.websocket_client.register_message_handler(
            "heartbeat", self.handle_heartbeat
        )

    async def handle_global_model_sync(self, data: Dict):
        """Handle incoming global model synchronization"""
        logger.info(f"Received global model synchronization")

        # Update local model with global knowledge
        sync_result = await self.model_synchronizer.synchronize_with_global_model(
            data["global_model_state"]
        )

        # Validate global model on client's local validation data
        logger.info(f"Validating global model on local validation data...")
        global_model_metrics = await self.validate_global_model()
        
        # Store global model metrics for later recording
        self.last_global_model_metrics = global_model_metrics
        
        logger.info(f"Global model validation complete: {global_model_metrics}")

        # Send acknowledgment
        await self.model_synchronizer.send_synchronization_acknowledgment(sync_result)

    async def handle_registration_ack(self, data: Dict):
        """Handle registration acknowledgment from server"""
        accepted = data.get("accepted", False)
        message = data.get("message", "")

        if accepted:
            logger.info(f"Registration acknowledged: {message}")
        else:
            logger.warning(f"Registration rejected: {message}")

    async def handle_heartbeat(self, data: Dict):
        """Handle heartbeat messages to keep connection alive"""
        # Send heartbeat response to keep connection alive
        heartbeat_response = MessageProtocol.create_heartbeat_message(self.client_id)
        await self.websocket_client.send(heartbeat_response)

    async def handle_training_request(self, data: Dict):
        """Handle training request from server"""
        round_num = data["round"]
        global_params = data.get("global_params", {})

        logger.info(f"Received training request for round {round_num}")
        logger.info(f"Starting training for client {self.client_id} with tasks: {self.tasks}")

        # Perform local training
        self.is_training = True
        self.current_round = round_num

        try:
            local_metrics = await self.perform_local_training()

            # Add global model validation metrics to the task metrics
            logger.info(f"[MERGE] Checking global model metrics: last_global_model_metrics={bool(self.last_global_model_metrics)}")
            logger.info(f"[MERGE] local_metrics keys: {local_metrics.keys()}")
            logger.info(f"[MERGE] last_global_model_metrics: {list(self.last_global_model_metrics.keys())}")
            
            if self.last_global_model_metrics and local_metrics:
                # Merge global model metrics into each task's metrics
                for task in self.tasks:
                    if task in local_metrics:
                        local_metrics[task].update(self.last_global_model_metrics)
                        logger.info(f"✅ [MERGE] Successfully added global model metrics to task {task}")
            else:
                logger.warning(f"⚠️ [MERGE] Failed to merge global model metrics! last_global={bool(self.last_global_model_metrics)}, local_metrics={bool(local_metrics)}")

            # Clear GPU cache after training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("[GPU-CLEANUP] Cleared GPU cache after training")

            # Extract model parameters
            model_updates = self.model.get_all_parameters()

            # Send update to server (include task label for MTL aggregation)
            update_message = MessageProtocol.create_client_update_message(
                self.client_id,
                round_num,
                model_updates,
                local_metrics,
                {}
            )
            # Add task labels for MTL-aware aggregation
            update_message['tasks'] = self.tasks

            # Use retry logic for sending updates
            logger.info(f"Attempting to send update to server for round {round_num}")
            success = await self.websocket_client.send(update_message, max_retries=5)
            if success:
                logger.info(f"Training completed and update sent for round {round_num}")
                logger.info(f"Client {self.client_id} metrics: {local_metrics}")
                logger.info(f"[SUCCESS] Training completed and update sent for round {round_num}")
                logger.info(f"[METRICS] Client {self.client_id} metrics: {local_metrics}")
            else:
                logger.error(f"Failed to send update for round {round_num} after retries")
                logger.error(f"WebSocket connection status: {self.websocket_client.is_connected}")
                logger.error(f"Update message size: {len(str(update_message))} characters")

        except Exception as e:
            logger.error(f"Error in local training for round {round_num}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.is_training = False
            # Clear GPU cache to prevent memory accumulation
            if torch.cuda.is_available():
                # Get memory before clearing
                allocated_before = torch.cuda.memory_allocated() / 1e9
                reserved_before = torch.cuda.memory_reserved() / 1e9
                
                torch.cuda.empty_cache()
                
                # Get memory after clearing
                allocated_after = torch.cuda.memory_allocated() / 1e9
                reserved_after = torch.cuda.memory_reserved() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                freed = reserved_before - reserved_after
                
                logger.info(f"=" * 60)
                logger.info(f"[COMPLETE] Round {round_num} completed on {self.device}")
                logger.info(f"[GPU-MEMORY] Final GPU Memory Status:")
                logger.info(f"[GPU-MEMORY] Memory freed in cleanup: {freed:.2f} GB")
                logger.info(f"[GPU-MEMORY] Allocated: {allocated_after:.2f} GB / {total:.2f} GB ({allocated_after/total*100:.1f}%)")
                logger.info(f"[GPU-MEMORY] Reserved:  {reserved_after:.2f} GB / {total:.2f} GB ({reserved_after/total*100:.1f}%)")
                logger.info(f"[GPU-MEMORY] Free:      {total - reserved_after:.2f} GB ({(total - reserved_after)/total*100:.1f}%)")
                logger.info(f"=" * 60)
            else:
                logger.info(f"[COMPLETE] Round {round_num} completed on CPU")
            
            # Record device usage at round end and save to CSV
            self.record_device_usage(round_num, 'all_tasks', 'round_complete')
            self.save_device_usage_to_csv()

    async def validate_global_model(self) -> Dict[str, float]:
        """Validate the global model on client's local validation data"""
        logger.info(f"[VALIDATION] Starting global model validation for tasks {self.tasks}")
        
        all_metrics = {}
        
        for task in self.tasks:
            if task not in self.dataset_handlers:
                logger.warning(f"[VALIDATION] No dataset handler available for task {task}")
                continue

            # Get validation data
            dataset = self.dataset_handlers[task]
            task_data = dataset.prepare_data()
            val_data = {
                'texts': task_data.get('val_texts', []),
                'labels': task_data.get('val_labels', [])
            }

            logger.info(f"[VALIDATION] Task {task}: {len(val_data['texts'])} validation samples")
            
            if not val_data['texts'] or not val_data['labels']:
                logger.warning(f"[VALIDATION] No validation data available for task {task}")
                continue

            # Get validation dataloader
            val_dataloader = self.model.get_task_dataloader(
                task, self.config.batch_size, dataset_data=val_data
            )

            # Set model to evaluation mode
            self.model.eval()

            # Validation metrics
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            # For F1, Pearson, Spearman
            all_predictions = []
            all_labels = []
            
            # For classification F1
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            is_regression = (task == 'stsb')

            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    logits = self.model(input_ids, attention_mask, task)

                    if is_regression:
                        # Regression task (STS-B)
                        logits = logits.squeeze(-1)
                        loss = F.mse_loss(logits, labels)
                        total_loss += loss.item() * len(labels)
                        
                        # Collect for correlation
                        all_predictions.extend(logits.cpu().numpy().tolist())
                        all_labels.extend(labels.cpu().numpy().tolist())
                        
                        total_samples += len(labels)
                    else:
                        # Classification task (SST-2, QQP)
                        loss = F.cross_entropy(logits, labels)
                        total_loss += loss.item() * len(labels)
                        
                        predictions = torch.argmax(logits, dim=1)
                        correct_predictions += (predictions == labels).sum().item()
                        total_samples += len(labels)
                        
                        # Collect for F1
                        all_predictions.extend(predictions.cpu().numpy().tolist())
                        all_labels.extend(labels.cpu().numpy().tolist())
                        
                        # Calculate TP, FP, FN for F1
                        for pred, label in zip(predictions.cpu().numpy(), labels.cpu().numpy()):
                            if pred == 1 and label == 1:
                                true_positives += 1
                            elif pred == 1 and label == 0:
                                false_positives += 1
                            elif pred == 0 and label == 1:
                                false_negatives += 1

            # Calculate metrics
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            metrics = {
                'global_model_val_loss': avg_loss,
                'global_model_val_samples': total_samples
            }

            if is_regression:
                # Calculate Pearson and Spearman correlation
                import numpy as np
                from scipy.stats import pearsonr, spearmanr
                
                predictions_np = np.array(all_predictions)
                labels_np = np.array(all_labels)
                
                pearson_corr, _ = pearsonr(predictions_np, labels_np)
                spearman_corr, _ = spearmanr(predictions_np, labels_np)
                
                metrics['global_model_val_pearson'] = pearson_corr
                metrics['global_model_val_spearman'] = spearman_corr
                
                # Also calculate "accuracy-like" metric for consistency
                mse = ((predictions_np - labels_np) ** 2).mean()
                accuracy_like = max(0, 1 - mse)
                metrics['global_model_val_accuracy'] = accuracy_like
            else:
                # Calculate accuracy and F1 for classification
                accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                metrics['global_model_val_accuracy'] = accuracy
                metrics['global_model_val_f1'] = f1
                metrics['global_model_val_precision'] = precision
                metrics['global_model_val_recall'] = recall
                metrics['global_model_val_correct_predictions'] = correct_predictions

            # Back to training mode
            self.model.train()
            
            logger.info(f"[VALIDATION] Task {task} global model metrics: {metrics}")
            all_metrics.update(metrics)  # Merge all task metrics
        
        logger.info(f"[VALIDATION] Complete. Total metrics: {list(all_metrics.keys())}")
        return all_metrics

    async def perform_local_training(self) -> Dict[str, float]:
        """Perform local training with KD"""
        # Record device usage at training start
        self.record_device_usage(self.current_round, 'all_tasks', 'training_start')
        
        # Log device information at the start of training
        logger.info(f"=" * 60)
        logger.info(f"Starting local training on device: {self.device}")
        
        if torch.cuda.is_available():
            # Clear GPU cache before training to ensure clean start
            torch.cuda.empty_cache()
            
            # Log GPU memory status
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"[GPU-MEMORY] GPU Memory Status:")
            logger.info(f"[GPU-MEMORY] Allocated: {allocated:.2f} GB / {total:.2f} GB ({allocated/total*100:.1f}%)")
            logger.info(f"[GPU-MEMORY] Reserved:  {reserved:.2f} GB / {total:.2f} GB ({reserved/total*100:.1f}%)")
            logger.info(f"[GPU-MEMORY] Free:      {total - reserved:.2f} GB")
        else:
            logger.info(f"[CPU] Training on CPU (CUDA not available)")
        
        logger.info(f"=" * 60)
        
        local_metrics = {}

        for task in self.tasks:
            if task in self.dataset_handlers:
                # Get data for this task
                dataset = self.dataset_handlers[task]
                task_data = dataset.prepare_data()

                # Train on this task with KD
                task_metrics = await self.train_task_with_kd(task, task_data)
                local_metrics[task] = task_metrics

        return local_metrics

    async def train_task_with_kd(self, task: str, task_data: Dict) -> Dict[str, float]:
        """Train on a specific task with KD"""
        
        # Split data into training and validation
        # Split data into training and validation
        # Dataset handler returns: texts/labels (train) and val_texts/val_labels (validation)
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
            logger.info(f"Validation dataloader created for task {task} with batch_size {self.config.batch_size}")
            logger.info(f"[VALIDATION] Validation dataloader created for task {task} with batch_size {self.config.batch_size}")

        logger.info(f"Training dataloader created for task {task} with batch_size {self.config.batch_size}")
        logger.info(f"[TRAINING] Training dataloader created for task {task} with batch_size {self.config.batch_size}")

        # Set model to training mode
        self.model.train()
        
        logger.info(f"Starting training for task {task} with {len(train_dataloader)} batches")
        logger.info(f"[TRAINING] Starting training for task {task} with {len(train_dataloader)} batches")
        logger.info(f"[DEVICE] Training tensors will be moved to: {self.device}")
        if val_dataloader:
            logger.info(f"Starting validation for task {task} with {len(val_dataloader)} batches")
            logger.info(f"[VALIDATION] Validation dataloader created for task {task} with {len(val_dataloader)} batches")

        # Training loop with proper metrics calculation
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        correct_predictions = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_dataloader):
            try:
                # Unpack batch tuple (input_ids, attention_mask, labels)
                input_ids, attention_mask, labels = batch

                # Move tensors to the correct device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Log device confirmation for first batch
                if num_batches == 0:
                    logger.info(f"[DEVICE] Batch tensors moved to {self.device} (input_ids device: {input_ids.device})")

                # Add check for empty or scalar batches
                if len(input_ids) == 0 or input_ids.dim() == 0:
                    logger.warning(f"Skipping empty or scalar batch for task {task}")
                    continue
                
                # Validate batch dimensions
                if input_ids.dim() != 2:
                    logger.error(f"Invalid input_ids dimensions: {input_ids.shape}, expected 2D tensor")
                    continue
                if input_ids.size(0) == 0:
                    logger.error(f"Batch size is 0, skipping")
                    continue

                # Ensure labels are not scalars
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(
                    input_ids,
                    attention_mask,
                    task
                )

                # Calculate loss based on task type
                if task == 'stsb':
                    # Regression task - use MSE loss
                    predictions = logits.squeeze()
                    loss = F.mse_loss(predictions, labels.float())
                else:
                    # Classification tasks - use cross-entropy loss
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

                # Store loss value before deleting tensor
                batch_loss = loss.item()
                total_loss += batch_loss
                num_batches += 1

                # Log progress every few batches
                if num_batches % 5 == 0:
                    logger.info(f"Task {task} - Batch {num_batches}, Loss: {batch_loss:.4f}")
                    logger.info(f"[STATS] Task {task} - Batch {num_batches}, Loss: {batch_loss:.4f}")
                
                # Periodic GPU cache clearing during long training
                if torch.cuda.is_available() and num_batches % 100 == 0:
                    torch.cuda.empty_cache()
                    logger.debug(f"Periodic GPU cache clear at batch {num_batches}")

                # Calculate predictions and accuracy
                with torch.no_grad():
                    if task == 'stsb':  # Regression task
                        predictions = logits.squeeze()
                        # For regression, use a tolerance-based accuracy
                        # Consider predictions "correct" if they're within 0.1 of the true label
                        if labels.dim() == 0:
                            labels_reshaped = labels.unsqueeze(0)
                        else:
                            labels_reshaped = labels
                        
                        # Ensure predictions are not scalars
                        if predictions.dim() == 0:
                            predictions = predictions.unsqueeze(0)
                        
                        # Calculate tolerance-based accuracy for regression
                        tolerance = 0.05  # Within 0.05 of true value (5% tolerance)
                        correct_predictions += (torch.abs(predictions - labels_reshaped) <= tolerance).sum().item()
                    else:  # Classification tasks
                        predictions = torch.argmax(logits, dim=1)
                        correct_predictions += (predictions == labels).sum().item()
                    
                    total_samples += labels.size(0)
                    
                    # Only extend lists if predictions and labels are arrays, not scalars
                    pred_cpu = predictions.cpu()
                    if pred_cpu.numel() > 1:  # More than one element
                        all_predictions.extend(pred_cpu.numpy().flatten())
                    else:
                        all_predictions.append(pred_cpu.item())
                    
                    label_cpu = labels.cpu()
                    if label_cpu.numel() > 1:
                        all_labels.extend(label_cpu.numpy().flatten())
                    else:
                        all_labels.append(label_cpu.item())
                
                # Explicitly delete tensors to free memory
                del input_ids, attention_mask, labels, logits, loss
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx} for task {task}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Continue to next batch instead of crashing
                continue

        # Update learning rate scheduler
        self.scheduler.step()

        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        logger.info(f"Task {task} training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        logger.info(f"[SUCCESS] Task {task} training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Initialize metrics dictionary
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples_processed': total_samples,
            'correct_predictions': correct_predictions
        }
        
        # Add task-specific metrics
        if task == 'stsb' and len(all_predictions) > 0 and len(all_labels) > 0:
            # Calculate additional regression metrics
            pred_array = np.array(all_predictions)
            label_array = np.array(all_labels)
            
            # Ensure arrays are not empty and have proper shape
            if pred_array.size == 0 or label_array.size == 0:
                logger.warning(f"Empty prediction or label arrays for task {task}")
                pred_array = np.array([0.0])  # Fallback
                label_array = np.array([0.0])
            
            # Handle scalar arrays
            if pred_array.ndim == 0:
                pred_array = np.array([pred_array])
            if label_array.ndim == 0:
                label_array = np.array([label_array])
            
            # Mean Absolute Error
            mae = np.mean(np.abs(pred_array - label_array))
            # Mean Squared Error
            mse = np.mean((pred_array - label_array) ** 2)
            # Root Mean Squared Error
            rmse = np.sqrt(mse)
            
            # Pearson correlation coefficient (handle edge cases)
            if len(pred_array) > 1 and np.std(pred_array) > 0 and np.std(label_array) > 0:
                try:
                    pearson_corr = np.corrcoef(pred_array, label_array)[0, 1]
                    if np.isnan(pearson_corr) or np.isinf(pearson_corr):
                        pearson_corr = 0.0
                except (ValueError, IndexError):
                    pearson_corr = 0.0
            else:
                pearson_corr = 0.0
            
            # Spearman correlation coefficient
            if len(pred_array) > 1 and np.std(pred_array) > 0 and np.std(label_array) > 0:
                try:
                    spearman_corr, _ = spearmanr(pred_array, label_array)
                    if np.isnan(spearman_corr) or np.isinf(spearman_corr):
                        spearman_corr = 0.0
                except (ValueError, IndexError):
                    spearman_corr = 0.0
            else:
                spearman_corr = 0.0
            
            # For regression, use Pearson correlation as the primary accuracy metric
            regression_accuracy = max(0, pearson_corr)  # Clamp negative correlations to 0
            
            # Tolerance-based correct count
            tolerance = 0.1  # 10% tolerance
            tolerance_correct = np.sum(np.abs(pred_array - label_array) <= tolerance)
            
            logger.info(f"STSB Regression Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")
            logger.info(f"[REGRESSION] STSB Regression Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")
            
            metrics.update({
                'accuracy': float(regression_accuracy),
                'correct_predictions': int(tolerance_correct),
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'pearson_correlation': float(pearson_corr),
                'spearman_correlation': float(spearman_corr)
            })
        elif task in ['sst2', 'qqp'] and len(all_predictions) > 0 and len(all_labels) > 0:
            # Calculate F1 score for classification tasks
            pred_array = np.array(all_predictions)
            label_array = np.array(all_labels)
            
            # Calculate F1 score (weighted average for multi-class)
            f1 = f1_score(label_array, pred_array, average='weighted', zero_division=0)
            
            logger.info(f"{task.upper()} Classification Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            logger.info(f"[CLASSIFICATION] {task.upper()} Classification Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            metrics.update({
                'f1_score': float(f1)
            })

        # Add validation metrics if validation data is available
        if val_dataloader is not None:
            val_metrics = self.evaluate_on_validation(task, val_dataloader)
            metrics['val_accuracy'] = val_metrics['accuracy']
            metrics['val_loss'] = val_metrics['loss']
            metrics['val_samples'] = val_metrics['samples_processed']
            metrics['val_correct_predictions'] = val_metrics['correct_predictions']
            
            # Add task-specific validation metrics
            if 'pearson_correlation' in val_metrics:
                metrics['val_pearson_correlation'] = val_metrics['pearson_correlation']
                metrics['val_spearman_correlation'] = val_metrics['spearman_correlation']
                metrics['val_mae'] = val_metrics['mae']
                logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, Pearson: {val_metrics['pearson_correlation']:.4f}, Spearman: {val_metrics['spearman_correlation']:.4f}")
                logger.info(f"[VALIDATION] Validation - Loss: {val_metrics['loss']:.4f}, Pearson: {val_metrics['pearson_correlation']:.4f}, Spearman: {val_metrics['spearman_correlation']:.4f}")
            elif 'f1_score' in val_metrics:
                metrics['val_f1_score'] = val_metrics['f1_score']
                logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
                logger.info(f"[VALIDATION] Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            else:
                logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
                logger.info(f"[VALIDATION] Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

        # Record device usage after task completion
        self.record_device_usage(self.current_round, task, 'task_complete', metrics)

        return metrics

    def evaluate_on_validation(self, task: str, val_dataloader) -> Dict[str, float]:
        """Evaluate model on validation data"""
        # Clear GPU cache before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"Cleared GPU cache before validation for task {task}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = batch
                
                # Move tensors to the correct device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Add check for empty or scalar batches
                if len(input_ids) == 0 or input_ids.dim() == 0:
                    logger.warning(f"Skipping empty or scalar validation batch for task {task}")
                    continue

                # Ensure labels are not scalars
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask, task)
                
                # Calculate loss based on task type
                if task == 'stsb':
                    # Regression task - use MSE loss
                    pred = logits.squeeze()
                    loss = F.mse_loss(pred, labels.float())
                else:
                    # Classification tasks - use cross-entropy loss
                    loss = F.cross_entropy(logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate predictions
                if task == 'stsb':  # Regression task
                    predictions = logits.squeeze()
                    tolerance = 0.1
                    if labels.dim() == 0:
                        labels_reshaped = labels.unsqueeze(0)
                    else:
                        labels_reshaped = labels
                    pred_reshaped = predictions.unsqueeze(0) if predictions.dim() == 0 else predictions
                    correct_predictions += (torch.abs(pred_reshaped - labels_reshaped) <= tolerance).sum().item()
                else:  # Classification tasks
                    predictions = torch.argmax(logits, dim=1)
                    correct_predictions += (predictions == labels).sum().item()
                
                total_samples += labels.size(0)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Clean up tensors after each validation batch
                del input_ids, attention_mask, labels, logits, loss, predictions
        
        # Clear GPU cache after validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"Cleared GPU cache after validation for task {task}")
        
        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples_processed': total_samples,
            'correct_predictions': correct_predictions
        }
        
        # Add task-specific metrics
        if task == 'stsb' and len(all_predictions) > 0 and len(all_labels) > 0:
            pred_array = np.array(all_predictions)
            label_array = np.array(all_labels)
            
            # Ensure arrays are not empty and have proper shape
            if pred_array.size == 0 or label_array.size == 0:
                logger.warning(f"Empty prediction or label arrays for validation task {task}")
                pred_array = np.array([0.0])  # Fallback
                label_array = np.array([0.0])
            
            # Handle scalar arrays
            if pred_array.ndim == 0:
                pred_array = np.array([pred_array])
            if label_array.ndim == 0:
                label_array = np.array([label_array])
            
            mae = np.mean(np.abs(pred_array - label_array))
            mse = np.mean((pred_array - label_array) ** 2)
            
            # Pearson correlation coefficient
            if len(pred_array) > 1 and np.std(pred_array) > 0 and np.std(label_array) > 0:
                try:
                    pearson_corr = np.corrcoef(pred_array, label_array)[0, 1]
                    if np.isnan(pearson_corr) or np.isinf(pearson_corr):
                        pearson_corr = 0.0
                except (ValueError, IndexError):
                    pearson_corr = 0.0
            else:
                pearson_corr = 0.0
            
            # Spearman correlation coefficient
            if len(pred_array) > 1 and np.std(pred_array) > 0 and np.std(label_array) > 0:
                try:
                    spearman_corr, _ = spearmanr(pred_array, label_array)
                    if np.isnan(spearman_corr) or np.isinf(spearman_corr):
                        spearman_corr = 0.0
                except (ValueError, IndexError):
                    spearman_corr = 0.0
            else:
                spearman_corr = 0.0
            
            regression_accuracy = max(0, pearson_corr)
            tolerance_correct = np.sum(np.abs(pred_array - label_array) <= 0.1)
            
            metrics.update({
                'accuracy': float(regression_accuracy),
                'correct_predictions': int(tolerance_correct),
                'mae': float(mae),
                'mse': float(mse),
                'pearson_correlation': float(pearson_corr),
                'spearman_correlation': float(spearman_corr)
            })
        elif task in ['sst2', 'qqp'] and len(all_predictions) > 0 and len(all_labels) > 0:
            # Calculate F1 score for classification tasks
            pred_array = np.array(all_predictions)
            label_array = np.array(all_labels)
            
            # Calculate F1 score (weighted average for multi-class)
            f1 = f1_score(label_array, pred_array, average='weighted', zero_division=0)
            
            metrics.update({
                'f1_score': float(f1)
            })
        
        # Set model back to training mode
        self.model.train()
        
        return metrics

    def collect_device_metrics(self, round_num: int, task: str, phase: str, metrics: Optional[Dict] = None) -> Dict:
        """Collect current device usage metrics"""
        device_metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'client_id': self.client_id,
            'round': round_num,
            'task': task,
            'phase': phase,  # 'start', 'end', 'training', 'validation'
            'device_type': self.device.type,
        }
        
        # Add CUDA metrics if available
        if torch.cuda.is_available() and self.device.type == 'cuda':
            device_metrics.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'gpu_memory_utilization_pct': (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100,
            })
        else:
            device_metrics.update({
                'gpu_name': 'N/A',
                'gpu_memory_allocated_gb': 0.0,
                'gpu_memory_reserved_gb': 0.0,
                'gpu_memory_total_gb': 0.0,
                'gpu_memory_utilization_pct': 0.0,
            })
        
        # Add training metrics if provided
        if metrics:
            device_metrics.update({
                'loss': metrics.get('loss', 0.0),
                'accuracy': metrics.get('accuracy', 0.0),
                'samples_processed': metrics.get('samples_processed', 0),
            })
        else:
            device_metrics.update({
                'loss': 0.0,
                'accuracy': 0.0,
                'samples_processed': 0,
            })
        
        return device_metrics

    def record_device_usage(self, round_num: int, task: str, phase: str, metrics: Optional[Dict] = None):
        """Record device usage at a specific point"""
        device_metrics = self.collect_device_metrics(round_num, task, phase, metrics)
        self.device_usage_records.append(device_metrics)
        
        # Log the metrics
        if self.device.type == 'cuda':
            logger.info(f"[DEVICE-METRICS] Round {round_num} | Task {task} | Phase {phase} | "
                       f"GPU Memory: {device_metrics['gpu_memory_allocated_gb']:.2f}GB / "
                       f"{device_metrics['gpu_memory_total_gb']:.2f}GB "
                       f"({device_metrics['gpu_memory_utilization_pct']:.1f}%)")
        else:
            logger.info(f"[DEVICE-METRICS] Round {round_num} | Task {task} | Phase {phase} | Device: CPU")

    def save_device_usage_to_csv(self):
        """Append device usage records to CSV file (single file per client session)"""
        if not self.device_usage_records:
            logger.debug("No new device usage records to save")
            return
        
        try:
            # Get all unique keys from current records
            all_keys = set()
            for record in self.device_usage_records:
                all_keys.update(record.keys())
            
            fieldnames = sorted(list(all_keys))
            
            # Check if we need to write header (first time or file doesn't exist)
            write_header = not self.csv_initialized or not os.path.exists(self.csv_filename)
            
            # Append to the same CSV file
            mode = 'w' if write_header else 'a'
            with open(self.csv_filename, mode, newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if write_header:
                    writer.writeheader()
                    logger.info(f"[CSV-EXPORT] Created new device usage file: {self.csv_filename}")
                    self.csv_initialized = True
                
                writer.writerows(self.device_usage_records)
            
            logger.info(f"[CSV-EXPORT] Appended {len(self.device_usage_records)} records to: {self.csv_filename}")
            
            # Clear records after saving to avoid duplication
            self.device_usage_records.clear()
            
        except Exception as e:
            logger.error(f"Error saving device usage to CSV: {e}")
            import traceback
            logger.error(traceback.format_exc())

def run_client(client_id: str, tasks: List[str], config: FederatedConfig):
    """Run a federated learning client"""
    client = FederatedClient(client_id, tasks, config)
    asyncio.run(client.run_client())

if __name__ == "__main__":
    import argparse
    from federated_config import create_argument_parser, load_config

    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate arguments for client mode
    if args.mode != "client":
        parser.error("This script is for client mode only.")

    if not args.client_id:
        parser.error("Client ID is required for client mode.")

    if not args.tasks:
        parser.error("Tasks are required for client mode.")

    config = load_config(args)
    config.print_summary()

    run_client(args.client_id, args.tasks, config)
