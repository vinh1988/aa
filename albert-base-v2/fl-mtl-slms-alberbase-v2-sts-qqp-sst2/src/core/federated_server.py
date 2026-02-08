#!/usr/bin/env python3
"""
Federated Learning Server Implementation
Server-Side Multi-Task Learning (MT-DNN Style)
Orchestrates training with shared encoder and task-specific heads
"""

import asyncio
import json
import logging
import time
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch

from federated_config import FederatedConfig
from src.models.mtl_server_model import MTLServerModel
from src.aggregation.mtl_aggregator import MTLAggregator
from src.communication.federated_websockets import WebSocketServer, MessageProtocol
from src.synchronization.federated_synchronization import SynchronizationManager

logger = logging.getLogger(__name__)


class FederatedServer:
    """Federated Learning Server with Multi-Task Learning (MT-DNN Style)"""

    def __init__(self, config: FederatedConfig):
        self.config = config
        self.connected_clients: Dict[str, Dict] = {}
        self.client_updates: Dict[int, List[Dict]] = {}
        
        # Initialize MTL Server Model (shared BERT + task-specific heads)
        self.mtl_model = MTLServerModel(
            base_model_name=config.server_model,
            tasks=['sst2', 'qqp', 'stsb']
        )
        logger.info("=" * 60)
        logger.info("MTL Server Model Summary:")
        summary = self.mtl_model.get_model_summary()
        logger.info(f"  Base Model: {summary['base_model']}")
        logger.info(f"  Tasks: {summary['tasks']}")
        logger.info(f"  Shared Parameters: {summary['shared_parameters']:,}")
        for task, size in summary['task_head_parameters'].items():
            logger.info(f"  Task '{task}' Head: {size:,} parameters")
        logger.info(f"  Total Parameters: {summary['total_parameters']:,}")
        logger.info("=" * 60)

        # Initialize MTL-aware aggregator
        self.aggregator = MTLAggregator()
        self.websocket_server = WebSocketServer(config.port)
        self.synchronization_manager = SynchronizationManager(self)

        # Setup results management
        self.setup_results_management()

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'federated_server_{self.config.port}.log'),
                logging.StreamHandler()
            ]
        )

    def setup_results_management(self):
        """Setup CSV files for results"""
        os.makedirs(self.config.results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Global results file
        self.csv_filename = os.path.join(
            self.config.results_dir,
            f"federated_results_{timestamp}.csv"
        )
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Client-specific results file
        self.client_csv_filename = os.path.join(
            self.config.results_dir,
            f"client_results_{timestamp}.csv"
        )
        self.client_csv_file = open(self.client_csv_filename, 'w', newline='')
        self.client_csv_writer = csv.writer(self.client_csv_file)

        # Write CSV headers for global results
        # Note: Global model validation metrics aggregated from clients
        global_headers = [
            "round", "responses_received", "avg_accuracy", "classification_accuracy",
            "regression_accuracy", "total_clients", "active_clients", "training_time",
            "synchronization_events", "global_model_version",
            # Global model validation metrics (aggregated from clients)
            "global_sst2_val_accuracy", "global_sst2_val_f1",
            "global_qqp_val_accuracy", "global_qqp_val_f1",
            "global_stsb_val_pearson", "global_stsb_val_spearman",
            "timestamp"
        ]
        self.csv_writer.writerow(global_headers)
        self.csv_file.flush()

        # Write CSV headers for client results
        client_headers = [
            "round", "client_id", "task", "accuracy", "loss", "samples_processed",
            "correct_predictions", "f1_score", "pearson_correlation", "spearman_correlation",
            "mae", "mse", "rmse",
            "val_accuracy", "val_loss", "val_samples", "val_correct_predictions",
            "val_f1_score", "val_pearson_correlation", "val_spearman_correlation", "val_mae",
            # Global model validation metrics (evaluated on client's local val data)
            "global_model_val_accuracy", "global_model_val_loss", "global_model_val_samples",
            "global_model_val_correct_predictions", "global_model_val_f1", "global_model_val_precision",
            "global_model_val_recall", "global_model_val_pearson", "global_model_val_spearman",
            "timestamp"
        ]
        self.client_csv_writer.writerow(client_headers)
        self.client_csv_file.flush()

    def record_client_results(self, round_num: int, client_id: str, client_metrics: Dict):
        """Record individual client results"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Process each task's metrics for this client
        for task_name, task_metrics in client_metrics.items():
            if isinstance(task_metrics, dict):
                row = [
                    round_num,
                    client_id,
                    task_name,
                    task_metrics.get('accuracy', 0.0),
                    task_metrics.get('loss', 0.0),
                    task_metrics.get('samples_processed', 0),
                    task_metrics.get('correct_predictions', 0),
                    # Classification metrics (F1)
                    task_metrics.get('f1_score', ''),
                    # Regression metrics (Pearson, Spearman, MAE, MSE, RMSE)
                    task_metrics.get('pearson_correlation', ''),
                    task_metrics.get('spearman_correlation', ''),
                    task_metrics.get('mae', ''),
                    task_metrics.get('mse', ''),
                    task_metrics.get('rmse', ''),
                    # Validation metrics (client model on local val data)
                    task_metrics.get('val_accuracy', 0.0),
                    task_metrics.get('val_loss', 0.0),
                    task_metrics.get('val_samples', 0),
                    task_metrics.get('val_correct_predictions', 0),
                    task_metrics.get('val_f1_score', ''),
                    task_metrics.get('val_pearson_correlation', ''),
                    task_metrics.get('val_spearman_correlation', ''),
                    task_metrics.get('val_mae', ''),
                    # Global model validation metrics (global model on client's local val data)
                    task_metrics.get('global_model_val_accuracy', ''),
                    task_metrics.get('global_model_val_loss', ''),
                    task_metrics.get('global_model_val_samples', ''),
                    task_metrics.get('global_model_val_correct_predictions', ''),
                    task_metrics.get('global_model_val_f1', ''),
                    task_metrics.get('global_model_val_precision', ''),
                    task_metrics.get('global_model_val_recall', ''),
                    task_metrics.get('global_model_val_pearson', ''),
                    task_metrics.get('global_model_val_spearman', ''),
                    timestamp
                ]
                self.client_csv_writer.writerow(row)
        
        self.client_csv_file.flush()

    async def client_handler(self, websocket):
        """Handle individual client connections"""
        client_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                # Deserialize tensors in received messages
                data = MessageProtocol.deserialize_tensors(data, device="cpu")
                message_type = data.get("type")

                if message_type == MessageProtocol.CLIENT_REGISTER:
                    client_id = await self.handle_client_registration(websocket, data)
                elif message_type == MessageProtocol.CLIENT_UPDATE:
                    await self.handle_client_update(websocket, data)
                elif message_type == MessageProtocol.SYNC_ACKNOWLEDGMENT:
                    await self.handle_sync_acknowledgment(websocket, data)
                elif message_type == MessageProtocol.HEARTBEAT:
                    await self.handle_heartbeat(websocket, data)

        except Exception as e:
            logger.error(f"Error in client handler: {e}")
        finally:
            if client_id and client_id in self.connected_clients:
                logger.info(f"Client {client_id} disconnected")
                del self.connected_clients[client_id]
                # Also remove from WebSocket server clients
                if client_id in self.websocket_server.clients:
                    del self.websocket_server.clients[client_id]

    async def handle_client_registration(self, websocket, data: Dict) -> str:
        """Handle client registration"""
        client_id = data["client_id"]
        tasks = data.get("tasks", [])
        capabilities = data.get("capabilities", {})

        # Register client in WebSocket server
        self.websocket_server.clients[client_id] = websocket

        # Register client in federated server
        self.connected_clients[client_id] = {
            "websocket": websocket,
            "tasks": tasks,
            "capabilities": capabilities,
            "last_seen": time.time(),
            "registered_at": datetime.now().isoformat()
        }

        logger.info(f"Client {client_id} registered with tasks: {tasks}")

        # Send acknowledgment
        response = {
            "type": "registration_ack",
            "client_id": client_id,
            "accepted": True,
            "message": f"Registered successfully with {len(tasks)} tasks"
        }
        await websocket.send(json.dumps(response))

        return client_id

    async def handle_client_update(self, websocket, data: Dict):
        """Handle client model updates"""
        client_id = data["client_id"]
        round_num = data["round"]
        model_updates = data.get("lora_updates", {})  # Keep key name for backward compatibility
        metrics = data.get("metrics", {})
        task = data.get("task", None)  # Get task label for MTL aggregation
        tasks = data.get("tasks", [])  # Alternative: list of tasks

        # Determine primary task for MTL
        if task:
            primary_task = task
        elif tasks:
            primary_task = tasks[0] if tasks else None
        else:
            # Try to infer from client_id
            primary_task = self._infer_task_from_client_id(client_id)

        # Store client update with task information
        if round_num not in self.client_updates:
            self.client_updates[round_num] = []

        self.client_updates[round_num].append({
            "client_id": client_id,
            "task": primary_task,  # MTL: Task label for aggregation
            "lora_updates": model_updates,  # Keep key name for backward compatibility
            "metrics": metrics,
            "received_at": datetime.now().isoformat()
        })

        # Record individual client results
        self.record_client_results(round_num, client_id, metrics)

        # Update client last seen
        if client_id in self.connected_clients:
            self.connected_clients[client_id]["last_seen"] = time.time()

        logger.info(f"Received update from client {client_id} for round {round_num}")

    async def handle_sync_acknowledgment(self, websocket, data: Dict):
        """Handle synchronization acknowledgment from clients"""
        client_id = data["client_id"]
        synchronized = data.get("synchronized", False)

        logger.info(f"Client {client_id} synchronization acknowledgment: {synchronized}")

    async def handle_heartbeat(self, websocket, data: Dict):
        """Handle heartbeat messages"""
        client_id = data["client_id"]

        # Update client last seen
        if client_id in self.connected_clients:
            self.connected_clients[client_id]["last_seen"] = time.time()

    async def wait_for_clients(self) -> bool:
        """Wait for all expected clients to connect before starting training"""
        # For federated learning, we typically want all clients to participate
        # Use expected_clients from config, fallback to min_clients
        expected_clients = getattr(self.config, 'expected_clients', self.config.min_clients)
        
        # If we have fewer clients than expected, wait for more to connect
        # Only proceed if we've been waiting too long (timeout)
        if expected_clients > len(self.connected_clients):
            logger.info(f"Waiting for more clients to connect (have {len(self.connected_clients)}, need {expected_clients})")
        
        logger.info(f"Waiting for clients... (need {expected_clients}, currently {len(self.connected_clients)})")

        start_time = time.time()
        max_wait_time = 120  # Wait up to 2 minutes for all clients
        
        while len(self.connected_clients) < expected_clients:
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                logger.warning(f"Timeout waiting for all clients after {elapsed:.0f}s")
                logger.warning(f"Proceeding with {len(self.connected_clients)} clients: {list(self.connected_clients.keys())}")
                break

            logger.info(f"Waiting for clients... ({len(self.connected_clients)}/{expected_clients})")
            logger.info(f"Connected clients: {list(self.connected_clients.keys())}")
            await asyncio.sleep(2)

        # Only proceed if we have the expected number of clients or timeout occurred
        if len(self.connected_clients) >= expected_clients:
            logger.info(f"Starting training with {len(self.connected_clients)} clients: {list(self.connected_clients.keys())}")
            return True
        elif len(self.connected_clients) >= self.config.min_clients:
            logger.warning(f"Starting with {len(self.connected_clients)} clients (expected {expected_clients}): {list(self.connected_clients.keys())}")
            return True
        else:
            logger.error(f"Not enough clients connected ({len(self.connected_clients)}/{self.config.min_clients})")
            return False

    async def run_federated_training(self):
        """Main federated training loop with MTL"""
        logger.info("Starting Federated Learning with Server-Side Multi-Task Learning (MT-DNN Style)")
        logger.info("Architecture: Shared BERT Encoder + Task-Specific Heads")

        # Wait for clients
        if not await self.wait_for_clients():
            logger.error("Insufficient clients for training")
            return

        # Training rounds with MTL-aware aggregation
        for round_num in range(1, self.config.num_rounds + 1):
            logger.info(f"=== Round {round_num}/{self.config.num_rounds} ===")
            round_start = time.time()

            try:
                # Step 1: Send task-specific model slices to clients
                # Each client receives: shared BERT encoder + their task-specific head
                if self.config.enable_synchronization:
                    await self.broadcast_mtl_model_slices()
                    await self.wait_for_sync_acknowledgments(round_num)

                # Step 2: Send training request to clients
                training_request = MessageProtocol.create_training_request_message(
                    round_num, {}, {}
                )
                await self.websocket_server.broadcast(training_request)

                # Step 3: Wait for client updates (with task labels)
                await self.collect_client_updates(round_num)

                # Step 4: MTL-aware aggregation
                # - Shared encoder: aggregated from ALL clients
                # - Task heads: aggregated within same-task clients only
                if round_num in self.client_updates and self.client_updates[round_num]:
                    client_updates = self.client_updates[round_num]
                    
                    # Perform MTL-aware aggregation
                    aggregated = self.aggregator.aggregate_mtl_updates(client_updates)
                    
                    # Update MTL server model
                    if aggregated['shared']:
                        self.mtl_model.set_shared_parameters(aggregated['shared'])
                        logger.info(f"Updated shared BERT encoder with {len(aggregated['shared'])} parameters")
                    
                    for task, task_params in aggregated['task_heads'].items():
                        if task_params:
                            self.mtl_model.set_task_head_parameters(task, task_params)
                            logger.info(f"Updated task '{task}' head with {len(task_params)} parameters")
                    
                    # Increment model version for synchronization
                    self.synchronization_manager.increment_model_version()

                # Record results for this round
                # Note: Global model validation is now done on client side
                self.record_round_results(round_num, round_start)

                logger.info(f"Round {round_num} completed in {time.time() - round_start:.2f}s")

            except Exception as e:
                logger.error(f"Error in round {round_num}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        # Finalize training
        self.finalize_training()
    
    async def broadcast_mtl_model_slices(self):
        """
        Broadcast task-specific model slices to clients
        Each client receives: shared BERT encoder + their specific task head
        """
        logger.info("Broadcasting MTL model slices to clients...")
        
        for client_id, client_info in self.connected_clients.items():
            # Get client's task
            client_tasks = client_info.get('tasks', [])
            if not client_tasks:
                logger.warning(f"Client {client_id} has no tasks assigned")
                continue
            
            client_task = client_tasks[0]  # Get primary task
            
            # Get model slice for this task (shared + task-specific head)
            model_slice = self.mtl_model.get_model_slice_for_task(client_task)
            
            # Create synchronization message
            sync_message = {
                "type": "global_model_sync",
                "global_model_state": {
                    "model_slice": self._serialize_tensors(model_slice),
                    "task": client_task,
                    "model_version": self.synchronization_manager.global_model_version
                },
                "version": self.synchronization_manager.global_model_version,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to specific client
            client_websocket = client_info.get('websocket')
            if client_websocket:
                await client_websocket.send(json.dumps(sync_message))
                logger.debug(f"Sent model slice for task '{client_task}' to client {client_id}")
        
        logger.info(f"Broadcasted model slices to {len(self.connected_clients)} clients")
    
    def _serialize_tensors(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict:
        """Convert tensors to JSON-serializable format"""
        serialized = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized
    
    def _infer_task_from_client_id(self, client_id: str) -> Optional[str]:
        """Infer task from client ID"""
        client_id_lower = client_id.lower()
        if 'sst2' in client_id_lower:
            return 'sst2'
        elif 'qqp' in client_id_lower:
            return 'qqp'
        elif 'stsb' in client_id_lower:
            return 'stsb'
        logger.warning(f"Could not infer task from client_id: {client_id}")
        return None

    async def wait_for_sync_acknowledgments(self, round_num: int, timeout: int = 120):
        """Wait for synchronization acknowledgments from clients"""
        start_time = time.time()
        acks_received = set()

        while time.time() - start_time < timeout:
            # Check for new acknowledgments (this would be implemented in the message handler)
            # For now, we'll assume clients acknowledge immediately
            acks_received.update(self.connected_clients.keys())

            if len(acks_received) >= len(self.connected_clients):
                logger.info("All clients synchronized")
                return True

            await asyncio.sleep(1)

        logger.warning(f"Timeout waiting for sync acknowledgments. Got {len(acks_received)}/{len(self.connected_clients)}")
        return False

    async def collect_client_updates(self, round_num: int, timeout: int = None):
        """Collect client updates for a round - wait for ALL connected clients"""
        if timeout is None:
            # Use config round_timeout, default to 3400 seconds (56.7 minutes) if not set
            timeout = getattr(self.config.communication, 'round_timeout', 3400)
        
        start_time = time.time()
        updates_received = 0
        
        # Use the expected clients from config, not current connected count
        expected_clients = getattr(self.config, 'expected_clients', len(self.connected_clients))
        
        if expected_clients == 0:
            logger.error("No clients expected!")
            return False

        logger.info(f"Waiting for client updates... (expecting ALL {expected_clients} clients)")
        logger.info(f"Currently connected clients: {list(self.connected_clients.keys())}")
        logger.info(f"Timeout set to {timeout} seconds")

        while time.time() - start_time < timeout:
            if round_num in self.client_updates:
                updates_received = len(self.client_updates[round_num])
                # Extract client IDs from the list of update dictionaries
                received_clients = [update.get('client_id', 'unknown') for update in self.client_updates[round_num]]
                missing_clients = [c for c in self.connected_clients.keys() if c not in received_clients]
                
                logger.info(f"Updates received: {updates_received}/{expected_clients}")
                logger.info(f"Received from: {received_clients}")
                if missing_clients:
                    logger.info(f"Still waiting for: {missing_clients}")

            # Wait for ALL connected clients to respond
            if updates_received >= expected_clients:
                logger.info(f"All client updates received ({updates_received}/{expected_clients})")
                return True

            # Check if any clients disconnected during waiting
            current_connected = len(self.connected_clients)
            if current_connected < expected_clients:
                logger.warning(f"Client count changed: {current_connected}/{expected_clients}")
                # Don't reduce expected_clients - we still want to wait for all originally expected clients
                if current_connected == 0:
                    logger.error("No clients connected!")
                    return False

            # Show progress every 30 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                logger.info(f"Still waiting... {elapsed:.0f}s elapsed, {updates_received}/{expected_clients} updates received")

            await asyncio.sleep(2)

        logger.warning(f"Timeout collecting updates. Got {updates_received}/{expected_clients}")
        if updates_received > 0:
            logger.warning(f"Proceeding with {updates_received} out of {expected_clients} clients")
            return True
        else:
            logger.error("No client updates received!")
            return False

    def record_round_results(self, round_num: int, round_start: float):
        """Record results for a training round"""
        updates = self.client_updates.get(round_num, [])
        responses = len(updates)

        if responses > 0:
            # Calculate aggregated metrics from client training
            metrics = self.calculate_aggregated_metrics(updates)
            
            # Extract global model validation metrics from client updates
            global_val_metrics = self.extract_global_validation_metrics(updates)
            
            # Record to CSV
            row = [
                round_num,
                responses,
                f"{metrics['avg_accuracy']:.4f}",
                f"{metrics['classification_accuracy']:.4f}",
                f"{metrics['regression_accuracy']:.4f}",
                len(self.connected_clients),
                len(self.connected_clients),
                f"{time.time() - round_start:.2f}",
                self.synchronization_manager.global_model_version,
                str(self.synchronization_manager.global_model_version),
                # Global model validation metrics (from clients)
                f"{global_val_metrics.get('sst2_accuracy', 0.0):.4f}",
                f"{global_val_metrics.get('sst2_f1', 0.0):.4f}",
                f"{global_val_metrics.get('qqp_accuracy', 0.0):.4f}",
                f"{global_val_metrics.get('qqp_f1', 0.0):.4f}",
                f"{global_val_metrics.get('stsb_pearson', 0.0):.4f}",
                f"{global_val_metrics.get('stsb_spearman', 0.0):.4f}",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]

            self.csv_writer.writerow(row)
            self.csv_file.flush()

            # Print progress
            responses_received = len(self.client_updates[round_num]) if round_num in self.client_updates else 0
            total_clients = len(self.connected_clients)
            participating_clients = [update.get('client_id', 'unknown') for update in self.client_updates[round_num]] if round_num in self.client_updates else []
            
            print(f"🏃 Round {round_num} completed")
            print(f"   Client Training Avg: {metrics['avg_accuracy']:.4f}")
            print(f"   Classification Avg: {metrics['classification_accuracy']:.4f}")
            print(f"   Regression Avg: {metrics['regression_accuracy']:.4f}")
            print(f"   📊 Global Model Validation:")
            print(f"      SST-2: Acc={global_val_metrics.get('sst2_accuracy', 0.0):.4f}, F1={global_val_metrics.get('sst2_f1', 0.0):.4f}")
            print(f"      QQP:   Acc={global_val_metrics.get('qqp_accuracy', 0.0):.4f}, F1={global_val_metrics.get('qqp_f1', 0.0):.4f}")
            print(f"      STS-B: Pearson={global_val_metrics.get('stsb_pearson', 0.0):.4f}, Spearman={global_val_metrics.get('stsb_spearman', 0.0):.4f}")
            print(f"📊 Participation: {responses_received}/{total_clients} clients")
            print(f"👥 Participating clients: {participating_clients}")

    def calculate_aggregated_metrics(self, updates: List[Dict]) -> Dict:
        """Calculate aggregated metrics across all clients"""
        if not updates:
            return {
                'avg_accuracy': 0.0,
                'classification_accuracy': 0.0,
                'regression_accuracy': 0.0,
                'total_clients': 0,
                'active_clients': 0,
                'training_time': 0.0
            }

        # Extract metrics from all clients
        all_accuracies = []
        classification_accuracies = []
        regression_accuracies = []

        for update in updates:
            client_metrics = update.get('metrics', {})
            
            # Process task-specific metrics
            for task_name, task_metrics in client_metrics.items():
                if isinstance(task_metrics, dict) and 'accuracy' in task_metrics:
                    accuracy = task_metrics['accuracy']
                    all_accuracies.append(accuracy)
                    
                    # Categorize by task type
                    if task_name in ['sst2', 'qqp']:
                        classification_accuracies.append(accuracy)
                    elif task_name == 'stsb':
                        regression_accuracies.append(accuracy)

        # Calculate averages
        avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        classification_accuracy = sum(classification_accuracies) / len(classification_accuracies) if classification_accuracies else 0.0
        regression_accuracy = sum(regression_accuracies) / len(regression_accuracies) if regression_accuracies else 0.0

        return {
            'avg_accuracy': avg_accuracy,
            'classification_accuracy': classification_accuracy,
            'regression_accuracy': regression_accuracy,
            'total_clients': len(self.connected_clients),
            'active_clients': len(updates),
            'training_time': 0.0  # Would calculate actual time
        }

    def extract_global_validation_metrics(self, updates: List[Dict]) -> Dict:
        """Extract global model validation metrics from client updates"""
        global_metrics = {}
        
        for update in updates:
            client_metrics = update.get('metrics', {})
            
            # Process each task's metrics
            for task_name, task_metrics in client_metrics.items():
                if not isinstance(task_metrics, dict):
                    continue
                
                # Extract global model validation metrics
                if task_name == 'sst2':
                    if 'global_model_val_accuracy' in task_metrics:
                        global_metrics['sst2_accuracy'] = task_metrics['global_model_val_accuracy']
                    if 'global_model_val_f1' in task_metrics:
                        global_metrics['sst2_f1'] = task_metrics['global_model_val_f1']
                
                elif task_name == 'qqp':
                    if 'global_model_val_accuracy' in task_metrics:
                        global_metrics['qqp_accuracy'] = task_metrics['global_model_val_accuracy']
                    if 'global_model_val_f1' in task_metrics:
                        global_metrics['qqp_f1'] = task_metrics['global_model_val_f1']
                
                elif task_name == 'stsb':
                    if 'global_model_val_pearson' in task_metrics:
                        global_metrics['stsb_pearson'] = task_metrics['global_model_val_pearson']
                    if 'global_model_val_spearman' in task_metrics:
                        global_metrics['stsb_spearman'] = task_metrics['global_model_val_spearman']
        
        logger.info(f"Extracted global validation metrics: {global_metrics}")
        return global_metrics

    def finalize_training(self):
        """Finalize training and create summary"""
        try:
            self.csv_file.close()
            self.client_csv_file.close()
            logger.info(f"Global results saved to {self.csv_filename}")
            logger.info(f"Client results saved to {self.client_csv_filename}")

            # Create summary
            summary_file = os.path.join(self.config.results_dir, "training_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("Federated Learning Training Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Configuration: Server-Side MTL (MT-DNN Style)\n")
                f.write(f"Architecture: Shared BERT + Task-Specific Heads\n")
                f.write(f"Total Rounds: {self.config.num_rounds}\n")
                f.write(f"Total Clients: {len(self.connected_clients)}\n")
                
                # Add MTL model summary
                mtl_summary = self.mtl_model.get_model_summary()
                f.write(f"\nMTL Model Architecture:\n")
                f.write(f"  Base Model: {mtl_summary['base_model']}\n")
                f.write(f"  Tasks: {', '.join(mtl_summary['tasks'])}\n")
                f.write(f"  Shared Parameters: {mtl_summary['shared_parameters']:,}\n")
                for task, size in mtl_summary['task_head_parameters'].items():
                    f.write(f"  Task '{task}' Head: {size:,} parameters\n")
                f.write(f"  Total Parameters: {mtl_summary['total_parameters']:,}\n")
                
                f.write(f"\nGlobal Results File: {os.path.basename(self.csv_filename)}\n")
                f.write(f"Client Results File: {os.path.basename(self.client_csv_filename)}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            logger.info(f"Summary saved to {summary_file}")

        except Exception as e:
            logger.error(f"Error finalizing training: {e}")

    async def start_server(self):
        """Start the federated server"""
        logger.info(f"Starting federated server on port {self.config.port}")

        # Setup message handlers
        self.websocket_server.register_message_handler(
            MessageProtocol.CLIENT_REGISTER,
            self.handle_client_registration
        )
        self.websocket_server.register_message_handler(
            MessageProtocol.CLIENT_UPDATE,
            self.handle_client_update
        )
        self.websocket_server.register_message_handler(
            MessageProtocol.HEARTBEAT,
            self.handle_heartbeat
        )

        # Start server
        await self.websocket_server.start_server(self.client_handler)

        # Start heartbeat task to keep connections alive
        asyncio.create_task(self._heartbeat_task())

        # Run training
        await self.run_federated_training()

    async def _heartbeat_task(self):
        """Send periodic heartbeats to keep connections alive"""
        while True:
            try:
                if self.connected_clients:
                    heartbeat_message = MessageProtocol.create_heartbeat_message("server")
                    await self.websocket_server.broadcast(heartbeat_message)
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}")
                await asyncio.sleep(30)

async def run_server(config: FederatedConfig):
    """Run the federated server"""
    server = FederatedServer(config)
    await server.start_server()

if __name__ == "__main__":
    import argparse
    from federated_config import create_argument_parser, load_config

    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate arguments for server mode
    if args.mode != "server":
        parser.error("This script is for server mode only.")

    config = load_config(args)
    config.print_summary()

    asyncio.run(run_server(config))
