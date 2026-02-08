#!/usr/bin/env python3
"""
MTL Model Synchronization Implementation
Manages task-specific model slice synchronization for federated multi-task learning
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
import torch
import logging

logger = logging.getLogger(__name__)

class SynchronizationManager:
    """Manages MTL model synchronization between server and clients"""

    def __init__(self, server):
        self.server = server
        self.synchronization_history = []
        self.global_model_version = 0

    async def broadcast_global_model(self, global_state: Dict):
        """
        Broadcast MTL model slices to clients
        Note: In MTL mode, use broadcast_mtl_model_slices() instead
        This is kept for backward compatibility
        """
        sync_message = {
            "type": "global_model_sync",
            "global_model_state": global_state,
            "version": self.global_model_version,
            "timestamp": datetime.now().isoformat()
        }

        # Send to all connected clients
        await self.server.websocket_server.broadcast(sync_message)

        # Record synchronization
        self.synchronization_history.append({
            "type": "broadcast",
            "timestamp": datetime.now(),
            "clients_notified": len(self.server.connected_clients),
            "version": self.global_model_version
        })

        logger.info(f"Broadcasted global model version {self.global_model_version} to {len(self.server.connected_clients)} clients")

    def increment_model_version(self):
        """Increment global model version"""
        self.global_model_version += 1
        return self.global_model_version

    def get_global_model_state(self) -> Dict:
        """Get current MTL model state for synchronization"""
        # Convert MTL model to JSON-serializable format
        serializable_params = {}
        
        if hasattr(self.server, 'mtl_model'):
            # MTL mode: serialize all parameters
            all_params = self.server.mtl_model.get_all_parameters()
            for param_name, param_value in all_params.items():
                if isinstance(param_value, torch.Tensor):
                    serializable_params[param_name] = param_value.tolist()
                else:
                    serializable_params[param_name] = param_value
        elif hasattr(self.server, 'global_parameters') and self.server.global_parameters:
            # Legacy mode: use global_parameters
            for param_name, param_value in self.server.global_parameters.items():
                if isinstance(param_value, torch.Tensor):
                    serializable_params[param_name] = param_value.tolist()
                else:
                    serializable_params[param_name] = param_value

        return {
            "global_parameters": serializable_params,
            "model_version": self.global_model_version,
            "aggregation_round": getattr(self.server, 'current_round', 0)
        }

    def get_model_slice_for_task(self, task: str) -> Dict:
        """
        Get MTL model slice for a specific task
        Returns: shared BERT encoder + task-specific head
        """
        if not hasattr(self.server, 'mtl_model'):
            logger.warning("MTL model not available, returning empty slice")
            return {}
        
        model_slice = self.server.mtl_model.get_model_slice_for_task(task)
        
        # Serialize tensors
        serializable_slice = {}
        for param_name, param_value in model_slice.items():
            if isinstance(param_value, torch.Tensor):
                serializable_slice[param_name] = param_value.tolist()
            else:
                serializable_slice[param_name] = param_value
        
        return {
            "model_slice": serializable_slice,
            "task": task,
            "model_version": self.global_model_version
        }

    def update_global_model_from_aggregation(self, aggregated_params: Dict):
        """
        Update MTL model with aggregated parameters
        In MTL mode, this is handled directly in server training loop
        """
        # In MTL mode, aggregation is handled in run_federated_training
        # This is kept for backward compatibility
        if not hasattr(self.server, 'mtl_model'):
            # Legacy mode
            if not hasattr(self.server, 'global_parameters'):
                self.server.global_parameters = {}
            self.server.global_parameters.update(aggregated_params)

        # Increment version
        self.increment_model_version()

        logger.info(f"Updated global model to version {self.global_model_version}")

    def get_synchronization_summary(self) -> Dict:
        """Get summary of synchronization history"""
        if not self.synchronization_history:
            return {"total_sync_events": 0}

        return {
            "total_sync_events": len(self.synchronization_history),
            "current_version": self.global_model_version,
            "average_clients_per_sync": sum(
                event["clients_notified"] for event in self.synchronization_history
            ) / len(self.synchronization_history),
            "last_sync_timestamp": self.synchronization_history[-1]["timestamp"].isoformat() if self.synchronization_history else None
        }

class ClientModelSynchronizer:
    """Handles client-side MTL model synchronization"""

    def __init__(self, local_model, websocket_client, task: str = None):
        self.local_model = local_model
        self.websocket_client = websocket_client
        self.task = task  # Task for MTL mode
        self.global_model_cache = None
        self.synchronization_log = []
        self.is_synchronized = False
        
        logger.info(f"ClientModelSynchronizer initialized for task: {task}")

    async def synchronize_with_global_model(self, global_state: Dict) -> Dict:
        """
        Update local model with MTL model slice from server
        Handles both MTL mode (model_slice) and legacy mode (global_parameters)
        """
        try:
            # Determine which format we received
            if "model_slice" in global_state:
                # MTL mode: task-specific model slice
                model_slice = global_state.get("model_slice", {})
                task = global_state.get("task")
                
                if task and self.task and task != self.task:
                    logger.warning(f"Received model slice for task '{task}' but expected '{self.task}'")
                
                await self.update_model_with_global_params(model_slice)
                params_updated = bool(model_slice)
            
            elif "global_parameters" in global_state:
                # Legacy mode or full model broadcast
                global_params = global_state.get("global_parameters", {})
                await self.update_model_with_global_params(global_params)
                params_updated = bool(global_params)
            
            else:
                logger.warning("No parameters found in global_state")
                params_updated = False

            # Update cache
            self.global_model_cache = global_state

            # Mark as synchronized
            self.is_synchronized = True

            # Log synchronization
            sync_record = {
                "timestamp": datetime.now(),
                "global_knowledge_integrated": True,
                "params_updated": params_updated,
                "model_version": global_state.get("model_version", 0),
                "task": self.task
            }
            self.synchronization_log.append(sync_record)

            logger.info(f"Client model synchronized with global model (task: {self.task})")

            return {
                "synchronized": True,
                "global_knowledge_integrated": True,
                "params_updated": params_updated,
                "model_version": global_state.get("model_version", 0)
            }

        except Exception as e:
            logger.error(f"Error during model synchronization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "synchronized": False,
                "error": str(e)
            }

    async def update_model_with_global_params(self, global_params: Dict):
        """Update local model parameters with global parameters"""
        # Convert lists back to tensors for model parameters
        tensor_params = {}
        for param_name, param_value in global_params.items():
            if isinstance(param_value, list):
                # Get device from the model's parameters
                try:
                    model_device = next(self.local_model.parameters()).device
                except StopIteration:
                    model_device = torch.device("cpu")
                tensor_params[param_name] = torch.tensor(param_value, device=model_device)
            else:
                # Ensure existing tensors are on the correct device
                if isinstance(param_value, torch.Tensor):
                    try:
                        model_device = next(self.local_model.parameters()).device
                    except StopIteration:
                        model_device = torch.device("cpu")
                    tensor_params[param_name] = param_value.to(model_device)
                else:
                    tensor_params[param_name] = param_value

        # Update local model with global parameters
        if hasattr(self.local_model, 'set_parameters'):
            self.local_model.set_parameters(tensor_params)
            logger.debug(f"Updated model with {len(tensor_params)} parameters (task: {self.task})")
        else:
            logger.warning("Local model does not have set_parameters method")

    async def send_synchronization_acknowledgment(self, sync_result: Dict):
        """Send synchronization acknowledgment to server"""
        ack_message = {
            "type": "sync_acknowledgment",
            "client_id": self.websocket_client.client_id,
            "synchronized": sync_result["synchronized"],
            "model_version": sync_result.get("model_version", 0),
            "task": self.task,
            "timestamp": datetime.now().isoformat()
        }

        await self.websocket_client.send(ack_message)

    def get_synchronization_status(self) -> Dict:
        """Get current synchronization status"""
        return {
            "is_synchronized": self.is_synchronized,
            "task": self.task,
            "global_model_version": self.global_model_cache.get("model_version", 0) if self.global_model_cache else 0,
            "last_sync_timestamp": self.synchronization_log[-1]["timestamp"].isoformat() if self.synchronization_log else None,
            "sync_events_count": len(self.synchronization_log),
            "global_knowledge_cached": bool(self.global_model_cache)
        }

class ModelStateManager:
    """Manages model state for synchronization"""

    def __init__(self):
        self.model_states = {}
        self.state_history = []

    def save_model_state(self, model_id: str, state: Dict):
        """Save model state for synchronization"""
        self.model_states[model_id] = state
        self.state_history.append({
            "model_id": model_id,
            "timestamp": datetime.now(),
            "state_size": len(str(state))
        })

    def get_model_state(self, model_id: str) -> Dict:
        """Get saved model state"""
        return self.model_states.get(model_id, {})

    def get_state_summary(self) -> Dict:
        """Get summary of saved states"""
        return {
            "total_states": len(self.model_states),
            "state_history_length": len(self.state_history),
            "latest_state_timestamp": self.state_history[-1]["timestamp"].isoformat() if self.state_history else None,
            "model_ids": list(self.model_states.keys())
        }

class SynchronizationProtocol:
    """Defines synchronization protocols and message formats"""

    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    SYNC_BROADCAST = "sync_broadcast"
    SYNC_ACKNOWLEDGMENT = "sync_acknowledgment"

    @staticmethod
    def create_sync_request(client_id: str, requested_version: int = None) -> Dict:
        """Create synchronization request message"""
        return {
            "type": SynchronizationProtocol.SYNC_REQUEST,
            "client_id": client_id,
            "requested_version": requested_version,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def create_sync_response(sync_data: Dict, version: int) -> Dict:
        """Create synchronization response message"""
        return {
            "type": SynchronizationProtocol.SYNC_RESPONSE,
            "sync_data": sync_data,
            "version": version,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def create_sync_broadcast(global_state: Dict, version: int) -> Dict:
        """Create synchronization broadcast message"""
        return {
            "type": SynchronizationProtocol.SYNC_BROADCAST,
            "global_state": global_state,
            "version": version,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def create_sync_acknowledgment(client_id: str, version: int, success: bool) -> Dict:
        """Create synchronization acknowledgment message"""
        return {
            "type": SynchronizationProtocol.SYNC_ACKNOWLEDGMENT,
            "client_id": client_id,
            "version": version,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
