#!/usr/bin/env python3
"""
WebSocket Communication Layer
Real-time bidirectional communication for federated learning
"""

import asyncio
import json
import logging
import websockets
import torch
from typing import Dict, List, Any, Callable
import time

logger = logging.getLogger(__name__)

class WebSocketServer:
    """WebSocket server for federated learning"""

    def __init__(self, port: int = 8771):
        self.port = port
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.server = None
        self.message_handlers: Dict[str, Callable] = {}

    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message types"""
        self.message_handlers[message_type] = handler

    async def start_server(self, message_handler):
        """Start WebSocket server"""
        self.server = await websockets.serve(
            message_handler,
            "localhost",
            self.port,
            ping_interval=None,  # Disable ping to prevent timeouts
            ping_timeout=None,   # Disable ping timeout
            close_timeout=None,  # Disable close timeout
            max_size=1024 * 1024 * 1024  # 1GB for large BERT-Medium updates
        )
        logger.info(f"WebSocket server started on port {self.port}")

    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.clients:
            logger.warning("No clients connected for broadcast")
            return

        # Serialize tensors before broadcasting
        serialized_message = MessageProtocol.serialize_tensors(message)
        message_str = json.dumps(serialized_message)
        disconnected_clients = []

        for client_id, client in self.clients.items():
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Client {client_id} disconnected during broadcast")
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            del self.clients[client_id]

    async def send_to_client(self, client_id: str, message: Dict):
        """Send message to specific client"""
        if client_id not in self.clients:
            logger.warning(f"Client {client_id} not found")
            return False

        try:
            # Serialize tensors before sending
            serialized_message = MessageProtocol.serialize_tensors(message)
            message_str = json.dumps(serialized_message)
            await self.clients[client_id].send(message_str)
            return True
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")
            del self.clients[client_id]
            return False

    def get_connected_clients(self) -> List[str]:
        """Get list of connected client IDs"""
        return list(self.clients.keys())

    def get_server_stats(self) -> Dict:
        """Get server statistics"""
        return {
            'port': self.port,
            'connected_clients': len(self.clients),
            'registered_handlers': len(self.message_handlers),
            'uptime': time.time()  # Would need to track actual uptime
        }

class WebSocketClient:
    """WebSocket client for federated learning"""

    def __init__(self, server_url: str, client_id: str, send_timeout: int = 3600):
        self.server_url = server_url
        self.client_id = client_id
        self.websocket = None
        self.message_handlers: Dict[str, Callable] = {}
        self.is_connected = False
        self.send_timeout = send_timeout  # Configurable send timeout

    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message types"""
        self.message_handlers[message_type] = handler

    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=None,  # Disable ping to prevent timeouts
                ping_timeout=None,   # Disable ping timeout
                close_timeout=None,  # Disable close timeout
                max_size=1024 * 1024 * 1024  # 1GB for large BERT-Medium updates
            )
            self.is_connected = True
            logger.info(f"Client {self.client_id} connected to server at {self.server_url}")

            # Start message handling loop
            asyncio.create_task(self._message_loop())
            
            # Start connection health check
            asyncio.create_task(self._connection_health_check())

        except Exception as e:
            logger.error(f"Failed to connect client {self.client_id}: {e}")
            self.is_connected = False
            raise

    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info(f"Client {self.client_id} disconnected")

    async def send(self, message: Dict, max_retries: int = 3):
        """Send message to server with retry logic"""
        for attempt in range(max_retries):
            if not self.is_connected or not self.websocket:
                logger.warning(f"Client {self.client_id} not connected, attempting reconnection (attempt {attempt + 1}/{max_retries})")
                try:
                    await self.connect()
                except Exception as e:
                    logger.error(f"Reconnection failed: {e}")
                    if attempt == max_retries - 1:
                        logger.error(f"Client {self.client_id} failed to reconnect after {max_retries} attempts")
                        return False
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue

            try:
                # Serialize tensors before sending
                serialized_message = MessageProtocol.serialize_tensors(message)
                message_str = json.dumps(serialized_message)
                logger.info(f"Client {self.client_id} sending message of size {len(message_str)} characters")
                logger.info(f"Client {self.client_id} using send timeout of {self.send_timeout} seconds")
                
                # Send with timeout to prevent hanging (use configurable timeout)
                await asyncio.wait_for(self.websocket.send(message_str), timeout=self.send_timeout)
                logger.info(f"Message sent successfully from client {self.client_id}")
                return True
            except asyncio.TimeoutError:
                logger.error(f"Send timeout for client {self.client_id} after {self.send_timeout}s (message size: {len(message_str)} chars)")
                self.is_connected = False
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Connection closed during send for client {self.client_id}, attempting reconnection (attempt {attempt + 1}/{max_retries})")
                self.is_connected = False
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
            except json.JSONDecodeError as e:
                logger.error(f"JSON serialization error for client {self.client_id}: {e}")
                return False
            except Exception as e:
                logger.error(f"Error sending message from client {self.client_id}: {e}")
                self.is_connected = False
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
        
        logger.error(f"Failed to send message after {max_retries} attempts")
        return False

    async def _message_loop(self):
        """Handle incoming messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    # Deserialize tensors in received messages
                    data = MessageProtocol.deserialize_tensors(data, device="cpu")
                    message_type = data.get("type")

                    if message_type in self.message_handlers:
                        await self.message_handlers[message_type](data)
                    else:
                        logger.warning(f"Unknown message type received: {message_type}")

                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {self.client_id} connection closed by server")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error in message loop for client {self.client_id}: {e}")
            self.is_connected = False

    async def _connection_health_check(self):
        """Periodic connection health check"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not self.is_connected or not self.websocket:
                    logger.warning(f"Connection lost for client {self.client_id}, attempting reconnection...")
                    try:
                        await self.connect()
                        logger.info(f"Client {self.client_id} reconnected successfully")
                    except Exception as e:
                        logger.error(f"Failed to reconnect client {self.client_id}: {e}")
                else:
                    # Send a ping to keep connection alive
                    try:
                        await self.websocket.ping()
                    except Exception as e:
                        logger.warning(f"Ping failed for client {self.client_id}: {e}")
                        self.is_connected = False
                        
            except Exception as e:
                logger.error(f"Error in connection health check for client {self.client_id}: {e}")
                await asyncio.sleep(30)

class MessageProtocol:
    """Defines message protocols for federated learning"""

    # Client registration
    CLIENT_REGISTER = "register"

    # Training messages
    TRAINING_REQUEST = "training_request"
    CLIENT_UPDATE = "client_update"
    GLOBAL_MODEL_SYNC = "global_model_sync"
    SYNC_ACKNOWLEDGMENT = "sync_acknowledgment"

    # Knowledge distillation messages
    TEACHER_KNOWLEDGE = "teacher_knowledge"
    STUDENT_KNOWLEDGE = "student_knowledge"

    # System messages
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    DISCONNECT = "disconnect"

    @staticmethod
    def serialize_tensors(data: Dict) -> Dict:
        """Convert tensors to JSON-serializable format"""
        if isinstance(data, dict):
            return {k: MessageProtocol.serialize_tensors(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [MessageProtocol.serialize_tensors(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return {
                "__tensor__": True,
                "data": data.detach().cpu().tolist(),
                "shape": list(data.shape),
                "dtype": str(data.dtype)
            }
        else:
            return data

    @staticmethod
    def deserialize_tensors(data: Dict, device: str = "cpu") -> Dict:
        """Convert serialized tensors back to torch tensors"""
        if isinstance(data, dict):
            if "__tensor__" in data:
                # Reconstruct tensor
                dtype_str = data["dtype"]
                # Handle different dtype formats
                if dtype_str.startswith("torch."):
                    dtype_str = dtype_str[6:]  # Remove "torch." prefix
                
                # Map string to torch dtype
                dtype_map = {
                    "float32": torch.float32,
                    "float64": torch.float64,
                    "int32": torch.int32,
                    "int64": torch.int64,
                    "long": torch.long,
                    "float": torch.float,
                    "double": torch.double,
                    "int": torch.int
                }
                
                dtype = dtype_map.get(dtype_str, torch.float32)
                tensor_data = torch.tensor(data["data"], dtype=dtype)
                tensor_data = tensor_data.reshape(data["shape"])
                return tensor_data.to(device)
            else:
                return {k: MessageProtocol.deserialize_tensors(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [MessageProtocol.deserialize_tensors(item, device) for item in data]
        else:
            return data

    @staticmethod
    def create_registration_message(client_id: str, tasks: List[str], capabilities: Dict) -> Dict:
        """Create client registration message"""
        return {
            "type": MessageProtocol.CLIENT_REGISTER,
            "client_id": client_id,
            "tasks": tasks,
            "capabilities": capabilities,
            "timestamp": time.time()
        }

    @staticmethod
    def create_training_request_message(round_num: int, global_params: Dict, teacher_knowledge: Dict) -> Dict:
        """Create training request message"""
        return {
            "type": MessageProtocol.TRAINING_REQUEST,
            "round": round_num,
            "global_params": global_params,
            "teacher_knowledge": teacher_knowledge,
            "timestamp": time.time()
        }

    @staticmethod
    def create_client_update_message(client_id: str, round_num: int, lora_updates: Dict, metrics: Dict, student_knowledge: Dict = None) -> Dict:
        """Create client update message"""
        message = {
            "type": MessageProtocol.CLIENT_UPDATE,
            "client_id": client_id,
            "round": round_num,
            "lora_updates": lora_updates,
            "metrics": metrics,
            "timestamp": time.time()
        }
        
        if student_knowledge is not None:
            message["student_knowledge"] = student_knowledge
            
        return message

    @staticmethod
    def create_global_model_sync_message(global_model_state: Dict) -> Dict:
        """Create global model synchronization message"""
        return {
            "type": MessageProtocol.GLOBAL_MODEL_SYNC,
            "global_model_state": global_model_state,
            "timestamp": time.time()
        }

    @staticmethod
    def create_sync_acknowledgment_message(client_id: str, synchronized: bool) -> Dict:
        """Create synchronization acknowledgment message"""
        return {
            "type": MessageProtocol.SYNC_ACKNOWLEDGMENT,
            "client_id": client_id,
            "synchronized": synchronized,
            "timestamp": time.time()
        }

    @staticmethod
    def create_heartbeat_message(client_id: str) -> Dict:
        """Create heartbeat message"""
        return {
            "type": MessageProtocol.HEARTBEAT,
            "client_id": client_id,
            "timestamp": time.time()
        }

class CommunicationManager:
    """Manages WebSocket communication between server and clients"""

    def __init__(self, config):
        self.config = config
        self.websocket_server = WebSocketServer(config.port)
        self.message_protocol = MessageProtocol()

        # Setup message handlers
        self.setup_message_handlers()

    def setup_message_handlers(self):
        """Setup default message handlers"""
        # These would be implemented by the server/client classes
        pass

    def get_communication_stats(self) -> Dict:
        """Get communication statistics"""
        return {
            'server_stats': self.websocket_server.get_server_stats(),
            'protocol_version': '1.0',
            'supported_message_types': list(MessageProtocol.__dict__.keys())
        }
