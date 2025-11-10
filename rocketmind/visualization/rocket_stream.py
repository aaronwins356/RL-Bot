"""
Rocket Stream - WebSocket streaming for live telemetry.
Enables real-time data streaming to dashboard and external clients.
"""

import asyncio
import json
from typing import Dict, Any, Set, Optional
from collections import deque
import time

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("⚠ websockets not available. Install with: pip install websockets")


class RocketStream:
    """
    WebSocket server for streaming live telemetry data.
    Allows multiple clients to connect and receive real-time updates.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.clients: Set = set()
        self.data_buffer = deque(maxlen=1000)
        self.is_running = False
    
    async def register(self, websocket):
        """Register new client."""
        self.clients.add(websocket)
        print(f"✓ Client connected: {websocket.remote_address}")
    
    async def unregister(self, websocket):
        """Unregister client."""
        self.clients.remove(websocket)
        print(f"✗ Client disconnected: {websocket.remote_address}")
    
    async def send_to_all(self, message: Dict[str, Any]):
        """
        Send message to all connected clients.
        
        Args:
            message: Dictionary to send (will be JSON serialized)
        """
        if self.clients:
            message_json = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_json) for client in self.clients],
                return_exceptions=True
            )
    
    async def handler(self, websocket, path):
        """Handle client connection."""
        await self.register(websocket)
        try:
            # Send initial buffer
            for data in self.data_buffer:
                await websocket.send(json.dumps(data))
            
            # Keep connection alive
            async for message in websocket:
                # Echo or handle client messages if needed
                pass
        finally:
            await self.unregister(websocket)
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """
        Broadcast update to all clients.
        
        Args:
            data: Data to broadcast
        """
        # Add timestamp
        data['timestamp'] = time.time()
        
        # Store in buffer
        self.data_buffer.append(data)
        
        # Send to all clients
        await self.send_to_all(data)
    
    def start(self):
        """Start WebSocket server (non-blocking)."""
        if not WEBSOCKETS_AVAILABLE:
            print("✗ WebSocket streaming not available")
            return
        
        print(f"🌐 Starting RocketStream server on ws://{self.host}:{self.port}")
        self.is_running = True
        
        async def run_server():
            async with websockets.serve(self.handler, self.host, self.port):
                await asyncio.Future()  # Run forever
        
        # Run in background
        loop = asyncio.get_event_loop()
        loop.create_task(run_server())
    
    def stop(self):
        """Stop WebSocket server."""
        self.is_running = False
        print("✓ RocketStream server stopped")


class TelemetryStreamer:
    """
    High-level telemetry streamer that integrates with training loop.
    Automatically streams metrics during training.
    """
    
    def __init__(
        self,
        stream: Optional[RocketStream] = None,
        update_interval: float = 0.1
    ):
        """
        Args:
            stream: RocketStream instance (creates new if None)
            update_interval: Minimum time between updates (seconds)
        """
        self.stream = stream or RocketStream()
        self.update_interval = update_interval
        self.last_update_time = 0.0
    
    async def stream_training_metrics(self, metrics: Dict[str, Any]):
        """
        Stream training metrics.
        
        Args:
            metrics: Training metrics dictionary
        """
        current_time = time.time()
        
        # Throttle updates
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Broadcast metrics
        await self.stream.broadcast_update({
            'type': 'training_metrics',
            'data': metrics
        })
    
    async def stream_game_state(self, state: Dict[str, Any]):
        """
        Stream game state.
        
        Args:
            state: Game state dictionary
        """
        await self.stream.broadcast_update({
            'type': 'game_state',
            'data': state
        })
    
    async def stream_episode_complete(self, episode_data: Dict[str, Any]):
        """
        Stream episode completion data.
        
        Args:
            episode_data: Episode statistics
        """
        await self.stream.broadcast_update({
            'type': 'episode_complete',
            'data': episode_data
        })


# Example client code
async def example_client():
    """Example client that connects to RocketStream."""
    if not WEBSOCKETS_AVAILABLE:
        return
    
    uri = "ws://localhost:8765"
    
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("✓ Connected to RocketStream")
        
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                print(f"Received: {data['type']} at {data['timestamp']}")
            except Exception as e:
                print(f"Error: {e}")
                break


# Discord webhook integration (bonus feature)
class DiscordNotifier:
    """Send notifications to Discord via webhook."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Args:
            webhook_url: Discord webhook URL
        """
        self.webhook_url = webhook_url
    
    async def send_notification(self, message: str, embed: Optional[Dict] = None):
        """
        Send notification to Discord.
        
        Args:
            message: Message text
            embed: Optional Discord embed
        """
        if not self.webhook_url:
            return
        
        try:
            import aiohttp
            
            data = {'content': message}
            if embed:
                data['embeds'] = [embed]
            
            async with aiohttp.ClientSession() as session:
                await session.post(self.webhook_url, json=data)
        except Exception as e:
            print(f"Failed to send Discord notification: {e}")
    
    async def notify_training_milestone(
        self,
        timesteps: int,
        reward: float,
        checkpoint_path: str
    ):
        """Notify about training milestone."""
        embed = {
            'title': '🚀 Training Milestone Reached',
            'color': 0x667eea,
            'fields': [
                {'name': 'Timesteps', 'value': f'{timesteps:,}', 'inline': True},
                {'name': 'Mean Reward', 'value': f'{reward:.2f}', 'inline': True},
                {'name': 'Checkpoint', 'value': checkpoint_path, 'inline': False}
            ]
        }
        
        await self.send_notification('Training milestone reached!', embed)
