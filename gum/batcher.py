import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class BatchedObservation:
    """Represents a batched observation waiting for processing."""
    id: str
    observer_name: str
    content: str
    content_type: str
    timestamp: datetime
    processed: bool = False

class ObservationBatcher:
    """Handles batching of observations to reduce API calls."""
    
    def __init__(self, data_directory: str, batch_interval_minutes: float = 2, max_batch_size: int = 50):
        self.data_directory = Path(data_directory)
        self.batch_interval_minutes = batch_interval_minutes
        self.max_batch_size = max_batch_size
        self.batch_file = self.data_directory / "batches" / "pending_observations.json"
        self.batch_file.parent.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("gum.batcher")
        self._pending_observations: List[BatchedObservation] = []
        
    async def start(self):
        """Start the batching system."""
        self._load_pending_observations()
        self.logger.info(f"Started batcher with {len(self._pending_observations)} pending observations")
        
    async def stop(self):
        """Stop the batching system."""
        self._save_pending_observations()
        self.logger.info("Stopped batcher")
        
    def add_observation(self, observer_name: str, content: str, content_type: str) -> str:
        """Add an observation to the batch queue.
        
        Args:
            observer_name: Name of the observer
            content: Observation content
            content_type: Type of content
            
        Returns:
            str: Observation ID
        """
        import uuid
        
        observation = BatchedObservation(
            id=str(uuid.uuid4()),
            observer_name=observer_name,
            content=content,
            content_type=content_type,
            timestamp=datetime.now(timezone.utc)
        )
        
        self._pending_observations.append(observation)
        self.logger.debug(f"Added observation {observation.id} to batch (total: {len(self._pending_observations)})")
        
        # Save immediately to prevent data loss
        self._save_pending_observations()
        
        return observation.id
        
    def get_pending_count(self) -> int:
        """Get the number of pending observations."""
        return len([obs for obs in self._pending_observations if not obs.processed])
        
    def get_batch(self, max_size: Optional[int] = None) -> List[BatchedObservation]:
        """Get a batch of unprocessed observations.
        
        Args:
            max_size: Maximum number of observations to return
            
        Returns:
            List of batched observations
        """
        unprocessed = [obs for obs in self._pending_observations if not obs.processed]
        max_size = max_size or self.max_batch_size
        return unprocessed[:max_size]
        
    def mark_processed(self, observation_ids: List[str]):
        """Mark observations as processed.
        
        Args:
            observation_ids: List of observation IDs to mark as processed
        """
        for obs in self._pending_observations:
            if obs.id in observation_ids:
                obs.processed = True
                
        # Remove processed observations older than 24 hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        self._pending_observations = [
            obs for obs in self._pending_observations 
            if not obs.processed or obs.timestamp > cutoff_time
        ]
        
        self._save_pending_observations()
        self.logger.debug(f"Marked {len(observation_ids)} observations as processed")
                        
    def _load_pending_observations(self):
        """Load pending observations from disk."""
        if self.batch_file.exists():
            try:
                with open(self.batch_file, 'r') as f:
                    data = json.load(f)
                    self._pending_observations = [
                        BatchedObservation(**obs_data) 
                        for obs_data in data
                    ]
                    # Convert timestamp strings back to datetime objects
                    for obs in self._pending_observations:
                        if isinstance(obs.timestamp, str):
                            obs.timestamp = datetime.fromisoformat(obs.timestamp.replace('Z', '+00:00'))
            except Exception as e:
                self.logger.error(f"Error loading pending observations: {e}")
                self._pending_observations = []
        else:
            self._pending_observations = []
            
    def _save_pending_observations(self):
        """Save pending observations to disk."""
        try:
            # Convert datetime objects to ISO format strings
            data = []
            for obs in self._pending_observations:
                obs_dict = asdict(obs)
                obs_dict['timestamp'] = obs.timestamp.isoformat()
                data.append(obs_dict)
                
            with open(self.batch_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving pending observations: {e}") 