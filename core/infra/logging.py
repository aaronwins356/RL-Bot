"""Logging infrastructure for training and inference.

This module provides structured logging with TensorBoard integration.
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that safely handles NumPy types and other non-serializable objects.
    
    This encoder prevents 'Object of type int32 is not JSON serializable' errors
    by converting NumPy types to native Python types.
    """
    
    def default(self, obj):
        """Convert object to JSON-serializable format.
        
        Args:
            obj: Object to encode
            
        Returns:
            JSON-serializable representation
        """
        # Handle NumPy integers
        if isinstance(obj, np.integer):
            return int(obj)
        # Handle NumPy floats
        if isinstance(obj, np.floating):
            return float(obj)
        # Handle NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle NumPy bool
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Fallback to default encoder
        return super().default(obj)


def safe_log(logger: logging.Logger, level: int, msg: str):
    """Safely log message, handling Unicode/encoding errors.
    
    This function prevents UnicodeEncodeError on Windows terminals by
    falling back to ASCII encoding when needed.
    
    Args:
        logger: Logger instance
        level: Logging level (logging.INFO, logging.DEBUG, etc.)
        msg: Message to log
    """
    try:
        logger.log(level, msg)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe version with visible replacement
        safe_msg = msg.encode('ascii', 'replace').decode('ascii')
        logger.log(level, safe_msg)
        logger.debug("Unicode encoding fallback used for log message")


class MetricsLogger:
    """Logger for training metrics with TensorBoard support."""
    
    def __init__(
        self,
        log_dir: Path,
        use_tensorboard: bool = True,
        log_name: str = "training"
    ):
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory for logs
            use_tensorboard: Whether to use TensorBoard
            log_name: Name for log file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.log_dir))
            except ImportError:
                print("TensorBoard not available, skipping tensorboard logging")
                self.use_tensorboard = False
        
        # JSONL file for structured logs
        self.jsonl_path = self.log_dir / f"{log_name}.jsonl"
        
        # Metrics buffer
        self.metrics = {}
        self.step = 0
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log scalar value.
        
        Args:
            tag: Metric name
            value: Metric value
            step: Global step (if None, uses internal counter)
        """
        if step is None:
            step = self.step
        
        # TensorBoard
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        
        # Buffer for JSONL
        if step not in self.metrics:
            self.metrics[step] = {}
        self.metrics[step][tag] = value
    
    def log_dict(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log dictionary of metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Global step
        """
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)
    
    def flush(self):
        """Flush buffered metrics to JSONL file."""
        if not self.metrics:
            return
        
        try:
            with open(self.jsonl_path, "a", encoding='utf-8') as f:
                for step, metrics in self.metrics.items():
                    log_entry = {
                        "step": step,
                        "timestamp": datetime.now().isoformat(),
                        **metrics
                    }
                    # Use SafeJSONEncoder to handle NumPy types
                    f.write(json.dumps(log_entry, cls=SafeJSONEncoder) + "\n")
        except Exception as e:
            # Log error but don't crash training
            logging.getLogger(__name__).warning(f"Failed to flush metrics: {e}")
        
        self.metrics.clear()
        
        if self.writer:
            try:
                self.writer.flush()
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to flush TensorBoard writer: {e}")
    
    def increment_step(self):
        """Increment internal step counter."""
        self.step += 1
    
    def close(self):
        """Close logger and flush remaining metrics."""
        self.flush()
        if self.writer:
            self.writer.close()


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup standard Python logger.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
