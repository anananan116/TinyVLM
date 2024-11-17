from transformers.trainer_callback import TrainerCallback
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
import logging
from typing import Dict, Any

class TensorBoardCallback(TrainerCallback):
    """Custom callback for logging metrics to TensorBoard during training and evaluation."""
    
    def __init__(self, log_dir: str = None):
        """Initialize the TensorBoard callback.
        
        Args:
            log_dir: Directory where TensorBoard logs will be saved. If None, defaults to './logs/runs/TIMESTAMP'
        """
        super().__init__()
        if log_dir is None:
            current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            log_dir = os.path.join('./logs', 'runs', current_time)
        
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logging_dir = log_dir
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TensorBoard logs will be saved to: {log_dir}")

    def on_train_begin(self, args, state, control, **kwargs):
        """Log hyperparameters at the start of training."""
        hparams = {
            'batch_size': args.per_device_train_batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'warmup_steps': args.warmup_steps,
            'num_train_epochs': args.num_train_epochs,
        }
        self.writer.add_hparams(hparams, {})

    def on_log(self, args, state, control, logs: Dict[str, Any] = None, **kwargs):
        """Log metrics during training and evaluation.
        
        Args:
            logs: Dictionary containing the metrics to log
        """
        if logs is None:
            return
        
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                if not key.startswith('eval_'):
                    # Training metrics
                    if key == 'loss':
                        self.writer.add_scalar(f'training/{key}', value, state.global_step)
                    elif key == 'grad_norm':
                        self.writer.add_scalar(f'training/{key}', value, state.global_step)
                    elif key == 'learning_rate':
                        self.writer.add_scalar('training/learning_rate', value, state.global_step)
                    else:
                        self.writer.add_scalar(f'other/{key}', value, state.global_step)

    def on_evaluate(self, args, state, control, metrics: Dict[str, float] = None, **kwargs):
        """Log evaluation metrics.
        
        Args:
            metrics: Dictionary containing the evaluation metrics
        """
        if metrics is None:
            return
            
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'evaluation/{key}', value, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Cleanup when training ends."""
        self.writer.close()