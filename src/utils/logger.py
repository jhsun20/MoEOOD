import os
import logging
import wandb
import torch
import yaml
from datetime import datetime

class Logger:
    """
    Logger class for tracking and saving experiment results.
    """
    def __init__(self, config):
        """
        Initialize the logger.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.experiment_name = config['experiment']['name']
        self.model_type = config['model']['type']

        self.best_checkpoint_path = None
        self.best_metric_value = float('-inf')
        self.metric_type = None
        
        self.log_dir = os.path.join(
            "logs", 
            f"{self.experiment_name}_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(os.path.join(self.log_dir, "experiment.log"))
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        if config['logging']['wandb']['enable']:
            wandb.init(
                project=config['logging']['wandb']['project'],
                name=f"{self.experiment_name}_{self.model_type}",
                config=config
            )
        
        with open(os.path.join(self.log_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, f)
        
    
    def log_metrics(self, metrics, epoch, phase="train"):
        """
        Log metrics for the current epoch.
        
        Args:
            metrics (dict): Dictionary of metrics
            epoch (int): Current epoch
            phase (str): Phase (train, val, test)
        """
        metric_type = metrics.get('metric_type', 'Accuracy')
        primary_metric_name = self._get_primary_metric_name(metric_type)
        
        if primary_metric_name in metrics:
            primary_metric_value = metrics[primary_metric_name]
            loss_value = metrics.get('loss', 0.0)
            
            log_message = f"Epoch {epoch} - {phase} - Loss: {loss_value:.4f} - {metric_type}: {primary_metric_value:.4f}"
            
            if metric_type not in ['Accuracy', 'RMSE', 'MAE'] and 'accuracy' in metrics:
                log_message += f" - Accuracy: {metrics['accuracy']:.4f}"
                
            self.logger.info(log_message)
            
            loss_keys = [key for key in metrics.keys() if 'loss' in key.lower()]
            if loss_keys:
                loss_str = " | ".join([f"{key}: {metrics[key]:.4f}" for key in sorted(loss_keys)])
                self.logger.info(f"{phase} Losses: {loss_str}")
        else:
            self.logger.info(f"Epoch {epoch} - {phase} - Metrics: {metrics}")
        
        if self.config['logging']['wandb']['enable']:
            wandb_metrics = {f"{phase}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
            wandb_metrics['epoch'] = epoch
            wandb.log(wandb_metrics)
    
    def _get_primary_metric_name(self, metric_type):
        """
        Get the primary metric name based on the metric type.
        
        Args:
            metric_type (str): Type of metric
            
        Returns:
            str: Primary metric name
        """
        metric_mapping = {
            'Accuracy': 'accuracy',
            'F1': 'f1',
            'ROC-AUC': 'roc_auc',
            'Average Precision': 'average_precision',
            'RMSE': 'rmse',
            'MAE': 'mae'
        }
        return metric_mapping.get(metric_type, 'accuracy')
    
    def save_model(self, model, epoch, metrics):
        """
        Save model checkpoint and update best checkpoint if needed.
        
        Args:
            model (torch.nn.Module): Model to save
            epoch (int): Current epoch
            metrics (dict): Dictionary of metrics
        """
        checkpoint_dir = os.path.join(
            "checkpoints", 
            f"{self.experiment_name}_{self.model_type}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        primary_metric_name = self._get_primary_metric_name(self.metric_type)
        primary_metric_value = metrics.get(primary_metric_name, 0.0)
        
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f"epoch_{epoch}_{primary_metric_name}_{primary_metric_value:.4f}.pt"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }, checkpoint_path)
        
        is_better = (primary_metric_value < self.best_metric_value) if self.metric_type in ['RMSE', 'MAE'] else (primary_metric_value > self.best_metric_value)
        if is_better:
            self.best_metric_value = primary_metric_value
            self.best_checkpoint_path = checkpoint_path
    
    def load_best_model(self, model):
        """
        Load the best model checkpoint.
        
        Args:
            model (torch.nn.Module): Model to load weights into
            
        Returns:
            dict: Metrics from the best checkpoint
        """
        if self.best_checkpoint_path is None:
            self.logger.warning("No best checkpoint found!")
            return None
            
        checkpoint = torch.load(self.best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with metrics: {checkpoint['metrics']}")
        return checkpoint['metrics']
    
    def close(self):
        """
        Close the logger.
        """
        if self.config['logging']['wandb']['enable']:
            wandb.finish()
        
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
    
    def set_metric_type(self, metric_type):
        """ 
        Set the metric type and initialize best metric value accordingly.
        
        Args:
            metric_type (str): Type of metric (e.g., 'Accuracy', 'RMSE', 'MAE')
        """
        self.metric_type = metric_type
        self.best_metric_value = float('-inf') if metric_type not in ['RMSE', 'MAE'] else float('inf')
    
    def reset(self):
        """
        Reset the best checkpoint tracking for a new run.
        """
        self.best_checkpoint_path = None
        self.best_metric_value = float('-inf') if self.metric_type not in ['RMSE', 'MAE'] else float('inf')