from models.MoE import MoE

def get_model(config, dataset_info):
    """
    Factory function to create a model based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        dataset_info (dict): Dataset information including num_features and num_classes
        
    Returns:
        nn.Module: Instantiated model
    """
    model_config = config['model']
    model_type = model_config['type']
    
    # Common parameters
    num_features = dataset_info['num_features']
    num_classes = dataset_info['num_classes']
    hidden_dim = model_config['hidden_dim']
    num_layers = model_config['num_layers']
    dropout = model_config['dropout']
    
    # Create model based on type
    if model_type == 'moe':
        model = MoE(config, dataset_info)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model 