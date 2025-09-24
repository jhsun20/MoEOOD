import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import datetime
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch.cuda.amp import autocast, GradScaler
import gc
import time

from data.dataset_loader import load_dataset
from models.model_factory import get_model
from utils.logger import Logger
from utils.metrics import compute_metrics

import warnings

# Ignore all FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="An issue occurred while importing 'pyg-lib'")
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'")
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


def _named_children_safe(m):
    try:
        return dict(m.named_children())
    except Exception:
        return {}

def get_gate_modules(model):
    """
    Returns (gate_modules_dict, gate_param_list).
    Covers: gate_enc, gate_mlp, and the tiny _gate_mlp used in _gate_logits_expert_features.
    Works with DataParallel and plain modules.
    """
    moem = model.module if hasattr(model, "module") else model
    mods = _named_children_safe(moem)
    gate_mods = {}

    # Canonical gate parts
    if hasattr(moem, "gate_enc"):  gate_mods["gate_enc"]  = moem.gate_enc
    if hasattr(moem, "gate_mlp"):  gate_mods["gate_mlp"]  = moem.gate_mlp

    # The small per-expert-feature MLP is created lazily; include it if it exists
    if hasattr(moem, "_gate_mlp") and (moem._gate_mlp is not None):
        gate_mods["_gate_mlp"] = moem._gate_mlp

    # Collect parameters
    gate_params = []
    for m in gate_mods.values():
        gate_params += list(m.parameters())
    return gate_mods, gate_params

def freeze_all_but_gate(model):
    """
    Sets requires_grad=False for every parameter except those in the gate (enc+MLPs).
    Keeps BatchNorms etc. in eval() for frozen parts; gate stays train().
    """
    moem = model.module if hasattr(model, "module") else model
    # 1) Find gate params
    _, gate_params = get_gate_modules(model)
    gate_param_ids = {id(p) for p in gate_params}

    # 2) Freeze everything not in gate
    for p in moem.parameters():
        p.requires_grad = (id(p) in gate_param_ids)

    # 3) Put frozen submodules to eval() for stability; gate to train()
    #    (This avoids BN running-stat updates in frozen experts.)
    if hasattr(moem, "shared"):
        moem.shared.eval()
    if hasattr(moem, "gate_enc"):
        moem.gate_enc.eval()
    if hasattr(moem, "gate_mlp"):
        moem.gate_mlp.eval()
    if hasattr(moem, "_gate_mlp") and moem._gate_mlp is not None:
        moem._gate_mlp.train()


@torch.no_grad()
def _bump_epoch_counter(model, steps: int = 1):
    moem = model.module if hasattr(model, "module") else model
    moem.set_epoch(moem.current_epoch + int(steps))

def finetune_gate_only(model, train_loader, val_loader, id_val_loader, dataset_info, device, config, logger, best):
    """
    After loading the best checkpoint: freeze experts, optimize only the gate for a few epochs.
    Uses your existing train_epoch_moe/evaluate_moe so gradients flow to the gate via:
      - gate KL/load-balance loss, and
      - task losses through gate-weighted aggregation (experts are frozen).
    """
    gate_epochs = int(config.get('gate', {}).get('finetune_epochs', 5))
    if gate_epochs <= 0:
        return  # nothing to do

    # Build gate-only optimizer
    _, gate_params = get_gate_modules(model)
    ft_lr = float(config.get('gate', {}).get('finetune_lr', config['training']['lr']))
    ft_wd = float(config.get('gate', {}).get('finetune_weight_decay', config['training']['weight_decay']))
    gate_opt = torch.optim.Adam(gate_params, lr=ft_lr, weight_decay=ft_wd)

    # Freeze everything else
    freeze_all_but_gate(model)

    metric_type = dataset_info['metric']
    primary_metric = 'accuracy' if metric_type == 'Accuracy' else metric_type.lower().replace('-', '_')
    best_val_metric = best
    for e in range(1, gate_epochs + 1):
        # keep epoch counter advancing so routing uses learned gate (post-warmup)
        _bump_epoch_counter(model, steps=1)

        # Train one epoch (only gate params require grad)
        train_metrics = train_epoch_moe(model, train_loader, gate_opt, dataset_info, device, epoch=100, config=config)

        # (Optional) quick val to watch overfitting
        val_metrics = evaluate_moe(model, val_loader, device, metric_type, epoch=100, config=config)
        val_id_metrics = evaluate_moe(model, id_val_loader, device, metric_type, epoch=100, config=config)

        current_metric = val_metrics[primary_metric]
        is_better = (current_metric < best_val_metric - config['training']['early_stopping']['min_delta']) if primary_metric in ['RMSE', 'MAE', 'loss'] else (current_metric > best_val_metric + config['training']['early_stopping']['min_delta'])
        if is_better:
            best_val_metric = current_metric
            patience_counter = 0
            best_epoch = config['training']['epochs'] + e  # Update best epoch
            logger.save_model(model, best_epoch, val_metrics)
            logger.logger.info(f"New best model saved with {primary_metric}: {best_val_metric:.4f}")
    # After finetune, keep experts frozen or unfreeze as you wish (we keep frozen).


def train_epoch_moe(model, loader, optimizer, dataset_info, device, epoch, config):
    model.train()
    scaler = GradScaler()

    total_loss = total_ce_loss = total_reg_loss = 0
    total_div_loss = total_gate_loss = 0
    all_targets = []
    all_aggregated_outputs = []

    verbose = model.verbose
    model.set_epoch(epoch)

    pbar = tqdm(loader, desc='Training MoEUIL', leave=False)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        with autocast():
            aggregated_outputs = model(data)
            loss = aggregated_outputs['loss_total']

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = data.y.size(0)

        gate_weights = aggregated_outputs['gate_weights']

        total_loss += loss.item() * batch_size
        total_ce_loss += aggregated_outputs['loss_ce'].item() * batch_size
        total_reg_loss += aggregated_outputs['loss_reg'].item() * batch_size
        total_div_loss += aggregated_outputs['loss_div'].item() * batch_size
        total_gate_loss += aggregated_outputs['loss_gate'].item() * batch_size
        all_targets.append(data.y.detach())
        all_aggregated_outputs.append(aggregated_outputs['logits'].detach())

    # shapes now consistent: concat along batch dimension
    gate_weights_all = torch.cat(gate_weights, dim=0)
    load_balance = gate_weights_all.mean(dim=0)

    final_outputs = torch.cat(all_aggregated_outputs, dim=0)
    final_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(final_outputs, final_targets, dataset_info['metric'])
    metrics['loss'] = total_loss / len(loader.dataset)
    metrics['loss_ce'] = total_ce_loss / len(loader.dataset)
    metrics['loss_reg'] = total_reg_loss / len(loader.dataset)
    metrics['loss_div'] = total_div_loss / len(loader.dataset)
    metrics['loss_gate'] = total_gate_loss / len(loader.dataset)

    gc.collect(); torch.cuda.empty_cache()
    return metrics


def evaluate_moe(model, loader, device, metric_type, epoch, config):
    model.eval()
    total_loss = total_ce_loss = total_reg_loss = 0
    total_div_loss = total_gate_loss = 0
    all_targets = []
    all_aggregated_outputs = []
    verbose = model.verbose
    model.set_epoch(epoch)

    pbar = tqdm(loader, desc='Evaluating MoEUIL', leave=False)
    for data in pbar:
        data = data.to(device)
        with autocast():
            aggregated_outputs = model(data)

        batch_size = data.y.size(0)

        gate_weights = aggregated_outputs['gate_weights']
        expert_logits = aggregated_outputs['expert_logits']

        total_loss += aggregated_outputs['loss_total'].item() * batch_size
        total_ce_loss += aggregated_outputs['loss_ce'].item() * batch_size
        total_reg_loss += aggregated_outputs['loss_reg'].item() * batch_size
        total_div_loss += aggregated_outputs['loss_div'].item() * batch_size
        total_gate_loss += aggregated_outputs['loss_gate'].item() * batch_size
        all_targets.append(data.y.detach())
        all_aggregated_outputs.append(aggregated_outputs['logits'].detach())

    final_outputs = torch.cat(all_aggregated_outputs, dim=0)
    final_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(final_outputs, final_targets, metric_type)

    metrics['loss'] = total_loss / len(loader.dataset)
    metrics['loss_ce'] = total_ce_loss / len(loader.dataset)
    metrics['loss_reg'] = total_reg_loss / len(loader.dataset)
    metrics['loss_div'] = total_div_loss / len(loader.dataset)
    metrics['loss_gate'] = total_gate_loss / len(loader.dataset)
    gc.collect(); torch.cuda.empty_cache()
    return metrics


def train(config, trial=None):
    """Main training function."""
    
    # Set device
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else "cpu")
    verbose = config['experiment']['debug']['verbose']
    
    # Initialize logger
    logger = Logger(config)
    is_tuning = config.get("experiment", {}).get("hyper_search", {}).get("enable", False)
    if not is_tuning:
        logger.logger.info(f"Using device: {device}")
    else:
        start_time = time.time()
    
    # Load dataset
    if not is_tuning:
        logger.logger.info("Loading dataset...")
    data_loaders = load_dataset(config)
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']  # OOD validation
    test_loader = data_loaders['test_loader']  # OOD test
    id_val_loader = data_loaders['id_val_loader']  # In-distribution validation
    id_test_loader = data_loaders['id_test_loader']  # In-distribution test
    dataset_info = data_loaders['dataset_info']
    
    metric_type = dataset_info['metric']
    logger.set_metric_type(metric_type)  # Set the metric type in logger
 
    # Get today's date and current time
    now = datetime.datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H-%M-%S")

    # Prepare results directory with today's date and current time
    if not is_tuning:
        results_dir = os.path.join(
            "results",
            f"{config['experiment']['name']}_{config['dataset']['dataset_name']}_{today_date}_{current_time}"
        )
        os.makedirs(results_dir, exist_ok=True)

    all_test_ood_metrics = []
    all_test_id_metrics = []
    all_train_metrics = []
    all_val_ood_metrics = []
    all_val_id_metrics = []

    # Iterate over each seed
    for seed in config['experiment']['seeds']:
        # Set new seed
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        if not is_tuning:
            logger.logger.info(f"Running with seed: {seed}")
        logger.reset()

        # Initialize model
        if not is_tuning:
            logger.logger.info(f"Initializing {config['model']['type']} model...")
        model = get_model(config, dataset_info)
        model = model.to(device)

        experts_params = list(model.shared.parameters())
        gate_params    = list(model._gate_mlp.parameters())  # gate encoder + MLP
        optimizer = torch.optim.Adam([
            {"params": experts_params, "lr": config['training']['lr'], "weight_decay": config['training']['weight_decay'], "name": "experts"},
            {"params": gate_params,    "lr": config['training']['lr']*0.1, "weight_decay": config['training']['weight_decay'], "name": "gate"},
        ])
        lr_decay_factor = 0.5
        lr_min = 1e-5

        

        if not is_tuning:
            logger.logger.info("Starting training...")

        primary_metric = 'accuracy' if metric_type == 'Accuracy' else metric_type.lower().replace('-', '_')
        if primary_metric == 'accuracy' or primary_metric == 'roc_auc':
            best_val_metric = 0
        else:
            best_val_metric = 1000000
        patience_counter = 0
        patience = config['training']['early_stopping']['patience']
        best_epoch = 0
        
        num_epochs = config['experiment']['debug']['epochs'] if config['experiment']['debug']['enable'] else config['training']['epochs']
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = train_epoch_moe(model, train_loader, optimizer, dataset_info, device, epoch, config)
            val_metrics = evaluate_moe(model, val_loader, device, metric_type, epoch, config)
            id_val_metrics = evaluate_moe(model, id_val_loader, device, metric_type, epoch, config)
            if not is_tuning:
                logger.log_metrics(train_metrics, epoch, phase="train")
                logger.log_metrics(val_metrics, epoch, phase="val_ood")
                logger.log_metrics(id_val_metrics, epoch, phase="val_id")
            eval_metric = 'accuracy' if metric_type == 'Accuracy' else metric_type.lower().replace('-', '_')
            if primary_metric not in val_metrics:
                primary_metric = list(val_metrics.keys())[0]
                
            current_metric = val_metrics[primary_metric]
            current_eval_metric = val_metrics[eval_metric]
            is_better = (current_metric < best_val_metric - config['training']['early_stopping']['min_delta']) if primary_metric in ['rmse', 'mae', 'loss'] else (current_metric > best_val_metric + config['training']['early_stopping']['min_delta'])
            
            if is_better:
                best_val_metric = current_metric
                patience_counter = 0
                best_epoch = epoch
                logger.save_model(model, epoch, val_metrics)
                if not is_tuning:
                    logger.logger.info(f"New best model saved with {primary_metric}: {best_val_metric:.4f} and primary metric {eval_metric}: {current_eval_metric:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    for i, pg in enumerate(optimizer.param_groups):
                        old_lr = float(pg.get('lr', 0.0))
                        new_lr = max(old_lr * lr_decay_factor, lr_min)
                        if new_lr < old_lr:
                            pg['lr'] = new_lr
                            if not is_tuning:
                                name = pg.get('name', f'group_{i}')
                    patience_counter = 0
        
        if False:
            del optimizer
            if 'scaler' in locals():
                del scaler
            gc.collect()
            torch.cuda.empty_cache()
        
        else:
            logger.save_model(model, epoch, val_metrics)
            best_val_metric = current_metric
            logger.logger.info("Evaluating on test sets...")
            print("Evaluating on test sets...")
            del optimizer
            if 'scaler' in locals():
                del scaler
            gc.collect()
            torch.cuda.empty_cache()
            logger.load_best_model(model)
            try:
                finetune_gate_only(model, train_loader, val_loader, test_loader, dataset_info, device, config, logger, best_val_metric)
            except Exception as e:
                print(f"[WARN] Gate-only fine-tune skipped due to error: {e}")

            test_ood_metrics = evaluate_moe(model, test_loader, device, metric_type, epoch, config)
            test_id_metrics = evaluate_moe(model, id_test_loader, device, metric_type, epoch, config)
            
            # Log test metrics with the best epoch
            logger.log_metrics(test_ood_metrics, best_epoch, phase="test_ood")
            logger.log_metrics(test_id_metrics, best_epoch, phase="test_id")
            all_test_ood_metrics.append(test_ood_metrics)
            all_test_id_metrics.append(test_id_metrics)
            all_train_metrics.append(train_metrics)
            all_val_ood_metrics.append(val_metrics)
            all_val_id_metrics.append(id_val_metrics)
            if config['logging']['save_model']:
                final_checkpoint_path = os.path.join(results_dir, f"final_model_checkpoint_{seed}.pth")
                torch.save(model.state_dict(), final_checkpoint_path)

    if is_tuning:
        logger.close()
        elapsed_time = time.time() - start_time
        print(f"Trial completed in {elapsed_time:.2f} seconds with {primary_metric}: {test_ood_metrics[primary_metric]:.4f}")
        return test_ood_metrics[primary_metric]

    avg_test_ood_primary_metric = sum(metrics[eval_metric] for metrics in all_test_ood_metrics) / len(config['experiment']['seeds'])
    avg_test_id_primary_metric = sum(metrics[eval_metric] for metrics in all_test_id_metrics) / len(config['experiment']['seeds'])
    results_path = os.path.join(results_dir, "metrics.yaml")
    with open(results_path, 'w') as f:
        yaml.dump({
            'avg_test_ood_primary_metric': avg_test_ood_primary_metric,
            'avg_test_id_primary_metric': avg_test_id_primary_metric
        }, f)
    
    config_path = os.path.join(results_dir, "config_used.yaml")
    with open(config_path, 'w') as config_file:
        yaml.dump(config, config_file)

    logger.close()
    
    return {
        'test_ood': test_ood_metrics,
        'test_id': test_id_metrics
    }