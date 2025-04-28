"""
File adapted from https://github.com/pg2455/Hybrid-learn2branch
"""
import os
import argparse
import sys
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utilities
from utilities import log
from utilities_mlp import MLPDataset, load_batch

def process_batch(model, batch, device, optimizer=None):
    """Process a single batch through the model."""
    cand_features, n_cands, best_cands, cand_scores, weights = map(lambda x: x.to(device), batch)
    batch_size = n_cands.shape[0]
    weights /= batch_size  # normalize loss

    # Forward pass
    with torch.no_grad() if optimizer is None else torch.enable_grad():
        logits = model(cand_features)
        logits = model.pad_output(logits, n_cands)
        loss = torch.sum(torch.nn.CrossEntropyLoss(reduction='none')(logits, best_cands) * weights)

    # Backward pass if training
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate accuracy metrics
    true_scores = model.pad_output(torch.reshape(cand_scores, (1, -1)), n_cands)
    true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
    
    return loss.item() * batch_size, true_scores, true_bestscore

def evaluate(model, dataloader, top_k, device, optimizer=None):
    """Evaluate model on dataset."""
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))
    n_samples = 0
    
    for batch in dataloader:
        batch_loss, true_scores, true_bestscore = process_batch(model, batch, device, optimizer)
        batch_size = batch[1].shape[0]  # n_cands
        
        # Calculate top-k accuracy
        logits = model(batch[0].to(device))
        logits = model.pad_output(logits, batch[1].to(device))
        
        for i, k in enumerate(top_k):
            pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
            true_scores_np = true_scores.cpu().numpy()
            true_bestscore_np = true_bestscore.cpu().numpy()
            pred_top_k_scores = np.take_along_axis(true_scores_np, pred_top_k, axis=1)
            mean_kacc[i] += np.mean(np.any(pred_top_k_scores == true_bestscore_np, axis=1)) * batch_size
            
        mean_loss += batch_loss
        n_samples += batch_size
        
    return mean_loss / n_samples, mean_kacc / n_samples

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up configuration
    config = {
        'max_epochs': 50,
        'epoch_size': 32,
        'batch_size': 32,
        'valid_batch_size': 32,
        'lr': 0.003 if args.problem == "facilities" else 0.005,
        'patience': 3,
        'early_stopping': 5,
        'top_k': [1, 3, 5, 10],
        'num_workers': 2
    }
    
    # Set up directories
    problem_folders = {
        'setcover': '500r_1000c_0.05d',
        'cauctions': '100_500',
        'facilities': '100_100_5',
        'indset': '750_4',
    }
    
    model_suffix = f"_{args.node_weights}" if args.node_weights else ""
    running_dir = f"trained_models/{args.problem}/mlp{model_suffix}/{args.seed}"
    os.makedirs(running_dir, exist_ok=True)
    
    # Set up logging
    logfile = os.path.join(running_dir, 'log.txt')
    log_config(config, args, logfile)
    
    # Set up device
    device = setup_device(args.gpu)
    
    # Set up random seed
    set_random_seed(args.seed)
    
    # Load dataset
    problem_folder = problem_folders[args.problem]
    train_files, valid_files = get_dataset_files(args.data_path, args.problem, problem_folder)
    log(f"{len(train_files)} training samples", logfile)
    log(f"{len(valid_files)} validation samples", logfile)
    
    valid_data = DataLoader(
        MLPDataset(valid_files, args.node_weights),
        batch_size=config['valid_batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=load_batch
    )
    
    # Load model
    model = load_model(device)
    
    # Train model
    train_model(
        model, train_files, valid_data, device, 
        config, args, running_dir, logfile
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '--data_path',
        help='Path to data directory with train and valid folders.',
        type=str,
        default="data/samples",
    )
    parser.add_argument(
        '-w','--node_weights',
        help='Weighing scheme for loss',
        choices=['sigmoidal_decay', 'exponential_decay', 'linear_decay', 
                'constant', 'quadratic_decay', ''],
        default="sigmoidal_decay"
    )
    return parser.parse_args()

def log_config(config, args, logfile):
    """Log configuration parameters."""
    for key, value in config.items():
        log(f"{key}: {value}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed: {args.seed}", logfile)
    log(f"node weights: {args.node_weights}", logfile)

def setup_device(gpu_id):
    """Set up and return the appropriate device."""
    if gpu_id == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))
    return rng

def get_dataset_files(data_path, problem, problem_folder):
    """Get the training and validation files."""
    train_path = f"{data_path}/{problem}/{problem_folder}/train"
    valid_path = f"{data_path}/{problem}/{problem_folder}/valid"
    
    train_files = [str(x) for x in pathlib.Path(train_path).glob('sample_*.pkl')]
    valid_files = [str(x) for x in pathlib.Path(valid_path).glob('sample_*.pkl')]
    
    return train_files, valid_files

def load_model(device):
    """Load the MLP model."""
    sys.path.insert(0, os.path.abspath('models/mlp'))
    import model as model_module
    import importlib
    importlib.reload(model_module)
    model = model_module.Policy()
    del sys.path[0]
    model.to(device)
    return model

def train_model(model, train_files, valid_data, device, config, args, running_dir, logfile):
    """Train the model with early stopping."""
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, 
        patience=config['patience'], verbose=True
    )
    
    best_loss, plateau_count = float('inf'), 0
    rng = np.random.RandomState(args.seed)
    
    for epoch in range(config['max_epochs'] + 1):
        log(f"EPOCH {epoch}...", logfile)
        
        # Training phase (skip for epoch 0)
        if epoch > 0:
            epoch_train_files = rng.choice(
                train_files, 
                config['epoch_size'] * config['batch_size'], 
                replace=True
            )
            
            train_data = DataLoader(
                MLPDataset(epoch_train_files, args.node_weights),
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                collate_fn=load_batch
            )
            
            train_loss, train_kacc = evaluate(model, train_data, config['top_k'], device, optimizer)
            log_metrics("TRAIN", train_loss, train_kacc, config['top_k'], logfile)
        
        # Validation phase
        valid_loss, valid_kacc = evaluate(model, valid_data, config['top_k'], device)
        log_metrics("VALID", valid_loss, valid_kacc, config['top_k'], logfile)
        
        # Check for improvement
        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            model.save_state(os.path.join(running_dir, 'best_params.pkl'))
            log("  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count % config['early_stopping'] == 0:
                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
        
        scheduler.step(valid_loss)
    
    # Load best model and evaluate
    model.restore_state(os.path.join(running_dir, 'best_params.pkl'))
    valid_loss, valid_kacc = evaluate(model, valid_data, config['top_k'], device)
    log_metrics("BEST VALID", valid_loss, valid_kacc, config['top_k'], logfile)

def log_metrics(prefix, loss, kacc, top_k, logfile):
    """Log metrics in a consistent format."""
    acc_str = "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, kacc)])
    log(f"{prefix} LOSS: {loss:0.3f} {acc_str}", logfile)

if __name__ == '__main__':
    main()