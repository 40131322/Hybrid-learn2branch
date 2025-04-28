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
import importlib
from datetime import datetime

import utilities
from utilities_gcnn_torch import GCNNDataset, load_batch_gcnn

def log(message, logfile=None):
    """Log a message to both stdout and a log file if provided."""
    print(message)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(f"{message}\n")

def pretrain_norm_layers(model, dataloader, device):
    """Pre-normalize model's normalization layers using batched data."""
    model.pre_train_init()
    count = 0
    
    while True:
        for batch in dataloader:
            batch = [x.to(device) for x in batch]
            c, ei, ev, v, n_cs, n_vs = batch[:6]
            batched_states = (c, ei, ev, v, n_cs, n_vs)
            
            if not model.pre_train(batched_states):
                break
                
        res = model.pre_train_next()
        if res is None:
            break
        count += 1
        
    return count

def process_data(model, dataloader, top_k, device, optimizer=None):
    """Process data for training or evaluation."""
    total_loss = 0
    total_accuracy = np.zeros(len(top_k))
    total_samples = 0
    
    for batch in dataloader:
        batch = [x.to(device) for x in batch]
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores, weights = batch
        batched_states = (c, ei, ev, v, n_cs, n_vs)
        batch_size = n_cs.shape[0]
        weights /= batch_size
        
        if optimizer:  # Training mode
            optimizer.zero_grad()
            _, logits = model(batched_states)
            logits = torch.unsqueeze(torch.gather(input=torch.squeeze(logits, 0), dim=0, index=cands), 0)
            logits = model.pad_output(logits, n_cands)
            loss = calculate_loss(logits, best_cands, weights)
            loss.backward()
            optimizer.step()
        else:  # Evaluation mode
            with torch.no_grad():
                _, logits = model(batched_states)
                logits = torch.unsqueeze(torch.gather(input=torch.squeeze(logits, 0), dim=0, index=cands), 0)
                logits = model.pad_output(logits, n_cands)
                loss = calculate_loss(logits, best_cands, weights)
        
        # Calculate top-k accuracy metrics
        true_scores = model.pad_output(torch.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
        true_scores = true_scores.cpu().numpy()
        true_bestscore = true_bestscore.cpu().numpy()
        
        # Calculate accuracy for each k
        k_accuracy = []
        num_candidates = logits.shape[1]

        for k in top_k:
            if k <= num_candidates:
                pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
                pred_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
                k_accuracy.append(np.mean(np.any(pred_scores == true_bestscore, axis=1)))
            else:
                k_accuracy.append(0.0)  # or np.nan if you want to skip in average


        
        # Update totals
        total_loss += loss.item() * batch_size
        total_accuracy += np.array(k_accuracy) * batch_size
        total_samples += batch_size
    
    # Calculate means
    mean_loss = total_loss / total_samples
    mean_accuracy = total_accuracy / total_samples
    
    return mean_loss, mean_accuracy

def calculate_loss(logits, labels, weights):
    """Calculate weighted cross-entropy loss."""
    loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
    return torch.sum(loss * weights)

def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', help='MILP instance type', 
                        choices=['setcover', 'cauctions', 'facilities', 'indset'])
    parser.add_argument('-m', '--model', help='GCNN model name', type=str, default='baseline_torch')
    parser.add_argument('-s', '--seed', help='Random seed', type=utilities.valid_seed, default=0)
    parser.add_argument('-g', '--gpu', help='CUDA GPU id (-1 for CPU)', type=int, default=0)
    parser.add_argument('--data_path', help='Data directory path', type=str, default="")
    parser.add_argument('--l2', help='L2 regularization', type=float, default=0.0)
    args = parser.parse_args()
    
    # Hyperparameters
    config = {
        'max_epochs': 50,
        'epoch_size': 64,
        'batch_size': 32,
        'pretrain_batch_size': 32,
        'valid_batch_size': 32,
        'learning_rate': 0.003,
        'patience': 3,
        'early_stopping': 5,
        'top_k': [1, 3, 5, 10],
        'num_workers': 2
    }
    
    # Problem configuration
    problem_configs = {
        'setcover': '500r_1000c_0.05d',
        'cauctions': '100_500',
        'facilities': '100_100_5',
        'indset': '750_4'
    }
    problem_folder = problem_configs[args.problem]
    
    # Setup directories
    model_name = f"{args.model}" + (f"_l2_{args.l2}" if args.l2 > 0 else "")
    run_dir = f"trained_models/{args.problem}/{model_name}/{args.seed}"
    os.makedirs(run_dir, exist_ok=True)
    logfile = os.path.join(run_dir, 'log.txt')
    
    # Log configuration
    for key, value in config.items():
        log(f"{key}: {value}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed: {args.seed}", logfile)
    log(f"l2: {args.l2}", logfile)
    
    # Set device (CPU/GPU)
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))
    
    # Setup data paths
    data_dir = f'data/samples/{args.problem}/{problem_folder}'
    if args.data_path:
        data_dir = f"{args.data_path}/{args.problem}/{problem_folder}"
    
    # Find data files
    train_files = [str(x) for x in pathlib.Path(f'{data_dir}/train').glob('sample_*.pkl')]
    valid_files = [str(x) for x in pathlib.Path(f'{data_dir}/valid').glob('sample_*.pkl')]
    log(f"{len(train_files)} training samples", logfile)
    log(f"{len(valid_files)} validation samples", logfile)
    
    # Create data loaders
    valid_dataset = GCNNDataset(valid_files)
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config['valid_batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'], 
        collate_fn=load_batch_gcnn
    )
    
    # Create pretraining dataset (10% of training data)
    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    pretrain_dataset = GCNNDataset(pretrain_files)
    pretrain_loader = DataLoader(
        pretrain_dataset, 
        batch_size=config['pretrain_batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'], 
        collate_fn=load_batch_gcnn
    )
    
    # Import and initialize model
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model as model_module
    importlib.reload(model_module)
    model = model_module.GCNPolicy().to(device)
    del sys.path[0]
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=args.l2
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.2, 
        patience=config['patience'], 
        verbose=True
    )
    
    # Training loop
    best_loss = float('inf')
    plateau_count = 0
    
    for epoch in range(config['max_epochs'] + 1):
        log(f"EPOCH {epoch}...", logfile)
        
        # Pretraining or training
        if epoch == 0:
            # Pretrain normalization layers
            n_layers = pretrain_norm_layers(model, pretrain_loader, device)
            log(f"PRETRAINED {n_layers} LAYERS", logfile)
        else:
            # Regular training
            epoch_train_files = rng.choice(
                train_files, 
                config['epoch_size'] * config['batch_size'], 
                replace=True
            )
            train_dataset = GCNNDataset(epoch_train_files)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['batch_size'],
                shuffle=False, 
                num_workers=config['num_workers'], 
                collate_fn=load_batch_gcnn
            )
            
            train_loss, train_acc = process_data(model, train_loader, config['top_k'], device, optimizer)
            acc_str = "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(config['top_k'], train_acc)])
            log(f"TRAIN LOSS: {train_loss:0.3f}{acc_str}", logfile)
        
        # Validation
        valid_loss, valid_acc = process_data(model, valid_loader, config['top_k'], device)
        acc_str = "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(config['top_k'], valid_acc)])
        log(f"VALID LOSS: {valid_loss:0.3f}{acc_str}", logfile)
        
        # Model checkpoint and early stopping
        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            model.save_state(os.path.join(run_dir, 'best_params.pkl'))
            log("  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count % config['early_stopping'] == 0:
                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
        
        # Update learning rate
        scheduler.step(valid_loss)
    
    # Final evaluation with best model
    model.restore_state(os.path.join(run_dir, 'best_params.pkl'))
    final_loss, final_acc = process_data(model, valid_loader, config['top_k'], device)
    acc_str = "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(config['top_k'], final_acc)])
    log(f"BEST VALID LOSS: {final_loss:0.3f}{acc_str}", logfile)

if __name__ == '__main__':
    main()