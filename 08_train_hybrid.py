"""
File adapted from https://github.com/pg2455/Hybrid-learn2branch
"""
import os
import importlib
import argparse
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from utilities import log, _loss_fn, _distillation_loss, _compute_root_loss
from utilities_hybrid import HybridDataset as Dataset, load_batch

def pretrain(model, dataloader, device):
    """Pre-normalize model layers over the given samples."""
    model.pre_train_init()
    layers_processed = 0
    
    while True:
        for batch in dataloader:
            # Process batch data
            root_g, node_g, node_attr = [map(lambda x: x if x is None else x.to(device), y) for y in batch]
            root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, *_ = root_g
            g_c, g_ei, g_ev, g_v, g_n_cs, g_n_vs, candss = node_g
            cand_features, n_cands, best_cands, cand_scores, weights = node_attr
            
            # Create batched states
            batched_states = (root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, candss, cand_features, None)
            
            # Try to pre-train this batch
            if not model.pre_train(batched_states):
                break
                
        # Move to next layer or finish
        res = model.pre_train_next()
        if res is None:
            break
        layers_processed += 1
        
    return layers_processed

def process(model, teacher, dataloader, top_k, optimizer=None, config=None, device=None):
    """Execute forward and backward pass over the dataset."""
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))
    n_samples_processed = 0
    accum_iter = 0
    
    for batch in dataloader:
        # Process batch data
        root_g, node_g, node_attr = [map(lambda x: x if x is None else x.to(device), y) for y in batch]
        root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, root_cands, root_n_cands = root_g
        node_c, node_ei, node_ev, node_v, node_n_cs, node_n_vs, candss = node_g
        cand_features, n_cands, best_cands, cand_scores, weights = node_attr
        cands_root_v = None
        
        # Get teacher predictions if needed
        with torch.no_grad():
            if teacher is not None:
                if config.no_e2e:
                    root_v, _ = teacher((root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs))
                    cands_root_v = root_v[candss]
                    
                if config.distilled:
                    _, soft_targets = teacher((node_c, node_ei, node_ev, node_v, node_n_cs, node_n_vs))
                    soft_targets = torch.unsqueeze(torch.gather(input=torch.squeeze(soft_targets, 0), 
                                                               dim=0, index=candss), 0)
                    soft_targets = model.pad_output(soft_targets, n_cands)
        
        # Prepare inputs
        batched_states = (root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, candss, cand_features, cands_root_v)
        batch_size = n_cands.shape[0]
        weights /= batch_size
        
        # Forward pass (with or without gradients)
        if optimizer:
            optimizer.zero_grad()
            var_feats, logits, film_parameters = model(batched_states)
        else:
            with torch.no_grad():
                var_feats, logits, film_parameters = model(batched_states)
        
        # Apply padding
        logits = model.pad_output(logits, n_cands)
        
        # Calculate loss
        if config.distilled:
            loss = _distillation_loss(logits, soft_targets, best_cands, weights, config.T, config.alpha)
        else:
            loss = _loss_fn(logits, best_cands, weights)
            
        # Add auxiliary loss if needed
        if config.at:
            loss += config.beta_at * _compute_root_loss(config.at, model, var_feats, root_n_vs, 
                                                      root_cands, root_n_cands, batch_size, 
                                                      config.root_cands_separation)
        
        # Add L2 regularization if enabled
        if config.l2 > 0 and film_parameters is not None:
            beta_norm = (1-film_parameters[:, :, 0]).norm()
            gamma_norm = film_parameters[:, :, 1].norm()
            loss += config.l2 * (beta_norm + gamma_norm)
            
        # Backward pass for training
        if optimizer:
            loss.backward()
            accum_iter += 1
            if accum_iter % config.accum_steps == 0:
                optimizer.step()
                accum_iter = 0
                
        # Calculate top-k accuracy
        true_scores = model.pad_output(torch.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
        true_scores = true_scores.cpu().numpy()
        true_bestscore = true_bestscore.cpu().numpy()
        
        kacc = []
        for k in top_k:
            pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
        kacc = np.asarray(kacc)
        
        # Update metrics
        mean_loss += loss.detach_().item() * batch_size
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size
        
    # Compute final metrics
    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed
    
    return mean_loss, mean_kacc

def train_model(config):
    """Main training function with configuration object."""
    # Create output directory
    running_dir = f"trained_models/{config.problem}/{config.modeldir}/{config.seed}"
    os.makedirs(running_dir, exist_ok=True)
    logfile = os.path.join(running_dir, 'log.txt')
    
    # Log configuration
    for key, value in vars(config).items():
        if not key.startswith('__'):
            log(f"{key}: {value}", logfile)
    
    # Set up device
    if config.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{config.gpu}'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    rng = np.random.RandomState(config.seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))
    
    # Prepare dataset
    problem_folder = config.problem_folders[config.problem]
    train_files = list(pathlib.Path(f"{config.data_path}/{config.problem}/{problem_folder}/train").glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f"{config.data_path}/{config.problem}/{problem_folder}/valid").glob('sample_*.pkl'))
    
    log(f"{len(train_files)} training samples", logfile)
    log(f"{len(valid_files)} validation samples", logfile)
    
    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]
    
    # Create validation data loader
    valid_data = Dataset(valid_files, config.data_path)
    valid_loader = DataLoader(valid_data, batch_size=config.valid_batch_size,
                             shuffle=False, num_workers=config.num_workers, 
                             collate_fn=load_batch)
    
    # Create pre-training data loader
    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    pretrain_data = Dataset(pretrain_files, config.data_path)
    pretrain_loader = DataLoader(pretrain_data, batch_size=config.pretrain_batch_size,
                                shuffle=False, num_workers=config.num_workers, 
                                collate_fn=load_batch)
    
    # Load model
    sys.path.insert(0, os.path.abspath(f'models/{config.model}'))
    import model as model_module
    importlib.reload(model_module)
    model = model_module.Policy()
    sys.path.pop(0)
    model.to(device)
    
    # Load teacher model if needed
    teacher = None
    if config.distilled or config.no_e2e:
        sys.path.insert(0, os.path.abspath(f'models/{config.teacher_model}'))
        import model as teacher_module
        importlib.reload(teacher_module)
        teacher = teacher_module.GCNPolicy()
        sys.path.pop(0)
        teacher.restore_state(f"trained_models/{config.problem}/{config.teacher_model}/{config.seed}/best_params.pkl")
        teacher.to(device)
        teacher.eval()
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=config.patience, verbose=True)
    
    # Training loop
    best_loss = np.inf
    plateau_count = 0
    
    for epoch in range(config.max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        
        # Pre-training phase
        if epoch == 0 and not config.no_e2e:
            n_layers = pretrain(model=model, dataloader=pretrain_loader, device=device)
            log(f"PRETRAINED {n_layers} LAYERS", logfile)
        else:
            # Create training data for this epoch
            epoch_train_files = rng.choice(
                train_files, 
                config.epoch_size * config.batch_size * config.accum_steps, 
                replace=True
            )
            train_data = Dataset(epoch_train_files, config.data_path)
            train_loader = DataLoader(
                train_data, 
                batch_size=config.batch_size,
                shuffle=False, 
                num_workers=config.num_workers, 
                collate_fn=load_batch
            )
            
            # Training step
            train_loss, train_kacc = process(
                model, teacher, train_loader, config.top_k, 
                optimizer, config, device
            )
            log(f"TRAIN LOSS: {train_loss:0.3f} " + 
                "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(config.top_k, train_kacc)]), 
                logfile)
        
        # Validation step
        valid_loss, valid_kacc = process(
            model, teacher, valid_loader, config.top_k, 
            None, config, device
        )
        log(f"VALID LOSS: {valid_loss:0.3f} " + 
            "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(config.top_k, valid_kacc)]), 
            logfile)
        
        # Model checkpoint and early stopping
        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            model.save_state(os.path.join(running_dir, 'best_params.pkl'))
            log(f"  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count % config.early_stopping == 0:
                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
        
        # Update learning rate
        scheduler.step(valid_loss)
    
    # Final evaluation with best model
    model.restore_state(os.path.join(running_dir, 'best_params.pkl'))
    valid_loss, valid_kacc = process(
        model, teacher, valid_loader, config.top_k, 
        None, config, device
    )
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + 
        "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(config.top_k, valid_kacc)]), 
        logfile)
    
    return model, best_loss, valid_kacc

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-m', '--model',
        help='model to be trained.',
        type=str,
        default='film',
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--data_path',
        help='name of the folder',
        type=str,
        default="data/samples/",
    )
    parser.add_argument(
        '--no_e2e',
        help='if training is with a pretrained GCNN.',
        action="store_true"
    )
    parser.add_argument(
        '--distilled',
        help='if distillation should be used',
        action="store_true"
    )
    parser.add_argument(
        '--at',
        help='type of auxiliary task',
        type=str,
        default='',
        choices=['ED', 'MHE', '']
    )
    parser.add_argument(
        '--beta_at',
        help='weight for at loss function',
        type=float,
        default=0,
    )
    parser.add_argument(
        '--l2',
        help='regularization film weights',
        type=float,
        default=0.0
    )
    args = parser.parse_args()
    
    # Create configuration object
    class Config:
        def __init__(self, args):
            # Copy arguments
            for key, value in vars(args).items():
                setattr(self, key, value)
            
            # Update model name if using pre-trained embeddings
            if self.model in ['concat', 'film'] and self.no_e2e:
                self.model = f"{self.model}-pre"
                
            # Set hyperparameters
            self.max_epochs = 50
            self.epoch_size = 32
            self.batch_size = 32
            self.accum_steps = 1
            self.pretrain_batch_size = 32
            self.valid_batch_size = 32
            self.lr = 0.005
            self.patience = 3
            self.early_stopping = 3
            self.top_k = [1, 3, 5, 10]
            self.num_workers = 5
            self.teacher_model = "baseline_torch"
            self.T = 2
            self.alpha = 0.9
            self.root_cands_separation = False
            
            # Special settings for facilities problem
            if self.problem == "facilities":
                self.lr = 0.005
                self.epoch_size = 32
                self.batch_size = 32
                self.accum_steps = 2
                self.patience = 2
                self.early_stopping = 2
                self.pretrain_batch_size = 32
                self.valid_batch_size = 32
                self.root_cands_separation = True
                self.num_workers = 2
                
            # Problem folders mapping
            self.problem_folders = {
                'setcover': '500r_1000c_0.05d',
                'cauctions': '100_500',
                'facilities': '100_100_5',
                'indset': '750_4',
            }
            
            # Set model directory name
            self.modeldir = f"{self.model}"
            if self.distilled:
                self.modeldir = f"{self.model}_distilled"
            if self.at:
                self.modeldir = f"{self.modeldir}_{self.at}_{self.beta_at}"
            if self.l2 > 0:
                self.modeldir = f"{self.modeldir}_l2_{self.l2}"
                
    import sys
    config = Config(args)
    train_model(config)