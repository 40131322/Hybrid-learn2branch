"""
File adapted from https://github.com/pg2455/Hybrid-learn2branch
"""
import os
import sys
import importlib
import argparse
import csv
import numpy as np
import time
import pickle
import pathlib
import gzip

import torch

import utilities
from utilities_gcnn_torch import GCNNDataset as Dataset, load_batch_gcnn as load_batch

def process(model, dataloader, top_k, optimizer=None):
    """
    Executes a forward and backward pass of model over the dataset.

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : torch.utils.data.DataLoader
        Dataset to use for training the model.
    top_k : list
        list of `k` (int) to estimate for accuracy using these many candidates
    optimizer :  torch.optim
        optimizer to use for SGD. No gradient computation takes place if its None.

    Return
    ------
    mean_loss : np.float
        mean loss of model on data in dataloader
    mean_kacc : np.array
        computed accuracy for `top_k` candidates
    """
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores, weights = map(lambda x:x.to(device), batch)
        batched_states = (c, ei, ev, v, n_cs, n_vs)
        batch_size = n_cs.shape[0]
        weights /= batch_size # sum loss

        with torch.no_grad():
            _, logits = model(batched_states)  # eval mode
            logits = torch.unsqueeze(torch.gather(input=torch.squeeze(logits, 0), dim=0, index=cands), 0)  # filter candidate variables
            logits = model.pad_output(logits, n_cands)  # apply padding now
            loss = _loss_fn(logits, best_cands, weights)

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

        mean_loss += loss.detach_().item() * batch_size
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed

    return mean_loss, mean_kacc

def _loss_fn(logits, labels, weights):
    loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
    return torch.sum(loss * weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--config',
        help='dataset configuration (e.g., 200_50_3, 200_100_5)',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--model_name',
        help='name of the model to test (e.g., baseline_torch)',
        type=str,
        default='baseline_torch',
    )
    parser.add_argument(
        '--max_samples',
        help='maximum number of samples to use from the dataset (default: use all)',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--start_idx',
        help='starting index for sample selection (default: 0)',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    seeds = [0]
    test_batch_size = 21
    top_k = [1, 3, 5, 10]
    num_workers = 2

    ### OUTPUT ###
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    
    # Create output directories
    result_dir = f"test_results/experiment/{args.config}"
    os.makedirs(result_dir, exist_ok=True)
    
    result_file = f"{result_dir}/{args.problem}_{args.model_name}_{timestamp}.csv"

    ### NUMPY / TORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    ### SET-UP DATASET ###
    problem_folder = f"data/samples/{args.problem}/{args.config}/experiment"
    
    # Get all sample files
    all_sample_files = list(pathlib.Path(problem_folder).glob('sample_*.pkl'))
    all_sample_files.sort()  # Ensure consistent ordering
    
    # Apply sample selection parameters
    start_idx = args.start_idx
    if start_idx >= len(all_sample_files):
        print(f"Error: Start index {start_idx} is out of range. There are only {len(all_sample_files)} sample files.")
        sys.exit(1)
        
    selected_files = all_sample_files[start_idx:]
    
    # Apply max_samples limit if specified
    if args.max_samples is not None:
        if args.max_samples <= 0:
            print("Error: max_samples must be positive")
            sys.exit(1)
        selected_files = selected_files[:args.max_samples]
    
    test_files = [str(x) for x in selected_files]
    
    print(f"Using {len(test_files)} test samples from index {start_idx}")
    print(f"Sample files range: {os.path.basename(test_files[0])} to {os.path.basename(test_files[-1])}")
    
    test_data = Dataset(test_files)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size,
                            shuffle=False, num_workers=num_workers, collate_fn=load_batch)

    # Set up the model to test
    policy = {
        'name': args.model_name,
        'type': 'gcnn'
    }

    fieldnames = [
        'problem',
        'policy',
        'config',
        'seed',
        'samples_used',
        'start_idx',
    ] + [
        f'acc@{k}' for k in top_k
    ]

    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        print(f"Testing {policy['type']}:{policy['name']} on {args.config}...")
        
        for seed in seeds:
            rng = np.random.RandomState(seed)

            # Load model
            sys.path.insert(0, os.path.abspath(f"models/{policy['name']}"))
            import model
            importlib.reload(model)
            del sys.path[0]
            policy['model'] = model.GCNPolicy()
            policy['model'].restore_state(f"trained_models/{args.problem}/{policy['name']}/{seed}/best_params.pkl")
            policy['model'].to(device)

            test_loss, test_kacc = process(policy['model'], test_data, top_k)
            print(f"  Seed {seed} " + " ".join([f"acc@{k}: {100*acc:4.1f}" for k, acc in zip(top_k, test_kacc)]))

            writer.writerow({
                **{
                    'problem': args.problem,
                    'policy': f"{policy['type']}:{policy['name']}",
                    'config': args.config,
                    'seed': seed,
                    'samples_used': len(test_files),
                    'start_idx': start_idx,
                },
                **{
                    f'acc@{k}': test_kacc[i] for i, k in enumerate(top_k)
                },
            })
            csvfile.flush()
            
    print(f"Results saved to {result_file}")