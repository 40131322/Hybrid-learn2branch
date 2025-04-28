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
from utilities import log, _get_model_type
from utilities_hybrid import HybridDataset as Dataset, load_batch

def process(model, teacher, dataloader, top_k, no_e2e=False):
    """
    Executes only a forward pass of model over the dataset and computes accuracy

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    teacher : model.BaseModel
        A pretrained model when args.no_e2e is True, and an expert model when it is True.
    dataloader : torch.utils.data.DataLoader
        Dataset to use for training the model.
    top_k : list
        list of `k` (int) to estimate for accuracy using these many candidates
    no_e2e :  bool
        if True, assumes that the model needs `teacher` to compute its pretrained embedding

    Return
    ------
    mean_kacc : np.array
        computed accuracy for `top_k` candidates
    """

    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        root_g, node_g, node_attr = [map(lambda x:x if x is None else x.to(device) , y) for y in batch]
        root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, root_cands, root_n_cands = root_g
        node_c, node_ei, node_ev, node_v, node_n_cs, node_n_vs, candss = node_g
        cand_features, n_cands, best_cands, cand_scores, weights  = node_attr
        cands_root_v = None

        if no_e2e:
            with torch.no_grad():
                root_v, _ = teacher((root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs))
                cands_root_v = root_v[candss]

        batched_states = (root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, candss, cand_features, cands_root_v)
        batch_size = n_cands.shape[0]
        weights /= batch_size # sum loss

        with torch.no_grad():
            _, logits, _ = model(batched_states)  # eval mode
            logits = model.pad_output(logits, n_cands)  # apply padding now

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

        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_kacc /= n_samples_processed

    return mean_kacc

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
        '-m', '--model_string',
        help='searches for this string in respective trained_models folder',
        type=str,
        default='',
    )
    parser.add_argument(
        '--model_name',
        help='searches for this model_name in respective trained_models folder',
        type=str,
        default='',
    )
    parser.add_argument(
        '--config',
        help='dataset configuration (e.g., 200_50_3, 200_100_5)',
        type=str,
        required=True,
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
    teacher_model = "baseline_torch" # used if pretrained model is used
    seeds = [0]
    test_batch_size = 32
    top_k = [1, 3, 5, 10]
    num_workers = 2

    ### MODELS TO TEST ###
    if args.model_string != "":
        models_to_test = [y for y in pathlib.Path(f"trained_models/{args.problem}").iterdir() if args.model_string in y.name]
        assert len(models_to_test) > 0, f"no model matched the model_string: {args.model_string}"
    elif args.model_name != "":
        model_path = pathlib.Path(f"trained_models/{args.problem}/{args.model_name}")
        assert model_path.exists(), f"path: {model_path} doesn't exist"
        models_to_test = [model_path]
    else:
        models_to_test = [y for y in pathlib.Path(f"trained_models/{args.problem}").iterdir()]
        assert len(models_to_test) > 0, f"no model matched the model_string: {args.model_string}"

    ### OUTPUT ###
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    
    # Create output directories for experiment results
    result_dir = f"test_results/experiment/{args.config}"
    os.makedirs(result_dir, exist_ok=True)
    
    ### NUMPY / TORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    ### SET-UP DATASET ###
    # Use experiment folder with the specified configuration
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
    if test_files:
        print(f"Sample files range: {os.path.basename(test_files[0])} to {os.path.basename(test_files[-1])}")
    else:
        print("No sample files found!")
        sys.exit(1)
    
    test_data = Dataset(test_files)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size,
                            shuffle=False, num_workers=num_workers, collate_fn=load_batch)

    evaluated_policies = []
    for model in models_to_test:
        try:
            model_type = _get_model_type(model.name)
        except ValueError as e:
            print(e, " skipping it...")
            continue

        evaluated_policies += [[model_type, model]]

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

    for model_type, model_path in evaluated_policies:
        # Create a model-specific result file
        result_file = f"{result_dir}/{args.problem}_{model_path.name}_{timestamp}.csv"
        
        print(f"Testing {model_type}:{model_path.name} on {args.config}...")
        
        with open(result_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for seed in seeds:
                rng = np.random.RandomState(seed)

                policy = {}
                policy['name'] = model_path.name
                policy['type'] = model_type

                # load model
                best_params = str(model_path / f"{seed}/best_params.pkl")
                sys.path.insert(0, os.path.abspath(f"models/{model_type}"))
                import model
                importlib.reload(model)
                del sys.path[0]
                policy['model'] = model.Policy()
                policy['model'].restore_state(best_params)
                policy['model'].to(device)

                ### TEACHER MODEL LOADING ###
                no_e2e = "-pre" in model_type
                teacher=None
                if no_e2e:
                    sys.path.insert(0, os.path.abspath(f'models/{teacher_model}'))
                    import model
                    importlib.reload(model)
                    teacher = model.GCNPolicy()
                    del sys.path[0]
                    teacher.restore_state(f"trained_models/{args.problem}/{teacher_model}/{seed}/best_params.pkl")
                    teacher.to(device)
                    teacher.eval()

                test_kacc = process(policy['model'], teacher, test_data, top_k, no_e2e=no_e2e)
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