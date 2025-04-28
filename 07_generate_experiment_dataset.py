"""
File adapted from https://github.com/pg2455/Hybrid-learn2branch
"""
import os
import argparse
import pickle
import glob
import shutil
import gzip
import math
import numpy as np
import multiprocessing as mp

import pyscipopt as scip  # Python interface to the SCIP solver
import utilities

class VanillaFullstrongBranchingDataCollector(scip.Branchrule):
    """
    Implements a branching rule for SCIP that collects data for machine learning models.
    
    This class embeds data collection for hybrid ML+MILP branching models directly in the
    SCIP solver by overriding the branching rule. It collects state information at branching
    nodes and the decisions made by full strong branching (an expensive but high-quality
    branching strategy).
    
    The collector uses an exploration-exploitation approach:
    - With probability query_expert_prob: Use full strong branching (expert) and collect data
    - Otherwise: Use a simpler rule (exploration) without collecting data
    """
    def __init__(self, rng, query_expert_prob=0.60):
        """
        Initialize the data collector branching rule.
        
        Parameters:
        -----------
        rng : numpy.random.RandomState
            Random number generator for making stochastic decisions
        query_expert_prob : float
            Probability of querying the expert (full strong branching) at each node
            This creates an exploration-exploitation trade-off for data collection
            Higher values collect more data but may slow down the solving process
        """
        # Data storage structures
        self.khalil_root_buffer = {}  # Cache for storing Khalil features at the root node
        self.obss = []                # List to store observations (states)
        self.targets = []             # List to store targets (best branching variable indices)
        self.obss_feats = []          # List to store additional features for observations
        
        # Configuration parameters
        self.exploration_policy = "pscost"  # Default policy to use when not querying expert
        self.query_expert_prob = query_expert_prob  # Probability of using expert policy
        self.rng = rng                # Random number generator
        self.iteration_counter = 0    # Counter for branching iterations

    def branchinit(self):
        """
        Initialize branching rule data before solving starts.
        Required by SCIP's branching rule interface.
        
        This method is called once at the beginning of the branch-and-bound process.
        It resets counters and buffers used during the solving process.
        """
        self.ndomchgs = 0  # Counter for domain changes (when branching tightens variable domains)
        self.ncutoffs = 0  # Counter for cutoffs (when branching prunes infeasible branches)
        self.khalil_root_buffer = {}  # Reset the Khalil features buffer (stores features computed at the root node)

    def branchexeclp(self, allowaddcons):
        """
        Execute the branching rule at each node in the branch-and-bound tree.
        
        This is called by SCIP at each node where branching is needed.
        It either calls full strong branching (expert) to collect data or uses
        a simpler branching rule (exploration) depending on random chance.
        
        The key steps are:
        1. Decide whether to query expert or use exploration policy
        2. If using expert: extract state, run full strong branching, record data
        3. Actually perform the branching on the selected variable
        
        Parameters:
        -----------
        allowaddcons : bool
            Whether adding constraints is allowed during branching
            
        Returns:
        --------
        dict
            Result of the branching decision (BRANCHED, REDUCEDDOM, or CUTOFF)
        """
        self.iteration_counter += 1

        # Decide whether to query the expert policy (full strong branching) or use exploration policy
        # Always use expert policy at the root node (getNNodes()==1) to ensure data collection starts properly
        query_expert = self.rng.rand() < self.query_expert_prob
        if query_expert or self.model.getNNodes() == 1:  # Always query expert at root node
            # Get candidate variables for branching (variables that are fractional in the LP relaxation)
            candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]

            # Extract state information (features of the current node)
            # This captures the state of the LP relaxation, constraints, and variables
            state = utilities.extract_state(self.model)
            # Extract additional features as described in Khalil et al. (2016)
            # These features capture more information about variables and their context
            state_khalil = utilities.extract_khalil_variable_features(self.model, candidate_vars, self.khalil_root_buffer)

            # Execute full strong branching rule
            # This evaluates all candidate variables by temporarily branching on them
            # and computing the resulting bound improvements
            result = self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            # Get results from full strong branching
            cands_, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
            best_var = cands_[bestcand]  # Get the best variable selected by full strong branching

            # Add this observation to our dataset
            self.add_obs(best_var, (state, state_khalil), (cands_, scores))
            if self.model.getNNodes() == 1:  # If this is the root node, store its state separately
                # This allows sharing root node information across all node samples
                self.state = [state, state_khalil, self.obss[0]]

            # Actually perform the branching
            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED
        else:
            # Use exploration policy (pseudocost branching by default)
            # This is faster than full strong branching but doesn't collect data
            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)

        # Count node results for metrics
        # REDUCEDDOM means the branching resulted in domain reductions
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        # CUTOFF means the branching resulted in an infeasible subproblem
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}

    def add_obs(self, best_var, state_, cands_scores=None):
        """
        Add a sample to the observation list for later processing.
        
        This records the state of the problem at a branching node and the decision made
        by the expert policy (full strong branching). The state includes features of
        variables, constraints, and their relationships, along with candidate variables
        and their scores from full strong branching.
        
        Parameters:
        -----------
        best_var : pyscipopt.Variable
            The variable selected for branching by the expert policy
        state_ : tuple
            Extracted features of constraints and variables at the current node
        cands_scores : tuple
            Tuple containing (candidate variables, their scores) from full strong branching
            
        Returns:
        --------
        bool
            True if sample added successfully, False otherwise
        """
        # Reset observation lists at root node
        # This ensures clean data collection for each problem instance
        if self.model.getNNodes() == 1:
            self.obss = []
            self.targets = []
            self.obss_feats = []
            # Create a mapping for variable indices (to track original variable indices)
            self.map = sorted([x.getCol().getIndex() for x in self.model.getVars(transformed=True)])

        # Unpack candidate variables and their scores
        cands, scores = cands_scores
        # Skip if scores are inconsistent (negative scores indicate errors)
        # This can happen if SCIP was early stopped due to time limit
        if any([s < 0 for s in scores]):
            return False

        # Unpack state features
        state, state_khalil = state_
        var_features = state[2]['values']      # Variable features (e.g., objective coefficient, bounds)
        cons_features = state[0]['values']     # Constraint features (e.g., dual values, activities)
        edge_features = state[1]               # Edge features (constraint-variable relationships)

        # Add more features to variables
        # This combines standard variable features with Khalil features and candidate indicators
        cands_index = [x.getCol().getIndex() for x in cands]
        # Initialize Khalil features with -1 (for non-candidate variables)
        khalil_features = -np.ones((var_features.shape[0], state_khalil.shape[1]))
        # Binary indicator for candidate variables (1 if variable is a candidate, 0 otherwise)
        cand_ind = np.zeros((var_features.shape[0], 1))
        # Set actual Khalil features for candidate variables
        khalil_features[cands_index] = state_khalil
        cand_ind[cands_index] = 1
        # Concatenate all variable features into a single matrix
        var_features = np.concatenate([var_features, khalil_features, cand_ind], axis=1)

        # Initialize scores vector with -1 (for non-candidate variables)
        # This records the strong branching scores for all variables
        tmp_scores = -np.ones(len(self.map))
        if scores:
            tmp_scores[cands_index] = scores

        # Store the index of the best variable as target (what we want to predict)
        self.targets.append(best_var.getCol().getIndex())
        # Store the observation (state features)
        self.obss.append([var_features, cons_features, edge_features])
        # Store additional metadata about the observation
        depth = self.model.getCurrentNode().getDepth()  # Depth in the branch-and-bound tree
        self.obss_feats.append({'depth': depth, 'scores': np.array(tmp_scores), 'iteration': self.iteration_counter})

        return True

def make_samples(in_queue, out_queue, node_limit, node_record_prob):
    """
    Worker function that processes instances from the input queue and collects samples.
    
    This function runs in a separate process, solves MILP instances using SCIP with
    the custom branching rule, and sends collected samples to the output queue.
    
    The key steps are:
    1. Initialize and configure a SCIP model with the custom branching rule
    2. Solve the instance and collect branching decisions
    3. Save the collected data (root node and other nodes separately)
    4. Send completion notification
    
    Parameters:
    -----------
    in_queue : multiprocessing.Queue
        Queue from which to receive problem instances and settings
    out_queue : multiprocessing.Queue
        Queue to send collected samples
    node_limit : int
        Maximum number of branch-and-bound nodes to explore per instance
    node_record_prob : float
        Probability of recording data at each node (controls data sampling density)
    """
    while True:
        # Get next task from input queue
        # Each task specifies an instance to solve and parameters for the solution process
        episode, instance, seed, time_limit, outdir, rng = in_queue.get()

        # Create and configure SCIP model
        m = scip.Model()
        m.setIntParam('display/verblevel', 0)  # Disable output for cleaner logs
        m.readProblem(f'{instance}')           # Read the problem instance from file
        utilities.init_scip_params(m, seed=seed)  # Initialize solver parameters
        m.setIntParam('timing/clocktype', 2)     # Set clock type to CPU time
        m.setRealParam('limits/time', time_limit)  # Set time limit in seconds
        m.setLongintParam('limits/nodes', node_limit)  # Set node limit (max B&B nodes to explore)

        # Create and include custom branching rule for data collection
        # This overrides SCIP's default branching behavior to collect data
        branchrule = VanillaFullstrongBranchingDataCollector(rng, node_record_prob)
        m.includeBranchrule(
            branchrule=branchrule,
            name="Sampling branching rule", desc="",
            priority=666666,    # High priority to ensure our rule is called first
            maxdepth=-1,        # No depth limit (-1 means apply at all depths)
            maxbounddist=1)     # Maximum distance for bound-based filtering

        # Configure full strong branching parameters
        # These ensure we correctly capture the full strong branching decisions
        m.setBoolParam('branching/vanillafullstrong/integralcands', True)  # Only branch on integer variables
        m.setBoolParam('branching/vanillafullstrong/scoreall', True)       # Score all candidates
        m.setBoolParam('branching/vanillafullstrong/collectscores', True)  # Collect scores for later analysis
        m.setBoolParam('branching/vanillafullstrong/donotbranch', True)    # Only compute scores without branching
        m.setBoolParam('branching/vanillafullstrong/idempotent', True)     # Ensure consistent results

        # Notify that processing has started
        out_queue.put({
            "type": 'start',
            "episode": episode,
            "instance": instance,
            "seed": seed
        })

        # Solve the instance
        # This will trigger our custom branching rule at each node
        m.optimize()
        
        # Save collected data if we have observations and at least one node was processed
        if m.getNNodes() >= 1 and len(branchrule.obss) > 0:
            filenames = []
            # Find maximum tree depth reached (for statistics)
            max_depth = max(x['depth'] for x in branchrule.obss_feats)
            # Collect solving statistics
            stats = {'nnodes': m.getNNodes(), 'time': m.getSolvingTime(), 'gap': m.getGap(), 'nobs': len(branchrule.obss)}

            # Prepare root node data (special handling for the root)
            # Root node data includes additional information used by non-root nodes
            sample_state, sample_khalil_state, root_obss = branchrule.state
            sample_cand_scores = branchrule.obss_feats[0]['scores']
            sample_cands = np.where(sample_cand_scores != -1)[0]  # Identify candidate variables
            sample_cand_scores = sample_cand_scores[sample_cands]  # Get scores for candidates
            cand_choice = np.where(sample_cands == branchrule.targets[0])[0][0]  # Find chosen candidate index

            # Save root node data
            root_filename = f"{outdir}/sample_root_0_{episode}.pkl"
            filenames.append(root_filename)
            with gzip.open(root_filename, 'wb') as f:
                pickle.dump({
                    'type': 'root',               # Indicates this is root node data
                    'episode': episode,           # Episode number for tracking
                    'instance': instance,         # Source problem instance
                    'seed': seed,                 # Random seed used
                    'stats': stats,               # Solving statistics
                    'root_state': [sample_state, sample_khalil_state, sample_cands, cand_choice, sample_cand_scores],
                    'obss': [branchrule.obss[0], branchrule.targets[0], branchrule.obss_feats[0], None],
                    'max_depth': max_depth        # Maximum depth reached in the B&B tree
                }, f)

            # Save data for non-root nodes
            # These contain their own data but reference the root node file
            for i in range(1, len(branchrule.obss)):
                iteration_counter = branchrule.obss_feats[i]['iteration']
                filenames.append(f"{outdir}/sample_node_{iteration_counter}_{episode}.pkl")
                with gzip.open(filenames[-1], 'wb') as f:
                    pickle.dump({
                        'type': 'node',           # Indicates this is non-root node data
                        'episode': episode,       # Episode number for tracking
                        'instance': instance,     # Source problem instance
                        'seed': seed,             # Random seed used
                        'stats': stats,           # Solving statistics
                        'root_state': f"{outdir}/sample_root_0_{episode}.pkl",  # Reference to root data file
                        'obss': [branchrule.obss[i], branchrule.targets[i], branchrule.obss_feats[i], None],
                        'max_depth': max_depth    # Maximum depth reached in the B&B tree
                    }, f)

            # Notify that processing is done and send filenames of created files
            out_queue.put({
                "type": "done",
                "episode": episode,
                "instance": instance,
                "seed": seed,
                "filenames": filenames,
                "nnodes": len(filenames),
            })

        # Free the SCIP problem to release memory
        # This is important to prevent memory leaks during long runs
        m.freeProb()

def send_orders(orders_queue, instances, seed, time_limit, outdir, start_episode):
    """
    Continuously sends problem instances to the worker queue.
    
    This function runs in a separate process and feeds tasks to worker processes.
    It randomly selects instances from the provided list and generates unique seeds
    for each solving task.
    
    Parameters:
    -----------
    orders_queue : multiprocessing.Queue
        Queue to send tasks to workers
    instances : list
        List of file paths to MILP instances
    seed : int
        Initial seed for the random number generator
    time_limit : int
        Time limit for solving each instance (in seconds)
    outdir : str
        Directory to save collected data
    start_episode : int
        Episode number to resume from (for continuing interrupted jobs)
    """
    rng = np.random.RandomState(seed)  # Initialize random number generator
    episode = 0
    while True:
        # Randomly select an instance from the available ones
        instance = rng.choice(instances)
        # Generate random seed for this instance (important for reproducibility and diversity)
        seed = rng.randint(2**31-1)  # Maximum value for 32-bit signed integer
        
        # Skip episodes that have already been processed
        # This supports resuming interrupted data collection jobs
        if episode <= start_episode:
            episode += 1
            continue

        # Send task to worker queue
        # Each task contains all information needed to solve one instance
        orders_queue.put([episode, instance, seed, time_limit, outdir, rng])
        episode += 1

def collect_samples(instances, outdir, rng, n_samples, n_jobs, time_limit):
    """
    Main function to orchestrate parallel data collection.
    
    This function creates multiple worker processes, sends them tasks,
    and collects the results until the desired number of samples is reached.
    
    The key steps are:
    1. Create worker processes and queues for communication
    2. Start a dispatcher process to send tasks to workers
    3. Process completed samples and move them to the output directory
    4. Continue until enough samples are collected
    
    Parameters:
    -----------
    instances : list
        List of MILP instance file paths
    outdir : str
        Directory to save the collected data
    rng : numpy.random.RandomState
        Random number generator
    n_samples : int
        Total number of samples to collect
    n_jobs : int
        Number of parallel worker processes
    time_limit : int
        Time limit for solving each instance (in seconds)
    """
    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Create queues for communication between processes
    orders_queue = mp.Queue(maxsize=2*n_jobs)  # Limited size to prevent memory issues
    answers_queue = mp.SimpleQueue()  # Simple queue for collecting results
    
    # Start worker processes
    workers = []
    for i in range(n_jobs):
        p = mp.Process(
                target=make_samples,
                args=(orders_queue, answers_queue, node_limit, node_record_prob),
                daemon=True)  # Daemon processes terminate when main process exits
        workers.append(p)
        p.start()

    # Create temporary directory for samples
    # This helps manage partial results and keeps accurate count of completed samples
    tmp_samples_dir = f'{outdir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # Check for existing samples to resume from
    # This allows continuing interrupted data collection jobs
    existing_samples = glob.glob(f"{outdir}/*.pkl")
    last_episode, last_i = -1, 0
    if existing_samples:
        # Find the highest episode and sample numbers from existing files
        last_episode = max(int(x.split("/")[-1].split(".pkl")[0].split("_")[-2]) for x in existing_samples)
        last_i = max(int(x.split("/")[-1].split(".pkl")[0].split("_")[-1]) for x in existing_samples)

    # Start dispatcher process to send tasks to workers
    dispatcher = mp.Process(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**31-1), time_limit, tmp_samples_dir, last_episode),
            daemon=True)
    dispatcher.start()

    # Main loop: collect and process samples
    i = last_i  # Resume from last sample number if interrupted
    in_buffer = 0  # Track number of instances being processed
    while i <= n_samples:
        sample = answers_queue.get()  # Wait for a result from any worker

        if sample['type'] == 'start':
            in_buffer += 1  # Track number of instances being processed

        if sample['type'] == 'done':
            # Move completed samples from temp directory to final directory
            for filename in sample['filenames']:
                x = filename.split('/')[-1].split(".pkl")[0]
                os.rename(filename, f"{outdir}/{x}.pkl")
                i += 1
                print(f"[m {os.getpid()}] {i} / {n_samples} samples written, ep {sample['episode']} ({in_buffer} in buffer).")

                # If we've collected enough samples, stop
                if i == n_samples:
                    if dispatcher.is_alive():
                        dispatcher.terminate()
                        print(f"[m {os.getpid()}] dispatcher stopped...")
                    break

        # If dispatcher has finished, exit loop
        # This should only happen if we run out of instances
        if not dispatcher.is_alive():
            break

    # Terminate all worker processes
    # This ensures clean shutdown and prevents zombie processes
    for p in workers:
        p.terminate()

    # Clean up temporary directory
    shutil.rmtree(tmp_samples_dir, ignore_errors=True)

if __name__ == "__main__":
    """
    Script entry point - Parse command line arguments and run data collection.
    
    This specialized version only handles facility location problems with two specific configurations:
    1. 200 facilities, 100 customers, 5 facilities per customer
    2. 200 facilities, 50 customers, 3 facilities per customer
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'configuration',
        help='Facility location configuration to process.',
        choices=['200_100_5', '200_50_3'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    args = parser.parse_args()

    # Dataset configuration
    experiment_size = 25000  # Number of samples to collect for experiment
    time_limit = 600         # Time limit per instance (in seconds)
    node_limit = 500         # Maximum nodes to process per instance
    node_record_prob = 1.0   # Always record all nodes

    basedir = "data/samples/facilities"
    
    # Set up configuration-specific instance directories and parameters
    if args.configuration == '200_100_5':
        # Facility Location Problem (200 facilities, 100 customers, 5 facilities per customer)
        instances = glob.glob('data/instances/facilities/experiment_200_100_5/*.lp')
        out_dir = f'{basedir}/200_100_5/experiment'
        
    elif args.configuration == '200_50_3':
        # Facility Location Problem (200 facilities, 50 customers, 3 facilities per customer)
        instances = glob.glob('data/instances/facilities/experiment_200_50_3/*.lp')
        out_dir = f'{basedir}/200_50_3/experiment'

    # Print dataset statistics
    print(f"{len(instances)} instances for {experiment_size} samples")

    # Collect experimental data
    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances, out_dir, rng, experiment_size, args.njobs, time_limit)
    print(f"Success: Experimental data collection for {args.configuration}")