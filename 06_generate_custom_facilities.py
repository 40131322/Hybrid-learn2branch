import os
import argparse
import numpy as np
from generate_instances import generate_capacited_facility_location

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    ratio = 5  # Standard capacity/demand ratio
    
    # Create directories if they don't exist
    os.makedirs('data/instances/facilities', exist_ok=True)
    
    # Dataset 1: 10 facilities, 10 customers
    n_customers_small = 100
    n_facilities_small = 100
    
    # Train dataset (small)
    n_train_small = 1000
    lp_dir = f'data/instances/facilities/train_{n_customers_small}_{n_facilities_small}_{ratio}'
    os.makedirs(lp_dir, exist_ok=True)
    print(f"Generating {n_train_small} train instances with {n_customers_small} customers and {n_facilities_small} facilities")
    for i in range(n_train_small):
        filename = os.path.join(lp_dir, f'instance_{i+1}.lp')
        print(f"  generating file {filename} ...")
        generate_capacited_facility_location(random=rng, filename=filename, n_customers=n_customers_small, n_facilities=n_facilities_small, ratio=ratio)
    
    # Validation dataset (small)
    n_valid_small = 200
    lp_dir = f'data/instances/facilities/valid_{n_customers_small}_{n_facilities_small}_{ratio}'
    os.makedirs(lp_dir, exist_ok=True)
    print(f"Generating {n_valid_small} validation instances with {n_customers_small} customers and {n_facilities_small} facilities")
    for i in range(n_valid_small):
        filename = os.path.join(lp_dir, f'instance_{i+1}.lp')
        print(f"  generating file {filename} ...")
        generate_capacited_facility_location(random=rng, filename=filename, n_customers=n_customers_small, n_facilities=n_facilities_small, ratio=ratio)
    
    # Test dataset (small)
    n_test_small = 200
    lp_dir = f'data/instances/facilities/test_{n_customers_small}_{n_facilities_small}_{ratio}'
    os.makedirs(lp_dir, exist_ok=True)
    print(f"Generating {n_test_small} test instances with {n_customers_small} customers and {n_facilities_small} facilities")
    for i in range(n_test_small):
        filename = os.path.join(lp_dir, f'instance_{i+1}.lp')
        print(f"  generating file {filename} ...")
        generate_capacited_facility_location(random=rng, filename=filename, n_customers=n_customers_small, n_facilities=n_facilities_small, ratio=ratio)
    
    # Dataset 2: 20 facilities, 60 customers (for experimentation)
    n_customers_exp = 200
    n_facilities_exp = 50
    ratio = 3
    n_exp = 200  # Number of experimental instances
    
    lp_dir = f'data/instances/facilities/experiment_{n_customers_exp}_{n_facilities_exp}_{ratio}'
    os.makedirs(lp_dir, exist_ok=True)
    print(f"Generating {n_exp} experimental instances with {n_customers_exp} customers and {n_facilities_exp} facilities")
    for i in range(n_exp):
        filename = os.path.join(lp_dir, f'instance_{i+1}.lp')
        print(f"  generating file {filename} ...")
        generate_capacited_facility_location(random=rng, filename=filename, n_customers=n_customers_exp, n_facilities=n_facilities_exp, ratio=ratio)
    

        # Dataset 3: 200 customers, 100 facilities, ratio 5
    n_customers_new = 200
    n_facilities_new = 100
    ratio = 5
    n_new = 200  # Number of instances to generate

    lp_dir = f'data/instances/facilities/custom_{n_customers_new}_{n_facilities_new}_{ratio}'
    os.makedirs(lp_dir, exist_ok=True)
    print(f"Generating {n_new} instances with {n_customers_new} customers and {n_facilities_new} facilities")
    for i in range(n_new):
        filename = os.path.join(lp_dir, f'instance_{i+1}.lp')
        print(f"  generating file {filename} ...")
        generate_capacited_facility_location(
            random=rng,
            filename=filename,
            n_customers=n_customers_new,
            n_facilities=n_facilities_new,
            ratio=ratio
        )
print("Dataset generation complete!")
