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
    
    
    # Dataset 4: Experimental dataset - 200 customers, 100 facilities, ratio 5
    n_customers_exp2 = 200
    n_facilities_exp2 = 100
    ratio = 5
    n_exp2 = 200

    lp_dir = f'data/instances/facilities/experiment_{n_customers_exp2}_{n_facilities_exp2}_{ratio}'
    os.makedirs(lp_dir, exist_ok=True)
    print(f"Generating {n_exp2} experimental instances with {n_customers_exp2} customers and {n_facilities_exp2} facilities")
    for i in range(n_exp2):
        filename = os.path.join(lp_dir, f'instance_{i+1}.lp')
        print(f"  generating file {filename} ...")
        generate_capacited_facility_location(
            random=rng,
            filename=filename,
            n_customers=n_customers_exp2,
            n_facilities=n_facilities_exp2,
            ratio=ratio
        )

print("Dataset generation complete!")
