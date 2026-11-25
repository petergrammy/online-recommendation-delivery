"""Run an online policy on any arrival sequence.

Examples (from the code directory):
    python run_policy.py "../online recommendation/data/arrivals_dataset_1.csv"
    python run_policy.py my_arrivals.csv --policy LP-FD --reps 1000
    python run_policy.py my_arrivals.csv --instance instance.json --output trace.csv

Input CSV: columns t, i_t with t = 1..T and types 1..m. The optional JSON file
holds prices, probabilities, capacities, arrival_rates and K (see instance.json);
without it the small instance of the assignment is used.

Online use: create the policy once, call
    policy.recommend(customer_type_zero_based, current_inventory, remaining_periods_including_current)
in every period, update the inventory with the observed purchase, and call
policy.close() at the end.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from recommendation import default_instance, Instance, FluidLP, FixedTopK, Greedy, LPLookahead, LPDual, simulate

POLICIES = {'LP-FD': LPLookahead, 'LP-Dual': LPDual, 'Greedy': Greedy, 'Alg1': FixedTopK}


def read_arrivals(path, m):
    frame = pd.read_csv(path)
    if not {'t', 'i_t'}.issubset(frame.columns):
        raise ValueError('Expected columns t and i_t.')
    raw = frame['i_t'].to_numpy(float)
    if not np.array_equal(frame['t'].to_numpy(), np.arange(1, len(frame) + 1)) or np.any(~np.isfinite(raw)) \
            or np.any(raw != np.floor(raw)) or np.any(raw < 1) or np.any(raw > m):
        raise ValueError('t must be 1,...,T and i_t an integer type in 1..m.')
    return raw.astype(int) - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--instance', type=Path, help='JSON with prices, probabilities, capacities, arrival_rates, K')
    parser.add_argument('--policy', choices=sorted(POLICIES), default='LP-Dual', help='LP-Dual is the main policy')
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--reps', type=int, default=1, help='number of purchase replications')
    parser.add_argument('--output', type=Path, default=Path('policy_trace.csv'), help='trace of the first replication')
    args = parser.parse_args()
    ins = Instance.from_json(args.instance) if args.instance else default_instance()
    arrivals = read_arrivals(args.csv, ins.m)
    policy = POLICIES[args.policy](ins)
    lp = FluidLP(ins)
    try:
        F = lp.solve(np.bincount(arrivals, minlength=ins.m), ins.capacities)
        rng = np.random.default_rng(args.seed)
        runs = [simulate(ins, arrivals, policy, rng.random(len(arrivals)), trace=(b == 0)) for b in range(args.reps)]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(runs[0].trace).to_csv(args.output, index=False)
        revenue = np.array([r.revenue for r in runs])
        remaining = np.array([r.remaining for r in runs])
        report = {'policy': args.policy, 'T': len(arrivals), 'type_counts': np.bincount(arrivals, minlength=ins.m).tolist(),
                  'hindsight_LP': F, 'replications': args.reps,
                  'revenue': revenue[0] if args.reps == 1 else float(revenue.mean()),
                  'regret': F - (revenue[0] if args.reps == 1 else float(revenue.mean())),
                  'remaining_inventory': remaining[0].tolist() if args.reps == 1 else remaining.mean(axis=0).round(4).tolist(),
                  'out_of_stock_recommendations': int(sum(r.oos_slots for r in runs)), 'trace': str(args.output)}
        print(json.dumps(report, indent=2))
    finally:
        lp.close()
        if hasattr(policy, 'close'):
            policy.close()


if __name__ == '__main__':
    main()
