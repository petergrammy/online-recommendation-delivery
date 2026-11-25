"""All numerical experiments of the report. Run from the code directory.

    python experiments.py all            # everything (about 30-40 minutes on 16 cores)
    python experiments.py exact          # exact evaluation: 5 datasets, block orders, 1000 generated sequences
    python experiments.py count loss     # remaining-count forecast; decomposition of the policy loss
    python experiments.py simulate       # Monte-Carlo replications on the 5 datasets
    python experiments.py q1 offline policy-map sensitivity scaling large

Results are written to ../results. Every experiment uses fixed seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path
from time import perf_counter
import multiprocessing as mp
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import t as student_t

from recommendation import (count_forecast, Instance, default_instance, FluidLP, FixedTopK, Greedy, LPLookahead, LPDual, DPPolicy,
                            simulate, latent_requests, dp_benchmark, exact_forward, Lattice, offer_fixed,
                            offer_greedy, offer_dp_table, LPTables, lp_offer_masks,
                            lp_value_table, action_matrix, top_k, actions, _bellman_type)

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
DATA = ROOT / 'online recommendation' / 'data'
RES = ROOT / 'results'
CACHE = Path(tempfile.gettempdir()) / 'online_recommendation_cache'
SEED = 20261107
WORKERS = max(1, min(14, (os.cpu_count() or 4) - 2))
EXACT_WORKERS = min(WORKERS, 8)   # each exact worker holds the offer tables in memory
POLICIES = ['Alg1', 'Greedy', 'LP-FD', 'LP-Dual', 'IID-DP']


def read_arrivals(path, m):
    frame = pd.read_csv(path)
    if not {'t', 'i_t'}.issubset(frame.columns):
        raise ValueError('Expected t and i_t columns.')
    raw = frame['i_t'].to_numpy(float)
    if not np.array_equal(frame['t'].to_numpy(), np.arange(1, len(frame) + 1)) or np.any(~np.isfinite(raw)) \
            or np.any(raw != np.floor(raw)) or np.any(raw < 1) or np.any(raw > m):
        raise ValueError('t must be 1,...,T and i_t must be a valid integer type.')
    return raw.astype(int) - 1


def datasets(m=3):
    files = sorted(DATA.glob('arrivals_dataset_*.csv'))
    return {f'D{k + 1}': read_arrivals(f, m) for k, f in enumerate(files)}


def ci95(a):
    a = np.asarray(a, float)
    return float(student_t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float('nan')


def hindsight(ins, arrivals, lp=None):
    own = lp is None
    lp = lp or FluidLP(ins)
    try:
        return lp.solve(np.bincount(arrivals, minlength=ins.m), ins.capacities)
    finally:
        if own:
            lp.close()


def block_orders(counts):
    names = {'123': [0, 1, 2], '132': [0, 2, 1], '213': [1, 0, 2], '231': [1, 2, 0], '312': [2, 0, 1], '321': [2, 1, 0]}
    return {f'B{k}': np.concatenate([np.full(counts[i], i) for i in order]) for k, order in names.items()}


def save(frame, name):
    RES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RES / name, index=False)
    print(f'  -> results/{name} ({len(frame)} rows)', flush=True)


# ---------------------------------------------------------------------------
# Exact evaluation machinery shared by workers
# ---------------------------------------------------------------------------
_W = {}


def _cache_tables(ins, T, tag=None):
    """Build (or reuse) LP-FD / LP-Dual offer masks for an instance and store them on disk."""
    import hashlib
    CACHE.mkdir(parents=True, exist_ok=True)
    key = json.dumps({'ins': ins.to_dict(), 'T': T, 'version': 3}, sort_keys=True)
    tag = 'lp_' + hashlib.sha1(key.encode()).hexdigest()[:12]
    path = CACHE / f'{tag}.npz'
    meta = CACHE / f'{tag}.json'
    if path.exists() and meta.exists() and meta.read_text() == key:
        return path, None
    tables = LPTables(ins, T, workers=WORKERS)
    fd = np.stack([np.stack([tables.fd[h, i] for i in range(ins.m)]) for h in range(1, T + 1)])
    du = np.stack([np.stack([tables.dual[h, i] for i in range(ins.m)]) for h in range(1, T + 1)])
    np.savez(path, fd=fd, dual=du, values=tables.values)
    meta.write_text(key)
    return path, tables.seconds


def _init_worker(ins_dict, T, table_path):
    ins = Instance(**ins_dict)
    L = Lattice(ins)
    data = np.load(table_path)
    _W.clear()
    _W.update(ins=ins, T=T, L=L, fd=data['fd'], dual=data['dual'])
    _, iid, _ = dp_benchmark(ins, T, keep_policy=True)
    _W['offers'] = {
        'Alg1': offer_fixed(ins, L),
        'Greedy': offer_greedy(ins, L),
        'LP-FD': lambda t, i: _W['fd'][T - t - 1, i],
        'LP-Dual': lambda t, i: _W['dual'][T - t - 1, i],
        'IID-DP': offer_dp_table(ins, iid, T),
    }
    _W['lp'] = FluidLP(ins)


def _evaluate_sequence(task):
    """Exact metrics of all policies on one sequence (runs inside a worker)."""
    name, arrivals, details = task
    ins, T, L = _W['ins'], _W['T'], _W['L']
    F = hindsight(ins, arrivals, _W['lp'])
    seq_value, seq_table, _ = dp_benchmark(ins, T, sequence=arrivals, keep_policy=details)
    row = {'sequence': name, 'F': F, 'V_seq': seq_value, **{f'N{i + 1}': int(v) for i, v in enumerate(np.bincount(arrivals, minlength=ins.m))}}
    extra = {}
    offers = dict(_W['offers'])
    if details:
        offers['Seq-DP'] = offer_dp_table(ins, seq_table, T)
    for pol, offer in offers.items():
        res = exact_forward(ins, arrivals, offer, L)
        row[f'ER_{pol}'] = res.expected_revenue
        row[f'oos_{pol}'] = res.expected_oos_slots
        for j in range(ins.n):
            row[f'Ec{j + 1}_{pol}'] = res.expected_remaining[j]
            row[f'Pso{j + 1}_{pol}'] = res.stockout_path[-1, j]
        if details:
            values, pmf = res.revenue_pmf(L)
            mean = float(values @ pmf)
            row[f'sd_{pol}'] = float(np.sqrt(max(pmf @ (values - mean) ** 2, 0)))
            row[f'Pneg_{pol}'] = float(pmf[values > F + 1e-9].sum())
            extra[pol] = {'mean_path': res.mean_path.tolist(), 'stockout_path': res.stockout_path.tolist(),
                          'pmf_values': values.tolist(), 'pmf': pmf.tolist()}
    return row, extra


def exact_suite(ins, T, sequences, tag=None, details=False, workers=EXACT_WORKERS):
    """Evaluate every sequence exactly; returns DataFrame (+ path details if requested)."""
    path, seconds = _cache_tables(ins, T, tag)
    tasks = [(name, np.asarray(seq, dtype=int), details) for name, seq in sequences.items()]
    rows, extras = [], {}
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(workers, len(tasks)), initializer=_init_worker, initargs=(ins.to_dict(), T, str(path))) as pool:
        for row, extra in pool.imap(_evaluate_sequence, tasks, chunksize=1):
            rows.append(row)
            if details:
                extras[row['sequence']] = extra
    return pd.DataFrame(rows), extras, seconds


# ---------------------------------------------------------------------------
# 1) Exact evaluation: 5 datasets, block orders, generated sequences
# ---------------------------------------------------------------------------
def run_exact(n_random=500):
    ins = default_instance()
    T = 100
    start = perf_counter()
    main = datasets()
    blocks = block_orders([50, 30, 20])
    frame, extras, seconds = exact_suite(ins, T, {**main, **blocks}, 'base', details=True)
    save(frame, 'exact_datasets.csv')
    (RES / 'exact_paths.json').write_text(json.dumps(extras), encoding='utf-8')
    # generated sequences under the small-instance setting
    rng = np.random.default_rng(SEED + 1)
    base = np.repeat([0, 1, 2], [50, 30, 20])
    perms = {f'P{k + 1:03d}': rng.permutation(base) for k in range(n_random)}
    rng = np.random.default_rng(SEED + 2)
    iid = {f'I{k + 1:03d}': rng.choice(3, size=T, p=ins.arrival_rates) for k in range(n_random)}
    gen, _, _ = exact_suite(ins, T, {**perms, **iid}, 'base')
    gen['set'] = np.where(gen.sequence.str.startswith('P'), 'permutation', 'iid')
    save(gen, 'exact_generated.csv')
    meta = {'lp_table_seconds': seconds, 'elapsed_seconds': perf_counter() - start, 'n_random': n_random,
            'lattice_states': int(np.prod(ins.capacities + 1))}
    (RES / 'exact_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')


# ---------------------------------------------------------------------------
# 1b) Remaining-count forecast (totals N known): exact, 5 datasets + block orders
# ---------------------------------------------------------------------------
def run_count():
    ins = default_instance()
    T = 100
    L = Lattice(ins)
    rows, paths = [], {}
    for name, seq in {**datasets(), **block_orders([50, 30, 20])}.items():
        start = perf_counter()
        totals = np.bincount(seq, minlength=ins.m)
        seen = np.zeros(ins.m)
        demands, taus = [], []
        for t, i in enumerate(seq):
            seen[int(i)] += 1
            taus.append(T - t - 1)
            demands.append(count_forecast(totals, seen, T - t - 1, ins.arrival_rates))
        values, duals = lp_value_table(ins, demands, taus, want_duals=True, workers=WORKERS)
        row = {'sequence': name}
        for pol, use_duals in [('LP-FD-Count', False), ('LP-Count', True)]:
            masks = [lp_offer_masks(ins, values[t], L, int(seq[t]), taus[t], duals[t] if use_duals else None)
                     for t in range(T)]
            res = exact_forward(ins, seq, lambda t, i, masks=masks: masks[t], L)
            row[f'ER_{pol}'] = res.expected_revenue
            for j in range(ins.n):
                row[f'Ec{j + 1}_{pol}'] = res.expected_remaining[j]
                row[f'Pso{j + 1}_{pol}'] = res.stockout_path[-1, j]
            paths.setdefault(name, {})[pol] = {'mean_path': res.mean_path.tolist(),
                                               'stockout_path': res.stockout_path.tolist()}
        rows.append(row)
        print(f"  {name}: LP-Count={row['ER_LP-Count']:.4f} LP-FD-Count={row['ER_LP-FD-Count']:.4f} "
              f"({perf_counter() - start:.0f}s)", flush=True)
    save(pd.DataFrame(rows), 'exact_count.csv')
    (RES / 'exact_paths_count.json').write_text(json.dumps(paths), encoding='utf-8')


# ---------------------------------------------------------------------------
# 1c) Where is revenue lost? Performance-difference decomposition vs Seq-DP
# ---------------------------------------------------------------------------
def seq_dp_values(ins, seq):
    """V[t] = optimal value at the start of period t (0 based) when the type sequence is known."""
    acts = actions(ins)
    V = [None] * (len(seq) + 1)
    V[len(seq)] = np.zeros(tuple(ins.capacities + 1))
    for t in range(len(seq) - 1, -1, -1):
        V[t] = _bellman_type(ins, V[t + 1], int(seq[t]), acts)[0]
    return V


def product_gains(ins, Vn, i):
    """g_j(c) = p_ij (r_j + V(c - e_j) - V(c)); -inf when product j is sold out."""
    g = np.empty(Vn.shape + (ins.n,))
    for j in range(ins.n):
        gj = ins.probabilities[i, j] * (ins.prices[j] + np.roll(Vn, 1, axis=j) - Vn)
        edge = [slice(None)] * ins.n
        edge[j] = 0
        gj[tuple(edge)] = -np.inf
        g[..., j] = gj
    return g


def loss_decomposition(ins, seq, offer, V, L):
    """Exact identity: V_seq - E R^pi = sum_t E_pi[ max_S Q_t(c,S) - Q_t(c,S^pi) ].
    Returns per-period losses and losses by (type, product-1 decision)."""
    n, K = ins.n, ins.K
    P, r = ins.probabilities, ins.prices
    D = np.zeros(L.shape)
    D[tuple(ins.capacities)] = 1.0
    per_period = np.zeros(len(seq))
    by_cat = {}
    revenue = 0.0
    for t, i in enumerate(np.asarray(seq, dtype=int)):
        mask = offer(t, int(i)) & L.instock
        g = product_gains(ins, V[t + 1], i)
        gpos = np.maximum(g, 0.0)
        best = np.sort(gpos, axis=-1)[..., n - K:].sum(axis=-1) if K > 0 else np.zeros(L.shape)
        q = np.where(mask, g, 0.0).sum(axis=-1)
        gap = np.maximum(best - q, 0.0)
        per_period[t] = float((D * gap).sum())
        # product-1 decision of an optimal action (ties count as "may offer")
        larger = (g[..., 1:] > g[..., :1]).sum(axis=-1)
        opt1 = (g[..., 0] > 1e-12) & (larger < K)
        pol1 = mask[..., 0]
        oversell = pol1 & ~opt1
        overreserve = ~pol1 & opt1 & L.instock[..., 0]
        cats = {'oversell': oversell, 'overreserve': overreserve, 'other': ~(oversell | overreserve)}
        for cat, where in cats.items():
            key = (int(i) + 1, cat)
            by_cat[key] = by_cat.get(key, 0.0) + float((D * gap * where).sum())
        new = D.copy()
        for j in range(n):
            if P[i, j] <= 0:
                continue
            flow = D * (P[i, j] * mask[..., j])
            new -= flow
            src = [slice(None)] * n
            dst = [slice(None)] * n
            src[j], dst[j] = slice(1, None), slice(0, -1)
            new[tuple(dst)] += flow[tuple(src)]
            revenue += r[j] * float(flow.sum())
        D = new
    return per_period, by_cat, revenue


def run_loss():
    base = default_instance()
    T = 100
    rows, period_rows, checks = [], [], []
    scarce = np.floor(0.7 * base.capacities + 0.5).astype(int)
    configs = [('base', base, {**datasets(), 'B213': block_orders([50, 30, 20])['B213']}),
               ('K1', base.with_changes(K=1), datasets()),
               ('theta0.7', base.with_changes(capacities=scarce), datasets())]
    for config, ins, seqs in configs:
        K = ins.K
        path, _ = _cache_tables(ins, T)
        data = np.load(path)
        fd, dual = data['fd'], data['dual']
        L = Lattice(ins)
        _, iid, _ = dp_benchmark(ins, T, keep_policy=True)
        offers = {'Greedy': offer_greedy(ins, L),
                  'LP-FD': lambda t, i, fd=fd: fd[T - t - 1, i],
                  'LP-Dual': lambda t, i, dual=dual: dual[T - t - 1, i],
                  'IID-DP': offer_dp_table(ins, iid, T)}
        for name, seq in seqs.items():
            V = seq_dp_values(ins, seq)
            v_seq = float(V[0][tuple(ins.capacities)])
            for pol, offer in offers.items():
                per_period, by_cat, revenue = loss_decomposition(ins, seq, offer, V, L)
                checks.append(abs(per_period.sum() - (v_seq - revenue)))
                for (i, cat), value in by_cat.items():
                    rows.append({'config': config, 'K': K, 'sequence': name, 'policy': pol, 'type': i, 'category': cat, 'loss': value})
                for t, value in enumerate(per_period):
                    period_rows.append({'config': config, 'K': K, 'sequence': name, 'policy': pol, 't': t + 1, 'loss': value})
                print(f'  {config} {name} {pol}: loss={per_period.sum():.4f} (V_seq-ER={v_seq - revenue:.4f})', flush=True)
    save(pd.DataFrame(rows), 'loss_decomposition.csv')
    save(pd.DataFrame(period_rows), 'loss_by_period.csv')
    (RES / 'loss_meta.json').write_text(json.dumps({'max_identity_error': max(checks)}, indent=2), encoding='utf-8')
    print(f'  identity check: max |sum loss - (V_seq - ER)| = {max(checks):.2e}', flush=True)


# ---------------------------------------------------------------------------
# 2) Monte-Carlo replications on the 5 datasets (online policy objects)
# ---------------------------------------------------------------------------
def make_policy(name, ins, arrivals=None, iid_table=None):
    if name == 'Alg1':
        return FixedTopK(ins)
    if name == 'Greedy':
        return Greedy(ins)
    if name == 'LP-FD':
        return LPLookahead(ins)
    if name == 'LP-Dual':
        return LPDual(ins)
    if name == 'LP-Count':
        return LPDual(ins, forecast='count', totals=np.bincount(arrivals, minlength=ins.m))
    if name == 'IID-DP':
        return DPPolicy(*iid_table)
    raise KeyError(name)


def run_simulation(reps=2000):
    ins = default_instance()
    T = 100
    _, table, sets = dp_benchmark(ins, T, keep_policy=True)
    names = ['Alg1', 'Greedy', 'LP-FD', 'LP-Dual', 'LP-Count', 'IID-DP']
    rows, summary, timing = [], [], []
    traces = RES / 'traces'
    traces.mkdir(parents=True, exist_ok=True)
    lp = FluidLP(ins)
    persistent = {k: make_policy(k, ins, iid_table=(table, sets)) for k in ['Alg1', 'Greedy', 'LP-FD', 'LP-Dual', 'IID-DP']}
    for s, (name, arrivals) in enumerate(datasets().items()):
        F = lp.solve(np.bincount(arrivals, minlength=3), ins.capacities)
        policies = dict(persistent)
        policies['LP-Count'] = make_policy('LP-Count', ins, arrivals)
        U = np.random.default_rng(np.random.SeedSequence([SEED, s])).random((reps, T))
        out = {k: [] for k in names}
        start = perf_counter()
        for b in range(reps):
            for k in names:
                run = simulate(ins, arrivals, policies[k], U[b], trace=(b == 0))
                out[k].append(run)
                if b == 0:
                    pd.DataFrame(run.trace).to_csv(traces / f'trace_{name}_{k}.csv', index=False)
        policies['LP-Count'].close()
        for k in names:
            rev = np.array([r.revenue for r in out[k]])
            rem = np.array([r.remaining for r in out[k]])
            oos = np.array([r.oos_slots for r in out[k]])
            summary.append({'dataset': name, 'policy': k, 'B': reps, 'F': F, 'mean_revenue': rev.mean(),
                            'sd_revenue': rev.std(ddof=1), 'hw95': ci95(rev), 'mean_regret': F - rev.mean(),
                            'share_negative_regret': float(np.mean(rev > F + 1e-9)), 'mean_oos_slots': oos.mean(),
                            **{f'mean_c{j + 1}': rem[:, j].mean() for j in range(3)},
                            **{f'share_soldout{j + 1}': float(np.mean(rem[:, j] == 0)) for j in range(3)}})
            for b, r in enumerate(out[k]):
                rows.append({'dataset': name, 'rep': b, 'policy': k, 'revenue': r.revenue, 'oos_slots': r.oos_slots,
                             **{f'c{j + 1}': int(r.remaining[j]) for j in range(3)}})
        print(f'  {name}: {perf_counter() - start:.1f}s ' + ', '.join(f'{k}={np.mean([r.revenue for r in out[k]]):.3f}' for k in names), flush=True)
    for k in ['LP-FD', 'LP-Dual']:
        p = persistent[k]
        timing.append({'policy': k, 'decisions': p.calls, 'uncached': p.misses, 'lp_solves': p.lp.solves,
                       'median_ms': 1000 * np.median(p.cold_seconds), 'p95_ms': 1000 * np.percentile(p.cold_seconds, 95)})
        p.close()
    lp.close()
    raw = pd.DataFrame(rows)
    save(raw, 'sim_replications.csv')
    save(pd.DataFrame(summary), 'sim_summary.csv')
    # paired differences against LP-FD and Greedy (common random numbers)
    paired = []
    wide = raw.pivot_table(index=['dataset', 'rep'], columns='policy', values='revenue')
    for name in wide.index.get_level_values(0).unique():
        sub = wide.loc[name]
        for a, b in [('LP-FD', 'Greedy'), ('LP-Dual', 'Greedy'), ('LP-Dual', 'LP-FD'), ('LP-Count', 'LP-Dual'),
                     ('Greedy', 'Alg1'), ('IID-DP', 'Greedy')]:
            d = (sub[a] - sub[b]).to_numpy()
            paired.append({'dataset': name, 'a': a, 'b': b, 'mean_diff': d.mean(), 'hw95': ci95(d),
                           'share_equal': float(np.mean(np.abs(d) < 1e-9))})
    save(pd.DataFrame(paired), 'sim_paired.csv')
    save(pd.DataFrame(timing), 'sim_timing.csv')


# ---------------------------------------------------------------------------
# 3) Question 1: the counterexample family
# ---------------------------------------------------------------------------
def run_q1():
    rows = []
    for T in [5, 10, 20, 50, 100, 200]:
        ins = Instance([10, 8, 6], [[.5, .3, .2]], [1, 1, T], [1], 2)
        seq = np.zeros(T, dtype=int)
        L = Lattice(ins)
        tables = LPTables(ins, T, workers=1)
        opt, _, _ = dp_benchmark(ins, T, sequence=seq)
        row = {'T': T, 'DP': opt, 'F': hindsight(ins, seq),
               'closed_alg1': 10 * (1 - .5 ** T) + 8 * (1 - .7 ** T),
               'closed_feasible': 10 * (1 - .5 ** T) + 1.2 * T}
        for pol, offer in [('Alg1', offer_fixed(ins, L)), ('Greedy', offer_greedy(ins, L)),
                           ('LP-FD', tables.offer_fd()), ('LP-Dual', tables.offer_dual())]:
            res = exact_forward(ins, seq, offer, L)
            row[f'ER_{pol}'] = res.expected_revenue
            row[f'oos_{pol}'] = res.expected_oos_slots
        rows.append(row)
        print(f'  T={T}: ' + ', '.join(f'{k}={v:.4f}' for k, v in row.items() if k != 'T'), flush=True)
    save(pd.DataFrame(rows), 'q1_family.csv')


# ---------------------------------------------------------------------------
# 4) Question 2: offline formulations, LP solution and parametric analysis
# ---------------------------------------------------------------------------
def run_offline():
    ins = default_instance()
    P, r, C = ins.probabilities, ins.prices, ins.capacities
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    rows = []
    N = np.array([50, 30, 20])
    for vtype, label in [(GRB.BINARY, 'type-binary'), (GRB.CONTINUOUS, 'type-LP')]:
        model = gp.Model(env=env)
        x = model.addVars(3, 3, vtype=vtype, lb=0, ub=1)
        model.setObjective(gp.quicksum(N[i] * r[j] * P[i, j] * x[i, j] for i in range(3) for j in range(3)), GRB.MAXIMIZE)
        for j in range(3):
            model.addConstr(gp.quicksum(N[i] * P[i, j] * x[i, j] for i in range(3)) <= C[j])
        for i in range(3):
            model.addConstr(x.sum(i, '*') <= ins.K)
        model.optimize()
        X = np.array([[x[i, j].X for j in range(3)] for i in range(3)])
        rows.append({'model': label, 'dataset': 'all', 'value': model.ObjVal, 'variables': model.NumVars,
                     'constraints': model.NumConstrs, 'x': json.dumps(np.round(X, 6).tolist()),
                     'sales': json.dumps(np.round((N[:, None] * P * X).sum(axis=0), 6).tolist())})
        model.dispose()
    for name, seq in datasets().items():
        model = gp.Model(env=env)
        x = model.addVars(len(seq), 3, vtype=GRB.BINARY)
        model.setObjective(gp.quicksum(r[j] * P[seq[t], j] * x[t, j] for t in range(len(seq)) for j in range(3)), GRB.MAXIMIZE)
        for j in range(3):
            model.addConstr(gp.quicksum(P[seq[t], j] * x[t, j] for t in range(len(seq))) <= C[j])
        for t in range(len(seq)):
            model.addConstr(x.sum(t, '*') <= ins.K)
        model.Params.MIPGap = 0
        model.optimize()
        X = np.array([[x[t, j].X for j in range(3)] for t in range(len(seq))])
        sales = np.array([sum(P[seq[t], j] * X[t, j] for t in range(len(seq))) for j in range(3)])
        rows.append({'model': 'customer-binary', 'dataset': name, 'value': model.ObjVal, 'variables': model.NumVars,
                     'constraints': model.NumConstrs, 'x': '', 'sales': json.dumps(np.round(sales, 6).tolist()),
                     'mip_gap': model.MIPGap, 'runtime': model.Runtime})
        model.dispose()
    env.dispose()
    save(pd.DataFrame(rows), 'offline_models.csv')
    lp = FluidLP(ins)
    sol = lp.solve(N, C, details=True)
    (RES / 'lp_solution.json').write_text(json.dumps({k: np.asarray(v).tolist() if not np.isscalar(v) else v
                                                      for k, v in sol.items()}, indent=2), encoding='utf-8')
    par = []
    for theta in np.round(np.arange(0.05, 1.6001, 0.01), 2):
        res = lp.solve(N, theta * C, details=True)
        par.append({'theta': theta, 'F': res['value'], **{f'beta{j + 1}': res['duals'][j] for j in range(3)},
                    **{f'sales{j + 1}': res['sales'][j] for j in range(3)}})
    lp.close()
    save(pd.DataFrame(par), 'lp_parametric.csv')


# ---------------------------------------------------------------------------
# 5) Policy illustration: decisions and reservation thresholds
# ---------------------------------------------------------------------------
def run_policy_map():
    ins = default_instance()
    T = 100
    path, _ = _cache_tables(ins, T, 'base')
    data = np.load(path)
    fd, dual = data['fd'], data['dual']
    _, iid, sets = dp_benchmark(ins, T, keep_policy=True)
    A = action_matrix(ins)
    # worked example: first period and a late scarce state
    rows = []
    lp = FluidLP(ins)
    for (H, c) in [(100, (40, 35, 25)), (40, (18, 20, 15)), (30, (8, 15, 15)), (20, (4, 12, 14))]:
        tau = H - 1
        base = lp.solve(tau * ins.arrival_rates, np.minimum(c, tau), details=True, cold=True)
        for i in range(3):
            delta = []
            for j in range(3):
                less = np.array(c)
                less[j] -= 1
                delta.append(base['value'] - lp.solve(tau * ins.arrival_rates, np.minimum(less, tau)))
            delta = np.round(delta, 6)
            p = ins.probabilities[i]
            rows.append({'H': H, 'c': json.dumps(list(c)), 'type': i + 1, 'F': base['value'],
                         **{f'Delta{j + 1}': delta[j] for j in range(3)},
                         **{f'beta{j + 1}': round(base['duals'][j], 6) for j in range(3)},
                         **{f'score{j + 1}': round(p[j] * (ins.prices[j] - delta[j]), 6) for j in range(3)},
                         'S_FD': '|'.join(str(j + 1) for j in np.flatnonzero(fd[H - 1, i][c])),
                         'S_Dual': '|'.join(str(j + 1) for j in np.flatnonzero(dual[H - 1, i][c])),
                         'S_Greedy': '|'.join(str(j + 1) for j in top_k(ins.values[i], np.array(c), 2, ins.values[i])),
                         'S_IIDDP': '|'.join(str(j + 1) for j in np.flatnonzero(A[iid[H, i][c]]))})
    lp.close()
    save(pd.DataFrame(rows), 'policy_examples.csv')
    # reservation threshold: smallest c1 at which a type-1 / type-2 customer is offered product 1
    thr = []
    for H in range(1, T + 1):
        for c23 in [(35, 25), (20, 12)]:
            row = {'H': H, 'c2': c23[0], 'c3': c23[1]}
            for i in range(2):
                for pol, table in [('FD', fd), ('Dual', dual)]:
                    offered = [table[H - 1, i][c1, c23[0], c23[1], 0] for c1 in range(41)]
                    row[f'withhold_max_{pol}_type{i + 1}'] = max([c1 for c1 in range(1, 41) if not offered[c1]], default=0)
                offered = [A[iid[H, i][c1, c23[0], c23[1]]][0] for c1 in range(41)]
                row[f'withhold_max_IIDDP_type{i + 1}'] = max([c1 for c1 in range(1, 41) if not offered[c1]], default=0)
            thr.append(row)
    save(pd.DataFrame(thr), 'reservation_threshold.csv')


# ---------------------------------------------------------------------------
# 6) Sensitivity: inventory scale, K, (block orders are part of run_exact)
# ---------------------------------------------------------------------------
def run_sensitivity():
    base = default_instance()
    T = 100
    rows = []
    seqs = datasets()
    for theta in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        C = np.floor(theta * base.capacities + 0.5).astype(int)
        ins = base.with_changes(capacities=C)
        frame, _, _ = exact_suite(ins, T, seqs, f'theta_{theta:.1f}')
        frame['theta'] = theta
        frame['C'] = json.dumps(C.tolist())
        rows.append(frame)
        print(f'  theta={theta}: ' + ', '.join(f'{p}={frame[f"ER_{p}"].mean():.3f}' for p in POLICIES) + f', seq={frame.V_seq.mean():.3f}', flush=True)
    save(pd.concat(rows, ignore_index=True), 'sensitivity_inventory.csv')
    rows = []
    for K in [1, 2, 3]:
        ins = base.with_changes(K=K)
        frame, _, _ = exact_suite(ins, T, seqs, f'K_{K}')
        frame['K'] = K
        rows.append(frame)
        print(f'  K={K}: ' + ', '.join(f'{p}={frame[f"ER_{p}"].mean():.3f}' for p in POLICIES), flush=True)
    save(pd.concat(rows, ignore_index=True), 'sensitivity_K.csv')


# ---------------------------------------------------------------------------
# 7) Scaling of the horizon: T and C grow proportionally
# ---------------------------------------------------------------------------
def _scaling_worker(task):
    T, k, reps, seed = task
    base = default_instance()
    scale = T / 100
    ins = base.with_changes(capacities=np.floor(scale * base.capacities + 0.5).astype(int))
    counts = np.floor(T * ins.arrival_rates + 0.5).astype(int)
    counts[0] = T - counts[1:].sum()
    rng = np.random.default_rng([seed, T, k])
    seq = rng.permutation(np.repeat([0, 1, 2], counts))
    U = rng.random((reps, T))
    F = hindsight(ins, seq)
    out = {'T': T, 'k': k, 'F': F, 'reps': reps}
    pols = {'Greedy': Greedy(ins), 'LP-FD': LPLookahead(ins), 'LP-Dual': LPDual(ins)}
    revenue = {}
    for name, p in pols.items():
        runs = [simulate(ins, seq, p, u) for u in U]
        rev = np.array([r.revenue for r in runs])
        revenue[name] = rev
        out[f'mean_{name}'] = rev.mean()
        out[f'sd_{name}'] = rev.std(ddof=1)
        out[f'mean_c1_{name}'] = float(np.mean([r.remaining[0] for r in runs]))
        if hasattr(p, 'close'):
            p.close()
    for a, b in [('LP-Dual', 'Greedy'), ('LP-FD', 'Greedy'), ('LP-Dual', 'LP-FD')]:
        d = revenue[a] - revenue[b]
        out[f'diff_{a}_{b}'] = d.mean()
        out[f'diffsd_{a}_{b}'] = d.std(ddof=1)
    if T <= 200:
        out['V_seq'] = dp_benchmark(ins, T, sequence=seq)[0]
    return out


def run_scaling(reps=300, per_T=10):
    horizons = [25, 50, 100, 200, 400, 800, 1600]
    tasks = [(T, k, reps if T <= 400 else (reps // 2 if T <= 800 else reps // 3), SEED + 7)
             for T in horizons[::-1] for k in range(per_T)]
    with mp.get_context('spawn').Pool(WORKERS) as pool:
        rows = []
        for row in pool.imap_unordered(_scaling_worker, tasks):
            rows.append(row)
            print(f"  T={row['T']} k={row['k']}: F={row['F']:.1f} LP-Dual={row['mean_LP-Dual']:.2f} "
                  f"Greedy={row['mean_Greedy']:.2f} diff={row['diff_LP-Dual_Greedy']:.3f}", flush=True)
    save(pd.DataFrame(rows).sort_values(['T', 'k']), 'scaling.csv')


# ---------------------------------------------------------------------------
# 8) Larger random instances (m, n, T, K)
# ---------------------------------------------------------------------------
def random_instance(m, n, K, T, rho, rng):
    """Dirichlet purchase probabilities with a no-purchase option, integer prices,
    and capacities equal to rho times the expected demand under fixed Top-K."""
    P = np.round(rng.dirichlet(np.full(n + 1, 0.6), size=m)[:, :n] * 0.95, 4)   # 4 decimals, rows < 1
    prices = rng.integers(4, 21, size=n).astype(float)
    rates = rng.dirichlet(np.ones(m) * 2)
    v = P * prices
    top = np.zeros_like(P)
    for i in range(m):
        top[i, np.argsort(-v[i], kind='stable')[:K]] = 1
    demand = T * (rates[:, None] * P * top).sum(axis=0)
    C = np.maximum(1, np.floor(rho * demand + 0.5)).astype(int)
    return Instance(prices, P, C, rates, K)


def _large_worker(task):
    (m, n, T, K), k, rho, reps, seed = task
    rng = np.random.default_rng([seed, m, n, T, K, k, int(100 * rho)])
    ins = random_instance(m, n, K, T, rho, rng)
    seq = rng.choice(m, size=T, p=ins.arrival_rates)
    U = rng.random((reps, T))
    F = hindsight(ins, seq)
    out = {'m': m, 'n': n, 'T': T, 'K': K, 'rho': rho, 'k': k, 'F': F, 'reps': reps,
           'capacity_total': int(ins.capacities.sum())}
    revenue = {}
    for name in ['Alg1', 'Greedy', 'LP-FD', 'LP-Dual']:
        p = make_policy(name, ins)
        delays = []
        if name.startswith('LP'):
            class Timed:
                respects_inventory = True

                def reset(self):
                    p.reset()

                def recommend(self, i, c, H):
                    a = perf_counter()
                    action = p.recommend(i, c, H)
                    delays.append(perf_counter() - a)
                    return action
            runner = Timed()
        else:
            runner = p
        runs = [simulate(ins, seq, runner, u) for u in U]
        rev = np.array([r.revenue for r in runs])
        revenue[name] = rev
        out[f'mean_{name}'] = rev.mean()
        out[f'sd_{name}'] = rev.std(ddof=1)
        out[f'oos_{name}'] = float(np.mean([r.oos_slots for r in runs]))
        if name.startswith('LP'):
            out[f'median_ms_{name}'] = 1000 * float(np.median(delays))
            out[f'p95_ms_{name}'] = 1000 * float(np.percentile(delays, 95))
            out[f'lp_solves_{name}'] = p.lp.solves
            out[f'decisions_{name}'] = len(delays)
            p.close()
    for a, b in [('LP-Dual', 'Greedy'), ('LP-FD', 'Greedy'), ('Greedy', 'Alg1')]:
        d = revenue[a] - revenue[b]          # common random numbers -> paired differences
        out[f'diff_{a}_{b}'] = d.mean()
        out[f'diffsd_{a}_{b}'] = d.std(ddof=1)
    return out


def run_large(reps=60, per_config=8):
    configs = [(2, 5, 50, 1), (3, 6, 100, 2), (5, 10, 100, 3), (10, 20, 200, 5), (20, 30, 300, 4)]
    tasks = []
    for cfg in configs:
        for rho in [0.6, 0.9]:
            for k in range(per_config):
                tasks.append((cfg, k, rho, reps if cfg[0] < 20 else reps // 2, SEED + 11))
    rows = []
    with mp.get_context('spawn').Pool(WORKERS) as pool:
        for row in pool.imap_unordered(_large_worker, tasks):
            rows.append(row)
            print(f"  {row['m']},{row['n']},{row['T']},{row['K']} rho={row['rho']} k={row['k']}: "
                  f"LP-FD/F={row['mean_LP-FD'] / row['F']:.4f} Greedy/F={row['mean_Greedy'] / row['F']:.4f}", flush=True)
    save(pd.DataFrame(rows).sort_values(['m', 'rho', 'k']), 'large_instances.csv')


def run_verify(reps=300):
    """Tables vs online policy objects, decision by decision, and exact vs simulated means."""
    ins = default_instance()
    T = 100
    path, _ = _cache_tables(ins, T)
    data = np.load(path)
    tables = {'LP-FD': data['fd'], 'LP-Dual': data['dual']}
    L = Lattice(ins)
    out = {'reps_per_dataset': reps, 'policies': {}}
    for pol, cls in [('LP-FD', LPLookahead), ('LP-Dual', LPDual)]:
        policy = cls(ins)
        decisions = mismatches = 0
        try:
            for s, (name, seq) in enumerate(datasets().items()):
                U = np.random.default_rng([SEED + 17, s]).random((reps, T))
                for u in U:
                    policy.reset()
                    c = ins.capacities.copy()
                    req = latent_requests(ins, seq, u)
                    for t, i in enumerate(seq):
                        action = policy.recommend(int(i), c.copy(), T - t)
                        table_action = tuple(np.flatnonzero(tables[pol][T - t - 1, int(i)][tuple(c)]))
                        decisions += 1
                        mismatches += int(action != table_action)
                        j = req[t]
                        if j in action and c[j] > 0:
                            c[j] -= 1
        finally:
            policy.close()
        out['policies'][pol] = {'decisions': decisions, 'mismatches': mismatches}
        print(f'  {pol}: {mismatches} mismatches in {decisions} decisions', flush=True)
    seq = datasets()['D1']
    start = perf_counter()
    for _ in range(5):
        exact_forward(ins, seq, lambda t, i: tables['LP-Dual'][T - t - 1, i], L)
    out['exact_forward_seconds'] = (perf_counter() - start) / 5
    start = perf_counter()
    dp_benchmark(ins, T, keep_policy=True)
    out['iid_dp_seconds'] = perf_counter() - start
    (RES / 'verification.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out), flush=True)


def write_environment():
    env = {'python': platform.python_version(), 'platform': platform.platform(), 'gurobi': '.'.join(map(str, gp.gurobi.version())),
           'numpy': np.__version__, 'pandas': pd.__version__, 'workers': WORKERS, 'seed': SEED}
    (RES / 'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tasks', nargs='+', choices=['all', 'exact', 'count', 'loss', 'simulate', 'verify', 'q1',
                                                     'offline', 'policy-map', 'sensitivity', 'scaling', 'large'])
    parser.add_argument('--reps', type=int, default=2000)
    parser.add_argument('--random', type=int, default=500)
    args = parser.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    order = ['exact', 'count', 'loss', 'simulate', 'verify', 'q1', 'offline', 'policy-map', 'sensitivity', 'scaling',
             'large']
    tasks = order if 'all' in args.tasks else args.tasks
    write_environment()
    for task in tasks:
        start = perf_counter()
        print(f'[{task}]', flush=True)
        {'exact': lambda: run_exact(args.random), 'count': run_count, 'loss': run_loss, 'verify': run_verify,
         'simulate': lambda: run_simulation(args.reps), 'q1': run_q1,
         'offline': run_offline, 'policy-map': run_policy_map, 'sensitivity': run_sensitivity,
         'scaling': run_scaling, 'large': run_large}[task]()
        print(f'[{task}] done in {perf_counter() - start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
