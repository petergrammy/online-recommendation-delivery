"""Online Top-K recommendation under inventory constraints (gurobipy).

Model
-----
T periods, one customer per period. A type-i customer has a latent first
choice J_t with Pr(J_t = j) = P[i, j] (the remaining mass means "no purchase
intention"). The customer buys J_t iff J_t is recommended and in stock, so the
purchase probability of an offered set S is sum_{j in S} P[i, j]. Rows of P
must therefore sum to at most one.

Contents
--------
Instance        problem data r, P, C, lambda, K (validated)
FluidLP         reusable continuous LP  F(N, c)  solved with gurobipy
Policies        FixedTopK  (Algorithm 1 of the assignment, ignores inventory)
                Greedy     (stock-aware, ranks p_ij r_j)
                LPDual     (main policy, LP dual prices of the inventory rows)
                LPLookahead (comparison policy, finite differences of F)
simulate        Monte-Carlo run with common random numbers
dp_benchmark    exact stochastic DP (IID arrivals or a known type sequence)
exact_forward   exact distribution of the inventory path for table policies
lp_value_table  F on the whole inventory lattice (multiprocessing, for exact
                evaluation of the LP policies without Monte-Carlo error)

All indices are zero based inside Python and one based in CSV files/paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from time import perf_counter
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB

TOL_SCORE = 1e-9      # scores are compared after rounding to 9 decimals
TOL_POSITIVE = 1e-10  # a product with p_ij r_j <= this is never recommended


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------
@dataclass
class Instance:
    prices: np.ndarray          # r_j, shape (n,)
    probabilities: np.ndarray   # P[i, j], shape (m, n), row sums <= 1
    capacities: np.ndarray      # C_j, nonnegative integers, shape (n,)
    arrival_rates: np.ndarray   # lambda_i, shape (m,), sums to one
    K: int = 2                  # at most K recommended products per customer

    def __post_init__(self):
        self.prices = np.asarray(self.prices, dtype=float)
        self.probabilities = np.asarray(self.probabilities, dtype=float)
        raw = np.asarray(self.capacities, dtype=float)
        self.arrival_rates = np.asarray(self.arrival_rates, dtype=float)
        if raw.ndim != 1 or not np.all(np.isfinite(raw)) or np.any(raw < 0) or np.any(raw != np.floor(raw)):
            raise ValueError('Capacities must be nonnegative integers.')
        self.capacities = raw.astype(int)
        if self.probabilities.ndim != 2:
            raise ValueError('P must be a matrix.')
        m, n = self.probabilities.shape
        if m == 0 or n == 0 or self.prices.shape != (n,) or raw.shape != (n,) or self.arrival_rates.shape != (m,):
            raise ValueError('Inconsistent dimensions.')
        if not isinstance(self.K, (int, np.integer)) or self.K < 0:
            raise ValueError('K must be a nonnegative integer.')
        if not all(np.all(np.isfinite(a)) for a in (self.prices, self.probabilities, self.arrival_rates)):
            raise ValueError('Inputs must be finite.')
        if np.any(self.prices < 0) or np.any(self.probabilities < 0) or np.any(self.probabilities.sum(axis=1) > 1 + 1e-12):
            raise ValueError('Use nonnegative prices and probabilities with row sums <= 1.')
        if np.any(self.arrival_rates < 0) or not np.isclose(self.arrival_rates.sum(), 1, atol=1e-12, rtol=0):
            raise ValueError('Arrival rates must sum to one.')
        self.K = int(min(self.K, n))

    @property
    def m(self):
        return len(self.arrival_rates)

    @property
    def n(self):
        return len(self.prices)

    @property
    def values(self):
        """Immediate expected revenue v_ij = p_ij r_j."""
        return self.probabilities * self.prices

    def with_changes(self, **kw):
        data = dict(prices=self.prices, probabilities=self.probabilities, capacities=self.capacities,
                    arrival_rates=self.arrival_rates, K=self.K)
        data.update(kw)
        return Instance(**data)

    def to_dict(self):
        return {'prices': self.prices.tolist(), 'probabilities': self.probabilities.tolist(),
                'capacities': self.capacities.tolist(), 'arrival_rates': self.arrival_rates.tolist(), 'K': self.K}

    @staticmethod
    def from_json(path):
        with open(path, encoding='utf-8') as f:
            return Instance(**json.load(f))


def default_instance():
    """The small instance of the assignment (T = 100)."""
    return Instance([10, 8, 6], [[.6, .2, .1], [.4, .5, 0], [.1, .2, .6]], [40, 35, 25], [.5, .3, .2], 2)


# ---------------------------------------------------------------------------
# Fluid LP
# ---------------------------------------------------------------------------
class FluidLP:
    """Continuous LP; y[i, j] is the expected number of times j is shown to type i.

        F(N, c) = max  sum_ij r_j p_ij y_ij
                  s.t. sum_i p_ij y_ij <= c_j          (inventory, dual beta_j)
                       sum_j y_ij      <= K N_i        (display slots)
                       0 <= y_ij <= N_i

    The model is built once; every solve only changes right-hand sides and
    variable upper bounds, so the dual simplex can reuse the previous basis.
    cold=True discards the basis first, which makes the returned dual vector
    a deterministic function of (N, c) even when the LP is degenerate.
    """

    def __init__(self, instance: Instance):
        self.ins = instance
        self.env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', 0)
        self.env.start()
        self.model = gp.Model('fluid_recommendation', env=self.env)
        self.model.Params.Threads = 1
        self.model.Params.Method = 1           # dual simplex
        self.model.Params.FeasibilityTol = 1e-9
        self.model.Params.OptimalityTol = 1e-9
        m, n = instance.m, instance.n
        P, r = instance.probabilities, instance.prices
        self.y = self.model.addVars(m, n, lb=0, name='y')
        self.flat = [self.y[i, j] for i in range(m) for j in range(n)]
        self.stock = [self.model.addConstr(gp.quicksum(P[i, j] * self.y[i, j] for i in range(m)) <= 0, name=f'stock_{j}')
                      for j in range(n)]
        self.slots = [self.model.addConstr(gp.quicksum(self.y[i, j] for j in range(n)) <= 0, name=f'slots_{i}')
                      for i in range(m)]
        self.model.setObjective(gp.quicksum(r[j] * P[i, j] * self.y[i, j] for i in range(m) for j in range(n)), GRB.MAXIMIZE)
        self.solves = 0
        self.solve_seconds = 0.0

    def solve(self, demand, inventory, details=False, cold=False):
        d = np.asarray(demand, dtype=float)
        c = np.asarray(inventory, dtype=float)
        if d.shape != (self.ins.m,) or c.shape != (self.ins.n,) or not np.all(np.isfinite(d)) \
                or not np.all(np.isfinite(c)) or np.any(d < 0) or np.any(c < 0):
            raise ValueError('Invalid LP right hand side.')
        self.model.setAttr('UB', self.flat, np.repeat(d, self.ins.n).tolist())
        self.model.setAttr('RHS', self.stock, c.tolist())
        self.model.setAttr('RHS', self.slots, (self.ins.K * d).tolist())
        if cold:
            self.model.reset()
        start = perf_counter()
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            # y = 0 is always feasible, so a non-optimal status is numerical:
            # retry from scratch, then with the most careful numerics.
            self.model.reset()
            self.model.optimize()
            if self.model.Status != GRB.OPTIMAL:
                self.model.Params.NumericFocus = 3
                self.model.reset()
                self.model.optimize()
                self.model.Params.NumericFocus = 0
        self.solve_seconds += perf_counter() - start
        self.solves += 1
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f'LP failed: Gurobi status {self.model.Status}')
        value = self.model.ObjVal
        if details:
            y = np.array(self.model.getAttr('X', self.flat)).reshape(self.ins.m, self.ins.n)
            return {'value': value, 'y': y, 'sales': (y * self.ins.probabilities).sum(axis=0),
                    'duals': np.array(self.model.getAttr('Pi', self.stock)),
                    'slot_duals': np.array(self.model.getAttr('Pi', self.slots))}
        return value

    def close(self):
        self.model.dispose()
        self.env.dispose()


def truncate(c, tau):
    """At most one unit is sold per period, so stock above tau is never binding."""
    return tuple(min(int(v), int(tau)) for v in c)


def round9(x):
    return np.round(x, 9)


# ---------------------------------------------------------------------------
# Ranking rule shared by all policies (online and table versions)
# ---------------------------------------------------------------------------
def top_k(scores, inventory, K, immediate=None):
    """Recommend at most K in-stock products with the largest scores.

    Eligible products: in stock, rounded score >= -1e-9 and p_ij r_j > 0.
    A zero score ties with "leave the slot empty"; the product is kept because
    it still earns its immediate revenue when the LP values a unit at its full
    price. Ties: larger score, then larger immediate revenue, then smaller index.
    """
    scores = np.asarray(scores, dtype=float)
    immediate = scores if immediate is None else np.asarray(immediate, dtype=float)
    key = round9(scores)
    eligible = [j for j in range(len(scores)) if inventory[j] > 0 and key[j] >= -TOL_SCORE and immediate[j] > TOL_POSITIVE]
    eligible.sort(key=lambda j: (-key[j], -immediate[j], j))
    return tuple(sorted(eligible[:K]))


def top_k_mask(scores, immediate, instock, K):
    """Vectorized top_k: scores (..., n), immediate (n,), instock (..., n) -> offer mask (..., n)."""
    key = round9(scores)
    imm = np.broadcast_to(np.asarray(immediate, dtype=float), key.shape)
    eligible = instock & (key >= -TOL_SCORE) & (imm > TOL_POSITIVE)
    n = key.shape[-1]
    index = np.broadcast_to(np.arange(n), key.shape)
    order = np.lexsort((index, -imm, -key, ~eligible), axis=-1)  # last key = primary
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(n), key.shape).copy(), axis=-1)
    return eligible & (rank < K)


# ---------------------------------------------------------------------------
# Policies: recommend(customer, inventory, remaining) -> tuple of products
# remaining counts the current period, i.e. remaining = T - t + 1.
# ---------------------------------------------------------------------------
class FixedTopK:
    """Algorithm 1 of the assignment: fixed Top-K list by p_ij r_j, ignores inventory."""
    name = 'Alg1'
    respects_inventory = False

    def __init__(self, instance):
        self.ins = instance
        v = instance.values
        self.sets = [tuple(sorted(sorted(range(instance.n), key=lambda j: (-v[i, j], j))[:instance.K]))
                     for i in range(instance.m)]

    def recommend(self, customer, inventory, remaining):
        return self.sets[customer]


class Greedy:
    """Stock-aware myopic policy: top-K in-stock products by p_ij r_j."""
    name = 'Greedy'
    respects_inventory = True

    def __init__(self, instance):
        self.ins = instance

    def recommend(self, customer, inventory, remaining):
        v = self.ins.values[customer]
        return top_k(v, inventory, self.ins.K, v)


def count_forecast(totals, seen, tau, rates):
    """Remaining-count forecast: max(N - seen, 0) rescaled to tau future customers."""
    if tau <= 0:
        return np.zeros(len(rates))
    rest = np.maximum(np.asarray(totals, float) - np.asarray(seen, float), 0)
    if rest.sum() <= 0:
        return tau * np.asarray(rates, float)
    return rest * (tau / rest.sum())


class LPLookahead:
    """LP-FD: one-step lookahead with the fluid LP as continuation value.

    forecast='rate'  : future type counts h * lambda (uses only lambda and T).
    forecast='count' : future type counts N - (types already observed), for
                       settings where the totals N of the horizon are known.
    The opportunity cost of one unit of product j is the finite difference
    Delta_j = F(d, c) - F(d, c - e_j); the score is p_ij (r_j - Delta_j).
    """
    name = 'LP-FD'
    respects_inventory = True

    def __init__(self, instance, forecast='rate', totals=None, rates=None):
        if forecast not in ('rate', 'count'):
            raise ValueError("forecast must be 'rate' or 'count'")
        self.ins = instance
        self.forecast = forecast
        self.rates = instance.arrival_rates if rates is None else np.asarray(rates, dtype=float)
        if self.rates.shape != (instance.m,) or np.any(self.rates < 0) or not np.isclose(self.rates.sum(), 1):
            raise ValueError('Invalid forecast rates.')
        self.totals = None if totals is None else np.asarray(totals, dtype=float)
        if forecast == 'count' and self.totals is None:
            raise ValueError("forecast='count' needs the totals N.")
        self.lp = FluidLP(instance)
        self.values = {}
        self.action_cache = {}
        self.calls = 0
        self.misses = 0
        self.cold_seconds = []
        self.reset()

    def reset(self):
        """Call before each new arrival sequence (only the count forecast keeps memory)."""
        self.seen = np.zeros(self.ins.m)

    def demand(self, tau):
        if self.forecast == 'rate':
            return tau * self.rates, ('rate', tau)
        d = count_forecast(self.totals, self.seen, tau, self.rates)
        return d, tuple(round9(d).tolist())

    def fluid_value(self, d, d_key, tau, c):
        if tau <= 0 or not any(c):
            return 0.0
        key = (d_key,) + truncate(c, tau)
        if key not in self.values:
            self.values[key] = float(round9(self.lp.solve(d, key[1:])))
        return self.values[key]

    def opportunity_costs(self, inventory, tau, d, d_key):
        base = self.fluid_value(d, d_key, tau, inventory)
        delta = np.zeros(self.ins.n)
        for j in range(self.ins.n):
            if inventory[j] > 0:
                less = np.array(inventory, copy=True)
                less[j] -= 1
                delta[j] = base - self.fluid_value(d, d_key, tau, less)
        return delta

    def observe_arrival(self, customer):
        """Advance the observed prefix once for this period."""
        if self.forecast == 'count':
            self.seen[customer] += 1

    def recommend(self, customer, inventory, remaining):
        """Online entry point: observe one customer, then choose their offer."""
        self.observe_arrival(customer)
        return self.recommend_after_arrival(customer, inventory, remaining)

    def recommend_after_arrival(self, customer, inventory, remaining):
        """Choose an offer at this inventory with the current prefix held fixed."""
        self.calls += 1
        tau = remaining - 1                           # customers after the current one
        d, d_key = self.demand(tau)
        key = (customer, d_key) + tuple(int(v) for v in inventory)
        if key in self.action_cache:
            return self.action_cache[key]
        start = perf_counter()
        self.misses += 1
        delta = self.opportunity_costs(inventory, tau, d, d_key)
        p, r = self.ins.probabilities[customer], self.ins.prices
        action = top_k(p * (r - delta), inventory, self.ins.K, p * r)
        self.action_cache[key] = action
        self.cold_seconds.append(perf_counter() - start)
        return action

    def close(self):
        self.lp.close()


class LPDual(LPLookahead):
    """Main policy: use the dual price of each inventory constraint as its opportunity cost.

    One LP per decision (instead of n + 1). The LP is cold-started so that the
    dual vector is a deterministic function of the state at degenerate points.
    """
    name = 'LP-Dual'

    def duals(self, d, d_key, tau, c):
        if tau <= 0 or not any(c):
            return np.zeros(self.ins.n)
        key = ('dual', d_key) + truncate(c, tau)
        if key not in self.values:
            self.values[key] = round9(self.lp.solve(d, key[2:], details=True, cold=True)['duals'])
        return self.values[key]

    def opportunity_costs(self, inventory, tau, d, d_key):
        return self.duals(d, d_key, tau, inventory)


class DPPolicy:
    """Look-up table produced by dp_benchmark(keep_policy=True)."""
    name = 'IID-DP'
    respects_inventory = True

    def __init__(self, table, action_sets):
        self.table, self.action_sets = table, action_sets

    def recommend(self, customer, inventory, remaining):
        return self.action_sets[int(self.table[(remaining, customer) + tuple(int(v) for v in inventory)])]


# ---------------------------------------------------------------------------
# Simulation with common random numbers
# ---------------------------------------------------------------------------
def latent_requests(instance, arrivals, uniforms):
    """Latent first choice J_t in {0..n-1} or n (no purchase intention), independent of the offer."""
    return (np.asarray(uniforms)[:, None] >= np.cumsum(instance.probabilities[arrivals], axis=1)).sum(axis=1)


@dataclass
class Run:
    revenue: float
    remaining: np.ndarray
    oos_slots: int = 0                       # recommended products that were out of stock
    trace: list = field(default_factory=list)


def simulate(instance, arrivals, policy, uniforms, trace=False):
    """One sample path. Policies with respects_inventory=True are checked every period."""
    arrivals = np.asarray(arrivals, dtype=int)
    if arrivals.ndim != 1 or np.any(arrivals < 0) or np.any(arrivals >= instance.m):
        raise ValueError('Invalid customer type.')
    uniforms = np.asarray(uniforms, dtype=float)
    if uniforms.shape != arrivals.shape or np.any(uniforms < 0) or np.any(uniforms >= 1):
        raise ValueError('Uniform draws must be in [0,1).')
    strict = getattr(policy, 'respects_inventory', True)
    if hasattr(policy, 'reset'):
        policy.reset()
    c = instance.capacities.copy()
    requests = latent_requests(instance, arrivals, uniforms)
    revenue, oos, records = 0.0, 0, []
    T = len(arrivals)
    for t, i in enumerate(arrivals):
        before = c.copy()
        action = policy.recommend(int(i), c.copy(), T - t)
        if len(action) > instance.K or len(action) != len(set(action)) or any(j < 0 or j >= instance.n for j in action):
            raise AssertionError('Invalid recommendation: duplicate or unknown product, or K exceeded.')
        sold_out = sum(1 for j in action if c[j] <= 0)
        if strict and sold_out:
            raise AssertionError('Infeasible recommendation: out-of-stock product.')
        oos += sold_out
        j = int(requests[t])
        bought = j if (j in action and j < instance.n and c[j] > 0) else -1
        if bought >= 0:
            c[bought] -= 1
            revenue += instance.prices[bought]
        if trace:
            records.append({'t': t + 1, 'customer_type': int(i) + 1,
                            'recommendation': '|'.join(str(k + 1) for k in action),
                            'latent_product': j + 1 if j < instance.n else 0, 'purchase': bought + 1,
                            'revenue_cumulative': revenue,
                            **{f'before_{k+1}': int(before[k]) for k in range(instance.n)},
                            **{f'after_{k+1}': int(c[k]) for k in range(instance.n)}})
    return Run(revenue, c, oos, records)


# ---------------------------------------------------------------------------
# Exact stochastic dynamic programming
# ---------------------------------------------------------------------------
def actions(instance):
    """All subsets with at most K products, ordered by size then lexicographically."""
    return [s for k in range(instance.K + 1) for s in combinations(range(instance.n), k)]


def action_matrix(instance):
    sets = actions(instance)
    A = np.zeros((len(sets), instance.n), dtype=bool)
    for a, s in enumerate(sets):
        A[a, list(s)] = True
    return A


def _bellman_type(instance, V, customer, all_actions):
    """max over feasible S of sum_{j in S} p_ij (r_j + V(c - e_j) - V(c)), vectorized over the lattice."""
    gains = []
    for j in range(instance.n):
        shifted = np.roll(V, 1, axis=j)
        diff = instance.probabilities[customer, j] * (instance.prices[j] + shifted - V)
        boundary = [slice(None)] * instance.n
        boundary[j] = 0
        diff[tuple(boundary)] = -np.inf          # product j sold out
        gains.append(diff)
    best = np.zeros_like(V)
    chosen = np.zeros(V.shape, dtype=np.uint16)
    for a, s in enumerate(all_actions[1:], 1):
        candidate = sum(gains[j] for j in s)
        improve = candidate > best + 1e-10
        chosen[improve] = a
        best = np.maximum(best, candidate)
    return V + best, chosen


def dp_benchmark(instance, horizon, sequence=None, keep_policy=False, max_states=60_000_000):
    """Exact DP. sequence=None: IID arrivals with rates lambda (online optimal policy).
    sequence given: the type sequence is known in advance, purchases stay random
    (diagnostic benchmark V^seq, never used by an online policy).
    Returns (value at full inventory, policy table or None, action list)."""
    shape = tuple(instance.capacities + 1)
    states = int(np.prod(shape))
    if states * (horizon + 1) > max_states:
        raise ValueError('DP state budget exceeded; use the LP policy.')
    all_actions = actions(instance)
    if len(all_actions) > 65535:
        raise ValueError('Too many DP actions.')
    if sequence is not None and len(sequence) != horizon:
        raise ValueError('Sequence length differs from horizon.')
    V = np.zeros(shape)
    policies = np.zeros((horizon + 1, instance.m) + shape, dtype=np.uint16) if keep_policy else None
    for h in range(1, horizon + 1):
        if sequence is None:
            current = np.zeros(shape)
            for i, rate in enumerate(instance.arrival_rates):
                value, action = _bellman_type(instance, V, i, all_actions)
                current += rate * value
                if keep_policy:
                    policies[h, i] = action
            V = current
        else:
            i = int(sequence[horizon - h])
            V, action = _bellman_type(instance, V, i, all_actions)
            if keep_policy:
                policies[h, i] = action
    return float(V[tuple(instance.capacities)]), policies, all_actions


def exact_table_policy_value(instance, arrivals, table=None):
    """Backward evaluation of a table policy (table=None: Greedy). Used to cross-check exact_forward."""
    shape = tuple(instance.capacities + 1)
    V = np.zeros(shape)
    aset = actions(instance)
    greedy_actions = [_bellman_type(instance, np.zeros(shape), i, aset)[1] for i in range(instance.m)] if table is None else None
    for t in range(len(arrivals) - 1, -1, -1):
        i = int(arrivals[t])
        selected = greedy_actions[i] if table is None else table[len(arrivals) - t, i]
        gain = np.zeros(shape)
        for a, s in enumerate(aset[1:], 1):
            mask = selected == a
            for j in s:
                delta = instance.probabilities[i, j] * (instance.prices[j] + np.roll(V, 1, axis=j) - V)
                gain += np.where(mask, delta, 0)
        V = V + gain
    return float(V[tuple(instance.capacities)])


def exact_policy_value(instance, arrivals, policy):
    """Evaluate an inventory policy on a fixed sequence of customer types.

    Policies with arrival memory expose observe_arrival and recommend_after_arrival:
    all inventory states in a period must share the same observed prefix. For
    policies without arrival memory, recommend depends only on the state supplied.
    """
    if hasattr(policy, 'reset'):
        policy.reset()
    observe = getattr(policy, 'observe_arrival', None)
    recommend = policy.recommend if observe is None else policy.recommend_after_arrival
    distribution = {tuple(instance.capacities): 1.0}
    total = 0.0
    for t, i in enumerate(arrivals):
        if observe is not None:
            observe(int(i))
        nxt = {}
        for c, mass in distribution.items():
            s = recommend(int(i), np.asarray(c), len(arrivals) - t)
            p0 = 1.0
            for j in s:
                if c[j] <= 0:
                    continue
                p = instance.probabilities[i, j]
                p0 -= p
                less = list(c)
                less[j] -= 1
                less = tuple(less)
                nxt[less] = nxt.get(less, 0.0) + mass * p
                total += mass * p * instance.prices[j]
            nxt[c] = nxt.get(c, 0.0) + mass * p0
        distribution = {c: p for c, p in nxt.items() if p > 0}
    return total


# ---------------------------------------------------------------------------
# Exact forward evaluation on the inventory lattice
# ---------------------------------------------------------------------------
class Lattice:
    """Inventory states c with 0 <= c_j <= C_j, stored as an n-dimensional array."""

    def __init__(self, instance):
        self.ins = instance
        self.shape = tuple(int(v) + 1 for v in instance.capacities)
        self.grid = np.indices(self.shape)                       # (n, *shape)
        self.instock = np.stack([g > 0 for g in self.grid], axis=-1)   # (*shape, n)
        sold = instance.capacities.reshape((-1,) + (1,) * instance.n) - self.grid
        self.revenue = np.tensordot(instance.prices, sold, axes=1)  # revenue if the horizon ends at c

    def offer_from_table(self, table_slice, A):
        return A[table_slice]


@dataclass
class ExactResult:
    expected_revenue: float
    final_distribution: np.ndarray
    expected_remaining: np.ndarray
    expected_oos_slots: float
    mean_path: np.ndarray          # (T+1, n): E[c_t]
    stockout_path: np.ndarray      # (T+1, n): Pr(c_tj = 0)

    def revenue_pmf(self, lattice):
        values, inverse = np.unique(np.round(lattice.revenue.ravel(), 9), return_inverse=True)
        return values, np.bincount(inverse, weights=self.final_distribution.ravel())


def exact_forward(instance, arrivals, offer_fn, lattice=None):
    """Exact law of the inventory process for a policy given as offer masks.

    offer_fn(t, i) returns a boolean array (*lattice, n): product j is shown at
    state c in period t (0 based) to a type-i customer. Products shown while
    sold out (Algorithm 1) waste the slot and are counted in expected_oos_slots.
    Revenue is a function of the final inventory, so the final distribution
    gives the exact distribution of R_T.
    """
    L = lattice or Lattice(instance)
    n = instance.n
    P, r = instance.probabilities, instance.prices
    D = np.zeros(L.shape)
    D[tuple(instance.capacities)] = 1.0
    revenue, oos = 0.0, 0.0
    mean_path = [instance.capacities.astype(float)]
    stockout_path = [(instance.capacities == 0).astype(float)]
    for t, i in enumerate(np.asarray(arrivals, dtype=int)):
        offer = offer_fn(t, int(i))
        new = D.copy()
        for j in range(n):
            shown = offer[..., j]
            oos += float((D * (shown & ~L.instock[..., j])).sum())
            if P[i, j] <= 0:
                continue
            flow = D * (P[i, j] * (shown & L.instock[..., j]))
            new -= flow
            src = [slice(None)] * n
            dst = [slice(None)] * n
            src[j], dst[j] = slice(1, None), slice(0, -1)
            new[tuple(dst)] += flow[tuple(src)]
            revenue += r[j] * float(flow.sum())
        D = new
        mean_path.append([float((D * L.grid[j]).sum()) for j in range(n)])
        stockout_path.append([float(D[tuple(slice(0, 1) if k == j else slice(None) for k in range(n))].sum()) for j in range(n)])
    expected_remaining = np.array([(D * L.grid[j]).sum() for j in range(n)])
    return ExactResult(revenue, D, expected_remaining, oos, np.array(mean_path), np.array(stockout_path))


def offer_fixed(instance, lattice):
    """Algorithm 1: constant offer per type, regardless of the state."""
    sets = FixedTopK(instance).sets
    masks = []
    for s in sets:
        mask = np.zeros(lattice.shape + (instance.n,), dtype=bool)
        mask[..., list(s)] = True
        masks.append(mask)
    return lambda t, i: masks[i]


def offer_greedy(instance, lattice):
    v = instance.values
    masks = [top_k_mask(np.broadcast_to(v[i], lattice.shape + (instance.n,)), v[i], lattice.instock, instance.K)
             for i in range(instance.m)]
    return lambda t, i: masks[i]


def offer_dp_table(instance, table, horizon):
    """IID-DP (table[h, i]) or sequence DP; h = remaining periods = horizon - t."""
    A = action_matrix(instance)
    return lambda t, i: A[table[horizon - t, i]]


def lp_offer_masks(instance, values, lattice, customer, tau_index, duals=None):
    """Offer mask of LP-FD (values) or LP-Dual (duals) for one continuation slice.

    values: array (*lattice) with F(d, c) for the continuation after the current customer.
    """
    n = instance.n
    p, r = instance.probabilities[customer], instance.prices
    if duals is None:
        delta = np.zeros(lattice.shape + (n,))
        for j in range(n):
            less = np.roll(values, 1, axis=j)
            dj = values - less
            boundary = [slice(None)] * n
            boundary[j] = 0
            dj[tuple(boundary)] = 0.0
            delta[..., j] = dj
    else:
        delta = duals
    return top_k_mask(p * (r - delta), p * r, lattice.instock, instance.K)


# ---------------------------------------------------------------------------
# LP values on the whole lattice (for exact evaluation of LP-FD / LP-Dual)
# ---------------------------------------------------------------------------
def _lp_rows_worker(payload):
    """Solve F(d_s, c) for all truncated states c of the rows given in payload."""
    ins_dict, rows, want_duals = payload
    ins = Instance(**ins_dict)
    lp = FluidLP(ins)
    out = []
    try:
        for s, d, tau in rows:
            dims = [min(int(C), int(tau)) + 1 for C in ins.capacities]
            vals = np.zeros(dims)
            dual = np.zeros(dims + [ins.n]) if want_duals else None
            if tau > 0:
                for c in product(*[range(k) for k in dims]):
                    if not any(c):
                        continue
                    if want_duals:
                        res = lp.solve(d, c, details=True, cold=True)
                        vals[c] = res['value']
                        dual[c] = res['duals']
                    else:
                        vals[c] = lp.solve(d, c)
            out.append((s, round9(vals), None if dual is None else round9(dual)))
    finally:
        lp.close()
    return out


def lp_value_table(instance, demands, taus, want_duals=False, workers=None):
    """F(d_s, c) and optionally the inventory duals for every lattice state c.

    demands: (S, m) future demand vectors; taus: (S,) number of future periods
    (truncation level). Returns values (S, *lattice) and duals (S, *lattice, n).
    """
    import multiprocessing as mp
    demands = np.asarray(demands, dtype=float)
    taus = np.asarray(taus, dtype=int)
    shape = tuple(int(v) + 1 for v in instance.capacities)
    rows = sorted(((s, demands[s], int(taus[s])) for s in range(len(taus))),
                  key=lambda row: -np.prod([min(C, row[2]) + 1 for C in instance.capacities]))
    workers = workers or max(1, min(14, mp.cpu_count() - 2))
    batches = [rows[k::workers * 4] for k in range(workers * 4)]
    payloads = [(instance.to_dict(), b, want_duals) for b in batches if b]
    values = np.zeros((len(taus),) + shape)
    duals = np.zeros((len(taus),) + shape + (instance.n,)) if want_duals else None
    grid = np.indices(shape)
    if workers == 1:
        results = map(_lp_rows_worker, payloads)
        pool = None
    else:
        pool = mp.get_context('spawn').Pool(workers)
        results = pool.imap_unordered(_lp_rows_worker, payloads)
    try:
        for chunk in results:
            for s, vals, dual in chunk:
                tau = int(taus[s])
                idx = tuple(np.minimum(grid[j], tau) for j in range(instance.n))
                values[s] = vals[idx]
                if want_duals:
                    duals[s] = dual[idx]
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return values, duals


class LPTables:
    """Offer masks of LP-FD and LP-Dual (rate forecast) for every (h, i, c)."""

    def __init__(self, instance, horizon, workers=None, duals=True):
        self.ins, self.T = instance, horizon
        self.lattice = Lattice(instance)
        taus = np.arange(horizon)                              # tau = h - 1 = 0..T-1
        demands = taus[:, None] * instance.arrival_rates[None, :]
        start = perf_counter()
        self.values, self.duals = lp_value_table(instance, demands, taus, want_duals=duals, workers=workers)
        self.seconds = perf_counter() - start
        self.fd = {}
        self.dual = {}
        for h in range(1, horizon + 1):
            for i in range(instance.m):
                self.fd[h, i] = lp_offer_masks(instance, self.values[h - 1], self.lattice, i, h - 1)
                if duals:
                    self.dual[h, i] = lp_offer_masks(instance, None, self.lattice, i, h - 1, self.duals[h - 1])

    def offer_fd(self):
        return lambda t, i: self.fd[self.T - t, i]

    def offer_dual(self):
        return lambda t, i: self.dual[self.T - t, i]


def count_lp_offer(instance, arrivals, totals, lattice=None, workers=None, duals=False):
    """Exact offers with known type totals, using dual prices or finite differences."""
    L = lattice or Lattice(instance)
    T = len(arrivals)
    seen = np.zeros(instance.m)
    demands, taus = [], []
    for t, i in enumerate(arrivals):
        seen[int(i)] += 1
        tau = T - t - 1
        demands.append(count_forecast(totals, seen, tau, instance.arrival_rates))
        taus.append(tau)
    values, dual = lp_value_table(instance, demands, taus, want_duals=duals, workers=workers)
    masks = [lp_offer_masks(instance, values[t], L, int(arrivals[t]), taus[t],
                            None if not duals else dual[t]) for t in range(T)]
    return lambda t, i: masks[t]
