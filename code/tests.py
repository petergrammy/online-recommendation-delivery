"""Verification of the model, the policies and the exact evaluators.

Run from the code directory:  python tests.py
"""
from functools import lru_cache
import unittest
import numpy as np
import gurobipy as gp
from recommendation import (Instance, default_instance, FluidLP, FixedTopK, LPLookahead, LPDual,
                            DPPolicy, simulate, latent_requests, actions, dp_benchmark, exact_table_policy_value,
                            exact_policy_value, exact_forward, Lattice, offer_fixed, offer_greedy, offer_dp_table,
                            LPTables, count_lp_offer, top_k, top_k_mask)


def small_instance(K=2):
    """Course instance with reduced stock, small enough for brute-force checks."""
    return default_instance().with_changes(capacities=[6, 5, 4], K=K)


class FluidModel(unittest.TestCase):
    def test_known_fluid_solution_and_duality(self):
        ins = default_instance()
        lp = FluidLP(ins)
        try:
            a = lp.solve([50, 30, 20], ins.capacities, True)
            self.assertAlmostEqual(a['value'], 706, places=7)
            np.testing.assert_allclose(a['sales'], [40, 29, 37 / 3], atol=1e-7)
            np.testing.assert_allclose(a['duals'], [9, 0, 0], atol=1e-7)
            self.assertEqual(lp.model.NumIntVars, 0)
            # dual objective of the decomposed dual equals the primal value
            beta = a['duals']
            P, r = ins.probabilities, ins.prices
            inner = [max(sum(P[i, j] * (r[j] - beta[j]) for j in s) for s in actions(ins)) for i in range(3)]
            self.assertAlmostEqual(float(beta @ ins.capacities + np.dot([50, 30, 20], inner)), 706, places=7)
        finally:
            lp.close()

    def test_item_lp_equals_assortment_lp(self):
        ins = Instance([7, 5, 3, 2], [[.2, .3, .1, .2], [.4, .1, .2, 0]], [3, 2, 4, 1], [.6, .4], 2)
        lp = FluidLP(ins)
        try:
            value = lp.solve([6, 4], ins.capacities)
            model = gp.Model(env=lp.env)
            sets = actions(ins)
            z = model.addVars(ins.m, len(sets), lb=0)
            model.setObjective(gp.quicksum(z[i, a] * sum(ins.probabilities[i, j] * ins.prices[j] for j in s)
                                           for i in range(ins.m) for a, s in enumerate(sets)), gp.GRB.MAXIMIZE)
            for i, N in enumerate([6, 4]):
                model.addConstr(gp.quicksum(z[i, a] for a in range(len(sets))) == N)
            for j in range(ins.n):
                model.addConstr(gp.quicksum(z[i, a] * ins.probabilities[i, j] for i in range(ins.m)
                                            for a, s in enumerate(sets) if j in s) <= ins.capacities[j])
            model.optimize()
            self.assertAlmostEqual(value, model.ObjVal, places=7)
            model.dispose()
        finally:
            lp.close()


class DynamicProgramming(unittest.TestCase):
    def test_dp_against_independent_recursive_enumeration(self):
        ins = Instance([7, 4], [[.4, .3], [.1, .6]], [2, 1], [.6, .4], 1)
        seq = (0, 1, 1, 0)

        @lru_cache(None)
        def oracle(t, c):
            if t == len(seq):
                return 0.0
            best = 0.0
            for s in actions(ins):
                if any(c[j] == 0 for j in s):
                    continue
                value = (1 - sum(ins.probabilities[seq[t], j] for j in s)) * oracle(t + 1, c)
                for j in s:
                    less = list(c)
                    less[j] -= 1
                    value += ins.probabilities[seq[t], j] * (ins.prices[j] + oracle(t + 1, tuple(less)))
                best = max(best, value)
            return best

        value, table, sets = dp_benchmark(ins, 4, seq, True)
        self.assertAlmostEqual(value, oracle(0, tuple(ins.capacities)), places=10)
        self.assertAlmostEqual(value, exact_policy_value(ins, seq, DPPolicy(table, sets)), places=10)

    def test_forward_equals_backward_for_table_policies(self):
        ins = small_instance()
        rng = np.random.default_rng(5)
        seq = rng.integers(ins.m, size=20)
        L = Lattice(ins)
        _, iid, _ = dp_benchmark(ins, 20, keep_policy=True)
        seq_value, seq_table, _ = dp_benchmark(ins, 20, sequence=seq, keep_policy=True)
        self.assertAlmostEqual(exact_forward(ins, seq, offer_dp_table(ins, iid, 20), L).expected_revenue,
                               exact_table_policy_value(ins, seq, iid), places=9)
        self.assertAlmostEqual(exact_forward(ins, seq, offer_greedy(ins, L), L).expected_revenue,
                               exact_table_policy_value(ins, seq), places=9)
        self.assertAlmostEqual(exact_forward(ins, seq, offer_dp_table(ins, seq_table, 20), L).expected_revenue,
                               seq_value, places=9)


class Policies(unittest.TestCase):
    def test_vectorized_ranking_matches_scalar_ranking(self):
        rng = np.random.default_rng(11)
        for _ in range(3000):
            n = rng.integers(1, 6)
            K = rng.integers(0, n + 1)
            scores = np.round(rng.normal(size=n), 1) * rng.integers(0, 2, size=n)
            immediate = np.abs(np.round(rng.normal(size=n), 1))
            inv = rng.integers(0, 2, size=n)
            mask = top_k_mask(scores[None], immediate, (inv > 0)[None], K)[0]
            self.assertEqual(tuple(np.flatnonzero(mask)), top_k(scores, inv, K, immediate))

    def test_lp_tables_reproduce_online_policies_exactly(self):
        ins = small_instance()
        T = 18
        tables = LPTables(ins, T, workers=1)
        L = tables.lattice
        rng = np.random.default_rng(3)
        for _ in range(3):
            seq = rng.integers(ins.m, size=T)
            for cls, offer in [(LPLookahead, tables.offer_fd()), (LPDual, tables.offer_dual())]:
                policy = cls(ins)
                try:
                    online = exact_policy_value(ins, seq, policy)
                finally:
                    policy.close()
                self.assertAlmostEqual(exact_forward(ins, seq, offer, L).expected_revenue, online, places=9)

    def test_count_forecast_table_matches_online(self):
        ins = small_instance()
        seq = np.array([1] * 5 + [0] * 8 + [2] * 5)
        totals = np.bincount(seq, minlength=3)
        L = Lattice(ins)
        exact = exact_forward(ins, seq, count_lp_offer(ins, seq, totals, L, workers=1), L).expected_revenue
        policy = LPLookahead(ins, forecast='count', totals=totals)
        try:
            rng = np.random.default_rng(9)
            U = rng.random((4000, len(seq)))
            sims = [simulate(ins, seq, policy, u).revenue for u in U]
        finally:
            policy.close()
        self.assertLess(abs(np.mean(sims) - exact), 4 * np.std(sims) / np.sqrt(len(sims)))

    def test_generic_feasibility_and_prefix_causality(self):
        rng = np.random.default_rng(163)
        for m, n, T, K in [(1, 1, 8, 0), (2, 4, 12, 1), (4, 5, 15, 3), (2, 3, 10, 8)]:
            P = rng.dirichlet(np.ones(n + 1), size=m)[:, :n]
            ins = Instance(np.arange(1, n + 1), P, np.arange(n) % 3, np.ones(m) / m, K)
            for cls in (LPLookahead, LPDual):
                policy = cls(ins)
                try:
                    seq = rng.integers(m, size=T)
                    u = rng.random(T)
                    run = simulate(ins, seq, policy, u, True)
                    changed = seq.copy()
                    changed[T // 2:] = rng.integers(m, size=T - T // 2)
                    other = simulate(ins, changed, policy, u, True)
                    self.assertEqual(run.trace[:T // 2], other.trace[:T // 2])
                    self.assertTrue(np.all(run.remaining >= 0))
                    self.assertEqual(run.oos_slots, 0)
                    self.assertAlmostEqual(run.revenue, np.dot(ins.capacities - run.remaining, ins.prices))
                finally:
                    policy.close()

    def test_algorithm1_wastes_slots_only_after_stockouts(self):
        ins = default_instance()
        self.assertEqual(FixedTopK(ins).sets, [(0, 1), (0, 1), (1, 2)])
        rng = np.random.default_rng(1)
        seq = rng.integers(3, size=100)
        L = Lattice(ins)
        exact = exact_forward(ins, seq, offer_fixed(ins, L), L)
        U = rng.random((3000, 100))
        runs = [simulate(ins, seq, FixedTopK(ins), u) for u in U]
        oos = np.array([r.oos_slots for r in runs])
        self.assertLess(abs(oos.mean() - exact.expected_oos_slots), 4 * oos.std() / np.sqrt(len(oos)) + 1e-9)
        with self.assertRaises(AssertionError):
            class Bad(FixedTopK):
                respects_inventory = True
            simulate(ins.with_changes(capacities=[0, 0, 0]), [0], Bad(ins), [0.1])

    def test_choice_marginals(self):
        ins = default_instance()
        N = 100000
        latent = latent_requests(ins, np.zeros(N, dtype=int), np.random.default_rng(321).random(N))
        np.testing.assert_allclose(np.bincount(latent, minlength=4) / N, [.6, .2, .1, .1], atol=.004)

    def test_zero_opportunity_score_serves_ties(self):
        ins = Instance([10], [[1]], [1], [1], 1)
        policy = LPLookahead(ins)
        try:
            self.assertEqual(policy.recommend(0, np.array([1]), 100), (0,))
            self.assertEqual(simulate(ins, [0, 0], policy, [.1, .1]).revenue, 10)
        finally:
            policy.close()

    def test_all_products_available_is_pathwise_optimal(self):
        ins = default_instance().with_changes(K=3)
        rng = np.random.default_rng(170)
        seq = rng.integers(ins.m, size=80)
        u = rng.random(80)
        policy = LPLookahead(ins)
        try:
            run = simulate(ins, seq, policy, u)
            requests = latent_requests(ins, seq, u)
            sales = np.minimum(np.bincount(requests, minlength=ins.n + 1)[:ins.n], ins.capacities)
            self.assertAlmostEqual(run.revenue, float(sales @ ins.prices))
            np.testing.assert_array_equal(run.remaining, ins.capacities - sales)
        finally:
            policy.close()

    def test_reject_ambiguous_or_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Instance([1, 2], [[.8, .7]], [1, 1], [1], 2)
        with self.assertRaises(ValueError):
            Instance([1], [[.5]], [.5], [1], 1)
        with self.assertRaises(ValueError):
            Instance([1], [[.5]], [1], [.8], 1)


class Question1(unittest.TestCase):
    def test_counterexample_closed_form_and_exact(self):
        ins = Instance([10, 8, 6], [[.5, .3, .2]], [1, 1, 20], [1], 2)
        naive = 10 * (1 - .5 ** 20) + 8 * (1 - .7 ** 20)
        feasible = 10 * (1 - .5 ** 20) + 20 * .2 * 6
        self.assertAlmostEqual((feasible - naive) / feasible, .470776, places=5)
        L = Lattice(ins)
        seq = np.zeros(20, dtype=int)
        self.assertAlmostEqual(exact_forward(ins, seq, offer_fixed(ins, L), L).expected_revenue, naive, places=9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
