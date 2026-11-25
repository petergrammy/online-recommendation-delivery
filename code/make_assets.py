"""Build every table, figure and number of the report from ../results.

    python make_assets.py      # writes ../paper/tables, ../paper/figures, ../paper/generated/numbers.tex

All numbers quoted in the text are LaTeX macros defined in generated/numbers.tex,
so the paper always matches the stored results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binom, t as student_t

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from recommendation import default_instance, FluidLP          # noqa: E402
from experiments import datasets, SEED                         # noqa: E402

ROOT = CODE.parent
RES = ROOT / 'results'
PAPER = ROOT / 'paper'
TAB, FIG, GEN = PAPER / 'tables', PAPER / 'figures', PAPER / 'generated'
for folder in (TAB, FIG, GEN):
    folder.mkdir(parents=True, exist_ok=True)

INS = default_instance()
SSTAR = np.array([40, 29, 37 / 3])               # fluid sales at N = (50, 30, 20)
CBAR = INS.capacities - SSTAR                   # fluid leftover inventory
DS = ['D1', 'D2', 'D3', 'D4', 'D5']

# ---------------------------------------------------------------------------
# Style. One fixed colour per policy; benchmarks in neutral ink.
# ---------------------------------------------------------------------------
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
COLOR = {'LP-Dual': '#2a78d6', 'LP-FD': '#eb6834', 'Greedy': '#1baf7a', 'IID-DP': INK2, 'Seq-DP': INK,
         'Alg1': '#e87ba4', 'LP-Count': '#008300'}
MARKER = {'LP-Dual': 'o', 'LP-FD': 's', 'Greedy': '^', 'IID-DP': 'D', 'Seq-DP': None, 'Alg1': 'v', 'LP-Count': 'P'}
STYLE = {'LP-Dual': '-', 'LP-FD': '-', 'Greedy': '-', 'IID-DP': '--', 'Seq-DP': ':', 'Alg1': '-', 'LP-Count': '-.'}
TEX = {'Alg1': r'\AlgOne', 'Greedy': r'\Greedy', 'LP-FD': r'\LPFD', 'LP-Dual': r'\LPDual', 'LP-Count': r'\LPCount',
       'IID-DP': r'\IIDDP', 'Seq-DP': r'\SeqDP'}
KEY = {'Alg1': 'Alg', 'Greedy': 'Greedy', 'LP-FD': 'FD', 'LP-Dual': 'Dual', 'LP-Count': 'Count', 'IID-DP': 'IID',
       'Seq-DP': 'Seq'}
plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'mathtext.fontset': 'cm', 'font.size': 8.5,
    'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'legend.fontsize': 7.8, 'lines.linewidth': 1.5, 'lines.markersize': 4.5,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
WIDTH = 6.5

NUM = {}


def no_negative_zero(text):
    """'-0.000' (a floating-point zero) is printed as '0.000'."""
    return text[1:] if text.startswith('-') and not text.strip('-0.') else text


def num(name, value, fmt='{:.3f}'):
    assert name.isalpha(), name
    NUM[name] = value if isinstance(value, str) else no_negative_zero(fmt.format(value))
    return NUM[name]


def f3(x):
    return no_negative_zero(f'{float(x):.3f}')


def pct(x, d=1):
    return f'{100 * float(x):.{d}f}\\%'


def write(name, text):
    (TAB / f'{name}.tex').write_text(text, encoding='utf-8')


def table(caption, label, header, rows, spec, note=None, size=r'\small', placement='htbp', pre=None):
    out = [f'\\begin{{table}}[{placement}]', r'\centering' + size, f'\\caption{{{caption}}}\\label{{{label}}}',
           f'\\begin{{tabular}}{{{spec}}}', r'\toprule']
    if pre:
        out += pre
    out += [' & '.join(header) + r' \\', r'\midrule']
    for row in rows:
        out.append(row if isinstance(row, str) else ' & '.join(str(v) for v in row) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}']
    if note:
        out.append(r'\par\vspace{0.35em}{\footnotesize ' + note + '}')
    out.append(r'\end{table}')
    return '\n'.join(out) + '\n'


def save_fig(fig, name):
    fig.savefig(FIG / f'{name}.pdf')
    plt.close(fig)


def end_labels(ax, x, ys, labels, gap, dx=None, color=INK, fontsize=7.5):
    """Direct labels at the right end of lines, pushed apart by at least `gap` (data units)."""
    order = np.argsort(ys)
    placed = []
    for idx in order:
        y = ys[idx] if not placed else max(ys[idx], placed[-1] + gap)
        placed.append(y)
    shift = max(0.0, placed[-1] - max(ys)) / 2 if placed else 0
    xr = ax.get_xlim()
    dx = dx if dx is not None else 0.012 * (xr[1] - xr[0])
    for idx, y in zip(order, placed):
        ax.annotate(labels[idx], (x, ys[idx]), xytext=(x + dx, y - shift), textcoords='data', va='center',
                    fontsize=fontsize, color=color, annotation_clip=False)


def legend_handles(policies, line=True):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=COLOR[p], ls=STYLE[p] if line else 'none', marker=MARKER[p], label=p,
                   markersize=4.5, lw=1.5) for p in policies]


# ---------------------------------------------------------------------------
# Section 2: instance facts
# ---------------------------------------------------------------------------
def instance_facts():
    def runs(a):
        return 1 + int(np.sum(a[1:] != a[:-1]))
    observed = [runs(seq) for seq in datasets().values()]
    rng = np.random.default_rng(SEED + 1)
    base = np.repeat([0, 1, 2], [50, 30, 20])
    sample = np.array([runs(rng.permutation(base)) for _ in range(500)])
    num('RunsMin', min(observed), '{:d}')
    num('RunsMax', max(observed), '{:d}')
    num('RunsLo', np.percentile(sample, 2.5), '{:.0f}')
    num('RunsHi', np.percentile(sample, 97.5), '{:.0f}')
    # product-1 requests if every type-1/2 customer sees product 1: D = Bin(50,.6) + Bin(30,.4)
    pmf = np.convolve(binom.pmf(np.arange(51), 50, .6), binom.pmf(np.arange(31), 30, .4))
    k = np.arange(len(pmf))
    num('DemandOneSd', np.sqrt(pmf @ (k - 42) ** 2), '{:.2f}')
    num('DemandOneShort', pmf[k < 40].sum() * 100, '{:.1f}')
    num('DemandOneLeft', pmf @ np.maximum(40 - k, 0), '{:.3f}')


# ---------------------------------------------------------------------------
# Section 3: Question 1
# ---------------------------------------------------------------------------
def q1():
    d = pd.read_csv(RES / 'q1_family.csv')
    rows = []
    for _, r in d.iterrows():
        rows.append([int(r['T']), f3(r['ER_Alg1']), f3(r['closed_feasible']), f3(r['ER_Greedy']), f3(r['ER_LP-FD']),
                     f3(r['ER_LP-Dual']), f3(r['DP']), pct(1 - r['ER_Alg1'] / r['DP']), f"{r['oos_Alg1']:.1f}"])
    header = ['$T$', r'$\E R^{\AlgOne}$', r"$\E R^{\pi'}$", r'$\E R^{\Greedy}$', r'$\E R^{\LPFD}$',
              r'$\E R^{\LPDual}$', '$V^{*}$', '$L^{*}(T)$', '$W(T)$']
    write('q1', table(r'算例\ref{ex:q1}的精确期望收入、算法 1 的相对损失 $L^{*}(T)$ 与缺货推荐次数 $W(T)$', 'tab:q1',
                      header, rows, 'r' * 9))
    r20, r100 = d[d['T'] == 20].iloc[0], d[d['T'] == 100].iloc[0]
    num('QoneLossOptTwenty', 100 * (1 - r20['ER_Alg1'] / r20['DP']), '{:.1f}')
    num('QoneLossOptHundred', 100 * (1 - r100['ER_Alg1'] / r100['DP']), '{:.1f}')
    num('QoneWasteTwenty', r20['oos_Alg1'], '{:.1f}')
    num('QoneLPMaxGap', max((d.DP - d['ER_LP-FD']).max(), (d.DP - d['ER_LP-Dual']).max()), '{:.2f}')
    num('QoneGreedyGap', (d.DP - d.ER_Greedy)[d['T'] >= 20].mean(), '{:.1f}')


# ---------------------------------------------------------------------------
# Section 4: Question 2
# ---------------------------------------------------------------------------
def q2():
    d = pd.read_csv(RES / 'offline_models.csv')
    names = {'type-binary': r'(a) 类型级 0-1 模型', 'customer-binary': r'(b) 顾客级 0-1 模型',
             'type-LP': r'(c) 流体线性规划\eqref{eq:lp}'}
    rows = []
    for key in ['type-binary', 'customer-binary', 'type-LP']:
        sub = d[d.model == key]
        r = sub.iloc[0]
        sales = json.loads(r.sales)
        rows.append([names[key], int(r.variables), int(r.constraints), f'{sub.value.mean():.1f}',
                     '$(' + ',\\ '.join(f'{v:.2f}' for v in sales) + ')$'])
    note = '顾客级模型在五份数据上的最优值与期望销量都相同，变量数与约束数按 $T=100$ 计算；流体线性规划的约束数不含变量上界。'
    write('offline', table('课程实例上三种离线模型的规模、最优值与期望销量', 'tab:offline',
                           ['模型', '变量数', '约束数', '最优值', '期望销量'], rows, 'lrrrl', note=note))
    p = pd.read_csv(RES / 'lp_parametric.csv')
    lp = FluidLP(INS)
    N = np.array([50, 30, 20])

    def duals(theta):
        return np.round(lp.solve(N, theta * INS.capacities, details=True, cold=True)['duals'], 6)

    def bisect(lo, hi):
        a = duals(lo)
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.array_equal(duals(mid), a) else (lo, mid)
        return (lo + hi) / 2
    theta_c, theta_b, theta_a = bisect(0.60, 0.75), bisect(0.75, 0.90), bisect(0.95, 1.10)
    lp.close()
    num('ThetaA', theta_a, '{:.2f}')
    num('ThetaB', theta_b, '{:.3f}')
    num('ThetaC', theta_c, '{:.3f}')
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.3))
    ax[0].plot(p.theta, p.F, color=INK, lw=1.6)
    ax[0].axvline(1, color=MUTED, lw=0.8)
    ax[0].set(xlabel=r'库存缩放系数 $\theta$', ylabel=r'$F(N,\theta C)$', xlim=(0, 1.6), ylim=(0, 780))
    ax[0].annotate('724', (1.5, 724), xytext=(0, 4), textcoords='offset points', color=INK2, ha='center')
    ax[0].annotate(r'$\theta=1$：706', (1, 706), xytext=(-52, -2), textcoords='offset points', color=INK2)
    shades = [INK, INK2, MUTED]
    dashes = ['-', '--', '-.']
    for j in range(3):
        ax[1].step(p.theta, p[f'beta{j + 1}'], where='post', color=shades[j], lw=1.6, ls=dashes[j])
        ax[1].annotate(rf'$\beta_{j + 1}$', (0.3, [10, 8, 6][j]), xytext=(0, 3), textcoords='offset points',
                       color=shades[j], fontsize=9)
    ax[1].axvline(1, color=MUTED, lw=0.8)
    for x in (theta_c, theta_b, theta_a):
        ax[1].axvline(x, color=GRID, lw=0.8, zorder=0)
    ax[1].set(xlabel=r'库存缩放系数 $\theta$', ylabel='影子价格', xlim=(0, 1.6), ylim=(-0.5, 11.5))
    fig.tight_layout(w_pad=2)
    save_fig(fig, 'lp_parametric')


# ---------------------------------------------------------------------------
# Section 5: policy behaviour
# ---------------------------------------------------------------------------
def policy_behaviour():
    ex = pd.read_csv(RES / 'policy_examples.csv')
    num('FirstPeriodF', ex[ex.H == 100].F.iloc[0], '{:.2f}')
    th = pd.read_csv(RES / 'reservation_threshold.csv')
    th = th[(th.c2 == 35) & (th.c3 == 25)].sort_values('H')
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.3))
    H = th.H.to_numpy()
    ax.plot(H, 0.12 * (H - 1), color=AXIS, lw=3, solid_capstyle='butt', zorder=1)
    ends = []
    for pol, col in [('IID-DP', 'IIDDP'), ('LP-FD', 'FD'), ('LP-Dual', 'Dual')]:
        y = th[f'withhold_max_{col}_type1'].to_numpy()
        ax.step(H, y, where='post', color=COLOR[pol], ls=STYLE[pol], lw=1.6, zorder=3)
        ends.append((pol, y[-1]))
    ax.set_xlim(1, 100)
    end_labels(ax, 100, np.array([e[1] for e in ends], float), [e[0] for e in ends], gap=1.9)
    ax.annotate(r'$0.12(H-1)$', (60, 0.12 * 59), xytext=(6, -12), textcoords='offset points', color=INK2)
    ax.set(xlabel='剩余期数 $H$', ylabel=r'预留阈值 $\bar c_1(H)$', xlim=(1, 100), ylim=(0, 25))
    fig.tight_layout()
    save_fig(fig, 'threshold')
    row = th[th.H == 100].iloc[0]
    num('ThrDualHundred', row.withhold_max_Dual_type1, '{:.0f}')
    num('ThrFDHundred', row.withhold_max_FD_type1, '{:.0f}')
    num('ThrIIDHundred', row.withhold_max_IIDDP_type1, '{:.0f}')


# ---------------------------------------------------------------------------
# Section 7: experiments
# ---------------------------------------------------------------------------
def load_exact():
    d = pd.read_csv(RES / 'exact_datasets.csv').set_index('sequence')
    if (RES / 'exact_count.csv').exists():
        c = pd.read_csv(RES / 'exact_count.csv').set_index('sequence')
        d = d.join(c[[col for col in c.columns if col.endswith('_LP-Count')]])
    for pol in ['Alg1', 'Greedy', 'LP-FD', 'LP-Dual', 'IID-DP', 'Seq-DP', 'LP-Count']:
        if f'ER_{pol}' in d:
            d[f'G_{pol}'] = d.F - d[f'ER_{pol}']
            d[f'L_{pol}'] = d.V_seq - d[f'ER_{pol}']
    return d


def main_results():
    d = load_exact()
    m = d.loc[DS]
    sim = pd.read_csv(RES / 'sim_summary.csv') if (RES / 'sim_summary.csv').exists() else None
    pols = [p for p in ['Alg1', 'Greedy', 'LP-FD', 'LP-Dual', 'LP-Count', 'IID-DP', 'Seq-DP'] if f'ER_{p}' in m]
    for p in pols:
        k = KEY[p]
        num(f'ER{k}', m[f'ER_{p}'].mean())
        num(f'Gavg{k}', m[f'G_{p}'].mean())
        num(f'Gmax{k}', m[f'G_{p}'].max())
        num(f'Lavg{k}', m[f'L_{p}'].mean())
        num(f'Econe{k}', m[f'Ec1_{p}'].mean())
        num(f'Ectwo{k}', m[f'Ec2_{p}'].mean(), '{:.2f}')
        num(f'Ecthree{k}', m[f'Ec3_{p}'].mean(), '{:.2f}')
    num('FluidGap', (m.F - m.V_seq).mean())
    num('FluidGapMin', (m.F - m.V_seq).min())
    num('FluidGapMax', (m.F - m.V_seq).max())
    num('FluidShare', 100 * (m.F - m.V_seq).mean() / m['G_LP-Dual'].mean(), '{:.1f}')
    num('RatioDual', 100 * m['ER_LP-Dual'].mean() / 706, '{:.2f}')
    num('AlgOneER', m.ER_Alg1.mean())
    num('AlgOneOOS', m.oos_Alg1.mean(), '{:.2f}')
    num('AlgOneLossGreedy', m.ER_Greedy.mean() - m.ER_Alg1.mean())
    diff = m['ER_LP-Dual'] - m.ER_Greedy
    num('DualGreedyMin', diff.min())
    num('DualGreedyMax', diff.max())
    num('DualGreedyAvg', diff.mean())
    diff = m['ER_LP-Dual'] - m['ER_LP-FD']
    num('DualFDAvg', diff.mean())
    num('DualFDWins', int((diff > 0).sum()), '{:d}')
    num('SdDual', m['sd_LP-Dual'].mean(), '{:.1f}')
    num('PnegDual', 100 * m['Pneg_LP-Dual'].mean(), '{:.1f}')
    num('PnegDualMin', 100 * m['Pneg_LP-Dual'].min(), '{:.1f}')
    num('PnegDualMax', 100 * m['Pneg_LP-Dual'].max(), '{:.1f}')
    # ---- table: LP-Dual per dataset
    rows = []
    for s in DS:
        r = d.loc[s]
        mc = sim[(sim.dataset == s) & (sim.policy == 'LP-Dual')].iloc[0] if sim is not None else None
        rows.append([s[1], f3(r['ER_LP-Dual']), f3(r['G_LP-Dual']), f3(r['L_LP-Dual']),
                     f3(mc.mean_revenue) if mc is not None else '--', f3(mc.hw95) if mc is not None else '--',
                     f3(r['Ec1_LP-Dual']), f3(r['Ec2_LP-Dual']), f3(r['Ec3_LP-Dual']), pct(r['Pso1_LP-Dual'])])
    rows.append(r'\midrule')
    avg_mc = sim[(sim.policy == 'LP-Dual')].mean_revenue.mean() if sim is not None else None
    rows.append(['平均', f3(m['ER_LP-Dual'].mean()), f3(m['G_LP-Dual'].mean()), f3(m['L_LP-Dual'].mean()),
                 f3(avg_mc) if avg_mc is not None else '--', '', f3(m['Ec1_LP-Dual'].mean()), f3(m['Ec2_LP-Dual'].mean()),
                 f3(m['Ec3_LP-Dual'].mean()), pct(m['Pso1_LP-Dual'].mean())])
    header = ['$s$', r'$\E R_T$', '$G_s$', '$L_s$', r'$\overline R_s$', '$h_s$', r'$\E c_{T+1,1}$', r'$\E c_{T+1,2}$',
              r'$\E c_{T+1,3}$', '$q_1$']
    note = (r'$\E R_T$、$G_s$、$L_s$、$\E c_{T+1,j}$ 与 $q_1=\Pr(c_{T+1,1}=0)$ 为精确值；$\overline R_s$ 与 $h_s$ 来自 $B=2000$ '
            r'次蒙特卡洛重复，定义见式\eqref{eq:mcstats}。五份数据的 $F_s$ 均为 706。')
    write('main', table(r'主策略 \LPDual{} 在五份数据上的收入、遗憾与期末库存', 'tab:main', header, rows,
                        'c' + 'r' * 9, note=note))
    # ---- table: aggregate over policies
    rows = []
    for p in pols:
        mc = sim[sim.policy == p] if sim is not None else pd.DataFrame()
        g_mc = f3((706 - mc.mean_revenue).mean()) if len(mc) else '--'
        w = m[f'oos_{p}'].mean() if f'oos_{p}' in m else 0.0
        rows.append([TEX[p] + '{}', f3(m[f'ER_{p}'].mean()), f3(m[f'G_{p}'].mean()), f3(m[f'G_{p}'].max()), g_mc,
                     f3(m[f'L_{p}'].mean()), f'{w:.2f}' if p != 'Seq-DP' else '--'])
    header = ['策略', r'$\overline{\E R_T}$', r'$\overline G$', r'$G_{\max}$', r'$\overline G^{\,\mathrm{MC}}$',
              r'$\overline L$', r'$\overline W$']
    note = (r'$\overline G$ 与 $G_{\max}$ 按式\eqref{eq:avgmax}由精确期望计算；$\overline G^{\,\mathrm{MC}}=\frac15\sum_s(F_s-\overline R_s)$ '
            r'为模拟估计；$\overline L$ 为五份数据上策略损失 $L_s$ 的平均；$\overline W$ 为每条路径上缺货推荐的期望次数，定义见式\eqref{eq:waste}。'
            r'\LPCount{} 使用了总人数 $N$；\SeqDP{} 预知顾客类型序列，只作为基准。')
    write('aggregate', table('各策略在五份数据上的综合表现', 'tab:aggregate', header, rows, 'lrrrrrr', note=note))
    # ---- table: product decomposition
    rows = []
    for p in ['Alg1', 'Greedy', 'LP-FD', 'LP-Dual', 'IID-DP', 'Seq-DP']:
        ec = np.array([m[f'Ec{j + 1}_{p}'].mean() for j in range(3)])
        contrib = INS.prices * (ec - CBAR)
        rows.append([TEX[p] + '{}'] + [f3(v) for v in ec] + [f3(v) for v in contrib] + [f3(contrib.sum())])
    header = ['策略', r'$\E c_{T+1,1}$', r'$\E c_{T+1,2}$', r'$\E c_{T+1,3}$', r'$\delta_1$', r'$\delta_2$', r'$\delta_3$',
              r'$\overline G=\sum_j\delta_j$']
    write('product', table(r'按商品分解的期望遗憾（五份数据平均），$\delta_j$ 的定义见式\eqref{eq:productgap}', 'tab:product',
                           header, rows, 'l' + 'r' * 7, size=r'\small'))
    ec = np.array([m[f'Ec{j + 1}_LP-Dual'].mean() for j in range(3)])
    contrib = INS.prices * (ec - CBAR)
    num('ContribOneDual', contrib[0])
    num('ContribTwoDual', contrib[1])
    num('ContribThreeDual', contrib[2])
    num('ContribOneShare', 100 * contrib[0] / contrib.sum(), '{:.0f}')
    # ---- figure: policy loss per dataset
    show = ['Greedy', 'LP-FD', 'LP-Dual', 'IID-DP']
    fig, ax = plt.subplots(figsize=(WIDTH, 2.25))
    x = np.arange(5)
    for k, p in enumerate(show):
        ax.plot(x + (k - 1.5) * 0.14, m[f'L_{p}'], ls='none', marker=MARKER[p], color=COLOR[p], ms=5.5,
                markeredgecolor='white', markeredgewidth=0.6, label=p)
    ax.set(xticks=x, xticklabels=[f'数据 {s[1]}' for s in DS], ylabel=r'策略损失 $L_s=V^{\mathrm{seq}}_s-\mathbb{E}R_T$',
           ylim=(-0.03, max(1.1, m[[f'L_{p}' for p in show]].to_numpy().max() * 1.1)))
    ax.legend(handles=legend_handles(show, line=False), ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    save_fig(fig, 'main_loss')
    # ---- figure: exact revenue distribution of LP-Dual on D1
    paths = json.loads((RES / 'exact_paths.json').read_text())
    pv = np.array(paths['D1']['LP-Dual']['pmf_values'])
    pm = np.array(paths['D1']['LP-Dual']['pmf'])
    bins = np.arange(600, 760, 4)
    hist = np.array([pm[(pv >= a) & (pv < a + 4)].sum() for a in bins])
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.2))
    ax.bar(bins + 2, hist, width=3.4, color=COLOR['LP-Dual'], alpha=0.85, edgecolor='none')
    mean = float(pv @ pm)
    top = hist.max() * 1.22
    ax.plot([706, 706], [0, top * 0.93], color=INK, lw=1, zorder=4)
    ax.plot([mean, mean], [0, top * 0.93], color=INK, lw=1, ls='--', zorder=4)
    ax.annotate(r'$F=706$', (706, top * 0.93), xytext=(3, 0), textcoords='offset points', color=INK, va='center')
    ax.annotate(rf'$\mathbb{{E}}R_T={mean:.1f}$', (mean, top * 0.93), xytext=(-3, 0), textcoords='offset points',
                ha='right', color=INK, va='center')
    ax.set(xlabel=r'总收入 $R_T$', ylabel='概率（组距 4）', xlim=(610, 770), ylim=(0, top))
    fig.tight_layout()
    save_fig(fig, 'revenue_dist')
    num('ProbAboveFDOne', 100 * pm[pv > 706 + 1e-9].sum(), '{:.1f}')
    # ---- Monte-Carlo consistency
    if sim is not None:
        z = []
        for p in ['Alg1', 'Greedy', 'LP-FD', 'LP-Dual', 'IID-DP']:
            for s in DS:
                r = sim[(sim.dataset == s) & (sim.policy == p)].iloc[0]
                z.append(abs(r.mean_revenue - d.loc[s, f'ER_{p}']) / (r.sd_revenue / np.sqrt(r.B)))
        num('MCmaxZ', max(z), '{:.2f}')
        num('MCinside', int(np.sum(np.array(z) <= student_t.ppf(.975, 1999))), '{:d}')
        num('MCtotal', len(z), '{:d}')
        num('MChw', sim[sim.policy == 'LP-Dual'].hw95.mean(), '{:.2f}')
        pair = pd.read_csv(RES / 'sim_paired.csv')
        sub = pair[(pair.a == 'LP-Dual') & (pair.b == 'Greedy')]
        num('MCpairhw', sub.hw95.mean(), '{:.2f}')
        num('MCpairEqual', 100 * sub.share_equal.mean(), '{:.1f}')
        sub = pair[(pair.a == 'LP-Dual') & (pair.b == 'LP-FD')]
        num('MCpairEqualFD', 100 * sub.share_equal.mean(), '{:.1f}')
        tm = pd.read_csv(RES / 'sim_timing.csv').set_index('policy')
        num('TimeDualMedian', tm.loc['LP-Dual', 'median_ms'], '{:.3f}')
        num('TimeDualP', tm.loc['LP-Dual', 'p95_ms'], '{:.2f}')
        num('TimeFDMedian', tm.loc['LP-FD', 'median_ms'], '{:.3f}')
        num('TimeFDP', tm.loc['LP-FD', 'p95_ms'], '{:.2f}')
        num('SolvesDual', f"{int(tm.loc['LP-Dual', 'lp_solves']):,}".replace(',', '{,}'))
        num('SolvesFD', f"{int(tm.loc['LP-FD', 'lp_solves']):,}".replace(',', '{,}'))


def loss_analysis():
    d = pd.read_csv(RES / 'loss_decomposition.csv')
    per = pd.read_csv(RES / 'loss_by_period.csv')
    pols = ['Greedy', 'LP-FD', 'LP-Dual', 'IID-DP']

    def block(config, seqs):
        s = d[(d.config == config) & d.sequence.isin(seqs)]
        p = per[(per.config == config) & per.sequence.isin(seqs)]
        out = {}
        for pol in pols:
            a = s[s.policy == pol]
            b = p[p.policy == pol]
            sell = a[(a.type == 1) & (a.category == 'oversell')].loss.sum() / len(seqs)
            hold = a[(a.type == 1) & (a.category == 'overreserve')].loss.sum() / len(seqs)
            total = a.loss.sum() / len(seqs)
            out[pol] = {'total': total, 'sell': sell, 'hold': hold, 'other': total - sell - hold,
                        'w1': b[b.t <= 60].loss.sum() / len(seqs), 'w2': b[(b.t > 60) & (b.t <= 90)].loss.sum() / len(seqs),
                        'w3': b[b.t > 90].loss.sum() / len(seqs),
                        'type2sell': a[(a.type == 2) & (a.category == 'oversell')].loss.sum() / len(seqs)}
        return out
    base = block('base', DS)
    rows = [[TEX[p] + '{}'] + [f3(base[p][k]) for k in ['total', 'sell', 'hold', 'other', 'w1', 'w2', 'w3']] for p in pols]
    header = ['策略', r'$\overline L$', r'$\overline L^{\mathrm{sell}}$', r'$\overline L^{\mathrm{hold}}$', r'$\overline L^{\mathrm{oth}}$',
              r'$\overline L_{[1,60]}$', r'$\overline L_{[61,90]}$', r'$\overline L_{[91,100]}$']
    write('loss', table('策略损失按决策类型与时段的分解（五份数据平均）', 'tab:loss', header, rows, 'l' + 'r' * 7))
    for p in pols:
        k = KEY[p]
        for part, name in [('sell', 'Sell'), ('hold', 'Hold'), ('w2', 'Mid'), ('w3', 'End')]:
            num(f'Loss{name}{k}', base[p][part])
    meta = json.loads((RES / 'loss_meta.json').read_text())
    num('LossIdentityErr', f"{meta['max_identity_error']:.0e}".replace('e-', r'\times10^{-') + '}')
    b213 = block('base', ['B213'])
    for p in pols:
        num(f'LossBlock{KEY[p]}', b213[p]['total'])
        num(f'HoldBlock{KEY[p]}', b213[p]['hold'])
    k1 = block('K1', DS)
    for p in pols:
        num(f'LossKone{KEY[p]}', k1[p]['total'])
        num(f'TypeTwoSellKone{KEY[p]}', k1[p]['type2sell'])
    th7 = block('theta0.7', DS)
    for p in pols:
        num(f'LossThetaSeven{KEY[p]}', th7[p]['total'])
        num(f'SellThetaSeven{KEY[p]}', th7[p]['sell'])
        num(f'HoldThetaSeven{KEY[p]}', th7[p]['hold'])
    # ---- figure: cumulative loss and product-1 stock-out probability over time
    paths = json.loads((RES / 'exact_paths.json').read_text())
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.35))
    t = np.arange(1, 101)
    ends = []
    for pol in pols:
        b = per[(per.config == 'base') & per.sequence.isin(DS) & (per.policy == pol)]
        cum = b.groupby('t').loss.sum().reindex(t, fill_value=0).cumsum() / len(DS)
        ax[0].plot(t, cum, color=COLOR[pol], ls=STYLE[pol], label=pol)
        ends.append(cum.iloc[-1])
    ax[0].set(xlabel='期数 $t$', ylabel='累计策略损失', xlim=(1, 100))
    end_labels(ax[0], 100, np.array(ends), pols, gap=0.05)
    for pol in pols + ['Seq-DP']:
        so = np.mean([np.array(paths[s][pol]['stockout_path'])[:, 0] for s in DS], axis=0)
        ax[1].plot(np.arange(1, 102), so, color=COLOR[pol], ls=STYLE[pol], lw=1.4 if pol != 'Seq-DP' else 1.2, label=pol)
    ax[1].set(xlabel='期数 $t$', ylabel=r'$\Pr(c_{t,1}=0)$', xlim=(50, 101), ylim=(0, 0.8))
    ax[1].legend(handles=legend_handles(pols + ['Seq-DP']), loc='upper left', fontsize=7.2)
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'loss_time')
    so = {pol: np.mean([np.array(paths[s][pol]['stockout_path'])[:, 0] for s in DS], axis=0) for pol in pols + ['Seq-DP']}
    for pol in pols + ['Seq-DP']:
        num(f'SoldOutEighty{KEY[pol]}', 100 * so[pol][80], '{:.1f}')      # Pr(c_{81,1} = 0): sold out by end of t = 80
        num(f'SoldOutEnd{KEY[pol]}', 100 * so[pol][100], '{:.1f}')


def random_sequences():
    g = pd.read_csv(RES / 'exact_generated.csv')
    pols = ['Alg1', 'Greedy', 'LP-FD', 'LP-Dual', 'IID-DP']
    rows = []
    stats = {}
    for p in pols:
        row = [TEX[p] + '{}']
        for st in ['permutation', 'iid']:
            s = g[g.set == st]
            G = s.F - s[f'ER_{p}']
            L = s.V_seq - s[f'ER_{p}']
            row += [f3(G.mean()), f3(np.percentile(G, 95)), f3(G.max()), f3(L.mean())]
            stats[(p, st)] = (G, L)
        rows.append(row)
    pre = [r' & \multicolumn{4}{c}{随机排列（500 条）} & \multicolumn{4}{c}{独立同分布到达（500 条）} \\',
           r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}']
    header = ['策略', r'$\overline G$', r'$G_{0.95}$', r'$G_{\max}$', r'$\overline L$', r'$\overline G$', r'$G_{0.95}$',
              r'$G_{\max}$', r'$\overline L$']
    note = r'$G_{0.95}$ 为 500 条序列上期望遗憾的 95\% 分位数。两组序列的 $F_s$ 与 $V^{\seq}_s$ 逐条计算。'
    write('random', table('生成序列上的精确期望遗憾与策略损失', 'tab:random', header, rows, 'l' + 'r' * 8, note=note, pre=pre))
    for st, tag in [('permutation', 'Perm'), ('iid', 'Iid')]:
        s = g[g.set == st]
        for p in pols:
            G, L = stats[(p, st)]
            num(f'Gavg{KEY[p]}{tag}', G.mean())
            num(f'Lavg{KEY[p]}{tag}', L.mean())
        d = s['ER_LP-Dual'] - s.ER_Greedy
        num(f'DualBeatsGreedy{tag}', 100 * np.mean(d >= -1e-9), '{:.1f}')
        num(f'DualMinusGreedy{tag}', d.mean())
        d = s['ER_LP-Dual'] - s['ER_LP-FD']
        num(f'DualBeatsFD{tag}', 100 * np.mean(d >= -1e-9), '{:.1f}')
        d = s['ER_LP-Dual'] - s['ER_IID-DP']
        num(f'DualBeatsIID{tag}', 100 * np.mean(d >= -1e-9), '{:.1f}')
        num(f'FluidGap{tag}', (s.F - s.V_seq).mean())
        num(f'Fmean{tag}', s.F.mean())
        num(f'Fmin{tag}', s.F.min(), '{:.1f}')
        num(f'Fmax{tag}', s.F.max(), '{:.1f}')
        num(f'AlgOneSd{tag}', s.ER_Alg1.std(), '{:.3f}')
        num(f'LmaxFD{tag}', (s.V_seq - s['ER_LP-FD']).max())
        num(f'LmaxDual{tag}', (s.V_seq - s['ER_LP-Dual']).max())
    # ---- which permutations are hard? (same generator as experiments.run_exact)
    rng = np.random.default_rng(SEED + 1)
    base = np.repeat([0, 1, 2], [50, 30, 20])
    perms = [rng.permutation(base) for _ in range(500)]
    s = g[g.set == 'permutation'].reset_index(drop=True)
    assert len(s) == 500 and s.sequence.iloc[0] == 'P001'
    last10 = np.array([np.sum(a[-10:] == 1) for a in perms])
    last20 = np.array([np.sum(a[-20:] == 1) for a in perms])
    loss = {p: (s.V_seq - s[f'ER_{p}']).to_numpy() for p in ['Greedy', 'LP-FD', 'LP-Dual']}
    num('CorrFDLastTen', np.corrcoef(loss['LP-FD'], last10)[0, 1], '{:.2f}')
    num('CorrDualLastTwenty', np.corrcoef(loss['LP-Dual'], last20)[0, 1], '{:.2f}')
    num('CorrGreedyLastTwenty', np.corrcoef(loss['Greedy'], last20)[0, 1], '{:.2f}')
    top = np.argsort(-loss['LP-FD'])[:10]
    num('TopFDLastTenMax', int(last10[top].max()), '{:d}')
    # ---- figure: distribution of the policy loss
    show = ['Greedy', 'LP-FD', 'LP-Dual', 'IID-DP']
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.3), sharey=True)
    for a, st, title in [(ax[0], 'permutation', '(a) 人数为 (50,30,20) 的随机排列'), (ax[1], 'iid', '(b) 按 $\\lambda$ 独立同分布到达')]:
        data = [stats[(p, st)][1].to_numpy() for p in show]
        bp = a.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=True,
                       medianprops={'color': INK, 'lw': 1.2}, whiskerprops={'color': INK2, 'lw': 0.8},
                       capprops={'color': INK2, 'lw': 0.8},
                       flierprops={'marker': 'o', 'ms': 2, 'markeredgecolor': 'none', 'markerfacecolor': MUTED, 'alpha': 0.6})
        for patch, p in zip(bp['boxes'], show):
            patch.set(facecolor=COLOR[p], alpha=0.75, edgecolor='none')
        a.set(yticks=range(1, len(show) + 1), yticklabels=show, xlabel=r'策略损失 $L_s$', title=title)
        a.title.set_fontsize(8.5)
        a.set_xlim(-0.1, None)
    ax[0].invert_yaxis()
    fig.tight_layout(w_pad=2)
    save_fig(fig, 'random')


def block_order_results():
    d = load_exact()
    order = ['B123', 'B132', 'B213', 'B231', 'B312', 'B321']
    pols = ['Greedy', 'LP-FD', 'LP-Dual', 'LP-Count', 'IID-DP', 'Seq-DP']
    pols = [p for p in pols if f'ER_{p}' in d]
    rows = []
    for s in order:
        r = d.loc[s]
        label = r'$' + r'\to'.join(s[1:]) + r'$'
        rows.append([label] + [f3(r[f'ER_{p}']) for p in pols] + [f3(r['Ec1_LP-Dual'])])
    header = ['到达顺序'] + [TEX[p] + '{}' for p in pols] + [r'$\E c^{\LPDual}_{T+1,1}$']
    note = r'表中为精确期望收入，各顺序的 $F_s$ 均为 706。$1\to2\to3$ 表示先到达 50 位类型 1 顾客，再到达 30 位类型 2 与 20 位类型 3 顾客，其余类推。算法 1 在任何顺序下的期望收入都是 \AlgOneER{}。'
    write('order', table('分块到达顺序下的期望收入', 'tab:order', header, rows, 'l' + 'r' * (len(pols) + 1), note=note,
                         size=r'\small'))
    r = d.loc['B213']
    for p in pols:
        num(f'Block{KEY[p]}', r[f'ER_{p}'])
        num(f'BlockEcone{KEY[p]}', r[f'Ec1_{p}'])
    for tag, seq in [('OTT', 'B123'), ('OTW', 'B132')]:
        r = d.loc[seq]
        for p in pols:
            num(f'Block{tag}{KEY[p]}', r[f'ER_{p}'])
        if 'Ec1_LP-Count' in r:
            num(f'Block{tag}EconeCount', r['Ec1_LP-Count'])
        num(f'Block{tag}EconeSeq', r['Ec1_Seq-DP'])
    # newsvendor view of the reservation for 30 type-2 customers: D2 ~ Bin(30, 0.4)
    k = np.arange(31)
    pmf = binom.pmf(k, 30, 0.4)
    cdf = np.cumsum(pmf)
    q_opt = int(np.argmax(cdf >= 0.1))

    def cost(q):
        return 9 * pmf @ np.maximum(q - k, 0) + 1 * pmf @ np.maximum(k - q, 0)
    num('NewsQuantile', q_opt, '{:d}')
    num('NewsLeft', pmf @ np.maximum(12 - k, 0), '{:.2f}')
    num('NewsCostMean', cost(12), '{:.2f}')
    num('NewsCostOpt', cost(q_opt), '{:.2f}')
    # figure: E c_{t,1} under B213
    paths = json.loads((RES / 'exact_paths.json').read_text())
    if (RES / 'exact_paths_count.json').exists():
        cp = json.loads((RES / 'exact_paths_count.json').read_text())
        paths['B213']['LP-Count'] = cp['B213']['LP-Count']
    show = [p for p in ['Greedy', 'LP-FD', 'LP-Dual', 'LP-Count', 'IID-DP', 'Seq-DP'] if p in paths['B213']]
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.35), gridspec_kw={'width_ratios': [1.25, 1]})
    t = np.arange(1, 102)
    for a in ax:
        for x in (31, 81):
            a.axvline(x, color=AXIS, lw=0.8)
    ends = []
    for p in show:
        y = np.array(paths['B213'][p]['mean_path'])[:, 0]
        for a in ax:
            a.plot(t, y, color=COLOR[p], ls=STYLE[p], lw=1.5 if p != 'Seq-DP' else 1.3)
        ends.append(y[-1])
    for x, lab in [(16, '类型 2'), (56, '类型 1'), (91, '类型 3')]:
        ax[0].text(x, 42, lab, ha='center', color=INK2, fontsize=8)
    ax[0].set(xlabel='期数 $t$', ylabel=r'$\mathbb{E}\,c_{t,1}$', xlim=(1, 101), ylim=(0, 45))
    ax[0].legend(handles=legend_handles(show), loc='lower left', fontsize=7.2)
    ax[1].set(xlabel='期数 $t$（局部放大）', xlim=(60, 101), ylim=(0, 10))
    end_labels(ax[1], 101, np.array(ends), show, gap=0.75)
    fig.tight_layout(w_pad=2)
    save_fig(fig, 'order_b213')


def sensitivity():
    s = pd.read_csv(RES / 'sensitivity_inventory.csv')
    s = s[s.sequence.isin(DS)]
    pols = ['Alg1', 'Greedy', 'LP-FD', 'LP-Dual', 'IID-DP']
    g = s.groupby('theta')
    rows = []
    agg = {}
    for theta, sub in g:
        C = json.loads(sub.C.iloc[0])
        L = {p: (sub.V_seq - sub[f'ER_{p}']).mean() for p in pols}
        agg[theta] = L | {'W': sub.oos_Alg1.mean(), 'F': sub.F.mean(), 'V': sub.V_seq.mean()}
        rows.append([f'{theta:.1f}', '$(' + ',\\,'.join(map(str, C)) + ')$', f'{sub.F.mean():.1f}', f3(sub.V_seq.mean())]
                    + [f3(L[p]) for p in pols] + [f'{sub.oos_Alg1.mean():.2f}'])
    header = [r'$\theta$', '$C$', '$F$', r'$\overline{V^{\seq}}$'] + [r'$\overline L^{' + TEX[p] + '}$' for p in pols] + [r'$\overline W^{\AlgOne}$']
    write('theta', table(r'库存缩放 $C\to\lfloor\theta C\rceil$ 下的策略损失（五份数据平均）', 'tab:theta', header, rows,
                         'cc' + 'r' * 8, size=r'\footnotesize'))
    for theta, key in [(0.5, 'Half'), (0.6, 'Six'), (0.7, 'Seven'), (0.8, 'Eight'), (1.0, 'One'), (1.2, 'Twelve')]:
        for p in pols:
            num(f'Theta{key}L{KEY[p]}', agg[theta][p])
        num(f'Theta{key}W', agg[theta]['W'], '{:.2f}')
    worst = max(agg, key=lambda th: agg[th]['LP-Dual'])
    num('ThetaWorstDual', worst, '{:.1f}')
    num('ThetaWorstDualL', agg[worst]['LP-Dual'])
    thetas = sorted(agg)
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.35), gridspec_kw={'width_ratios': [1.6, 1]})
    for p in ['Greedy', 'LP-FD', 'LP-Dual', 'IID-DP']:
        y = [agg[th][p] for th in thetas]
        ax[0].plot(thetas, y, color=COLOR[p], ls=STYLE[p], marker=MARKER[p], ms=4)
    ax[0].axvline(1, color=AXIS, lw=0.8)
    ax[0].legend(handles=legend_handles(['Greedy', 'LP-FD', 'LP-Dual', 'IID-DP']), loc='upper right', fontsize=7.2)
    ax[0].set(xlabel=r'库存缩放系数 $\theta$', ylabel=r'平均策略损失 $\overline{L}$', ylim=(-0.2, None))
    ax[1].plot(thetas, [agg[th]['Alg1'] for th in thetas], color=COLOR['Alg1'], marker=MARKER['Alg1'], ms=4)
    ax[1].set(xlabel=r'库存缩放系数 $\theta$', ylabel=r'算法 1 的平均策略损失')
    ax[1].axvline(1, color=AXIS, lw=0.8)
    fig.tight_layout(w_pad=2)
    save_fig(fig, 'theta')
    # K
    k = pd.read_csv(RES / 'sensitivity_K.csv')
    k = k[k.sequence.isin(DS)]
    rows = []
    for K, sub in k.groupby('K'):
        rows.append([int(K), f'{sub.F.mean():.1f}', f3(sub.V_seq.mean())] + [f3(sub[f'ER_{p}'].mean()) for p in pols])
        for p in pols:
            num(f'KER{["", "one", "two", "three"][int(K)]}{KEY[p]}', sub[f'ER_{p}'].mean())
        num(f'KV{["", "one", "two", "three"][int(K)]}', sub.V_seq.mean())
        num(f'KF{["", "one", "two", "three"][int(K)]}', sub.F.mean(), '{:.0f}')
    header = ['$K$', '$F$', r'$\overline{V^{\seq}}$'] + [r'$\overline{\E R}^{' + TEX[p] + '}$' for p in pols]
    write('K', table('推荐位数 $K$ 对期望收入的影响（五份数据平均）', 'tab:K', header, rows, 'c' + 'r' * 7, size=r'\small'))


def scaling():
    path = RES / 'scaling.csv'
    if not path.exists():
        return
    d = pd.read_csv(path)
    if 'diff_LP-Dual_Greedy' not in d:
        return
    rows, info = [], {}
    for T, sub in d.groupby('T'):
        reps = int(sub.reps.iloc[0])
        k = len(sub)

        def hw(col):
            """1.96 x pooled standard error of a mean over k sequences x reps replications."""
            return 1.96 * np.sqrt((sub[col] ** 2 / reps).sum()) / k
        G = {p: (sub.F - sub[f'mean_{p}']).mean() for p in ['Greedy', 'LP-FD', 'LP-Dual']}
        diff = sub['diff_LP-Dual_Greedy'].mean()
        h_diff, h_dual = hw('diffsd_LP-Dual_Greedy'), hw('sd_LP-Dual')
        gap = (sub.F - sub.V_seq).mean() if 'V_seq' in sub and sub.V_seq.notna().all() else np.nan
        C = np.floor(T / 100 * INS.capacities + 0.5).astype(int)
        info[T] = (G['LP-Dual'], h_dual, diff, h_diff)
        rows.append([int(T), '$(' + ','.join(map(str, C)) + ')$', f'{sub.F.mean():.0f}',
                     f"{G['Greedy']:.2f}", f"{G['LP-FD']:.2f}", f"${G['LP-Dual']:.2f}" + r'\pm' + f"{h_dual:.2f}$",
                     '$' + no_negative_zero(f'{diff:.2f}') + r'\pm' + f'{h_diff:.2f}$', f'{gap:.2f}' if np.isfinite(gap) else '--',
                     pct(sub['mean_LP-Dual'].mean() / sub.F.mean(), 2)])
    header = ['$T$', '$C$', '$F$', r'$\widehat G^{\Greedy}$', r'$\widehat G^{\LPFD}$', r'$\widehat G^{\LPDual}\pm h$',
              r'$\bar d^{\,\LPDual,\Greedy}\pm h$', r'$\overline{F-V^{\seq}}$', r'$\overline R^{\LPDual}/F$']
    paths = {int(T): int(sub.reps.iloc[0]) * len(sub) for T, sub in d.groupby('T')}
    counts = '、'.join(f'{v}' for v in sorted(set(paths.values()), reverse=True))
    note = (r'每个 $T$ 生成 10 条人数为 $T\lambda$ 的随机排列，各策略使用公共随机数，路径总数依 $T$ 由小到大为 ' + counts +
            r'；$\widehat G=F-\overline R$ 为模拟估计，$h$ 为合并 10 条序列后的 95\% 置信半宽。'
            r'$T>200$ 时动态规划的状态数过大，未计算 $V^{\seq}$。')
    write('scaling', table(r'销售期与库存同比例放大时的遗憾', 'tab:scaling', header, rows, 'rcrrrrrrr', note=note,
                           size=r'\footnotesize'))
    Tmax = int(d['T'].max())
    big = d[d['T'] == Tmax]
    num('ScaleTmax', Tmax, '{:d}')
    num('ScaleGreedyGapMax', info[Tmax][2], '{:.2f}')
    num('ScaleGreedyGapMaxHw', info[Tmax][3], '{:.2f}')
    num('ScaleDualHwMax', info[Tmax][1], '{:.1f}')
    num('ScaleRatioMax', 100 * big['mean_LP-Dual'].mean() / big.F.mean(), '{:.2f}')
    num('ScaleGDualMin', min(v[0] for v in info.values()), '{:.1f}')
    num('ScaleGDualMax', max(v[0] for v in info.values()), '{:.1f}')
    num('ScaleGreedyFluid', 0.0057 * Tmax, '{:.1f}')
    small = [T for T in info if T <= 100]
    num('ScaleGapSmall', max(abs(info[T][2]) for T in small), '{:.2f}')


def large():
    path = RES / 'large_instances.csv'
    if not path.exists():
        return
    d = pd.read_csv(path)
    paired = 'diff_LP-Dual_Greedy' in d
    rows = []
    for (m, n, T, K, rho), sub in d.groupby(['m', 'n', 'T', 'K', 'rho']):
        r = {p: (sub[f'mean_{p}'] / sub.F).mean() for p in ['Alg1', 'Greedy', 'LP-FD', 'LP-Dual']}
        row = [f'$({m},{n},{T},{K})$', f'{rho:.1f}'] + [f'{100 * r[p]:.1f}' for p in ['Alg1', 'Greedy', 'LP-FD', 'LP-Dual']]
        if paired:
            reps = sub.reps.to_numpy()
            rel = (sub['diff_LP-Dual_Greedy'] / sub.F).mean()
            hw = 1.96 * np.sqrt(((sub['diffsd_LP-Dual_Greedy'] / sub.F) ** 2 / reps).sum()) / len(sub)
            row.append('$' + no_negative_zero(f'{100 * rel:.2f}') + r'\pm' + f'{100 * hw:.2f}$')
        row += [f"{sub['oos_Alg1'].mean():.0f}", f"{sub['median_ms_LP-Dual'].mean():.2f}", f"{sub['p95_ms_LP-Dual'].mean():.2f}",
                f"{sub['median_ms_LP-FD'].mean():.2f}"]
        rows.append(row)
    header = ['$(m,n,T,K)$', r'$\rho$', r'\AlgOne', r'\Greedy', r'\LPFD', r'\LPDual']
    header += [r'$\bar d/F\pm h$'] if paired else []
    header += [r'$\overline W^{\AlgOne}$', r'$t^{\LPDual}_{50}$', r'$t^{\LPDual}_{95}$', r'$t^{\LPFD}_{50}$']
    k = 7 if paired else 6
    pre = [r' & & \multicolumn{4}{c}{$\overline R/F$ (\%)} & ' + (r'(\%) ' if paired else '') + r'& & \multicolumn{3}{c}{单期决策时间/ms} \\',
           r'\cmidrule(lr){3-6}' + f'\\cmidrule(lr){{{k + 2}-{k + 4}}}']
    note = (r'每种配置随机生成 8 个实例，每个实例 ' + f'{int(d.reps.max()) if "reps" in d else 40}' +
            r' 次购买重复（$m=20$ 时减半），各策略使用公共随机数。$\rho$ 为库存相对于算法 1 期望需求的比例；'
            r'$\bar d/F$ 为 \LPDual{} 减 \Greedy{} 的配对收入差占 $F$ 的百分比，$h$ 为 95\% 置信半宽；'
            r'$\overline W$ 为每条路径上缺货推荐的平均次数；$t_{50}$、$t_{95}$ 为单期决策时间的中位数与 95\% 分位数，包含线性规划求解与排序。')
    write('large', table('随机生成的较大实例', 'tab:large', header, rows, 'lr' + 'r' * (len(header) - 2), note=note, pre=pre,
                         size=r'\footnotesize'))
    big = d[d.m == d.m.max()]
    num('LargeTimeMedian', big['median_ms_LP-Dual'].mean(), '{:.2f}')
    num('LargeTimeP', big['p95_ms_LP-Dual'].mean(), '{:.2f}')
    num('LargeTimeFD', big['median_ms_LP-FD'].mean(), '{:.1f}')
    num('LargeVars', int((big.m * big.n).max()), '{:d}')
    num('LargeRows', int((big.m + big.n).max()), '{:d}')
    num('LargeAlgOneMin', 100 * (d.mean_Alg1 / d.F).groupby([d.m, d.rho]).mean().min(), '{:.1f}')
    num('LargeAlgOneMax', 100 * (d.mean_Alg1 / d.F).groupby([d.m, d.rho]).mean().max(), '{:.1f}')
    num('LargeGreedyMin', 100 * (d.mean_Greedy / d.F).groupby([d.m, d.rho]).mean().min(), '{:.1f}')
    if paired:
        rel = d['diff_LP-Dual_Greedy'] / d.F
        se = d['diffsd_LP-Dual_Greedy'] / d.F / np.sqrt(d.reps)
        num('LargePairMin', 100 * rel.groupby([d.m, d.rho]).mean().min(), '{:.2f}')
        num('LargePairMax', 100 * rel.groupby([d.m, d.rho]).mean().max(), '{:.2f}')
        num('LargeSigWins', int((rel > 1.96 * se).sum()), '{:d}')
        num('LargeSigLosses', int((rel < -1.96 * se).sum()), '{:d}')
        num('LargeInstances', len(d), '{:d}')


def verification():
    path = RES / 'verification.json'
    if not path.exists():
        return
    v = json.loads(path.read_text())
    total = sum(p['decisions'] for p in v['policies'].values())
    bad = sum(p['mismatches'] for p in v['policies'].values())
    num('VerifyDecisions', f'{total:,}'.replace(',', '{,}'))
    num('VerifyMismatch', bad, '{:d}')
    num('ExactSeconds', v['exact_forward_seconds'], '{:.2f}')
    num('IIDSeconds', v['iid_dp_seconds'], '{:.1f}')
    meta = json.loads((RES / 'exact_meta.json').read_text())
    num('TableSeconds', meta['lp_table_seconds'], '{:.0f}')


def write_numbers():
    lines = ['% generated by code/make_assets.py -- do not edit by hand']
    lines += [f'\\newcommand{{\\{k}}}{{{NUM[k]}}}' for k in sorted(NUM)]
    (GEN / 'numbers.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    for step in [instance_facts, q1, q2, policy_behaviour, main_results, loss_analysis, random_sequences,
                 block_order_results, sensitivity, scaling, large, verification]:
        step()
        print(f'  {step.__name__}: ok', flush=True)
    write_numbers()
    print(f'{len(NUM)} numbers written.')
