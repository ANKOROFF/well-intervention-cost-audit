"""Fig. 3: operations of selection year 2024 whose selection flips between the cost representations. Rebuilds the ex-ante chain with the project modules, keeps the per-operation table only in a temporary directory and deletes it; writes only counts and shares."""
import os, pathlib, shutil, sys
import numpy as np
import polars as pl
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config as cfg
import ep19_crossruler as e19
import stats
from ep06_ablation import _цены
from sdt_analysis import цепь
from style import *
YEAR, N_DRAWS, EXPECT_DIFF = (2024, 500, 229)
import tempfile
SCRATCH = pathlib.Path(tempfile.mkdtemp(prefix='flips_'))
OUT_TXT = pathlib.Path(cfg.out('flips_2024_summary.txt'))
C = float(pl.read_parquet(cfg.data('interim/capex_physical.parquet'))['capex'].sum())
LINE = e19.LINE
SCRATCH.mkdir(exist_ok=True)
tab = SCRATCH / 'ops_2024.parquet'
if not tab.exists():
    log, tag = ([], [None])
    _greedy, _year_run = (stats.greedy, stats.year_run)

    def greedy_rec(score, capex, dur, budget, cap, positive=True):
        r = _greedy(score, capex, dur, budget, cap, positive)
        if positive:
            log.append((tag[0], np.array(score), np.array(dur), r.copy(), budget, cap, np.array(capex)))
        return r

    def year_run_rec(d, руб_т, Y, *a, **k):
        if Y == 2016:
            tag[0] = (0 if tag[0] is None else tag[0][0] + 1, Y)
        tag[0] = (tag[0][0], Y)
        return _year_run(d, руб_т, Y, *a, **k)
    stats.greedy, stats.year_run = (greedy_rec, year_run_rec)
    cap = pl.read_parquet(e19.CAP_ПП)
    item, day = цепь('flips', cap)
    stats.greedy, stats.year_run = (_greedy, _year_run)

    def hat(run, res, Y):
        g = next((g for g in res['отборы'] if g['год'] == Y))
        calls = [c for c in log if c[0] == (run, Y)]
        j = next((k for k, c in enumerate(calls) if (c[3] == g['отбор'][LINE]).all() and k >= 2))
        sc, dur = (calls[j][1], calls[j][2])
        h = sc * dur
        assert np.allclose(h, calls[j - 2][1])
        return (dict(zip(g['op_id'].tolist(), h)), calls[j][4], calls[j][5], g)
    h_i, bud_i, cap_i, g24i = hat(0, item, YEAR)
    h_d, bud_d, cap_d, g24d = hat(1, day, YEAR)
    p_i, p_d = (_цены(item), _цены(day))
    sel = lambda r: {(g['год'], i) for g in r['отборы'] for i in g['op_id'][g['отбор'][LINE]].tolist()}
    S_i, S_d = (sel(item), sel(day))
    diff_all = len(S_i ^ S_d)
    per_year = {y: len({i for g, i in S_i ^ S_d if g == y}) for y in sorted({g['год'] for g in item['отборы']})}
    g24 = next((g for g in item['отборы'] if g['год'] == YEAR))
    ids = g24['op_id'].tolist()
    freq = dict(pl.read_parquet(cfg.data('interim/ep30_pselect_ops.parquet')).iter_rows())
    pl.DataFrame({'id': ids, 'npv_i': [h_i[i] / C for i in ids], 'npv_d': [h_d[i] / C for i in ids], 'real_i': [p_i[i] / C for i in ids], 'sel_i': [(YEAR, i) in S_i for i in ids], 'sel_d': [(YEAR, i) in S_d for i in ids], 'freq': [freq.get(i, 0) / N_DRAWS for i in ids]}).write_parquet(tab)
    fact = {i for g in item['отборы'] for i in g['op_id'][g['отбор']['факт инженеров']].tolist()}
    U = set(freq) | fact
    sens = {i for i in U if 0.05 < freq.get(i, 0) / N_DRAWS < 0.95}
    dis = {i for _, i in S_i ^ S_d}
    pool24 = set(ids)
    dis24 = {i for y, i in S_i ^ S_d if y == YEAR}
    agg = dict(pool_all=sum((len(g['op_id']) for g in item['отборы'])), fact_all=len(fact), dis=len(dis), sens=len(sens), inter=len(dis & sens), union=len(dis | sens), sens24=len(sens & pool24), dis24=len(dis24), inter24=len(dis24 & sens), union24=len(dis24 | sens & pool24), nz=len(pool24))
    pl.DataFrame({k: [v] for k, v in agg.items()}).write_parquet(SCRATCH / 'tot.parquet')
    print('aggregates:', agg)
    print('per-year symmetric difference:', per_year, 'total', diff_all, '|S_i|,|S_d|', len(S_i), len(S_d))
    if diff_all != EXPECT_DIFF:
        print(f'STOP: nine-year disagreement {diff_all} != manuscript {EXPECT_DIFF}')
        sys.exit(2)
d = pl.read_parquet(tab).sort('npv_i', descending=True)
x = np.arange(1, d.height + 1)
ni, nd = (d['npv_i'].to_numpy(), d['npv_d'].to_numpy())
_si, _sd = (d['sel_i'].to_numpy(), d['sel_d'].to_numpy())
print('selected with forecast <= 0 (itemized, day-rate):', int((_si & (ni <= 0)).sum()), int((_sd & (nd <= 0)).sum()), '| unselected with forecast > 0:', int((~_si & (ni > 0)).sum()), int((~_sd & (nd > 0)).sum()))
si, sd = (d['sel_i'].to_numpy(), d['sel_d'].to_numpy())
both, only_i, only_d = (si & sd, si & ~sd, ~si & sd)
none = ~si & ~sd
fq = d['freq'].to_numpy()
a = pl.read_parquet(SCRATCH / 'tot.parquet').row(0, named=True)
sens_g = [int(((fq[m] > 0.05) & (fq[m] < 0.95)).sum()) for m in (both, only_i, only_d)]
sizes = [int(both.sum()), int(only_i.sum()), int(only_d.sum())]
lines = ['# 2024 pool (forecast NPV = response-model forecast, itemized representation; shares of C in per cent)', f'pool_size {d.height}', f'kept_by_neither {int(none.sum())}', f'group_size both {sizes[0]}', f'group_size itemized_only {sizes[1]}', f'group_size dayrate_only {sizes[2]}', f'group_forecast_sum_pct_C both {100 * ni[both].sum():.2f}', f'group_forecast_sum_pct_C itemized_only {100 * ni[only_i].sum():.2f}', f'group_forecast_sum_pct_C dayrate_only_under_itemized {100 * ni[only_d].sum():.2f}', f'group_forecast_sum_pct_C dayrate_only_under_dayrate {100 * nd[only_d].sum():.2f}', f'mean_draw_freq both {fq[both].mean():.3f}', f'mean_draw_freq itemized_only {fq[only_i].mean():.3f}', f'mean_draw_freq dayrate_only {fq[only_d].mean():.3f}', f'draw_sensitive_count both {sens_g[0]} of {sizes[0]}', f'draw_sensitive_count itemized_only {sens_g[1]} of {sizes[1]}', f'draw_sensitive_count dayrate_only {sens_g[2]} of {sizes[2]}', f'draw_sensitive_share both {sens_g[0] / sizes[0]:.3f}', f'draw_sensitive_share itemized_only {sens_g[1] / sizes[1]:.3f}', f'draw_sensitive_share dayrate_only {sens_g[2] / sizes[2]:.3f}', f'forecast_sign_changes {int((np.sign(ni[only_i | only_d]) != np.sign(nd[only_i | only_d])).sum())} of {sizes[1] + sizes[2]}', '# 2024 alone', f"disagreement {a['dis24']}", f"draw_sensitive_in_pool {a['sens24']}", f"intersection {a['inter24']}", f"union {a['union24']}", f"union_share_of_pool {a['union24'] / a['nz']:.3f} of {a['nz']}", '# nine selection years (5,806-operation chain)', f"pool_operations_in_nine_years {a['pool_all']}", f"engineers_program_operations {a['fact_all']}", f"disagreement {a['dis']}", f"draw_sensitive {a['sens']}", f"intersection {a['inter']}", f"union {a['union']}", f"union_share_of_engineers_program {a['union'] / a['fact_all']:.3f}"]
OUT_TXT.write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
print('pool', d.height, 'neither', none.sum(), 'mean price-draw freq both/it/dr:', [round(float(fq[m].mean()), 3) if m.any() else None for m in (both, only_i, only_d)], 'share with 0.05<freq<0.95:', [round(float(((fq[m] > 0.05) & (fq[m] < 0.95)).mean()), 3) if m.any() else None for m in (both, only_i, only_d)], 'sign differs in disagreement:', int((np.sign(ni[only_i | only_d]) != np.sign(nd[only_i | only_d])).sum()))
ni, nd = 100 * ni, 100 * nd   # plotted in per cent of C
YLO, YHI = -0.15, ni.max() + 0.01
off = ni < YLO                    # drawn as an open triangle on the lower edge
assert not (nd < YLO).any() and off.sum() == 1
fig, ax = plt.subplots(figsize=(W2, 3.3))
grid(ax)
ax.axhline(0, color=INK2, lw=0.6, zorder=2)
ax.plot(x[none & ~off], ni[none & ~off], ".", ms=2, color=GRAY, label="kept by neither", rasterized=True, zorder=2)
ax.plot(x[off], [YLO] * int(off.sum()), "v", ms=4.5, mfc="white", mec=INK, mew=0.7, ls="none", clip_on=False,
        label="off scale (below range)", zorder=5)
ax.plot(x, nd, "-", lw=LW_THIN, color=DAY, label="day-rate forecast (line)", zorder=2)
ax.plot(x[both], ni[both], "o", ms=2.6, mfc="none", mec=INK2, mew=0.5, label="kept by both", zorder=3)
ax.plot(x[only_i], ni[only_i], "s", ms=4.2, mfc=ITEM, mec=INK, mew=0.5, label="kept only by itemized", zorder=4)
ax.plot(x[only_d], ni[only_d], "^", ms=5, mfc=DAY, mec=INK, mew=0.5, label="kept only by day-rate", zorder=4)
ax.set_xlabel("Operations of the 2024 pool, ranked by itemized forecast")
ax.set_ylabel("Forecast NPV, % of C")
ax.set_ylim(YLO, YHI)
ax.set_xlim(0, d.height + 1)
legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, markerscale=1.1)
save(fig, "fig_flips.png")
shutil.rmtree(SCRATCH, ignore_errors=True)
