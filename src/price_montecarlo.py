"""Probabilistic price draws on the main ex-ante chain."""
import config as cfg
import argparse
import contextlib
import csv
import io
import multiprocessing as mp
import os
import pathlib
import sys
import tempfile
from dataclasses import fields, replace
for _v in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import functools
import lightgbm as lgb
import numpy as np
import polars as pl
_lgb_init = lgb.LGBMRegressor.__init__

@functools.wraps(_lgb_init)
def _lgb_one_thread(self, *a, **k):
    k.setdefault('n_jobs', 1)
    _lgb_init(self, *a, **k)
lgb.LGBMRegressor.__init__ = _lgb_one_thread
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import capex_exante as cx
import ep19_crossruler as e19
from cost_physical import Prices, capex
from ep06_ablation import калибровать, написать_вариант
SEED = 20261012
GC_HEAD, TOL = (0.2631, 0.0001)
REG = ROOT / cfg.out('price_registry.tsv')
OUT_DRAWS = ROOT / cfg.out('price_montecarlo.tsv')
PRICES = ['проппант_руб_т', 'проппант_rcp_руб_т', 'гель_руб_м3', 'кислота_руб_м3', 'азот_руб_т', 'цемент_руб_м3', 'насос_руб_м3сут_м', 'бурение_руб_м', 'крс_руб_сутки', 'флот_грп_руб_сутки', 'станок_збс_руб_сутки', 'мобилизация_руб']
TYPES = ['ВБД', 'ВПН', 'ГРП', 'ДП', 'ЗБС', 'ЛА', 'ЛНЭК', 'ОПЗ', 'ОПТ', 'ППР', 'ПиП', 'РИР']
DIV_MIN, R_SHARE = (100, 0.95)
_T = _IDS = _TMP = None

def num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None

def spec():
    d = pl.read_csv(REG, separator='\t', infer_schema_length=0)
    base = Prices()
    out = {}
    for p in PRICES:
        r = d.filter(pl.col('price') == p)
        pick = None
        for lvl in ('T1', 'T2'):
            c = r.filter(pl.col('level_effective') == lvl)
            if p == 'проппант_руб_т':
                c = c.filter(pl.col('note').str.contains('нет слова RCP'))
            c = [x for x in c.iter_rows(named=True) if num(x['median']) is not None]
            if c:
                pick = (lvl, max(c, key=lambda x: int(x['n_independent_sources'] or 0)))
                break
        if pick is None:
            out[p] = ('T3', getattr(base, p), None)
        elif pick[0] == 'T1':
            q25, q75 = (num(pick[1]['q25']), num(pick[1]['q75']))
            out[p] = ('T1', (np.log(q25) + np.log(q75)) / 2, (np.log(q75) - np.log(q25)) / 1.349)
        else:
            m = num(pick[1]['median'])
            out[p] = ('T2', m, (0.5, 2.0) if p == 'флот_грп_руб_сутки' else (0.7, 1.3))
    return out

def draw(sp, rng):
    pr = {}
    for p, (lvl, a, b) in sp.items():
        if lvl == 'T1':
            pr[p] = float(np.exp(rng.normal(a, b)))
        elif lvl == 'T2':
            pr[p] = float(a * rng.uniform(*b))
        else:
            pr[p] = float(a * np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    return pr

def init():
    global _T, _IDS, _TMP
    ids = pl.read_parquet(e19.CAP_ПП)['op_id'].to_list()
    vp = pl.read_parquet(cx.VOL_SRC)
    with contextlib.redirect_stdout(io.StringIO()):
        _T = cx.подменить_объёмы(cx.ресурсы(True), vp).filter(pl.col('op_id').is_in(ids))
    _IDS = set(ids)
    _TMP = pathlib.Path(tempfile.mkdtemp(prefix='pmc_'))

def sel_sets(r):
    s = set()
    for g in r['отборы']:
        s |= set(g['op_id'][g['отбор'][e19.LINE]].tolist())
    return s

def cell(args):
    i, pr = args
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c = capex(_T, replace(Prices(), **pr))
        pa = _TMP / f'd{i}.parquet'
        c.write_parquet(pa)
        наш = e19.прогон_exante(f'd{i}', e19.параметры(str(pa)))
        ставка, _ = калибровать(c, set(наш['d']['op_id'].to_list()), 'dur')
        pl1 = _TMP / f'l{i}.parquet'
        написать_вариант(c, 'dur', ставка, pl1)
        л1 = e19.прогон_exante(f'l{i}', e19.параметры(str(pl1)))
    g = наш['линии'][e19.LINE]
    a, b = (sel_sets(наш), sel_sets(л1))
    sh = c.group_by('тип').agg(m=pl.col('c_материалы_исп').sum(), k=pl.col('capex').sum()).with_columns(s=pl.col('m') / pl.col('k'))
    shm = dict(zip(sh['тип'].to_list(), sh['s'].to_list()))
    for f in (pa, pl1):
        f.unlink()
    return {'i': i, 'pr': pr, 'n': наш['n'], 'GC': g['GC'], 'lo': g['lo'], 'hi': g['hi'], 'div': len(a ^ b), 'jac': len(a & b) / max(len(a | b), 1), 'share': {t: shm.get(t, float('nan')) for t in TYPES}}

def write_draws(path, rs):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['draw', *PRICES, 'n', 'GC', 'GC_lo', 'GC_hi', 'sel_diff', 'jaccard'] + [f'mat_{t}' for t in TYPES])
        for r in rs:
            w.writerow([r['i'], *[f"{r['pr'][p]:.6g}" for p in PRICES], r['n'], f"{r['GC']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", r['div'], f"{r['jac']:.4f}"] + [f"{r['share'][t]:.4f}" for t in TYPES])

def rank(x):
    return np.argsort(np.argsort(x)).astype(float)

def spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(rank(x), rank(y))[0, 1])

def r2_ranks(Xs, y):
    X = np.column_stack([np.ones(len(y))] + [rank(x) for x in Xs])
    yy = rank(y)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    res = yy - X @ beta
    return float(1 - res.var() / yy.var())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    sp = spec()
    for p, (lvl, x, y) in sp.items():
        print(f'{p}: {lvl}')
    rng = np.random.default_rng(SEED)
    draws = [(i, draw(sp, rng)) for i in range(1, a.n + 1)]
    base_pr = {p: getattr(Prices(), p) for p in PRICES}
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.jobs, initializer=init) as pool:
        r0 = pool.apply(cell, ((0, base_pr),))
        print(f"РЕГРЕССИЯ: GC(Prices()) = {r0['GC']:.4f} против {GC_HEAD}")
        if abs(r0['GC'] - GC_HEAD) > TOL:
            print('ОСТАНОВ: регрессия не сошлась')
            return 1
        res = []
        for r in pool.imap(cell, draws, chunksize=1):
            res.append(r)
            if len(res) % 5 == 0:
                print(f'розыгрышей {len(res)} из {a.n}', flush=True)
            if len(res) % 50 == 0:
                write_draws(OUT_DRAWS.with_suffix('.partial.tsv'), [r0] + res)
    write_draws(OUT_DRAWS, [r0] + res)
    OUT_DRAWS.with_suffix('.partial.tsv').unlink(missing_ok=True)
    gc = np.array([r['GC'] for r in res])
    lo = np.array([r['lo'] for r in res])
    dv = np.array([r['div'] for r in res])
    jc = np.array([r['jac'] for r in res])
    sh = np.array([[r['share'][t] for t in TYPES] for r in res])
    r1 = float(np.mean((gc > 0) & (lo > 0)))
    r2 = float(np.mean(dv >= DIV_MIN))
    pos = np.nan_to_num(sh, nan=0.0) > 0
    r3 = bool(pos.all())
    rows = [['n_draws', str(len(res)), '', '', 'цель правила 500' + ('' if len(res) == 500 else '; ОТСТУПЛЕНИЕ от правила')]]
    for nm, v in (('GC', gc), ('GC_lo', lo), ('sel_diff', dv.astype(float)), ('jaccard', jc)):
        q = np.percentile(v, [5, 50, 95])
        rows.append([nm, f'{q[0]:.4f}', f'{q[1]:.4f}', f'{q[2]:.4f}', ''])
    for j, t in enumerate(TYPES):
        q = np.percentile(np.nan_to_num(sh[:, j]), [5, 50, 95])
        rows.append([f'mat_{t}', f'{q[0]:.4f}', f'{q[1]:.4f}', f'{q[2]:.4f}', f'розыгрышей с долей>0: {int(pos[:, j].sum())}'])
    rows.append(['R1 (GC>0 и lo>0 в >=95%)', f'{r1:.3f}', '', '', 'выполнено' if r1 >= R_SHARE else 'нет'])
    rows.append(['R2 (расхождение>=100 в >=95%)', f'{r2:.3f}', '', '', 'выполнено' if r2 >= R_SHARE else 'нет'])
    rows.append(['R3 (доля материалов>0 во всех 12 типах)', f'типов с долей>0 во всех розыгрышах: {int(pos.all(axis=0).sum())} из 12', '', '', 'выполнено' if r3 else 'нет'])
    P = {p: np.array([r['pr'][p] for r in res]) for p in PRICES}
    for p in PRICES:
        rows.append([f'spearman GC~{p} [{sp[p][0]}]', f'{spearman(P[p], gc):+.3f}', f'lo: {spearman(P[p], lo):+.3f}', '', ''])
    for g in ('T1', 'T2', 'T3'):
        ps = [p for p in PRICES if sp[p][0] == g]
        if ps:
            rows.append([f'R2_рангов GC~группа {g} ({len(ps)} цен)', f'{r2_ranks([P[p] for p in ps], gc):.3f}', f'lo: {r2_ranks([P[p] for p in ps], lo):.3f}', '', ', '.join(ps)])
    with open(ROOT / cfg.out('price_montecarlo_summary.tsv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['metric', 'q05_or_value', 'q50', 'q95', 'note'])
        w.writerows(rows)
    for r in rows:
        print(' | '.join(r))
    return 0
if __name__ == '__main__':
    sys.exit(main())
