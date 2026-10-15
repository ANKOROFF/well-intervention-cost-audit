"""Pump-component multiple sweep on the main ex-ante chain."""
import config as cfg
import contextlib
import csv
import io
import pathlib
import sys
import tempfile
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import ep19_crossruler as e19
from ep06_ablation import _цены
LAMBDAS = [0, 0.25, 0.5, 1, 1.5, 2]
GC_HEAD, TOL, HALF = (0.2631, 0.0001, 0.084)
OUT = ROOT / cfg.out('lambda_sweep.tsv')

def pump_share(r, pump_ids):
    npv = _цены(r)
    inc = {True: 0.0, False: 0.0}
    for g in r['отборы']:
        for line, sign in ((e19.LINE, 1.0), ('факт инженеров', -1.0)):
            for i in g['op_id'][g['отбор'][line]].tolist():
                inc[i in pump_ids] += sign * npv[i]
    tot = inc[True] + inc[False]
    return inc[True] / tot if tot else float('nan')

def main() -> int:
    base = pl.read_parquet(e19.CAP_ПП)
    pump_ids = set(base.filter(pl.col('c_насос').fill_null(0.0) > 0)['op_id'].to_list())
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='lam_'))
    res = {}
    for lam in LAMBDAS:
        c = base.with_columns(capex=pl.col('capex') + lam * pl.col('c_насос').fill_null(0.0))
        pa = tmp / f'l{lam}.parquet'
        c.write_parquet(pa)
        with contextlib.redirect_stdout(io.StringIO()):
            r = e19.прогон_exante(f'lam{lam}', e19.параметры(str(pa)))
        res[lam] = r
    g1 = res[0]['линии'][e19.LINE]
    print(f"РЕГРЕССИЯ: GC(λ=0) = {g1['GC']:.4f} [{g1['lo']:.4f}, {g1['hi']:.4f}] против {GC_HEAD}")
    if abs(g1['GC'] - GC_HEAD) > TOL:
        print('ОСТАНОВ: регрессия не сошлась')
        return 1
    rows, viol = ([], [])
    for lam in LAMBDAS:
        r = res[lam]
        g = r['линии'][e19.LINE]
        d = g['GC'] - g1['GC']
        bad = g['lo'] <= 0 or abs(d) > HALF
        if bad:
            viol.append(lam)
        rows.append([lam, f"{g['GC']:.4f}", f"{g['lo']:.4f}", f"{g['hi']:.4f}", f'{d:+.4f}', f"{g['p']:.4f}", f'{pump_share(r, pump_ids):.3f}', 'нарушено' if bad else 'в пределах'])
    star = f'λ* = {viol[0]} (наименьшее нарушившее в сетке)' if viol else 'устойчиво до λ = 2'
    rows.append(['итог', '', '', '', '', '', '', f'{star}; насосных операций {len(pump_ids)} из {base.height}'])
    with open(OUT, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['lambda', 'GC', 'GC_lo', 'GC_hi', 'dGC', 'DM_p', 'pump_share_of_gain', 'outcome'])
        w.writerows(rows)
    for r in rows:
        print(' | '.join(map(str, r)))
    return 0
if __name__ == '__main__':
    sys.exit(main())
