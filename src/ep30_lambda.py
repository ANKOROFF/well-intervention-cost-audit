"""Threshold multiple of the pump price for omitted pump components."""
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
from lambda_sweep import pump_share
LAMBDAS = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
GC_HEAD, LO_HEAD, HI_HEAD, ЗНАМ_HEAD, TOL, HALF = (0.2631, 0.1772, 0.3453, float('nan'), 0.0001, 0.084)
ПРАВДОПОДОБНО = 1.0
OUT_MD = ROOT / cfg.results('ep30_lambda.md')
OUT_TSV = ROOT / cfg.out('ep30_lambda.tsv')
ПРОЧЬ = []

def вывод(s=''):
    ПРОЧЬ.append(s)
    print(s)

def якорь(имя, знач, ф='.4f'):
    вывод(f'ЯКОРЬ {имя} = {format(знач, ф)}')

def множество(r):
    return {i for g in r['отборы'] for i in g['op_id'][g['отбор'][e19.LINE]].tolist()}

def main() -> int:
    base = pl.read_parquet(e19.CAP_ПП)
    pump_ids = set(base.filter(pl.col('c_насос').fill_null(0.0) > 0)['op_id'].to_list())
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='ep30lam_'))
    ст = {}
    for lam in LAMBDAS:
        c = base.with_columns(capex=pl.col('capex') + lam * pl.col('c_насос').fill_null(0.0))
        pa = tmp / f'l{lam}.parquet'
        c.write_parquet(pa)
        with contextlib.redirect_stdout(io.StringIO()):
            r = e19.прогон_exante(f'lam{lam}', e19.параметры(str(pa)))
        g = r['линии'][e19.LINE]
        ст[lam] = (r, g)
        print(f"λ={lam}: GC {g['GC']:.4f} [{g['lo']:.4f}, {g['hi']:.4f}] повт {g['повторов']}", flush=True)
        if lam == 0:
            ok = abs(g['GC'] - GC_HEAD) <= TOL and abs(g['lo'] - LO_HEAD) <= TOL and (abs(g['hi'] - HI_HEAD) <= TOL) and (r['n'] == 5806) and (r['лет'] == 9) and (abs(r['разрыв'] - ЗНАМ_HEAD) <= TOL)
            print(f"РЕГРЕССИЯ: n {r['n']}, лет {r['лет']}, знаменатель {r['разрыв']:.4f}, {('сошлась' if ok else 'НЕ СОШЛАСЬ')}")
            if not ok:
                print('ОСТАНОВ: регрессия на головное число не сошлась')
                return 1
    with open(ROOT / cfg.out('lambda_sweep.tsv'), encoding='utf-8') as fh:
        старый = {float(r['lambda']): r for r in csv.DictReader(fh, delimiter='\t') if r['lambda'] != 'итог'}
    for lam, r0 in старый.items():
        g = ст[int(lam) if lam == int(lam) else lam][1]
        for k, f in (('GC', 'GC'), ('lo', 'GC_lo'), ('hi', 'GC_hi')):
            if abs(g[k] - float(r0[f])) > 5e-05:
                print(f'ОСТАНОВ: узел λ={lam} {k} {g[k]:.4f} против файла {r0[f]}')
                return 1
    print('РЕГРЕССИЯ узлов 0.25-2 против lambda_sweep.tsv: сошлась')
    r0, g0 = ст[0]
    S0 = множество(r0)
    вывод('# Б1.2. Порог λ* по недостающим компонентам ЭЦН (ep30_lambda)')
    вывод('')
    вывод(f"Цепь ППП, n {r0['n']}, лет {r0['лет']}, знаменатель {r0['разрыв']:.4f} млрд руб; насосных операций в совокупности {len(pump_ids & set(r0['d']['op_id'].to_list()))}; сетка закрыта, λ_max = 8. Регрессия на 0.2631 и на узлы 0.25-2 lambda_sweep.tsv сошлась.")
    вывод('| λ | GC | lo | hi | повт. | ДМ p | dGC | NPV отбора млрд | число расхождений отбора | Жаккар | доля насосных в приросте |')
    вывод('|---|---|---|---|---|---|---|---|---|---|---|')
    rows, э = ([], {'Э1': None, 'Э2': None, 'Э3': None, 'Э4': None})
    for lam in LAMBDAS:
        r, g = ст[lam]
        S = множество(r)
        sd, jac = (len(S ^ S0), len(S & S0) / len(S | S0))
        d = g['GC'] - g0['GC']
        ps = pump_share(r, pump_ids)
        rows.append([lam, f"{g['GC']:.4f}", f"{g['lo']:.4f}", f"{g['hi']:.4f}", g['повторов'], f"{g['p']:.4f}", f'{d:+.4f}', f"{g['NPV']:.4f}", sd, f'{jac:.4f}', f'{ps:.3f}', len(S)])
        вывод(f"| {lam} | {g['GC']:.4f} | {g['lo']:.4f} | {g['hi']:.4f} | {g['повторов']} | {g['p']:.4f} | {d:+.4f} | {g['NPV']:.4f} | {sd} | {jac:.4f} | {ps:.3f} |")
        if lam > 0:
            if э['Э1'] is None and S != S0:
                э['Э1'] = lam
            if э['Э2'] is None and g['lo'] <= 0:
                э['Э2'] = lam
            if э['Э3'] is None and g['GC'] <= 0:
                э['Э3'] = lam
            if э['Э4'] is None and abs(d) > HALF:
                э['Э4'] = lam
    вывод('')
    for k, lam in э.items():
        if lam is None:
            вывод(f'{k}: не наступило на сетке до λ_max = 8')
        else:
            вывод(f'{k}: узел λ = {lam}, интервал узлов ({LAMBDAS[LAMBDAS.index(lam) - 1]}, {lam}]')
    нар = [lam for k, lam in э.items() if k != 'Э1' and lam is not None]
    if not нар:
        исход = 'Л-1: до λ_max событий нет; «не достигнут до λ_max = 8» (восьмикратная добавка к насосной компоненте, далеко за правдоподобной областью λ ≤ 1)'
    elif min(нар) <= ПРАВДОПОДОБНО:
        исход = 'Л-2: событие внутри правдоподобной области (узел λ ≤ 1); ' + ', '.join((f'{k} на λ = {э[k]}' for k in ('Э2', 'Э3', 'Э4') if э[k] is not None and э[k] <= ПРАВДОПОДОБНО))
    else:
        m = min(нар)
        исход = f'Л-3: первое событие вне правдоподобной области: вывод держится до λ_ниж = {LAMBDAS[LAMBDAS.index(m) - 1]}, нарушается на λ_верх = {m}; события: ' + ', '.join((f'{k} на λ = {э[k]}' for k in ('Э2', 'Э3', 'Э4') if э[k] is not None))
    вывод(f'ИСХОД: {исход}')
    вывод('')
    вывод('Ограничение: добавку получают операции с c_насос > 0; R_i есть собственная цена модели (price_registry, T3, независимых источников 0), λ - чувствительность к недостающим компонентам в долях этой цены, не оценка их стоимости. Данных по кабелю, ПЭД и НКТ нет.')
    ps = [float(x[10]) for x in rows]
    вывод('')
    якорь('ep30 lambda GC λ=0', g0['GC'])
    for lam in LAMBDAS[1:]:
        g = ст[lam][1]
        якорь(f'ep30 lambda GC λ={lam}', g['GC'])
        якорь(f'ep30 lambda GC_lo λ={lam}', g['lo'])
        якорь(f'ep30 lambda GC_hi λ={lam}', g['hi'])
    якорь('ep30 lambda повторов λ=0', g0['повторов'], 'd')
    якорь('ep30 lambda λ_max', LAMBDAS[-1], 'g')
    якорь('ep30 lambda доля насосных min', min(ps), '.3f')
    якорь('ep30 lambda доля насосных max', max(ps), '.3f')
    якорь('ep30 lambda Жаккар λ=8', float(rows[-1][9]))
    якорь('ep30 lambda расхождение отбора λ=8', rows[-1][8], 'd')
    for k, lam in э.items():
        якорь(f'ep30 lambda {k} узел (-1 = не наступило)', -1 if lam is None else lam, 'g')
    OUT_MD.write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
    with open(OUT_TSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['lambda', 'GC', 'GC_lo', 'GC_hi', 'повторов', 'DM_p', 'dGC', 'NPV_отбора_млрд', 'расхождение_отбора', 'Жаккар', 'доля_насосных', 'операций_отбора'])
        w.writerows(rows)
    return 0
if __name__ == '__main__':
    sys.exit(main())
