"""Partial identification of the sign of NPV over the admissible box of prices."""
import config as cfg
import contextlib
import csv
import io
import math
import pathlib
import sys
from dataclasses import replace
import numpy as np
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import ep19_crossruler as e19
import price_montecarlo as pmc
import stats
from cost_physical import Prices, capex
GC_HEAD, TOL, N_HEAD, ЛЕТ_HEAD = (0.2631, 0.0001, 5806, 9)
ПОРОГ_УЗ, ПОРОГ_ВЕРХ = (0.25, 0.5)
SEED_ВНУТР, N_ВНУТР = (20261013, 200)
МАЛО = 10
OUT_MD = ROOT / cfg.results('ep30_ident.md')
OUT_TSV = ROOT / cfg.out('ep30_ident.tsv')
ПРОЧЬ = []

def вывод(s=''):
    ПРОЧЬ.append(s)
    print(s)

def якорь(имя, знач, ф='.4f'):
    вывод(f'ЯКОРЬ {имя} = {format(знач, ф)}')

def уилсон(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)

def коробка():
    base = Prices()
    out = {}
    for p, (lvl, a, b) in pmc.spec().items():
        if lvl == 'T1':
            h = 1.349 / 2 * b
            out[p] = (lvl, float(np.exp(a - h)), float(np.exp(a + h)))
        elif lvl == 'T2':
            out[p] = (lvl, a * b[0], a * b[1])
        else:
            out[p] = (lvl, getattr(base, p) * 0.5, getattr(base, p) * 2.0)
    return out

def угол(T, box, pr, сторона):
    return capex(T, replace(Prices(), **{p: box[p][1 if сторона == 'lo' else 2] for p in pr}))['capex'].to_numpy()

def классы(npv_min, npv_max):
    return np.where(npv_min > 0, '+', np.where(npv_max < 0, '-', '?'))

def main() -> int:
    pmc.init()
    T = pmc._T
    prm = e19.параметры(e19.CAP_ПП)
    with contextlib.redirect_stdout(io.StringIO()):
        r = e19.прогон_exante('ident', prm)
    g = r['линии'][e19.LINE]
    print(f"РЕГРЕССИЯ GC {g['GC']:.4f}, n {r['n']}, лет {r['лет']}")
    if abs(g['GC'] - GC_HEAD) > TOL or r['n'] != N_HEAD or r['лет'] != ЛЕТ_HEAD:
        print('ОСТАНОВ: регрессия на головное число не сошлась')
        return 1
    with contextlib.redirect_stdout(io.StringIO()):
        d, _ = stats.build(prm)
    d = d.select('op_id', 'тип', 'npv', 'capex', 'доход_диск')
    d = d.sort('op_id')
    T = T.filter(pl.col('op_id').is_in(d['op_id'].to_list())).sort('op_id')
    assert d.height == N_HEAD and T.height == N_HEAD and (d['op_id'] == T['op_id']).all()
    дох = d['доход_диск'].to_numpy()
    cap_ппп = d['capex'].to_numpy()
    npv_мод = d['npv'].to_numpy()
    потеря = дох - cap_ппп - npv_мод
    cap_мод = capex(T, Prices())['capex'].to_numpy()
    if np.max(np.abs(cap_мод - cap_ппп) / np.maximum(np.abs(cap_ппп), 1.0)) > 1e-09:
        print('ОСТАНОВ: CAPEX(Prices()) не равен CAP_ПП')
        return 1
    print('РЕГРЕССИЯ угловой NPV при модельных ценах = npv_table: сошлась')
    box = коробка()
    все = list(box)
    уз = [p for p in все if box[p][0] == 'T1']
    рез = {}
    for имя, pr in (('полн', все), ('уз', уз)):
        c_lo, c_hi = (угол(T, box, pr, 'lo'), угол(T, box, pr, 'hi'))
        nmin, nmax = (дох - потеря - c_hi, дох - потеря - c_lo)
        if (nmin > nmax + 1e-06).any():
            print('ОСТАНОВ: NPV_min > NPV_max')
            return 1
        rng = np.random.default_rng(SEED_ВНУТР)
        for _ in range(N_ВНУТР):
            pt = {p: float(rng.uniform(box[p][1], box[p][2])) for p in pr}
            n_pt = дох - потеря - capex(T, replace(Prices(), **pt))['capex'].to_numpy()
            if (n_pt < nmin - 1e-06).any() or (n_pt > nmax + 1e-06).any():
                print(f'ОСТАНОВ: внутренняя точка вне [NPV_min, NPV_max] ({имя})')
                return 1
        рез[имя] = (nmin, nmax, классы(nmin, nmax))
    print(f'РЕГРЕССИЯ {N_ВНУТР} внутренних точек в обоих вариантах: сошлась')
    типы = d['тип'].to_numpy()
    абс = np.abs(npv_мод)
    вывод('# Б7.1. Частичная идентификация знака NPV (ep30_ident)')
    вывод('')
    вывод(f'Цепь ППП, n {N_HEAD}, лет {ЛЕТ_HEAD}; регрессия GC 0.2631, CAPEX(Prices()) = CAP_ПП, угловой NPV = npv_table, {N_ВНУТР} внутренних точек (зерно {SEED_ВНУТР}) сошлась. Цен в P_полн {len(все)}, из них T1 {len(уз)}: ' + ', '.join(уз) + '.')
    вывод('')
    вывод('| P | класс | n | доля | Уилсон 95 % | доля в Σ|NPV| |')
    вывод('|---|---|---|---|---|---|')
    rows, дол = ([], {})
    for имя in ('полн', 'уз'):
        cl = рез[имя][2]
        for k, нм in (('+', 'устойчиво +'), ('-', 'устойчиво -'), ('?', 'зависит от модели затрат')):
            m = cl == k
            n = int(m.sum())
            lo, hi = уилсон(n, N_HEAD)
            sh = float(абс[m].sum() / абс.sum())
            дол[имя, k] = (n / N_HEAD, lo, hi, sh, n)
            вывод(f'| P_{имя} | {нм} | {n} | {n / N_HEAD:.4f} | [{lo:.4f}, {hi:.4f}] | {sh:.4f} |')
            rows.append([f'P_{имя}', k, n, f'{n / N_HEAD:.4f}', f'{lo:.4f}', f'{hi:.4f}', f'{sh:.4f}'])
    вывод('')
    вывод('По типам (P_полн; типы с числом операций < 10 -- без разбивки по классам):')
    вывод('| тип | n | «+» | «-» | «?» доля | Уилсон 95 % |')
    вывод('|---|---|---|---|---|---|')
    cl = рез['полн'][2]
    for t in sorted(set(типы)):
        m = типы == t
        n = int(m.sum())
        if n < МАЛО:
            вывод(f'| {t} | <{МАЛО} | | | | |')
            rows.append([f'тип {t}', '', f'<{МАЛО}', '', '', '', ''])
            continue
        kp, km, kq = (int((cl[m] == k).sum()) for k in ('+', '-', '?'))
        lo, hi = уилсон(kq, n)
        вывод(f'| {t} | {n} | {kp} | {km} | {kq / n:.3f} | [{lo:.3f}, {hi:.3f}] |')
        rows.append([f'тип {t}', '', n, kp, km, f'{kq / n:.3f}', f'[{lo:.3f}, {hi:.3f}]'])
    D, Dуз = (дол['полн', '?'][0], дол['уз', '?'][0])
    if D > ПОРОГ_ВЕРХ:
        исход = 'И-1: D > 50 %; отбор в основном зависит от цен, знак NPV инвариантен менее чем для половины операций'
    elif D > ПОРОГ_УЗ:
        исход = f'И-2: 25 % < D <= 50 %; знак NPV инвариантен ко всем допустимым ценам для {100 * (1 - D):.1f} % операций, для {100 * D:.1f} % зависит от модели затрат'
    else:
        исход = 'И-3: D <= 25 %; знак инвариантен не менее чем для 75 % операций (при P_полн, границы P названы)'
    вывод('')
    вывод(f'ИСХОД: {исход}. D(P_полн) = {D:.4f}, D(P_уз) = {Dуз:.4f}.')
    вывод('Классы описывают знак NPV операции при допустимых ценах, а не отбор: бюджет и ёмкость связывают операции. Границы T3 назначены правилом [0.5, 2], не взяты из данных. Класс «?» не складывается в «долю устойчивых решений».')
    вывод('')
    for имя in ('полн', 'уз'):
        for k, нм in (('+', 'плюс'), ('-', 'минус'), ('?', 'вопрос')):
            f, lo, hi, sh, n = дол[имя, k]
            якорь(f'ep30 ident P_{имя} доля {нм}', f)
            якорь(f'ep30 ident P_{имя} доля {нм} Уилсон lo', lo)
            якорь(f'ep30 ident P_{имя} доля {нм} Уилсон hi', hi)
            якорь(f'ep30 ident P_{имя} доля {нм} в Σ|NPV|', sh)
            якорь(f'ep30 ident P_{имя} число {нм}', n, 'd')
    якорь('ep30 ident число цен T1', len(уз), 'd')
    якорь('ep30 ident число цен P_полн', len(все), 'd')
    якорь('ep30 ident операций', N_HEAD, 'd')
    OUT_MD.write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
    with open(OUT_TSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['строка', 'класс', 'n', 'доля_или_плюс', 'lo_или_минус', 'hi_или_вопрос', 'доля_в_сумме_модулей'])
        w.writerows(rows)
    return 0
if __name__ == '__main__':
    sys.exit(main())
