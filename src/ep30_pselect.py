"""Selection frequency under price draws."""
import config as cfg
import argparse
import contextlib
import csv
import io
import math
import pathlib
import sys
import time
from collections import Counter
from dataclasses import replace
import numpy as np
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import ep19_crossruler as e19
import price_montecarlo as pmc
from cost_physical import Prices, capex
from ep06_ablation import калибровать, написать_вариант
SEED_M = 20261013
ПОРОГ_ВЫС, ПОРОГ_НИЗ, S1, S2 = (0.95, 0.05, 0.25, 0.5)
OUT_MD = ROOT / cfg.results('ep30_pselect.md')
OUT_TSV = ROOT / cfg.out('ep30_pselect.tsv')
OUT_OPS = ROOT / cfg.data('interim/ep30_pselect_ops.parquet')
PROG = ROOT / cfg.results('ep30_progress.txt')
ПРОЧЬ = []

def вывод(s=''):
    ПРОЧЬ.append(s)
    print(s, flush=True)

def якорь(имя, знач, ф='.4f'):
    вывод(f'ЯКОРЬ {имя} = {format(знач, ф)}')

def прогресс(s):
    with open(PROG, 'a', encoding='utf-8') as f:
        f.write(time.strftime('%H:%M:%S ') + s + '\n')

def уилсон(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def розыгрыш(i, pr, с_л1):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c = capex(pmc._T, replace(Prices(), **pr))
        pa = pmc._TMP / f'd{i}.parquet'
        c.write_parquet(pa)
        наш = e19.прогон_exante(f'd{i}', e19.параметры(str(pa)))
        out = {'i': i, 'n': наш['n'], 'GC': наш['линии'][e19.LINE]['GC'], 'a': pmc.sel_sets(наш)}
        if с_л1:
            ставка, _ = калибровать(c, set(наш['d']['op_id'].to_list()), 'dur')
            pl1 = pmc._TMP / f'l{i}.parquet'
            написать_вариант(c, 'dur', ставка, pl1)
            л1 = e19.прогон_exante(f'l{i}', e19.параметры(str(pl1)))
            b = pmc.sel_sets(л1)
            out['div'] = len(out['a'] ^ b)
            out['jac'] = len(out['a'] & b) / max(len(out['a'] | b), 1)
            pl1.unlink()
        if i == 0:
            out['fact'] = set()
            for g in наш['отборы']:
                out['fact'] |= set(g['op_id'][g['отбор']['факт инженеров']].tolist())
    pa.unlink()
    return out

def ряд_tsv():
    with open(pmc.OUT_DRAWS, encoding='utf-8') as f:
        return {int(r['draw']): r for r in csv.DictReader(f, delimiter='\t')}

def сверка(r, t):
    return r['n'] == int(t['n']) and abs(r['GC'] - float(t['GC'])) <= 0.0001 and (r['div'] == int(t['sel_diff'])) and (abs(r['jac'] - float(t['jaccard'])) <= 0.0001)

def отчёт(метка, cnt, n, fact, типы, всего, ключ):
    U = set(cnt) | fact
    P = {i: cnt.get(i, 0) / n for i in U}
    кл = {i: 'выс' if p >= ПОРОГ_ВЫС else 'низ' if p <= ПОРОГ_НИЗ else 'чувств' for i, p in P.items()}
    k = Counter(кл.values())
    S = k['чувств'] / len(U)
    lo, hi = уилсон(k['чувств'], len(U))
    исх = 'В-1' if S > S2 else 'В-2' if S > S1 else 'В-3'
    вывод(f'### {метка}')
    вывод(f"Операций хотя бы раз в отборе или в факте: {len(U)} из {всего}; в отборе хотя бы раз {len(cnt)}; в факте {len(fact)}. Классы: robustly selected {k['выс']}, robustly rejected {k['низ']}, cost-sensitive {k['чувств']}.")
    вывод(f"S = {S:.4f} (95 % Уилсон [{lo:.4f}, {hi:.4f}], знаменатель {len(U)}; операции не независимы, интервал описательный); доля cost-sensitive в {всего} = {k['чувств'] / всего:.4f}. **ИСХОД {исх}.**")
    h = np.histogram([P[i] for i in U], bins=[0, 0.05 + 1e-09, 0.25, 0.5, 0.75, 0.95 - 1e-09, 1.0 + 1e-09])[0]
    вывод(f"Распределение P_i (0-0.05 / .05-.25 / .25-.5 / .5-.75 / .75-.95 / 0.95-1): {' / '.join(map(str, h))}")
    вывод('')
    вывод('| тип | операций в U | выс | низ | чувств | S типа | Уилсон |')
    вывод('|---|---|---|---|---|---|---|')
    tsv = []
    for t in pmc.TYPES:
        ids = [i for i in U if типы[i] == t]
        a = Counter((кл[i] for i in ids))
        l, h2 = уилсон(a['чувств'], len(ids))
        s = a['чувств'] / len(ids) if ids else float('nan')
        вывод(f"| {t} | {len(ids)} | {a['выс']} | {a['низ']} | {a['чувств']} | {s:.3f} | [{l:.3f}, {h2:.3f}] |")
        tsv.append([метка, t, len(ids), a['выс'], a['низ'], a['чувств'], f'{s:.4f}', f'{l:.4f}', f'{h2:.4f}'])
    tsv.append([метка, 'ВСЕ', len(U), k['выс'], k['низ'], k['чувств'], f'{S:.4f}', f'{lo:.4f}', f'{hi:.4f}'])
    вывод('')
    якорь(f'ep30 pselect {ключ} S', S)
    якорь(f'ep30 pselect {ключ} S Уилсон низ', lo)
    якорь(f'ep30 pselect {ключ} S Уилсон верх', hi)
    якорь(f'ep30 pselect {ключ} знаменатель U', len(U), 'd')
    якорь(f'ep30 pselect {ключ} robustly selected', k['выс'], 'd')
    якорь(f'ep30 pselect {ключ} robustly rejected', k['низ'], 'd')
    якорь(f'ep30 pselect {ключ} cost-sensitive', k['чувств'], 'd')
    якорь(f'ep30 pselect {ключ} номер исхода (1 В-1, 2 В-2, 3 В-3)', {'В-1': 1, 'В-2': 2, 'В-3': 3}[исх], 'd')
    return tsv

def главная():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    N = ap.parse_args().n
    t0 = time.time()
    sp = pmc.spec()
    pmc.init()
    cap = pl.read_parquet(e19.CAP_ПП)
    типы = dict(zip(cap['op_id'].to_list(), cap['тип'].to_list()))
    ref = ряд_tsv()
    rng = np.random.default_rng(pmc.SEED)
    draws = [(i, pmc.draw(sp, rng)) for i in range(1, N + 1)]
    base_pr = {p: getattr(Prices(), p) for p in pmc.PRICES}
    r0 = розыгрыш(0, base_pr, True)
    print(f"РЕГРЕССИЯ: GC(Prices()) = {r0['GC']:.4f} против {pmc.GC_HEAD}; n {r0['n']}", flush=True)
    if abs(r0['GC'] - pmc.GC_HEAD) > pmc.TOL or not сверка(r0, ref[0]):
        print('ОСТАНОВ: розыгрыш 0 не совпал с price_montecarlo.tsv')
        return 1
    fact = r0['fact']
    cnt = Counter()
    for i, pr in draws:
        r = розыгрыш(i, pr, True)
        if not сверка(r, ref[i]):
            print(f"ОСТАНОВ: розыгрыш {i} не совпал с price_montecarlo.tsv: n {r['n']}/{ref[i]['n']}, GC {r['GC']:.4f}/{ref[i]['GC']}, div {r['div']}/{ref[i]['sel_diff']}, jac {r['jac']:.4f}/{ref[i]['jaccard']}")
            прогресс(f'ОСТАНОВ на розыгрыше {i}')
            return 1
        cnt.update(r['a'])
        if i % 10 == 0:
            прогресс(f'независимые: {i} из {N}, {time.time() - t0:.0f} с')
    вывод('# Б7.2. Доля розыгрышей цен, в которых операция входит в отбор (ep30_pselect)')
    вывод('')
    вывод(f'Регрессия: все {N + 1} розыгрышей (включая нулевой) совпали с price_montecarlo.tsv (GC 1e-4, n и расхождение точно, Жаккар 1e-4). P_i есть ДОЛЯ ИЗ 500 РОЗЫГРЫШЕЙ цен при заданных распределениях, а не вероятность верности решения.')
    вывод('')
    t1 = time.time() - t0
    tsv1 = отчёт('независимые розыгрыши', cnt, N, fact, типы, len(типы), 'независимые')
    OUT_OPS.parent.mkdir(exist_ok=True)
    pl.DataFrame({'op_id': list(cnt), 'k': list(cnt.values())}).write_parquet(OUT_OPS)
    rm = np.random.default_rng(SEED_M)
    m = np.exp(rm.uniform(np.log(0.5), np.log(2.0), size=N))
    cnt2 = Counter()
    for (i, pr), md in zip(draws, m):
        r = розыгрыш(i, {p: v * float(md) for p, v in pr.items()}, False)
        cnt2.update(r['a'])
        if i % 10 == 0:
            прогресс(f'общий множитель: {i} из {N}, {time.time() - t0:.0f} с')
    tsv2 = отчёт('общий множитель', cnt2, N, fact, типы, len(типы), 'множитель')
    вывод(f'Вердикт по независимым розыгрышам (п. 6.2.5); вариант с множителем печатается рядом. Время счёта: независимые {t1:.0f} с, всего {time.time() - t0:.0f} с.')
    якорь('ep30 pselect время счёта с', time.time() - t0, '.0f')
    OUT_MD.write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
    with open(OUT_TSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['вариант', 'тип', 'операций_U', 'robustly_selected', 'robustly_rejected', 'cost_sensitive', 'S', 'Уилсон_низ', 'Уилсон_верх'])
        w.writerows(tsv1 + tsv2)
    return 0
if __name__ == '__main__':
    sys.exit(главная())
