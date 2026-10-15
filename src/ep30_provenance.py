"""Shares of provenance marks by type; selection on the subset without filled values."""
import config as cfg
import csv
import pathlib
import sys
import time
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import ep19_crossruler as e19
import price_montecarlo as pmc
from provenance import КОЛИЧЕСТВА, МЕТКИ, СТОЛБЕЦ, метки, проверить_полноту
from sdt_analysis import цепь
TOL, HALF, B = (0.0001, 0.084, 1000000000.0)
J1, J2 = (0.75, 0.5)
OUT_MD = ROOT / cfg.results('ep30_provenance.md')
OUT_TSV = ROOT / cfg.out('ep30_provenance.tsv')
OUT_SUB = ROOT / cfg.out('ep30_provenance_subset.tsv')
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

def множество(рез, линия=e19.LINE):
    return {i for г in рез['отборы'] for i in г['op_id'][г['отбор'][линия]].tolist()}

def исход(лет, lo, d):
    if лет < 9:
        return 'М-4'
    if lo <= 0:
        return 'М-3'
    return 'М-1' if abs(d) <= HALF else 'М-2'

def исход_ж(j):
    return 'Ж-1' if j >= J1 else 'Ж-2' if j >= J2 else 'Ж-3'

def таблица_долей(m, подпись):
    строки = []
    for q in КОЛИЧЕСТВА:
        g = m.group_by('тип', СТОЛБЕЦ[q]).len().rename({СТОЛБЕЦ[q]: 'метка'}).pivot(on='метка', index='тип', values='len').fill_null(0).sort('тип')
        for м in МЕТКИ:
            if м not in g.columns:
                g = g.with_columns(pl.lit(0).alias(м))
        for r in g.iter_rows(named=True):
            n = sum((r[м] for м in МЕТКИ))
            строки.append([подпись, q, r['тип'], n] + [r[м] for м in МЕТКИ] + [f"{r['восполнено'] / n:.4f}"])
    return строки

def главная() -> int:
    t0 = time.time()
    cap = pl.read_parquet(e19.CAP_ПП)
    m = метки()
    проверить_полноту(m, 6800)
    полн, _ = цепь('prov_full', cap)
    ids = полн['d']['op_id'].to_list()
    m5 = m.filter(pl.col('op_id').is_in(ids))
    assert m5.height == len(ids) == 5806, f'цепь: {len(ids)}, меток {m5.height}'
    g0 = полн['линии'][e19.LINE]
    print(f"РЕГРЕССИЯ полная: GC {g0['GC']:.4f} [{g0['lo']:.4f}, {g0['hi']:.4f}], n {полн['n']}", flush=True)
    if not (abs(g0['GC'] - 0.2631) <= TOL and abs(g0['lo'] - 0.1772) <= TOL and (abs(g0['hi'] - 0.3453) <= TOL) and (полн['n'] == 5806) and (полн['лет'] == 9)):
        print('ОСТАНОВ: полная цепь не воспроизводит 0.2631')
        return 1
    S0 = множество(полн)
    F0 = множество(полн, 'факт инженеров')
    cpx = dict(zip(cap['op_id'].to_list(), cap['capex'].to_list()))
    вывод('# Б1.5. Метки происхождения количеств и отбор без восполненных (ep30_provenance)')
    вывод('')
    вывод('Метки: `src/provenance.py`, правила `docs/epic/b1_reading_rules.md` п. 7.1-7.2. Метка описывает источник количества в факте, не его правильность; привязка к операции не проверяется; проходка ЗБС определяется ПРОКСИ (значение равно подставленной медиане).')
    вывод('')
    вывод('## Доля «восполнено» по типам и количествам (пул 6 800; полная таблица меток в TSV)')
    вывод('')
    типы = sorted(m['тип'].unique().to_list())
    вывод('| количество | ' + ' | '.join(типы) + ' | всего |')
    вывод('|---|' + '---|' * (len(типы) + 1))
    for q in КОЛИЧЕСТВА:
        c = СТОЛБЕЦ[q]
        ячейки = []
        for t in типы:
            x = m.filter(pl.col('тип') == t)
            ячейки.append(f"{(x[c] == 'восполнено').mean():.3f}")
        вывод(f'| {q} | ' + ' | '.join(ячейки) + f" | {(m[c] == 'восполнено').mean():.3f} |")
    вывод('')
    вывод('## Распределение меток по количествам (число операций, пул 6 800 / цепь 5 806)')
    вывод('')
    вывод('| количество | ' + ' | '.join(МЕТКИ) + ' |')
    вывод('|---|' + '---|' * len(МЕТКИ))
    for q in КОЛИЧЕСТВА:
        c = СТОЛБЕЦ[q]
        a = {k: int((m[c] == k).sum()) for k in МЕТКИ}
        b = {k: int((m5[c] == k).sum()) for k in МЕТКИ}
        вывод(f'| {q} | ' + ' | '.join((f'{a[k]} / {b[k]}' for k in МЕТКИ)) + ' |')
        for k in МЕТКИ:
            якорь(f'ep30 provenance {q} {k} пул', a[k], 'd')
            якорь(f'ep30 provenance {q} {k} цепь', b[k], 'd')
    вывод('')
    tsv = таблица_долей(m, 'пул 6800') + таблица_долей(m5, 'цепь 5806')
    восп = lambda c: pl.col(СТОЛБЕЦ[c]) == 'восполнено'
    подмножества = [('П-А', m5.filter(восп('проходка_м'))['op_id'].to_list()), ('П-Б', m5.filter(восп('проходка_м') | восп('станок_фикс') | восп('материальная_часть'))['op_id'].to_list())]
    вывод('## Отбор на подмножестве без операций с восполненным ключевым количеством')
    вывод('')
    вывод('Базовая цепь: GC 0.2631 [0.1772, 0.3453], n 5 806. Жаккар считается между отбором усечённой цепи и базовым отбором, ограниченным подмножеством.')
    вывод('')
    вывод('| подмножество | исключено | доля | n | лет | GC | lo | hi | dGC | исход | в отборе полной | в факте | Жаккар | исход Ж | расхождений | dK, млрд |')
    вывод('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    итоги, подтсв = ({}, [])
    for имя, excl in подмножества:
        прогресс(f'provenance {имя}, {time.time() - t0:.0f} с')
        Е = set(excl)
        наш, _ = цепь(f'prov_{имя}', cap.filter(~pl.col('op_id').is_in(excl)))
        g = наш['линии'][e19.LINE]
        S = множество(наш)
        Sб = {i for i in S0 if i not in Е}
        d = g['GC'] - g0['GC']
        j = len(S & Sб) / max(len(S | Sб), 1)
        расх = len(S ^ Sб)
        dK = (sum((cpx[i] for i in S)) - sum((cpx[i] for i in Sб))) / B
        ис = исход(наш['лет'], g['lo'], d)
        итоги[имя] = ис
        sel_e, fact_e = (len(S0 & Е), len(F0 & Е))
        вывод(f"| {имя} | {len(Е)} | {len(Е) / len(ids):.4f} | {наш['n']} | {наш['лет']} | {g['GC']:.4f} | {g['lo']:.4f} | {g['hi']:.4f} | {d:+.4f} | {ис} | {sel_e} | {fact_e} | {j:.4f} | {исход_ж(j)} | {расх} | {dK:+.3f} |")
        подтсв.append([имя, len(Е), наш['n'], наш['лет'], f"{g['GC']:.4f}", f"{g['lo']:.4f}", f"{g['hi']:.4f}", f'{d:+.4f}', ис, sel_e, fact_e, f'{j:.4f}', исход_ж(j), расх, f'{dK:+.4f}'])
        якорь(f'ep30 provenance {имя} исключено', len(Е), 'd')
        якорь(f'ep30 provenance {имя} n', наш['n'], 'd')
        якорь(f'ep30 provenance {имя} GC', g['GC'])
        якорь(f'ep30 provenance {имя} GC_lo', g['lo'])
        якорь(f'ep30 provenance {имя} GC_hi', g['hi'])
        якорь(f'ep30 provenance {имя} dGC', d, '+.4f')
        якорь(f'ep30 provenance {имя} Жаккар', j)
        якорь(f'ep30 provenance {имя} расхождений', расх, 'd')
        якорь(f'ep30 provenance {имя} dK млрд', dK, '+.4f')
        якорь(f'ep30 provenance {имя} исключённых в отборе полной цепи', sel_e, 'd')
        якорь(f'ep30 provenance {имя} исключённых в факте', fact_e, 'd')
    вывод('')
    вывод(f"**ИСХОД (вердикт по П-Б, п. 7.5)**: {итоги['П-Б']}; П-А: {итоги['П-А']}." + ('' if итоги['П-А'] == итоги['П-Б'] else ' Исходы П-А и П-Б РАЗОШЛИСЬ.'))
    якорь('ep30 provenance номер исхода П-Б (1 М-1, 2 М-2, 3 М-3, 4 М-4)', int(итоги['П-Б'][-1]), 'd')
    якорь('ep30 provenance время счёта с', time.time() - t0, '.0f')
    OUT_MD.write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
    with open(OUT_TSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['совокупность', 'количество', 'тип', 'n'] + list(МЕТКИ) + ['доля_восполнено'])
        w.writerows(tsv)
    with open(OUT_SUB, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['подмножество', 'исключено', 'n', 'лет', 'GC', 'GC_lo', 'GC_hi', 'dGC', 'исход', 'в_отборе_полной', 'в_факте', 'Жаккар', 'исход_Ж', 'расхождений', 'dK_млрд'])
        w.writerows(подтсв)
    return 0
if __name__ == '__main__':
    sys.exit(главная())
