"""Four cost representations and the 4x4 matrix."""
import config as cfg
import csv
import itertools
import pathlib
import sys
import tempfile
import time
import numpy as np
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import ep19_crossruler as e19
import price_montecarlo as pmc
from ep06_ablation import _ди_суммы, _ряд_линейкой, _цены, калибровать, написать_вариант
from sdt_analysis import тихо, цепь
TOL, B, LAMBDA_D = (0.0001, 1000000000.0, 1.0)
LINE = e19.LINE
REP = ['A', 'B', 'C', 'D']
OUT_MD = ROOT / cfg.results('ep30_repr.md')
OUT_TSV = ROOT / cfg.out('ep30_repr.tsv')
ПРОЧЬ = []

def вывод(s=''):
    ПРОЧЬ.append(s)
    print(s, flush=True)

def якорь(имя, знач, ф='.4f'):
    вывод(f'ЯКОРЬ {имя} = {format(знач, ф)}')

def ставки_по_типу(cap, ids, числитель, d='dur'):
    t = cap.filter(pl.col('op_id').is_in(list(ids))).group_by('тип').agg(num=pl.col(числитель).sum(), den=pl.col(d).fill_null(0.0).sum())
    return dict(zip(t['тип'].to_list(), zip(t['num'].to_list(), t['den'].to_list())))

def варианты(cap, ids):
    r_A, _ = калибровать(cap, ids, 'dur')
    d = pl.col('dur').fill_null(0.0)
    rB = {t: n / dd if dd > 0 else r_A for t, (n, dd) in ставки_по_типу(cap, ids, 'capex').items()}
    rC = {t: n / dd if dd > 0 else 0.0 for t, (n, dd) in ставки_по_типу(cap, ids, 'c_время').items()}
    rb = pl.col('тип').replace_strict(rB, default=r_A, return_dtype=pl.Float64)
    rc = pl.col('тип').replace_strict(rC, default=0.0, return_dtype=pl.Float64)
    return {'B': cap.select('op_id', capex=d * rb), 'C': cap.select('op_id', capex=d * rc + pl.col('c_материалы_исп')), 'D': cap.select('op_id', capex=d * rc + pl.col('c_материалы_исп') + LAMBDA_D * pl.col('c_насос'))}

def главная() -> int:
    t0 = time.time()
    cap = pl.read_parquet(e19.CAP_ПП)
    j = cap.join(pl.read_parquet(e19.DUR_SRC).select('op_id', 'dur_пред'), on='op_id')
    assert float((j['dur'] - j['dur_пред']).abs().max()) == 0.0, 'd != dur_пред'
    рез = {}
    рез['Н'], рез['A'] = цепь('repr', cap)
    ids = set(рез['Н']['d']['op_id'].to_list())
    g = рез['Н']['линии'][LINE]
    print(f"РЕГРЕССИЯ GC(Н) = {g['GC']:.4f}, n {рез['Н']['n']}", flush=True)
    if abs(g['GC'] - 0.2631) > TOL or рез['Н']['n'] != e19.N_ОБЪЯВЛЕНО:
        print('ОСТАНОВ: GC(Н) не 0.2631')
        return 1
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='repr_'))
    уровень = {'Н': float(cap.filter(pl.col('op_id').is_in(list(ids)))['capex'].sum()) / B}
    уровень['A'] = float(калибровать(cap, ids, 'dur')[0] * калибровать(cap, ids, 'dur')[1]) / B
    for k, df in варианты(cap, ids).items():
        п = tmp / f'{k}.parquet'
        df.with_columns(мат_измерены=pl.lit(False)).write_parquet(п)
        рез[k] = тихо(e19.прогон_exante, f'repr {k}', e19.параметры(str(п)))
        уровень[k] = float(df.filter(pl.col('op_id').is_in(list(ids)))['capex'].sum()) / B
        print(f"цепь {k}: n {рез[k]['n']}, GC собственный {рез[k]['линии'][LINE]['GC']:.4f}, {time.time() - t0:.0f} с", flush=True)
    for k in рез:
        if рез[k]['n'] != e19.N_ОБЪЯВЛЕНО:
            print(f"ОСТАНОВ: n({k}) = {рез[k]['n']}")
            return 1
    цены = {k: _цены(рез[k]) for k in рез}
    f = {k: рез[k]['df']['факт инженеров|NPV'].to_numpy() for k in рез}
    o = {k: рез[k]['df']['оракул по факту|NPV'].to_numpy() for k in рез}
    ряд = {(x, y): _ряд_линейкой(рез[x]['отборы'], цены[y], LINE) for x in REP for y in ['Н'] + REP}
    яч = {(x, y): e19.доля(r, f[y], o[y]) for (x, y), r in ряд.items()}
    ok = abs(яч['A', 'A']['доля'] - 0.3479) <= TOL and abs(яч['A', 'Н']['доля'] - 0.2287) <= TOL
    print(f"РЕГРЕССИЯ V_AA {яч['A', 'A']['доля']:.4f} против 0.3479; V_AН {яч['A', 'Н']['доля']:.4f} против 0.2287: {('сошлась' if ok else 'НЕ СОШЛАСЬ')}", flush=True)
    if not ok:
        print('ОСТАНОВ: регрессия матрицы не сошлась')
        return 1
    вывод('# Б7.4. Четыре представления затрат A-D, матрица 4x4 (ep30_repr)')
    вывод('')
    вывод('Регрессия: GC(Н) 0.2631, V_AA 0.3479, V_AН 0.2287. Это чувствительность к представлению затрат, не выбор лучшего: внешнего арбитра в данных нет. Знаменатель столбца зависит только от оценщика Y; столбцы между собой не складываются.')
    вывод(f'Уровень Sum capex на 5 806 операциях, млрд руб: ' + ', '.join((f'{k} {v:.3f}' for k, v in уровень.items())) + '. Уровень у A и B выровнен калибровкой, у C близок к Н по построению, у D выше на Sum c_насос (свойство стресса).')
    вывод('')
    вывод('| ячейка V_XY (отбор X, оценка Y) | NPV, млрд | GC | lo | hi | ДМ | p | знаменатель, млрд |')
    вывод('|---|---|---|---|---|---|---|---|')
    tsv = []
    for (x, y), д in яч.items():
        знам = рез[y]['разрыв']
        вывод(f"| V_{x}{y} | {д['NPV']:.4f} | {д['доля']:.4f} | {д['lo']:.4f} | {д['hi']:.4f} | {д['ДМ']:.2f} | {д['p']:.4f} | {знам:.4f} |")
        tsv.append([f'V_{x}{y}', x, y, f"{д['NPV']:.4f}", f"{д['доля']:.4f}", f"{д['lo']:.4f}", f"{д['hi']:.4f}", f"{д['ДМ']:.2f}", f"{д['p']:.4f}", f'{знам:.4f}'])
        якорь(f'ep30 repr V_{x}{y} NPV млрд', д['NPV'])
        якорь(f'ep30 repr V_{x}{y} GC', д['доля'])
        якорь(f'ep30 repr V_{x}{y} GC ДИ низ', д['lo'])
        якорь(f'ep30 repr V_{x}{y} GC ДИ верх', д['hi'])
    вывод('')
    порядки, лидеры = ({}, {})
    вывод('Порядок отборов по сумме NPV в столбце Y (лидер; разность лидер - второй, ДИ бутстрэпа по годам):')
    for y in REP + ['Н']:
        упор = sorted(REP, key=lambda x: -ряд[x, y].sum())
        порядки[y] = упор
        a, b = (упор[0], упор[1])
        d = ряд[a, y] - ряд[b, y]
        lo, hi, reps = _ди_суммы(d, np.random.default_rng(e19.SEED))
        нр = lo <= 0 <= hi
        вывод(f"- оценщик {y}: {' > '.join(упор)}; лидер {a}, {a} - {b} = {d.sum() / B:+.3f} [{lo / B:+.3f}, {hi / B:+.3f}] млрд" + ('; ДИ накрывает ноль, лидерство не различимо на девяти годах' if нр else '') + ('' if y != 'Н' else ' (столбец Н справочный, в исход не входит)'))
        лидеры[y] = a
        якорь(f'ep30 repr лидер-второй {y} млрд', d.sum() / B)
        якорь(f'ep30 repr лидер-второй {y} ДИ низ', lo / B)
        якорь(f'ep30 repr лидер-второй {y} ДИ верх', hi / B)
        якорь(f'ep30 repr лидер-второй {y} различим (1 да, 0 нет)', 0 if нр else 1, 'd')
    вывод('')
    if all((порядки[y] == порядки['A'] for y in REP)):
        код, т = ('Р-1', 'ранжирование представлений не зависит от оценщика')
    elif len({лидеры[y] for y in REP}) == 1:
        код, т = ('Р-2', f"лидер ({лидеры['A']}) один и тот же во всех четырёх столбцах, порядок остальных различается")
    else:
        код, т = ('Р-3', 'лидер зависит от оценщика: решение не инвариантно к представлению затрат')
    вывод(f'**ИСХОД {код}**: {т}. Это чувствительность, а не победа представления.')
    якорь('ep30 repr номер исхода (1 Р-1, 2 Р-2, 3 Р-3)', {'Р-1': 1, 'Р-2': 2, 'Р-3': 3}[код], 'd')
    вывод('')

    def мн(r):
        return {(г['год'], i) for г in r['отборы'] for i in г['op_id'][г['отбор'][LINE]].tolist()}
    S = {k: мн(рез[k]) for k in ['Н'] + REP}
    вывод('Жаккар отборов (пары, по (год, операция)) и размер отбора:')
    вывод('| пара | Жаккар | симм. разность |')
    вывод('|---|---|---|')
    for a, b in itertools.combinations(['Н'] + REP, 2):
        jj = len(S[a] & S[b]) / len(S[a] | S[b])
        вывод(f'| {a}-{b} | {jj:.4f} | {len(S[a] ^ S[b])} |')
        якорь(f'ep30 repr Жаккар {a}-{b}', jj)
    for k in S:
        якорь(f'ep30 repr размер отбора {k}', len(S[k]), 'd')
    якорь('ep30 repr время счёта с', time.time() - t0, '.0f')
    OUT_MD.write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
    with open(OUT_TSV, 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh, delimiter='\t', lineterminator='\n')
        w.writerow(['клетка', 'отбор', 'оценка', 'NPV_млрд', 'GC', 'ДИ_низ', 'ДИ_верх', 'ДМ', 'p', 'знаменатель_млрд'])
        w.writerows(tsv)
    return 0
if __name__ == '__main__':
    sys.exit(главная())
