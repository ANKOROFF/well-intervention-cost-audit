"""Exclusion of sidetracks, empirical drilling rate and reconciliation with the plan."""
import config as cfg
import contextlib
import csv
import io
import pathlib
import sys
import tempfile
import numpy as np
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import ep19_crossruler as e19
import price_registry as pr
from ep06_ablation import калибровать, написать_вариант
GC_HEAD = 0.2631
TOL = 0.0001
HALF = 0.084
МИН_N = 10
OUT = ROOT / cfg.out('sdt_analysis.tsv')
ДРМ = cfg.data('parquet/final/sgs_drm_plan_d0c9.parquet')
ОЦЕНКИ_СТАТЬИ = (2.36, 401.0)
ЯДРО = []

def строка(блок, метрика, сов, год, знач, прим=''):
    ЯДРО.append([блок, метрика, сов, год, знач, прим])

def тихо(f, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a)

def отборы_линии(рез):
    return {г['год']: set(г['op_id'][г['отбор'][e19.LINE]].tolist()) for г in рез['отборы']}

def расхождение(a, b, збс):
    рез = {}
    for y in sorted(a):
        x = a[y] ^ b[y]
        рез[y] = (len(x), len(x & збс), len(a[y] & b[y]), len(a[y] | b[y]))
    return рез

def цепь(метка, cap):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='sdt_'))
    п = tmp / f'{метка}.parquet'
    cap.write_parquet(п)
    наш = тихо(e19.прогон_exante, метка, e19.параметры(str(п)))
    ids = set(наш['d']['op_id'].to_list())
    ставка, _ = калибровать(cap, ids, 'dur')
    пл1 = tmp / f'{метка}_l1.parquet'
    написать_вариант(cap, 'dur', ставка, пл1)
    л1 = тихо(e19.прогон_exante, метка + ' Л1', e19.параметры(str(пл1)))
    return (наш, л1)

def b1() -> int:
    cap = pl.read_parquet(e19.CAP_ПП)
    збс = set(cap.filter(pl.col('тип') == 'ЗБС')['op_id'].to_list())
    полн_наш, полн_л1 = цепь('full', cap)
    gf = полн_наш['линии'][e19.LINE]
    print(f"РЕГРЕССИЯ: GC(полная) = {gf['GC']:.4f} против {GC_HEAD}; n = {полн_наш['n']}")
    if abs(gf['GC'] - GC_HEAD) > TOL or полн_наш['n'] != e19.N_ОБЪЯВЛЕНО:
        print('ОСТАНОВ: регрессия не сошлась')
        return 1
    ids_полн = set(полн_наш['d']['op_id'].to_list())
    n_збс = len(ids_полн & збс)
    б_наш, б_л1 = цепь('no_zbs', cap.filter(pl.col('тип') != 'ЗБС'))
    gb = б_наш['линии'][e19.LINE]
    d = gb['GC'] - gf['GC']
    for сов, r, g in (('полная', полн_наш, gf), ('без ЗБС', б_наш, gb)):
        строка('B1', 'n операций', сов, '', r['n'])
        строка('B1', 'GC', сов, '', f"{g['GC']:.4f}")
        строка('B1', 'GC ДИ низ', сов, '', f"{g['lo']:.4f}")
        строка('B1', 'GC ДИ верх', сов, '', f"{g['hi']:.4f}")
    строка('B1', 'операций ЗБС в полной совокупности', 'полная', '', n_збс)
    строка('B1', 'dGC (без ЗБС минус полная)', 'без ЗБС', '', f'{d:+.4f}')
    итог = {}
    for сов, н, л in (('полная', полн_наш, полн_л1), ('без ЗБС', б_наш, б_л1)):
        р = расхождение(отборы_линии(н), отборы_линии(л), збс)
        for y, (x, xз, i, u) in р.items():
            строка('B1', 'расхождение отбора наша-Л1, операций', сов, y, x)
            if сов == 'полная':
                строка('B1', 'из них ЗБС', сов, y, xз)
        X, Xз, I, U = (sum((v[k] for v in р.values())) for k in range(4))
        строка('B1', 'расхождение отбора наша-Л1, операций', сов, 'сумма', X)
        строка('B1', 'Жаккар отборов наша-Л1', сов, 'сумма', f'{I / U:.4f}')
        if сов == 'полная':
            строка('B1', 'из них ЗБС', сов, 'сумма', Xз)
            строка('B1', 'доля расхождения на ЗБС', сов, 'сумма', f'{Xз / X:.4f}')
        итог[сов] = (X, Xз)
    доля = итог['полная'][1] / итог['полная'][0]
    if gb['lo'] <= 0:
        sw = 'W'
    elif abs(d) <= HALF:
        sw = 'S'
    else:
        sw = 'вне S/W: нижняя граница > 0, но |dGC| > 0.084'
    c = 'C' if доля >= 0.5 else 'не C'
    строка('B1', 'исход S/W', 'без ЗБС', '', sw)
    строка('B1', 'исход C', 'полная', '', f"{c} (доля на ЗБС {доля:.4f}; ЗБС {n_збс / полн_наш['n']:.4f} совокупности)")
    print(f"B1: GC без ЗБС {gb['GC']:.4f} [{gb['lo']:.4f}, {gb['hi']:.4f}], n {б_наш['n']}, dGC {d:+.4f}; расхождение {итог['полная'][0]} / без ЗБС {итог['без ЗБС'][0]}, доля ЗБС {доля:.4f}; исход {sw}, {c}")
    return 0

def b2():
    pr.ШАПКИ.update({pr.норм(k): v for k, v in pr.ШАПКИ_ТЕКСТ.items()})
    А = тихо(pr.собрать_А)
    s = А.filter((pl.col('dn') > 0) & (pl.col('prokh') > 0)).with_columns(v=pl.col('prokh') / pl.col('dn'))
    for имя, t in (('все', s), ('проходка < 500 м', s.filter(pl.col('prokh') < 500))):
        v = t['v'].to_numpy()
        строка('B2', 'n скважин (дни>0, проходка>0, после дедупликации)', имя, '', len(v))
        строка('B2', 'обществ', имя, '', t['soc'].n_unique())
        if len(v) < МИН_N:
            continue
        q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
        for k, x in zip(('q05', 'q25', 'q50', 'q75', 'q95'), q):
            строка('B2', f'проходка/сутки {k}, м/сут', имя, '', f'{x:.1f}')
        внутри = ОЦЕНКИ_СТАТЬИ[0] <= q[1] and q[3] <= ОЦЕНКИ_СТАТЬИ[1]
        строка('B2', 'исход (q25-q75 между 2.36 и 401)', имя, '', 'внутри коридора' if внутри else 'вне коридора')
        print(f"B2 [{имя}]: n {len(v)}, q25 {q[1]:.1f}, q50 {q[2]:.1f}, q75 {q[3]:.1f} -> {('внутри' if внутри else 'вне')}")
    строка('B2', 'оговорка', '', '', '', "опросные листы: бурение при строительстве, новые скважины, не боковые стволы; сутки -- 'Количество дней бурения'")

def b3():
    cap = pl.read_parquet(e19.CAP_ПП).filter(pl.col('тип') == 'ЗБС')
    м = cap.group_by('год').agg(n=pl.len(), m=pl.col('capex').mean()).to_dicts()
    м = {r['год']: r for r in м}
    план = pl.read_parquet(ДРМ).filter(pl.col('pokazatel').str.starts_with('D0464_') & (pl.col('og') == 'SUBSIDIARY')).row(0, named=True)
    сум_м = сум_п = 0.0
    for y in range(2020, 2026):
        p = float(план[f'c_{y}']) * 1000.0
        r = м[y]
        if r['n'] < МИН_N:
            строка('B3', 'n ЗБС модели', 'SUBSIDIARY', y, r['n'], 'n < 10, отношение не публикуется')
            continue
        k = r['m'] / p
        o = 'уровень сопоставим' if k >= 0.8 else 'ниже плана' if k >= 0.5 else 'существенно ниже плана'
        строка('B3', 'n ЗБС модели', 'SUBSIDIARY', y, r['n'])
        строка('B3', 'отношение модель/план (средний CAPEX ЗБС / D0464)', 'SUBSIDIARY', y, f'{k:.3f}', o)
        сум_м += r['m'] * r['n']
        сум_п += p * r['n']
        print(f"B3 {y}: n {r['n']}, модель/план {k:.3f} -> {o}")
    k = сум_м / сум_п
    o = 'уровень сопоставим' if k >= 0.8 else 'ниже плана' if k >= 0.5 else 'существенно ниже плана; модельная стоимость ЗБС занижена или определения не сопоставимы'
    строка('B3', 'отношение модель/план, взвешено числом операций', 'SUBSIDIARY', '2020-2025', f'{k:.3f}', o)
    строка('B3', 'оговорка', '', '', '', 'определения различны: план -- вся скважина БС, модель -- ресурсная часть ЗБС; проверка только уровня')
    print(f'B3 пул: {k:.3f} -> {o}')

def main() -> int:
    if b1():
        return 1
    b2()
    b3()
    with open(OUT, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['block', 'metric', 'population', 'year', 'value', 'note'])
        w.writerows(ЯДРО)
    return 0
if __name__ == '__main__':
    sys.exit(main())
