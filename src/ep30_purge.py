"""Purge test of target-horizon leakage."""
import config as cfg
import contextlib
import csv
import io
import pathlib
import sys
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import ep19_crossruler as e19
import stats
from ep06_ablation import _цены
B = 1000000000.0
TOL, ПОРОГ = (0.0001, 0.02)
ПРОЧЬ = []
ОРИГ = stats.year_run

def вывод(s=''):
    ПРОЧЬ.append(s)
    print(s, flush=True)

def якорь(имя, знач, ф='.4f'):
    вывод(f'ЯКОРЬ {имя} = {format(знач, ф)}')

def обёртка(purge):

    def f(d, руб_т, Y, pool_span, n_rand=200, отборы=None):
        mi0 = d['vnr'].dt.year() * 12 + d['vnr'].dt.month() - 1
        старый = d['год'] < Y
        годен = mi0 + 24 <= 12 * Y - 1 if purge else pl.Series([True] * d.height)
        keep = ~старый | годен
        return ОРИГ(d.filter(keep.fill_null(False)), руб_т, Y, pool_span, n_rand=n_rand, отборы=отборы)
    return f

def запуск(метка, purge):
    stats.year_run = обёртка(purge)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return e19.прогон_exante(метка, e19.параметры(e19.CAP_ПП))
    finally:
        stats.year_run = ОРИГ

def множество(r):
    return {i for g in r['отборы'] for i in g['op_id'][g['отбор'][e19.LINE]].tolist()}

def доля_опт(r, опт):
    npv = _цены(r)
    inc = {True: 0.0, False: 0.0}
    for g in r['отборы']:
        for линия, s in ((e19.LINE, 1.0), ('факт инженеров', -1.0)):
            for i in g['op_id'][g['отбор'][линия]].tolist():
                inc[i in опт] += s * npv[i]
    return inc[True] / (inc[True] + inc[False])

def main() -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        r0 = e19.прогон_exante('head', e19.параметры(e19.CAP_ПП))
    g0 = r0['линии'][e19.LINE]
    ok = abs(g0['GC'] - 0.2631) <= TOL and abs(g0['lo'] - 0.1772) <= TOL and (abs(g0['hi'] - 0.3453) <= TOL) and (r0['n'] == 5806) and (r0['лет'] == 9) and (abs(r0['разрыв'] - float('nan')) <= TOL)
    print(f"РЕГРЕССИЯ (а): GC {g0['GC']:.4f} [{g0['lo']:.4f}, {g0['hi']:.4f}] n {r0['n']} знам. {r0['разрыв']:.4f}: {('сошлась' if ok else 'НЕ СОШЛАСЬ')}", flush=True)
    if not ok:
        print('ОСТАНОВ')
        return 1
    rw = запуск('wrap', False)
    расх = max((abs(a - b) / B for a, b in zip(r0['df']['модель отклика на сутки|NPV'], rw['df']['модель отклика на сутки|NPV'])))
    print(f'РЕГРЕССИЯ (б): обёртка без purge, макс. расхождение NPV линии по годам {расх:.2e} млрд', flush=True)
    if расх > 1e-06 or abs(rw['линии'][e19.LINE]['GC'] - g0['GC']) > 1e-09:
        print('ОСТАНОВ: обёртка не воспроизводит головную цепь')
        return 1
    rp = запуск('purge', True)
    gp = rp['линии'][e19.LINE]
    d = r0['d']
    опт = set(d.filter(pl.col('тип').cast(pl.Utf8) == 'ОПТ')['op_id'].to_list())
    mi0 = d['vnr'].dt.year() * 12 + d['vnr'].dt.month() - 1
    годы_пр = [int(y) for y in rp['df']['год'].to_list()]
    годы_0 = [int(y) for y in r0['df']['год'].to_list()]
    S0, Sp = (множество(r0), множество(rp))
    dG = gp['GC'] - g0['GC']
    jac = len(S0 & Sp) / len(S0 | Sp)
    if gp['lo'] <= 0:
        исход = 'Пг-3 (нижняя граница GC <= 0): СТОП'
    elif abs(dG) <= ПОРОГ:
        исход = 'Пг-1: утечка горизонта цели несущественна для головного вывода'
    else:
        исход = 'Пг-2: строгая цепь становится новой головной, обновление числа - решение lead'
    вывод('# Б14. Purge-тест утечки горизонта цели обучения (ep30_purge)')
    вывод('')
    вывод(f'Регрессия: головная цепь 0.2631 [0.1772, 0.3453], n 5806, знаменатель <withheld> сошлась; обёртка без purge воспроизводит NPV линии по годам (макс. расхождение {расх:.1e} млрд). Purge: обучение года Y только на операциях с mi0 + 24 <= 12*Y - 1; остальное как в головной цепи.')
    вывод(f"**ИСХОД: {исход}** (ΔGC = {dG:+.4f}, порог 0.02, нижняя граница {gp['lo']:.4f}).")
    вывод('')
    вывод('| цепь | n | лет | GC | lo | hi | повт. | ДМ | p | знаменатель | операций отбора | доля ОПТ в приросте |')
    вывод('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for имя, r, g in (('головная', r0, g0), ('purge', rp, gp)):
        вывод(f"| {имя} | {r['n']} | {r['лет']} | {g['GC']:.4f} | {g['lo']:.4f} | {g['hi']:.4f} | {g['повторов']} | {g['ДМ']:.2f} | {g['p']:.4f} | {r['разрыв']:.4f} | {len(множество(r))} | {доля_опт(r, опт):.3f} |")
    вывод('')
    вывод(f"Жаккар множеств отбора (линия «модель отклика на сутки», 9 лет) purge против головной: {jac:.4f}, расхождение {len(S0 ^ Sp)} операций. Годы purge: {годы_пр}; потерянные годы: {sorted(set(годы_0) - set(годы_пр)) or 'нет'}.")
    вывод('')
    вывод('| Y | обучение, головная | обучение, purge | потеря |')
    вывод('|---|---|---|---|')
    tsv = []
    for Y in годы_0:
        n0 = int((d['год'] < Y).sum())
        np_ = int(((d['год'] < Y) & (mi0 + 24 <= 12 * Y - 1)).sum())
        вывод(f'| {Y} | {n0} | {np_} | {n0 - np_} ({100 * (n0 - np_) / n0:.1f} %) |')
        tsv.append([Y, n0, np_, n0 - np_])
    вывод('')
    вывод('Ограничение: purge касается только модели прироста; модели длительности и объёмов не менялись; число обучающих операций считано по совокупности цепи ППП (5 806 операций).')
    вывод('')
    якорь('ep30 purge GC головная', g0['GC'])
    for k, g in (('purge', gp),):
        якорь(f'ep30 {k} GC', g['GC'])
        якорь(f'ep30 {k} GC_lo', g['lo'])
        якорь(f'ep30 {k} GC_hi', g['hi'])
        якорь(f'ep30 {k} повторов', g['повторов'], 'd')
        якорь(f'ep30 {k} ДМ p', g['p'])
    якорь('ep30 purge ΔGC', dG, '+.4f')
    якорь('ep30 purge знаменатель', rp['разрыв'])
    якорь('ep30 purge Жаккар', jac)
    якорь('ep30 purge расхождение отбора', len(S0 ^ Sp), 'd')
    якорь('ep30 purge доля ОПТ в приросте', доля_опт(rp, опт), '.3f')
    якорь('ep30 purge доля ОПТ в приросте головная', доля_опт(r0, опт), '.3f')
    якорь('ep30 purge лет', rp['лет'], 'd')
    (ROOT / cfg.results('ep30_purge.md')).write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
    with open(ROOT / cfg.out('ep30_purge.tsv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['Y', 'обучение_головная', 'обучение_purge', 'потеря'])
        w.writerows(tsv)
    return 0
if __name__ == '__main__':
    sys.exit(main())
