"""Leave-one-year-out, BCa intervals and the exact sign-flip test."""
from __future__ import annotations
import config as cfg
import itertools
import pathlib
import sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ep19_crossruler as e19
import stats
from ep06_ablation import _ряд_линейкой, _цены
from ep30_crossacct import ОЖИД, собрать, сверить, якорь
from perm_control import ПОРОГ_БОНФ, перестановочное_p
B = 1000000000.0
LINE = e19.LINE
TSV = cfg.out('ep30_loyo_bca.tsv')
ЦЕПЬ = 'цепь ППП, 5 806 операций'

def реплики(v, f, o, rng, reps=2000):
    n = len(f)
    gcs = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        den = (o[i] - f[i]).sum()
        if abs(den) > 1e-09 * max(abs(o[i].sum()), abs(f[i].sum()), 1.0):
            gcs.append((v[i] - f[i]).sum() / den)
    return np.array(gcs)

def gc_точка(v, f, o):
    return float((v - f).sum() / (o - f).sum())

def джекнайф(v, f, o):
    return np.array([gc_точка(np.delete(v, k), np.delete(f, k), np.delete(o, k)) for k in range(len(f))])

def bca(g, theta, jack):
    доля = float((g < theta).mean())
    if not 0.0 < доля < 1.0:
        return None
    z0 = float(norm.ppf(доля))
    m = jack.mean()
    num, den = (((m - jack) ** 3).sum(), 6.0 * ((m - jack) ** 2).sum() ** 1.5)
    if not np.isfinite(den) or den == 0:
        return None
    a = float(num / den)
    гр = []
    for q in (0.025, 0.975):
        zq = norm.ppf(q)
        знам = 1.0 - a * (z0 + zq)
        if знам <= 0:
            return None
        гр.append(100 * norm.cdf(z0 + (z0 + zq) / знам))
    lo, hi = (np.percentile(g, гр[0]), np.percentile(g, гр[1]))
    return (float(lo), float(hi), z0, a)

def главная() -> int:
    print(f'=== Б1.3: LOYO, BCa, точный знаковый тест; {ЦЕПЬ}, 9 лет ===')
    print('правило чтения: docs/epic/b1_reading_rules.md, раздел 3 (записано до расчёта)')
    наш, л1 = собрать()
    g0 = наш['линии'][LINE]
    ok = all([сверить('GC(ППП)', g0['GC'], ОЖИД['GC_НН']), сверить('GC(ППП) ДИ низ', g0['lo'], ОЖИД['lo_НН']), сверить('GC(ППП) ДИ верх', g0['hi'], ОЖИД['hi_НН']), сверить('знаменатель Н', наш['разрыв'], ОЖИД['знам_Н'])])
    print(f"  n = {наш['n']} (объявлено {e19.N_ОБЪЯВЛЕНО}), лет {наш['лет']}")
    ok &= наш['n'] == e19.N_ОБЪЯВЛЕНО and наш['лет'] == 9
    if not ok:
        print('\nОСТАНОВ по п. 0.3 правил: регрессия не сошлась.')
        return 1
    годы = [г['год'] for г in наш['отборы']]
    f = наш['df']['факт инженеров|NPV'].to_numpy()
    o = наш['df']['оракул по факту|NPV'].to_numpy()
    v = наш['df'][f'{LINE}|NPV'].to_numpy()
    fл = л1['df']['факт инженеров|NPV'].to_numpy()
    oл = л1['df']['оракул по факту|NPV'].to_numpy()
    vл = _ряд_линейкой(наш['отборы'], _цены(л1), LINE)
    сумма_d = float((v - f).sum()) / B
    gc_сумм = сумма_d / (float((o - f).sum()) / B)
    if not сверить('GC из сумм годовых рядов', gc_сумм, ОЖИД['GC_НН']):
        print('\nОСТАНОВ: ряды не воспроизводят GC.')
        return 1
    print(f'  сумма d_i = {сумма_d:.4f} млрд, знаменатель {(o - f).sum() / B:.4f} млрд')
    print('\nГОДОВЫЕ СЫРЫЕ РАЗНОСТИ d_i = v_i - f_i (млрд руб) И GC ПО ГОДАМ:')
    gcy = stats.gc_per_year(v, f, o)
    for Y, d, gy in zip(годы, (v - f) / B, gcy):
        print(f'  {Y}: d = {d:+.4f}, GC года {gy:+.4f}')
        якорь(f'd_i {Y} млрд', d)
        якорь(f'GC года {Y}', gy)
    print(f'\nLOYO, линия «{LINE}»: GC без года k, ДИ `stats.bootstrap_gc` на 8 годах (перцентильный, зерно {e19.SEED} перед каждым k)')
    print(f"{'без года':>9s} {'GC(-k)':>8s} {'95% ДИ':>20s} {'повт.':>6s} {'знам.,млрд':>11s}")
    лоё = []
    for k, Y in enumerate(годы):
        i = np.delete(np.arange(9), k)
        stats.rng = np.random.default_rng(e19.SEED)
        gc, lo, hi, reps = stats.bootstrap_gc(v[i], f[i], o[i])
        зн = float((o[i] - f[i]).sum()) / B
        лоё.append((Y, gc, lo, hi, reps, зн))
        print(f'{Y:9d} {gc:8.4f} [{lo:8.4f},{hi:8.4f}] {reps:6d} {зн:11.4f}')
        якорь(f'LOYO без {Y} GC', gc)
        якорь(f'LOYO без {Y} ДИ низ', lo)
        якорь(f'LOYO без {Y} ДИ верх', hi)
        якорь(f'LOYO без {Y} повторов', reps, 'd')
    if all((r[1] > 0 for r in лоё)):
        if all((r[2] > 0 for r in лоё)):
            л_код, л_текст = ('Л-A', 'знак и различимость устойчивы к исключению любого одного года')
        else:
            плохие = [r[0] for r in лоё if r[2] <= 0]
            л_код = 'Л-B'
            л_текст = 'знак устойчив; различимость теряется при исключении лет: ' + ', '.join(map(str, плохие))
    else:
        плохие = [r[0] for r in лоё if r[1] <= 0]
        л_код, л_текст = ('Л-C', 'знак зависит от года: ' + ', '.join(map(str, плохие)))
    print(f'\nИСХОД LOYO (п. 3.4): {л_код} -- {л_текст}')
    якорь('номер исхода LOYO (1 Л-A, 2 Л-B, 3 Л-C)', {'Л-A': 1, 'Л-B': 2, 'Л-C': 3}[л_код], 'd')
    якорь('минимум GC(-k)', min((r[1] for r in лоё)))
    якорь('максимум GC(-k)', max((r[1] for r in лоё)))
    якорь('минимум нижней границы LOYO', min((r[2] for r in лоё)))
    величины = {'GC(ППП)': (v, f, o), 'ДОР-Л1(ППП)': (vл, fл, oл)}
    итог = {}
    print('\nBCa (п. 3.3). Сверка реплик с `stats.bootstrap_gc` до 1e-12 (зерно пересеивается):')
    for имя, (vv, ff, oo) in величины.items():
        stats.rng = np.random.default_rng(e19.SEED)
        gc, lo_p, hi_p, reps = stats.bootstrap_gc(vv, ff, oo)
        g = реплики(vv, ff, oo, np.random.default_rng(e19.SEED))
        d1, d2 = (abs(np.percentile(g, 2.5) - lo_p), abs(np.percentile(g, 97.5) - hi_p))
        print(f'  {имя}: перцентили реплик против bootstrap_gc: расхождение {d1:.2e} / {d2:.2e}, повторов {len(g)} против {reps}')
        if d1 > 1e-12 or d2 > 1e-12 or len(g) != reps:
            print('ОСТАНОВ по п. 3.3: реплики не воспроизводят интервал bootstrap_gc.')
            return 1
        jack = джекнайф(vv, ff, oo)
        r = bca(g, gc, jack)
        доля_ниже = float((g < gc).mean())
        итог[имя] = (gc, lo_p, hi_p, reps, r, доля_ниже, jack)
    print(f"\n{'величина':14s} {'оценка':>7s} {'перцентильный':>20s} {'BCa':>20s} {'z0':>7s} {'a':>8s} {'доля<оц.':>8s}")
    коды = {}
    for имя, (gc, lo_p, hi_p, reps, r, дн, jack) in итог.items():
        if r is None:
            bs = 'не определён'
            код, z0s, as_ = ('Б-4', '-', '-')
        else:
            lo_b, hi_b, z0, a = r
            bs = f'[{lo_b:7.4f},{hi_b:7.4f}]'
            z0s, as_ = (f'{z0:7.4f}', f'{a:8.4f}')
            код = 'Б-1' if lo_b > 0 and lo_p > 0 else 'Б-2' if lo_b <= 0 and lo_p > 0 else 'Б-3' if lo_b <= 0 and lo_p <= 0 else 'вне перечисления: BCa>0, перц.<=0'
        коды[имя] = код
        print(f'{имя:14s} {gc:7.4f} [{lo_p:8.4f},{hi_p:8.4f}] {bs:>20s} {z0s:>7s} {as_:>8s} {дн:8.4f}')
        якорь(f'{имя} точка', gc)
        якорь(f'{имя} перцентильный низ', lo_p)
        якорь(f'{имя} перцентильный верх', hi_p)
        якорь(f'{имя} повторов', reps, 'd')
        якорь(f'{имя} доля реплик ниже точки', дн)
        if r is not None:
            якорь(f'{имя} BCa низ', r[0])
            якорь(f'{имя} BCa верх', r[1])
            якорь(f'{имя} z0', r[2])
            якорь(f'{имя} ускорение a', r[3])
        якорь(f'{имя} номер исхода BCa (1..4 = Б-1..Б-4)', int(код[2]) if код[:2] == 'Б-' and код[2].isdigit() and (len(код) == 3) else 0, 'd')
        якорь(f'{имя} джекнайф min', float(jack.min()))
        якорь(f'{имя} джекнайф max', float(jack.max()))
    for имя, код in коды.items():
        print(f'ИСХОД BCa (п. 3.5), {имя}: {код}')
    print('  решающим для текста объявлен BCa; перцентильный напечатан рядом. Число лет -- девять.')
    знаки = np.array(list(itertools.product([1, -1], repeat=9)))
    print(f'\nТОЧНЫЙ ЗНАКОВЫЙ ТЕСТ, {ЦЕПЬ}: `perm_control.перестановочное_p`, 2^9 = {len(знаки)} расстановок, квант 2/512 = {2 / 512:.6f}, порог Бонферрони {ПОРОГ_БОНФ:.4f} (ведомость 5.2)')
    print('  Это замер на цепи ППП; отчёт perm_control (p = 0.003906 для линии «модель отклика на сутки») сделан на БАЗОВОЙ цепи, 5 871 операция.')
    ряды = {l: (наш['df'][f'{l}|NPV'].to_numpy() - f) / B for l in ('предсказ. абсолютный NPV', 'медиана типа на сутки', LINE)}
    ряды['ДОР-Л1: V_НЛ1 против факта Л1'] = (vл - fл) / B
    print(f"{'ряд d_i':34s} {'T':>7s} {'p':>9s} {'числ./512':>10s} {'уровней':>8s}")
    for имя, d in ряды.items():
        T, p, числ, всего, ур = перестановочное_p(d, знаки)
        print(f'{имя:34s} {T:7.3f} {p:9.6f} {числ:6d}/{всего:<3d} {ур:8d}')
        ключ = {'предсказ. абсолютный NPV': 'абсолютный', 'медиана типа на сутки': 'медиана', LINE: 'отклик'}.get(имя, 'ДОР-Л1')
        якорь(f'точный p ППП {ключ}', p, '.6f')
        якорь(f'точный p ППП {ключ} числитель', числ, 'd')
        якорь(f'точный p ППП {ключ} T', T, '.3f')
        print(f"    {('проходит' if p < ПОРОГ_БОНФ else 'не проходит')} порог Бонферрони {ПОРОГ_БОНФ:.4f}; {('p < 0.05' if p < 0.05 else 'p >= 0.05: на девяти годах точный тест не отвергает')}")
    print('  тест и интервалы не являются двумя независимыми подтверждениями (п. 3.6).')
    with open(TSV, 'w', encoding='utf-8') as fh:
        fh.write('без_года\tGC\tДИ_низ\tДИ_верх\tповторов\tзнаменатель_млрд\n')
        for Y, gc, lo, hi, reps, зн in лоё:
            fh.write(f'{Y}\t{gc:.4f}\t{lo:.4f}\t{hi:.4f}\t{reps}\t{зн:.4f}\n')
        fh.write('\nвеличина\tточка\tперц_низ\tперц_верх\tBCa_низ\tBCa_верх\tz0\ta\n')
        for имя, (gc, lo_p, hi_p, reps, r, дн, jack) in итог.items():
            rr = ['', '', '', ''] if r is None else [f'{x:.4f}' for x in r]
            fh.write('\t'.join([имя, f'{gc:.4f}', f'{lo_p:.4f}', f'{hi_p:.4f}'] + rr) + '\n')
    print(f'\nтаблица: {TSV}')
    return 0
if __name__ == '__main__':
    sys.exit(главная())
