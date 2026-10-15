"""Out-of-sample quality of the production-gain (response) model by year and type."""
import config as cfg
import contextlib
import csv
import io
import pathlib
import sys
import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import binomtest, spearmanr
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import ep19_crossruler as e19
import stats
B = 1000000000.0
TOL = 0.0001
SEED_BOOT = 20260927
МИН_ГОД_ОПТ = 30
ПРОЧЬ = []

def вывод(s=''):
    ПРОЧЬ.append(s)
    print(s, flush=True)

def якорь(имя, знач, ф='.4f'):
    вывод(f'ЯКОРЬ {имя} = {format(знач, ф)}')

def метрики(y, m, b, npv, nh_m, nh_b):

    def r2(p):
        sst = ((y - y.mean()) ** 2).sum()
        return 1 - ((y - p) ** 2).sum() / sst

    def phi(p):
        a, c = ((p > 0).astype(float), (y > 0).astype(float))
        return float(np.corrcoef(a, c)[0, 1]) if a.std() > 0 and c.std() > 0 else float('nan')
    return {'n': len(y), 'MAE_м': np.abs(y - m).mean(), 'MAE_б': np.abs(y - b).mean(), 'MdAE_м': np.median(np.abs(y - m)), 'MdAE_б': np.median(np.abs(y - b)), 'R2_м': r2(m), 'R2_б': r2(b), 'Спирмен_м': spearmanr(y, m).statistic, 'Спирмен_б': spearmanr(y, b).statistic, 'знак_эффекта_м': float(((m > 0) == (y > 0)).mean()), 'знак_эффекта_б': float(((b > 0) == (y > 0)).mean()), 'phi_м': phi(m), 'phi_б': phi(b), 'знак_npv_м': float(((nh_m > 0) == (npv > 0)).mean()), 'знак_npv_б': float(((nh_b > 0) == (npv > 0)).mean())}

def main() -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        r = e19.прогон_exante('rq', e19.параметры(e19.CAP_ПП))
    g = r['линии'][e19.LINE]
    ok = abs(g['GC'] - 0.2631) <= TOL and abs(g['lo'] - 0.1772) <= TOL and (abs(g['hi'] - 0.3453) <= TOL) and (r['n'] == 5806) and (r['лет'] == 9) and (abs(r['разрыв'] - float('nan')) <= TOL)
    print(f"РЕГРЕССИЯ GC: {g['GC']:.4f} [{g['lo']:.4f}, {g['hi']:.4f}], n {r['n']}, знаменатель {r['разрыв']:.4f}: {('сошлась' if ok else 'НЕ СОШЛАСЬ')}", flush=True)
    if not ok:
        print('ОСТАНОВ')
        return 1
    d, руб_т = (r['d'], r['руб_т'])
    df = r['df']
    X, CAT = (stats.X, stats.CAT)
    нпв_линий = {int(y): (a, b_) for y, a, b_ in zip(df['год'], df['модель отклика на сутки|NPV'], df['медиана типа на сутки|NPV'])}
    годы = [int(y) for y in df['год'].to_list()]
    пред = {}
    for Y in годы:
        tr = d.filter(pl.col('год') < Y)
        pool = d.filter(pl.col('год') == Y)
        capex = pool['capex'].to_numpy()
        dur = pool['бригадо_сутки'].to_numpy().clip(1, 400)
        m = lgb.LGBMRegressor(objective='l1', n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42, n_jobs=1)
        m.fit(tr.select(X).to_pandas(), tr['inc_oil_t'].to_numpy(), categorical_feature=CAT)
        pm = m.predict(pool.select(X).to_pandas())
        med = tr.group_by('тип').agg(mm=pl.col('inc_oil_t').median())
        pb = pool.select('тип').join(med, on='тип', how='left')['mm'].fill_null(float(tr['inc_oil_t'].median())).to_numpy()
        fact = np.ones(pool.height, bool)
        budget, cap = (capex.sum(), dur.sum())
        npv_hat = pm * руб_т - capex
        npv = pool['npv'].to_numpy()
        sel = stats.greedy(npv_hat / dur, capex, dur, budget, cap)
        расх = abs(npv[sel].sum() / B - нпв_линий[Y][0] / B)
        if расх > 1e-06:
            print(f'ОСТАНОВ: год {Y}, NPV линии при повторной подгонке расходится на {расх:.2e} млрд')
            return 1
        пред[Y] = dict(y=pool['inc_oil_t'].to_numpy(), m=pm, b=pb, npv=npv, nh_m=npv_hat, nh_b=pb * руб_т - capex, тип=pool['тип'].cast(pl.Utf8).to_numpy())
        print(f'{Y}: n {pool.height}, обучение {tr.height}, NPV линии воспроизведён', flush=True)
    print('РЕГРЕССИЯ NPV линии по годам: сошлась', flush=True)

    def сводка(Ys, оп):
        sel = lambda k: np.concatenate([пред[Y][k][пред[Y]['тип'] == 'ОПТ' if оп else slice(None)] for Y in Ys])
        return метрики(sel('y'), sel('m'), sel('b'), sel('npv'), sel('nh_m'), sel('nh_b'))
    строки, рез = ([], {})
    for Y in годы:
        for оп in (False, True):
            n = int((пред[Y]['тип'] == 'ОПТ').sum()) if оп else len(пред[Y]['y'])
            if оп and n < МИН_ГОД_ОПТ:
                continue
            рез[Y, оп] = сводка([Y], оп)
    for оп in (False, True):
        рез['все', оп] = сводка(годы, оп)
    k = sum((рез[Y, False]['MAE_м'] < рез[Y, False]['MAE_б'] for Y in годы))
    k_опт = sum((рез[Y, True]['MAE_м'] < рез[Y, True]['MAE_б'] for Y in годы if (Y, True) in рез))
    n_опт_лет = sum(((Y, True) in рез for Y in годы))
    O = рез['все', True]['MAE_м'] < рез['все', True]['MAE_б']
    исход = 'М-A' if k >= 5 and O else 'М-B' if k <= 4 and (not O) else 'М-C'
    p_бин = binomtest(int(k), len(годы), 0.5).pvalue
    ae = np.array([[np.abs(пред[Y]['y'] - пред[Y]['b']).sum(), np.abs(пред[Y]['y'] - пред[Y]['m']).sum(), len(пред[Y]['y'])] for Y in годы])
    rg = np.random.default_rng(SEED_BOOT)
    diffs = []
    for _ in range(2000):
        i = rg.integers(0, len(годы), len(годы))
        s = ae[i].sum(axis=0)
        diffs.append((s[0] - s[1]) / s[2])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pt = (ae[:, 0].sum() - ae[:, 1].sum()) / ae[:, 2].sum()
    вывод('# Б11. Качество модели прироста добычи (ep30_response_quality)')
    вывод('')
    вывод(f'Цепь ППП, регрессия GC 0.2631 и NPV линии по годам сошлась. Тест: Y = {годы[0]}…{годы[-1]}, обучение на годах < Y, цель inc_oil_t (т за 24 мес), базовый прогноз - медиана типа на годах < Y. Описание модели: docs/epic/v3/response_model_spec.md.')
    вывод(f"**ИСХОД: {исход}** (k = {k} из {len(годы)} лет с MAE модели < базового; O = {('да' if O else 'нет')}); биномиальный p (k из 9, p0 = 0.5) = {p_бин:.4f}; ОПТ по годам: модель лучше в {k_опт} из {n_опт_лет} лет с n >= {МИН_ГОД_ОПТ}. Разность MAE (база − модель) на объединении: {pt:+.1f} т, 95 % ДИ [{lo:+.1f}, {hi:+.1f}], повторов {len(diffs)}.")
    вывод('')
    вывод('| год | группа | n | MAE мод/баз, т | MdAE мод/баз | R² мод/баз | Спирмен мод/баз | знак эффекта мод/баз | φ мод/баз | знак NPV мод/баз | NPV линий мод/медиана типа, млрд |')
    вывод('|---|---|---|---|---|---|---|---|---|---|---|')
    ключи = [(Y, о) for Y in годы for о in (False, True) if (Y, о) in рез] + [('все', False), ('все', True)]
    for kk in ключи:
        m = рез[kk]
        Y, оп = kk
        л = f'{нпв_линий[Y][0] / B:.3f} / {нпв_линий[Y][1] / B:.3f}' if Y != 'все' and (not оп) else ''
        вывод(f"| {Y} | {('ОПТ' if оп else 'все типы')} | {m['n']} | {m['MAE_м']:.0f} / {m['MAE_б']:.0f} | {m['MdAE_м']:.0f} / {m['MdAE_б']:.0f} | {m['R2_м']:.3f} / {m['R2_б']:.3f} | {m['Спирмен_м']:.3f} / {m['Спирмен_б']:.3f} | {m['знак_эффекта_м']:.3f} / {m['знак_эффекта_б']:.3f} | {m['phi_м']:.3f} / {m['phi_б']:.3f} | {m['знак_npv_м']:.3f} / {m['знак_npv_б']:.3f} | {л} |")
        строки.append([Y, 'ОПТ' if оп else 'все типы'] + [f'{m[c]:.4f}' if c != 'n' else m[c] for c in ('n', 'MAE_м', 'MAE_б', 'MdAE_м', 'MdAE_б', 'R2_м', 'R2_б', 'Спирмен_м', 'Спирмен_б', 'знак_эффекта_м', 'знак_эффекта_б', 'phi_м', 'phi_б', 'знак_npv_м', 'знак_npv_б')])
    вывод('')
    вывод('Ограничение: R² и MAE в тоннах за 24 мес на операцию; тест - все операции пула года Y (не только отобранные); качество предсказания эффекта не есть верность модели затрат и не есть доказательство «точности». nan: у базового прогноза знак/ранг постоянен внутри группы (медиана типа > 0 для всех операций типа), коэффициент не определён.')
    вывод('')
    for Y in годы:
        якорь(f'ep30 rq MAE модель {Y}', рез[Y, False]['MAE_м'], '.0f')
        якорь(f'ep30 rq MAE база {Y}', рез[Y, False]['MAE_б'], '.0f')
        якорь(f'ep30 rq Спирмен модель {Y}', рез[Y, False]['Спирмен_м'], '.3f')
        якорь(f'ep30 rq Спирмен база {Y}', рез[Y, False]['Спирмен_б'], '.3f')
    for nm, kk in (('все', ('все', False)), ('ОПТ', ('все', True))):
        m = рез[kk]
        for c in ('n', 'MAE_м', 'MAE_б', 'MdAE_м', 'MdAE_б', 'R2_м', 'R2_б', 'Спирмен_м', 'Спирмен_б', 'знак_эффекта_м', 'знак_эффекта_б', 'знак_npv_м', 'знак_npv_б'):
            якорь(f'ep30 rq {nm} {c}', m[c], 'd' if c == 'n' else '.0f' if c.startswith(('MAE', 'MdAE')) else '.3f')
    якорь('ep30 rq k лет модель лучше базового', k, 'd')
    якорь('ep30 rq k лет ОПТ модель лучше базового', k_опт, 'd')
    якорь('ep30 rq биномиальный p', p_бин)
    якорь('ep30 rq разность MAE база минус модель, т', pt, '+.1f')
    якорь('ep30 rq разность MAE ДИ низ, т', lo, '+.1f')
    якорь('ep30 rq разность MAE ДИ верх, т', hi, '+.1f')
    якорь('ep30 rq повторов бутстрэпа', len(diffs), 'd')
    (ROOT / cfg.results('ep30_response_quality.md')).write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
    with open(ROOT / cfg.out('response_quality.tsv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['год', 'группа', 'n', 'MAE_мод', 'MAE_баз', 'MdAE_мод', 'MdAE_баз', 'R2_мод', 'R2_баз', 'Спирмен_мод', 'Спирмен_баз', 'знак_эффекта_мод', 'знак_эффекта_баз', 'phi_мод', 'phi_баз', 'знак_npv_мод', 'знак_npv_баз'])
        w.writerows(строки)
    return 0
if __name__ == '__main__':
    sys.exit(main())
