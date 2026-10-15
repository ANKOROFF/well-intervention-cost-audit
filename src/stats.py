"""Statistics of the chronological run (intervals, tests)."""
import config as cfg
import numpy as np
import polars as pl
import lightgbm as lgb
from scipy import stats as st
import oracle
from npv import Params, npv_table
B = 1000000000.0
FEATS = ['t_qn', 't_qzh', 't_wcut', 't_pzab', 't_ppl', 't_kpr', 't_kh', 't_skin', 't_hdin', 't_f', 't_hsp', 'пр_kpr', 'пр_qn', 'мес_до_снимка', 'dur']
CAT = ['тип', 't_state', 't_method']
X = FEATS + CAT
LINES = ['факт инженеров', 'случайный отбор', 'предсказ. абсолютный NPV', 'медиана типа на сутки', 'модель отклика на сутки', 'оракул по факту']
rng = np.random.default_rng(20260927)

def build(p: Params | None=None):
    n = npv_table(p or Params()).select('op_id', 'npv', 'capex', 'бригадо_сутки', 'доход_диск')
    ev = pl.read_parquet(cfg.data('interim/gtm_events.parquet')).filter(pl.col('выбираемая'))
    dc = pl.read_parquet(cfg.data('interim/decline.parquet')).select('op_id', 'inc_oil_t')
    dr = pl.read_parquet(cfg.data('interim/drawdown.parquet'))
    pr = dr.pivot(values='просадка', index=['well', 'mi'], on='величина', aggregate_function='max').rename({'kpr': 'пр_kpr', 'qn': 'пр_qn'})
    d = ev.join(dc, on='op_id').join(n, on='op_id').with_columns(mi=pl.col('beg').dt.year() * 12 + pl.col('beg').dt.month() - 2, мес_до_снимка=(pl.col('beg') - pl.col('t_date')).dt.total_days()).join(pr, on=['well', 'mi'], how='left').drop_nulls(['npv', 'capex', 'бригадо_сутки', 'год'])
    for c in CAT:
        d = d.with_columns(pl.col(c).cast(pl.Utf8).fill_null('нет').cast(pl.Categorical))
    руб_т = float((d['доход_диск'] / d['inc_oil_t'].abs().clip(1, None)).median())
    return (d, руб_т)

def greedy(score, capex, dur, budget, cap, positive=True):
    sel = np.zeros(len(score), bool)
    b = c = 0.0
    for i in np.argsort(-score):
        if positive and score[i] <= 0:
            continue
        if b + capex[i] <= budget and c + dur[i] <= cap:
            sel[i] = True
            b += capex[i]
            c += dur[i]
    return sel

def year_run(d, руб_т, Y, pool_span, n_rand=200, отборы=None):
    tr = d.filter(pl.col('год') < Y)
    pool = d.filter(pl.col('год').is_between(Y, Y + pool_span - 1))
    fact = (pool['год'] == Y).to_numpy()
    if tr.height < 300 or fact.sum() < 20:
        return None
    npv = pool['npv'].to_numpy()
    capex = pool['capex'].to_numpy()
    dur = pool['бригадо_сутки'].to_numpy().clip(1, 400)
    budget, cap = (capex[fact].sum(), dur[fact].sum())
    m = lgb.LGBMRegressor(objective='l1', n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42)
    m.fit(tr.select(X).to_pandas(), tr['inc_oil_t'].to_numpy(), categorical_feature=CAT)
    npv_hat = m.predict(pool.select(X).to_pandas()) * руб_т - capex
    med = tr.group_by('тип').agg(mm=pl.col('inc_oil_t').median())
    npv_med = pool.select('тип').join(med, on='тип', how='left')['mm'].fill_null(float(tr['inc_oil_t'].median())).to_numpy() * руб_т - capex
    rand = np.mean([npv[greedy(rng.permutation(len(npv)).astype(float), capex, dur, budget, cap, positive=False)].sum() for _ in range(n_rand)])
    sel = {'факт инженеров': fact, 'предсказ. абсолютный NPV': greedy(npv_hat, capex, dur, budget, cap), 'медиана типа на сутки': greedy(npv_med / dur, capex, dur, budget, cap), 'модель отклика на сутки': greedy(npv_hat / dur, capex, dur, budget, cap), 'оракул по факту': oracle.exact(npv, capex, dur, budget, cap, метка=f'stats {Y} пул {pool_span}')}
    out = {'год': Y, 'кандидатов': pool.height, 'бюджет_млрд': budget / B, 'случайный отбор|NPV': float(rand), 'оп_с_npv_hat_gt0': int((npv_hat > 0).sum()), 'линии_совпали': bool((sel['предсказ. абсолютный NPV'] == sel['модель отклика на сутки']).all())}
    for k, s in sel.items():
        out[f'{k}|NPV'] = float(npv[s].sum())
        out[f'{k}|оп'] = int(s.sum())
    out['случайный отбор|оп'] = -1
    ж = greedy(npv / dur, capex, dur, budget, cap)
    out['оракул жадный (до сведения)|NPV'] = float(npv[ж].sum())
    out['оракул жадный (до сведения)|оп'] = int(ж.sum())
    if отборы is not None:
        отборы.append({'год': Y, 'op_id': pool['op_id'].to_numpy(), 'npv': npv, 'capex': capex, 'dur': dur, 'бюджет': budget, 'ёмкость': cap, 'отбор': {k: s.copy() for k, s in sel.items()}})
    return out

def dm_test(diff):
    n = len(diff)
    dbar = diff.mean()
    gamma0 = ((diff - dbar) ** 2).sum() / n
    if gamma0 <= 0:
        return (np.nan, np.nan, n)
    dm = dbar / np.sqrt(gamma0 / n)
    hln = dm * np.sqrt((n - 1) / n)
    p = 2 * (1 - st.t.cdf(abs(hln), df=n - 1))
    return (float(hln), float(p), n)

def bootstrap_gc(v, f, o, reps=2000):
    n = len(f)
    gcs = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        den = (o[i] - f[i]).sum()
        if abs(den) > 1e-09 * max(abs(o[i].sum()), abs(f[i].sum()), 1.0):
            gcs.append((v[i] - f[i]).sum() / den)
    if not gcs:
        return (np.nan, np.nan, np.nan, 0)
    g = np.array(gcs)
    return (float((v - f).sum() / (o - f).sum()), float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5)), len(gcs))

def gc_per_year(v, f, o):
    den = o - f
    _sc = np.maximum(np.maximum(np.abs(o), np.abs(f)), 1.0)
    return np.where(np.abs(den) > 1e-09 * _sc, (v - f) / np.where(den == 0, 1, den), np.nan)

def report(d, руб_т, pool_span, label):
    rows = [r for Y in range(2016, 2025) if (r := year_run(d, руб_т, Y, pool_span))]
    df = pl.DataFrame(rows)
    print(f"\n{'=' * 78}\n{label}  |  лет в прогоне: {df.height}\n{'=' * 78}")
    show = df.select('год', 'кандидатов', бюджет=pl.col('бюджет_млрд').round(2))
    for k in LINES:
        show = show.with_columns((pl.Series(df[f'{k}|NPV']) / B).round(2).alias(k))
    print('NPV по годам, млрд ₽:')
    print(show.to_pandas().to_string(index=False))
    v = {k: df[f'{k}|NPV'].to_numpy() for k in LINES}
    f, o = (v['факт инженеров'], v['оракул по факту'])
    print('\nGC ПО ГОДАМ (доля закрытого разрыва в каждом году отдельно):')
    gy = df.select('год')
    for k in LINES[1:-1]:
        gy = gy.with_columns(pl.Series(gc_per_year(v[k], f, o)).round(3).alias(k))
    print(gy.to_pandas().to_string(index=False))
    for k in LINES[1:-1]:
        g = gc_per_year(v[k], f, o)
        g = g[~np.isnan(g)]
        print(f'  {k:26s} медиана по годам {np.median(g):6.3f}  разброс [{g.min():6.3f}, {g.max():6.3f}]  лет с GC > 0: {int((g > 0).sum())} из {len(g)}')
    print(f'\nАГРЕГИРОВАННЫЙ GC, ДИ бутстрэпом по годам, ДМ против факта:')
    print(f"{'линия':26s} {'NPV, млрд':>10s} {'GC':>7s} {'95% ДИ GC':>18s} {'повт.':>6s} {'ДМ стат.':>9s} {'p':>7s}")
    res = []
    for k in LINES:
        gc, lo, hi, reps = bootstrap_gc(v[k], f, o)
        if k == 'факт инженеров':
            dm, p, n = (np.nan, np.nan, len(f))
        else:
            dm, p, n = dm_test((v[k] - f) / B)
        print(f'{k:26s} {v[k].sum() / B:10.2f} {gc:7.3f} [{lo:7.3f}, {hi:6.3f}] {reps:6d} {dm:9.2f} {p:7.4f}')
        res.append({'линия': k, 'NPV_млрд': v[k].sum() / B, 'GC': gc, 'GC_lo': lo, 'GC_hi': hi, 'повторов': reps, 'ДМ': dm, 'p': p, 'лет': n})
    pairs = [('модель отклика на сутки', 'предсказ. абсолютный NPV'), ('модель отклика на сутки', 'медиана типа на сутки'), ('предсказ. абсолютный NPV', 'медиана типа на сутки')]
    print(f'\nПАРНЫЕ СРАВНЕНИЯ ЛИНИЙ МЕЖДУ СОБОЙ (ДМ на разности NPV):')
    print('  без этого теста утверждать, что одна линия лучше другой, нельзя; пересечение\n  доверительных интервалов само по себе ничего не доказывает.')
    print(f'  ВНИМАНИЕ: сравнений {len(pairs)}, поправки на множественность нет.\n  Порог Бонферрони при {len(pairs)} сравнениях: p < {0.05 / len(pairs):.4f}')
    for a, b in pairs:
        dm, pv, n = dm_test((v[a] - v[b]) / B)
        d = (v[a] - v[b]).sum() / B
        if pv != pv:
            verdict = 'тождественны' if abs(d) < 1e-09 else f'разность постоянна ({d:+.2f} млрд каждый год), тест Дибольда-Мариано неприменим'
        elif pv < 0.05:
            verdict = 'значимо лучше' if d > 0 else 'значимо хуже'
        else:
            verdict = 'не различимы'
        bonf = '' if pv != pv else ' [проходит Бонферрони]' if pv < 0.05 / len(pairs) else ' [Бонферрони НЕ проходит]' if pv < 0.05 else ''
        print(f'  {a:26s} против {b:26s}: разность {d:7.2f} млрд, ДМ {dm:6.2f}, p = {pv:.4f} -> {verdict}{bonf}')
    ident = int(df['линии_совпали'].sum())
    print(f'\nТОЖДЕСТВЕННОСТЬ ЛИНИЙ «предсказ. абсолютный NPV» и «модель отклика на сутки»:')
    print(f'  линии отличаются ТОЛЬКО порядком жадного отбора (npv_hat против npv_hat/сутки);')
    print(f'  отобранные множества совпали в {ident} годах из {df.height}.')
    print(f'  причина измерима -- отобрано операций против числа операций с npv_hat > 0:')
    for r in df.select('год', 'оп_с_npv_hat_gt0', 'предсказ. абсолютный NPV|оп', 'модель отклика на сутки|оп', 'линии_совпали').iter_rows():
        print(f"    {r[0]}: с npv_hat > 0 {r[1]:>5}, отобрано {r[2]:>5} и {r[3]:>5} -> {('совпали' if r[4] else 'РАЗНЫЕ')}")
    if ident == df.height:
        print(f'  все {df.height} лет: обе линии берут ВСЁ с npv_hat > 0, порядок отбора не играет роли, и это ОДНА линия, а не две.')
    print(f'\nразрыв оракул - факт: {(o - f).sum() / B:.2f} млрд ₽ за {len(f)} лет')
    print(f'лет, где линия строго лучше факта:')
    for k in LINES[1:]:
        print(f'  {k:26s} {int((v[k] > f).sum()):2d} из {len(f)}')
    return (df, pl.DataFrame(res))
if __name__ == '__main__':
    d, руб_т = build()
    print(f'операций: {d.height:,} | цена тонны из фактического NPV: {руб_т:,.0f} ₽/т')
    df1, r1 = report(d, руб_т, 1, 'ПУЛ = ГОД Y (основная постановка)')
    df3, r3 = report(d, руб_т, 3, 'ПУЛ = Y..Y+2 (расширение, заднее знание о составе)')
    r1.write_parquet(cfg.data('interim/sh7_pool1.parquet'))
    r3.write_parquet(cfg.data('interim/sh7_pool3.parquet'))
    print('\nВНИМАНИЕ: выборка мала -- девять лет. Тест Дибольда-Мариано на девяти')
    print('наблюдениях имеет слабую мощность, доверительные интервалы GC широки.')
