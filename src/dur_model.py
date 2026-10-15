"""Ex-ante forecast of operation duration and well downtime at the decision date."""
import config as cfg
import numpy as np
import polars as pl
import lightgbm as lgb
MAX_DUR = 400
MIN_TRAIN = 200
HIST_WIN = 50
NUM = ['t_qn', 't_qzh', 't_wcut', 't_pzab', 't_ppl', 't_kpr', 't_kh', 't_skin', 't_hdin', 't_ppr', 't_f', 't_hsp', 'мес_до_снимка', 'qn_do', 'qzh_do', 'pzab_do', 'ppl_do', 'wcut_do', 'уровень_мэр_т_сут', 'доля_работы_мес', 'w_оп_прошлых', 'w_dur_посл', 'w_dur_сред', 'w_простой_сред', 'w_мес_с_прошлой', 'тип_dur_мед', 'тип_простой_мед', 'тип_оп_прошлых', 'мес_года']
CAT = ['тип', 't_state', 't_method', 't_pump', 'метод_до', 'сост_до']
FEATS = NUM + CAT
TARGETS = ['dur_цель', 'простой_цель']
EXCLUDED = ['p_beg', 'delay', 'qn_plan', 'qn_po', 'qzh_po', 'wcut_po', 'пласты', 'end', 'vnr', 'dur', 'vnr_lag']

def mi(col):
    return col.dt.year() * 12 + col.dt.month() - 1

def plan_is_retro(ev):
    p = ev.drop_nulls(['p_beg', 'beg'])
    same = int((p['p_beg'] == p['beg']).sum())
    return (same, p.height, 100.0 * same / p.height)

def history(ev):
    done = ev.drop_nulls(['well', 'end', 'dur_цель']).select('well', 'тип', 'dur_цель', 'простой_цель', mi_end=mi(pl.col('end'))).sort('mi_end')
    w = done.sort(['well', 'mi_end']).with_columns(n=pl.col('dur_цель').cum_count().over('well'), s=pl.col('dur_цель').cum_sum().over('well'), sp=pl.col('простой_цель').cum_sum().over('well')).group_by(['well', 'mi_end']).agg(w_оп_прошлых=pl.col('n').max().cast(pl.Float64), w_dur_посл=pl.col('dur_цель').last(), w_dur_сред=pl.col('s').max() / pl.col('n').max(), w_простой_сред=pl.col('sp').max() / pl.col('n').max()).sort('mi_end')
    t = done.sort('mi_end').with_columns(dm=pl.col('dur_цель').rolling_median(HIST_WIN, min_samples=5).over('тип'), pm=pl.col('простой_цель').rolling_median(HIST_WIN, min_samples=5).over('тип'), tn=pl.col('dur_цель').cum_count().over('тип')).group_by(['тип', 'mi_end']).agg(тип_dur_мед=pl.col('dm').last(), тип_простой_мед=pl.col('pm').last(), тип_оп_прошлых=pl.col('tn').max().cast(pl.Float64)).sort('mi_end')
    return (w, t)

def features():
    ev = pl.read_parquet(cfg.data('interim/gtm_events.parquet')).filter(pl.col('выбираемая')).with_columns(dur_цель=(pl.col('end') - pl.col('beg')).dt.total_days().clip(0, MAX_DUR).cast(pl.Float64), простой_цель=(pl.col('vnr') - pl.col('beg')).dt.total_days().clip(0, MAX_DUR).cast(pl.Float64))
    lvl = pl.read_parquet(cfg.data('interim/preop_level.parquet')).select('op_id', 'уровень_мэр_т_сут', 'доля_работы_мес')
    w, t = history(ev)
    d = ev.join(lvl, on='op_id', how='left').with_columns(mi_beg=mi(pl.col('beg')), мес_года=pl.col('beg').dt.month().cast(pl.Float64), мес_до_снимка=(pl.col('beg') - pl.col('t_date')).dt.total_days().cast(pl.Float64)).with_columns(mi_src=pl.col('mi_beg') - 1)
    d = d.sort('mi_src').join_asof(w, left_on='mi_src', right_on='mi_end', by='well', strategy='backward').rename({'mi_end': 'mi_src_w'})
    d = d.sort('mi_src').join_asof(t, left_on='mi_src', right_on='mi_end', by='тип', strategy='backward').rename({'mi_end': 'mi_src_t'})
    for c, что in (('mi_src_w', 'истории скважины'), ('mi_src_t', 'истории типа')):
        bad = d.filter(pl.col(c).is_not_null() & (pl.col(c) >= pl.col('mi_beg')))
        assert bad.height == 0, f'утечка будущего в признаках {что}: {bad.height} операций'
    d = d.with_columns(w_мес_с_прошлой=(pl.col('mi_src') - pl.col('mi_src_w')).cast(pl.Float64), w_оп_прошлых=pl.col('w_оп_прошлых').fill_null(0.0), тип_оп_прошлых=pl.col('тип_оп_прошлых').fill_null(0.0))
    for c in CAT:
        d = d.with_columns(pl.col(c).cast(pl.Utf8).fill_null('нет').cast(pl.Categorical))
    for c in NUM:
        d = d.with_columns(pl.col(c).cast(pl.Float64, strict=False))
    return d

def fit(tr, target, objective='l1'):
    m = lgb.LGBMRegressor(objective=objective, n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42)
    m.fit(tr.select(FEATS).to_pandas(), tr[target].to_numpy(), categorical_feature=CAT)
    return m

def type_median(tr, te, target):
    med = tr.group_by('тип').agg(m=pl.col(target).median())
    g = float(tr[target].median())
    return te.select('тип').join(med, on='тип', how='left')['m'].fill_null(g).to_numpy()

def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

def holdout(d, split_year=2022, last_year=2024):
    tr = d.filter(pl.col('end').dt.year() < split_year).drop_nulls(TARGETS)
    te = d.filter(pl.col('beg').dt.year().is_between(split_year, last_year)).drop_nulls(TARGETS)
    out = {'n_train': tr.height, 'n_test': te.height, 'train_end_max': tr['end'].max(), 'test_beg_min': te['beg'].min()}
    for target in TARGETS:
        y = te[target].to_numpy()
        base = type_median(tr, te, target)
        r = {'база_MAE': mae(y, base)}
        for obj, name in (('l1', 'L1_MAE'), ('l2', 'L2_MAE')):
            p = fit(tr, target, obj).predict(te.select(FEATS).to_pandas())
            r[name] = mae(y, np.clip(p, 0, MAX_DUR))
        r['выигрыш_L1_пц'] = 100.0 * (r['база_MAE'] - r['L1_MAE']) / r['база_MAE']
        r['выигрыш_L2_пц'] = 100.0 * (r['база_MAE'] - r['L2_MAE']) / r['база_MAE']
        out[target] = r
    return (out, tr, te)

def walk_forward(d):
    rows, diag = ([], [])
    years = sorted(d['beg'].dt.year().unique().drop_nulls().to_list())
    for Y in years:
        te = d.filter(pl.col('beg').dt.year() == Y)
        tr = d.filter(pl.col('end').dt.year() < Y).drop_nulls(TARGETS)
        r = te.select('op_id')
        if tr.height >= MIN_TRAIN:
            how = 'модель'
            for target, name in zip(TARGETS, ['dur_пред', 'простой_пред']):
                p = fit(tr, target).predict(te.select(FEATS).to_pandas())
                r = r.with_columns(pl.Series(name, np.clip(p, 0, MAX_DUR)))
        elif tr.height:
            how = 'медиана типа'
            for target, name in zip(TARGETS, ['dur_пред', 'простой_пред']):
                r = r.with_columns(pl.Series(name, type_median(tr, te, target)))
        else:
            how = 'истории нет'
            for name in ('dur_пред', 'простой_пред'):
                r = r.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
        rows.append(r)
        diag.append({'год_начала': Y, 'операций': te.height, 'обучающих': tr.height, 'способ': how})
    return (pl.concat(rows), pl.DataFrame(diag))
if __name__ == '__main__':
    ev = pl.read_parquet(cfg.data('interim/gtm_events.parquet')).filter(pl.col('выбираемая'))
    allev = pl.read_parquet(cfg.data('interim/gtm_events.parquet'))
    print('=== ПОЧЕМУ ex-ante ПЛАНА В РЕЕСТРЕ НЕТ ===')
    for label, df in (('весь реестр событий', allev), ('выбираемые операции', ev)):
        same, n, pc = plan_is_retro(df)
        print(f'  план.Начало == факт.Начало, {label}: {same:,} из {n:,} ({pc:.1f}%)')
    same, n, pc = plan_is_retro(ev)
    print('поэтому в признаки НЕ входят: ' + ', '.join(EXCLUDED))
    d = features()
    print(f'\n=== ВЫБОРКА: {d.height:,} выбираемых операций, {len(FEATS)} признаков ({len(NUM)} числовых, {len(CAT)} категориальных) ===')
    nw = int(d['mi_src_w'].is_not_null().sum())
    nt = int(d['mi_src_t'].is_not_null().sum())
    print(f'строгая обратность признаков истории: mi(источник) <= mi_beg - 1 у ВСЕХ {nw:,} операций с найденной историей скважины и {nt:,} с историей типа -- это ТАВТОЛОГИЯ ПО ПОСТРОЕНИЮ (backward-join плюс сдвиг ключа на -1), assert страхует от смены стратегии join, а не доказывает отсутствие утечки')
    print('\n=== ЧАСТЬ 1: ОТЛОЖЕННЫЙ ПЕРИОД (обучение до 2021 вкл., проверка 2022-2024) ===')
    res, tr, te = holdout(d)
    print(f"обучающих {res['n_train']:,} (последнее окончание {res['train_end_max']}), проверочных {res['n_test']:,} (первое начало {res['test_beg_min']})")
    hdr = f"{'величина':16s} {'MAE медианы типа':>17s} {'MAE L1':>9s} {'MAE L2':>9s} {'выигрыш L1':>11s} {'выигрыш L2':>11s} {'порог 15%':>12s}"
    print(hdr)
    ok = {}
    for target in TARGETS:
        r = res[target]
        v = r['выигрыш_L1_пц'] >= 15.0
        ok[target] = v
        print(f"{target:16s} {r['база_MAE']:17.2f} {r['L1_MAE']:9.2f} {r['L2_MAE']:9.2f} {r['выигрыш_L1_пц']:10.1f}% {r['выигрыш_L2_пц']:10.1f}% {('ВЫПОЛНЕН' if v else 'НЕ ВЫПОЛНЕН'):>12s}")
    print('выбор L1 ИЗМЕРЕН, а не объявлен: выигрыш L1 против выигрыша L2 напечатан выше')
    print('\nMAE по типам на отложенном периоде (L1 против медианы типа):')
    for target in TARGETS:
        p = fit(tr, target).predict(te.select(FEATS).to_pandas())
        b = type_median(tr, te, target)
        z = te.select('тип', y=pl.col(target)).with_columns(pl.Series('p', np.clip(p, 0, MAX_DUR)), pl.Series('b', b))
        g = z.group_by('тип').agg(n=pl.len(), MAE_база=(pl.col('y') - pl.col('b')).abs().mean().round(2), MAE_L1=(pl.col('y') - pl.col('p')).abs().mean().round(2)).with_columns(выигрыш_пц=(100 * (pl.col('MAE_база') - pl.col('MAE_L1')) / pl.col('MAE_база')).round(1)).sort('n', descending=True)
        print(f'  -- {target} --')
        print(g.to_pandas().to_string(index=False))
    print('\nважность признаков L1 (по числу разбиений), топ-12:')
    for target in TARGETS:
        m = fit(tr, target)
        imp = sorted(zip(FEATS, m.feature_importances_), key=lambda x: -x[1])[:12]
        print(f'  {target}: ' + ', '.join((f'{k}={v}' for k, v in imp)))
    print('\n=== ПРОГНОЗ ВНЕ ОБУЧЕНИЯ ДЛЯ ВСЕХ ОПЕРАЦИЙ (walk-forward) ===')
    pred, diag = walk_forward(d)
    print(diag.to_pandas().to_string(index=False))
    fact = d.select('op_id', 'тип', 'год', 'dur_цель', 'простой_цель')
    j = pred.join(fact, on='op_id', how='inner')
    print(f"\nпрогнозов: {pred.height:,}, из них не-null {int(pred['dur_пред'].is_not_null().sum()):,}; сверка с фактом по типам (медианы, сутки):")
    print(j.group_by('тип').agg(n=pl.len(), dur_факт=pl.col('dur_цель').median().round(1), dur_пред=pl.col('dur_пред').median().round(1), простой_факт=pl.col('простой_цель').median().round(1), простой_пред=pl.col('простой_пред').median().round(1)).sort('n', descending=True).to_pandas().to_string(index=False))
    pred.write_parquet(cfg.data('interim/dur_pred.parquet'), compression='zstd')
    print(f'\n-> data/interim/dur_pred.parquet ({pred.height:,} x {pred.width})')
    print('\nПорог части 1 по каждой величине: ' + ', '.join((f"{k} {('ВЫПОЛНЕН' if v else 'НЕ ВЫПОЛНЕН')}" for k, v in ok.items())))
