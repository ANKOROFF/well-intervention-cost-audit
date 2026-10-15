"""Audit of the dates of the features of the response model."""
import config as cfg
import datetime as dt
import pathlib
import sys
import numpy as np
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
ПРОЧЬ = []

def вывод(s=''):
    ПРОЧЬ.append(s)
    print(s)
ev = pl.read_parquet(cfg.data('interim/gtm_events.parquet')).filter(pl.col('выбираемая'))
dc = pl.read_parquet(cfg.data('interim/decline.parquet')).select('op_id', 'inc_oil_t')
d = ev.join(dc, on='op_id').drop_nulls(['inc_oil_t', 'год']).with_columns(лаг=(pl.col('beg') - pl.col('t_date')).dt.total_days(), mi_b=pl.col('beg').dt.year() * 12 + pl.col('beg').dt.month() - 1)
вывод('# Б11. Даты признаков относительно 01.01.Y (ep30_feature_dates)')
вывод('')
вывод('| Y | n пула | снимок t_* без снимка | снимок t_* в году Y (после 01.01.Y) | лаг начало-снимок, сут (медиана / p90) | снимок позже начала | просадка: месяц в Y | обучающих с окном цели за 01.01.Y |')
вывод('|---|---|---|---|---|---|---|---|')
tot = [0, 0, 0, 0]
for Y in range(2016, 2025):
    p = d.filter(pl.col('год') == Y)
    tr = d.filter(pl.col('год') < Y)
    if p.height == 0:
        continue
    без = int(p['t_date'].is_null().sum())
    в_Y = int((p['t_date'] >= dt.date(Y, 1, 1)).sum())
    лаг = p['лаг'].drop_nulls().to_numpy()
    позже = int((p['лаг'] < 0).sum())
    пр = int((p['mi_b'] - 1 >= Y * 12).sum())
    vn = tr['vnr'].dt.year() * 12 + tr['vnr'].dt.month() - 1 + 24
    за = int((vn > Y * 12).sum())
    вывод(f'| {Y} | {p.height} | {без} | {в_Y} ({100 * в_Y / p.height:.1f} %) | {np.median(лаг):.0f} / {np.percentile(лаг, 90):.0f} | {позже} | {пр} ({100 * пр / p.height:.1f} %) | {за} из {tr.height} ({100 * за / tr.height:.1f} %) |')
    for i, x in enumerate((p.height, в_Y, пр)):
        tot[i] += x
    tot[3] += 0
вывод('')
вывод(f'ЯКОРЬ ep30 fd пул всего = {tot[0]}')
вывод(f'ЯКОРЬ ep30 fd снимок t_* в году Y, доля = {tot[1] / tot[0]:.3f}')
вывод(f'ЯКОРЬ ep30 fd просадка с месяцем в Y, доля = {tot[2] / tot[0]:.3f}')
(ROOT / cfg.results('ep30_feature_dates.md')).write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
