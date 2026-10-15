"""Validation of the pump parser against labels; physical control of proppant mass."""
import config as cfg
import csv
import sys
import numpy as np
import polars as pl
sys.path.insert(0, 'src')
from wells import norm_well
import price_registry as pr
ПУЛ = cfg.data('interim/capex_physical_pp.parquet')
РЕЖИМ = cfg.data('parquet/tr_regim_monthly.parquet')
ВЫХОД = cfg.out('parser_validation.tsv')
ДОПУСК = 0.05
МИН_N = 10
СТРОКИ = []

def fmt(x):
    return '' if x is None or (isinstance(x, float) and (not np.isfinite(x))) else f'{x:.4g}'

def строка(пакет, метрика, группа, n, значение, примечание=''):
    if n < МИН_N:
        значение, примечание = ('', 'n<10, подавлено')
    СТРОКИ.append([пакет, метрика, группа, n, fmt(значение) if значение != '' else '', примечание])
    print(f"[{пакет}] {метрика:34s} {группа:26s} n={n:<6d} {(fmt(значение) if значение != '' else '-'):>9s} {примечание}")

def в_идекс(col):
    return col.str.slice(0, 4).cast(pl.Int32) * 12 + col.str.slice(5, 2).cast(pl.Int32)

def совп(a, b):
    return ((a - b).abs() <= ДОПУСК * b.abs()) & (b.abs() > 0)
пул = pl.read_parquet(ПУЛ).with_columns(norm_well(pl.col('well')).alias('w'))
реж = pl.read_parquet(РЕЖИМ, columns=['period', 'Скважина', 'Месторождение | Название', 'Тип насоса', 'Номинальная производительность | м3/сут', 'Параметры работы ЭЦН | Номинальный напор | м', 'Перф. | м', 'Мощность | Нефтенас. | м']).filter(pl.col('Месторождение | Название').str.starts_with('FIELD')).rename({'Номинальная производительность | м3/сут': 'Q', 'Параметры работы ЭЦН | Номинальный напор | м': 'H', 'Перф. | м': 'перф', 'Мощность | Нефтенас. | м': 'нн'}).with_columns(norm_well(pl.col('Скважина')).alias('w'), в_идекс(pl.col('period')).alias('m'))
c1 = пул.filter((pl.col('beg') >= pl.date(2023, 12, 1)) & pl.col('произв_м3сут').is_not_null() & pl.col('напор_м').is_not_null())
c1 = c1.with_columns((pl.col('end').dt.year() * 12 + pl.col('end').dt.month()).alias('me'), (pl.col('beg').dt.year() * 12 + pl.col('beg').dt.month()).alias('mb'), pl.int_range(pl.len()).alias('i'))
n_all = c1.height
строка('C1', 'операций в выборке', 'все', n_all, float(n_all))
for t, k in c1.group_by('тип').agg(pl.len()).sort('тип').iter_rows():
    строка('C1', 'доля типа в выборке', t, k, k / n_all)
р = реж.filter(pl.col('Q').is_not_null() & pl.col('H').is_not_null() & (pl.col('Q') > 0))
рг = р.group_by(['w', 'm']).agg(pl.col('Q').first(), pl.col('H').first(), pl.col('Q').n_unique().alias('nq'), pl.col('H').n_unique().alias('nh'))
пары_в_режиме = реж.select(['w', 'm']).unique()

def прицепить(k, смещение_от, имя):
    ключ = pl.col('me') + k if смещение_от == 'end' else pl.col('mb') + k
    x = c1.select('i', 'w', ключ.alias('m')).join(рг, on=['w', 'm'], how='left')
    return x.select('i', pl.col('Q').alias(f'Q_{имя}'), pl.col('H').alias(f'H_{имя}'), pl.col('nq').alias(f'nq_{имя}'), pl.col('nh').alias(f'nh_{имя}'))
c1 = c1.join(прицепить(1, 'end', 'p'), on='i')
for k, имя in [(2, 'p2'), (3, 'p3'), (0, 'p0')]:
    c1 = c1.join(прицепить(k, 'end', имя), on='i')
c1 = c1.join(прицепить(-1, 'beg', 'pre'), on='i')
скв_в_режиме = set(реж['w'].unique().to_list())
c1 = c1.with_columns(pl.col('w').is_in(list(скв_в_режиме)).alias('скв_есть'), (pl.col('me') + 1).alias('m1'))
мес_режима = реж.select('m').unique()
макс_м = int(реж['m'].max())
c1 = c1.with_columns((pl.col('m1') <= макс_м).alias('в_окне'))
найд = c1.filter(pl.col('Q_p').is_not_null())
строка('C1', 'доля найденных пар (с Q,H режима)', 'все', n_all, найд.height / n_all)
строка('C1', 'пар найдено', 'все', найд.height, float(найд.height))
строка('C1', 'скважина есть в режиме вообще', 'все', n_all, c1['скв_есть'].mean())
строка('C1', 'месяц после конца в окне режима', 'все', n_all, c1['в_окне'].mean())
строка('C1', 'найдено среди операций в окне', 'в окне', int(c1['в_окне'].sum()), найд.height / max(1, int(c1['в_окне'].sum())))
строка('C1', 'найдено / скважина есть и в окне', 'скв+окно', int((c1['скв_есть'] & c1['в_окне']).sum()), найд.height / max(1, int((c1['скв_есть'] & c1['в_окне']).sum())))
найд = найд.with_columns(совп(pl.col('произв_м3сут'), pl.col('Q_p')).alias('мQ'), совп(pl.col('напор_м'), pl.col('H_p')).alias('мH'), совп(pl.col('произв_м3сут') * pl.col('напор_м'), pl.col('Q_p') * pl.col('H_p')).alias('мП'))
n = найд.height
строка('C1', 'совпадение подача x напор (головное)', 'найденные', n, найд['мП'].mean())
строка('C1', 'совпадение подачи', 'найденные', n, найд['мQ'].mean())
строка('C1', 'совпадение напора', 'найденные', n, найд['мH'].mean())
строка('C1', 'совпали обе величины', 'найденные', n, (найд['мQ'] & найд['мH']).mean())
строка('C1', 'нижняя граница на всех 697', 'все', n_all, найд['мП'].sum() / n_all)
строка('C1', 'пары с неоднозначным режимом (>1 значения Q/H)', 'найденные', n, ((найд['nq_p'] > 1) | (найд['nh_p'] > 1)).mean())
for t in sorted(найд['тип'].unique().to_list()):
    g = найд.filter(pl.col('тип') == t)
    строка('C1', 'совпадение произведения по типу', t, g.height, g['мП'].mean() if g.height else None)
    строка('C1', 'найдено по типу (от выборки типа)', t, int((c1['тип'] == t).sum()), g.height / max(1, int((c1['тип'] == t).sum())))
for нз, имя in [('насос_случай', 'случай')]:
    for t in sorted(найд[нз].unique().to_list()):
        g = найд.filter(pl.col(нз) == t)
        строка('C1', f'совпадение произведения по {имя}', t, g.height, g['мП'].mean() if g.height else None)
мп2 = совп(найд['произв_м3сут'] * найд['напор_м'], найд['Q_p2'] * найд['H_p2']).fill_null(False)
мп3 = совп(найд['произв_м3сут'] * найд['напор_м'], найд['Q_p3'] * найд['H_p3']).fill_null(False)
строка('C1', 'ПОСТФАКТУМ совпадение в одном из +1..+3', 'найденные', n, (найд['мП'] | мп2 | мп3).mean(), 'не по правилу, диагностика лага режима')
нм = найд.filter(~pl.col('мП'))
строка('C1', 'несовпавших по произведению', 'найденные', n, нм.height / n if n else None)
if нм.height:
    rq = нм['произв_м3сут'] / нм['Q_p']
    rh = нм['напор_м'] / нм['H_p']
    шаблоны = {'подача совпала, напор нет': нм['мQ'] & ~нм['мH'], 'напор совпал, подача нет': ~нм['мQ'] & нм['мH'], 'обе различны, подача пула больше': ~нм['мQ'] & ~нм['мH'] & (rq > 1), 'обе различны, подача пула меньше': ~нм['мQ'] & ~нм['мH'] & (rq < 1), 'произведение совпало, по отдельности нет': нм['мП']}
    for имя, маска in шаблоны.items():
        строка('C1', f'шаблон несовпадения: {имя}', 'несовпавшие', нм.height, float(маска.sum()) / нм.height)
    pre = нм.filter(pl.col('Q_pre').is_not_null())
    if pre.height:
        mp = совп(pre['произв_м3сут'] * pre['напор_м'], pre['Q_pre'] * pre['H_pre'])
        строка('C1', 'несовпавшие: пул совпал с режимом ДО операции', 'несовпавшие с до-режимом', pre.height, float(mp.mean()))
    for имя, k in [('p0', 'месяц конца'), ('p2', '+2 месяца'), ('p3', '+3 месяца')]:
        x = нм.filter(pl.col(f'Q_{имя}').is_not_null())
        if x.height:
            mp = совп(x['произв_м3сут'] * x['напор_м'], x[f'Q_{имя}'] * x[f'H_{имя}'])
            строка('C1', f'несовпавшие: пул совпал с режимом {k}', 'несовпавшие', x.height, float(mp.mean()))
    строка('C1', 'несовпавшие: медиана |Q_пул/Q_режим-1|', 'несовпавшие', нм.height, float((rq - 1).abs().median()))
    строка('C1', 'несовпавшие: медиана |H_пул/H_режим-1|', 'несовпавшие', нм.height, float((rh - 1).abs().median()))
    строка('C1', 'несовпавшие: подача пула = подаче режима (округл.)', 'несовпавшие', нм.height, float((нм['произв_м3сут'].round(0) == нм['Q_p'].round(0)).mean()))
пул2 = pl.read_parquet(ПУЛ).with_columns(norm_well(pl.col('well')).alias('w'))
h = реж.group_by('w').agg(pl.col('перф').median().alias('h_перф'), pl.col('нн').median().alias('h_нн'))
o = пул2.filter((pl.col('проппант_т') > 0) & (pl.col('Xf_м') > 0)).join(h, on='w', how='left')
строка('C2', 'операций с проппантом и Xf', 'все', o.height, float(o.height))
for t in sorted(o['тип'].unique().to_list()):
    строка('C2', 'операций с проппантом и Xf по типу', t, int((o['тип'] == t).sum()), float((o['тип'] == t).sum()))
φ = 0.35

def ширина_мм(M_т, Xf, h, ρ):
    return M_т * 1000.0 / (2.0 * Xf * h * ρ * (1.0 - φ)) * 1000.0
for источник, имя_h in [('h_перф', 'Перф. | м (головное)'), ('h_нн', 'Нефтенас. мощность (справочно)')]:
    z = o.filter(pl.col(источник).is_not_null() & (pl.col(источник) > 0))
    строка('C2', f'операций с мощностью: {имя_h}', 'все', z.height, z.height / max(1, o.height))
    if z.height < МИН_N:
        continue
    ρs = {1500: None, 1700: None, 1900: None}
    for ρ in ρs:
        w = ширина_мм(z['проппант_т'], z['Xf_м'], z[источник], ρ)
        ρs[ρ] = w
        строка('C2', f'доля w в [2,20] мм, rho={ρ}', имя_h, z.height, float(((w >= 2) & (w <= 20)).mean()))
    w1500, w1900 = (ρs[1500], ρs[1900])
    строка('C2', 'доля w в [2,20] при ВСЕХ rho 1500-1900', имя_h, z.height, float(((w1900 >= 2) & (w1500 <= 20)).mean()))
    строка('C2', 'доля w в [2,20] хотя бы при одном rho', имя_h, z.height, float(((w1500 >= 2) & (w1900 <= 20) | (w1500 >= 2) & (w1500 <= 20) | (w1900 >= 2) & (w1900 <= 20)).mean()))
    w = ρs[1700]
    строка('C2', 'w мм: квартиль 25', имя_h, z.height, float(w.quantile(0.25)))
    строка('C2', 'w мм: медиана', имя_h, z.height, float(w.median()))
    строка('C2', 'w мм: квартиль 75', имя_h, z.height, float(w.quantile(0.75)))
    строка('C2', 'доля w < 2 мм (rho=1700)', имя_h, z.height, float((w < 2).mean()))
    строка('C2', 'доля w > 20 мм (rho=1700)', имя_h, z.height, float((w > 20).mean()))
    for t in sorted(z['тип'].unique().to_list()):
        g = z.filter(pl.col('тип') == t)
        wt = ширина_мм(g['проппант_т'], g['Xf_м'], g[источник], 1700)
        строка('C2', 'доля w в [2,20] по типу, rho=1700', f'{t} / {имя_h[:8]}', g.height, float(((wt >= 2) & (wt <= 20)).mean()) if g.height else None)
    s = z.filter(pl.col('стадий').is_not_null() & (pl.col('стадий') >= 1))
    if s.height >= МИН_N:
        wm = ширина_мм(s['проппант_т'] / s['стадий'], s['Xf_м'], s[источник], 1700)
        строка('C2', 'доля w в [2,20], масса на стадию, rho=1700', имя_h, s.height, float(((wm >= 2) & (wm <= 20)).mean()))
        строка('C2', 'доля операций со стадий>1', имя_h, s.height, float((s['стадий'] > 1).mean()))
pr.ШАПКИ.update({pr.норм(k): v for k, v in pr.ШАПКИ_ТЕКСТ.items()})
А = pr.собрать_А()
z = А.filter((pl.col('mass') > 0) & (pl.col('mass') <= 2000) & (pl.col('vol') > 0) & pl.col('y').is_between(2018, 2024)).select('soc', 'y', 'mass', 'vol', 'ngrp')
строка('C3', 'заданий ГРП после дедупликации и фильтра', 'все', z.height, float(z.height))
строка('C3', 'отброшено массой > 2000 т', 'все', int(А.filter(pl.col('mass') > 2000).height), float(А.filter(pl.col('mass') > 2000).height))
y = z['y'].to_numpy()
M = z['mass'].to_numpy()
V = z['vol'].to_numpy()
ng = z['ngrp'].fill_null(0).to_numpy()

def mdape(a, p):
    return float(np.median(np.abs(p - a) / a))

def cv(X):
    res_r, res_m, n_fold = ([], [], [])
    for yy in sorted(set(y.tolist())):
        te, tr = (y == yy, y != yy)
        if te.sum() < МИН_N:
            continue
        A_ = np.column_stack([np.ones(tr.sum()), X[tr]])
        b, *_ = np.linalg.lstsq(A_, np.log(M[tr]), rcond=None)
        p = np.exp(np.column_stack([np.ones(te.sum()), X[te]]) @ b)
        res_r.append(np.abs(p - M[te]) / M[te])
        res_m.append(np.abs(np.median(M[tr]) - M[te]) / M[te])
        n_fold.append((yy, int(te.sum()), float(np.median(res_r[-1])), float(np.median(res_m[-1]))))
    return (np.concatenate(res_r), np.concatenate(res_m), n_fold)
Xv = np.log(V).reshape(-1, 1)
rr, rm, folds = cv(Xv)
строка('C3', 'MdAPE медианы (оставить год)', 'все', len(rr), float(np.median(rm)))
строка('C3', 'MdAPE регрессии лог-лог (оставить год)', 'масса~объём', len(rr), float(np.median(rr)))
строка('C3', 'выигрыш регрессии, п.п. (головное)', 'масса~объём', len(rr), 100 * float(np.median(rm) - np.median(rr)))
for yy, k, a, b in folds:
    строка('C3', 'MdAPE регрессии / медианы по году', f'{yy}: регр {fmt(a)}', k, b, f'медиана={fmt(b)}')
m2 = ng > 0
if m2.sum() >= МИН_N:
    y0, M0, V0, ng0 = (y, M, V, ng)
    y, M, V, ng = (y[m2], M[m2], V[m2], ng[m2])
    rr2, rm2, _ = cv(np.column_stack([np.log(V), np.log(ng)]))
    rr1, _, _ = cv(np.log(V).reshape(-1, 1))
    строка('C3', 'MdAPE медианы (подвыборка ngrp>0)', 'подвыборка', len(rm2), float(np.median(rm2)))
    строка('C3', 'MdAPE масса~объём (подвыборка ngrp>0)', 'подвыборка', len(rr1), float(np.median(rr1)))
    строка('C3', 'MdAPE масса~объём+число ГРП', 'подвыборка', len(rr2), float(np.median(rr2)))
    строка('C3', 'выигрыш объём+число ГРП, п.п.', 'подвыборка', len(rr2), 100 * float(np.median(rm2) - np.median(rr2)))
    y, M, V, ng = (y0, M0, V0, ng0)
soc = z['soc'].to_numpy()
rs_r, rs_m = ([], [])
for c in set(soc.tolist()):
    te, tr = (soc == c, soc != c)
    if te.sum() < МИН_N:
        continue
    A_ = np.column_stack([np.ones(tr.sum()), Xv[tr]])
    bb, *_ = np.linalg.lstsq(A_, np.log(M[tr]), rcond=None)
    rs_r.append(np.abs(np.exp(np.column_stack([np.ones(te.sum()), Xv[te]]) @ bb) - M[te]) / M[te])
    rs_m.append(np.abs(np.median(M[tr]) - M[te]) / M[te])
rs_r, rs_m = (np.concatenate(rs_r), np.concatenate(rs_m))
строка('C3', 'MdAPE медианы (оставить общество)', 'все', len(rs_r), float(np.median(rs_m)))
строка('C3', 'MdAPE регрессии (оставить общество)', 'масса~объём', len(rs_r), float(np.median(rs_r)))
строка('C3', 'выигрыш (оставить общество), п.п.', 'масса~объём', len(rs_r), 100 * float(np.median(rs_m) - np.median(rs_r)))
b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(M)), np.log(V)]), np.log(M), rcond=None)
строка('C3', 'наклон лог-лог (все данные)', 'масса~объём', len(M), float(b[1]))
r2 = 1 - np.var(np.log(M) - (b[0] + b[1] * np.log(V))) / np.var(np.log(M))
строка('C3', 'R2 лог-лог (все данные)', 'масса~объём', len(M), float(r2))
with open(ВЫХОД, 'w', newline='', encoding='utf-8') as f:
    w_ = csv.writer(f, delimiter='\t')
    w_.writerow(['package', 'metric', 'group', 'n', 'value', 'note'])
    w_.writerows(СТРОКИ)
print(f'строк {len(СТРОКИ)} -> {ВЫХОД}')
