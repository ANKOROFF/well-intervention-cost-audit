"""Economic model of an operation (NPV, scenario parameters)."""
import config as cfg
from dataclasses import dataclass, field, replace
import numpy as np
import pint
import polars as pl
u = pint.UnitRegistry()
u.define('rub = [currency] = ₽')
u.define('tonne_oil = [oil_mass]')
u.define('day_crew = [crew_time]')
M_RUB = 1000000.0
DAYS_MONTH = 30.4
MAX_DUR = 400

@dataclass(frozen=True)
class Params:
    netback_rub_t: float = 1.0
    crew_season_rub: float = 1.0
    discount_annual: float = 0.15
    horizon_months: int = 24
    capex_scale: float = 1.0
    downtime_scale: float = 1.0
    level_source: str = 'мэр'
    preop_source: str = cfg.data('interim/preop_level.parquet')
    capex_source: str = cfg.data('interim/capex_physical.parquet')
    dur_source: str | None = None
    use_pred_dur: bool = False
    use_pred_downtime: bool = False

def check_dimensions(p: Params) -> str:
    q = 1000.0 * u.tonne_oil
    price = p.netback_rub_t * u.rub / u.tonne_oil
    capex = 2000000.0 * u.rub
    q_pre = 14.1 * u.tonne_oil / u.day
    down = 11.0 * u.day
    loss = q_pre * down * price
    assert loss.units == u.rub, f'размерность потери не рубли: {loss.units}'
    npv = q * price - capex - loss
    rate = p.crew_season_rub * u.rub / u.year
    dur = 6.0 * u.day_crew
    assert (dur / (1.0 * u.day_crew)).dimensionless
    assert npv.units == u.rub, f'размерность NPV не рубли: {npv.units}'
    return f'{npv.units:~P}'

def npv_table(p: Params) -> pl.DataFrame:
    prof = pl.read_parquet(cfg.data('interim/decline_profile.parquet')).filter(pl.col('k') <= p.horizon_months)
    ops = pl.read_parquet(cfg.data('interim/decline.parquet')).select('op_id', 'тип', 'год', 'well')
    ev = pl.read_parquet(cfg.data('interim/gtm_events.parquet')).select('op_id', 'dur', 'beg', 'vnr', 't_state')
    LEVELS = {'мэр': 'уровень_мэр_т_сут', 'мэр_раб': 'уровень_мэр_раб_т_сут', 'техрежим': 'уровень_техрежим_т_сут'}
    if p.level_source not in LEVELS:
        raise ValueError(f'level_source должен быть одним из {sorted(LEVELS)}, получено {p.level_source!r}')
    lvl = pl.read_parquet(p.preop_source).select('op_id', уровень_сырой=pl.col(LEVELS[p.level_source]))
    r_m = (1 + p.discount_annual) ** (1 / 12) - 1
    disc = prof.with_columns(dcf=pl.col('прирост_т') * p.netback_rub_t / (1 + r_m) ** pl.col('k')).group_by('op_id').agg(доход_диск=pl.col('dcf').sum(), прирост_т=pl.col('прирост_т').sum(), месяцев=pl.len())
    t = ops.join(disc, on='op_id', how='inner').join(ev, on='op_id', how='left').join(lvl, on='op_id', how='left')
    cx = pl.read_parquet(p.capex_source).select('op_id', capex_phys='capex', мат_измерены='мат_измерены')
    med = t.group_by('тип').agg(dur_med=pl.col('dur').median())
    t = t.join(cx, on='op_id', how='inner').join(med, on='тип', how='left')
    if (p.use_pred_dur or p.use_pred_downtime) and (not p.dur_source):
        raise ValueError('use_pred_dur/use_pred_downtime требуют dur_source')
    if p.dur_source:
        pr = pl.read_parquet(p.dur_source).select('op_id', 'dur_пред', 'простой_пред').drop_nulls(['dur_пред', 'простой_пред'])
        t = t.join(pr, on='op_id', how='inner')
    dur_expr = pl.col('dur_пред') if p.use_pred_dur else pl.coalesce(pl.col('dur'), pl.col('dur_med'))
    down_expr = pl.col('простой_пред') if p.use_pred_downtime else pl.coalesce((pl.col('vnr') - pl.col('beg')).dt.total_days(), pl.col('dur'), pl.col('dur_med'))
    t = t.with_columns(dur_use=dur_expr.clip(0, MAX_DUR), capex=pl.col('capex_phys') * p.capex_scale, простой_сут=down_expr.clip(0, MAX_DUR), уровень_до_т_сут=pl.col('уровень_сырой').fill_null(0.0).clip(0, None))
    return t.with_columns(бригадо_сутки=pl.col('dur_use'), потеря_диск=pl.col('уровень_до_т_сут') * pl.col('простой_сут') * p.netback_rub_t * p.downtime_scale / (1 + r_m) ** (pl.col('простой_сут') / (2 * DAYS_MONTH))).with_columns(npv=pl.col('доход_диск') - pl.col('capex') - pl.col('потеря_диск')).with_columns(npv_на_бригадо_сутки=pl.col('npv') / pl.col('бригадо_сутки').clip(1, None)).drop_nulls('npv')

def selftest():
    p = Params()
    print('размерность NPV:', check_dimensions(p))
    base = npv_table(p)['npv'].sum()
    checks = [('NPV растёт с ценой', npv_table(replace(p, netback_rub_t=p.netback_rub_t * 2))['npv'].sum() > base), ('NPV падает с CAPEX', npv_table(replace(p, capex_scale=2.0))['npv'].sum() < base), ('NPV не зависит от ставки бригады (она -- ограничение, не затрата)', npv_table(replace(p, crew_season_rub=p.crew_season_rub * 2))['npv'].sum() == base), ('NPV падает со ставкой дисконта', npv_table(replace(p, discount_annual=0.3))['npv'].sum() < base), ('NPV растёт с горизонтом', npv_table(replace(p, horizon_months=12))['npv'].sum() < base), ('NPV не возрастает при росте простоя [ТАВТОЛОГИЯ по построению]', npv_table(replace(p, downtime_scale=2.0))['npv'].sum() < base)]
    ok = True
    for name, res in checks:
        print(f"  [{('OK ' if res else 'ПРОВАЛ')}] {name}")
        ok &= bool(res)
    return ok
if __name__ == '__main__':
    print('=== ПРОВЕРКА МОНОТОННОСТЕЙ (инварианты ребра гиперграфа) ===')
    assert selftest(), 'инварианты нарушены'
    p = Params()
    t = npv_table(p)
    print(f'\n=== БАЗОВЫЙ СЦЕНАРИЙ: netback {p.netback_rub_t:,.0f} ₽/т, дисконт {p.discount_annual:.0%}, горизонт {p.horizon_months} мес ===')
    print(f'операций с NPV: {t.height:,}')
    print(t.group_by('тип').agg(n=pl.len(), доп_т=pl.col('прирост_т').median().round(0), доход_млн=(pl.col('доход_диск') / M_RUB).median().round(1), capex_млн=(pl.col('capex') / M_RUB).median().round(1), простой_сут=pl.col('простой_сут').median().round(0), потеря_млн=(pl.col('потеря_диск') / M_RUB).median().round(2), бр_сутки=pl.col('бригадо_сутки').median().round(0), npv_на_бр_сутки_млн=(pl.col('npv_на_бригадо_сутки') / M_RUB).median().round(2), npv_млн=(pl.col('npv') / M_RUB).median().round(1), доля_npv_gt0=(pl.col('npv') > 0).mean().round(3)).sort('n', descending=True).to_pandas().to_string(index=False))
    print(f"\nвсего NPV портфеля: {t['npv'].sum() / 1000000000.0:,.1f} млрд ₽ | доля операций с NPV > 0: {(t['npv'] > 0).mean():.3f}")
    print(f'\n=== ЧУВСТВИТЕЛЬНОСТЬ: доля операций с NPV > 0 ===')
    print(f"{'netback ₽/т':>12s} | " + ' | '.join((f'  capex x{s:<4.1f}' for s in (0.5, 1.0, 2.0))))
    for nb in (4000, 8000, 16000):
        cells = []
        for s in (0.5, 1.0, 2.0):
            tt = npv_table(replace(p, netback_rub_t=nb, capex_scale=s))
            cells.append(f"{(tt['npv'] > 0).mean():12.3f}")
        print(f'{nb:>12,} | ' + ' | '.join(cells))
    print(f'\n=== ЧУВСТВИТЕЛЬНОСТЬ ПОРЯДКА ТИПОВ (медианный NPV, млн ₽) ===')
    print('порядок по NPV на бригадо-сутки -- именно он определяет выбор при дефиците бригад;\nесли он не меняется по сетке, решения устойчивы')
    for nb in (4000, 8000, 16000):
        for cs in (0.5, 1.0, 2.0):
            tt = npv_table(replace(p, netback_rub_t=nb, capex_scale=cs))
            r = tt.group_by('тип').agg(m=pl.col('npv_на_бригадо_сутки').median(), n=pl.len()).filter(pl.col('n') >= 20).sort('m', descending=True)
            print(f'  цена {nb:>6,}, capex x{cs:<4.1f}: ' + ' > '.join(r['тип'].to_list()))
