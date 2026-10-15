"""Resource model: rig and crew occupancy from dated events."""
import config as cfg
import polars as pl
from wells import norm_well, base_well
M = 1000000.0
ZBS_WIN = (15, 365)
FRAC_WIN = (15, 45)
DATES_KONSTR = ['Дата спуска', 'Дата цементирования', 'Дата установки секции', 'Дата опрессовки']
GNKT_PHASES = ['переезд_ч', 'куст_ч', 'работа_ч']
ФИ_ОЖИДАЕМАЯ = 0.266

def productive_fraction() -> tuple[float, int]:
    g = pl.read_parquet(cfg.data('parquet/gnkt_ops.parquet')).filter(pl.col('работа_ч').is_between(1, 24 * 60))
    tot = sum((g[c].sum() for c in GNKT_PHASES))
    return (float(g['чистое_ч'].sum() / tot), g.height)

def _konstr_events() -> pl.DataFrame:
    kk = pl.read_parquet(cfg.data('parquet/konstrukciya.parquet'))
    k = kk.select(well=norm_well(pl.col('SRC_WELL_NAME')), тип_кол=pl.col('Тип колонны').cast(pl.Utf8).str.strip_chars(), **{f'd{i}': pl.col(c).str.to_date('%d.%m.%y', strict=False) for i, c in enumerate(DATES_KONSTR)}).filter(pl.col('тип_кол') == 'хвостов').with_columns(base=base_well(pl.col('well')))
    long = pl.concat([k.select('base', дата=pl.col(f'd{i}')) for i in range(len(DATES_KONSTR))]).drop_nulls('дата')
    return long.filter(pl.col('дата').is_between(pl.date(1990, 1, 1), pl.date(2026, 1, 1)))

def _frac_events() -> pl.DataFrame:
    return pl.read_parquet(cfg.data('parquet/frac_stages.parquet')).select('well', 'дата').drop_nulls()

def _last_event(ops: pl.DataFrame, ev: pl.DataFrame, on: str, before: int, after_end: bool, after: int) -> pl.DataFrame:
    край = pl.coalesce(pl.col('end'), pl.col('beg')) if after_end else pl.col('beg')
    j = ops.join(ev, on=on, how='inner').filter((pl.col('дата') >= pl.col('beg') - pl.duration(days=before)) & (pl.col('дата') <= край + pl.duration(days=after)))
    return j.group_by('op_id').agg(посл=pl.col('дата').max(), дней=pl.col('дата').n_unique())

def occupancy(ev: pl.DataFrame | None=None) -> pl.DataFrame:
    if ev is None:
        ev = pl.read_parquet(cfg.data('interim/gtm_events.parquet')).filter(pl.col('выбираемая'))
    ops = ev.select('op_id', 'тип', 'well', 'beg', 'end', 'dur')
    с_датой = ops.drop_nulls(['well', 'beg'])
    zbs = с_датой.filter(pl.col('тип') == 'ЗБС').with_columns(base=base_well(pl.col('well')))
    станок = _last_event(zbs, _konstr_events(), 'base', ZBS_WIN[0], False, ZBS_WIN[1]).rename({'посл': 'посл_станок', 'дней': 'дней_станок'})
    флот = _last_event(с_датой, _frac_events(), 'well', FRAC_WIN[0], True, FRAC_WIN[1]).rename({'посл': 'посл_флот', 'дней': 'дней_флот'})
    фи, n_гнкт = productive_fraction()
    окно = lambda c: (pl.col(c) - pl.col('beg')).dt.total_days() + 1
    t = ops.join(станок, on='op_id', how='left').join(флот, on='op_id', how='left').with_columns(T_станок=окно('посл_станок'), T_флот=окно('посл_флот')).with_columns(T_техники=pl.max_horizontal('T_станок', 'T_флот'))
    d = pl.col('dur').fill_null(0).clip(0, 400)
    t = t.with_columns(T_техники=pl.col('T_техники').clip(1, None)).with_columns(восстановлено=pl.col('T_техники').is_not_null(), база_сут=pl.min_horizontal(pl.col('T_техники').fill_null(1000000000.0), d)).with_columns(занятость_крс_сут=pl.col('база_сут') * фи, фи=pl.lit(фи), n_гнкт=pl.lit(n_гнкт))
    return t.select('op_id', 'тип', 'dur', 'T_станок', 'T_флот', 'T_техники', 'дней_станок', 'дней_флот', 'восстановлено', 'база_сут', 'занятость_крс_сут', 'фи', 'n_гнкт')

def _order(t: pl.DataFrame, вар: str) -> list[str]:
    r = t.group_by('тип').agg(m=pl.col(вар).median(), n=pl.len()).filter(pl.col('n') >= 20).sort('m', descending=True)
    return r['тип'].to_list()

def main() -> None:
    фи, n_гнкт = productive_fraction()
    print('=' * 78)
    print('1. КОЭФФИЦИЕНТ КАЛЕНДАРЬ -> ЗАНЯТОСТЬ (измерен, не назначен)')
    print('=' * 78)
    print(f'  чистое время / (переезд + на кусту + работа) = {100 * фи:.4f}%  по {n_гнкт:,} постановкам флота ГНКТ')
    assert abs(фи - ФИ_ОЖИДАЕМАЯ) < 0.001, f'расчётная доля {фи:.5f} расходится с опубликованными {ФИ_ОЖИДАЕМАЯ}: источник изменился, число надо переизмерить'
    t = occupancy()
    print('\n' + '=' * 78)
    print('2. ПОКРЫТИЕ ДАТИРОВАННЫМИ СОБЫТИЯМИ ПО ТИПАМ ТЕХНИКИ')
    print('=' * 78)
    print(t.group_by('тип').agg(n=pl.len(), станок=pl.col('T_станок').is_not_null().sum(), флот=pl.col('T_флот').is_not_null().sum(), восст=pl.col('восстановлено').sum(), доля_восст=pl.col('восстановлено').mean().round(3), dur_мед=pl.col('dur').median(), T_мед=pl.col('T_техники').median(), занятость_мед=pl.col('занятость_крс_сут').median().round(1)).sort('n', descending=True).to_pandas().to_string(index=False))
    for тип, порог in (('ЗБС', 0.6), ('ГРП', 0.6)):
        s = t.filter(pl.col('тип') == тип)
        доля = float(s['восстановлено'].mean())
        print(f"  критерий 1, {тип}: восстановлено {s['восстановлено'].sum()} из {s.height} = {доля:.3f} при пороге {порог:.2f} -> {('ВЫПОЛНЕН' if доля >= порог else 'НЕ ВЫПОЛНЕН')}")
    print('\n' + '=' * 78)
    print('3. ТЕХРЕЖИМ КАК ИСТОЧНИК: ОТРИЦАТЕЛЬНЫЙ РЕЗУЛЬТАТ')
    print('=' * 78)
    tr = pl.read_parquet(cfg.data('parquet/source_regime_all_prb.parquet'))
    колонка = 'Дата изм. параметров насоса'
    заполнена = float(tr[колонка].is_not_null().mean())
    pm = tr.select(well=norm_well(pl.col('Номер скважины')), дата=pl.col(колонка).str.to_date('%d.%m.%Y', strict=False)).drop_nulls('дата').unique()
    ev = pl.read_parquet(cfg.data('interim/gtm_events.parquet')).filter(pl.col('выбираемая')).select('op_id', 'тип', 'well', 'beg', 'end', 'dur')
    a = _last_event(ev.drop_nulls(['well', 'beg']), pm, 'well', FRAC_WIN[0], True, FRAC_WIN[1])
    a = ev.join(a, on='op_id', how='inner').with_columns(T=(pl.col('посл') - pl.col('beg')).dt.total_days() + 1)
    print(f'  «{колонка}» заполнена {100 * заполнена:.1f}%, уникальных пар (скважина, дата): {pm.height:,}')
    print(f"  отметка попадает в окно операции у {a.height:,} операций из {ev.drop_nulls(['well', 'beg']).height:,}")
    print(a.group_by('тип').agg(n=pl.len(), T_мед=pl.col('T').median(), dur_мед=pl.col('dur').median()).sort('n', descending=True).to_pandas().to_string(index=False))
    хуже = a.group_by('тип').agg(n=pl.len(), T=pl.col('T').median(), d=pl.col('dur').median()).filter(pl.col('T') > pl.col('d'))
    print(f'  типов, где «окно» по этой отметке БОЛЬШЕ календарной длительности: {хуже.height} -- точечная отметка занятость не ограничивает, для длительности источник отвергнут')
    print('\n' + '=' * 78)
    print('4. ПОРЯДОК ТИПОВ ПО NPV НА БРИГАДО-СУТКИ: ЧТО ЕГО МЕНЯЕТ')
    print('=' * 78)
    from cost_physical import Prices
    from npv import Params, npv_table
    ставка = Prices().крс_руб_сутки
    p = Params()
    c = pl.read_parquet(cfg.data('interim/capex_physical.parquet')).select('op_id', 'c_крс')
    q = npv_table(p).join(c, on='op_id', how='inner').join(t.select('op_id', 'T_техники', 'база_сут', 'восстановлено'), on='op_id', how='left').with_columns(capex_без_крс=pl.col('capex') - pl.col('c_крс'), бс=pl.col('бригадо_сутки').clip(1, None))
    d = pl.col('бригадо_сутки')
    вар = {'dur (как сейчас)': d, f'dur x {фи:.4f} (только коэффициент)': d * фи, 'T (только дифф. длительности)': pl.coalesce(pl.col('база_сут'), d), f'T x {фи:.4f} (полная модель Ш9)': pl.coalesce(pl.col('база_сут'), d) * фи}
    q = q.with_columns(**{f'v{i}': (pl.col('доход_диск') - pl.col('capex_без_крс') - e * ставка) / pl.col('бс') / M for i, e in enumerate(вар.values())})
    имена = list(вар)
    базовый = _order(q, 'v0')
    print(f'  {имена[0]:42s}: ' + ' > '.join(базовый))
    for i, имя in enumerate(имена[1:], 1):
        o = _order(q, f'v{i}')
        print(f'  {имя:42s}: ' + ' > '.join(o))
        print(f"  {'':42s}  порядок {('НЕ изменился' if o == базовый else 'ИЗМЕНИЛСЯ')}")
    print('\n  сдвиг медианного NPV на бригадо-сутки, млн ₽:')
    сд = q.group_by('тип').agg(n=pl.len(), **{f'{i}': pl.col(f'v{i}').median() for i in range(len(вар))}).filter(pl.col('n') >= 20)
    сд = сд.with_columns(**{f'сдвиг_{i}': (pl.col(f'{i}') - pl.col('0')).round(4) for i in (1, 2, 3)}).sort('0', descending=True)
    print(сд.select('тип', 'n', база=pl.col('0').round(3), сдвиг_коэф=pl.col('сдвиг_1'), сдвиг_T=pl.col('сдвиг_2'), сдвиг_полн=pl.col('сдвиг_3')).to_pandas().to_string(index=False))
    for i, имя in ((1, 'коэффициент'), (2, 'дифф. длительности'), (3, 'полная модель')):
        u = сд[f'сдвиг_{i}'].n_unique()
        print(f"  уникальных сдвигов, {имя}: {u} -> {('порядок изменить НЕ может' if u == 1 else 'порядок может измениться')}")
    print('\n' + '=' * 78)
    print('5. c_крс ЗБС: БЫЛО И СТАЛО (критерий 2)')
    print('=' * 78)
    z = t.filter(pl.col('тип') == 'ЗБС')
    from cost_physical import Prices
    ставка = Prices().крс_руб_сутки
    было = float((z['dur'].fill_null(0).clip(0, 400) * ставка).median())
    стало = float((z['занятость_крс_сут'] * ставка).median())
    print(f"  dur медиана {z['dur'].median():.0f} сут -> c_крс {было / M:.1f} млн ₽")
    print(f"  занятость медиана {z['занятость_крс_сут'].median():.1f} сут -> c_крс {стало / M:.1f} млн ₽")
    print(f"  снижение {100 * (1 - стало / было):.1f}% при пороге 50% -> {('ВЫПОЛНЕН' if стало <= 0.5 * было else 'НЕ ВЫПОЛНЕН')}")
if __name__ == '__main__':
    main()
