"""Physical cost model of an operation: quantities x unit prices; pump-designation parser (pump_nomenclature)."""
import config as cfg
from dataclasses import dataclass
import polars as pl
from crew_model import ZBS_WIN, occupancy
from wells import CYR2LAT, base_well
M = 1000000.0
WIN_BEFORE, WIN_AFTER = (15, 45)
FRAC_TYPES = ['ГРП', 'ЗБС', 'ВПН', 'ПиП']
MAT = ['c_проппант', 'c_гель', 'c_кислота', 'c_азот', 'c_бурение', 'c_цемент', 'c_насос']
TIME = ['c_крс', 'c_флот_грп', 'c_станок', 'c_моб']

def norm_well(col):
    c = col.cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    for a, b in CYR2LAT:
        c = c.str.replace_all(a, b, literal=True)
    return c.str.replace_all('\\s+', '')

@dataclass(frozen=True)
class Prices:
    проппант_руб_т: float = 1.0
    проппант_rcp_руб_т: float = 1.0
    гель_руб_м3: float = 1.0
    кислота_руб_м3: float = 1.0
    азот_руб_т: float = 1.0
    цемент_руб_м3: float = 1.0
    насос_руб_м3сут_м: float = 1.0
    бурение_руб_м: float = 1.0
    мех_скорость_м_сутки: float = 40.0
    крс_руб_сутки: float = 1.0
    флот_грп_руб_сутки: float = 1.0
    станок_збс_руб_сутки: float = 1.0
    мобилизация_руб: float = 1.0
    scale: float = 1.0

def pump_nomenclature() -> pl.DataFrame:
    g = pl.read_parquet(cfg.data('parquet/gtm_registry.parquet'))
    raw = lambda c: pl.col(c).cast(pl.Utf8).fill_null('').str.strip_chars().str.to_uppercase()

    def canon(col):
        x = raw(col)
        for a, b in CYR2LAT:
            x = x.str.replace_all(a, b, literal=True)
        return x.str.replace_all('\\s+', '').str.replace_all('-', '').str.replace_all('\\.', '')
    num = lambda c: pl.col(c).cast(pl.Utf8).str.replace_all(',', '.').cast(pl.Float64, strict=False)
    цифры = lambda c: raw(c).str.extract_all('\\d+')
    валидна = lambda c: (цифры(c).list.len() >= 2) & raw(c).str.contains('[А-ЯA-Z]')
    ex = цифры('после.Насос')
    t = g.select(op_id=pl.int_range(pl.len()).cast(pl.Int32), тип=raw('общ.Тип'), сырое_до=raw('до.Насос'), сырое_после=raw('после.Насос'), нас_до=canon('до.Насос'), нас_по=canon('после.Насос'), валид_до=валидна('до.Насос'), валид_по=валидна('после.Насос'), произв_м3сут=ex.list.get(-2, null_on_oob=True).cast(pl.Float64, strict=False), напор_м=ex.list.get(-1, null_on_oob=True).cast(pl.Float64, strict=False), произв_план=num('плннасос.Произв-сть, м3/сут'), напор_план=num('плннасос.Напор ЭЦН, м')).with_columns(насос_случай=pl.when(~pl.col('валид_до') | ~pl.col('валид_по')).then(pl.lit('в неопределимо')).when(pl.col('нас_до') == pl.col('нас_по')).then(pl.lit('а тот же типоразмер')).otherwise(pl.lit('б другой типоразмер')), в_границах=pl.col('произв_м3сут').is_between(5, 1500) & pl.col('напор_м').is_between(300, 3500)).with_columns(насос_мощность=pl.when((pl.col('насос_случай') == 'б другой типоразмер') & pl.col('в_границах')).then(pl.col('произв_м3сут') * pl.col('напор_м')).otherwise(None), насос_нулевая=pl.col('насос_случай') == 'а тот же типоразмер', насос_неопределим=pl.col('насос_случай') == 'в неопределимо')
    только_начертание = t.filter((pl.col('сырое_до') != pl.col('сырое_после')) & (pl.col('нас_до') == pl.col('нас_по')) & pl.col('валид_до') & pl.col('валид_по')).height
    сч = t.group_by('насос_случай').len().sort('насос_случай')
    print('насос, три исхода по всем строкам реестра:')
    print('  ' + ' | '.join((f'{r[0]}: {r[1]:,}' for r in сч.iter_rows())))
    print(f"  мощность измерена (только случай б, в границах типоразмеров): {int(t['насос_мощность'].is_not_null().sum()):,}")
    print(f'  ВКЛАД НОРМАЛИЗАЦИИ: пар, различающихся ТОЛЬКО начертанием или пробелами, {только_начертание} -- нормализация случай (б) НЕ создаёт')
    ev_пул = pl.read_parquet(cfg.data('interim/gtm_events.parquet')).filter(pl.col('выбираемая')).select(op_id=pl.col('op_id').cast(pl.Int64))
    пул = ev_пул.join(t.select(op_id=pl.col('op_id').cast(pl.Int64), насос_случай='насос_случай'), on='op_id', how='left')
    сч_пул = пул.group_by('насос_случай').len().sort('насос_случай')
    д = {str(r[0])[:1]: r[1] for r in сч_пул.iter_rows() if r[0] is not None}
    n_пул = пул.height
    print(f'три исхода ПО ПУЛУ выбираемых операций ({n_пул:,}):')
    for к in ('а', 'б', 'в'):
        v = int(д.get(к, 0))
        print(f'  случай {к} в пуле: {v:,} ({100 * v / max(n_пул, 1):.1f}%)')
    сум = sum((int(д.get(к, 0)) for к in ('а', 'б', 'в')))
    print(f'  сумма трёх исходов по пулу: {сум:,} -- обязана равняться {n_пул:,}')
    assert сум == n_пул, f'три исхода по пулу дают {сум}, а операций {n_пул}: сопоставление по op_id или разбор случая неполны'
    z = t.filter(pl.col('в_границах')).drop_nulls(['произв_план', 'напор_план'])
    print(f"  СВЕРКА С НЕЗАВИСИМЫМИ ПОЛЯМИ плннасос ({z.height:,} строк): производительность совпала точно у {100 * float((z['произв_м3сут'] == z['произв_план']).mean()):.1f}%, напор у {100 * float((z['напор_м'] == z['напор_план']).mean()):.1f}%, медиана |отн. расхождения| напора {float(((z['напор_м'] - z['напор_план']).abs() / z['напор_план']).median()):.4f}")
    print('  точное совпадение напора низкое ЗАКОНОМЕРНО: плановый напор -- требуемый, номенклатурный -- ближайшего доступного типоразмера')
    return t.select('op_id', 'насос_случай', 'насос_мощность', 'насос_нулевая', 'насос_неопределим', 'произв_м3сут', 'напор_м', тип_реестр='тип')

def станко_фикс(ev: pl.DataFrame) -> pl.DataFrame:
    ok = lambda e: e.is_between(pl.date(1990, 1, 1), pl.date(2026, 1, 1))
    k = pl.read_parquet(cfg.data('parquet/konstrukciya.parquet')).select(well=norm_well(pl.col('SRC_WELL_NAME')), тип_кол=pl.col('Тип колонны').cast(pl.Utf8).str.strip_chars(), спуск=pl.col('Дата спуска').str.to_date('%d.%m.%y', strict=False), опр=pl.col('Дата опрессовки').str.to_date('%d.%m.%y', strict=False)).filter(pl.col('тип_кол') == 'хвостов').with_columns(base=base_well(pl.col('well')), спуск=pl.when(ok(pl.col('спуск'))).then('спуск'), опр=pl.when(ok(pl.col('опр'))).then('опр')).drop_nulls(['спуск', 'опр'])
    zbs = ev.filter(pl.col('тип') == 'ЗБС').select('op_id', 'well', 'beg').drop_nulls(['well', 'beg']).with_columns(base=base_well(pl.col('well')))
    в_окне = lambda c: (pl.col(c) >= pl.col('beg') - pl.duration(days=ZBS_WIN[0])) & (pl.col(c) <= pl.col('beg') + pl.duration(days=ZBS_WIN[1]))
    пары = zbs.join(k.select('base', 'спуск', 'опр'), on='base', how='inner').filter(в_окне('спуск') & в_окне('опр'))
    сыр = пары.with_columns(инт=(pl.col('опр') - pl.col('спуск')).dt.total_days())
    негодн = сыр.filter(pl.col('инт') < 0).height
    зам = сыр.filter(pl.col('инт') >= 0).group_by('op_id').agg(станко_фикс_сут=pl.col('инт').median())
    n_збс = zbs.height
    мед = float(зам['станко_фикс_сут'].median()) if зам.height else 0.0
    print(f'  записей отброшено как негодные (опрессовка раньше спуска): {негодн}')
    print(f'станко-сутки ЗБС (фикс. часть) ИЗМЕРЕНЫ у {зам.height} из {n_збс} ({100 * зам.height / max(n_збс, 1):.1f}%), медиана {мед:.1f} сут; остальным подставляется она же')
    if зам.height:
        q = зам['станко_фикс_сут']
        print(f'  квартили {q.quantile(0.25):.0f} / {мед:.0f} / {q.quantile(0.75):.0f}; q90 {q.quantile(0.9):.0f}; максимум {q.max():.0f}; меньше прежних 10 сут у {100 * (q < 10).mean():.1f}%')
    return зам.with_columns(станко_фикс_мед=pl.lit(мед))

def _window(df):
    return df.filter((pl.col('дата') >= pl.col('beg') - pl.duration(days=WIN_BEFORE)) & (pl.col('дата') <= pl.coalesce(pl.col('end'), pl.col('beg')) + pl.duration(days=WIN_AFTER)))

def load_resources(ev: pl.DataFrame | None=None) -> pl.DataFrame:
    if ev is None:
        ev = pl.read_parquet(cfg.data('interim/gtm_events.parquet')).filter(pl.col('выбираемая')).select('op_id', 'well', 'тип', 'год', 'beg', 'end', 'vnr', 'dur')
    else:
        ev = ev.select('op_id', 'well', 'тип', 'год', 'beg', 'end', 'vnr', 'dur')
    base = ev.select('op_id', 'well', 'beg', 'end').drop_nulls(['well', 'beg'])
    fr = pl.read_parquet(cfg.data('parquet/frac_stages.parquet')).filter(pl.col('дата').is_not_null()).with_columns(rcp=pl.col('тип_проппанта').cast(pl.Utf8).str.contains('RCP').fill_null(False))
    mp = pl.coalesce(pl.col('Mp_факт_т'), pl.col('Mp_план_т'))
    grp = _window(ev.filter(pl.col('тип').is_in(FRAC_TYPES)).select('op_id', 'well', 'beg', 'end').drop_nulls(['well', 'beg']).join(fr.select('well', 'дата', 'Mp_факт_т', 'Mp_план_т', 'Vгель_м3', 'Vподушки_м3', 'rcp', 'успех', 'Xf_м'), on='well')).group_by('op_id').agg(стадий=pl.len(), проппант_т=mp.sum(), проппант_rcp_т=pl.when(pl.col('rcp')).then(mp).otherwise(0.0).sum(), гель_м3=(pl.col('Vгель_м3').fill_null(0) + pl.col('Vподушки_м3').fill_null(0)).sum(), Xf_м=pl.col('Xf_м').mean(), успех_грп=pl.col('успех').cast(pl.Float64, strict=False).mean())
    print(f'операций со стадиями ГРП: {grp.height:,}')
    g = pl.read_parquet(cfg.data('parquet/gnkt_ops.parquet')).filter(pl.col('работа_оцз').is_not_null()).with_columns(well=norm_well(pl.col('well_raw')), дата=pl.col('работа_оцз').dt.date())
    gn = _window(base.join(g.select('well', 'дата', 'кислота_м3', 'опз_м3', 'азот_т', 'работа_ч', 'нпв_подр_ч'), on='well')).group_by('op_id').agg(кислота_м3=(pl.col('кислота_м3').fill_null(0) + pl.col('опз_м3').fill_null(0)).sum(), азот_т=pl.col('азот_т').sum(), гнкт_постановок=pl.len(), гнкт_нпв_ч=pl.col('нпв_подр_ч').sum())
    print(f'операций с постановками ГНКТ: {gn.height:,}')
    ink = pl.read_parquet(cfg.data('parquet/inkl_wells.parquet')).with_columns(w=norm_well(pl.col('well'))).group_by('w').agg(md=pl.col('md_max').max(), offset=pl.col('offset_max').max())
    lat = ev.filter(pl.col('тип') == 'ЗБС').select('op_id', 'well').with_columns(base=pl.col('well').str.replace('[_/-].*$', '')).join(ink.rename({'w': 'well', 'md': 'md_с', 'offset': 'off_с'}), on='well', how='left').join(ink.rename({'w': 'base', 'md': 'md_б'}).drop('offset'), on='base', how='left').with_columns(проходка_м=pl.when(pl.col('md_с') > pl.col('md_б')).then(pl.col('md_с') - pl.col('md_б')).otherwise(None)).select('op_id', 'проходка_м', смещение_м='off_с')
    n_lat = lat['проходка_м'].drop_nulls().len()
    med_lat = lat['проходка_м'].median()
    print(f'ЗБС с восстановленной проходкой: {n_lat} из {lat.height}, медиана {med_lat:.0f} м -- остальным подставляется она же')
    num = lambda c: pl.col(c).str.replace_all(',', '.').cast(pl.Float64, strict=False)
    k = pl.read_parquet(cfg.data('parquet/konstrukciya.parquet')).select(well=norm_well(pl.col('SRC_WELL_NAME')), тип_колонны=pl.col('Тип колонны').cast(pl.Utf8).str.strip_chars(), башмак=num('Глубина установки башмака, м'), нач=num('Глубина начала, м'), объем_цем=num('Объем тамп раствора, м3'), дата=pl.col('Дата спуска').str.to_date('%d.%m.%y', strict=False))
    эк = k.filter(pl.col('тип_колонны') == 'э/к').group_by('well').agg(башмак_эк=pl.col('башмак').max())
    хв = k.filter(pl.col('тип_колонны') == 'хвостов').drop_nulls('дата')
    zbs = _window(ev.filter(pl.col('тип') == 'ЗБС').select('op_id', 'well', 'beg', 'end').drop_nulls(['well', 'beg']).join(хв.select('well', 'дата', 'объем_цем'), on='well')).group_by('op_id').agg(цемент_м3=pl.col('объем_цем').sum())
    print(f'операций ЗБС с цементированием хвостовика: {zbs.height:,}')
    стан = станко_фикс(ev)
    occ = occupancy(ev).select('op_id', 'T_техники', 'база_сут', 'восстановлено', 'занятость_крс_сут')
    print(f"операций с восстановленной по датам занятостью: {occ['восстановлено'].sum():,} из {occ.height:,}")
    pump = pump_nomenclature()
    chk = ev.join(pump.select('op_id', 'тип_реестр'), on='op_id', how='inner')
    расх = chk.filter(pl.col('тип').str.to_uppercase() != pl.col('тип_реестр')).height
    assert расх == 0, f'op_id реестра и gtm_events не совпадают: тип различается у {расх} из {chk.height} операций'
    print(f'сверка индексов: тип операции совпал у всех {chk.height:,} состыкованных op_id (проверка сдвига индекса внешним полем, не тавтология)')
    pump = pump.drop('тип_реестр')
    t = ev.join(grp, on='op_id', how='left').join(gn, on='op_id', how='left').join(zbs, on='op_id', how='left').join(lat, on='op_id', how='left').join(occ, on='op_id', how='left').join(pump, on='op_id', how='left').join(стан, on='op_id', how='left').with_columns(станко_фикс_мед=pl.col('станко_фикс_мед').fill_null(стан['станко_фикс_мед'].max() if стан.height else 0.0)).with_columns(проходка_м=pl.when(pl.col('тип') == 'ЗБС').then(pl.col('проходка_м').fill_null(med_lat)).otherwise(pl.col('проходка_м')))
    print('\nпокрытие физикой по типам:')
    print(t.group_by('тип').agg(n=pl.len(), стадии=pl.col('стадий').is_not_null().sum(), гнкт=pl.col('гнкт_постановок').is_not_null().sum(), проходка=(pl.col('проходка_м') > 10).sum(), смещение=pl.col('смещение_м').is_not_null().sum(), длительность=pl.col('dur').is_not_null().sum(), насос_б=pl.col('насос_мощность').is_not_null().sum(), насос_а=pl.col('насос_нулевая').sum(), насос_в=pl.col('насос_неопределим').sum(), мощн_мед=pl.col('насос_мощность').median().round(0), занятость=pl.col('восстановлено').sum(), dur_мед=pl.col('dur').median(), занят_мед=pl.col('занятость_крс_сут').median().round(1)).sort('n', descending=True).to_pandas().to_string(index=False))
    return t

def capex(t: pl.DataFrame, pr: Prices) -> pl.DataFrame:
    z = lambda c: pl.col(c).fill_null(0.0)
    прочий = (z('проппант_т') - z('проппант_rcp_т')).clip(0, None)
    t = t.with_columns(c_проппант=прочий * pr.проппант_руб_т + z('проппант_rcp_т') * pr.проппант_rcp_руб_т, c_гель=z('гель_м3') * pr.гель_руб_м3, c_кислота=z('кислота_м3') * pr.кислота_руб_м3, c_азот=z('азот_т') * pr.азот_руб_т, c_цемент=z('цемент_м3') * pr.цемент_руб_м3, c_бурение=z('проходка_м').clip(0, 4000) * pr.бурение_руб_м, c_насос=z('насос_мощность') * pr.насос_руб_м3сут_м, c_крс=z('занятость_крс_сут') * pr.крс_руб_сутки, c_флот_грп=z('стадий') * pr.флот_грп_руб_сутки, c_станок=pl.when(pl.col('тип') == 'ЗБС').then((z('проходка_м') / pr.мех_скорость_м_сутки + pl.coalesce(pl.col('станко_фикс_сут'), pl.col('станко_фикс_мед'))) * pr.станок_збс_руб_сутки).otherwise(0.0), c_моб=pl.lit(pr.мобилизация_руб), мат_ненулевая=pl.col('стадий').is_not_null() | pl.col('гнкт_постановок').is_not_null() | (pl.col('проходка_м') > 10).fill_null(False) | pl.col('насос_мощность').is_not_null(), мат_нуль_измерен=pl.col('насос_нулевая').fill_null(False)).with_columns(мат_измерены=pl.col('мат_ненулевая') | pl.col('мат_нуль_измерен'), c_материалы=sum((pl.col(c) for c in MAT)), c_время=sum((pl.col(c) for c in TIME)))
    med = t.filter('мат_измерены').group_by('тип').agg(мат_med=pl.col('c_материалы').median())
    return t.join(med, on='тип', how='left').with_columns(c_материалы_исп=pl.when(pl.col('мат_измерены')).then(pl.col('c_материалы')).otherwise(pl.coalesce(pl.col('мат_med'), pl.lit(0.0)))).with_columns(capex=(pl.col('c_время') + pl.col('c_материалы_исп')) * pr.scale)
if __name__ == '__main__':
    t = load_resources()
    pr = Prices()
    c = capex(t, pr)
    c.write_parquet(cfg.data('interim/capex_physical.parquet'), compression='zstd')
    print('\n=== CAPEX ИЗ ФИЗИКИ, млн ₽ ===')
    print(c.group_by('тип').agg(n=pl.len(), мат_изм=pl.col('мат_измерены').sum(), p10=(pl.col('capex') / M).quantile(0.1).round(1), медиана=(pl.col('capex') / M).median().round(1), p90=(pl.col('capex') / M).quantile(0.9).round(1), разброс=(pl.col('capex').quantile(0.9) / pl.col('capex').quantile(0.1)).round(2)).sort('n', descending=True).to_pandas().to_string(index=False))
    print('\n=== СОСТАВ, медиана по типу, млн ₽ ===')
    print('ВНИМАНИЕ: это медианы СЛАГАЕМЫХ, и они НЕ складываются в медиану')
    print('CAPEX: медиана суммы не равна сумме медиан. Ниже напечатано и то,')
    print('и другое, чтобы разность нельзя было принять за ошибку счёта.')
    print(c.group_by('тип').agg(n=pl.len(), проппант=(pl.col('c_проппант') / M).median().round(1), гель=(pl.col('c_гель') / M).median().round(1), кислота=(pl.col('c_кислота') / M).median().round(2), бурение=(pl.col('c_бурение') / M).median().round(1), цемент=(pl.col('c_цемент') / M).median().round(2), насос=(pl.col('c_насос') / M).median().round(2), крс=(pl.col('c_крс') / M).median().round(1), флот=(pl.col('c_флот_грп') / M).median().round(1), станок=(pl.col('c_станок') / M).median().round(1)).sort('n', descending=True).to_pandas().to_string(index=False))
    parts = MAT + TIME
    cmp = c.group_by('тип').agg(n=pl.len(), медиана_суммы=(pl.col('capex') / M).median().round(1), сумма_медиан=sum(((pl.col(k) / M).median() for k in parts)).round(1)).with_columns(расхождение=(pl.col('сумма_медиан') - pl.col('медиана_суммы')).round(1)).sort('n', descending=True)
    print(f'\nмедиана суммы против суммы медиан (по всем {len(parts)} слагаемым), млн ₽:')
    print(cmp.to_pandas().to_string(index=False))
    print('\n=== ДОЛЯ ОПЕРАЦИЙ С ИЗМЕРЕННОЙ МАТЕРИАЛЬНОЙ ЧАСТЬЮ ===')
    print('ТРИ исхода по насосу, а не два: (а) тот же типоразмер -- объём измерен и')
    print('равен нулю; (б) другой типоразмер -- объём измерен и положителен;')
    print('(в) неопределимо -- одно из полей пусто или не разбирается.')
    share = c.group_by('тип').agg(n=pl.len(), насос_а=pl.col('насос_нулевая').sum(), насос_б=pl.col('насос_мощность').is_not_null().sum(), насос_в=pl.col('насос_неопределим').sum(), стадии=pl.col('стадий').is_not_null().sum(), гнкт=pl.col('гнкт_постановок').is_not_null().sum(), проходка=(pl.col('проходка_м') > 10).fill_null(False).sum(), ненулевая=pl.col('мат_ненулевая').sum(), определена=pl.col('мат_измерены').sum()).with_columns(доля_ненул=(pl.col('ненулевая') / pl.col('n')).round(3), доля_опред=(pl.col('определена') / pl.col('n')).round(3)).sort('n', descending=True)
    print(share.to_pandas().to_string(index=False))
    всего = c.height
    нен = int(c['мат_ненулевая'].sum())
    опр = int(c['мат_измерены'].sum())
    нул = int(c['мат_нуль_измерен'].sum())
    print(f'\nДВА числа, и путать их нельзя:')
    print(f'  с ИЗМЕРЕННЫМ ПОЛОЖИТЕЛЬНЫМ материальным объёмом: {нен:,} из {всего:,} ({100 * нен / всего:.1f}%)')
    print(f'  с ОПРЕДЕЛЁННОЙ материальной частью (включая измеренный ноль у {нул:,} операций случая «а»): {опр:,} из {всего:,} ({100 * опр / всего:.1f}%)')
    print(f'  НЕОПРЕДЕЛИМЫХ, получающих медиану по типу: {всего - опр:,} из {всего:,} ({100 * (всего - опр) / всего:.1f}%)')
    пересеч = c.filter(pl.col('мат_ненулевая') & pl.col('мат_нуль_измерен'))
    assert нен + нул - пересеч.height == опр, f'объединение не замыкается: {нен} + {нул} - {пересеч.height} != {опр}'
    print(f'  в ОБОИХ множествах сразу: {пересеч.height} операций -- множества НЕ РАЗДЕЛЬНЫ, поэтому {нен:,} + {нул:,} = {нен + нул:,} величиной этой работы НЕ является')
    print(f'  замыкание: {нен:,} + {нул:,} - {пересеч.height} = {опр:,} определённых, и {опр:,} + {всего - опр:,} = {всего:,}')
    print('  состав пересечения по типам и источнику ПОЛОЖИТЕЛЬНОГО объёма')
    print('  (столбец «насос» обязан быть нулевым: случай «а» -- это нуль по насосу):')
    print(пересеч.group_by('тип').agg(n=pl.len(), стадии=pl.col('стадий').is_not_null().sum(), гнкт=pl.col('гнкт_постановок').is_not_null().sum(), проходка=(pl.col('проходка_м') > 10).fill_null(False).sum(), насос=pl.col('насос_мощность').is_not_null().sum()).sort('n', descending=True).to_pandas().to_string(index=False))
    print('Первое число -- то, что модель действительно измерила как расход.')
    print('Второе включает измеренный НУЛЬ и поэтому выше; оно отвечает на другой')
    print('вопрос -- по скольким операциям объём не пришлось подставлять.')
    print('ОГОВОРКА к случаю (а): совпадение типоразмера доказывает, что типоразмер')
    print('не изменился, а НЕ то, что спустили тот же физический насос. Нуль там --')
    print('КОНСЕРВАТИВНОЕ допущение, а не измерение; оно занижает затраты, и его')
    print('вес мал -- см. долю случая (а) в таблице выше.')
    отн = c.group_by('тип').agg(n=pl.len(), отношение=(pl.col('c_крс') + pl.col('c_моб')).sum() / pl.col('capex').sum(), доля_насос=pl.col('c_насос').sum() / pl.col('capex').sum(), доля_матер=pl.col('c_материалы_исп').sum() / pl.col('capex').sum()).sort('n', descending=True)
    плоские = отн.filter(pl.col('отношение') >= 0.9995)
    print('\n=== ДОЛЯ ВРЕМЕНИ И МОБИЛИЗАЦИИ В ЗАТРАТАХ ПО ТИПАМ ===')
    print(отн.select('тип', 'n', отношение=pl.col('отношение').round(3), доля_насос=pl.col('доля_насос').round(3), доля_матер=pl.col('доля_матер').round(3)).to_pandas().to_string(index=False))
    print(f"типов с отношением (c_крс + c_моб)/capex = 1.000: {плоские.height} из {отн.height}; операций в них: {int(плоские['n'].sum()):,}")
    print('\n=== Э3.1. ПОКРЫТИЕ ПО ТИПАМ: ДОЛЯ МАТЧАСТИ В CAPEX, СТРОКОЙ НА ТИП ===')
    for r in отн.iter_rows(named=True):
        print(f"  тип {r['тип']}: доля матчасти {r['доля_матер']:.3f}, доля насоса {r['доля_насос']:.3f}, операций {r['n']:,}")
    _см = отн.sort('доля_матер')
    _мин, _макс = (_см.row(0, named=True), _см.row(_см.height - 1, named=True))
    print(f"  размах доли матчасти по {отн.height} типам: {_мин['доля_матер']:.3f} ({_мин['тип']}) .. {_макс['доля_матер']:.3f} ({_макс['тип']})")
    print('  НИ ОДНОГО типа с нулевой долей матчасти нет, и это не следует из')
    print('  размаха: минимум размаха мог бы принадлежать типу с нулём.')
    print(f"  типов с долей матчасти РОВНО 0.000 (по той же округлённой до трёх знаков величине, что напечатана выше): {отн.filter(pl.col('доля_матер').round(3) == 0).height} из {отн.height}")
    print('\n=== Э3.2. ПОКРЫТИЕ ПО ТИПАМ: ДОЛЯ ОПРЕДЕЛЁННОГО ОБЪЁМА, СТРОКОЙ НА ТИП ===')
    for r in share.iter_rows(named=True):
        print(f"  тип {r['тип']}: определена {r['доля_опред']:.3f} ({r['определена']:,} из {r['n']:,}), ненулевая {r['доля_ненул']:.3f}")
    _слаб = share.filter(pl.col('доля_опред') < 0.7)
    _сл_n = int(_слаб['n'].sum())
    print(f'  типов с долей определённого объёма ниже 0.7: {_слаб.height} из {share.height}; операций в них: {_сл_n:,} из {всего:,} ({100 * _сл_n / всего:.1f}%)')
    _круп = share.filter(pl.col('n') >= 100)
    print(f"  среди типов от 100 операций ({_круп.height} типов, {int(_круп['n'].sum()):,} операций) МИНИМУМ доли определённого объёма: {_круп['доля_опред'].min():.3f}")
    отн2 = c.group_by('тип').agg(n=pl.len(), кап=pl.col('capex').sum(), крс_моб=(pl.col('c_крс') + pl.col('c_моб')).sum(), флот_станок=(pl.col('c_флот_грп') + pl.col('c_станок')).sum(), матер=pl.col('c_материалы_исп').sum()).with_columns(д_крс_моб=pl.col('крс_моб') / pl.col('кап'), д_прочее=pl.col('флот_станок') / pl.col('кап'), д_матер=pl.col('матер') / pl.col('кап')).with_columns(д_не_матчасть=1 - pl.col('д_матер'), замыкание=(pl.col('крс_моб') + pl.col('флот_станок') + pl.col('матер')) / pl.col('кап')).sort('n', descending=True)
    print('\n=== Э3.3. ДОЛЯ ВРЕМЕНИ В CAPEX, СТРОКОЙ НА ТИП (ТРИ РАЗНЫЕ ДОЛИ) ===')
    for r in отн2.iter_rows(named=True):
        print(f"  тип {r['тип']}: крс и моб {r['д_крс_моб']:.3f}, флот и станок {r['д_прочее']:.3f}, не матчасть {r['д_не_матчасть']:.3f}")
    for имя, к in (('(крс+моб)', 'д_крс_моб'), ('НЕ матчасти', 'д_не_матчасть')):
        s = отн2.sort(к)
        мин, макс = (s.row(0, named=True), s.row(s.height - 1, named=True))
        print(f"  размах доли {имя} по {отн2.height} типам: {мин[к]:.3f} ({мин['тип']}) .. {макс[к]:.3f} ({макс['тип']})")
    зам = отн2.filter(pl.col('замыкание').round(3) == 1)
    print(f'  типов, у которых три доли складываются в 1.000: {зам.height} из {отн2.height} -- это РАЗЛОЖЕНИЕ, а не три независимых числа')
    print('\n=== ПРЕЖНИЕ ВЫДУМАННЫЕ КОНСТАНТЫ ПРОТИВ ФИЗИКИ ===')
    old = {'ОПЗ': 2.0, 'ОПТ': 3.0, 'ГРП': 20.0, 'ППР': 2.5, 'ВБД': 5.0, 'ЗБС': 110.0, 'ПиП': 6.0, 'ЛА': 4.0, 'РИР': 5.0}
    r = c.group_by('тип').agg(физ=(pl.col('capex') / M).median().round(1), n=pl.len()).with_columns(было=pl.col('тип').replace_strict(old, default=None)).with_columns(отношение=(pl.col('физ') / pl.col('было')).round(2)).sort('n', descending=True)
    print(r.to_pandas().to_string(index=False))
