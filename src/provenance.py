"""Provenance marks of the quantities of an operation (observed / parsed / inferred / filled / none)."""
import config as cfg
import pathlib
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
ПУТЬ = ROOT / cfg.data('interim/capex_physical.parquet')
МЕТКИ = ('наблюдено', 'разобрано', 'выведено', 'восполнено', 'нет')
КОЛИЧЕСТВА = ['проппант_т', 'проппант_rcp_т', 'гель_м3', 'стадий', 'кислота_м3', 'азот_т', 'цемент_м3', 'проходка_м', 'насос_мощность', 'занятость_крс_сут', 'станок_фикс', 'материальная_часть']
СТОЛБЕЦ = {q: f'{q}_происхождение' for q in КОЛИЧЕСТВА}

def метки(путь=ПУТЬ) -> pl.DataFrame:
    t = pl.read_parquet(путь)
    збс = pl.col('тип') == 'ЗБС'
    vc = t.filter(збс & pl.col('проходка_м').is_not_null())['проходка_м'].value_counts().sort('count', descending=True)
    мода = vc['проходка_м'][0]
    assert vc['count'][0] > 0.5 * t.filter(збс).height, 'медианная подстановка проходки ЗБС не видна (мода < 50 %)'
    наблюдено_если = lambda ок: pl.when(ок).then(pl.lit('наблюдено')).otherwise(pl.lit('нет'))
    грп = наблюдено_если(pl.col('стадий').is_not_null())
    гнкт = наблюдено_если(pl.col('гнкт_постановок').is_not_null())
    return t.select('op_id', 'тип', проппант_т_происхождение=грп, проппант_rcp_т_происхождение=грп, гель_м3_происхождение=грп, стадий_происхождение=грп, кислота_м3_происхождение=гнкт, азот_т_происхождение=гнкт, цемент_м3_происхождение=наблюдено_если(pl.col('цемент_м3').is_not_null()), проходка_м_происхождение=pl.when(збс & (pl.col('проходка_м') == мода)).then(pl.lit('восполнено')).when(pl.col('проходка_м').is_not_null()).then(pl.lit('наблюдено')).otherwise(pl.lit('нет')), насос_мощность_происхождение=pl.when(pl.col('насос_неопределим')).then(pl.lit('нет')).when(pl.col('насос_нулевая') | pl.col('насос_мощность').is_not_null() | pl.col('насос_случай').str.starts_with('б')).then(pl.lit('разобрано')).otherwise(pl.lit('нет')), занятость_крс_сут_происхождение=pl.when(pl.col('восстановлено')).then(pl.lit('выведено')).otherwise(pl.lit('восполнено')), станок_фикс_происхождение=pl.when(pl.col('станко_фикс_сут').is_not_null()).then(pl.lit('наблюдено')).when(збс).then(pl.lit('восполнено')).otherwise(pl.lit('нет')), материальная_часть_происхождение=pl.when(pl.col('мат_ненулевая')).then(pl.lit('наблюдено')).when(pl.col('мат_нуль_измерен')).then(pl.lit('разобрано')).otherwise(pl.lit('восполнено')))

def проверить_полноту(m: pl.DataFrame, n_ожидается: int) -> None:
    assert m.height == n_ожидается and m['op_id'].n_unique() == n_ожидается, 'число операций или op_id'
    for q in КОЛИЧЕСТВА:
        c = m[СТОЛБЕЦ[q]]
        assert c.null_count() == 0, f'{q}: пустые метки'
        assert set(c.unique().to_list()) <= set(МЕТКИ), f'{q}: метка вне словаря'
