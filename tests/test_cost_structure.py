"""Structural tests of the cost model (require the confidential tables)."""
import functools
import os
import pathlib
import sys
from dataclasses import replace
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)
import capex_exante as cx
import cost_physical
from cost_physical import Prices, capex
ОТН = 1e-09
ОБЪЁМЫ = ['проппант_т', 'стадий', 'проходка_м', 'насос_мощность']
МНОЖИТЕЛИ = [1.1, 1.5, 2.0]
ЦЕНА = {'проппант_т': 'проппант_руб_т', 'стадий': 'флот_грп_руб_сутки', 'проходка_м': 'бурение_руб_м', 'насос_мощность': 'насос_руб_м3сут_м'}

class НеПрименим(Exception):
    pass

def _skip(msg):
    try:
        import pytest
    except ImportError:
        raise НеПрименим(msg)
    pytest.skip(msg)

@functools.lru_cache(maxsize=1)
def сырая():
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return cx.ресурсы(True)

@functools.lru_cache(maxsize=1)
def заранее():
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        t = cx.ресурсы(True)
        vp = pl.read_parquet(cx.VOL_SRC)
        return cx.подменить_объёмы(t, vp)

def _s1(T, нарушения):
    base = capex(T, Prices())['capex'].to_numpy()
    цены = Prices()
    for q in ОБЪЁМЫ:
        цена = getattr(цены, ЦЕНА[q])
        есть = (T[q].fill_null(0.0) != 0).to_numpy()
        for k in МНОЖИТЕЛИ:
            c = capex(T.with_columns(pl.col(q) * k), Prices())['capex'].to_numpy()
            уменьшил = (c < base - ОТН * abs(base))[есть].sum()
            не_вырос = (c <= base)[есть].sum() if цена > 0 and k > 1.0 else 0
            нарушения[q, k, 'не убывает'] = int(уменьшил)
            нарушения[q, k, 'строго растёт'] = int(не_вырос)
    return нарушения

def _s1_отчёт(T):
    return _s1(T, {})

def test_s1_монотонность():
    итог = {}
    for имя, T in (('факт', сырая()), ('ex-ante', заранее())):
        for k, v in _s1_отчёт(T).items():
            итог[имя, *k] = v
    плохо = {k: v for k, v in итог.items() if v}
    assert not плохо, f'S1 нарушений: {sum(плохо.values())}; {плохо}'

def test_s2_нулевая_замена_насоса():
    T = сырая()
    c = capex(T, Prices())
    m = c.filter(pl.col('насос_нулевая').fill_null(False))
    assert m.height > 0, 'в таблице нет операций с насос_нулевая'
    плохо_насос = int((m['c_насос'].abs() > 0).sum())
    плохо_мат = int(((m['c_материалы_исп'] - m['c_материалы']).abs() > 0).sum())
    assert плохо_насос == 0 and плохо_мат == 0, f'S2: операций {m.height}; c_насос != 0: {плохо_насос}; c_материалы_исп != c_материалы: {плохо_мат}'

def test_s3_написание_кода_оборудования():
    _skip('не применим: функции разбора строки типоразмера в src/ нет (pump_nomenclature разбирает реестр с диска встроенными выражениями)')

def test_s4_единицы():
    _skip('не применим: нормализации единиц (кг/т, л/м3) в src/ нет, таблица уже в т и м3')

def test_s5_масштаб():
    for имя, T in (('факт', сырая()), ('ex-ante', заранее())):
        c1 = capex(T, Prices())['capex'].to_numpy()
        c2 = capex(T, replace(Prices(), scale=2.0))['capex'].to_numpy()
        плохо = int((abs(c2 - 2 * c1) > ОТН * abs(2 * c1)).sum())
        assert плохо == 0, f'S5 ({имя}): нарушений {плохо} из {len(c1)}'
if __name__ == '__main__':
    код = 0
    for имя, f in sorted(((n, f) for n, f in globals().items() if n.startswith('test_'))):
        try:
            f()
            print(f'PASSED  {имя}')
        except НеПрименим as e:
            print(f'SKIPPED {имя}: {e}')
        except AssertionError as e:
            print(f'FAILED  {имя}: {e}')
            код = 1
    sys.exit(код)
