"""Structural tests of the cost model."""
import contextlib
import importlib.util
import io
import pathlib
import re
import sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('test_cost_structure', ROOT / 'tests/test_cost_structure.py')
tcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tcs)

def якорь(имя, знач):
    print(f'ЯКОРЬ ep30 structure {имя} = {знач}')

def прогнать(f):
    try:
        f()
        return (0, None)
    except tcs.НеПрименим as e:
        return (None, str(e))
    except AssertionError as e:
        return (sum((int(x) for x in re.findall('(?:нарушений|!= 0|c_материалы): (\\d+)', str(e)))), None)
s1_всего = s1_убывание = 0
for T in (tcs.сырая(), tcs.заранее()):
    for (q, k, вид), v in tcs._s1_отчёт(T).items():
        s1_всего += v
        if вид == 'не убывает':
            s1_убывание += v
s2, _ = прогнать(tcs.test_s2_нулевая_замена_насоса)
s5 = 0
for имя, T in (('факт', tcs.сырая()), ('ex-ante', tcs.заранее())):
    import numpy as np
    from dataclasses import replace
    c1 = tcs.capex(T, tcs.Prices())['capex'].to_numpy()
    c2 = tcs.capex(T, replace(tcs.Prices(), scale=2.0))['capex'].to_numpy()
    s5 += int((abs(c2 - 2 * c1) > tcs.ОТН * abs(2 * c1)).sum())
_, s3 = прогнать(tcs.test_s3_написание_кода_оборудования)
_, s4 = прогнать(tcs.test_s4_единицы)
T = tcs.заранее()
ненулевой = T['проппант_т'].fill_null(0.0) != 0
rcp_больше = T['проппант_rcp_т'].fill_null(0.0) > T['проппант_т'].fill_null(0.0)
якорь('S1 нарушений', s1_всего)
якорь('S1 нарушений убывания', s1_убывание)
якорь('S2 нарушений', s2)
якорь('S5 нарушений', s5)
якорь('S3 не применим (1 да)', int(s3 is not None))
якорь('S4 не применим (1 да)', int(s4 is not None))
якорь('операций с rcp больше общего проппанта', int(rcp_больше.sum()))
якорь('операций с ненулевым проппантом', int(ненулевой.sum()))
якорь('операций с rcp больше общего среди ненулевых', int((rcp_больше & ненулевой).sum()))
якорь('операций таблицы', T.height)
