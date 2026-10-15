"""Evaluation of the pump parser on the synthetic test set (no operator data)."""
import config as cfg
import collections
import contextlib
import io
import json
import os
import pathlib
import sys
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)
import cost_physical
ФИКСТУРА = ROOT / 'tests/fixtures/parser_synthetic.jsonl'
ОТЧЁТ = ROOT / cfg.results('ep30_parser_synthetic.md')
ПОЛЯ = ['случай', 'произв', 'напор', 'мощность']
СЛУЧАЙ = {'а тот же типоразмер': 'а', 'б другой типоразмер': 'б', 'в неопределимо': 'в'}
ИЗВЕСТНЫЕ = {'разделители', 'латиница', 'кириллица+латиница', 'гомоглиф в числе', 'обрезано', 'B без серии', 'диапазон подачи', 'не ЭЦН', 'серия и одно число', 'не номенклатура', 'не номенклатура с цифрами', 'старая/новая номенклатура'}
ПОРОГ_ДОЛИ = 0.69

def строки():
    return [json.loads(s) for s in ФИКСТУРА.read_text(encoding='utf-8').splitlines() if s.strip()]

def разобрать(rows):
    n = len(rows)
    синт = pl.DataFrame({'до.Насос': [r['до_насос'] for r in rows], 'после.Насос': [r['после_насос'] for r in rows], 'общ.Тип': ['ОПЗ'] * n, 'плннасос.Произв-сть, м3/сут': ['1'] * n, 'плннасос.Напор ЭЦН, м': ['1'] * n})
    события = pl.DataFrame({'op_id': list(range(n)), 'выбираемая': [True] * n})
    подмена = {'gtm_registry.parquet': синт, 'gtm_events.parquet': события}
    orig = pl.read_parquet
    pl.read_parquet = lambda p, *a, **k: next((v for f, v in подмена.items() if str(p).endswith(f)), None) if str(p).endswith(tuple(подмена)) else orig(p, *a, **k)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            t = cost_physical.pump_nomenclature()
    finally:
        pl.read_parquet = orig
    return t.sort('op_id').to_dicts()

def прогон():
    rows = строки()
    вых = разобрать(rows)
    итог = []
    for r, o in zip(rows, вых):
        got = {'случай': СЛУЧАЙ[o['насос_случай']], 'произв': o['произв_м3сут'], 'напор': o['напор_м'], 'мощность': o['насос_мощность']}
        ож = r['ожидание']
        ок = {k: got[k] == ож[k] for k in ПОЛЯ}
        итог.append({**r, 'получено': got, 'совпало': ок})
    return итог

def test_целостность_набора():
    rows = строки()
    assert 150 <= len(rows) <= 250, f'строк {len(rows)}'
    assert len({r['id'] for r in rows}) == len(rows), 'id не уникальны'
    for r in rows:
        assert {'до_насос', 'после_насос', 'истина_до', 'истина_после', 'ожидание', 'класс'} <= set(r)
        assert r['ожидание']['случай'] in ('а', 'б', 'в')

def test_нет_строк_пула():
    p = ROOT / cfg.data('parquet/gtm_registry.parquet')
    if not p.exists():
        print('  (данных нет, проверка пересечения с пулом пропущена)')
        return
    пул = pl.read_parquet(p)
    norm = lambda s: ' '.join(str(s).upper().split())
    в_пуле = set()
    for c in ('до.Насос', 'после.Насос'):
        в_пуле |= {norm(x) for x in пул[c].drop_nulls().to_list()}
    наши = set()
    for r in строки():
        for k in ('до_насос', 'после_насос'):
            if r[k].strip():
                наши.add(norm(r[k]))
    общие = наши & в_пуле
    assert not общие, f'строк набора, совпавших со строкой пула: {len(общие)}'

def test_дубли_дают_одинаковый_выход():
    п = прогон()
    by = collections.defaultdict(set)
    for r in п:
        by[r['до_насос'], r['после_насос']].add(json.dumps(r['получено'], sort_keys=True))
    плохо = [k for k, v in by.items() if len(v) > 1]
    assert not плохо, f'одна пара строк -- разный выход: {len(плохо)}'

def test_совпадение_с_ожиданием():
    п = прогон()
    полное = sum((all(r['совпало'].values()) for r in п)) / len(п)
    классы = {r['класс'] for r in п if not all(r['совпало'].values())}
    новые = классы - ИЗВЕСТНЫЕ
    assert not новые, f'новые классы расхождений: {sorted(новые)}'
    assert полное >= ПОРОГ_ДОЛИ, f'доля полного совпадения {полное:.4f} < {ПОРОГ_ДОЛИ}'

def отчёт():
    п = прогон()
    n = len(п)
    дол = lambda f: sum((f(r) for r in п)) / n
    полное = дол(lambda r: all(r['совпало'].values()))
    затр = дол(lambda r: r['совпало']['случай'] and r['совпало']['мощность'])
    по_полю = {k: дол(lambda r, k=k: r['совпало'][k]) for k in ПОЛЯ}
    классы = collections.OrderedDict()
    for r in п:
        c = классы.setdefault(r['класс'], {'n': 0, **{k: 0 for k in ПОЛЯ}, 'полн': 0})
        c['n'] += 1
        c['полн'] += all(r['совпало'].values())
        for k in ПОЛЯ:
            c[k] += not r['совпало'][k]
    расх = [(k, v) for k, v in классы.items() if v['полн'] < v['n']]
    out = ['# Синтетический набор для парсера насоса (Б1.7; ПРОБА)', '', 'Набор: `tests/fixtures/parser_synthetic.jsonl`, вымышленные пары строк; ожидаемый выход задан построением по `docs/epic/parser_labeling_guide.md` (генератор `tests/fixtures/make_parser_synthetic.py`), не копией выхода парсера. Проверяется `cost_physical.pump_nomenclature` без правок. Совпадение с ожиданием не есть точность относительно истины: ожидание отражает договорённость инструкции, а не отраслевой стандарт.', '', '| класс | строк | полное совпадение | расхождений: случай | подача | напор | мощность |', '|---|---|---|---|---|---|---|']
    for k, v in классы.items():
        out.append(f"| {k} | {v['n']} | {v['полн']} | {v['случай']} | {v['произв']} | {v['напор']} | {v['мощность']} |")
    out += ['', '## Классы расхождений (хотя бы одна строка не совпала)', '']
    for k, v in расх:
        ex = next((r for r in п if r['класс'] == k and (not all(r['совпало'].values()))))
        out.append(f"- {k}: {v['n'] - v['полн']} из {v['n']}; пример id {ex['id']}, ожидалось {ex['ожидание']}, получено {ex['получено']}")
    out.append('')
    A = lambda имя, з, ф: out.append(f'ЯКОРЬ ep30 parser_synthetic {имя} = {format(з, ф)}')
    A('строк', n, 'd')
    A('полное совпадение', полное, '.4f')
    A('совпадение случая и мощности', затр, '.4f')
    for k in ПОЛЯ:
        A(f'совпадение {k}', по_полю[k], '.4f')
    A('классов в наборе', len(классы), 'd')
    A('классов с расхождением', len(расх), 'd')
    for k, v in расх:
        out.append(f"ЯКОРЬ ep30 parser_synthetic расхождение {k} = {v['n'] - v['полн']}/{v['n']}")
    ОТЧЁТ.parent.mkdir(parents=True, exist_ok=True)
    ОТЧЁТ.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('\n'.join(out))
if __name__ == '__main__':
    if '--report' in sys.argv:
        отчёт()
        sys.exit(0)
    код = 0
    for имя, f in sorted(((n, f) for n, f in globals().items() if n.startswith('test_'))):
        try:
            f()
            print(f'PASSED  {имя}')
        except AssertionError as e:
            print(f'FAILED  {имя}: {e}')
            код = 1
    sys.exit(код)
