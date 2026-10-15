"""Causes of disagreement between the parser and the labels."""
import config as cfg
import json
import re
from collections import Counter
import polars as pl
O = {'тот_же': 'а', 'другой': 'б', 'нельзя_определить': 'в'}

def reasons(r):
    out = []
    for s in ('до', 'после'):
        x = r[s]
        if x['форма'] == 'B':
            out.append('формат Q/H без серии («50/1550»)')
        if x['Q'] is not None and x['Q'] > 1500:
            out.append('подача > 1500')
        if x['H'] is not None and x['H'] < 300:
            out.append('напор < 300 (усечён)')
        if x['форма'] == 'C':
            out.append('серия и одно число')
        if x['форма'] == 'D':
            out.append('диапазон подачи')
        if x['серия'] and re.match('^\\d', x['серия']):
            out.append('слитый числовой префикс серии')
        if 'не ЭЦН' in (r['заметка'] or ''):
            out.append('не ЭЦН')
        if x['гомоглиф']:
            out.append('гомоглиф')
        if x['форма'] == 'E':
            out.append('не номенклатура')
    return sorted(set(out)) or ['прочее']

def main():
    k = pl.read_parquet(cfg.data('interim/parser_sample_key.parquet'))
    sel = pl.read_parquet(cfg.data('interim/ep30_pselect_ops.parquet')).get_column('op_id').cast(pl.Int64).to_list()
    sel = set(sel)
    L = {}
    for line in open(cfg.data('interim/parser_labels_final.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        L[r['id']] = r
    ce, cq = (Counter(), Counter())
    we, wq = (Counter(), Counter())
    acc = {True: [0, 0.0, 0, 0, 0.0, 0.0], False: [0, 0.0, 0, 0, 0.0, 0.0]}
    for row in k.iter_rows(named=True):
        r = L[row['id']]
        w = row['вес']
        e = row['насос_случай'][0] != O[r['сравнение_строгое']]
        pq, lq = (row['произв_м3сут'], r['после']['Q'])
        qe = (pq is None) != (lq is None) or (pq is not None and lq is not None and (pq != lq))
        for flag, c, cw in ((e, ce, we), (qe, cq, wq)):
            if flag:
                for q in reasons(r) + ['ВСЕГО строк']:
                    c[q] += 1
                    cw[q] += w
        a = acc[row['op_id'] in sel]
        a[0] += 1
        a[1] += w
        a[2] += e
        a[3] += qe
        a[4] += w * e
        a[5] += w * qe
    L2 = ['# Причины расхождений парсера и осуществимость возмущения q (ПРОБА)', '', 'Агрегаты: строки выборки (n) и оценка числа операций пула (вес). Причины перекрываются. Причина берётся из метки (форма, значения, заметка «не ЭЦН»), не из текста строки.', '', '| причина | исход неверен, n | вес | подача неверна, n | вес |', '|---|---|---|---|---|']
    for q in sorted(set(ce) | set(cq), key=lambda z: (z == 'ВСЕГО строк', -cq[z])):
        L2.append(f'| {q} | {ce[q]} | {we[q]:.0f} | {cq[q]} | {wq[q]:.0f} |')
    L2 += ['', '## Операции пула, несущие решение отбора (в отборе хотя бы раз из 500 розыгрышей цен)', '', '| группа | n выборки | вес | ошибка исхода (по весу) | ошибка подачи (по весу) |', '|---|---|---|---|---|']
    for nm, f in (('в отборе', True), ('вне отбора', False)):
        a = acc[f]
        L2.append(f'| {nm} | {a[0]} | {a[1]:.0f} | {a[4] / a[1]:.4f} | {a[5] / a[1]:.4f} |')
    open(cfg.results('ep30_parser_causes.md'), 'w', encoding='utf-8').write('\n'.join(L2) + '\n')
    print('\n'.join(L2))
main()
