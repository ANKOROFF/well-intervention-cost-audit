"""Table with and without sidetracks."""
import config as cfg
import csv
import pathlib
import sys
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import ep19_crossruler as e19
from ep06_ablation import _цены
from sdt_analysis import отборы_линии, цепь
TOL, HALF = (0.0001, 0.084)
ПРОЧЬ = []

def вывод(s=''):
    ПРОЧЬ.append(s)
    print(s, flush=True)

def якорь(имя, знач, ф='.4f'):
    вывод(f'ЯКОРЬ {имя} = {format(знач, ф)}')

def объед(s):
    return set().union(*s.values())

def main() -> int:
    cap = pl.read_parquet(e19.CAP_ПП)
    збс = set(cap.filter(pl.col('тип') == 'ЗБС')['op_id'].to_list())
    ex = dict(zip(cap['op_id'].to_list(), cap['capex'].to_list()))
    ст = {}
    for имя, c in (('полная', cap), ('без ЗБС', cap.filter(pl.col('тип') != 'ЗБС'))):
        наш, л1 = цепь('t_' + имя.replace(' ', '_'), c)
        ст[имя] = (наш, л1)
    g1, g2 = (ст['полная'][0]['линии'][e19.LINE], ст['без ЗБС'][0]['линии'][e19.LINE])
    if not (abs(g1['GC'] - 0.2631) <= TOL and abs(g1['lo'] - 0.1772) <= TOL and (abs(g1['hi'] - 0.3453) <= TOL) and (ст['полная'][0]['n'] == 5806) and (abs(g2['GC'] - 0.2544) <= TOL) and (abs(g2['lo'] - 0.1909) <= TOL) and (abs(g2['hi'] - 0.3161) <= TOL) and (ст['без ЗБС'][0]['n'] == 5533)):
        print('ОСТАНОВ: регрессия GC не сошлась')
        return 1
    рядки, tsv = ({}, [])
    for имя in ('полная', 'без ЗБС'):
        наш, л1 = ст[имя]
        g = наш['линии'][e19.LINE]
        a, b = (объед(отборы_линии(наш)), объед(отборы_линии(л1)))
        n_ = _цены(наш)
        только_н, только_л = (a - b, b - a)
        d = len(только_н) + len(только_л)
        jac = len(a & b) / len(a | b)
        dK = sum((ex[i] for i in a)) - sum((ex[i] for i in b))
        dnpv = sum((n_[i] for i in только_н)) - sum((n_[i] for i in только_л))
        рядки[имя] = (g, наш, a, b, n_, только_н, только_л, d, jac, dK, dnpv)
    p = рядки['полная']
    if not (p[7] == 229 and abs(p[8] - 0.907) < 5e-05 and (рядки['без ЗБС'][7] == 145)):
        print(f"ОСТАНОВ: расхождение {p[7]}/{p[8]:.4f}, без ЗБС {рядки['без ЗБС'][7]} против 229/0.9070/145")
        return 1
    вывод('# Таблица «полная совокупность / без ЗБС» (ep30_sdt_table)')
    вывод('')
    вывод('Определения (до цифр): расхождение = |S_Н Δ S_Л1| за 9 лет по op_id, линия «модель отклика на сутки»; ΔK = Σcapex(S_Н) − Σcapex(S_Л1) по CAPEX модели Н; NPV-разногласие = V_Н(S_Н \\ S_Л1) − V_Н(S_Л1 \\ S_Н) нашей линейкой строки (одно мерило; совпадает с V_Н(S_Н) − V_Н(S_Л1)). Регрессии GC и 229/0.907/145 сошлись. Структура «только Н / только Л1» из 3.117 (116/229) относится к базовой совокупности 5 871 и здесь не сверяется.')
    вывод('| строка | n | GC [ДИ] | расх. | только Н | только Л1 | Жаккар | ΔK млрд | NPV-разногласие млрд | исход |')
    вывод('|---|---|---|---|---|---|---|---|---|---|')
    for имя in ('полная', 'без ЗБС'):
        g, наш, a, b, n_, tн, tл, d, jac, dK, dn = рядки[имя]
        ис = 'W' if g['lo'] <= 0 else 'S' if abs(g['GC'] - g1['GC']) <= HALF else 'Д'
        вывод(f"| {имя} | {наш['n']} | {g['GC']:.4f} [{g['lo']:.4f}, {g['hi']:.4f}] | {d} | {len(tн)} | {len(tл)} | {jac:.4f} | {dK / 1000000000.0:+.4f} | {dn / 1000000000.0:+.4f} | {ис} |")
        tsv.append([имя, наш['n'], f"{g['GC']:.4f}", f"{g['lo']:.4f}", f"{g['hi']:.4f}", d, len(tн), len(tл), f'{jac:.4f}', f'{dK / 1000000000.0:+.4f}', f'{dn / 1000000000.0:+.4f}', ис])
    g, наш, a, b, n_, tн, tл, d, jac, dK, dn = p
    вывод('')
    r = {}
    for k, f in (('ЗБС', lambda i: i in збс), ('не-ЗБС', lambda i: i not in збс)):
        hн, hл = ([i for i in tн if f(i)], [i for i in tл if f(i)])
        r[k] = (len(hн), len(hл), (sum((n_[i] for i in hн)) - sum((n_[i] for i in hл))) / 1000000000.0, (sum((ex[i] for i in hн)) - sum((ex[i] for i in hл))) / 1000000000.0)
        вывод(f'Полная, {k}: расхождений {len(hн) + len(hл)} (только Н {len(hн)}, только Л1 {len(hл)}), вклад в NPV-разногласие {r[k][2]:+.4f} млрд, в ΔK {r[k][3]:+.4f} млрд')
        tsv.append(['полная|' + k, '', '', '', '', len(hн) + len(hл), len(hн), len(hл), '', f'{r[k][3]:+.4f}', f'{r[k][2]:+.4f}', ''])
    доля = (r['ЗБС'][0] + r['ЗБС'][1]) / d
    вывод(f"Доля ЗБС в расхождениях {доля:.3f} (ЗБС в совокупности {len(збс & set(наш['d']['op_id'].to_list()))} из {наш['n']}).")
    gб = рядки['без ЗБС']
    Sп = {i for i in объед(отборы_линии(наш)) if i not in збс}
    Sб = объед(отборы_линии(ст['без ЗБС'][0]))
    рx = len(Sб ^ Sп)
    вывод(f'Отбор «без ЗБС» против отбора полной цепи без ЗБС: расхождение {рx}, Жаккар {len(Sб & Sп) / len(Sб | Sп):.4f}, ΔK = Σcapex(S_без) − Σcapex(S_полн\\ЗБС) = {(sum((ex[i] for i in Sб)) - sum((ex[i] for i in Sп))) / 1000000000.0:+.4f} млрд.')
    вывод('')
    for имя in ('полная', 'без ЗБС'):
        g, наш, a, b, n_, tн, tл, d, jac, dK, dn = рядки[имя]
        к = 'полная' if имя == 'полная' else 'без ЗБС'
        якорь(f'ep30 sdt {к} GC', g['GC'])
        якорь(f'ep30 sdt {к} GC_lo', g['lo'])
        якорь(f'ep30 sdt {к} GC_hi', g['hi'])
        якорь(f'ep30 sdt {к} расхождений', d, 'd')
        якорь(f'ep30 sdt {к} Жаккар', jac)
        якорь(f'ep30 sdt {к} ΔK млрд', dK / 1000000000.0, '+.4f')
        якорь(f'ep30 sdt {к} NPV-разногласие млрд', dn / 1000000000.0, '+.4f')
    for k in ('ЗБС', 'не-ЗБС'):
        якорь(f'ep30 sdt полная {k} расхождений', r[k][0] + r[k][1], 'd')
        якорь(f'ep30 sdt полная {k} вклад в NPV-разногласие млрд', r[k][2], '+.4f')
    якорь('ep30 sdt полная доля ЗБС в расхождениях', доля, '.3f')
    (ROOT / cfg.results('ep30_sdt_table.md')).write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
    with open(ROOT / cfg.out('sdt_table.tsv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['строка', 'n', 'GC', 'GC_lo', 'GC_hi', 'расхождений', 'только_Н', 'только_Л1', 'Жаккар', 'dK_млрд', 'NPV_разногласие_млрд', 'исход'])
        w.writerows(tsv)
    return 0
if __name__ == '__main__':
    sys.exit(main())
