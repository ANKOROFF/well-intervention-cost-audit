"""Leave-one-type-out over the twelve operation types."""
import config as cfg
import csv
import pathlib
import sys
import time
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import ep19_crossruler as e19
import price_montecarlo as pmc
from sdt_analysis import отборы_линии, цепь
TOL, HALF, B = (0.0001, 0.084, 1000000000.0)
TYPES = pmc.TYPES
OUT_MD = ROOT / cfg.results('ep30_loto.md')
OUT_TSV = ROOT / cfg.out('ep30_loto.tsv')
PROG = ROOT / cfg.results('ep30_progress.txt')
ПРОЧЬ = []

def вывод(s=''):
    ПРОЧЬ.append(s)
    print(s, flush=True)

def якорь(имя, знач, ф='.4f'):
    вывод(f'ЯКОРЬ {имя} = {format(знач, ф)}')

def прогресс(s):
    with open(PROG, 'a', encoding='utf-8') as f:
        f.write(time.strftime('%H:%M:%S ') + s + '\n')

def исход(лет, lo, d):
    if лет < 9:
        return 'не определён'
    if lo <= 0:
        return 'W'
    return 'S' if abs(d) <= HALF else 'Д'

def множество(рез, линия=e19.LINE):
    return {i for г in рез['отборы'] for i in г['op_id'][г['отбор'][линия]].tolist()}

def главная() -> int:
    t0 = time.time()
    cap = pl.read_parquet(e19.CAP_ПП)
    тип = dict(zip(cap['op_id'].to_list(), cap['тип'].to_list()))
    cpx = dict(zip(cap['op_id'].to_list(), cap['capex'].to_list()))
    полн, _ = цепь('loto_full', cap)
    g0 = полн['линии'][e19.LINE]
    print(f"РЕГРЕССИЯ полная: GC {g0['GC']:.4f} [{g0['lo']:.4f}, {g0['hi']:.4f}], n {полн['n']}", flush=True)
    if not (abs(g0['GC'] - 0.2631) <= TOL and abs(g0['lo'] - 0.1772) <= TOL and (abs(g0['hi'] - 0.3453) <= TOL) and (полн['n'] == 5806) and (полн['лет'] == 9)):
        print('ОСТАНОВ: полная цепь не воспроизводит 0.2631')
        return 1
    S0, F0 = (множество(полн), множество(полн, 'факт инженеров'))
    строки, рез = ([], {})
    for t in TYPES:
        прогресс(f'loto {t}, {time.time() - t0:.0f} с')
        try:
            наш, _ = цепь(f'loto_{t}', cap.filter(pl.col('тип') != t))
            g = наш['линии'][e19.LINE]
            лет, n = (наш['лет'], наш['n'])
        except Exception as e:
            вывод(f'{t}: цепь не строится ({type(e).__name__}: {e})')
            строки.append([t, None])
            continue
        if t == 'ЗБС':
            ok = abs(g['GC'] - 0.2544) <= TOL and abs(g['lo'] - 0.1909) <= TOL and (abs(g['hi'] - 0.3161) <= TOL) and (n == 5533)
            print(f"РЕГРЕССИЯ без ЗБС: GC {g['GC']:.4f} [{g['lo']:.4f}, {g['hi']:.4f}], n {n}: {('сошлась' if ok else 'НЕ СОШЛАСЬ')}", flush=True)
            if not ok:
                print('ОСТАНОВ: исключение ЗБС не воспроизводит 0.2544')
                return 1
        d = g['GC'] - g0['GC']
        S = множество(наш)
        Sб = {i for i in S0 if тип[i] != t}
        sel_t = sum((тип[i] == t for i in S0))
        fact_t = sum((тип[i] == t for i in F0))
        расх = len(S ^ Sб)
        жак = len(S & Sб) / max(len(S | Sб), 1)
        dK = (sum((cpx[i] for i in S)) - sum((cpx[i] for i in Sб))) / B
        строки.append([t, dict(n=n, лет=лет, g=g, d=d, ис=исход(лет, g['lo'], d), sel_t=sel_t, fact_t=fact_t, жак=жак, расх=расх, dK=dK, ops_t=int((cap['тип'] == t).sum()))])
    вывод('# Б7.3. Leave-one-type-out по 12 типам цепи ППП (ep30_loto)')
    вывод('')
    вывод('Регрессия: полная цепь 0.2631 [0.1772, 0.3453] n 5806; без ЗБС 0.2544 [0.1909, 0.3161] n 5533. Для каждого типа цепь пересчитана (модель отклика и Л1 заново). ЛНЭК, ДП, ВПН, ПиП, РИР, ЛА малы (операций в таблице: см. столбец); их вывод не читается как «тип неважен».')
    вывод('')
    вывод('| исключён | операций типа (таблица) | n | лет | GC | lo | hi | dGC | исход | тип в отборе (полн.) | тип в факте (полн.) | Жаккар S_-t / S_полн\\t | расхождений | dK, млрд |')
    вывод('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    tsv, исходы = ([], {})
    for t, r in строки:
        if r is None:
            вывод(f'| {t} | | | | не определён | | | | не определён | | | | | |')
            tsv.append([t, '', '', '', '', '', '', '', 'не определён', '', '', '', '', ''])
            исходы[t] = 'не определён'
            continue
        g = r['g']
        исходы[t] = r['ис']
        вывод(f"| {t} | {r['ops_t']} | {r['n']} | {r['лет']} | {g['GC']:.4f} | {g['lo']:.4f} | {g['hi']:.4f} | {r['d']:+.4f} | {r['ис']} | {r['sel_t']} | {r['fact_t']} | {r['жак']:.4f} | {r['расх']} | {r['dK']:+.3f} |")
        tsv.append([t, r['ops_t'], r['n'], r['лет'], f"{g['GC']:.4f}", f"{g['lo']:.4f}", f"{g['hi']:.4f}", f"{r['d']:+.4f}", r['ис'], r['sel_t'], r['fact_t'], f"{r['жак']:.4f}", r['расх'], f"{r['dK']:+.4f}"])
        якорь(f'ep30 loto {t} GC', g['GC'])
        якорь(f'ep30 loto {t} GC_lo', g['lo'])
        якорь(f'ep30 loto {t} GC_hi', g['hi'])
        якорь(f'ep30 loto {t} dGC', r['d'], '+.4f')
        якорь(f'ep30 loto {t} n', r['n'], 'd')
        якорь(f'ep30 loto {t} Жаккар', r['жак'])
        якорь(f'ep30 loto {t} расхождений', r['расх'], 'd')
        якорь(f'ep30 loto {t} dK млрд', r['dK'], '+.4f')
        якорь(f'ep30 loto {t} операций типа в отборе нашей линии', r['sel_t'], 'd')
        якорь(f'ep30 loto {t} операций типа в факте', r['fact_t'], 'd')
    вывод('')
    W = [t for t, v in исходы.items() if v == 'W']
    N_ = [t for t, v in исходы.items() if v == 'не определён']
    Д = [t for t, v in исходы.items() if v == 'Д']
    if N_:
        код, текст = ('Т-3', f'GC не определён для {N_}; исход по остальным типам не меняется')
    elif W:
        код, текст = ('Т-2', f'W при исключении {W}: вывод зависит от типа')
    else:
        код, текст = ('Т-1', 'во всех 12 строках исход S или Д' + (f' (Д: {Д})' if Д else ''))
    вывод(f'**ИСХОД {код}**: {текст}. Исходы по типам: ' + '; '.join((f'{t} {v}' for t, v in исходы.items())) + '.')
    gcs = [r['g']['GC'] for _, r in строки if r]
    якорь('ep30 loto номер исхода (1 Т-1, 2 Т-2, 3 Т-3)', {'Т-1': 1, 'Т-2': 2, 'Т-3': 3}[код], 'd')
    якорь('ep30 loto GC min', min(gcs))
    якорь('ep30 loto GC max', max(gcs))
    якорь('ep30 loto время счёта с', time.time() - t0, '.0f')
    OUT_MD.write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
    with open(OUT_TSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['исключён', 'операций_типа', 'n', 'лет', 'GC', 'GC_lo', 'GC_hi', 'dGC', 'исход', 'тип_в_отборе', 'тип_в_факте', 'Жаккар', 'расхождений', 'dK_млрд'])
        w.writerows(tsv)
    return 0
if __name__ == '__main__':
    sys.exit(главная())
