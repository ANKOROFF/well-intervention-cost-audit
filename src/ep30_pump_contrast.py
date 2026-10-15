"""Contrast: parsed pump power against the type median."""
import config as cfg
import contextlib
import csv
import io
import pathlib
import sys
import numpy as np
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import capex_exante as cx
import dur_model as dm
import ep19_crossruler as e19
import ep30_parser_perturb as pp
import stats
from cost_physical import Prices, capex
B = 1000000000.0
TOL = 0.0001
J_ПОРОГ, DGC_ПОРОГ = (0.95, 0.02)
COL = pp.COL
OUT_MD = ROOT / cfg.results('ep30_pump_contrast.md')
OUT_TSV = ROOT / cfg.out('ep30_pump_contrast.tsv')
ПРОЧЬ = []

def вывод(s=''):
    ПРОЧЬ.append(s)
    print(s)

def якорь(имя, знач, ф='.4f'):
    вывод(f'ЯКОРЬ {имя} = {format(знач, ф)}')

def по_годам(дв, f):
    куски = []
    for Y in дв.годы:
        te = дв.d.filter(pl.col('beg').dt.year() == Y)
        tr = дв.d.filter(pl.col('end').dt.year() < Y)
        p = f(tr, te) if tr.height else np.full(te.height, np.nan)
        куски.append(te.select('op_id').with_columns(pl.Series('пред', np.asarray(p, dtype=float))))
    return pl.concat(куски)

def вариант_А2(дв):
    return по_годам(дв, lambda tr, te: dm.type_median(tr, te, COL))

def вариант_А1(дв):

    def f(tr, te):
        if tr.height < dm.MIN_TRAIN:
            return dm.type_median(tr, te, COL)
        med = tr.group_by('тип').agg(m=pl.col(COL).median())
        tr2 = tr.join(med, on='тип', how='left').with_columns(pl.col('m').alias(COL)).drop('m')
        return np.clip(dm.fit(tr2, COL).predict(te.select(dm.FEATS).to_pandas()), 0, None)
    return по_годам(дв, f)

def вариант_Б(дв):
    return дв.d.select('op_id', пред=pl.col(COL))

def цепь(дв, метка, col):
    head = дв.vp.select('op_id', 'насос_мощность_пред')
    c = head.join(col, on='op_id', how='left').select('op_id', насос_мощность_пред=pl.coalesce(pl.col('пред'), pl.col('насос_мощность_пред')))
    пропусков = int(head.join(col, on='op_id', how='left')['пред'].is_null().sum())
    пропусков_цепи = int(head.filter(pl.col('op_id').is_in(дв.ids)).join(col, on='op_id', how='left')['пред'].is_null().sum())
    vp = дв.vp.drop('насос_мощность_пред').join(c, on='op_id', how='left')
    with contextlib.redirect_stdout(io.StringIO()):
        T = cx.подменить_объёмы(дв.T, vp).filter(pl.col('op_id').is_in(дв.ids))
        cap = capex(T, Prices())
        pa = дв.tmp / f'{метка}.parquet'
        cap.write_parquet(pa)
        наш = e19.прогон_exante(метка, e19.параметры(str(pa)))
    pa.unlink()
    g = наш['линии'][e19.LINE]
    S = {(г_i, i) for г_i, г in enumerate(наш['отборы']) for i in г['op_id'][г['отбор'][e19.LINE]].tolist()}
    return dict(GC=g['GC'], lo=g['lo'], hi=g['hi'], n=наш['n'], лет=наш['лет'], знам=наш['разрыв'], S=S, capex=float(cap['capex'].sum()), насос_c=float(cap['c_насос'].sum()), насос_пол=int((cap['насос_мощность'].fill_null(0) > 0).sum()), пропусков=пропусков_цепи, всего_пропусков=пропусков)

def main() -> int:
    дв = pp.Движок()
    with contextlib.redirect_stdout(io.StringIO()):
        d, _ = stats.build(e19.параметры(e19.CAP_ПП))
    npv_н = dict(zip(d['op_id'].to_list(), d['npv'].to_list()))
    голова = цепь(дв, 'голова', pl.DataFrame({'op_id': дв.vp['op_id'], 'пред': дв.vp['насос_мощность_пред']}))
    print(f"РЕГРЕССИЯ: GC {голова['GC']:.4f} [{голова['lo']:.4f}, {голова['hi']:.4f}], n {голова['n']}, лет {голова['лет']}", flush=True)
    if not (abs(голова['GC'] - 0.2631) <= TOL and abs(голова['lo'] - 0.1772) <= TOL and (abs(голова['hi'] - 0.3453) <= TOL) and (голова['n'] == 5806) and (голова['лет'] == 9)):
        print('ОСТАНОВ: головная цепь не воспроизводит 0.2631')
        return 1
    S0 = голова['S']
    cap0 = sum(дв.cpx.values())
    if abs(голова['capex'] - cap0) > 1e-06 * cap0:
        print('ОСТАНОВ: CAPEX головной таблицы не равен CAP_ПП')
        return 1
    npv0 = sum((npv_н[i] for _, i in S0))
    рез = {}
    for имя, f in (('А2', вариант_А2), ('А1', вариант_А1), ('Б', вариант_Б)):
        рез[имя] = цепь(дв, f'v{имя}', f(дв))
        print(f"{имя}: GC {рез[имя]['GC']:.4f}", flush=True)
    вывод('# Б12. Контраст К-П: разобранный насос против медианы типа (ep30_pump_contrast)')
    вывод('')
    вывод('Регрессия: головная цепь GC 0.2631 [0.1772, 0.3453], n 5806, лет 9, CAPEX = CAP_ПП. Правила: `docs/epic/b1_reading_rules.md` п. 11. Подменяется только прогноз мощности насоса; знаменатель GC пересчитан цепью варианта. ΔGC читается по точечной разности (ДИ разности не строится). Оценка и оракул - на прогнозном CAPEX цепи варианта; результат описывает чувствительность вывода, не верность разбора.')
    вывод('')
    вывод('| вариант | GC [интервал] | знаменатель млрд | ΔGC | Жаккар | |SΔS0| | ΔCAPEX млрд (%) | ΔNPV_Н млрд | насос > 0 | доля c_насос | пропусков прогноза |')
    вывод('|---|---|---|---|---|---|---|---|---|---|---|')
    ис, tsv = ({}, [])
    строки = [('голова', голова)] + list(рез.items())
    for имя, r in строки:
        J = len(r['S'] & S0) / max(len(r['S'] | S0), 1)
        X = len(r['S'] ^ S0)
        dG = r['GC'] - голова['GC']
        dC = (r['capex'] - голова['capex']) / B
        dN = (sum((npv_н[i] for _, i in r['S'])) - npv0) / B
        доля = r['насос_c'] / r['capex']
        ис[имя] = 'П-1' if J >= J_ПОРОГ and abs(dG) <= DGC_ПОРОГ else 'П-2'
        вывод(f"| {имя} | {r['GC']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] | {(r['знам'] / 1 if r['знам'] < 1000000.0 else r['знам'] / B):.4f} | {dG:+.4f} | {J:.4f} | {X} | {dC:+.4f} ({100 * dC * B / голова['capex']:+.2f}) | {dN:+.4f} | {r['насос_пол']} | {доля:.4f} | {r['пропусков']} |")
        tsv.append([имя, f"{r['GC']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", f'{dG:+.4f}', f'{J:.4f}', X, f'{dC:+.4f}', f'{dN:+.4f}', r['насос_пол'], f'{доля:.4f}'])
        if имя != 'голова':
            якорь(f'ep30 pump_contrast {имя} GC', r['GC'])
            якорь(f'ep30 pump_contrast {имя} GC_lo', r['lo'])
            якорь(f'ep30 pump_contrast {имя} GC_hi', r['hi'])
            якорь(f'ep30 pump_contrast {имя} dGC', dG, '+.4f')
            якорь(f'ep30 pump_contrast {имя} Жаккар', J)
            якорь(f'ep30 pump_contrast {имя} изменённых решений', X, 'd')
            якорь(f'ep30 pump_contrast {имя} dCAPEX млрд', dC, '+.4f')
            якорь(f'ep30 pump_contrast {имя} dCAPEX %', 100 * dC * B / голова['capex'], '+.2f')
            якорь(f'ep30 pump_contrast {имя} dNPV_Н млрд', dN, '+.4f')
            якорь(f'ep30 pump_contrast {имя} номер исхода (1 П-1, 2 П-2)', int(ис[имя][-1]), 'd')
        else:
            якорь('ep30 pump_contrast голова насос > 0', r['насос_пол'], 'd')
            якорь('ep30 pump_contrast голова доля c_насос', доля)
            якорь('ep30 pump_contrast голова знаменатель', r['знам'] if r['знам'] < 1000000.0 else r['знам'] / B)
    вывод('')
    вывод(f"**ИСХОД (вердикт по А2, п. 11.5): {ис['А2']}.** А1: {ис['А1']}; Б (ex-post, справочно): {ис['Б']}." + ('' if ис['А2'] == ис['А1'] == ис['Б'] else ' Классы вариантов РАЗОШЛИСЬ.'))
    OUT_MD.write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
    with open(OUT_TSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['вариант', 'GC', 'GC_lo', 'GC_hi', 'dGC', 'Жаккар', 'изменённых_решений', 'dCAPEX_млрд', 'dNPV_Н_млрд', 'насос_положит', 'доля_c_насос'])
        w.writerows(tsv)
    return 0
if __name__ == '__main__':
    sys.exit(main())
