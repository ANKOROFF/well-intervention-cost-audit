"""Library: pump-parser error -> CAPEX -> selection."""
import config as cfg
import argparse
import collections
import contextlib
import csv
import io
import json
import math
import pathlib
import re
import sys
import tempfile
import time
import numpy as np
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import capex_exante as cx
import dur_model as dm
import ep19_crossruler as e19
import price_montecarlo as pmc
from cost_physical import Prices, capex, pump_nomenclature
SEED = 20261014
TOL, B = (0.0001, 1000000000.0)
COL = 'y_насос_мощность'
OUT_MD = ROOT / cfg.results('ep30_parser_perturb.md')
OUT_TSV = ROOT / cfg.out('ep30_parser_perturb.tsv')
OUT_MAT = ROOT / cfg.out('ep30_parser_perturb_matrix.tsv')
OUT_DR = ROOT / cfg.out('ep30_parser_perturb_draws.tsv')
SETS = ROOT / cfg.data('interim/ep30_parser_perturb_sets.jsonl')
PROG = ROOT / cfg.results('ep30_progress.txt')
ПРИЧИНЫ = ['серия и одно число', 'диапазон подачи', 'не ЭЦН', 'подача > 1500', 'напор < 300', 'слитый числовой префикс', 'формат Q/H без серии', 'не номенклатура', 'прочее']
ОСОБАЯ = 'серия и одно число'
ДВОЙНИКИ = dict(zip('АВЕКМНОРСТХ', 'ABEKMHOPCTX'))
ПРОЧЬ = []

def вывод(s=''):
    ПРОЧЬ.append(s)
    print(s, flush=True)

def якорь(имя, знач, ф='.4f'):
    вывод(f'ЯКОРЬ {имя} = {format(знач, ф)}')

def прогресс(s):
    with open(PROG, 'a', encoding='utf-8') as f:
        f.write(time.strftime('%H:%M:%S ') + s + '\n')

def норм(raw):
    return ' '.join(str(raw or '').upper().split())

def форма_текста(raw):
    raw = норм(raw)
    if not raw:
        return ('F', 0, [])
    буквы = bool(re.search('[A-ZА-ЯЁ]', raw))
    числа = re.findall('(?<![A-ZА-ЯЁ\\d])\\d+(?![A-ZА-ЯЁ\\d])', raw)
    n = len(числа)
    if буквы and n == 3:
        return ('D', n, числа)
    if буквы and n == 2:
        return ('A', n, числа)
    if буквы and n == 1:
        return ('C', n, числа)
    if not буквы and n == 2:
        return ('B', n, числа)
    return ('E', n, числа)

def причина(raw, Q, H):
    форма, _, _ = форма_текста(raw)
    raw = норм(raw)
    буквенная = ''.join((ДВОЙНИКИ.get(ch, ch) for ch in re.sub('[^A-ZА-ЯЁ]', '', raw)))
    if форма == 'C':
        return 'серия и одно число'
    if форма == 'D':
        return 'диапазон подачи'
    if буквенная and (not any((k in буквенная for k in ('ЭЦH', 'ESP', 'REDA', 'DN')))):
        return 'не ЭЦН'
    if Q is not None and Q > 1500:
        return 'подача > 1500'
    if H is not None and H < 300:
        return 'напор < 300'
    if re.match('^\\d+[A-ZА-ЯЁ]', raw):
        return 'слитый числовой префикс'
    if форма == 'B':
        return 'формат Q/H без серии'
    if форма == 'E' and raw:
        return 'не номенклатура'
    return 'прочее'

def матрица():
    k = pl.read_parquet(ROOT / cfg.data('interim/parser_sample_key.parquet'))
    L = {}
    for line in open(ROOT / cfg.data('interim/parser_labels_final.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        L[r['id']] = r
    гр = collections.defaultdict(lambda: {'w': [], 'e': [], 'исх': []})
    согл_формы = [0, 0]
    for row in k.iter_rows(named=True):
        r = L[row['id']]
        pq, H, lab = (row['произв_м3сут'], row['напор_м'], r['после'])
        c = причина(row['после'], pq, H)
        lq = lab['Q']
        if lq is None and lab['Q_min'] is not None and (lab['Q_max'] is not None):
            lq_исх = (lab['Q_min'] + lab['Q_max']) / 2
        else:
            lq_исх = lq
        err = (pq is None) != (lq is None) or (pq is not None and lq is not None and (pq != lq))
        g = гр[c]
        g['w'].append(row['вес'])
        g['e'].append(err)
        if err and pq is not None:
            g['исх'].append(0.0 if lq_исх is None else lq_исх / pq)
        согл_формы[0] += форма_текста(row['после'])[0] == lab['форма']
        согл_формы[1] += 1
    res = {}
    for c in ПРИЧИНЫ:
        g = гр.get(c)
        if not g:
            res[c] = dict(n=0, k=0, p=0.0, neff=0.0, исх=[])
            continue
        w = np.array(g['w'])
        e = np.array(g['e'], float)
        p = float((w * e).sum() / w.sum())
        res[c] = dict(n=len(w), k=int(e.sum()), p=p, neff=float(w.sum() ** 2 / (w ** 2).sum()), исх=g['исх'])
    return (res, согл_формы)

def пул_данные(d):
    with contextlib.redirect_stdout(io.StringIO()):
        pn = pump_nomenclature()
    g = pl.read_parquet(ROOT / cfg.data('parquet/gtm_registry.parquet'))
    сырое = g['после.Насос'].cast(pl.Utf8).to_list()
    t = d.select('op_id', 'тип', COL).join(pn.select('op_id', 'насос_случай', 'произв_м3сут', 'напор_м'), on='op_id', how='left')
    ops = t['op_id'].to_list()
    Q = t['произв_м3сут'].to_list()
    H = t['напор_м'].to_list()
    б = [(s or '').startswith('б') for s in t['насос_случай'].to_list()]
    c = [причина(сырое[o], q, h) for o, q, h in zip(ops, Q, H)]
    qsep = []
    for o in ops:
        форма, _, ч = форма_текста(сырое[o])
        qsep.append(float(ч[0]) if форма == 'C' else None)
    return (t, ops, Q, H, б, c, qsep)

def в_границах(q, h):
    return 5 <= q <= 1500 and 300 <= h <= 3500

class Движок:

    def __init__(self):
        self.t0 = time.time()
        self.cap = pl.read_parquet(e19.CAP_ПП)
        self.ids = self.cap['op_id'].to_list()
        self.cpx = dict(zip(self.cap['op_id'].to_list(), self.cap['capex'].to_list()))
        with contextlib.redirect_stdout(io.StringIO()):
            self.t_all, _ = cx.volume_targets()
            self.d = cx.build_features(self.t_all)
            self.T = cx.ресурсы(True)
        self.vp = pl.read_parquet(cx.VOL_SRC)
        self.годы = sorted(self.d['beg'].dt.year().unique().drop_nulls().to_list())
        self.y0 = self.d[COL].to_numpy().astype(float)
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix='pp_'))
        self.t, self.ops, self.Q, self.H, self.б, self.c, self.qsep = пул_данные(self.d)
        self.mode = None
        self.pred0 = None

    def прогноз(self, y):
        dd = self.d.with_columns(pl.Series(COL, y))
        куски = []
        for Y in self.годы:
            te = dd.filter(pl.col('beg').dt.year() == Y)
            tr = dd.filter(pl.col('end').dt.year() < Y)
            r = te.select('op_id')
            if tr.height >= dm.MIN_TRAIN:
                p = np.clip(dm.fit(tr, COL).predict(te.select(dm.FEATS).to_pandas()), 0, None)
            elif tr.height:
                p = dm.type_median(tr, te, COL)
            else:
                p = np.full(te.height, np.nan)
            куски.append(r.with_columns(pl.Series('пред', p)))
        return pl.concat(куски)

    def подготовить_базу(self):
        self.pred0 = self.прогноз(self.y0)
        файл = self.vp.select('op_id', 'насос_мощность_пред').join(self.pred0, on='op_id', how='left')
        a = файл['насос_мощность_пред'].to_numpy().astype(float)
        b = файл['пред'].to_numpy().astype(float)
        ok = ~np.isnan(a) & ~np.isnan(b)
        отн = float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(a[ok]), 1.0)))
        self.mode = 'прямой' if отн <= 1e-06 and (np.isnan(a) == np.isnan(b)).all() else 'поправка'
        вывод(f'Пересчёт базового прогноза насоса против capex_vol_pred: макс. отн. расхождение {отн:.2e}; режим «{self.mode}» (п. 8.2).')
        якорь('ep30 parser_perturb расхождение базового прогноза', отн, '.3e')

    def цепь_по_y(self, метка, y):
        pred = self.прогноз(y)
        if self.mode == 'прямой':
            col = pred
        else:
            col = self.pred0.rename({'пред': 'пред0'}).join(pred, on='op_id').join(self.vp.select('op_id', 'насос_мощность_пред'), on='op_id').select('op_id', пред=(pl.col('насос_мощность_пред') + pl.col('пред') - pl.col('пред0')).clip(0, None))
        vp = self.vp.drop('насос_мощность_пред').join(col.rename({'пред': 'насос_мощность_пред'}), on='op_id', how='left')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T = cx.подменить_объёмы(self.T, vp).filter(pl.col('op_id').is_in(self.ids))
            c = capex(T, Prices())
            pa = self.tmp / f'{метка}.parquet'
            c.write_parquet(pa)
            наш = e19.прогон_exante(метка, e19.параметры(str(pa)))
        pa.unlink()
        g = наш['линии'][e19.LINE]
        S = sorted({i for г in наш['отборы'] for i in г['op_id'][г['отбор'][e19.LINE]].tolist()})
        return dict(GC=g['GC'], lo=g['lo'], hi=g['hi'], n=наш['n'], лет=наш['лет'], S=S)

def возмущение(дв, мат, i, без_C):
    rng = np.random.default_rng([SEED, i])
    p = {}
    for c in ПРИЧИНЫ:
        m = мат[c]
        p[c] = float(rng.beta(m['p'] * m['neff'] + 0.5, (1 - m['p']) * m['neff'] + 0.5)) if m['n'] else 0.0
    N = len(дв.ops)
    u, ur, uh = (rng.random(N), rng.random(N), rng.random(N))
    y = дв.y0.copy()
    k = 0
    for j in range(N):
        c = дв.c[j]
        q, h = (дв.Q[j], дв.H[j])
        if not дв.б[j] or q is None or h is None or (u[j] >= p[c]):
            continue
        if c == ОСОБАЯ:
            if без_C or дв.qsep[j] is None:
                continue
            hs = дв.Hпул.get(дв.t['тип'][j]) or дв.Hпул['*']
            q2, h2 = (дв.qsep[j], hs[int(uh[j] * len(hs))])
        else:
            o = мат[c]['исх']
            if not o:
                continue
            q2, h2 = (q * o[int(ur[j] * len(o))], h)
        y[j] = q2 * h2 if в_границах(q2, h2) else 0.0
        k += 1
    return (y, k)

def главная() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--n-noC', type=int, default=250)
    a = ap.parse_args()
    t0 = time.time()
    мат, согл = матрица()
    дв = Движок()
    ф = np.array([q * h if б and q is not None and (h is not None) and в_границах(q, h) else 0.0 for q, h, б in zip(дв.Q, дв.H, дв.б)])
    расхождение_y = float(np.max(np.abs(ф - дв.y0)))
    if расхождение_y > 1e-06:
        print(f'ОСТАНОВ: базовый таргет не равен q*h парсера, макс. расхождение {расхождение_y}')
        return 1
    дв.Hпул = collections.defaultdict(list)
    for j, c in enumerate(дв.c):
        if c == 'прочее' and дв.б[j] and (дв.H[j] is not None):
            дв.Hпул[дв.t['тип'][j]].append(дв.H[j])
            дв.Hпул['*'].append(дв.H[j])
    дв.подготовить_базу()
    готово = {}
    if SETS.exists():
        for line in open(SETS, encoding='utf-8'):
            r = json.loads(line)
            готово[r['i'], r['v']] = r

    def считать(i, v):
        if (i, v) in готово:
            return готово[i, v]
        if i == 0:
            y, k = (дв.y0, 0)
        else:
            y, k = возмущение(дв, мат, i, v == 'безC')
        r = дв.цепь_по_y(f'pp{i}{v}', y)
        r.update(i=i, v=v, k=k)
        with open(SETS, 'a', encoding='utf-8') as f:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        готово[i, v] = r
        return r
    r0 = считать(0, 'полн')
    print(f"РЕГРЕССИЯ: GC {r0['GC']:.4f} [{r0['lo']:.4f}, {r0['hi']:.4f}], n {r0['n']}", flush=True)
    if not (abs(r0['GC'] - 0.2631) <= TOL and abs(r0['lo'] - 0.1772) <= TOL and (abs(r0['hi'] - 0.3453) <= TOL) and (r0['n'] == 5806) and (r0['лет'] == 9)):
        print('ОСТАНОВ: розыгрыш 0 не воспроизводит 0.2631')
        return 1
    S0 = set(r0['S'])
    K0 = sum((дв.cpx[i] for i in S0))
    for v, n in (('полн', a.n), ('безC', a.n_noC)):
        for i in range(1, n + 1):
            считать(i, v)
            if i % 10 == 0:
                прогресс(f'parser_perturb {v}: {i} из {n}, {time.time() - t0:.0f} с')
    вывод('# Б7.6. Ошибка парсера подачи насоса -> CAPEX -> отбор (ep30_parser_perturb)')
    вывод('')
    вывод(f"Регрессия: розыгрыш 0 (без возмущения) GC {r0['GC']:.4f} [{r0['lo']:.4f}, {r0['hi']:.4f}], n {r0['n']}. Правила: `docs/epic/b1_reading_rules.md` п. 8. Размер возмущения задан выборкой 450 с автоматической разметкой (согласие парсера с разметкой, не истина); исход а/б/в, строка `до.Насос` и потерянные числа не возмущаются; влияние идёт через таргет прогноза объёма, не через фактический CAPEX той же операции.")
    вывод('')
    вывод(f'Согласие формы по тексту с формой метки на выборке (450): {согл[0]} из {согл[1]} ({согл[0] / согл[1]:.3f}); причины определяются по тексту (п. 8.3).')
    вывод('')
    вывод('## Матрица ошибок подачи по причинам (выборка)')
    вывод('')
    вывод('| причина | строк | с ошибкой подачи | доля взвеш. | n эфф. | исходов ошибки | операций пула случая б с подачей |')
    вывод('|---|---|---|---|---|---|---|')
    пул_c = collections.Counter((c for c, б, q, h in zip(дв.c, дв.б, дв.Q, дв.H) if б and q is not None and (h is not None)))
    строки_мат = []
    for c in ПРИЧИНЫ:
        m = мат[c]
        вывод(f"| {c} | {m['n']} | {m['k']} | {m['p']:.4f} | {m['neff']:.1f} | {len(m['исх'])} | {пул_c[c]} |")
        строки_мат.append([c, m['n'], m['k'], f"{m['p']:.4f}", f"{m['neff']:.1f}", len(m['исх']), пул_c[c]])
        якорь(f'ep30 parser_perturb матрица {c} доля ошибки', m['p'])
        якорь(f'ep30 parser_perturb матрица {c} операций пула', пул_c[c], 'd')
    вывод('')
    ref = []
    with open(ROOT / cfg.out('price_montecarlo.tsv'), encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            ref.append(float(r['jaccard']))
    j_цен = float(np.median(ref))
    итоги, tsv, драфт = ({}, [], [])
    for v, имя in (('полн', 'с источником C (вердикт)'), ('безC', 'без источника C (первые жребии)')):
        R = [готово[i, v] for i in sorted((i for i, vv in готово if vv == v and i > 0))]
        J = np.array([len(set(r['S']) & S0) / max(len(set(r['S']) | S0), 1) for r in R])
        X = np.array([len(set(r['S']) ^ S0) for r in R])
        dG = np.array([r['GC'] - r0['GC'] for r in R])
        GC = np.array([r['GC'] for r in R])
        dK = np.array([(sum((дв.cpx[i] for i in r['S'])) - K0) / B for r in R])
        lo_pos = float(np.mean([r['lo'] > 0 for r in R]))
        kk = np.array([r['k'] for r in R])
        Jmed = float(np.median(J))
        ис = 'Р-1' if Jmed >= 0.95 and lo_pos >= 0.95 else 'Р-2' if Jmed >= 0.8 else 'Р-3'
        итоги[v] = ис
        q = lambda x, p: float(np.percentile(x, p))
        keep = collections.Counter((i for r in R for i in r['S'] if i in S0))
        доли = np.array([keep[i] / len(R) for i in S0])
        вывод(f'## Розыгрышей {len(R)}, {имя}')
        вывод('')
        вывод('| величина | медиана | 2.5 % | 97.5 % |')
        вывод('|---|---|---|---|')
        for nm, x, ф in (('Жаккар с базовым отбором', J, '.4f'), ('изменённых решений |S_d Δ S_0|', X, '.1f'), ('GC', GC, '.4f'), ('ΔGC к 0.2631', dG, '+.4f'), ('ΔK, млрд руб (базовый CAPEX)', dK, '+.4f'), ('возмущённых операций', kk, '.1f')):
            вывод(f'| {nm} | {format(q(x, 50), ф)} | {format(q(x, 2.5), ф)} | {format(q(x, 97.5), ф)} |')
            tsv.append([v, nm, len(R), format(q(x, 50), ф), format(q(x, 2.5), ф), format(q(x, 97.5), ф)])
        вывод('')
        вывод(f'Доля розыгрышей с нижней границей GC > 0: {lo_pos:.4f}. Операций базового отбора: {len(S0)}; сохраняются в отборе не менее чем в 95 % розыгрышей: {int(np.sum(доли >= 0.95))}, менее чем в 50 %: {int(np.sum(доли < 0.5))}. Медиана Жаккара розыгрышей цен (price_montecarlo.tsv, для сравнения, не для исхода): {j_цен:.4f}.')
        вывод('')
        вывод(f'**ИСХОД {ис}**.')
        вывод('')
        p_ = 'ep30 parser_perturb ' + ('' if v == 'полн' else 'безC ')
        якорь(p_ + 'розыгрышей', len(R), 'd')
        якорь(p_ + 'Жаккар медиана', Jmed)
        якорь(p_ + 'Жаккар 2.5', q(J, 2.5))
        якорь(p_ + 'Жаккар 97.5', q(J, 97.5))
        якорь(p_ + 'изменённых решений медиана', q(X, 50), '.1f')
        якорь(p_ + 'изменённых решений 2.5', q(X, 2.5), '.1f')
        якорь(p_ + 'изменённых решений 97.5', q(X, 97.5), '.1f')
        якорь(p_ + 'GC медиана', q(GC, 50))
        якорь(p_ + 'dGC медиана', q(dG, 50), '+.4f')
        якорь(p_ + 'dGC 2.5', q(dG, 2.5), '+.4f')
        якорь(p_ + 'dGC 97.5', q(dG, 97.5), '+.4f')
        якорь(p_ + 'dK млрд медиана', q(dK, 50), '+.4f')
        якорь(p_ + 'доля розыгрышей GC нижняя > 0', lo_pos)
        якорь(p_ + 'возмущённых операций медиана', q(kk, 50), '.1f')
        якорь(p_ + 'операций базового отбора в >= 95 % розыгрышей', int(np.sum(доли >= 0.95)), 'd')
        якорь(p_ + 'операций базового отбора в < 50 % розыгрышей', int(np.sum(доли < 0.5)), 'd')
        якорь(p_ + 'номер исхода (1 Р-1, 2 Р-2, 3 Р-3)', int(ис[-1]), 'd')
        драфт += [[v, r['i'], f"{r['GC']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", r['k'], f"{len(set(r['S']) & S0) / max(len(set(r['S']) | S0), 1):.4f}", len(set(r['S']) ^ S0)] for r in R]
    вывод(f"**ИТОГО (вердикт по варианту с источником C, п. 8.8)**: {итоги['полн']}; без C: {итоги['безC']}." + ('' if итоги['полн'] == итоги['безC'] else ' Исходы РАЗОШЛИСЬ.'))
    якорь('ep30 parser_perturb время счёта с', time.time() - t0, '.0f')
    OUT_MD.write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
    for путь, шапка, строки in ((OUT_TSV, ['вариант', 'величина', 'розыгрышей', 'медиана', 'q2.5', 'q97.5'], tsv), (OUT_MAT, ['причина', 'строк', 'с_ошибкой', 'доля_взвеш', 'n_эфф', 'исходов', 'операций_пула'], строки_мат), (OUT_DR, ['вариант', 'розыгрыш', 'GC', 'GC_lo', 'GC_hi', 'возмущено', 'Жаккар', 'расхождений'], драфт)):
        with open(путь, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter='\t', lineterminator='\n')
            w.writerow(шапка)
            w.writerows(строки)
    return 0
if __name__ == '__main__':
    sys.exit(главная())
