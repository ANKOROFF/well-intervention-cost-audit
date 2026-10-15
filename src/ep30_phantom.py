"""Spurious proppant in the volume forecast: diagnosis and corrections."""
import config as cfg
import contextlib
import csv
import io
import pathlib
import sys
import time
import lightgbm as lgb
import numpy as np
import polars as pl
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import capex_exante as cx
import dur_model as dm
import ep19_crossruler as e19
import price_montecarlo as pmc
from cost_physical import Prices, capex
from ep06_ablation import _цены
from sdt_analysis import цепь
TOL, HALF, B = (0.0001, 0.084, 1000000000.0)
LINE = e19.LINE
ТИПЫ = pmc.TYPES
ФАНТ_ТИПЫ = ['ОПЗ', 'ОПТ', 'ППР', 'ВБД']
МАТ5 = ['проппант_т', 'гель_м3', 'кислота_м3', 'азот_т', 'цемент_м3']
ДОЛЯ_ОПТ0, ПОРОГ_ОПТ, ПОРОГ_J, ПОРОГ_ФАКТ, МИН_ОП = (68.5, 10.0, 0.85, 0.05, 10)
ПОРОГ_ПРОППАНТ = 1748
OUT_MD = ROOT / cfg.results('ep30_phantom.md')
OUT_TSV = ROOT / cfg.out('ep30_phantom.tsv')
ПРОЧЬ = []

def вывод(s=''):
    ПРОЧЬ.append(s)
    print(s, flush=True)

def якорь(имя, знач, ф='.4f'):
    вывод(f'ЯКОРЬ {имя} = {format(знач, ф)}')

def тихо(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)

def auc(y, p):
    y = np.asarray(y) > 0
    if y.all() or not y.any():
        return float('nan')
    r = np.argsort(np.argsort(p, kind='stable'), kind='stable') + 1.0
    for v in np.unique(p):
        m = p == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n0 = (y.sum(), (~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def классификатор(tr, col):
    y = (tr[col].to_numpy() > 0).astype(int)
    if y.min() == y.max():
        return (None, float(y[0]))
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42, n_jobs=1)
    m.fit(tr.select(dm.FEATS).to_pandas(), y, categorical_feature=dm.CAT)
    return (m, None)

def прогноз_в1(d, vp):
    куски, метрики = ([], [])
    for Y in sorted(d['beg'].dt.year().unique().drop_nulls().to_list()):
        te = d.filter(pl.col('beg').dt.year() == Y)
        tr = d.filter(pl.col('end').dt.year() < Y)
        r = te.select('op_id', 'тип')
        if tr.height < dm.MIN_TRAIN:
            куски.append(r.select('op_id'))
            continue
        Xte = te.select(dm.FEATS).to_pandas()
        for q in МАТ5:
            col = f'y_{q}'
            m, const = классификатор(tr, col)
            p = np.full(te.height, const) if m is None else m.predict_proba(Xte)[:, 1]
            pos = tr.filter(pl.col(col) > 0)
            cond = np.clip(dm.fit(pos, col).predict(Xte), 0, None) if pos.height >= 2 else np.zeros(te.height)
            r = r.with_columns(pl.Series(f'{q}_пред', np.where(p >= 0.5, cond, 0.0)))
            if q == 'проппант_т':
                метрики.append((Y, te.height, auc(te[col].to_numpy(), p), float(((p >= 0.5) == (te[col].to_numpy() > 0)).mean())))
        s = tr.filter((pl.col('y_проппант_rcp_т') > 0) & (pl.col('y_проппант_т') > 0)).group_by('тип').agg(доля=(pl.col('y_проппант_rcp_т') / pl.col('y_проппант_т')).median())
        r = r.join(s, on='тип', how='left').with_columns(**{'проппант_rcp_т_пред': pl.col('проппант_т_пред') * pl.col('доля').fill_null(0.0)}).drop('доля', 'тип')
        куски.append(r)
    new = pl.concat(куски, how='diagonal')
    cols = [f'{q}_пред' for q in МАТ5] + ['проппант_rcp_т_пред']
    out = vp.join(new.select('op_id', *[pl.col(c).alias(c + '_н') for c in cols if c in new.columns]), on='op_id', how='left')
    for c in cols:
        out = out.with_columns(pl.coalesce(pl.col(c + '_н'), pl.col(c)).alias(c)).drop(c + '_н')
    return (out, метрики)

def прогноз_в2(d, vp):
    zero_ops, rows = ([], [])
    for Y in sorted(d['beg'].dt.year().unique().drop_nulls().to_list()):
        te = d.filter(pl.col('beg').dt.year() == Y)
        tr = d.filter(pl.col('end').dt.year() < Y)
        if not tr.height:
            continue
        g = tr.group_by('тип').agg(n=pl.len(), доля=(pl.col('y_проппант_т') > 0).mean())
        обн = g.filter((pl.col('n') >= МИН_ОП) & (pl.col('доля') < ПОРОГ_ФАКТ))['тип'].to_list()
        zero_ops += te.filter(pl.col('тип').is_in(обн))['op_id'].to_list()
        rows.append((Y, len(обн)))
    z = pl.col('op_id').is_in(zero_ops)
    out = vp.with_columns(**{c: pl.when(z).then(0.0).otherwise(pl.col(c)) for c in ['проппант_т_пред', 'проппант_rcp_т_пред', 'гель_м3_пред']})
    return (out, len(set(zero_ops)))

def главная() -> int:
    t0 = time.time()
    cap = pl.read_parquet(e19.CAP_ПП)
    ids = cap['op_id'].to_list()
    тип = dict(zip(cap['op_id'].to_list(), cap['тип'].to_list()))
    факт = pl.read_parquet(ROOT / cfg.data('interim/capex_physical.parquet')).filter(pl.col('op_id').is_in(ids))
    t_all, _ = cx.volume_targets()
    d = тихо(cx.build_features, t_all)
    T = тихо(cx.ресурсы, True)
    vp = pl.read_parquet(cx.VOL_SRC)

    def cap_из(vp_):
        with contextlib.redirect_stdout(io.StringIO()):
            return capex(cx.подменить_объёмы(T, vp_).filter(pl.col('op_id').is_in(ids)), Prices())
    c3 = cap_из(vp)
    j = c3.select('op_id', a='capex').join(cap.select('op_id', b='capex'), on='op_id')
    отн = float(((j['a'] - j['b']).abs() / j['b'].abs().clip(1.0, None)).max())
    print(f'РЕГРЕССИЯ пересборки CAPEX В3: макс. отн. расхождение {отн:.2e}, строк {j.height} из {len(ids)}', flush=True)
    if отн > 1e-09 or j.height != len(ids):
        print('ОСТАНОВ: пересборка CAPEX не воспроизводит capex_physical_ПП')
        return 1
    вывод('# Б10. Фантомный проппант в прогнозе объёмов (ep30_phantom)')
    вывод('')
    вывод(f'Правило чтения: docs/epic/b1_reading_rules.md, п. 9 (записано до расчёта). Регрессия пересборки CAPEX В3: макс. отн. расхождение {отн:.1e}.')
    пп, фф = (cap, факт)
    n_пп, n_ф = (int((пп['проппант_т'].fill_null(0) > 0).sum()), int((фф['проппант_т'].fill_null(0) > 0).sum()))
    ph = пп.filter((pl.col('проппант_т') > 0) & pl.col('тип').is_in(ФАНТ_ТИПЫ))
    ph_n = {t: int((ph['тип'] == t).sum()) for t in ФАНТ_ТИПЫ}
    rcp_пп = int(((пп['проппант_т'] > 0) & (пп['проппант_rcp_т'] > пп['проппант_т'])).sum())
    rcp_ф = int(((фф['проппант_т'].fill_null(0) > 0) & (фф['проппант_rcp_т'].fill_null(0) > фф['проппант_т'].fill_null(0))).sum())
    деньги = float(ph.select((pl.col('c_проппант') + pl.col('c_гель') + pl.col('c_флот_грп')).sum()).item())
    всего = float(пп['capex'].sum())
    ok_цел = (n_пп, n_ф, rcp_пп, rcp_ф) == (4777, 1398, 1955, 0) and ph_n == {'ОПЗ': 1082, 'ОПТ': 945, 'ППР': 754, 'ВБД': 272}
    вывод('')
    вывод(f"Подтверждение аудита (таблица {пп.height} операций цепи): ненулевой проппант прогноз {n_пп} / факт {n_ф}; ОПЗ {ph_n['ОПЗ']}, ОПТ {ph_n['ОПТ']}, ППР {ph_n['ППР']}, ВБД {ph_n['ВБД']}; RCP > общего прогноз {rcp_пп} / факт {rcp_ф}: {('целые числа совпали' if ok_цел else 'ЦЕЛЫЕ ЧИСЛА НЕ СОВПАЛИ')}.")
    вывод(f'Сумма c_проппант + c_гель + c_флот_грп на этих {sum(ph_n.values())} операциях: {деньги / B:.3f} млрд руб = {100 * деньги / всего:.2f} % ex-ante CAPEX {всего / B:.2f} млрд. Цифры аудита <withheld> и 3.3 % НЕ воспроизведены ни этим, ни другими перебранными определениями (сообщено lead; в исходы не входят).')
    якорь('ep30 phantom ненулевой проппант прогноз', n_пп, 'd')
    якорь('ep30 phantom ненулевой проппант факт', n_ф, 'd')
    for t in ФАНТ_ТИПЫ:
        якорь(f'ep30 phantom {t} операций с проппантом', ph_n[t], 'd')
    якорь('ep30 phantom RCP больше общего прогноз', rcp_пп, 'd')
    якорь('ep30 phantom RCP больше общего факт', rcp_ф, 'd')
    пг = float(ph.select((pl.col('c_проппант') + pl.col('c_гель')).sum()).item())
    вывод(f'Только проппант + гель: {пг / B:.3f} млрд = {100 * пг / всего:.2f} %. По решению lead расхождение суммы с цифрой аудита (<withheld>, 3.3 %) допущено к расчёту (п. 9.1, пояснение); целые числа совпали.')
    якорь('ep30 phantom сумма проппант+гель млрд', пг / B)
    якорь('ep30 phantom доля проппант+гель в ex-ante CAPEX', пг / всего)
    якорь('ep30 phantom сумма проппант+гель+флот млрд', деньги / B)
    якорь('ep30 phantom доля в ex-ante CAPEX', деньги / всего)
    if not ok_цел:
        print('ОСТАНОВ: целые числа аудита не воспроизведены')
        return 1
    вывод('')
    вывод('| тип | операций | факт: проппант>0 | прогноз: проппант>0 | факт: гель>0 | прогноз: гель>0 | факт: кислота>0 | прогноз: кислота>0 | факт: азот>0 | прогноз: азот>0 | факт: цемент>0 | прогноз: цемент>0 | доля нулей таргета проппанта в обучении (все годы / до 2024) |')
    вывод('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    d24 = d.filter(pl.col('end').dt.year() < 2024)
    fj = фф.select('op_id', *[pl.col(q).fill_null(0) for q in МАТ5])
    for t in ТИПЫ:
        a, b = (пп.filter(pl.col('тип') == t), фф.filter(pl.col('тип') == t))
        n = a.height
        if n == 0:
            continue
        дл = d.filter(pl.col('тип') == t)
        д24 = d24.filter(pl.col('тип') == t)
        z_all = float((дл['y_проппант_т'] == 0).mean()) if дл.height else float('nan')
        z_24 = float((д24['y_проппант_т'] == 0).mean()) if д24.height else float('nan')
        cells = []
        for q in МАТ5:
            cells += [f'{(b[q].fill_null(0) > 0).sum() / n:.3f}', f'{(a[q].fill_null(0) > 0).sum() / n:.3f}']
        вывод(f'| {t} | {n} | ' + ' | '.join(cells) + f' | {z_all:.3f} / {z_24:.3f} |')
    вывод('')
    вывод(f"Причина (по числам таблицы): модель объёма регрессионная (L1, `dm.fit`), классификатора «есть материал» нет, и прогноз клипуется снизу нулём; у типов ОПЗ/ОПТ/ППР/ВБД доля нулей таргета проппанта в обучении у них 1.000 (все завершённые операции без проппанта, столбец справа), но прогноз проппанта ненулевой у {ph_n['ОПЗ']}/{ph_n['ОПТ']}/{ph_n['ППР']}/{ph_n['ВБД']} операций. RCP и общий проппант предсказываются двумя независимыми моделями, отсюда RCP > общего у {rcp_пп} операций.")
    вывод('')
    v1, метр = прогноз_в1(d, vp)
    v2, n_обн = прогноз_в2(d, vp)
    caps = {'В3': cap, 'В1': cap_из(v1), 'В2': cap_из(v2)}
    рез = {}
    for k, c in caps.items():
        наш, _ = цепь(f'phantom_{k}', c)
        нб, _ = цепь(f'phantom_{k}_bz', c.filter(pl.col('тип') != 'ЗБС'))
        рез[k] = (наш, нб)
        print(f"цепь {k}: n {наш['n']}, GC {наш['линии'][LINE]['GC']:.4f}, {time.time() - t0:.0f} с", flush=True)
    g3 = рез['В3'][0]['линии'][LINE]
    r3 = рез['В3'][0]
    ok = abs(g3['GC'] - 0.2631) <= TOL and abs(g3['lo'] - 0.1772) <= TOL and (abs(g3['hi'] - 0.3453) <= TOL) and (r3['n'] == 5806) and (r3['лет'] == 9) and (abs(r3['разрыв'] - float('nan')) <= TOL)
    print(f"РЕГРЕССИЯ В3: GC {g3['GC']:.4f} [{g3['lo']:.4f}, {g3['hi']:.4f}] n {r3['n']}: {('сошлась' if ok else 'НЕ СОШЛАСЬ')}", flush=True)
    if not ok:
        print('ОСТАНОВ: регрессия GC 0.2631 не сошлась')
        return 1

    def мн(r):
        return {(g['год'], i) for g in r['отборы'] for i in g['op_id'][g['отбор'][LINE]].tolist()}

    def доля_опт(r):
        npv = _цены(r)
        inc = {}
        for g in r['отборы']:
            for линия, s in ((LINE, 1.0), ('факт инженеров', -1.0)):
                for i in g['op_id'][g['отбор'][линия]].tolist():
                    inc[тип[i]] = inc.get(тип[i], 0.0) + s * npv[i]
        tot = sum(inc.values())
        return (100 * inc.get('ОПТ', 0.0) / tot, inc.get('ОПТ', 0.0) / B, tot / B)
    S3 = мн(r3)
    ids_ц = set(r3['d']['op_id'].to_list())
    cap3 = float(cap.filter(pl.col('op_id').is_in(list(ids_ц)))['capex'].sum())
    метрика, tsv = ({}, [])
    for k in ('В3', 'В1', 'В2'):
        r, rb = рез[k]
        g, gb = (r['линии'][LINE], rb['линии'][LINE])
        c = caps[k]
        d_gc = g['GC'] - g3['GC']
        S = мн(r)
        jac = len(S & S3) / len(S | S3)
        sh, inc_опт, inc_tot = доля_опт(r)
        dK = (float(c.filter(pl.col('op_id').is_in(list(ids_ц)))['capex'].sum()) - cap3) / B
        n_пр = int((c['проппант_т'].fill_null(0) > 0).sum())
        n_rcp = int(((c['проппант_т'] > 0) & (c['проппант_rcp_т'] > c['проппант_т'])).sum())
        n_пр_ц = int((c.filter(pl.col('op_id').is_in(list(ids_ц)))['проппант_т'].fill_null(0) > 0).sum())
        ис = 'W' if g['lo'] <= 0 else 'S' if abs(d_gc) <= HALF else 'Д'
        if g['GC'] <= 0 or g['lo'] <= 0 or inc_опт <= 0:
            ф = 3
        elif abs(d_gc) <= HALF and abs(sh - ДОЛЯ_ОПТ0) <= ПОРОГ_ОПТ and (jac >= ПОРОГ_J):
            ф = 1
        else:
            ф = 2
        метрика[k] = dict(g=g, gb=gb, d=d_gc, jac=jac, расх=len(S ^ S3), sh=sh, inc_опт=inc_опт, dK=dK, dKp=100 * dK * B / cap3, n_пр=n_пр, n_пр_ц=n_пр_ц, n_rcp=n_rcp, ис=ис, ф=ф, n=r['n'], лет=r['лет'], nb=rb['n'])
    принят = метрика['В1']['n_пр'] <= ПОРОГ_ПРОППАНТ and метрика['В1']['n_rcp'] == 0
    вывод('Классификатор проппанта (walk-forward, по годам обучения моделью): ' + '; '.join((f'{Y}: n {n}, AUC {a:.3f}, верно {p:.3f}' for Y, n, a, p in метр if Y >= 2016)) + '.')
    вывод(f'В2 обнуляет проппант, RCP и гель у {n_обн} операций (по годам, типы с долей ненулевого проппанта < 5 % при >= {МИН_ОП} операций в обучении).')
    вывод('')
    вывод('| вариант | n | лет | GC | lo | hi | dGC к В3 | исход S/Д/W | GC без ЗБС | lo | hi | n без ЗБС | доля ОПТ в приросте, % | прирост ОПТ, млрд | Жаккар к В3 | расхождений | dCAPEX цепи, млрд (%) | проппант>0 (6 735 / цепь) | RCP>общего | исход Ф |')
    вывод('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k in ('В3', 'В1', 'В2'):
        m = метрика[k]
        g, gb = (m['g'], m['gb'])
        вывод(f"| {k} | {m['n']} | {m['лет']} | {g['GC']:.4f} | {g['lo']:.4f} | {g['hi']:.4f} | {m['d']:+.4f} | {m['ис']} | {gb['GC']:.4f} | {gb['lo']:.4f} | {gb['hi']:.4f} | {m['nb']} | {m['sh']:.2f} | {m['inc_опт']:+.3f} | {m['jac']:.4f} | {m['расх']} | {m['dK']:+.3f} ({m['dKp']:+.2f} %) | {m['n_пр']} / {m['n_пр_ц']} | {m['n_rcp']} | Ф-{m['ф']} |")
        tsv.append([k, m['n'], m['лет'], f"{g['GC']:.4f}", f"{g['lo']:.4f}", f"{g['hi']:.4f}", f"{m['d']:+.4f}", m['ис'], f"{gb['GC']:.4f}", f"{gb['lo']:.4f}", f"{gb['hi']:.4f}", m['nb'], f"{m['sh']:.2f}", f"{m['inc_опт']:+.4f}", f"{m['jac']:.4f}", m['расх'], f"{m['dK']:+.4f}", f"{m['dKp']:+.3f}", m['n_пр'], m['n_пр_ц'], m['n_rcp'], f"Ф-{m['ф']}"])
        for имя, v, ф in (('GC', g['GC'], '.4f'), ('GC_lo', g['lo'], '.4f'), ('GC_hi', g['hi'], '.4f'), ('dGC', m['d'], '+.4f'), ('GC без ЗБС', gb['GC'], '.4f'), ('GC без ЗБС lo', gb['lo'], '.4f'), ('GC без ЗБС hi', gb['hi'], '.4f'), ('доля ОПТ в приросте %', m['sh'], '.2f'), ('Жаккар к В3', m['jac'], '.4f'), ('расхождений', m['расх'], 'd'), ('dCAPEX млрд', m['dK'], '+.4f'), ('ненулевых проппантов', m['n_пр'], 'd'), ('RCP больше общего', m['n_rcp'], 'd')):
            якорь(f'ep30 phantom {k} {имя}', v, ф)
    вывод('')
    вывод(f"Принятие В1 (проппант>0 не больше {ПОРОГ_ПРОППАНТ} и RCP>общего 0): {('принят' if принят else 'НЕ ПРИНЯТ')} (проппант>0 {метрика['В1']['n_пр']}, RCP>общего {метрика['В1']['n_rcp']}).")
    ф_верд = max(метрика['В1']['ф'], метрика['В2']['ф']) if принят else метрика['В2']['ф']
    вывод(f"**ИСХОД Ф-{ф_верд}** (вердикт по {('худшему из В1 и В2' if принят else 'В2, В1 не принят')}; В1: Ф-{метрика['В1']['ф']}, В2: Ф-{метрика['В2']['ф']}).")
    якорь('ep30 phantom номер исхода Ф', ф_верд, 'd')
    якорь('ep30 phantom В1 принят (1 да, 0 нет)', int(принят), 'd')
    якорь('ep30 phantom время счёта с', time.time() - t0, '.0f')
    OUT_MD.write_text('\n'.join(ПРОЧЬ) + '\n', encoding='utf-8')
    with open(OUT_TSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['вариант', 'n', 'лет', 'GC', 'GC_lo', 'GC_hi', 'dGC', 'исход', 'GC_без_ЗБС', 'lo_без_ЗБС', 'hi_без_ЗБС', 'n_без_ЗБС', 'доля_ОПТ_прироста_%', 'прирост_ОПТ_млрд', 'Жаккар_к_В3', 'расхождений', 'dCAPEX_млрд', 'dCAPEX_%', 'проппант_больше_0_6735', 'проппант_больше_0_цепь', 'RCP_больше_общего', 'исход_Ф'])
        w.writerows(tsv)
    return 0
if __name__ == '__main__':
    sys.exit(главная())
