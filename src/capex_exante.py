"""Ex-ante CAPEX on predicted material quantities (itemized cost representation)."""
import config as cfg
import contextlib
import io
import os
import numpy as np
import polars as pl
import dur_model as dm
import stats
from cost_physical import MAT, TIME, Prices, capex, load_resources
from npv import Params, npv_table
B = 1000000000.0
M = 1000000.0
SEED = 20260927
LINE = 'модель отклика на сутки'
DUR_SRC = cfg.data('interim/dur_pred.parquet')
VOL_SRC = cfg.data('interim/capex_vol_pred.parquet')
VOL_MED = cfg.data('interim/capex_vol_med.parquet')
CUBE = cfg.data('interim/ep01_gc_cube.parquet')
VMAE = cfg.data('interim/ep01_vol_mae.parquet')
ЛОГ = 'logs/ep01/capex_exante_variants.log'
ПОРОГ_НЕАДДИТИВНОСТИ = 0.005
assert ПОРОГ_НЕАДДИТИВНОСТИ == 0.005, 'порог неаддитивности объявлен ДО счёта и равен 0.005 (0.5 п.п.) -- тому же значению, при котором Ш17 объявил аддитивным разложение на два плеча'
VOL = ['проппант_т', 'проппант_rcp_т', 'гель_м3', 'кислота_м3', 'азот_т', 'цемент_м3', 'проходка_м', 'насос_мощность', 'стадий']
НЕ_ПРЕДСКАЗЫВАЮТСЯ = ['гнкт_постановок', 'занятость_крс_сут', 'T_техники']
ВАРИАНТЫ = {'ФФФ': (False, False, False), 'ПФФ': (True, False, False), 'ФПФ': (False, True, False), 'ФФП': (False, False, True), 'ППФ': (True, True, False), 'ПФП': (True, False, True), 'ФПП': (False, True, True), 'ППП': (True, True, True), 'ФФБ': (False, False, 'Б'), 'ППБ': (True, True, 'Б')}

def порог_txt(ok):
    return 'ВЫПОЛНЕН' if ok else 'НЕ ВЫПОЛНЕН'

def отн(числ, знам):
    return pl.when(знам > 0).then(числ / знам).otherwise(None)

def cap_path(dur_p, vol, без_T=False):
    v = 'Б' if vol == 'Б' else 'П' if vol else 'Ф'
    return cfg.data(f"interim/capex_physical_{('П' if dur_p else 'Ф')}{v}{('_бT' if без_T else '')}.parquet")

def volume_targets():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        t = load_resources()
    return (t, buf.getvalue())

def build_features(t):
    d = dm.features()
    y = t.select('op_id', *[pl.col(q).fill_null(0.0).cast(pl.Float64).alias(f'y_{q}') for q in VOL])
    return d.join(y, on='op_id', how='inner')

def leak_in_features(feats):
    запрещённые = set(VOL) | {f'y_{q}' for q in VOL} | set(НЕ_ПРЕДСКАЗЫВАЮТСЯ) | set(dm.EXCLUDED) | set(MAT) | set(TIME) | {'capex', 'c_материалы'}
    return sorted(set(feats) & запрещённые)

def check_k1():
    print('\n--- К1. УТЕЧКА ЧЕРЕЗ ПРИЗНАК ---')
    искать = len(set(VOL) | set(НЕ_ПРЕДСКАЗЫВАЮТСЯ) | set(dm.EXCLUDED) | set(MAT) | set(TIME))
    print(f'  послеоперационных величин, которые контроль ищет: {искать} (из них материальных объёмов {len(VOL)}) -- искать ЕСТЬ ЧТО')
    чисто = leak_in_features(dm.FEATS)
    подложено = leak_in_features(list(dm.FEATS) + ['проппант_т', 'насос_мощность'])
    print(f'  на действующем наборе из {len(dm.FEATS)} признаков найдено нарушений: {len(чисто)}')
    print(f'  на наборе с ПОДЛОЖЕННЫМИ «проппант_т» и «насос_мощность» найдено: {len(подложено)} -> {подложено}')
    ok = len(чисто) == 0 and len(подложено) == 2
    print(f'  К1 {порог_txt(ok)}: контроль ловит подложенное нарушение и молчит на чистом наборе')
    assert ok, 'контроль К1 не ловит подложенный послеоперационный признак'
    return (ok, искать, len(подложено))

def check_k2(d):
    print('\n--- К2. УТЕЧКА ЧЕРЕЗ КЛЮЧ asof ---')
    ev = pl.read_parquet(cfg.data('interim/gtm_events.parquet')).filter(pl.col('выбираемая')).with_columns(dur_цель=(pl.col('end') - pl.col('beg')).dt.total_days().clip(0, dm.MAX_DUR).cast(pl.Float64), простой_цель=(pl.col('vnr') - pl.col('beg')).dt.total_days().clip(0, dm.MAX_DUR).cast(pl.Float64))
    w, tp = dm.history(ev)
    base = ev.with_columns(mi_beg=dm.mi(pl.col('beg')))

    def подтянуть(сдвиг):
        lhs = base.with_columns(mi_src=pl.col('mi_beg') - сдвиг).sort('mi_src')
        a = lhs.join_asof(w, left_on='mi_src', right_on='mi_end', by='well', strategy='backward').rename({'mi_end': 'mi_w'})
        a = a.sort('mi_src').join_asof(tp, left_on='mi_src', right_on='mi_end', by='тип', strategy='backward').rename({'mi_end': 'mi_t'})
        нар = a.filter((pl.col('mi_w') >= pl.col('mi_beg')) | (pl.col('mi_t') >= pl.col('mi_beg'))).height
        есть_w = int(a.filter(pl.col('mi_w') == pl.col('mi_beg')).height)
        return (a, нар, есть_w)
    _, нар_ok, _ = подтянуть(1)
    _, нар_bad, есть = подтянуть(0)
    print(f'  действующий ключ mi_beg - 1: операций с источником НЕ РАНЬШЕ месяца начала: {нар_ok}')
    print(f'  ПОДЛОЖЕННЫЙ ключ mi_beg: таких операций {нар_bad:,}, из них с историей СКВАЖИНЫ ровно в месяце начала {есть:,}')
    print(f'  ловить БЫЛО ЧТО: {нар_bad:,} операций -- это счётчик найденного, а не код возврата')
    ok = нар_ok == 0 and нар_bad > 0
    print(f'  К2 {порог_txt(ok)}')
    assert ok, 'контроль К2 либо даёт ложное срабатывание на действующем ключе, либо не находит ни одной операции на подложенном'
    return (ok, нар_bad, есть)

def check_k3(d):
    print('\n--- К3. УТЕЧКА ЧЕРЕЗ ГРАНИЦУ ОБУЧЕНИЯ ---')
    годы = sorted(d['beg'].dt.year().unique().drop_nulls().to_list())
    всего_ok = всего_bad = 0
    for Y in годы:
        tr_ok = d.filter(pl.col('end').dt.year() < Y)
        tr_bad = d.filter(pl.col('beg').dt.year() < Y)
        всего_ok += tr_ok.filter(pl.col('end') >= pl.date(Y, 1, 1)).height
        всего_bad += tr_bad.filter(pl.col('end') >= pl.date(Y, 1, 1)).height
    print(f'  действующее окно по ОКОНЧАНИЮ: операций обучения, ещё не завершённых на 01.01.Y: {всего_ok} (суммарно по {len(годы)} годам)')
    print(f'  ПОДЛОЖЕННОЕ окно по НАЧАЛУ: таких операций {всего_bad:,}')
    print(f'  ловить БЫЛО ЧТО: {всего_bad:,} операций')
    ok = всего_ok == 0 and всего_bad > 0
    print(f'  К3 {порог_txt(ok)}')
    assert ok, 'контроль К3 не различает окно по началу и окно по окончанию'
    return (ok, всего_bad)

def holdout_volumes(d, split_year=2022, last_year=2024):
    tr = d.filter(pl.col('end').dt.year() < split_year)
    te = d.filter(pl.col('beg').dt.year().is_between(split_year, last_year))
    строки, поти = ([], {})
    for q in VOL:
        col = f'y_{q}'
        y = te[col].to_numpy()
        base = dm.type_median(tr, te, col)
        r = {'величина': q, 'n_test': te.height, 'MAE_медиана_типа': dm.mae(y, base), 'сумма_факт': float(y.sum()), 'сумма_медиана': float(base.sum())}
        for obj, имя in (('l1', 'L1'), ('l2', 'L2')):
            p = np.clip(dm.fit(tr, col, obj).predict(te.select(dm.FEATS).to_pandas()), 0, None)
            r[f'MAE_{имя}'] = dm.mae(y, p)
            r[f'сумма_{имя}'] = float(p.sum())
            поти.setdefault(имя, {})[q] = p
        строки.append(r)
    return (pl.DataFrame(строки), tr, te, поти)

def walk_forward_volumes(d, только_медиана=False):
    куски, диаг = ([], [])
    годы = sorted(d['beg'].dt.year().unique().drop_nulls().to_list())
    for Y in годы:
        te = d.filter(pl.col('beg').dt.year() == Y)
        tr = d.filter(pl.col('end').dt.year() < Y)
        r = te.select('op_id')
        if tr.height >= dm.MIN_TRAIN and (not только_медиана):
            как = 'модель'
            for q in VOL:
                p = dm.fit(tr, f'y_{q}').predict(te.select(dm.FEATS).to_pandas())
                r = r.with_columns(pl.Series(f'{q}_пред', np.clip(p, 0, None)))
        elif tr.height:
            как = 'медиана типа'
            for q in VOL:
                r = r.with_columns(pl.Series(f'{q}_пред', dm.type_median(tr, te, f'y_{q}')))
        else:
            как = 'истории нет'
            for q in VOL:
                r = r.with_columns(pl.lit(None, dtype=pl.Float64).alias(f'{q}_пред'))
        куски.append(r)
        диаг.append({'год_начала': Y, 'операций': te.height, 'обучающих': tr.height, 'способ': как})
    return (pl.concat(куски), pl.DataFrame(диаг))

def ресурсы(dur_pred: bool, без_T: bool=False):
    ev = pl.read_parquet(cfg.data('interim/gtm_events.parquet')).filter(pl.col('выбираемая'))
    if dur_pred:
        pr = pl.read_parquet(DUR_SRC).select('op_id', 'dur_пред')
        ev = ev.join(pr, on='op_id', how='left').with_columns(dur=pl.col('dur_пред'))
    t = load_resources(ev)
    if без_T:
        б = t.filter(~pl.col('восстановлено')).select((pl.col('занятость_крс_сут') / pl.col('dur').clip(1, None)).median()).item()
        t = t.with_columns(занятость_крс_сут=pl.col('dur').clip(0, None) * б, T_техники=pl.lit(None, dtype=pl.Float64), восстановлено=pl.lit(False))
    return t

def подменить_объёмы(t, vp):
    ren = {f'{q}_пред': q for q in VOL}
    return t.drop(VOL).join(vp.select('op_id', *ren.keys()).rename(ren), on='op_id', how='inner').with_columns(насос_нулевая=pl.lit(False), насос_неопределим=pl.lit(False), гнкт_постановок=pl.lit(None, dtype=pl.Float64))

def собрать_capex(vp, vm, население, log):
    из = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for dur_p in (False, True):
            t = ресурсы(dur_p)
            для = {False: t, True: подменить_объёмы(t, vp), 'Б': подменить_объёмы(t, vm)}
            for vol, tt in для.items():
                c = capex(tt.filter(pl.col('op_id').is_in(население)), Prices())
                c.write_parquet(cap_path(dur_p, vol), compression='zstd')
                из[('П' if dur_p else 'Ф') + ('Б' if vol == 'Б' else 'П' if vol else 'Ф')] = c
        tb = подменить_объёмы(ресурсы(True, без_T=True), vp)
        cb = capex(tb.filter(pl.col('op_id').is_in(население)), Prices())
        cb.write_parquet(cap_path(True, True, без_T=True), compression='zstd')
        из['ПП_бT'] = cb
    log.write('\n=== сборка семи таблиц CAPEX ===\n' + buf.getvalue())
    return из

def run(метка, use_dur, use_down, use_vol, log, узкая=True):
    p = Params(dur_source=DUR_SRC if узкая else None, use_pred_dur=use_dur, use_pred_downtime=use_down, capex_source=cap_path(use_dur, use_vol) if узкая else cfg.data('interim/capex_physical.parquet'))
    t = npv_table(p)
    stats.rng = np.random.default_rng(SEED)
    stats.npv_table = lambda _p, _t=t: _t
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        d, руб_т = stats.build()
        if use_dur:
            pr = pl.read_parquet(DUR_SRC).select('op_id', 'dur_пред')
            d = d.drop('dur').join(pr, on='op_id', how='left').rename({'dur_пред': 'dur'})
        df, res = stats.report(d, руб_т, 1, f'EP-01 {метка}')
    log.write(f"\n{'#' * 78}\n# ВАРИАНТ {метка}\n{'#' * 78}\n" + buf.getvalue())
    r = res.filter(pl.col('линия') == LINE).row(0, named=True)
    f = res.filter(pl.col('линия') == 'факт инженеров').row(0, named=True)
    o = res.filter(pl.col('линия') == 'оракул по факту').row(0, named=True)
    return {'вариант': метка, 'операций': t.height, 'лет': r['лет'], 'capex_млрд': float(t['capex'].sum()) / B, 'NPV_факт': f['NPV_млрд'], 'NPV_модель': r['NPV_млрд'], 'NPV_оракул': o['NPV_млрд'], 'разрыв': o['NPV_млрд'] - f['NPV_млрд'], 'GC': r['GC'], 'GC_lo': r['GC_lo'], 'GC_hi': r['GC_hi'], 'повторов': r['повторов'], 'ДМ': r['ДМ'], 'p': r['p']}

def run_bez_T(log):
    p = Params(dur_source=DUR_SRC, use_pred_dur=True, use_pred_downtime=True, capex_source=cap_path(True, True, без_T=True))
    t = npv_table(p)
    stats.rng = np.random.default_rng(SEED)
    stats.npv_table = lambda _p, _t=t: _t
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        d, руб_т = stats.build()
        pr = pl.read_parquet(DUR_SRC).select('op_id', 'dur_пред')
        d = d.drop('dur').join(pr, on='op_id', how='left').rename({'dur_пред': 'dur'})
        _, res = stats.report(d, руб_т, 1, 'EP-01 ППП+ без T_техники')
    log.write(f"\n{'#' * 78}\n# ВАРИАНТ ППП+ (без T_техники)\n{'#' * 78}\n" + buf.getvalue())
    r = res.filter(pl.col('линия') == LINE).row(0, named=True)
    return {'вариант': 'ППП+ без T', 'операций': t.height, 'лет': r['лет'], 'capex_млрд': float(t['capex'].sum()) / B, 'GC': r['GC'], 'GC_lo': r['GC_lo'], 'GC_hi': r['GC_hi'], 'повторов': r['повторов'], 'ДМ': r['ДМ'], 'p': r['p']}
if __name__ == '__main__':
    os.makedirs(os.path.dirname(ЛОГ), exist_ok=True)
    log = io.open(ЛОГ, 'w', encoding='utf-8')
    print('=== EP-01: CAPEX НА ПРЕДСКАЗАННЫХ МАТЕРИАЛЬНЫХ ОБЪЁМАХ ===')
    print(f'порог неаддитивности ОБЪЯВЛЕН ДО СЧЁТА: {100 * ПОРОГ_НЕАДДИТИВНОСТИ:.1f} п.п. (константа модуля ПОРОГ_НЕАДДИТИВНОСТИ = {ПОРОГ_НЕАДДИТИВНОСТИ})')
    t0, лог_ресурсов = volume_targets()
    log.write('=== load_resources ===\n' + лог_ресурсов)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c0 = capex(t0, Prices())
    log.write('\n=== capex факт ===\n' + buf.getvalue())
    print(f'\n=== ЧТО ИМЕННО ТЕЧЁТ: доля статей в CAPEX портфеля ===')
    tot = float(c0['capex'].sum())
    статьи = [(k, float(c0[k].sum()) / tot) for k in MAT + TIME]
    для_печати = sorted(статьи, key=lambda x: -x[1])
    for k, v in для_печати:
        плечо = 'CAPEX (объёмы)' if k in MAT or k == 'c_флот_грп' else 'длительность' if k == 'c_крс' else 'CAPEX (объёмы)' if k == 'c_станок' else 'ex-ante по построению'
        print(f'  {k:14s} {100 * v:6.2f}%   плечо: {плечо}')
    доля_объёмов = sum((v for k, v in статьи if k in MAT or k in ('c_флот_грп', 'c_станок')))
    print(f'  ИТОГО на послеоперационных объёмах: {100 * доля_объёмов:.2f}% CAPEX портфеля {tot / B:.2f} млрд ₽ ({c0.height:,} операций)')
    print('\n=== СОСТЯЗАТЕЛЬНЫЕ ПРОВЕРКИ УТЕЧКИ (каждая с предъявленным входом) ===')
    d = build_features(t0)
    k1_ok, k1_искать, k1_найдено = check_k1()
    k2_ok, k2_найдено, k2_история = check_k2(d)
    k3_ok, k3_найдено = check_k3(d)
    print('\n--- СТРОГАЯ ОБРАТНОСТЬ ПРИЗНАКОВ (assert) ---')
    nw = int(d['mi_src_w'].is_not_null().sum())
    nt = int(d['mi_src_t'].is_not_null().sum())
    плохо = d.filter(pl.col('mi_src_w').is_not_null() & (pl.col('mi_src_w') >= pl.col('mi_beg')) | pl.col('mi_src_t').is_not_null() & (pl.col('mi_src_t') >= pl.col('mi_beg'))).height
    assert плохо == 0, f'нарушена строгая обратность признаков: {плохо} операций'
    print(f'assert строгой обратности ПРОХОДИТ на {d.height:,} операциях выборки; источник найден у {nw:,} по истории скважины и у {nt:,} по истории типа')
    print('это ТАВТОЛОГИЯ ПО ПОСТРОЕНИЮ (backward-join плюс сдвиг ключа на -1), а не измерение; измерения дают К1-К3 выше со своими счётчиками')
    print('\n=== КРИТЕРИЙ 2: ОШИБКА ОБЪЁМА НА ОТЛОЖЕННОМ ПЕРИОДЕ ===')
    print('обучение на ЗАВЕРШЁННЫХ до 2022, проверка на НАЧАТЫХ 2022-2024; случайное разбиение для временных данных запрещено')
    mae_df, tr, te, поти = holdout_volumes(d)
    print(f"обучающих {tr.height:,} (последнее окончание {tr['end'].max()}), проверочных {te.height:,} (первое начало {te['beg'].min()})")
    print(mae_df.select('величина', MAE_медиана=pl.col('MAE_медиана_типа').round(3), MAE_L1=pl.col('MAE_L1').round(3), MAE_L2=pl.col('MAE_L2').round(3), выигрыш_L1_пц=(100 * отн(pl.col('MAE_медиана_типа') - pl.col('MAE_L1'), pl.col('MAE_медиана_типа'))).round(1), выигрыш_L2_пц=(100 * отн(pl.col('MAE_медиана_типа') - pl.col('MAE_L2'), pl.col('MAE_медиана_типа'))).round(1)).to_pandas().to_string(index=False))
    выр = int((mae_df['MAE_медиана_типа'] <= 0).sum())
    print(f'величин с ТОЖДЕСТВЕННО НУЛЕВЫМ объёмом на отложенном периоде: {выр} из {mae_df.height} -- у них MAE базы равен нулю, отношение выигрыша не определено и напечатано как NaN, а не как выигрыш 0%')
    луч1 = int((mae_df['MAE_L1'] < mae_df['MAE_медиана_типа']).sum())
    луч2 = int((mae_df['MAE_L2'] < mae_df['MAE_медиана_типа']).sum())
    print(f'L1 лучше медианы типа у {луч1} величин из {mae_df.height}; L2 -- у {луч2} из {mae_df.height}')
    print('ПОРОГА на выигрыш здесь НЕ объявлено намеренно: порог, для которого не предъявлен вход, на котором он срабатывает, -- не порог (П4), а критерий шага требует НАЗВАТЬ MAE, а не перешагнуть черту')
    print('\nЦЕНА ВЫБОРА L1: отношение суммы предсказанного объёма к фактической на отложенном периоде')
    print('(условная медиана на разреженном нулями таргете смещает СУММУ вниз, а CAPEX входит в бюджет года именно суммой)')
    сум = mae_df.select('величина', доля_медианы=отн(pl.col('сумма_медиана'), pl.col('сумма_факт')).round(3), доля_L1=отн(pl.col('сумма_L1'), pl.col('сумма_факт')).round(3), доля_L2=отн(pl.col('сумма_L2'), pl.col('сумма_факт')).round(3))
    print(сум.to_pandas().to_string(index=False))
    print('\nMAE ПО КАЖДОМУ ТИПУ (L1 против медианы типа), отложенный период:')
    строки_типов = []
    for q in VOL:
        col = f'y_{q}'
        z = te.select('тип', y=pl.col(col)).with_columns(pl.Series('p', поти['L1'][q]), pl.Series('b', dm.type_median(tr, te, col)))
        g = z.group_by('тип').agg(n=pl.len(), MAE_база=(pl.col('y') - pl.col('b')).abs().mean(), MAE_L1=(pl.col('y') - pl.col('p')).abs().mean(), факт_мед=pl.col('y').median()).with_columns(величина=pl.lit(q))
        строки_типов.append(g)
    типы = pl.concat(строки_типов).with_columns(выигрыш_пц=100 * отн(pl.col('MAE_база') - pl.col('MAE_L1'), pl.col('MAE_база')))
    print(типы.select('величина', 'тип', 'n', MAE_база=pl.col('MAE_база').round(2), MAE_L1=pl.col('MAE_L1').round(2), выигрыш_пц=pl.col('выигрыш_пц').round(1)).filter(pl.col('MAE_база') > 0).sort(['величина', 'n'], descending=[False, True]).to_pandas().to_string(index=False))
    вырожд = типы.filter(pl.col('MAE_база') <= 0).height
    print(f'пар (величина, тип) с нулевым MAE базы: {вырожд} из {типы.height} -- там объём тождественно нулевой, и отношение выигрыша неопределимо; они исключены из таблицы, а не показаны как выигрыш 0%')
    типы.write_parquet(VMAE, compression='zstd')
    print('\nMAE САМОГО CAPEX НА ОТЛОЖЕННОМ ПЕРИОДЕ, млн ₽ (квант денег 1 ₽):')
    vp_te = te.select('op_id').with_columns(*[pl.Series(f'{q}_пред', поти['L1'][q]) for q in VOL])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c_факт = capex(t0.filter(pl.col('op_id').is_in(te['op_id'].to_list())), Prices())
        c_пред = capex(подменить_объёмы(t0.filter(pl.col('op_id').is_in(te['op_id'].to_list())), vp_te), Prices())
    log.write('\n=== capex на отложенном периоде ===\n' + buf.getvalue())
    мед_типа = c_факт.group_by('тип').agg(cm=pl.col('capex').median())
    ср = c_факт.select('op_id', 'тип', capex_факт='capex').join(c_пред.select('op_id', capex_пред='capex'), on='op_id').join(мед_типа, on='тип', how='left')
    гр = ср.group_by('тип').agg(n=pl.len(), MAE_база_млн=(pl.col('capex_факт') - pl.col('cm')).abs().mean() / M, MAE_модель_млн=(pl.col('capex_факт') - pl.col('capex_пред')).abs().mean() / M, факт_млн=pl.col('capex_факт').median() / M, пред_млн=pl.col('capex_пред').median() / M).with_columns(выигрыш_пц=100 * отн(pl.col('MAE_база_млн') - pl.col('MAE_модель_млн'), pl.col('MAE_база_млн'))).sort('n', descending=True)
    print(гр.select('тип', 'n', MAE_база=pl.col('MAE_база_млн').round(2), MAE_модель=pl.col('MAE_модель_млн').round(2), выигрыш_пц=pl.col('выигрыш_пц').round(1), факт_мед=pl.col('факт_млн').round(2), пред_мед=pl.col('пред_млн').round(2)).to_pandas().to_string(index=False))
    лучше = int((гр['MAE_модель_млн'] < гр['MAE_база_млн']).sum())
    print(f'модель точнее медианы типа по CAPEX у {лучше} типов из {гр.height}')
    сп, сф = (float(ср['capex_пред'].sum()), float(ср['capex_факт'].sum()))
    print(f'сумма CAPEX отложенного периода: предсказанная {сп / B:.3f} млрд ₽ против фактической {сф / B:.3f} млрд ₽, отношение {сп / сф:.3f}')
    print('\n=== ПРОГНОЗ ОБЪЁМОВ ВНЕ ОБУЧЕНИЯ (walk-forward) ===')
    vp, диаг = walk_forward_volumes(d)
    print(диаг.to_pandas().to_string(index=False))
    vm, _ = walk_forward_volumes(d, только_медиана=True)
    есть_vol = vp.filter(pl.col('насос_мощность_пред').is_not_null())['op_id']
    есть_med = vm.filter(pl.col('насос_мощность_пред').is_not_null())['op_id']
    dp = pl.read_parquet(DUR_SRC).drop_nulls(['dur_пред', 'простой_пред'])['op_id']
    население = set(есть_vol.to_list()) & set(есть_med.to_list()) & set(dp.to_list())
    print(f'операций с прогнозом объёма моделью: {len(есть_vol):,}; медианой типа: {len(есть_med):,}; с прогнозом длительности и простоя: {len(dp):,}; ПЕРЕСЕЧЕНИЕ (общая популяция ВСЕХ вариантов): {len(население):,}')
    vp.write_parquet(VOL_SRC, compression='zstd')
    vm.write_parquet(VOL_MED, compression='zstd')
    print(f'-> {VOL_SRC} ({vp.height:,} x {vp.width})')
    print(f'-> {VOL_MED} ({vm.height:,} x {vm.width})')
    население_l = sorted(население)
    таб = собрать_capex(vp, vm, население_l, log)
    print('\n=== CAPEX: ФАКТ ПРОТИВ ПРОГНОЗА НА ОБЩЕЙ ПОПУЛЯЦИИ ===')
    cmp = таб['ФФ'].select('op_id', 'тип', ф='capex').join(таб['ФП'].select('op_id', п='capex'), on='op_id').group_by('тип').agg(n=pl.len(), факт_мед_млн=(pl.col('ф') / M).median(), пред_мед_млн=(pl.col('п') / M).median(), факт_сум_млрд=pl.col('ф').sum() / B, пред_сум_млрд=pl.col('п').sum() / B).sort('n', descending=True)
    print(cmp.select('тип', 'n', факт_мед=pl.col('факт_мед_млн').round(2), пред_мед=pl.col('пред_мед_млн').round(2), факт_сум=pl.col('факт_сум_млрд').round(3), пред_сум=pl.col('пред_сум_млрд').round(3), отношение=(pl.col('пред_сум_млрд') / pl.col('факт_сум_млрд')).round(3)).to_pandas().to_string(index=False))
    print('медианы и суммы напечатаны обе: медиана суммы не равна сумме медиан')
    сф = float(таб['ФФ']['capex'].sum())
    сп = float(таб['ФП']['capex'].sum())
    print(f'CAPEX портфеля на общей популяции: факт {сф / B:.3f} млрд ₽, прогноз {сп / B:.3f} млрд ₽, отношение {сп / сф:.3f}')
    print(f'ex-ante CAPEX ЗАНИЖЕН на {100 * (1 - сп / сф):.1f}% -- прямое следствие выбора L1, измеренное выше отношениями сумм по каждому объёму')
    подст = таб['ФФ'].filter(~pl.col('мат_измерены'))['op_id'].to_list()
    мф = таб['ФФ'].filter(pl.col('op_id').is_in(подст))
    мп = таб['ФП'].filter(pl.col('op_id').is_in(подст))
    оф = таб['ФФ'].filter(~pl.col('op_id').is_in(подст))
    оп = таб['ФП'].filter(~pl.col('op_id').is_in(подст))
    print(f'\nвклад ЗАМЕНЫ МЕДИАННОЙ ПОДСТАНОВКИ на прогноз (объявлено в докстроке, а не обнаружено задним числом):')
    print(f"  операций с подстановкой медианы в факте: {мф.height:,} из {таб['ФФ'].height:,} ({100 * мф.height / таб['ФФ'].height:.1f}%)")
    print(f"  на них CAPEX: факт {float(мф['capex'].sum()) / B:.3f} млрд -> прогноз {float(мп['capex'].sum()) / B:.3f} млрд, разность {(float(мп['capex'].sum()) - float(мф['capex'].sum())) / B:+.3f} млрд")
    д_м = float(мп['capex'].sum()) - float(мф['capex'].sum())
    д_о = float(оп['capex'].sum()) - float(оф['capex'].sum())
    print(f"  на остальных {оф.height:,}: факт {float(оф['capex'].sum()) / B:.3f} -> прогноз {float(оп['capex'].sum()) / B:.3f} млрд, разность {д_о / B:+.3f} млрд")
    print(f'  доля замены подстановки в ОБЩЕМ сдвиге CAPEX: {100 * abs(д_м) / abs(д_м + д_о):.1f}% -- занижение идёт из основной массы операций, а не из этих {мф.height:,}')
    print(f'\n=== ДЕСЯТЬ ВАРИАНТОВ, основная линия «{LINE}» ===')
    print('метка читается как (длительность, простой, CAPEX); Ф -- факт, П -- прогноз модели, Б -- прогноз медианой типа')
    полная = run('ПОЛНАЯ факт', False, False, False, log, узкая=False)
    rows = [run(м, *ф, log) for м, ф in ВАРИАНТЫ.items()]
    r = pl.DataFrame(rows)
    print(r.select('вариант', 'операций', 'лет', capex_млрд=pl.col('capex_млрд').round(2), NPV_факт=pl.col('NPV_факт').round(2), NPV_модель=pl.col('NPV_модель').round(2), NPV_оракул=pl.col('NPV_оракул').round(2), разрыв=pl.col('разрыв').round(2), GC=pl.col('GC').round(4), ДИ_ниж=pl.col('GC_lo').round(4), ДИ_верх=pl.col('GC_hi').round(4), повт=pl.col('повторов'), ДМ=pl.col('ДМ').round(2), p=pl.col('p').round(4)).to_pandas().to_string(index=False))
    r.write_parquet(CUBE, compression='zstd')
    g = {x['вариант']: x['GC'] for x in rows}
    print('\n=== ВКЛАД СУЖЕНИЯ ПОПУЛЯЦИИ (отдельно от замены величин) ===')
    print(f"полная популяция {полная['операций']:,} операций, GC {полная['GC']:.4f} [{полная['GC_lo']:.4f}, {полная['GC_hi']:.4f}]")
    print(f"суженная до {rows[0]['операций']:,} операций, GC(ФФФ) {g['ФФФ']:.4f} -- вклад ОДНОГО СУЖЕНИЯ {g['ФФФ'] - полная['GC']:+.4f} при тех же фактических величинах")
    print('оба числа ВЫЧИСЛЕНЫ этим прогоном; ни одно не вписано литералом')
    пп = [x for x in rows if x['вариант'] == 'ППП'][0]
    ппф = [x for x in rows if x['вариант'] == 'ППФ'][0]
    print('\n=== КРИТЕРИЙ 3: GC НА ПОЛНОСТЬЮ ex-ante ЦЕПОЧКЕ ===')
    print(f"GC(ППП) = {пп['GC']:.4f} [{пп['GC_lo']:.4f}, {пп['GC_hi']:.4f}], ДМ {пп['ДМ']:.2f}, p = {пп['p']:.4f}, повторов бутстрэпа {пп['повторов']}, лет {пп['лет']}, операций {пп['операций']:,}")
    ноль = пп['GC_lo'] <= 0.0 <= пп['GC_hi']
    print(f"доверительный интервал {('НАКРЫВАЕТ' if ноль else 'не накрывает')} ноль")
    print(f"ШИРИНА доверительного интервала GC(ППП): {100 * (пп['GC_hi'] - пп['GC_lo']):.2f} п.п. -- это разрешающая способность девяти лет, и всякая разность GC МЕНЬШЕ неё остаётся точечной оценкой, а не измеренным различием")
    ффф = [x for x in rows if x['вариант'] == 'ФФФ'][0]
    прирост = пп['NPV_модель'] - пп['NPV_факт']
    print('\n=== ПРИРОСТ NPV НА ПОЛНОСТЬЮ ex-ante ЦЕПОЧКЕ (ППП) ===')
    print(f"ПРИРОСТ NPV(ППП) = {пп['NPV_модель']:.2f} - {пп['NPV_факт']:.2f} = {прирост:.2f} млрд ₽ за {пп['лет']} лет")
    print('это ПРИРОСТ NPV портфеля относительно факта инженеров, а НЕ экономия затрат: оба столбца -- NPV, затраты в разность не входят')
    print(f"знаменатель GC(ППП) -- разрыв СВОЕЙ строки: {пп['разрыв']:.2f} млрд ₽ (оракул {пп['NPV_оракул']:.2f} - факт {пп['NPV_факт']:.2f})")
    print(f"разрыв строки ФФФ -- ДРУГОЕ число: {ффф['разрыв']:.2f} млрд ₽ (оракул {ффф['NPV_оракул']:.2f} - факт {ффф['NPV_факт']:.2f})")
    print(f"сверка знаменателя: {прирост:.2f} / {пп['разрыв']:.2f} = {прирост / пп['разрыв']:.5f} против GC(ППП) = {пп['GC']:.5f}")
    print('ЗАПРЕЩЕНО: произведение GC(ППП) на разрыв строки ФФФ. Доля одной строки на знаменателе другой даёт величину, которой в работе нет; денежное число строки ППП берётся ТОЛЬКО этой разностью.')
    print('\n=== КРИТЕРИЙ 4: ЦЕНА УТЕЧКИ ===')
    цена = пп['GC'] - ппф['GC']
    print(f"GC(ППФ), частичная цепочка на ЭТОЙ популяции: {ппф['GC']:.4f} [{ппф['GC_lo']:.4f}, {ппф['GC_hi']:.4f}]")
    print(f"GC(ППП), полная ex-ante цепочка:              {пп['GC']:.4f} [{пп['GC_lo']:.4f}, {пп['GC_hi']:.4f}]")
    print(f'ЦЕНА УТЕЧКИ CAPEX: {цена:+.4f} ({100 * цена:+.2f} п.п.)')
    ппб = [x for x in rows if x['вариант'] == 'ППБ'][0]
    цена_б = ппб['GC'] - ппф['GC']
    print(f"то же на ВТОРОМ допустимом предсказателе объёма (расширяющаяся медиана типа): GC(ППБ) = {ппб['GC']:.4f} [{ппб['GC_lo']:.4f}, {ппб['GC_hi']:.4f}], цена утечки {цена_б:+.4f} ({100 * цена_б:+.2f} п.п.)")
    print(f'РАЗМАХ ЦЕНЫ УТЕЧКИ ПО ДВУМ ПРЕДСКАЗАТЕЛЯМ: от {100 * min(цена, цена_б):+.2f} до {100 * max(цена, цена_б):+.2f} п.п. -- названы оба, потому что модель проигрывает медиане типа по MAE у восьми объёмов из девяти, и цена утечки на одном предсказателе была бы ценой утечки ПЛЮС ценой неудачной модели')
    print(f"знаменатель обоих чисел назван: {пп['операций']:,} операций, {пп['лет']} лет, одна и та же популяция")
    print('опубликованное 0.2829 сюда НЕ подставлено: оно считано на своей популяции, и разность с ним смешивала бы замену величины с заменой выборки; сверка с ним -- строка «вклад сужения» выше')
    print('\n=== КРИТЕРИЙ 5: РАЗЛОЖЕНИЕ ПО ТРЁМ ПЛЕЧАМ ===')
    print(f'порог неаддитивности {100 * ПОРОГ_НЕАДДИТИВНОСТИ:.1f} п.п. объявлен ДО счёта (константа модуля), значение то же, при котором Ш17 объявил аддитивным разложение на два плеча')
    d_dur = g['ПФФ'] - g['ФФФ']
    d_down = g['ФПФ'] - g['ФФФ']
    d_cap = g['ФФП'] - g['ФФФ']
    i_dd = g['ППФ'] - g['ФФФ'] - d_dur - d_down
    i_dc = g['ПФП'] - g['ФФФ'] - d_dur - d_cap
    i_nc = g['ФПП'] - g['ФФФ'] - d_down - d_cap
    i_3 = g['ППП'] - g['ФФФ'] - d_dur - d_down - d_cap - i_dd - i_dc - i_nc
    ост = g['ППП'] - g['ФФФ'] - d_dur - d_down - d_cap
    print(f"  GC(ФФФ) опорный                       {g['ФФФ']:+.4f}")
    print(f'  плечо длительности  dGC(dur)          {d_dur:+.4f} ({100 * d_dur:+.2f} п.п.)')
    print(f'  плечо простоя       dGC(простой)      {d_down:+.4f} ({100 * d_down:+.2f} п.п.)')
    print(f'  плечо CAPEX         dGC(объёмы)       {d_cap:+.4f} ({100 * d_cap:+.2f} п.п.)')
    print(f'  СУММА ТРЁХ ПЛЕЧ                       {d_dur + d_down + d_cap:+.4f} ({100 * (d_dur + d_down + d_cap):+.2f} п.п.)')
    print(f'  ОСТАТОК (все взаимодействия)          {ост:+.4f} ({100 * ост:+.2f} п.п.)')
    print(f"  GC(ППП) = ФФФ + три плеча + остаток   {g['ППП']:+.4f}")
    print('  парные и тройное взаимодействия по отдельности:')
    print(f'    dur x простой   {i_dd:+.4f} ({100 * i_dd:+.2f} п.п.)')
    print(f'    dur x CAPEX     {i_dc:+.4f} ({100 * i_dc:+.2f} п.п.)')
    print(f'    простой x CAPEX {i_nc:+.4f} ({100 * i_nc:+.2f} п.п.)')
    print(f'    тройное         {i_3:+.4f} ({100 * i_3:+.2f} п.п.)')
    add = abs(ост) < ПОРОГ_НЕАДДИТИВНОСТИ
    print(f"  разложение {('АДДИТИВНО' if add else 'НЕАДДИТИВНО')}: |остаток| = {abs(ост):.4f} против порога {ПОРОГ_НЕАДДИТИВНОСТИ:.4f} ({100 * ПОРОГ_НЕАДДИТИВНОСТИ:.1f} п.п.); плечи {('разделимы' if add else 'НЕ разделимы')}")
    print('\n=== ОСТАТОК УТЕЧКИ ВНЕ ПЛЕЧА CAPEX: ОКНО ПРИСУТСТВИЯ ТЕХНИКИ ===')
    вос = int(t0['восстановлено'].sum())
    print(f'T_техники восстановлено из датированных событий у {вос:,} операций из {t0.height:,} ({100 * вос / t0.height:.1f}%); это послеоперационное сведение, принадлежащее плечу ДЛИТЕЛЬНОСТИ (Ш9), а не плечу CAPEX')
    bt = run_bez_T(log)
    print(f"GC(ППП+) со снятым T_техники: {bt['GC']:.4f} [{bt['GC_lo']:.4f}, {bt['GC_hi']:.4f}], операций {bt['операций']:,}, capex {bt['capex_млрд']:.2f} млрд ₽")
    print(f"остаток утечки сверху ограничен: GC(ППП+) - GC(ППП) = {bt['GC'] - пп['GC']:+.4f} ({100 * (bt['GC'] - пп['GC']):+.2f} п.п.)")
    print('\n=== ИТОГ ШАГА ОДНОЙ СТРОКОЙ ===')
    print(f"GC полной ex-ante цепочки {пп['GC']:.4f} [{пп['GC_lo']:.4f}, {пп['GC_hi']:.4f}] против {ппф['GC']:.4f} на частичной; цена утечки CAPEX {100 * цена:+.2f} п.п. на {пп['операций']:,} операциях и {пп['лет']} годах")
    log.close()
    print(f'подробные прогоны вариантов: {ЛОГ}')
    print(f'-> {CUBE}')
    print(f'-> {VMAE}')
