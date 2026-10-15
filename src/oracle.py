"""Perfect-information (oracle) benchmark; one implementation used by several steps."""
from __future__ import annotations
import highspy
import numpy as np
INF = highspy.kHighsInf
ЗАЗОР = 1e-06
ЛИМИТ_С = 60.0

def exact_cert(npv, capex, dur, budget, cap, *, зазор=ЗАЗОР, лимит=ЛИМИТ_С):
    n = len(npv)
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.setOptionValue('mip_rel_gap', зазор)
    h.setOptionValue('time_limit', лимит)
    idx = np.arange(n)
    h.addVars(n, np.zeros(n), np.ones(n))
    h.changeColsIntegrality(n, idx, np.array([highspy.HighsVarType.kInteger] * n))
    h.changeColsCost(n, idx, -np.asarray(npv, float))
    h.addRow(-INF, float(budget), n, idx, np.asarray(capex, float))
    h.addRow(-INF, float(cap), n, idx, np.asarray(dur, float))
    h.run()
    i = h.getInfo()
    статус = h.modelStatusToString(h.getModelStatus())
    сведения = {'цель': -float(i.objective_function_value), 'граница': -float(i.mip_dual_bound), 'mip_gap': float(i.mip_gap), 'статус': статус, 'сертификат': статус == 'Optimal' and float(i.mip_gap) <= зазор, 'кандидатов': n}
    return (np.asarray(h.getSolution().col_value[:n]) > 0.5, сведения)

def exact(npv, capex, dur, budget, cap, *, метка='', зазор=ЗАЗОР, лимит=ЛИМИТ_С):
    sel, c = exact_cert(npv, capex, dur, budget, cap, зазор=зазор, лимит=лимит)
    if not c['сертификат']:
        print(f"  ВНИМАНИЕ оракул{(' ' + метка if метка else '')}: СЕРТИФИКАТ ОПТИМУМА НЕ ДОСТИГНУТ -- статус {c['статус']}, mip_gap {c['mip_gap']:.3e} при пороге {зазор:g}, кандидатов {c['кандидатов']}. Возвращена граница, а не оптимум: знаменатель GC ЗАНИЖЕН, число недействительно.")
    return sel
