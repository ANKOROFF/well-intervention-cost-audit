# Glossary: Russian names used in the code -> English

The code was written with Russian identifiers and Russian column names in the intermediate tables. Identifiers and the names of columns that the code reads from the operator's tables are kept so that the code runs unchanged on tables with the original schema. This table translates the ones a reader needs. Column headers and labels in `aggregates/` are already in English (below, the mapping used).

## Operation types (Latin codes of the paper, Table S1)

| Russian code | Code | Meaning |
|---|---|---|
| ОПТ | PMO | pump optimization |
| ППР | SWO | scheduled preventive repair (forced class) |
| ЗБС | SDT | sidetrack drilling |
| ОПЗ | BZT | bottomhole acid treatment |
| ЛА | LER | forced class (as in the paper) |
| РИР | RIS | remedial cementing |
| ЛНЭК | CLR | forced class (as in the paper) |
| ГРП | HF | hydraulic fracturing |
| ВБД | RIW | forced class (as in the paper) |
| ПиП | PCM | type with few operations |
| ВПН | ONW | type with few operations |
| ДП | APF | type with few operations |

## Representations, selections and cells

| Russian | English |
|---|---|
| Н (в именах V_НН, S_Н) | itemized (quantity-based) representation, ITM |
| Л1 | day-rate representation, DR |
| А, Б, В, Г (представления) | representations A, B, C, D of the 4x4 matrix |
| ППП, цепь ППП | main ex-ante chain (5,806 operations) |
| ДОР | gain ratio against the other ruler, defined in `src/ep19_crossruler.py` |
| оракул | perfect-information benchmark (oracle) |
| факт, факт инженеров | engineers' (historical) program |
| отбор | selection |
| розыгрыш | draw (price draw, bootstrap draw) |
| ДМ | Diebold-Mariano statistic |
| ДИ низ / ДИ верх | confidence interval, lower / upper |
| Жаккар | Jaccard similarity |
| знаменатель | denominator of GC (gap benchmark minus historical program) |

## Pump outcomes

| Russian | English |
|---|---|
| а, "а тот же типоразмер" | outcome (a): both fields valid, size identical |
| б, "б другой типоразмер" | outcome (b): both valid, size different |
| в, "в неопределимо" | outcome (c): field empty or unparsable |
| в1 материал из других драйверов | (c1) material part from other recorded drivers |
| в2 подстановка медианы типа | (c2) type-median substitution |
| произв (произв_м3сут) | pump delivery, m3/day |
| напор (напор_м) | pump head, m |
| мощность (насос_мощность) | delivery x head (QH), the hydraulic power proxy |
| до.Насос, после.Насос | pump designation before / after the operation |
| плннасос.Произв-сть, м3/сут; плннасос.Напор ЭЦН, м | planned pump delivery and head |
| ЭЦН | electric submersible pump (ESP) |
| гомоглиф | homoglyph: a letter of another alphabet that looks the same |

## Columns of the intermediate tables

| Russian | English |
|---|---|
| общ.Тип; тип | operation type |
| год | year |
| дата | date |
| выбираемая | selectable (member of the pool) |
| проппант_т, проппант_rcp_т | proppant, t; reinforced (resin-coated) proppant, t |
| гель_м3, кислота_м3, азот_т, цемент_м3 | gel, acid, cement volumes; nitrogen mass |
| проходка_м | footage, m |
| стадий | number of fracturing stages |
| гнкт_постановок | coiled-tubing setups |
| занятость_крс_сут | workover-rig occupancy, days |
| бригадо_сутки | crew-days |
| простой_сут, dur | downtime, days; duration, days |
| dur_пред, простой_пред | forecast duration, downtime |
| c_проппант, c_гель, c_насос, c_крс, c_флот_грп, c_станок, c_моб | cost parts: proppant, gel, pump, workover rig, fracturing fleet, drilling rig, mobilization |
| c_материалы, c_материалы_исп | material cost part; used material cost part |
| capex_факт, capex_пред | actual, forecast CAPEX |
| прирост_т, inc_oil_t | incremental oil over 24 months, t |
| npv_на_бригадо_сутки | NPV per crew-day (ranking score) |
| доход_диск, потеря_диск | discounted revenue, discounted loss during downtime |
| восстановлено | reconstructed |
| наблюдено, разобрано, выведено, восполнено, нет | provenance marks: observed, parsed, inferred, filled, none |
| мат_измерены, мат_ненулевая, мат_нуль_измерен | material part measured, non-zero, measured zero |
| T_техники, T_станок | dated occupancy of equipment, of the drilling rig |

## Column headers in `aggregates/`

| Russian header | English header |
|---|---|
| год, без_года | year, left-out year |
| тип, исключён | type, excluded type |
| клетка, отбор, оценка | cell, selection, valuation |
| ДИ_низ, ДИ_верх | CI_lo, CI_hi |
| повторов, розыгрышей | replicates, draws |
| знаменатель_pct_C, NPV_pct_C | denominator, NPV: percent of C |
| операций, n, лет | operations, n, years |
| Жаккар | Jaccard |
| расхождений, расхождение_отбора | disagreements (size of the symmetric difference) |
| только_Н, только_Л1 | only_ITM, only_DR |
| доля_ЗБС_в_расхождении | share_SDT_in_disagreement |
| вариант | variant |
| исход | outcome class of the pre-declared rule |
| подмножество | subset |
| совокупность | population |
| случай | pump outcome case |
| уровень, метрика, оценка | level, metric, estimate |
| причина | cause |

Words that appear inside translated headers: percent of C is the unit for every money quantity (C is the sum of the CAPEX of the pool and is not released); `_pct_C` marks such columns.
