# Domain model

The entities and terms of the audit, in the wording of the paper. The class diagram is in the [README](../README.md#domain-model). Money is a share of C, the CAPEX of all 6,800 operations in the pool, so it is dimensionless. Russian names used in the code are in [GLOSSARY.md](GLOSSARY.md).

## Entities and attributes

| entity | attribute | unit | meaning |
|---|---|---|---|
| Well | well key | text | normalized identifier that joins registry, production and event tables |
| Operation | type | code | one of twelve Latin codes (PMO, SDT, HF, BZT, RIS, SWO, RIW, LER, CLR, PCM, ONW, APF) |
| Operation | start date, end date | date | dates of the work in the registry |
| Operation | selectable | yes/no | member of the pool of selectable operations |
| CostDriver | duration | days | raw calendar duration of the operation |
| CostDriver | idle time | days | well downtime caused by the operation |
| CostDriver | material volume | t, m3, m, stages | proppant, reinforced proppant, gel, acid, nitrogen, cement, footage, fracturing stages |
| CostDriver | pump designation | text | string `ESP-delivery-head` before and after the operation |
| CostDriver | QH | m4/day | delivery times head of the pump, proxy for its hydraulic power |
| CostDriver | provenance | label | observed, parsed or inferred |
| CostRepresentation | kind | label | itemized or day-rate |
| CostRepresentation | CAPEX of an operation | share of C | cost of the operation under the representation |
| PriceList | unit price | share of C per unit | price of one recorded quantity |
| PriceList | day rate | share of C per day | rate of a machine class, or of calendar duration |
| PriceList | netback | share of C per t | value of a tonne of incremental oil in the realized NPV |
| PriceList | discount rate | 1 per year | 0.15 in the paper, applied monthly |
| ResponseForecast | incremental oil | t | forecast oil in the 24 months after start-up above the counterfactual |
| ResponseForecast | forecast NPV | share of C | forecast revenue minus CAPEX |
| ProgramYear | year | year | year of the decision |
| ProgramYear | budget | share of C | CAPEX of the historical program of the year |
| ProgramYear | crew-day capacity | days | crew-days of the historical program of the year |
| Selection | selected set | operations | operations taken in a year |
| Selection | Jaccard overlap | 0 to 1 | overlap of selected sets under two representations |
| Selection | share of the gap closed | 0 to 1 | GC of the paper |
| ReviewList | operations | operations | disagreement operations and draw-sensitive operations |

## Terms

| term | definition |
|---|---|
| Well | A producing well, identified by a normalized key. |
| Operation | A dated intervention on a well, with a type; the unit that is selected or not. |
| Pool | The 6,800 operations of twelve selectable types that the study draws on. |
| Forced class | Operation types that the operator must run: SWO, RIW, LER, CLR. |
| Discretionary class | Operation types the operator chooses: BZT, PMO, HF, SDT, PCM, RIS, ONW, APF. |
| Cost driver | A recorded quantity that causes cost: duration, idle time, equipment event, material volume, pump designation. |
| Observed quantity | A cost driver read from a structured table by the date window of the operation. |
| Parsed quantity | A cost driver read from a text field by a parser; only the pump designation. |
| Inferred quantity | A cost driver that depends on a forecast or a substituted value. |
| Pump designation | The registry string `ESP-delivery-head` that names the electric submersible pump before and after an operation. |
| QH | The product of pump delivery and head, a proxy for hydraulic power, priced by one commodity price per unit. |
| Pump-change outcome | The reading of the two designation strings: (a) both valid and the size identical, priced at zero; (b) both valid and the size different, QH read from the designation; (c) a field empty or unparsable, priced from other drivers (c1) or at the type median (c2). |
| Cost representation | A rule that turns the cost drivers of an operation into CAPEX. |
| Itemized representation | CAPEX as recorded quantities times unit prices plus machine time times day rates (quantity-based in the paper). |
| Day-rate representation | CAPEX as raw calendar duration times one rate, with no material part, calibrated to the same total CAPEX as the itemized one. |
| Price list | The unit prices, day rates, netback and discount rate that turn quantities into money. |
| Response model | A gradient-boosted tree ensemble that forecasts incremental oil, refitted each year on operations of earlier years only. |
| Forecast NPV | Forecast incremental oil times the oil price used by the selection, minus CAPEX. |
| Program year | One year of selection, with a budget and a crew-day capacity taken from the historical program of that year. |
| Selection | The operations taken in a program year: positive forecast NPV, filled greedily under the budget and the crew-day capacity. |
| Benchmark | The selection that knows the realized NPV of every operation, solved exactly. |
| Share of the gap closed (GC) | The NPV gain of a selection over the historical program, divided by the gain of the benchmark over the historical program. |
| Jaccard overlap | The size of the intersection of two selected sets divided by the size of their union. |
| Cross-accounting | Valuing the selection made under one representation with another representation. |
| Draw-sensitive operation | An operation selected in more than 5% and fewer than 95% of the 500 price draws. |
| Sign-uncertain operation | An operation whose sign of NPV is not fixed within the box of admissible prices; a diagnostic that does not enter the selection. |
| Disagreement operation | An operation selected under one representation and not under the other. |
| Review list | The disagreement operations together with the draw-sensitive ones; the operations sent to an engineer for a cost check. |
| C | The CAPEX of all 6,800 pool operations, used as the unit of scale for every monetary quantity. |
