# Auditing cost models in well-intervention selection: analysis code and aggregate results

This repository accompanies the paper "Auditing Cost Models in Well-Intervention
Selection: Reading Cost Drivers from Field Records". It releases the analysis code of
the method, a synthetic test set for the pump-designation parser, the scripts of the
figures, and the aggregate result tables behind the tables and figures of the paper.

## The audit in brief

The audit asks which operations of a selection program depend on the way their cost
is modeled. It recovers the physical drivers of cost for every candidate operation
from the registry (observed, parsed or inferred), builds two cost representations
calibrated to the same total CAPEX (itemized quantities times prices, and day rate),
and runs the same selection under each, and under many draws of the price vector.
The operations on which the selections disagree, or whose selection flips across
price draws, are the ones an engineer should check. The six steps are written out in
[`docs/AUDIT_PROCEDURE.md`](docs/AUDIT_PROCEDURE.md).

## What is not included, and why

The records the study is computed on belong to an operating company and are
confidential. They are not part of this repository, and neither is anything derived
from them at the level of a well, an operation or a row. In particular the
repository contains no production reports, regime tables, intervention registry,
survey workbooks, price registry, well identifiers, field or company names, and no
amounts in currency. Every monetary quantity in the released tables is given as a
percentage of the scale C defined in the paper (the sum of the CAPEX of the pool of
6,800 operations); C itself is not released. Unit prices and scenario parameters in
the code (`cost_physical.Prices`, `npv.Params`) are placeholders set to 1.0 and must
be replaced with your own values. Per-operation outputs (for example the table behind
Fig. 3) are neither stored nor shipped; the script writes only counts and shares.

Because the data are not released, the numerical results of the paper cannot be
reproduced from this repository alone. What can be re-run without any data is the
pump-designation parser on the synthetic test set (see below), and the figures that
are drawn from the aggregate tables.

## Directory map

| path | content |
|---|---|
| `src/` | analysis code: parser and cost representations (`cost_physical.py`, `capex_exante.py`, `crew_model.py`, `dur_model.py`), economic model (`npv.py`), response model and selection (`ep06_ablation.py`, `ep19_crossruler.py`), perfect-information benchmark (`oracle.py`), inference (`stats.py`, `perm_control.py`, `ep30_loyo_bca.py`), price scenarios (`price_montecarlo.py`, `ep30_pselect.py`, `ep30_ident.py`), robustness and structure tests (`ep30_*.py`), decomposition (`ep15_decomposition.py`, `ep27_forced.py`) |
| `src/config.py` | all data locations as placeholders resolved from environment variables |
| `src/figures/` | scripts of Figs. 1-6 (`fig_workflow.py`, `fig_years.py`, `fig_flips.py`, `fig_waterfall.py`, `fig_map_by_type.py`, `fig_budget.py`) and their shared style |
| `tests/test_parser_synthetic.py` | evaluation of the parser on the synthetic set (runs without data) |
| `tests/fixtures/parser_synthetic.jsonl` | the synthetic test set, 223 rows in 21 classes |
| `tests/fixtures/make_parser_synthetic.py` | deterministic generator of the synthetic set |
| `tests/test_cost_structure.py` | structural tests of the cost model (need your own tables) |
| `aggregates/` | aggregate result tables (TSV) and the 2024 flip summary, CC BY 4.0 |
| `docs/AUDIT_PROCEDURE.md` | the audit procedure, one page |
| `docs/GLOSSARY.md` | Russian names kept in the code -> English |

## The synthetic parser set

All 223 rows are invented. Each row is a pair of pump-designation strings with the
expected outcome fixed by construction (same size, different size, undeterminable),
not copied from the output of the parser. The generator draws delivery and head from
two fixed pools of nine round numbers and uses a fixed seed; running it reproduces
the file byte for byte. Agreement with the expectation is agreement with the labeling
instruction, not accuracy against the truth. To evaluate the parser:

    pip install -r requirements.txt
    PYTHONPATH=src python tests/test_parser_synthetic.py            # pass/fail checks
    PYTHONPATH=src python tests/test_parser_synthetic.py --report   # writes results/ep30_parser_synthetic.md

The report gives a full match with the expectation of 0.6996, a match of case and
power of 0.8072, and 12 of the 21 classes with at least one discrepancy.

## Running the pipeline on your own records

1. Install the requirements. The code targets Python 3.12 or later.
2. Set `PIPELINE_ROOT` to a working directory, or `PIPELINE_DATA` to a directory laid
   out like `data/` in `src/config.py` (parquet tables under `parquet/` for the
   intervention registry, regime tables, fracturing stages, well construction, and
   under `interim/` for the derived tables the steps pass to each other). The column
   names are those read in `src/cost_physical.py` and `src/npv.py` (Russian names;
   see `docs/GLOSSARY.md`); read the `pl.read_parquet` calls to see the schema each
   step expects.
3. Replace the placeholder unit prices in `cost_physical.Prices` and the scenario
   parameters in `npv.Params` with your own values.
4. Run the steps in the order of the dependencies: the cost model and the ex-ante
   chain (`cost_physical`, `capex_exante`), the main selection and benchmark
   (`ep06_ablation`, `ep19_crossruler`), then the `ep30_*` analyses. Each step reads
   the chain, reproduces its headline figures within a tolerance before printing
   anything new, and writes one report. `PIPELINE_RESULTS`, `PIPELINE_OUT` and
   `PIPELINE_FIGURES` set where reports, tables and figures go.
5. Some modules contain reproduction checks against the values of the original study
   (for example the main gap-closed share of 0.2631). On other data these checks will
   stop the run by design; adjust the expected values to your own first run.

The loader of the operator's survey price sheets is withheld; `src/price_registry.py`
is a stub, and the steps that reconcile against those sheets raise an error. The
scripts, identifiers and printed messages inside `src/` are in Russian; the glossary
gives the translation of the names a reader needs. The aggregate tables and the
figure scripts are in English.

## How the aggregates map to the paper

Money columns are named `*_pct_C` and are percentages of C. N and L1 of the original
code are named ITM (itemized) and DR (day-rate) in the tables; GC is the share of the
gap to the perfect-information benchmark closed by the selection; operation types use
the Latin codes of the paper (PMO pump optimization, SDT sidetrack, HF hydraulic
fracturing, BZT acid treatment, RIS remedial cementing).

| paper item | file in `aggregates/` or script |
|---|---|
| Table 1, populations | `ep30_measure_v3_C.tsv`, `ep30_provenance.tsv`, `ep30_zbs.tsv` |
| Table 2, pump outcomes | `ep30_measure_v3_pump.tsv` |
| Table 3, cross-accounting | `ep30_crossacct.tsv` (selection x valuation, numerator, denominator), `ep30_repr.tsv` |
| Table 4, sides of the disagreement | `table7_disagreement_sides.tsv` |
| Table 5, robustness | sidetracks and drilling rate: `ep30_zbs.tsv`, `ep30_sdt_table.tsv`; price draws: `price_montecarlo_summary.tsv`, `price_montecarlo_multipliers.tsv`; leave one type out: `ep30_loto.tsv`; pump components: `ep30_lambda.tsv`, `lambda_sweep.tsv`; leave one year out, BCa interval, sign-flip test: `ep30_loyo_bca.tsv` |
| Fig. 1, audit workflow | `src/figures/fig_workflow.py` (counts typed from Tables 1 and 3 and `ep30_pselect.tsv`, `flips_2024_summary.txt`) |
| Fig. 2, gain by year | `ep30_crossacct_years.tsv` -> `src/figures/fig_years.py` |
| Fig. 3, the 2024 pool | `flips_2024_summary.txt`; `src/figures/fig_flips.py` (needs the confidential tables) |
| Fig. 4, decomposition of the gap | `table8_fig2_decomposition.tsv` -> `src/figures/fig_waterfall.py` |
| Fig. 5, gain by type and leave-one-type-out | `fig5_share_of_gain_by_type.tsv`, `ep30_loto.tsv` -> `src/figures/fig_map_by_type.py` |
| Fig. 6, budget multiplier | `fig6_budget_alpha.tsv` -> `src/figures/fig_budget.py` |

Supplementary Material, best-effort mapping:

| supplement item | file in `aggregates/` |
|---|---|
| cross-accounting, four representations, yearly differences (S3, S16-S18) | `ep30_crossacct.tsv`, `ep30_repr.tsv`, `ep30_crossacct_years.tsv` |
| selection frequency under price draws, identification (S4, S19) | `ep30_pselect.tsv`, `ep30_ident.tsv`, `price_montecarlo_*.tsv` |
| leave-one-type-out, gain by type (S20, S21) | `ep30_loto.tsv`, `fig5_share_of_gain_by_type.tsv` |
| sidetracks, drilling rate, pump multiple (S4, S5, S14) | `ep30_zbs.tsv`, `ep30_sdt_table.tsv`, `ep30_lambda.tsv` |
| disagreement sides (S6) | `table7_disagreement_sides.tsv` |
| response model quality (S7) | `response_quality.tsv` |
| proppant forecast, pump power, parser perturbation (S8, S9, S22) | `ep30_phantom.tsv`, `ep30_pump_contrast.tsv`, `ep30_parser_perturb*.tsv` |
| parser validation, causes, synthetic set (S6.x of the parser section) | `ep30_parser_validation.tsv`, `ep30_parser_causes_t1.tsv`, `ep30_parser_causes_t2.tsv`, `ep30_parser_synthetic.tsv` |
| dates of the features, structure tests (S12) | `ep30_feature_dates.tsv`, `ep30_structure.tsv`, `ep30_purge.tsv` |
| provenance of the quantities | `ep30_provenance.tsv`, `ep30_provenance_subset.tsv` |
| intervals and tests (S13), budget grid (S15) | `ep30_loyo_bca.tsv`, `fig6_budget_alpha.tsv` |

## Citation

If you use this material, please cite the paper (see `CITATION.cff`). Citation
details, including the DOI of this repository, will be added when the paper is
published.

## License

The code is released under the MIT License (see `LICENSE`). The aggregate tables in
`aggregates/` are released under the Creative Commons Attribution 4.0 International
license (CC BY 4.0). The synthetic test set is released under the MIT License
together with the code.
