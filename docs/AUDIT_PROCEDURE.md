# Audit procedure

The audit tests whether the cost model behind an intervention-selection program can be trusted, using only what a registry of past operations already holds. It does not need new data collection or a new cost study. An operator can repeat it in six steps (Fig. 1 of the paper).

1. **Read the cost-driver quantities from the records.** For every candidate operation take durations, idle time, equipment events, recorded material volumes and the pump designation strings before and after the operation. Mark each quantity as observed (read from a structured table), parsed (read from a text field by a parser) or inferred (filled by a type-level rule or a forecast). Validate the parser on a labeled sample and on a synthetic test set (`tests/`).
2. **Build at least two cost representations of every candidate.** For example an itemized one (quantities times unit prices) and a day-rate one (duration times a daily rate), both calibrated to the same total CAPEX, so that the comparison concerns the structure of the cost model and not its level.
3. **Predict the effect.** Fit a response model on earlier years only (walk-forward) to forecast incremental oil of each candidate, and compute NPV under each cost representation with the same prices and discount rate.
4. **Run the same selection under each representation.** Use the same greedy selection under the budget and crew-day capacity of each year, once per representation and under many draws of the price vector. Compute the Jaccard overlap of the selected sets and the share of the gap closed (GC) against the perfect-information benchmark, with bootstrap intervals over years.
5. **List the operations that depend on the choice.** These are the operations selected under one representation and not the other, and the operations whose selection is sensitive to price draws. The sign-uncertain classes over the admissible price box are a diagnostic, not an input to the selection.
6. **Send only these operations to an engineer for a cost check.** In the paper's case this is about a sixth of the operations of the nine selection years, instead of the whole list.

The steps are implemented by the modules in `src/`; see the README for the mapping from the tables and figures of the paper to the files in `aggregates/` and the figure scripts in `src/figures/`.
