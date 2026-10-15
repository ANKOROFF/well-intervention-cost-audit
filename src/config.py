"""Paths of the pipeline. All data locations are placeholders.

The operator data behind the original study are confidential and are not distributed.
To run the pipeline on your own records, point PIPELINE_ROOT at a directory that holds
`data/` (raw tables, parquet exports and interim files with the schemas described in
README.md) and make sure the unit-price placeholders in `cost_physical.Prices` and
`npv.Params` are replaced by your own values.
"""
import os
import pathlib

ROOT = pathlib.Path(os.environ.get("PIPELINE_ROOT", pathlib.Path(__file__).resolve().parent.parent))
DATA = pathlib.Path(os.environ.get("PIPELINE_DATA", ROOT / "data"))
RESULTS = pathlib.Path(os.environ.get("PIPELINE_RESULTS", ROOT / "results"))
OUT = pathlib.Path(os.environ.get("PIPELINE_OUT", ROOT / "aggregates"))
FIGURES = pathlib.Path(os.environ.get("PIPELINE_FIGURES", ROOT / "figures"))
CATALOG = pathlib.Path(os.environ.get("PIPELINE_CATALOG", ROOT / "catalog"))


def data(rel):
    return str(DATA / rel)


def results(rel):
    return str(RESULTS / rel)


def out(rel):
    return str(OUT / rel)


def catalog(rel):
    return str(CATALOG / rel)


def figures(rel):
    return str(FIGURES / rel)
