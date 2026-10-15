"""Pipeline step that runs the synthetic parser test set."""
import importlib.util
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('test_parser_synthetic', ROOT / 'tests/test_parser_synthetic.py')
tps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tps)
tps.отчёт()
