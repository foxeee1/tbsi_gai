#!/usr/bin/env python3
"""Independent retry identity after the interrupted foreground-loss candidate retry1."""
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).with_name('run_mr_source_foreground_full_20261116.py')
spec = importlib.util.spec_from_file_location('mr_foreground_retry2_base', SOURCE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.FAMILY = 'MR-G1-FGLOSS-U936-S42-FULL15-FULL244-20261116-RETRY2'
base.RUN = base.ROOT / 'output/experiments' / base.FAMILY
base.__file__ = str(Path(__file__).resolve())

if __name__ == '__main__':
    base.main()
