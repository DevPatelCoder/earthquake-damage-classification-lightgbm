import sys
import types
from pathlib import Path


import pandas as pd


# Stub the mlebench package so prepare.py imports work without installing it.
mlebench = types.ModuleType("mlebench")
mlebench_utils = types.ModuleType("mlebench.utils")
mlebench_utils.read_csv = pd.read_csv
sys.modules["mlebench"] = mlebench
sys.modules["mlebench.utils"] = mlebench_utils


sys.path.insert(0, "task")
from prepare import prepare


public = Path("public")
private = Path("private")
public.mkdir(parents=True, exist_ok=True)
private.mkdir(parents=True, exist_ok=True)


prepare(Path("raw"), public, private)
print("Wrote prepared/public and prepared/private")
