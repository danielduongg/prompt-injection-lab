"""Run the full pipeline end-to-end: build data -> train -> evaluate -> figures."""
import runpy
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for step in ["01_build_data.py", "02_train_detector.py",
             "03_run_eval.py", "04_make_figures.py"]:
    print(f"\n=== {step} ===")
    sys.argv = [step]
    runpy.run_path(str(HERE / step), run_name="__main__")
