"""Step 1: generate the labeled detector dataset and the categorized attack suite."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_gen  # noqa: E402

if __name__ == "__main__":
    data_gen.main()
