"""Old name of pipeline.py, kept so job lists written before the rename keep working."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
