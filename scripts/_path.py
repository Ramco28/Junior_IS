"""Let the scripts import the adsb package.

I run the scripts as `python scripts/<name>.py`, so Python only looks inside
scripts/ for imports. Importing this module adds the repo root to the path.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
