# I run the scripts as `python scripts/<name>.py`, so Python only looks inside
# scripts/ for imports. This adds the repo root so `import adsb` works.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
